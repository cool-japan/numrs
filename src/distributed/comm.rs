//! Legacy point-to-point communication types, rebuilt on [`super::net`].
//!
//! # Status: retired in favour of [`super::net::Endpoint`]
//!
//! New code should use [`super::net::endpoint::Endpoint`] (bootstrapped by
//! [`super::bootstrap::bootstrap`]) rather than anything below. This module
//! survives only because [`super::prelude`] and `bench/distributed_benchmarks.rs`
//! still name its types; it is a compatibility surface, not a place to build
//! on. [`super::collective`] and [`super::communication`] were rebuilt
//! directly on [`super::net`] and no longer reference anything here (each
//! defines its own error type rather than reusing [`CommunicationError`]).
//!
//! # What was wrong with the original, and what replaced it
//!
//! The first version of this file owned its sockets directly and had two
//! defects that no amount of careful call ordering could work around:
//!
//! - `CommunicationChannel` held one `Arc<Mutex<TcpStream>>` across every
//!   `.await`. A `recv()` in flight held that lock across `read_exact().await`,
//!   so a concurrent `send()` on the same channel blocked forever trying to
//!   lock the same mutex to reach the write half — even though a real TCP
//!   socket's read and write directions never contend. Two ranks that both
//!   sent before either received deadlocked outright.
//! - `ConnectionManager::recv` performed one `accept()` per call. It could not
//!   demultiplex by source rank (whichever connection happened to arrive
//!   serviced the call), and because the accepted stream was dropped after a
//!   single message, a peer's *second* message down a pooled connection was
//!   never read at all.
//!
//! Both types are now thin shims over the redesigned transport:
//!
//! - [`CommunicationChannel`] wraps a [`super::net::link::Link`]. Sends are
//!   enqueue-and-return onto that link's bounded writer queue, receives pop a
//!   private [`super::net::mailbox::Mailbox`] filled by the link's reader task.
//!   Nothing is ever locked across a socket `.await`, so the bidirectional
//!   deadlock is gone *by construction* — see
//!   `tests::simultaneous_bidirectional_send_does_not_deadlock`.
//! - [`ConnectionManager`] runs one persistent accept loop that keeps every
//!   inbound connection alive and streams its frames into a shared queue.
//!   [`ConnectionManager::recv`] pops that queue (any source, in arrival
//!   order) and [`ConnectionManager::recv_from`] demultiplexes by source rank,
//!   stashing frames from other ranks instead of dropping them.
//!
//! Both speak the same 56-byte [`super::net::frame::FrameHeader`] wire format
//! as the rest of the transport rather than a second, private codec: a
//! [`Message`]'s oxicode encoding travels as the frame payload, under
//! `ctx = 0` and frame `tag = 0` (the legacy [`MessageTag`] lives inside the
//! serialized [`MessageHeader`], so per-peer FIFO order is preserved across
//! tags exactly as the original stream-of-messages semantics required).
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::comm::*;
//! use std::net::SocketAddr;
//!
//! # async fn example() -> Result<(), CommunicationError> {
//! let addr: SocketAddr = "127.0.0.1:5000".parse().expect("valid socket address literal");
//! let manager = ConnectionManager::new(addr).await?;
//!
//! // Send data to another process
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0];
//! let msg = Message::new(0, 1, 42, data)?;
//! manager.send(msg).await?;
//!
//! // Receive data
//! let received: Message<f64> = manager.recv().await?;
//! # Ok(())
//! # }
//! ```

use super::net::frame::FrameHeader;
use super::net::link::{read_frame, Link};
use super::net::mailbox::{Mailbox, MailboxKey};
use super::net::{EndpointConfig, NetError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;

/// Errors that can occur during communication operations
#[derive(Error, Debug, Clone)]
pub enum CommunicationError {
    #[error("Network I/O error: {0}")]
    IoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Connection error to {addr}: {msg}")]
    ConnectionError { addr: String, msg: String },

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid message format: {0}")]
    InvalidMessage(String),

    #[error("Buffer overflow: tried to write {size} bytes, buffer capacity {capacity}")]
    BufferOverflow { size: usize, capacity: usize },

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Invalid rank: {0}")]
    InvalidRank(usize),
}

impl From<NetError> for CommunicationError {
    /// Map a transport error onto the legacy error surface.
    ///
    /// `collective.rs` matches on these variants, so the mapping is chosen to
    /// preserve the meaning callers already depend on rather than to be
    /// lossless about the richer [`NetError`] taxonomy.
    fn from(err: NetError) -> Self {
        match err {
            NetError::Io(msg) => Self::IoError(msg),
            NetError::PeerClosed | NetError::MailboxClosed => Self::ConnectionClosed,
            NetError::Timeout(d) => Self::Timeout(d),
            NetError::RecvTimeout { timeout, .. } => Self::Timeout(timeout),
            NetError::ConnectFailed { addr, msg } => Self::ConnectionError { addr, msg },
            NetError::InvalidRank { rank, .. } => Self::InvalidRank(rank as usize),
            NetError::FrameTooLarge { size, max } => Self::BufferOverflow {
                size,
                capacity: max,
            },
            NetError::Compression(msg) => Self::DeserializationError(msg),
            other => Self::InvalidMessage(other.to_string()),
        }
    }
}

/// Message tag type for distinguishing different message types
pub type MessageTag = u32;

/// Message header containing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Source process rank
    pub source: usize,
    /// Destination process rank
    pub dest: usize,
    /// Message tag for identification
    pub tag: MessageTag,
    /// Size of payload in bytes
    pub payload_size: usize,
    /// Message sequence number
    pub sequence: u64,
}

impl MessageHeader {
    /// Create a new message header
    pub fn new(
        source: usize,
        dest: usize,
        tag: MessageTag,
        payload_size: usize,
        sequence: u64,
    ) -> Self {
        Self {
            source,
            dest,
            tag,
            payload_size,
            sequence,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, CommunicationError> {
        let config = oxicode::config::standard();
        oxicode::serde::encode_to_vec(self, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CommunicationError> {
        let config = oxicode::config::standard();
        let (header, _): (Self, usize) = oxicode::serde::decode_from_slice(bytes, config)
            .map_err(|e| CommunicationError::DeserializationError(e.to_string()))?;
        Ok(header)
    }
}

/// A message containing data to be sent between processes
#[derive(Debug, Clone)]
pub struct Message<T> {
    /// Message header with metadata
    pub header: MessageHeader,
    /// Message payload
    pub payload: Vec<T>,
}

impl<T: Serialize + for<'de> Deserialize<'de> + Clone> Message<T> {
    /// Create a new message
    pub fn new(
        source: usize,
        dest: usize,
        tag: MessageTag,
        payload: Vec<T>,
    ) -> Result<Self, CommunicationError> {
        // Serialize payload to get size
        let config = oxicode::config::standard();
        let payload_bytes = oxicode::serde::encode_to_vec(&payload, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))?;

        let header = MessageHeader::new(source, dest, tag, payload_bytes.len(), 0);

        Ok(Self { header, payload })
    }

    /// Create a message with a specific sequence number
    pub fn with_sequence(
        source: usize,
        dest: usize,
        tag: MessageTag,
        payload: Vec<T>,
        sequence: u64,
    ) -> Result<Self, CommunicationError> {
        let config = oxicode::config::standard();
        let payload_bytes = oxicode::serde::encode_to_vec(&payload, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))?;

        let header = MessageHeader::new(source, dest, tag, payload_bytes.len(), sequence);

        Ok(Self { header, payload })
    }

    /// Serialize message to bytes (header + payload)
    pub fn to_bytes(&self) -> Result<Vec<u8>, CommunicationError> {
        let header_bytes = self.header.to_bytes()?;
        let config = oxicode::config::standard();
        let payload_bytes = oxicode::serde::encode_to_vec(&self.payload, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))?;

        // Format: [header_size: u32][header][payload]
        let header_size = header_bytes.len() as u32;
        let mut bytes = Vec::with_capacity(4 + header_bytes.len() + payload_bytes.len());
        bytes.extend_from_slice(&header_size.to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload_bytes);

        Ok(bytes)
    }

    /// Deserialize message from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CommunicationError> {
        if bytes.len() < 4 {
            return Err(CommunicationError::InvalidMessage(
                "Insufficient bytes for header size".to_string(),
            ));
        }

        // Read header size
        let header_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        if bytes.len() < 4 + header_size {
            return Err(CommunicationError::InvalidMessage(format!(
                "Insufficient bytes for header: expected {}, got {}",
                header_size,
                bytes.len() - 4
            )));
        }

        // Read header
        let header = MessageHeader::from_bytes(&bytes[4..4 + header_size])?;

        // Read payload
        let payload_bytes = &bytes[4 + header_size..];
        let config = oxicode::config::standard();
        let (payload, _): (Vec<T>, usize) =
            oxicode::serde::decode_from_slice(payload_bytes, config)
                .map_err(|e| CommunicationError::DeserializationError(e.to_string()))?;

        Ok(Self { header, payload })
    }

    /// Get the source rank
    pub fn source(&self) -> usize {
        self.header.source
    }

    /// Get the destination rank
    pub fn dest(&self) -> usize {
        self.header.dest
    }

    /// Get the message tag
    pub fn tag(&self) -> MessageTag {
        self.header.tag
    }

    /// Get the sequence number
    pub fn sequence(&self) -> u64 {
        self.header.sequence
    }
}

/// Context id every legacy frame travels under.
///
/// Fixed at `0`: this compatibility layer has no notion of sub-communicators,
/// and the user-visible [`MessageTag`] lives inside the serialized
/// [`MessageHeader`] rather than in the frame, so one channel is one FIFO
/// stream of messages exactly as the original API promised.
const LEGACY_CTX: u64 = 0;

/// Frame tag every legacy frame travels under. See [`LEGACY_CTX`].
const LEGACY_TAG: u64 = 0;

/// Peer rank a standalone [`CommunicationChannel`]'s private mailbox is keyed
/// by. A channel owns exactly one link and one mailbox, so the value is
/// arbitrary — it only has to be consistent between the reader task and
/// [`CommunicationChannel::recv`].
const LEGACY_PEER: u32 = 0;

/// The mailbox key every frame arriving on a [`CommunicationChannel`] lands
/// under.
const LEGACY_KEY: MailboxKey = (LEGACY_PEER, LEGACY_CTX, LEGACY_TAG);

/// How often [`CommunicationChannel::recv`] re-checks whether the peer went
/// away while it is parked with nothing to deliver.
///
/// This is a liveness poll, not a latency floor: a frame that arrives wakes
/// the parked receive immediately. Only the "nothing has arrived yet" path
/// pays it, and only to notice a closed connection.
const LIVENESS_POLL: Duration = Duration::from_millis(50);

/// How long one [`ConnectionManager::next_inbound_turn`] holds the inbound
/// queue before yielding it to another receive.
///
/// This is a fairness bound, not a latency floor: a frame that arrives is
/// taken immediately. It only bounds how long a receive waiting for a *quiet*
/// source may hold the queue, which is what keeps two concurrent `recv_from`
/// calls for different ranks from deadlocking each other.
const INBOUND_TURN: Duration = Duration::from_millis(25);

/// Communication channel for sending and receiving messages
///
/// Wraps one [`Link`]: sends enqueue onto its writer task and return, receives
/// pop the private mailbox its reader task fills. The two directions share no
/// lock, which is what makes simultaneous bidirectional traffic complete
/// instead of deadlocking.
///
/// Construct this inside a tokio runtime context: the underlying link spawns
/// its reader and writer tasks eagerly.
pub struct CommunicationChannel {
    /// Framed connection to the peer (owns the reader/writer tasks).
    link: Link,
    /// Where this channel's own reader task delivers inbound payloads.
    mailbox: Arc<Mailbox<Vec<u8>>>,
    /// Remote address
    remote_addr: SocketAddr,
    /// Sequence counter for messages
    sequence: AtomicU64,
    /// Largest frame this channel will emit or accept.
    max_frame: usize,
}

impl std::fmt::Debug for CommunicationChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommunicationChannel")
            .field("remote_addr", &self.remote_addr)
            .field("sequence", &self.sequence.load(Ordering::Relaxed))
            .finish()
    }
}

impl CommunicationChannel {
    /// Create a new communication channel
    pub fn new(stream: TcpStream, remote_addr: SocketAddr) -> Self {
        Self::with_config(stream, remote_addr, EndpointConfig::default())
    }

    /// As [`Self::new`], with an explicit transport configuration (queue
    /// depth, frame ceiling, timeouts).
    pub fn with_config(stream: TcpStream, remote_addr: SocketAddr, config: EndpointConfig) -> Self {
        let mailbox = Arc::new(Mailbox::with_key_capacity(
            config.queue_depth.saturating_mul(64).max(1),
        ));
        let max_frame = config.max_frame;
        let link = Link::from_stream(
            stream,
            LEGACY_PEER,
            remote_addr,
            Arc::clone(&mailbox),
            &config,
        );
        Self {
            link,
            mailbox,
            remote_addr,
            sequence: AtomicU64::new(0),
            max_frame,
        }
    }

    /// Send a message through this channel.
    ///
    /// Enqueue-and-return: this awaits the link's bounded writer queue, never
    /// the socket itself, so it completes even while the peer is busy sending
    /// in the other direction.
    pub async fn send<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        mut message: Message<T>,
    ) -> Result<(), CommunicationError> {
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        message.header.sequence = seq;

        let bytes = message.to_bytes()?;
        let header = legacy_frame_header(
            message.source(),
            message.dest(),
            seq,
            &bytes,
            self.max_frame,
        )?;
        self.link.send_frame(header, bytes).await?;
        Ok(())
    }

    /// Receive a message from this channel.
    ///
    /// Returns the next message the peer sent, in send order. Waits
    /// indefinitely for one, but reports [`CommunicationError::ConnectionClosed`]
    /// as soon as the peer hangs up rather than hanging forever.
    ///
    /// # Cancel safety
    ///
    /// Dropping this future (as [`Self::recv_timeout`] does on expiry) never
    /// loses a message. The internal poll is
    /// [`super::net::mailbox::Mailbox::pop_timeout`], which on expiry re-takes
    /// the mailbox lock, unregisters its own waiter, and drains any payload a
    /// concurrent delivery had already committed to it — so a message either
    /// comes back from this call or is still queued for the next one. A
    /// message *is* lost only if the future is dropped in the instant between
    /// `pop_timeout` returning `Ok` and this function returning it, which is
    /// inherent to any `T`-by-value receive and matches the original API's
    /// contract.
    pub async fn recv<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
    ) -> Result<Message<T>, CommunicationError> {
        loop {
            match self.mailbox.pop_timeout(LEGACY_KEY, LIVENESS_POLL).await {
                Ok(payload) => return Message::from_bytes(&payload),
                Err(NetError::RecvTimeout { .. }) => {
                    if self.link.reader_is_finished() {
                        // Drain anything the reader delivered on its way out
                        // before declaring the connection dead.
                        if let Some(payload) = self.mailbox.try_pop(LEGACY_KEY)? {
                            return Message::from_bytes(&payload);
                        }
                        return Err(CommunicationError::ConnectionClosed);
                    }
                }
                Err(other) => return Err(other.into()),
            }
        }
    }

    /// Receive a message with timeout
    pub async fn recv_timeout<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        duration: Duration,
    ) -> Result<Message<T>, CommunicationError> {
        timeout(duration, self.recv())
            .await
            .map_err(|_| CommunicationError::Timeout(duration))?
    }

    /// Get the remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Flush queued frames and shut the connection down.
    pub async fn close(&self) -> Result<(), CommunicationError> {
        self.link.close().await?;
        Ok(())
    }
}

/// Build the wire header for one legacy message frame.
fn legacy_frame_header(
    source: usize,
    dest: usize,
    seq: u64,
    bytes: &[u8],
    max_frame: usize,
) -> Result<FrameHeader, CommunicationError> {
    if bytes.len() > max_frame {
        return Err(CommunicationError::BufferOverflow {
            size: bytes.len(),
            capacity: max_frame,
        });
    }
    let len = u32::try_from(bytes.len()).map_err(|_| CommunicationError::BufferOverflow {
        size: bytes.len(),
        capacity: max_frame,
    })?;
    // The wire header's rank fields are u32. A `usize` rank that does not fit
    // is rejected rather than silently truncated: a truncated rank would make
    // `ConnectionManager::recv_from` demultiplex to the wrong source.
    let src = u32::try_from(source).map_err(|_| CommunicationError::InvalidRank(source))?;
    let dst = u32::try_from(dest).map_err(|_| CommunicationError::InvalidRank(dest))?;
    FrameHeader::new(src, dst, LEGACY_CTX, LEGACY_TAG, seq, len, len, 0).map_err(Into::into)
}

/// One frame delivered by [`ConnectionManager`]'s accept loop.
type Inbound = (FrameHeader, Vec<u8>);

/// Frames [`ConnectionManager::recv_from`] pulled off the inbound queue while
/// looking for a different rank, held per source rank until someone asks for
/// them.
type StashedBySource = HashMap<usize, VecDeque<Vec<u8>>>;

/// Connection manager for managing multiple connections
///
/// Unlike the original, the listener is drained by one persistent accept loop
/// that keeps every inbound connection alive for the manager's lifetime: a
/// peer can send any number of messages down one pooled connection and all of
/// them are read. [`Self::recv`] takes the next message from any source;
/// [`Self::recv_from`] demultiplexes by source rank without dropping the
/// frames it skips.
pub struct ConnectionManager {
    /// Local bind address
    local_addr: SocketAddr,
    /// Connection pool: rank -> channel
    connections: Arc<RwLock<HashMap<usize, Arc<CommunicationChannel>>>>,
    /// Mapping of addresses to ranks
    rank_addresses: Arc<RwLock<HashMap<usize, SocketAddr>>>,
    /// Frames accepted from peers, in arrival order.
    inbound: Mutex<mpsc::Receiver<Inbound>>,
    /// Frames [`Self::recv_from`] pulled off `inbound` while looking for a
    /// different source, kept for the receive that does want them.
    ///
    /// A `std::sync::Mutex`, never held across an `.await`: that is what lets
    /// a frame be taken off the inbound queue and stashed in one uncancellable
    /// step.
    stash: std::sync::Mutex<StashedBySource>,
    /// The accept loop; aborted (with every reader task it owns) on drop.
    accept_task: JoinHandle<()>,
    /// Transport configuration shared by every channel this manager opens.
    config: EndpointConfig,
}

impl Drop for ConnectionManager {
    fn drop(&mut self) {
        self.accept_task.abort();
    }
}

impl std::fmt::Debug for ConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionManager")
            .field("local_addr", &self.local_addr)
            .finish()
    }
}

impl ConnectionManager {
    /// Create a new connection manager
    pub async fn new(bind_addr: SocketAddr) -> Result<Self, CommunicationError> {
        Self::with_config(bind_addr, EndpointConfig::default()).await
    }

    /// As [`Self::new`], with an explicit transport configuration.
    pub async fn with_config(
        bind_addr: SocketAddr,
        config: EndpointConfig,
    ) -> Result<Self, CommunicationError> {
        let listener = TcpListener::bind(bind_addr).await.map_err(|e| {
            CommunicationError::ConnectionError {
                addr: bind_addr.to_string(),
                msg: e.to_string(),
            }
        })?;

        let local_addr = listener
            .local_addr()
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;

        let (tx, rx) = mpsc::channel::<Inbound>(config.queue_depth.max(1));
        let max_frame = config.max_frame;
        let accept_task = tokio::spawn(accept_loop(listener, tx, max_frame));

        Ok(Self {
            local_addr,
            connections: Arc::new(RwLock::new(HashMap::new())),
            rank_addresses: Arc::new(RwLock::new(HashMap::new())),
            inbound: Mutex::new(rx),
            stash: std::sync::Mutex::new(HashMap::new()),
            accept_task,
            config,
        })
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Register a rank with its address
    pub async fn register_rank(&self, rank: usize, addr: SocketAddr) {
        let mut addrs = self.rank_addresses.write().await;
        addrs.insert(rank, addr);
    }

    /// The address registered for `rank`, if any.
    pub async fn address_of(&self, rank: usize) -> Option<SocketAddr> {
        self.rank_addresses.read().await.get(&rank).copied()
    }

    /// Get or create connection to a rank
    pub async fn get_connection(
        &self,
        rank: usize,
    ) -> Result<Arc<CommunicationChannel>, CommunicationError> {
        // Check if connection already exists
        {
            let conns = self.connections.read().await;
            if let Some(channel) = conns.get(&rank) {
                return Ok(Arc::clone(channel));
            }
        }

        // Get address for rank
        let addr = {
            let addrs = self.rank_addresses.read().await;
            addrs
                .get(&rank)
                .copied()
                .ok_or(CommunicationError::InvalidRank(rank))?
        };

        // Create new connection
        let stream =
            TcpStream::connect(addr)
                .await
                .map_err(|e| CommunicationError::ConnectionError {
                    addr: addr.to_string(),
                    msg: e.to_string(),
                })?;

        let channel = Arc::new(CommunicationChannel::with_config(
            stream,
            addr,
            self.config.clone(),
        ));

        // Store connection, keeping whichever one won the race so both callers
        // see the same channel.
        let mut conns = self.connections.write().await;
        let chosen = Arc::clone(conns.entry(rank).or_insert(channel));
        Ok(chosen)
    }

    /// Send a message to a specific rank
    pub async fn send<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        message: Message<T>,
    ) -> Result<(), CommunicationError> {
        let rank = message.dest();
        let channel = self.get_connection(rank).await?;
        channel.send(message).await
    }

    /// Receive a message (from any source)
    pub async fn recv<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
    ) -> Result<Message<T>, CommunicationError> {
        loop {
            // Anything a concurrent `recv_from` set aside is older than
            // whatever is still on the wire, so serve it first.
            if let Some(payload) = self.take_any_stashed() {
                return Message::from_bytes(&payload);
            }
            if let Some((_header, payload)) = self.next_inbound_turn().await? {
                return Message::from_bytes(&payload);
            }
        }
    }

    /// Receive the next message sent by `source`, setting aside (rather than
    /// discarding) anything that arrives from another rank meanwhile.
    ///
    /// This is the demultiplexing the original `recv` could not do: it read
    /// whichever connection happened to arrive next and had no way to tell the
    /// caller it was the wrong one.
    ///
    /// The stash is re-checked every turn, not just on entry: a *concurrent*
    /// receive may set aside this source's message between two of our turns,
    /// and only re-checking finds it.
    pub async fn recv_from<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        source: usize,
    ) -> Result<Message<T>, CommunicationError> {
        loop {
            if let Some(payload) = self.take_stashed(source) {
                return Message::from_bytes(&payload);
            }
            if let Some((header, payload)) = self.next_inbound_turn().await? {
                if header.src as usize == source {
                    return Message::from_bytes(&payload);
                }
                // No `.await` between taking the frame and stashing it, so a
                // cancelled `recv_from` (its `recv_from_timeout` expiring, say)
                // can never drop a frame on the floor here.
                self.stash_payload(header.src as usize, payload);
            }
        }
    }

    /// Receive a message with timeout
    pub async fn recv_timeout<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        duration: Duration,
    ) -> Result<Message<T>, CommunicationError> {
        timeout(duration, self.recv())
            .await
            .map_err(|_| CommunicationError::Timeout(duration))?
    }

    /// As [`Self::recv_from`], bounded by `duration`.
    pub async fn recv_from_timeout<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        source: usize,
        duration: Duration,
    ) -> Result<Message<T>, CommunicationError> {
        timeout(duration, self.recv_from(source))
            .await
            .map_err(|_| CommunicationError::Timeout(duration))?
    }

    /// Take one turn at the inbound queue, bounded so a caller looking for a
    /// specific source cannot monopolise it.
    ///
    /// `Ok(None)` means "nothing this turn, go round again" — the caller
    /// re-checks the stash, which is where a *concurrent* receive may have
    /// just put the frame it wants. Without that bound, two `recv_from` calls
    /// for different sources deadlock the pair: whichever grabs the queue
    /// first holds it forever waiting for a frame the other one would have to
    /// take off the queue and stash.
    ///
    /// The receiver lock is held across `recv().await`, but only for the
    /// length of one turn, and nothing else in this type ever needs that lock,
    /// so no other operation can be blocked behind it.
    async fn next_inbound_turn(&self) -> Result<Option<Inbound>, CommunicationError> {
        let mut guard = self.inbound.lock().await;
        match tokio::time::timeout(INBOUND_TURN, guard.recv()).await {
            Ok(Some(frame)) => Ok(Some(frame)),
            // The accept loop's senders all dropped: nothing more will arrive.
            Ok(None) => Err(CommunicationError::ConnectionClosed),
            Err(_elapsed) => Ok(None),
        }
    }

    /// Set a frame aside for the rank that sent it. Synchronous by design:
    /// with no `.await` between taking a frame off the queue and stashing it,
    /// a cancelled receive cannot lose one.
    fn stash_payload(&self, source: usize, payload: Vec<u8>) {
        self.lock_stash()
            .entry(source)
            .or_default()
            .push_back(payload);
    }

    fn take_stashed(&self, source: usize) -> Option<Vec<u8>> {
        let mut guard = self.lock_stash();
        let queue = guard.get_mut(&source)?;
        let item = queue.pop_front();
        if queue.is_empty() {
            guard.remove(&source);
        }
        item
    }

    fn take_any_stashed(&self) -> Option<Vec<u8>> {
        let mut guard = self.lock_stash();
        let source = *guard.keys().next()?;
        let queue = guard.get_mut(&source)?;
        let item = queue.pop_front();
        if queue.is_empty() {
            guard.remove(&source);
        }
        item
    }

    /// Lock the stash, recovering from a poisoned mutex rather than panicking
    /// (COOLJAPAN no-unwrap policy).
    fn lock_stash(&self) -> std::sync::MutexGuard<'_, StashedBySource> {
        self.stash
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Close all connections
    pub async fn close_all(&self) -> Result<(), CommunicationError> {
        let channels: Vec<Arc<CommunicationChannel>> = {
            let mut conns = self.connections.write().await;
            conns.drain().map(|(_rank, channel)| channel).collect()
        };
        for channel in &channels {
            let _ = channel.close().await;
        }
        Ok(())
    }
}

/// Accept connections until the manager goes away, streaming every frame from
/// every peer into `tx` in arrival order.
///
/// Reader tasks live in a [`JoinSet`] owned by this future, so aborting the
/// accept loop (which [`ConnectionManager::drop`] does) tears every one of
/// them down with it.
async fn accept_loop(listener: TcpListener, tx: mpsc::Sender<Inbound>, max_frame: usize) {
    let mut readers: JoinSet<()> = JoinSet::new();
    loop {
        // Reap finished readers so a long-lived manager does not accumulate
        // completed task handles.
        while readers.try_join_next().is_some() {}

        match listener.accept().await {
            Ok((stream, _addr)) => {
                let tx = tx.clone();
                readers.spawn(async move {
                    read_into(stream, tx, max_frame).await;
                });
            }
            // A transient accept error must not tear the manager down.
            Err(_) => tokio::time::sleep(Duration::from_millis(1)).await,
        }
    }
}

/// Forward every frame on `stream` into `tx` until the peer closes, the frame
/// stream goes bad, or the manager is dropped.
async fn read_into(mut stream: TcpStream, tx: mpsc::Sender<Inbound>, max_frame: usize) {
    loop {
        match read_frame(&mut stream, max_frame).await {
            Ok(frame) => {
                if tx.send(frame).await.is_err() {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loopback() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 0))
    }

    /// Two channels wired to each other over loopback TCP.
    async fn channel_pair() -> (CommunicationChannel, CommunicationChannel) {
        let listener = TcpListener::bind(loopback()).await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let dial = tokio::spawn(async move { TcpStream::connect(addr).await });
        let (accepted, dialer_addr) = listener.accept().await.expect("accept");
        let dialed = dial.await.expect("join").expect("connect");
        (
            CommunicationChannel::new(dialed, addr),
            CommunicationChannel::new(accepted, dialer_addr),
        )
    }

    #[test]
    fn test_message_header() {
        let header = MessageHeader::new(0, 1, 42, 1024, 100);
        assert_eq!(header.source, 0);
        assert_eq!(header.dest, 1);
        assert_eq!(header.tag, 42);
        assert_eq!(header.payload_size, 1024);
        assert_eq!(header.sequence, 100);
    }

    #[test]
    fn test_message_header_serialization() {
        let header = MessageHeader::new(0, 1, 42, 1024, 100);
        let bytes = header.to_bytes().expect("Serialization failed");
        let deserialized = MessageHeader::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(header.source, deserialized.source);
        assert_eq!(header.dest, deserialized.dest);
        assert_eq!(header.tag, deserialized.tag);
        assert_eq!(header.payload_size, deserialized.payload_size);
        assert_eq!(header.sequence, deserialized.sequence);
    }

    #[test]
    fn test_message_creation() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let msg = Message::new(0, 1, 42, data.clone()).expect("Message creation failed");

        assert_eq!(msg.source(), 0);
        assert_eq!(msg.dest(), 1);
        assert_eq!(msg.tag(), 42);
        assert_eq!(msg.payload, data);
    }

    #[test]
    fn test_message_serialization() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let msg = Message::new(0, 1, 42, data.clone()).expect("Message creation failed");

        let bytes = msg.to_bytes().expect("Serialization failed");
        let deserialized: Message<f64> =
            Message::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(msg.source(), deserialized.source());
        assert_eq!(msg.dest(), deserialized.dest());
        assert_eq!(msg.tag(), deserialized.tag());
        assert_eq!(msg.payload, deserialized.payload);
    }

    #[test]
    fn test_message_with_sequence() {
        let data = vec![1, 2, 3, 4];
        let msg = Message::with_sequence(0, 1, 42, data, 123).expect("Message creation failed");

        assert_eq!(msg.sequence(), 123);
    }

    #[tokio::test]
    async fn test_connection_manager_creation() {
        let manager = ConnectionManager::new(loopback())
            .await
            .expect("Manager creation failed");

        // Just verify it was created successfully
        assert!(manager.local_addr().port() > 0);
    }

    #[tokio::test]
    async fn test_register_rank() {
        let manager = ConnectionManager::new(loopback())
            .await
            .expect("Manager creation failed");

        let rank_addr: SocketAddr = "127.0.0.1:5001".parse().expect("Valid address");
        manager.register_rank(0, rank_addr).await;

        assert_eq!(manager.address_of(0).await, Some(rank_addr));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn channel_round_trips_a_message() {
        let (a, b) = channel_pair().await;
        let msg = Message::new(0, 1, 7, vec![1.5_f64, 2.5]).expect("message");
        a.send(msg).await.expect("send");
        let got: Message<f64> = b.recv().await.expect("recv");
        assert_eq!(got.payload, vec![1.5, 2.5]);
        assert_eq!(got.tag(), 7);
        assert_eq!(got.source(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn channel_preserves_send_order_across_tags() {
        let (a, b) = channel_pair().await;
        for i in 0..8u32 {
            let msg = Message::new(0, 1, i, vec![i as f64]).expect("message");
            a.send(msg).await.expect("send");
        }
        for i in 0..8u32 {
            let got: Message<f64> = b.recv().await.expect("recv");
            assert_eq!(got.tag(), i, "messages must arrive in send order");
        }
    }

    /// The regression the whole rewrite exists for.
    ///
    /// Both ends push a payload far larger than the kernel socket buffers
    /// *before* either starts receiving. The original implementation held one
    /// `Arc<Mutex<TcpStream>>` across every `.await`, so this deadlocked: each
    /// side's `send` waited on the mutex its own `recv` was holding across
    /// `read_exact`. Sends are now enqueue-and-return, so both complete.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn simultaneous_bidirectional_send_does_not_deadlock() {
        const BIG: usize = 512 * 1024;
        let (a, b) = channel_pair().await;
        let a = Arc::new(a);
        let b = Arc::new(b);

        let a_task = {
            let a = Arc::clone(&a);
            tokio::spawn(async move {
                let msg = Message::new(0, 1, 1, vec![1.0_f64; BIG]).expect("message");
                a.send(msg).await.expect("rank 0 send");
                let got: Message<f64> = a.recv().await.expect("rank 0 recv");
                got.payload.len()
            })
        };
        let b_task = {
            let b = Arc::clone(&b);
            tokio::spawn(async move {
                let msg = Message::new(1, 0, 1, vec![2.0_f64; BIG]).expect("message");
                b.send(msg).await.expect("rank 1 send");
                let got: Message<f64> = b.recv().await.expect("rank 1 recv");
                got.payload.len()
            })
        };

        let deadline = Duration::from_secs(30);
        let a_len = timeout(deadline, a_task)
            .await
            .expect("rank 0 must not deadlock")
            .expect("join");
        let b_len = timeout(deadline, b_task)
            .await
            .expect("rank 1 must not deadlock")
            .expect("join");
        assert_eq!((a_len, b_len), (BIG, BIG));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recv_reports_a_closed_peer_instead_of_hanging() {
        let (a, b) = channel_pair().await;
        drop(a);
        let err = b.recv::<f64>().await.expect_err("peer went away");
        assert!(matches!(err, CommunicationError::ConnectionClosed));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recv_timeout_expires_without_traffic() {
        let (_a, b) = channel_pair().await;
        let err = b
            .recv_timeout::<f64>(Duration::from_millis(50))
            .await
            .expect_err("nothing was sent");
        assert!(matches!(err, CommunicationError::Timeout(_)));
    }

    /// The original `ConnectionManager::recv` accepted a fresh connection per
    /// call, so a second message down the same pooled connection was never
    /// read. Both must arrive now.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn manager_reads_many_messages_from_one_pooled_connection() {
        let receiver = ConnectionManager::new(loopback()).await.expect("bind");
        let sender = ConnectionManager::new(loopback()).await.expect("bind");
        sender.register_rank(1, receiver.local_addr()).await;

        for i in 0..4usize {
            let msg = Message::new(0, 1, i as u32, vec![i as f64]).expect("message");
            sender.send(msg).await.expect("send");
        }
        for i in 0..4usize {
            let got: Message<f64> = receiver
                .recv_timeout(Duration::from_secs(10))
                .await
                .expect("recv");
            assert_eq!(got.tag(), i as u32);
        }
    }

    /// The demultiplexing the original could not do: a receive naming a source
    /// gets that source's message, and the other rank's message is still there
    /// afterwards rather than discarded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn manager_recv_from_demultiplexes_by_source() {
        let receiver = ConnectionManager::new(loopback()).await.expect("bind");
        let rank_one = ConnectionManager::new(loopback()).await.expect("bind");
        let rank_two = ConnectionManager::new(loopback()).await.expect("bind");
        rank_one.register_rank(0, receiver.local_addr()).await;
        rank_two.register_rank(0, receiver.local_addr()).await;

        // Rank 1 speaks first; the receiver asks rank 2 for its message first.
        rank_one
            .send(Message::new(1, 0, 11, vec![1.0_f64]).expect("message"))
            .await
            .expect("send");
        // Make the ordering deterministic: rank 1's frame is already in the
        // queue when rank 2's is sent.
        tokio::time::sleep(Duration::from_millis(50)).await;
        rank_two
            .send(Message::new(2, 0, 22, vec![2.0_f64]).expect("message"))
            .await
            .expect("send");

        let from_two: Message<f64> = receiver
            .recv_from_timeout(2, Duration::from_secs(10))
            .await
            .expect("recv_from rank 2");
        assert_eq!(from_two.source(), 2);
        assert_eq!(from_two.tag(), 22);

        // Rank 1's message was set aside, not dropped.
        let from_one: Message<f64> = receiver
            .recv_from_timeout(1, Duration::from_secs(10))
            .await
            .expect("recv_from rank 1");
        assert_eq!(from_one.source(), 1);
        assert_eq!(from_one.tag(), 11);
    }

    /// Two concurrent `recv_from` calls for *different* sources must both
    /// complete, in either arrival order.
    ///
    /// A naive implementation deadlocks here: whichever call grabs the inbound
    /// queue first holds it forever waiting for its own source, while the
    /// frame it is waiting for sits behind the other rank's frame that only
    /// the *other*, now-blocked call would have stashed. Turn-bounded queue
    /// access plus a stash re-checked every turn is what breaks the cycle.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_recv_from_for_two_sources_both_complete() {
        let receiver = Arc::new(ConnectionManager::new(loopback()).await.expect("bind"));
        let rank_one = ConnectionManager::new(loopback()).await.expect("bind");
        let rank_two = ConnectionManager::new(loopback()).await.expect("bind");
        rank_one.register_rank(0, receiver.local_addr()).await;
        rank_two.register_rank(0, receiver.local_addr()).await;

        // Both receives are parked before either sender speaks, so the queue
        // is genuinely contended rather than pre-filled.
        let wait_one = {
            let receiver = Arc::clone(&receiver);
            tokio::spawn(async move { receiver.recv_from::<f64>(1).await })
        };
        let wait_two = {
            let receiver = Arc::clone(&receiver);
            tokio::spawn(async move { receiver.recv_from::<f64>(2).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Rank 2 speaks first, so the receive that wants rank 1 is the one
        // most likely to be holding the queue when the "wrong" frame lands.
        rank_two
            .send(Message::new(2, 0, 22, vec![2.0_f64]).expect("message"))
            .await
            .expect("send");
        rank_one
            .send(Message::new(1, 0, 11, vec![1.0_f64]).expect("message"))
            .await
            .expect("send");

        let deadline = Duration::from_secs(20);
        let got_one = timeout(deadline, wait_one)
            .await
            .expect("the receive for rank 1 must not deadlock")
            .expect("join")
            .expect("recv_from rank 1");
        let got_two = timeout(deadline, wait_two)
            .await
            .expect("the receive for rank 2 must not deadlock")
            .expect("join")
            .expect("recv_from rank 2");

        assert_eq!((got_one.source(), got_one.tag()), (1, 11));
        assert_eq!((got_two.source(), got_two.tag()), (2, 22));
    }

    /// A `recv_from` that gives up must not take an unrelated rank's message
    /// down with it: the frame it set aside is still there for the next call.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cancelled_recv_from_does_not_lose_the_frames_it_stashed() {
        let receiver = ConnectionManager::new(loopback()).await.expect("bind");
        let rank_one = ConnectionManager::new(loopback()).await.expect("bind");
        rank_one.register_rank(0, receiver.local_addr()).await;

        rank_one
            .send(Message::new(1, 0, 11, vec![1.0_f64]).expect("message"))
            .await
            .expect("send");

        // Nobody will ever send as rank 2, so this expires — having stashed
        // rank 1's message on the way.
        let err = receiver
            .recv_from_timeout::<f64>(2, Duration::from_millis(300))
            .await
            .expect_err("rank 2 never sends");
        assert!(matches!(err, CommunicationError::Timeout(_)));

        let got: Message<f64> = receiver
            .recv_from_timeout(1, Duration::from_secs(10))
            .await
            .expect("rank 1's message survived the cancelled receive");
        assert_eq!((got.source(), got.tag()), (1, 11));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn manager_send_to_unregistered_rank_is_rejected() {
        let manager = ConnectionManager::new(loopback()).await.expect("bind");
        let err = manager
            .send(Message::new(0, 9, 0, vec![1.0_f64]).expect("message"))
            .await
            .expect_err("rank 9 was never registered");
        assert!(matches!(err, CommunicationError::InvalidRank(9)));
    }

    #[test]
    fn net_errors_map_onto_the_legacy_surface() {
        assert!(matches!(
            CommunicationError::from(NetError::PeerClosed),
            CommunicationError::ConnectionClosed
        ));
        assert!(matches!(
            CommunicationError::from(NetError::FrameTooLarge { size: 9, max: 4 }),
            CommunicationError::BufferOverflow {
                size: 9,
                capacity: 4
            }
        ));
        assert!(matches!(
            CommunicationError::from(NetError::InvalidRank { rank: 3, size: 2 }),
            CommunicationError::InvalidRank(3)
        ));
    }
}
