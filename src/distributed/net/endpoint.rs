//! The bound local socket that owns every peer [`super::link::Link`] and
//! demuxes inbound frames into a per-key [`super::mailbox::Mailbox`].
//!
//! Replaces [`super::super::comm::ConnectionManager`], whose `recv` did one
//! `accept()` per call and so could not demultiplex messages by source rank —
//! whichever connection happened to arrive serviced the call, regardless of
//! which rank the caller was actually waiting to hear from. `Endpoint`
//! instead runs one persistent reader task per peer link, feeding decoded
//! frames into a mailbox keyed by `(src, ctx, tag)`, and one accept loop that
//! turns each inbound connection into such a link.
//!
//! # Guarantees
//!
//! - **Send never blocks on the socket.** [`Endpoint::send_bytes`] hands the
//!   frame to that peer's writer task and returns; only the bounded queue can
//!   apply backpressure. Two ranks sending to each other simultaneously
//!   therefore both complete — the deadlock in `comm.rs` is gone by
//!   construction, not by careful call ordering.
//! - **Recv is demultiplexed and bounded.** A receive names its
//!   `(src, ctx, tag)` and can only ever be handed a message with exactly
//!   that key, in FIFO order, within a deadline. Expiry names the key.
//! - **Self-send never touches the network.** `dst == rank` goes straight
//!   into the local mailbox, so it works at `world_size == 1` with zero
//!   links.
//! - **Compression is opt-in and never lossy about it.** A payload is
//!   LZ4-compressed only when [`SendOpts::compress`] is set *and* it reaches
//!   [`EndpointConfig::compress_threshold`] *and* compression actually made
//!   it smaller; the [`FLAG_COMPRESSED`] bit always matches what is on the
//!   wire. A latency probe passing `SendOpts::default()` is never compressed
//!   at any size.
//! - **Oversized frames error, never truncate**, on both the send and the
//!   receive side, against [`EndpointConfig::max_frame`].

use super::frame::{FrameHeader, CTX_CONTROL, FLAG_COMPRESSED, TAG_HELLO};
use super::link::Link;
use super::mailbox::Mailbox;
use super::{EndpointConfig, NetError, SendOpts};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

/// Shared state behind an [`Endpoint`]. Held by `Arc` so links, the accept
/// loop (through a `Weak`), and clones of the endpoint all see one runtime.
struct EndpointInner {
    rank: u32,
    world_size: u32,
    local_addr: SocketAddr,
    config: EndpointConfig,
    mailbox: Arc<Mailbox<Vec<u8>>>,
    /// The link this endpoint *writes* to for each peer.
    links: RwLock<HashMap<u32, Arc<Link>>>,
    /// Every link ever created, kept alive so its reader task keeps
    /// delivering even if a second connection to the same peer later won the
    /// `links` slot (possible when both ranks dial each other at once).
    retained: Mutex<Vec<Arc<Link>>>,
    peers: RwLock<HashMap<u32, SocketAddr>>,
    /// Serializes dial-on-demand so two concurrent sends to a cold peer do
    /// not open two connections.
    dialing: tokio::sync::Mutex<()>,
    accept_task: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for EndpointInner {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.accept_task.lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
        self.mailbox.close();
    }
}

/// A bound local endpoint managing connections to every other rank.
///
/// Cheap to clone: every clone shares one accept loop, one mailbox, and one
/// set of links.
#[derive(Clone)]
pub struct Endpoint {
    inner: Arc<EndpointInner>,
}

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint")
            .field("rank", &self.inner.rank)
            .field("world_size", &self.inner.world_size)
            .field("local_addr", &self.inner.local_addr)
            .finish()
    }
}

impl Endpoint {
    /// Bind a new endpoint at `addr` (use `127.0.0.1:0` for an OS-assigned
    /// ephemeral port) as `rank` of `world_size`.
    pub async fn bind(
        addr: SocketAddr,
        rank: u32,
        world_size: u32,
        config: EndpointConfig,
    ) -> Result<Self, NetError> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| NetError::ConnectFailed {
                addr: addr.to_string(),
                msg: e.to_string(),
            })?;
        Self::from_listener(listener, rank, world_size, config)
    }

    /// Adopt an already-bound listener.
    ///
    /// This is the constructor a harness must use when something else has
    /// already published this rank's address (see
    /// [`super::super::testing::RankContext`]): binding a *second* socket
    /// would leave peers dialing an address nobody is accepting on.
    pub fn from_listener(
        listener: TcpListener,
        rank: u32,
        world_size: u32,
        config: EndpointConfig,
    ) -> Result<Self, NetError> {
        if world_size == 0 || rank >= world_size {
            return Err(NetError::InvalidRank {
                rank,
                size: world_size,
            });
        }
        let local_addr = listener
            .local_addr()
            .map_err(|e| NetError::Io(e.to_string()))?;

        let inner = Arc::new(EndpointInner {
            rank,
            world_size,
            local_addr,
            mailbox: Arc::new(Mailbox::with_key_capacity(Self::key_capacity_for(&config))),
            links: RwLock::new(HashMap::new()),
            retained: Mutex::new(Vec::new()),
            peers: RwLock::new(HashMap::new()),
            dialing: tokio::sync::Mutex::new(()),
            accept_task: Mutex::new(None),
            config,
        });

        // The accept loop holds only a `Weak`, so the endpoint's refcount can
        // still reach zero; `EndpointInner::drop` then aborts the task.
        let weak = Arc::downgrade(&inner);
        let handle = tokio::spawn(accept_loop(weak, listener));
        if let Ok(mut guard) = inner.accept_task.lock() {
            *guard = Some(handle);
        }

        Ok(Self { inner })
    }

    /// Per-key mailbox ceiling this endpoint uses, derived from
    /// [`EndpointConfig::queue_depth`].
    ///
    /// `queue_depth` bounds *in-flight sends per peer*; a receive queue must
    /// be able to absorb a good deal more than that before it is fair to
    /// call it a runaway, so this scales it up and applies a floor.
    pub fn key_capacity_for(config: &EndpointConfig) -> usize {
        config
            .queue_depth
            .saturating_mul(64)
            .max(super::mailbox::DEFAULT_KEY_CAPACITY)
    }

    /// The address this endpoint is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.inner.local_addr
    }

    /// This endpoint's rank.
    pub fn rank(&self) -> u32 {
        self.inner.rank
    }

    /// The world size this endpoint was created with.
    pub fn world_size(&self) -> u32 {
        self.inner.world_size
    }

    /// This endpoint's configuration.
    pub fn config(&self) -> &EndpointConfig {
        &self.inner.config
    }

    /// Register the address of peer `rank` so future sends can connect to it.
    pub async fn register_peer(&self, rank: u32, addr: SocketAddr) -> Result<(), NetError> {
        if rank >= self.inner.world_size {
            return Err(NetError::InvalidRank {
                rank,
                size: self.inner.world_size,
            });
        }
        if rank == self.inner.rank {
            return Ok(());
        }
        self.inner.peers.write().await.insert(rank, addr);
        Ok(())
    }

    /// Number of peers this endpoint currently holds a link to.
    pub async fn link_count(&self) -> usize {
        self.inner.links.read().await.len()
    }

    /// Register every rank's address and establish the full mesh.
    ///
    /// Pair rule: **`j` dials `i` whenever `i < j`**, so each unordered pair
    /// opens exactly one connection and neither side has to guess. Every
    /// dialer's first frame is a HELLO carrying its own rank in the header's
    /// `src`, which is how the accepting side knows who just arrived.
    ///
    /// Returns only once this rank holds `world_size - 1` links — dialed to
    /// every lower rank and accepted from every higher one — so a `send`
    /// immediately afterwards can never find a missing link. `world_size == 1`
    /// short-circuits: there is nobody to dial and nothing to wait for.
    pub async fn connect_mesh(&self, addrs: &[SocketAddr]) -> Result<(), NetError> {
        let world_size = self.inner.world_size;
        if world_size <= 1 {
            return Ok(());
        }
        if addrs.len() != world_size as usize {
            return Err(NetError::Bootstrap(format!(
                "address table has {} entries but world size is {world_size}",
                addrs.len()
            )));
        }
        for (rank, addr) in addrs.iter().enumerate() {
            let rank = u32::try_from(rank).map_err(|_| NetError::InvalidRank {
                rank: u32::MAX,
                size: world_size,
            })?;
            self.register_peer(rank, *addr).await?;
        }

        for peer in 0..self.inner.rank {
            self.connect_peer(peer).await?;
        }

        // Now wait for every higher rank to dial us.
        let expected = world_size as usize - 1;
        let deadline = tokio::time::Instant::now() + self.inner.config.connect_timeout;
        let mut backoff = Duration::from_micros(200);
        loop {
            if self.link_count().await >= expected {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                let have = self.link_count().await;
                return Err(NetError::Bootstrap(format!(
                    "rank {} timed out establishing the mesh: {have} of {expected} link(s) after {:?}",
                    self.inner.rank, self.inner.config.connect_timeout
                )));
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_millis(5));
        }
    }

    /// Dial peer `rank` (whose address must already be registered) and greet
    /// it with a HELLO frame. A no-op if a link already exists.
    pub async fn connect_peer(&self, rank: u32) -> Result<(), NetError> {
        let _ = self.link_for(rank).await?;
        Ok(())
    }

    /// The link to write to for `rank`, dialing on demand.
    async fn link_for(&self, rank: u32) -> Result<Arc<Link>, NetError> {
        if let Some(link) = self.inner.links.read().await.get(&rank) {
            return Ok(Arc::clone(link));
        }

        // One dial at a time; re-check under the guard in case another task
        // just finished connecting to the same peer.
        let _guard = self.inner.dialing.lock().await;
        if let Some(link) = self.inner.links.read().await.get(&rank) {
            return Ok(Arc::clone(link));
        }

        let addr = self
            .inner
            .peers
            .read()
            .await
            .get(&rank)
            .copied()
            .ok_or_else(|| NetError::NotConnected {
                rank,
                reason: "no address registered for this rank".to_string(),
            })?;

        let link = Arc::new(
            Link::connect(
                addr,
                rank,
                Arc::clone(&self.inner.mailbox),
                &self.inner.config,
            )
            .await?,
        );

        // Announce who we are before anything else goes down this socket.
        let hello = FrameHeader::new(
            self.inner.rank,
            rank,
            CTX_CONTROL,
            TAG_HELLO,
            link.next_seq(),
            0,
            0,
            0,
        )?;
        link.send_frame(hello, Vec::new()).await?;

        self.retain_link(Arc::clone(&link));
        let mut links = self.inner.links.write().await;
        let chosen = Arc::clone(links.entry(rank).or_insert_with(|| Arc::clone(&link)));
        Ok(chosen)
    }

    /// Keep a link (and therefore its reader task) alive for the lifetime of
    /// the endpoint, whether or not it won the `links` slot.
    fn retain_link(&self, link: Arc<Link>) {
        if let Ok(mut guard) = self.inner.retained.lock() {
            guard.push(link);
        }
    }

    /// Send `payload` to rank `dst` under `(ctx, tag)`.
    ///
    /// See [`SendOpts::compress`] for exactly when the payload is compressed.
    /// `dst == self.rank()` is a loopback delivery that never touches the
    /// network.
    pub async fn send_bytes(
        &self,
        dst: u32,
        ctx: u64,
        tag: u64,
        payload: &[u8],
        opts: SendOpts,
    ) -> Result<(), NetError> {
        self.check_send_target(dst, ctx, payload.len())?;
        if dst == self.inner.rank {
            return self.deliver_locally(ctx, tag, payload.to_vec());
        }
        match self.compress_if_worthwhile(payload, opts)? {
            Some(compressed) => {
                self.send_framed(dst, ctx, tag, compressed, payload.len(), FLAG_COMPRESSED)
                    .await
            }
            None => {
                self.send_framed(dst, ctx, tag, payload.to_vec(), payload.len(), 0)
                    .await
            }
        }
    }

    /// As [`Self::send_bytes`], taking ownership of the payload so an
    /// uncompressed send does not copy it.
    pub async fn send_owned(
        &self,
        dst: u32,
        ctx: u64,
        tag: u64,
        payload: Vec<u8>,
        opts: SendOpts,
    ) -> Result<(), NetError> {
        self.check_send_target(dst, ctx, payload.len())?;
        if dst == self.inner.rank {
            return self.deliver_locally(ctx, tag, payload);
        }
        let raw_len = payload.len();
        match self.compress_if_worthwhile(&payload, opts)? {
            Some(compressed) => {
                self.send_framed(dst, ctx, tag, compressed, raw_len, FLAG_COMPRESSED)
                    .await
            }
            None => self.send_framed(dst, ctx, tag, payload, raw_len, 0).await,
        }
    }

    /// Validate a send's destination, context, and size before any work.
    fn check_send_target(&self, dst: u32, ctx: u64, len: usize) -> Result<(), NetError> {
        if dst >= self.inner.world_size {
            return Err(NetError::InvalidRank {
                rank: dst,
                size: self.inner.world_size,
            });
        }
        if ctx == CTX_CONTROL {
            return Err(NetError::MalformedFrame(format!(
                "ctx {CTX_CONTROL} is reserved for transport control frames"
            )));
        }
        if len > self.inner.config.max_frame {
            return Err(NetError::FrameTooLarge {
                size: len,
                max: self.inner.config.max_frame,
            });
        }
        Ok(())
    }

    /// Loopback: straight into this endpoint's own mailbox under `src ==
    /// self.rank`, with no framing, no compression, and no socket.
    fn deliver_locally(&self, ctx: u64, tag: u64, payload: Vec<u8>) -> Result<(), NetError> {
        self.inner
            .mailbox
            .push((self.inner.rank, ctx, tag), payload)
    }

    /// `Some(compressed)` when compression is permitted, the payload is big
    /// enough to be worth it, and it actually got smaller; `None` means send
    /// the payload raw.
    ///
    /// LZ4 can *expand* incompressible input, so a compressed result that is
    /// no smaller than the original is discarded rather than paying to send
    /// more bytes than we started with.
    fn compress_if_worthwhile(
        &self,
        payload: &[u8],
        opts: SendOpts,
    ) -> Result<Option<Vec<u8>>, NetError> {
        if !opts.compress || payload.len() < self.inner.config.compress_threshold {
            return Ok(None);
        }
        let compressed = oxiarc_lz4::block::compress_block(payload)
            .map_err(|e| NetError::Compression(e.to_string()))?;
        if compressed.len() >= payload.len() {
            return Ok(None);
        }
        Ok(Some(compressed))
    }

    /// Frame `wire` and hand it to `dst`'s writer task.
    async fn send_framed(
        &self,
        dst: u32,
        ctx: u64,
        tag: u64,
        wire: Vec<u8>,
        raw_len: usize,
        flags: u16,
    ) -> Result<(), NetError> {
        if wire.len() > self.inner.config.max_frame {
            return Err(NetError::FrameTooLarge {
                size: wire.len(),
                max: self.inner.config.max_frame,
            });
        }
        let wire_len = u32::try_from(wire.len()).map_err(|_| NetError::FrameTooLarge {
            size: wire.len(),
            max: self.inner.config.max_frame,
        })?;
        let raw_len_u32 = u32::try_from(raw_len).map_err(|_| NetError::FrameTooLarge {
            size: raw_len,
            max: self.inner.config.max_frame,
        })?;

        let link = self.link_for(dst).await?;
        let header = FrameHeader::new(
            self.inner.rank,
            dst,
            ctx,
            tag,
            link.next_seq(),
            wire_len,
            raw_len_u32,
            flags,
        )?;
        link.send_frame(header, wire).await
    }

    /// Receive the next payload queued from rank `src` under `(ctx, tag)`,
    /// waiting up to [`EndpointConfig::recv_timeout`].
    pub async fn recv_bytes(&self, src: u32, ctx: u64, tag: u64) -> Result<Vec<u8>, NetError> {
        self.recv_bytes_timeout(src, ctx, tag, self.inner.config.recv_timeout)
            .await
    }

    /// As [`Self::recv_bytes`] with an explicit deadline. On expiry the error
    /// is [`NetError::RecvTimeout`], which names the exact `(src, ctx, tag)`
    /// that never arrived.
    pub async fn recv_bytes_timeout(
        &self,
        src: u32,
        ctx: u64,
        tag: u64,
        timeout: Duration,
    ) -> Result<Vec<u8>, NetError> {
        if src >= self.inner.world_size {
            return Err(NetError::InvalidRank {
                rank: src,
                size: self.inner.world_size,
            });
        }
        self.inner
            .mailbox
            .pop_timeout((src, ctx, tag), timeout)
            .await
    }

    /// Take an already-delivered payload for `(src, ctx, tag)` without
    /// waiting.
    pub fn try_recv_bytes(
        &self,
        src: u32,
        ctx: u64,
        tag: u64,
    ) -> Result<Option<Vec<u8>>, NetError> {
        self.inner.mailbox.try_pop((src, ctx, tag))
    }

    /// Number of payloads already delivered and waiting under
    /// `(src, ctx, tag)`.
    pub fn pending(&self, src: u32, ctx: u64, tag: u64) -> Result<usize, NetError> {
        self.inner.mailbox.len((src, ctx, tag))
    }

    /// Shut down every managed link, stop accepting, and wake every parked
    /// receive with [`NetError::MailboxClosed`]. Idempotent.
    pub async fn shutdown(&self) -> Result<(), NetError> {
        if let Ok(mut guard) = self.inner.accept_task.lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
        let links: Vec<Arc<Link>> = match self.inner.retained.lock() {
            Ok(mut guard) => guard.drain(..).collect(),
            Err(poisoned) => poisoned.into_inner().drain(..).collect(),
        };
        // Best effort: a peer that already vanished must not stop us from
        // closing the rest.
        for link in &links {
            let _ = link.close().await;
        }
        self.inner.links.write().await.clear();
        self.inner.mailbox.close();
        Ok(())
    }
}

/// Accept connections until the endpoint goes away, turning each into a link.
async fn accept_loop(endpoint: Weak<EndpointInner>, listener: TcpListener) {
    loop {
        let accepted = listener.accept().await;
        let Some(inner) = endpoint.upgrade() else {
            return;
        };
        match accepted {
            Ok((stream, addr)) => {
                let inner = Arc::clone(&inner);
                // Greeting a new peer must not stall the accept loop: a slow
                // or malicious dialer would otherwise hold every other rank
                // out of the mesh.
                tokio::spawn(async move {
                    let _ = adopt_inbound(inner, stream, addr).await;
                });
            }
            // A transient accept error (fd pressure, RST between the SYN and
            // the accept) should not tear the endpoint down.
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }
}

/// Read the dialer's HELLO, then register the connection as that rank's link.
async fn adopt_inbound(
    inner: Arc<EndpointInner>,
    mut stream: TcpStream,
    addr: SocketAddr,
) -> Result<(), NetError> {
    let (header, _payload) = tokio::time::timeout(
        inner.config.connect_timeout,
        super::link::read_frame(&mut stream, inner.config.max_frame),
    )
    .await
    .map_err(|_| NetError::Timeout(inner.config.connect_timeout))??;

    if header.ctx != CTX_CONTROL || header.tag != TAG_HELLO {
        return Err(NetError::MalformedFrame(format!(
            "expected a HELLO frame first, got ctx {} tag {}",
            header.ctx, header.tag
        )));
    }
    let peer = header.src;
    if peer >= inner.world_size || peer == inner.rank {
        return Err(NetError::InvalidRank {
            rank: peer,
            size: inner.world_size,
        });
    }

    let link = Arc::new(Link::from_stream(
        stream,
        peer,
        addr,
        Arc::clone(&inner.mailbox),
        &inner.config,
    ));
    match inner.retained.lock() {
        Ok(mut guard) => guard.push(Arc::clone(&link)),
        Err(poisoned) => poisoned.into_inner().push(Arc::clone(&link)),
    }
    inner
        .links
        .write()
        .await
        .entry(peer)
        .or_insert_with(|| Arc::clone(&link));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loopback() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 0))
    }

    /// Two endpoints, fully meshed, as ranks 0 and 1.
    async fn pair() -> (Endpoint, Endpoint) {
        let cfg = EndpointConfig::default();
        let a = Endpoint::bind(loopback(), 0, 2, cfg.clone())
            .await
            .expect("bind rank 0");
        let b = Endpoint::bind(loopback(), 1, 2, cfg)
            .await
            .expect("bind rank 1");
        let addrs = vec![a.local_addr(), b.local_addr()];
        let (ra, rb) = tokio::join!(a.connect_mesh(&addrs), b.connect_mesh(&addrs));
        ra.expect("mesh rank 0");
        rb.expect("mesh rank 1");
        (a, b)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn point_to_point_round_trip() {
        let (a, b) = pair().await;
        a.send_bytes(1, 7, 3, b"ping", SendOpts::default())
            .await
            .expect("send");
        let got = b.recv_bytes(0, 7, 3).await.expect("recv");
        assert_eq!(got, b"ping".to_vec());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn self_send_never_touches_the_network() {
        let cfg = EndpointConfig::default();
        // world_size 1: there is no peer, no mesh, and no link at all.
        let solo = Endpoint::bind(loopback(), 0, 1, cfg)
            .await
            .expect("bind solo");
        solo.connect_mesh(&[solo.local_addr()])
            .await
            .expect("single-process mesh is a no-op");
        assert_eq!(solo.link_count().await, 0);

        solo.send_bytes(0, 1, 2, b"loopback", SendOpts::default())
            .await
            .expect("self send");
        let got = solo.recv_bytes(0, 1, 2).await.expect("self recv");
        assert_eq!(got, b"loopback".to_vec());
        assert_eq!(solo.link_count().await, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tags_are_demultiplexed_out_of_order() {
        let (a, b) = pair().await;
        // Tag 2 goes on the wire first; the receiver asks for tag 1 first.
        a.send_bytes(1, 0, 2, b"second-tag", SendOpts::default())
            .await
            .expect("send tag 2");
        a.send_bytes(1, 0, 1, b"first-tag", SendOpts::default())
            .await
            .expect("send tag 1");

        let one = b.recv_bytes(0, 0, 1).await.expect("recv tag 1");
        assert_eq!(one, b"first-tag".to_vec());
        let two = b.recv_bytes(0, 0, 2).await.expect("recv tag 2");
        assert_eq!(two, b"second-tag".to_vec());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn contexts_are_demultiplexed_too() {
        let (a, b) = pair().await;
        a.send_bytes(1, 100, 1, b"ctx-100", SendOpts::default())
            .await
            .expect("send");
        a.send_bytes(1, 200, 1, b"ctx-200", SendOpts::default())
            .await
            .expect("send");
        assert_eq!(
            b.recv_bytes(0, 200, 1).await.expect("recv"),
            b"ctx-200".to_vec()
        );
        assert_eq!(
            b.recv_bytes(0, 100, 1).await.expect("recv"),
            b"ctx-100".to_vec()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recv_timeout_names_the_missing_key() {
        let (_a, b) = pair().await;
        let err = b
            .recv_bytes_timeout(0, 11, 22, Duration::from_millis(50))
            .await
            .expect_err("nothing was sent");
        match err {
            NetError::RecvTimeout { src, ctx, tag, .. } => assert_eq!((src, ctx, tag), (0, 11, 22)),
            other => panic!("expected RecvTimeout, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn oversized_send_errors_instead_of_truncating() {
        let cfg = EndpointConfig {
            max_frame: 1024,
            ..EndpointConfig::default()
        };
        let ep = Endpoint::bind(loopback(), 0, 2, cfg).await.expect("bind");
        let err = ep
            .send_bytes(1, 0, 0, &vec![0u8; 2048], SendOpts::default())
            .await
            .expect_err("over the endpoint's policy limit");
        assert!(matches!(
            err,
            NetError::FrameTooLarge {
                size: 2048,
                max: 1024
            }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn send_to_an_out_of_world_rank_is_rejected() {
        let cfg = EndpointConfig::default();
        let ep = Endpoint::bind(loopback(), 0, 2, cfg).await.expect("bind");
        let err = ep
            .send_bytes(9, 0, 0, b"x", SendOpts::default())
            .await
            .expect_err("rank 9 is not in a world of 2");
        assert!(matches!(err, NetError::InvalidRank { rank: 9, size: 2 }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reserved_control_context_is_refused_to_user_sends() {
        let cfg = EndpointConfig::default();
        let ep = Endpoint::bind(loopback(), 0, 2, cfg).await.expect("bind");
        let err = ep
            .send_bytes(1, CTX_CONTROL, 0, b"x", SendOpts::default())
            .await
            .expect_err("control ctx is reserved");
        assert!(matches!(err, NetError::MalformedFrame(_)));
    }

    // ===============================================================
    // Compression, observed on the wire rather than inferred.
    //
    // A round-trip assertion cannot tell a compressed frame from an
    // uncompressed one — both decode back to the same bytes. These tests
    // stand a raw `TcpListener` in for rank 1 and read the frame exactly as
    // it left this endpoint, so the FLAG_COMPRESSED bit and the
    // wire_len/raw_len pair are checked against what actually went out.
    // ===============================================================

    /// Drive one `send_bytes` at a raw socket standing in for rank 1 and hand
    /// back the data frame's header (as it appeared on the wire) together with
    /// the payload the receive path recovered from it. The HELLO the dialer
    /// sends first is consumed and checked on the way.
    async fn wire_frame_for(
        payload: &[u8],
        opts: SendOpts,
        config: EndpointConfig,
    ) -> (FrameHeader, Vec<u8>) {
        let peer = TcpListener::bind(loopback()).await.expect("bind fake peer");
        let peer_addr = peer.local_addr().expect("peer addr");
        let max_frame = config.max_frame;
        let endpoint = Endpoint::bind(loopback(), 0, 2, config)
            .await
            .expect("bind endpoint");
        endpoint
            .register_peer(1, peer_addr)
            .await
            .expect("register peer");

        let owned = payload.to_vec();
        let sender = {
            let endpoint = endpoint.clone();
            tokio::spawn(async move { endpoint.send_bytes(1, 0, 1, &owned, opts).await })
        };

        let (mut stream, _) = peer.accept().await.expect("accept the dial");
        let (hello, _) = crate::distributed::net::link::read_frame(&mut stream, max_frame)
            .await
            .expect("HELLO frame");
        assert_eq!(
            hello.ctx, CTX_CONTROL,
            "the dialer greets before anything else"
        );
        assert_eq!(hello.tag, TAG_HELLO);

        let frame = crate::distributed::net::link::read_frame(&mut stream, max_frame)
            .await
            .expect("data frame");
        sender.await.expect("join").expect("send");
        frame
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn large_compressible_payload_goes_on_the_wire_compressed() {
        let payload = vec![0xABu8; 256 * 1024];
        let (header, decoded) = wire_frame_for(
            &payload,
            SendOpts { compress: true },
            EndpointConfig::default(),
        )
        .await;

        assert!(
            header.is_compressed(),
            "a 256 KiB constant payload above the threshold must be compressed"
        );
        assert!(
            (header.wire_len as usize) < payload.len(),
            "wire_len {} should be well under raw_len {}",
            header.wire_len,
            payload.len()
        );
        assert_eq!(header.raw_len as usize, payload.len());
        assert_eq!(decoded, payload, "the round trip must be byte-exact");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compression_opt_out_puts_a_large_payload_on_the_wire_raw() {
        // The latency-probe path: the same payload that compresses 100:1 above
        // must go out untouched when the caller says so.
        let payload = vec![0xABu8; 256 * 1024];
        let (header, decoded) = wire_frame_for(
            &payload,
            SendOpts { compress: false },
            EndpointConfig::default(),
        )
        .await;

        assert!(
            !header.is_compressed(),
            "compress: false must never set FLAG_COMPRESSED"
        );
        assert_eq!(header.wire_len as usize, payload.len());
        assert_eq!(header.raw_len, header.wire_len);
        assert_eq!(decoded, payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn incompressible_payload_falls_back_to_raw_even_when_permitted() {
        // LZ4 *expands* this input, so compressing it would put more bytes on
        // the wire than sending it raw. The flag must match what was actually
        // sent, not what was requested.
        let payload = crate::distributed::testing::incompressible_bytes(256 * 1024);
        let (header, decoded) = wire_frame_for(
            &payload,
            SendOpts { compress: true },
            EndpointConfig::default(),
        )
        .await;

        assert!(
            !header.is_compressed(),
            "an incompressible payload must be sent raw, not expanded"
        );
        assert_eq!(header.wire_len as usize, payload.len());
        assert_eq!(header.raw_len, header.wire_len);
        assert_eq!(decoded, payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn payload_below_the_threshold_is_never_compressed() {
        // Highly compressible, but only 1 KiB: below compress_threshold the
        // overhead is not worth it, so `compress: true` still sends it raw.
        let payload = vec![0u8; 1024];
        assert!(payload.len() < EndpointConfig::default().compress_threshold);
        let (header, decoded) = wire_frame_for(
            &payload,
            SendOpts { compress: true },
            EndpointConfig::default(),
        )
        .await;

        assert!(!header.is_compressed());
        assert_eq!(header.wire_len as usize, payload.len());
        assert_eq!(decoded, payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compressed_frame_survives_a_full_endpoint_round_trip() {
        // The wire-level tests above prove the flag; this proves the receiving
        // endpoint decompresses what the flag promises.
        let (a, b) = pair().await;
        let payload = vec![0x7Fu8; 512 * 1024];
        a.send_bytes(1, 0, 5, &payload, SendOpts { compress: true })
            .await
            .expect("send");
        let got = b.recv_bytes(0, 0, 5).await.expect("recv");
        assert_eq!(got, payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn shutdown_wakes_parked_receives() {
        let (a, b) = pair().await;
        let waiter = {
            let b = b.clone();
            tokio::spawn(
                async move { b.recv_bytes_timeout(0, 0, 0, Duration::from_secs(30)).await },
            )
        };
        tokio::task::yield_now().await;
        b.shutdown().await.expect("shutdown");
        let err = waiter.await.expect("join").expect_err("closed");
        assert!(matches!(err, NetError::MailboxClosed));
        a.shutdown().await.expect("shutdown");
    }
}
