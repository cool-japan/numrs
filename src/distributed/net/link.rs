//! One point-to-point connection to a single peer.
//!
//! Replaces [`super::super::comm::CommunicationChannel`], whose `send` and
//! `recv` shared one `Arc<Mutex<TcpStream>>` held across `.await` points —
//! that is what deadlocks on bidirectional traffic: a `recv()` in progress
//! held the lock across `read_exact().await`, so a concurrent `send()` on
//! the same channel blocked forever waiting for the same mutex, even though
//! a real socket's read and write directions do not contend at all.
//!
//! # Shape
//!
//! [`Link::from_stream`] splits the accepted/dialed `TcpStream` into owned
//! halves and spawns two independent tasks:
//!
//! - the **writer** drains a bounded [`tokio::sync::mpsc`] channel
//!   (`queue_depth` frames deep) and serializes each frame as
//!   `header.encode()` followed by exactly `header.wire_len` payload bytes.
//!   The 56-byte header *is* the length prefix — there is no second one.
//! - the **reader** parses frames off the wire forever and routes each
//!   decoded payload into the shared [`super::mailbox::Mailbox`] under
//!   `(header.src, header.ctx, header.tag)`.
//!
//! Neither task ever takes a lock the other one needs, and
//! [`Link::send_frame`] is *enqueue-and-return*: it awaits only mpsc
//! capacity, never the socket. A rank that sends to a peer which is
//! simultaneously sending to it therefore cannot deadlock — the reader task
//! keeps draining the socket regardless of whether user code has called
//! `recv` yet.

use super::frame::{FrameHeader, CTX_CONTROL};
use super::mailbox::Mailbox;
use super::{EndpointConfig, NetError};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

/// How long [`Link::close`] waits for the writer task to drain and shut the
/// socket down before aborting it outright.
const CLOSE_GRACE: Duration = Duration::from_millis(500);

/// What the writer task consumes.
enum Cmd {
    /// Serialize this frame onto the socket.
    Frame(FrameHeader, Vec<u8>),
    /// Flush what is queued, then shut the write half down (sends FIN).
    Shutdown,
}

/// One bidirectional, framed connection to a single peer.
///
/// Dropping a `Link` aborts both of its background tasks. Prefer
/// [`Link::close`] for a graceful shutdown that flushes queued frames first.
pub struct Link {
    peer: u32,
    peer_addr: SocketAddr,
    tx: mpsc::Sender<Cmd>,
    reader: JoinHandle<()>,
    writer: JoinHandle<()>,
    /// Resolves (as a receive error) once the writer task has exited.
    writer_done: watch::Receiver<()>,
    /// Per-`(src, dst)` monotonically increasing sequence number.
    seq: AtomicU64,
    send_timeout: Duration,
}

impl std::fmt::Debug for Link {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Link")
            .field("peer", &self.peer)
            .field("peer_addr", &self.peer_addr)
            .field("seq", &self.seq.load(Ordering::Relaxed))
            .finish()
    }
}

impl Link {
    /// Dial `addr` and wrap the resulting stream, routing inbound frames
    /// into `mailbox`.
    ///
    /// Retries a refused connection until `config.connect_timeout` expires:
    /// during bootstrap the dialer routinely wins the race against the
    /// listener it is trying to reach.
    pub async fn connect(
        addr: SocketAddr,
        peer: u32,
        mailbox: Arc<Mailbox<Vec<u8>>>,
        config: &EndpointConfig,
    ) -> Result<Self, NetError> {
        let stream = connect_with_retry(addr, config.connect_timeout).await?;
        Ok(Self::from_stream(stream, peer, addr, mailbox, config))
    }

    /// Wrap an already-established stream (dialed or accepted), spawning the
    /// reader and writer tasks.
    pub fn from_stream(
        stream: TcpStream,
        peer: u32,
        peer_addr: SocketAddr,
        mailbox: Arc<Mailbox<Vec<u8>>>,
        config: &EndpointConfig,
    ) -> Self {
        // Latency over throughput: frames are already batched by the mpsc
        // queue, so Nagle would only add delay.
        let _ = stream.set_nodelay(true);
        let (read_half, write_half) = stream.into_split();

        let (tx, rx) = mpsc::channel::<Cmd>(config.queue_depth.max(1));
        let (done_tx, writer_done) = watch::channel(());
        let writer = tokio::spawn(async move {
            writer_loop(write_half, rx).await;
            drop(done_tx);
        });

        let max_frame = config.max_frame;
        let reader = tokio::spawn(async move {
            // A reader that stops (peer closed, malformed frame, oversized
            // frame) simply ends: pending receives on this peer surface as
            // `RecvTimeout`, which names the key that never arrived.
            let _ = reader_loop(read_half, peer, mailbox, max_frame).await;
        });

        Self {
            peer,
            peer_addr,
            tx,
            reader,
            writer,
            writer_done,
            seq: AtomicU64::new(0),
            send_timeout: config.send_timeout,
        }
    }

    /// The peer rank on the other end.
    pub fn peer(&self) -> u32 {
        self.peer
    }

    /// The peer's address.
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Take the next sequence number for this link.
    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Whether this link's reader task has stopped — the peer closed the
    /// connection, the frame stream went bad, or the link was closed.
    ///
    /// Nothing more will ever be delivered into the mailbox from this peer
    /// once this is true, so a caller that would otherwise wait indefinitely
    /// (see [`super::super::comm::CommunicationChannel::recv`]) can use it to
    /// report a dead peer instead of hanging forever.
    pub fn reader_is_finished(&self) -> bool {
        self.reader.is_finished()
    }

    /// Whether this link's writer task has stopped. A `send_frame` after this
    /// fails with [`NetError::NotConnected`] rather than queueing forever.
    pub fn writer_is_finished(&self) -> bool {
        self.writer.is_finished()
    }

    /// Enqueue one frame (header plus already-encoded payload bytes) for the
    /// writer task and return — this never waits on the socket, only on
    /// queue capacity, bounded by [`EndpointConfig::send_timeout`].
    ///
    /// `payload.len()` must equal `header.wire_len`.
    pub async fn send_frame(&self, header: FrameHeader, payload: Vec<u8>) -> Result<(), NetError> {
        if payload.len() != header.wire_len as usize {
            return Err(NetError::MalformedFrame(format!(
                "payload is {} bytes but header declares wire_len {}",
                payload.len(),
                header.wire_len
            )));
        }
        match tokio::time::timeout(self.send_timeout, self.tx.send(Cmd::Frame(header, payload)))
            .await
        {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_closed)) => Err(NetError::NotConnected {
                rank: self.peer,
                reason: "writer task has stopped".to_string(),
            }),
            Err(_elapsed) => Err(NetError::Timeout(self.send_timeout)),
        }
    }

    /// Flush queued frames, close the write half, and stop the reader.
    ///
    /// Idempotent, and safe to call on a link whose peer already went away.
    /// The graceful wait is bounded by [`CLOSE_GRACE`]: a writer blocked on a
    /// peer that stopped reading is aborted rather than hanging shutdown.
    pub async fn close(&self) -> Result<(), NetError> {
        // Best-effort: if the writer already stopped, there is nothing to
        // drain and the send simply fails.
        let _ = tokio::time::timeout(CLOSE_GRACE, self.tx.send(Cmd::Shutdown)).await;
        let mut done = self.writer_done.clone();
        // `changed()` yields `Err` once the writer task drops its sender,
        // which is exactly "the writer has exited".
        let _ = tokio::time::timeout(CLOSE_GRACE, async { while done.changed().await.is_ok() {} })
            .await;
        self.reader.abort();
        self.writer.abort();
        Ok(())
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        self.reader.abort();
        self.writer.abort();
    }
}

/// Dial `addr`, retrying a refused/unreachable connection until `deadline`
/// elapses. Bootstrap dials routinely beat the listener into existence.
pub(crate) async fn connect_with_retry(
    addr: SocketAddr,
    deadline: Duration,
) -> Result<TcpStream, NetError> {
    let start = tokio::time::Instant::now();
    let mut backoff = Duration::from_millis(1);
    loop {
        let attempt = match TcpStream::connect(addr).await {
            Ok(stream) => return Ok(stream),
            Err(e) => e,
        };
        if start.elapsed() >= deadline {
            return Err(NetError::ConnectFailed {
                addr: addr.to_string(),
                msg: attempt.to_string(),
            });
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_millis(50));
    }
}

/// Serialize frames from `rx` onto `write_half` until the channel closes or
/// the socket errors.
async fn writer_loop(mut write_half: OwnedWriteHalf, mut rx: mpsc::Receiver<Cmd>) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Cmd::Frame(header, payload) => {
                if write_frame_half(&mut write_half, &header, &payload)
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Cmd::Shutdown => break,
        }
    }
    let _ = write_half.shutdown().await;
}

/// Write one frame to an owned write half.
async fn write_frame_half(
    write_half: &mut OwnedWriteHalf,
    header: &FrameHeader,
    payload: &[u8],
) -> Result<(), NetError> {
    let encoded = header.encode();
    // Small frames go out as a single write so a latency probe costs one
    // packet; large ones skip the extra copy.
    if payload.len() <= 8 * 1024 {
        let mut buf = Vec::with_capacity(FrameHeader::WIRE_SIZE + payload.len());
        buf.extend_from_slice(&encoded);
        buf.extend_from_slice(payload);
        write_half
            .write_all(&buf)
            .await
            .map_err(|e| NetError::Io(e.to_string()))?;
    } else {
        write_half
            .write_all(&encoded)
            .await
            .map_err(|e| NetError::Io(e.to_string()))?;
        write_half
            .write_all(payload)
            .await
            .map_err(|e| NetError::Io(e.to_string()))?;
    }
    write_half
        .flush()
        .await
        .map_err(|e| NetError::Io(e.to_string()))
}

/// Parse frames off `read_half` forever, routing each payload into `mailbox`.
async fn reader_loop(
    mut read_half: OwnedReadHalf,
    peer: u32,
    mailbox: Arc<Mailbox<Vec<u8>>>,
    max_frame: usize,
) -> Result<(), NetError> {
    loop {
        let (header, payload) = match read_frame(&mut read_half, max_frame).await {
            Ok(frame) => frame,
            Err(NetError::PeerClosed) => return Ok(()),
            Err(other) => return Err(other),
        };
        // Control frames are transport-internal and never reach a user
        // mailbox; a steady-state link should not see any.
        if header.ctx == CTX_CONTROL {
            continue;
        }
        // Trust the wire's own `src` only as far as the link it arrived on:
        // a link to `peer` can only deliver messages from `peer`.
        let key = (peer, header.ctx, header.tag);
        mailbox.push(key, payload)?;
    }
}

/// Read exactly one frame (header + payload, decompressing if flagged).
///
/// Returns [`NetError::PeerClosed`] when the stream ends cleanly at a frame
/// boundary. Enforces `max_frame` against the *declared* lengths before any
/// allocation, so an oversized frame errors rather than truncating.
pub(crate) async fn read_frame<R>(
    reader: &mut R,
    max_frame: usize,
) -> Result<(FrameHeader, Vec<u8>), NetError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut header_bytes = [0u8; FrameHeader::WIRE_SIZE];
    match reader.read_exact(&mut header_bytes).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(NetError::PeerClosed)
        }
        Err(e) => return Err(NetError::Io(e.to_string())),
    }

    let header = FrameHeader::decode(&header_bytes)?;
    let wire_len = header.wire_len as usize;
    let raw_len = header.raw_len as usize;
    if wire_len > max_frame || raw_len > max_frame {
        return Err(NetError::FrameTooLarge {
            size: wire_len.max(raw_len),
            max: max_frame,
        });
    }

    let mut payload = vec![0u8; wire_len];
    if wire_len > 0 {
        reader
            .read_exact(&mut payload)
            .await
            .map_err(|e| NetError::Io(e.to_string()))?;
    }

    if header.is_compressed() {
        let decoded = oxiarc_lz4::block::decompress_block(&payload, raw_len)
            .map_err(|e| NetError::Compression(e.to_string()))?;
        if decoded.len() != raw_len {
            return Err(NetError::MalformedFrame(format!(
                "decompressed to {} bytes but header declares raw_len {raw_len}",
                decoded.len(),
            )));
        }
        return Ok((header, decoded));
    }

    if wire_len != raw_len {
        return Err(NetError::MalformedFrame(format!(
            "uncompressed frame declares wire_len {wire_len} != raw_len {raw_len}"
        )));
    }
    Ok((header, payload))
}

/// Write one frame to any async writer. Used by the bootstrap exchange,
/// which speaks the same codec on a throwaway socket rather than inventing
/// a second parser.
pub(crate) async fn write_frame<W>(
    writer: &mut W,
    header: &FrameHeader,
    payload: &[u8],
) -> Result<(), NetError>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    writer
        .write_all(&header.encode())
        .await
        .map_err(|e| NetError::Io(e.to_string()))?;
    if !payload.is_empty() {
        writer
            .write_all(payload)
            .await
            .map_err(|e| NetError::Io(e.to_string()))?;
    }
    writer
        .flush()
        .await
        .map_err(|e| NetError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::net::frame::{FLAG_COMPRESSED, TAG_HELLO};
    use tokio::net::TcpListener;

    async fn bound_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let dial = tokio::spawn(async move { TcpStream::connect(addr).await });
        let (accepted, _) = listener.accept().await.expect("accept");
        let dialed = dial.await.expect("join").expect("connect");
        (dialed, accepted)
    }

    #[tokio::test]
    async fn frames_round_trip_through_a_link_into_the_mailbox() {
        let (a, b) = bound_pair().await;
        let cfg = EndpointConfig::default();
        let mailbox_b: Arc<Mailbox<Vec<u8>>> = Arc::new(Mailbox::new());
        let addr_a = a.local_addr().expect("addr");
        let addr_b = b.local_addr().expect("addr");

        let link_a = Link::from_stream(a, 1, addr_b, Arc::new(Mailbox::new()), &cfg);
        let _link_b = Link::from_stream(b, 0, addr_a, Arc::clone(&mailbox_b), &cfg);

        let payload = b"hello link".to_vec();
        let header = FrameHeader::new(0, 1, 5, 9, 0, payload.len() as u32, payload.len() as u32, 0)
            .expect("header");
        link_a
            .send_frame(header, payload.clone())
            .await
            .expect("send");

        let got = mailbox_b
            .pop_timeout((0, 5, 9), Duration::from_secs(5))
            .await
            .expect("delivered");
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn reader_drops_control_frames_instead_of_delivering_them() {
        let (a, b) = bound_pair().await;
        let cfg = EndpointConfig::default();
        let mailbox_b: Arc<Mailbox<Vec<u8>>> = Arc::new(Mailbox::new());
        let addr_a = a.local_addr().expect("addr");
        let addr_b = b.local_addr().expect("addr");

        let link_a = Link::from_stream(a, 1, addr_b, Arc::new(Mailbox::new()), &cfg);
        let _link_b = Link::from_stream(b, 0, addr_a, Arc::clone(&mailbox_b), &cfg);

        let hello = FrameHeader::new(0, 1, CTX_CONTROL, TAG_HELLO, 0, 0, 0, 0).expect("header");
        link_a.send_frame(hello, Vec::new()).await.expect("send");
        let real = FrameHeader::new(0, 1, 0, 1, 1, 3, 3, 0).expect("header");
        link_a
            .send_frame(real, b"abc".to_vec())
            .await
            .expect("send");

        // The control frame must not have been routed anywhere the user can
        // see; the following data frame arrives normally.
        let got = mailbox_b
            .pop_timeout((0, 0, 1), Duration::from_secs(5))
            .await
            .expect("delivered");
        assert_eq!(got, b"abc".to_vec());
        assert_eq!(mailbox_b.try_pop((0, CTX_CONTROL, TAG_HELLO)), Ok(None));
    }

    #[tokio::test]
    async fn send_frame_rejects_a_payload_that_contradicts_the_header() {
        let (a, b) = bound_pair().await;
        let cfg = EndpointConfig::default();
        let addr_b = b.local_addr().expect("addr");
        let link = Link::from_stream(a, 1, addr_b, Arc::new(Mailbox::new()), &cfg);
        let header = FrameHeader::new(0, 1, 0, 0, 0, 10, 10, 0).expect("header");
        let err = link
            .send_frame(header, vec![0u8; 3])
            .await
            .expect_err("length mismatch");
        assert!(matches!(err, NetError::MalformedFrame(_)));
    }

    #[tokio::test]
    async fn read_frame_rejects_a_frame_larger_than_the_policy_limit() {
        let (mut a, mut b) = bound_pair().await;
        let payload = vec![7u8; 4096];
        let header = FrameHeader::new(0, 1, 0, 0, 0, payload.len() as u32, payload.len() as u32, 0)
            .expect("header");
        let writer = tokio::spawn(async move {
            let _ = write_frame(&mut a, &header, &payload).await;
            // Keep the socket alive so the reader sees the size error rather
            // than a clean EOF.
            tokio::time::sleep(Duration::from_millis(200)).await;
        });
        let err = read_frame(&mut b, 1024)
            .await
            .expect_err("over policy limit");
        assert!(matches!(
            err,
            NetError::FrameTooLarge {
                size: 4096,
                max: 1024
            }
        ));
        writer.abort();
    }

    #[tokio::test]
    async fn read_frame_reports_a_clean_eof_as_peer_closed() {
        let (a, mut b) = bound_pair().await;
        drop(a);
        let err = read_frame(&mut b, 1024).await.expect_err("closed");
        assert!(matches!(err, NetError::PeerClosed));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reader_is_finished_flips_once_the_peer_hangs_up() {
        let (a, b) = bound_pair().await;
        let cfg = EndpointConfig::default();
        let addr_a = a.local_addr().expect("addr");
        let addr_b = b.local_addr().expect("addr");

        let link_a = Link::from_stream(a, 1, addr_b, Arc::new(Mailbox::new()), &cfg);
        let link_b = Link::from_stream(b, 0, addr_a, Arc::new(Mailbox::new()), &cfg);
        assert!(!link_b.reader_is_finished(), "a live peer is not finished");

        drop(link_a);
        // The reader task observes EOF and returns; give the runtime a moment
        // to actually retire it.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while !link_b.reader_is_finished() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            link_b.reader_is_finished(),
            "the reader must retire once the peer closes the connection"
        );
    }

    #[tokio::test]
    async fn compressed_frames_round_trip_through_read_frame() {
        let (mut a, mut b) = bound_pair().await;
        let raw = vec![42u8; 200_000];
        let wire = oxiarc_lz4::block::compress_block(&raw).expect("compress");
        assert!(wire.len() < raw.len(), "constant data must compress");
        let header = FrameHeader::new(
            0,
            1,
            0,
            0,
            0,
            wire.len() as u32,
            raw.len() as u32,
            FLAG_COMPRESSED,
        )
        .expect("header");
        let writer = tokio::spawn(async move { write_frame(&mut a, &header, &wire).await });
        let (got_header, got) = read_frame(&mut b, super::super::frame::MAX_FRAME)
            .await
            .expect("read");
        writer.await.expect("join").expect("write");
        assert!(got_header.is_compressed());
        assert_eq!(got, raw);
    }
}
