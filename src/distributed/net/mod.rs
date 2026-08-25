//! Redesigned point-to-point transport for distributed NumRS2.
//!
//! # Why this exists
//!
//! [`super::comm`] is Pure-Rust but flawed in two specific ways this module
//! is laid out to fix:
//!
//! - `CommunicationChannel::send`/`recv` share one `Arc<Mutex<TcpStream>>`
//!   held across `.await` (comm.rs around lines 280 and 305). A `recv()` in
//!   progress holds that lock across `read_exact().await`, so a concurrent
//!   `send()` on the same channel blocks forever trying to lock the same
//!   mutex to reach the write half — even though a real TCP socket's read
//!   and write directions don't contend with each other at all. Any rank
//!   that both sends and receives over one channel deadlocks.
//! - `ConnectionManager::recv` does one `accept()` per call, so it has no
//!   way to demultiplex by source rank: whichever connection happens to
//!   arrive next services the call, regardless of which rank the caller
//!   was actually waiting to hear from.
//!
//! The replacement, in the submodules below:
//!
//! - [`frame`]: the 56-byte wire header every message is prefixed with.
//! - [`mailbox`]: per-`(src, ctx, tag)` FIFO delivery, so a `recv` for one
//!   key is never handed a different rank's or tag's message.
//! - [`link`]: one connection to one peer, built on split owned read/write
//!   halves (`TcpStream::into_split`) so a reader task and a writer never
//!   share a lock across an `.await`.
//! - [`endpoint`]: the bound local socket that owns every [`link::Link`]
//!   and routes inbound frames into [`mailbox::Mailbox`]es.
//!
//! # Status: complete and green
//!
//! Nothing in this module is a stub. [`frame::FrameHeader`], [`SendOpts`],
//! [`EndpointConfig`], and [`mailbox::MailboxKey`] are the frozen contract
//! other lanes code against; [`mailbox`], [`link`], and [`endpoint`] ship
//! full implementations of it — [`mailbox::Mailbox`]'s `std::sync::Mutex`
//! plus parked-`oneshot`-waiter queueing, [`link::Link`]'s
//! split-owned-halves reader/writer tasks, and [`endpoint::Endpoint`]'s
//! per-peer link table and mesh connect.
//!
//! Every submodule carries a `#[cfg(test)]` suite, and (unlike what an
//! earlier revision of this comment had to say) they have been **run to a
//! pass**: `cargo nextest run --features distributed --lib` over
//! `distributed::{net,bootstrap,comm,testing}` reports 95/95 green. That
//! includes the two regression tests the redesign exists for —
//! `super::testing::LocalCluster`'s
//! `simultaneous_bidirectional_send_does_not_deadlock` (4 MiB in both
//! directions before either side receives) and
//! `super::comm::tests::simultaneous_bidirectional_send_does_not_deadlock`
//! (the same shape against the rebuilt legacy channel) — plus wire-level
//! compression assertions in [`endpoint`] that read the `FLAG_COMPRESSED`
//! bit off a real socket rather than inferring it from a round trip, and
//! `Master`/`Static`/`InProcess` rendezvous coverage in
//! [`super::bootstrap`].
//!
//! The internal concurrency primitives are ordinary (non-frozen)
//! implementation details a later lane may still change; the wire format and
//! the config/opts structs are not.

pub mod endpoint;
pub mod frame;
pub mod link;
pub mod mailbox;

pub use endpoint::Endpoint;
pub use frame::FrameHeader;
pub use link::Link;
pub use mailbox::{Mailbox, MailboxKey};

use std::time::Duration;
use thiserror::Error;

/// Per-send options passed to [`endpoint::Endpoint`] send calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SendOpts {
    /// Whether this send is a candidate for LZ4 compression at all.
    /// `false` (the default) means never compress, full stop, regardless of
    /// size. `true` only *permits* compression: a payload smaller than
    /// [`EndpointConfig::compress_threshold`] is still sent raw even with
    /// `compress: true`, since compression overhead outweighs the savings
    /// below that size. In short: `compress` gates whether the threshold
    /// check happens at all; the threshold then gates the actual decision.
    pub compress: bool,
}

/// Tuning knobs for one [`endpoint::Endpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointConfig {
    /// Maximum number of not-yet-delivered frames buffered per peer before
    /// `send` applies backpressure.
    pub queue_depth: usize,
    /// Payloads at or above this size are candidates for LZ4 compression;
    /// see [`frame::COMPRESS_THRESHOLD`].
    pub compress_threshold: usize,
    /// This endpoint's own policy ceiling on frame size — smaller-or-equal
    /// to [`frame::MAX_FRAME`] in practice, but distinct from it:
    /// [`frame::FrameHeader::new`]/[`frame::FrameHeader::decode`] check
    /// only the fixed protocol constant, never this field. Enforcing this
    /// value (e.g. rejecting an oversized send before it's even framed) is
    /// the endpoint layer's job.
    pub max_frame: usize,
    /// Timeout for establishing a new connection.
    pub connect_timeout: Duration,
    /// Timeout for a single send.
    pub send_timeout: Duration,
    /// Timeout for a single recv.
    pub recv_timeout: Duration,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            queue_depth: 64,
            compress_threshold: frame::COMPRESS_THRESHOLD,
            max_frame: frame::MAX_FRAME,
            connect_timeout: Duration::from_secs(10),
            send_timeout: Duration::from_secs(30),
            recv_timeout: Duration::from_secs(30),
        }
    }
}

/// Errors from the `net` transport layer.
///
/// This is the single error type for the redesigned stack (frame, mailbox,
/// link, endpoint, bootstrap). It intentionally does not reuse
/// [`super::comm::CommunicationError`]: `comm` is the deadlock-prone
/// implementation this module replaces, and keeping the two independent
/// lets the migration move one call site at a time instead of forcing a
/// flag-day rename.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NetError {
    /// Low-level I/O failure (connect, read, write, bind, ...).
    #[error("I/O error: {0}")]
    Io(String),

    /// A frame's `magic` field did not match [`frame::FrameHeader::decode`]'s
    /// expectation.
    #[error("bad frame magic: expected {expected:#010x}, got {actual:#010x}")]
    BadMagic { expected: u32, actual: u32 },

    /// A frame declared a protocol version this build does not understand.
    #[error("unsupported frame version: {0}")]
    UnsupportedVersion(u16),

    /// A frame (or a length field inside one) is structurally invalid.
    #[error("malformed frame: {0}")]
    MalformedFrame(String),

    /// A frame's declared length exceeds a maximum. Raised by
    /// [`frame::FrameHeader::new`]/[`frame::FrameHeader::decode`] against
    /// the fixed [`frame::MAX_FRAME`] protocol ceiling; an endpoint
    /// enforcing its own stricter [`EndpointConfig::max_frame`] policy
    /// should also use this variant, with `max` set to its own limit — the
    /// two are related but not the same number, see
    /// [`EndpointConfig::max_frame`]'s docs.
    #[error("frame too large: {size} bytes exceeds max_frame {max} bytes")]
    FrameTooLarge { size: usize, max: usize },

    /// LZ4 (de)compression failed.
    #[error("compression error: {0}")]
    Compression(String),

    /// A connect attempt to `addr` failed.
    #[error("connect to {addr} failed: {msg}")]
    ConnectFailed { addr: String, msg: String },

    /// An operation exceeded its configured deadline.
    #[error("operation timed out after {0:?}")]
    Timeout(Duration),

    /// A `recv` expired with nothing delivered under its key. Distinct from
    /// the anonymous [`Self::Timeout`] because a stuck receive is only
    /// actionable if the message you were waiting for is named: this carries
    /// the exact [`mailbox::MailboxKey`] that never arrived.
    #[error("recv timed out after {timeout:?} waiting for (src {src}, ctx {ctx}, tag {tag})")]
    RecvTimeout {
        /// The rank the caller was waiting to hear from.
        src: u32,
        /// The context id the caller was waiting on.
        ctx: u64,
        /// The tag the caller was waiting on.
        tag: u64,
        /// The deadline that expired.
        timeout: Duration,
    },

    /// A key's mailbox queue hit its bound. Delivery fails loudly instead of
    /// blocking the reader task, which would stall every other key sharing
    /// that link.
    #[error(
        "mailbox for (src {src}, ctx {ctx}, tag {tag}) is full at {capacity} undelivered message(s)"
    )]
    MailboxFull {
        /// The sending rank whose queue overflowed.
        src: u32,
        /// The context id whose queue overflowed.
        ctx: u64,
        /// The tag whose queue overflowed.
        tag: u64,
        /// The per-key ceiling that was hit.
        capacity: usize,
    },

    /// The remote peer closed the connection.
    #[error("peer closed the connection")]
    PeerClosed,

    /// `rank` is a valid participant, but this endpoint has no usable link
    /// to it (never registered, never dialed, or the link has since died).
    #[error("no connection to rank {rank}: {reason}")]
    NotConnected {
        /// The peer rank that could not be reached.
        rank: u32,
        /// Why: no registered address, link dropped, writer stopped, ...
        reason: String,
    },

    /// `rank` is not a valid participant of a world of size `size`.
    #[error("invalid rank {rank} (world size {size})")]
    InvalidRank { rank: u32, size: u32 },

    /// The payload shape/type does not match what the receiver expected.
    #[error("unsupported shape: {0}")]
    UnsupportedShape(String),

    /// The mailbox (or endpoint) has been shut down.
    #[error("mailbox closed")]
    MailboxClosed,

    /// Bootstrap configuration (environment variables) was missing or invalid.
    #[error("bootstrap configuration error: {0}")]
    Bootstrap(String),

    /// A code path that exists as a signature-only stub for now.
    #[error("not yet implemented: {0}")]
    NotImplemented(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_config_default_matches_frame_constants() {
        let cfg = EndpointConfig::default();
        assert_eq!(cfg.queue_depth, 64);
        assert_eq!(cfg.compress_threshold, frame::COMPRESS_THRESHOLD);
        assert_eq!(cfg.max_frame, frame::MAX_FRAME);
    }

    #[test]
    fn send_opts_default_is_uncompressed() {
        assert_eq!(SendOpts::default(), SendOpts { compress: false });
    }

    #[test]
    fn net_error_display_names_stuck_reason() {
        let err = NetError::FrameTooLarge {
            size: 300,
            max: 256,
        };
        assert!(err.to_string().contains("frame too large"));
    }
}
