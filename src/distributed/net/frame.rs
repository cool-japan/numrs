//! Wire frame header for the redesigned point-to-point transport.
//!
//! Every message on the wire is prefixed with a fixed-size [`FrameHeader`]
//! followed by `wire_len` bytes of payload (LZ4-block-compressed when
//! [`FLAG_COMPRESSED`] is set in `flags`, raw otherwise).
//!
//! # Layout (56 bytes, little-endian)
//!
//! | field      | type | offset | width |
//! |------------|------|--------|-------|
//! | `magic`    | u32  | 0      | 4     |
//! | `version`  | u16  | 4      | 2     |
//! | `flags`    | u16  | 6      | 2     |
//! | `src`      | u32  | 8      | 4     |
//! | `dst`      | u32  | 12     | 4     |
//! | `ctx`      | u64  | 16     | 8     |
//! | `tag`      | u64  | 24     | 8     |
//! | `seq`      | u64  | 32     | 8     |
//! | `wire_len` | u32  | 40     | 4     |
//! | `raw_len`  | u32  | 44     | 4     |
//! | `reserved` | u64  | 48     | 8     |
//!
//! Total: [`FrameHeader::WIRE_SIZE`] = 56 bytes. `reserved` carries no
//! meaning yet: encoders must write `0` and decoders must accept and pass
//! through whatever value they find (forward-compatible with a future use,
//! e.g. a checksum, without another wire-format break).
//!
//! `raw_len` is the *uncompressed* payload length; pass it as the
//! `max_output` bound to [`oxiarc_lz4::block::decompress_block`] on the
//! receive side. `wire_len` is the number of payload bytes that actually
//! follow the header on the wire — equal to `raw_len` when
//! `flags & FLAG_COMPRESSED == 0`, and the compressed length otherwise.
//!
//! `MAX_FRAME` here is a fixed *protocol* ceiling: [`FrameHeader::new`] and
//! [`FrameHeader::decode`] only ever check `wire_len`/`raw_len` against this
//! constant. [`super::EndpointConfig::max_frame`] is a separate, smaller,
//! per-endpoint *policy* value — nothing in this file reads it. The
//! endpoint layer is responsible for enforcing its own configured limit
//! (e.g. rejecting a send before it ever reaches `FrameHeader::new`, or
//! double-checking a decoded header's lengths against its own config)
//! rather than assuming `decode` already did that narrower check.

use super::NetError;

/// The magic 4 bytes every frame starts with. Spelled out as a byte string
/// so the little-endian encoding of this constant is literally `b"NRS2"` on
/// the wire (readable in a hex dump).
pub const MAGIC: u32 = u32::from_le_bytes(*b"NRS2");

/// Current wire protocol version.
pub const VERSION: u16 = 1;

/// `flags` bit 0: the payload following this header is LZ4-block-compressed.
pub const FLAG_COMPRESSED: u16 = 0b0000_0001;

/// Frames whose `wire_len` or `raw_len` exceeds this are rejected outright:
/// 256 MiB.
pub const MAX_FRAME: usize = 256 * 1024 * 1024;

/// Payloads at or above this size are candidates for LZ4 compression
/// (smaller payloads are sent raw — compression overhead isn't worth it):
/// 64 KiB.
pub const COMPRESS_THRESHOLD: usize = 64 * 1024;

/// Reserved context id for transport-internal control frames (HELLO,
/// bootstrap address exchange). User traffic must never use it — a
/// [`super::endpoint::Endpoint`] never routes a frame carrying this `ctx`
/// into a user mailbox.
pub const CTX_CONTROL: u64 = u64::MAX;

/// `tag` (within [`CTX_CONTROL`]) of the HELLO frame a dialer sends
/// immediately after connecting, so the accepting side learns which rank is
/// on the other end. The rank travels in the header's `src` field, so the
/// payload is empty.
pub const TAG_HELLO: u64 = u64::MAX;

/// `tag` (within [`CTX_CONTROL`]) of the frame each rank sends to the master
/// during `Master`-mode bootstrap, carrying its own bound address as UTF-8.
pub const TAG_BOOTSTRAP_PUBLISH: u64 = u64::MAX - 1;

/// `tag` (within [`CTX_CONTROL`]) of the master's reply, carrying the full
/// rank-ordered address table as a comma-separated UTF-8 list.
pub const TAG_BOOTSTRAP_TABLE: u64 = u64::MAX - 2;

/// Fixed-size header prefixed to every frame on the wire.
///
/// All multi-byte fields are little-endian. This is a frozen wire contract
/// (see module docs for the exact byte layout) — every field, its type, and
/// the 56-byte total size are fixed; do not resize or reorder without a
/// protocol version bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameHeader {
    /// Must equal [`MAGIC`] on the wire; [`Self::decode`] rejects anything else.
    pub magic: u32,
    /// Protocol version; [`Self::decode`] rejects anything other than [`VERSION`].
    pub version: u16,
    /// Bit flags — see [`FLAG_COMPRESSED`].
    pub flags: u16,
    /// Sending rank.
    pub src: u32,
    /// Destination rank.
    pub dst: u32,
    /// Logical communicator/context id (distinguishes concurrent
    /// sub-communicators sharing one physical link).
    pub ctx: u64,
    /// User-facing message tag.
    pub tag: u64,
    /// Per-`(src, dst)` monotonically increasing sequence number.
    pub seq: u64,
    /// Number of payload bytes following this header on the wire.
    pub wire_len: u32,
    /// Uncompressed payload length (equals `wire_len` when not compressed).
    pub raw_len: u32,
    /// Reserved for future use. Always `0` today; decoders must not reject
    /// a nonzero value.
    pub reserved: u64,
}

impl FrameHeader {
    /// Encoded size of a `FrameHeader` on the wire, in bytes. Frozen at 56.
    pub const WIRE_SIZE: usize = 56;

    /// Build a header with `magic`/`version` filled in and `reserved`
    /// zeroed, validating that `wire_len`/`raw_len` fit within [`MAX_FRAME`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        src: u32,
        dst: u32,
        ctx: u64,
        tag: u64,
        seq: u64,
        wire_len: u32,
        raw_len: u32,
        flags: u16,
    ) -> Result<Self, NetError> {
        if wire_len as usize > MAX_FRAME || raw_len as usize > MAX_FRAME {
            return Err(NetError::FrameTooLarge {
                size: (wire_len.max(raw_len)) as usize,
                max: MAX_FRAME,
            });
        }
        Ok(Self {
            magic: MAGIC,
            version: VERSION,
            flags,
            src,
            dst,
            ctx,
            tag,
            seq,
            wire_len,
            raw_len,
            reserved: 0,
        })
    }

    /// Whether the payload following this header is LZ4-block-compressed.
    pub fn is_compressed(&self) -> bool {
        self.flags & FLAG_COMPRESSED != 0
    }

    /// Encode this header to its fixed 56-byte little-endian wire form.
    pub fn encode(&self) -> [u8; Self::WIRE_SIZE] {
        let mut buf = [0u8; Self::WIRE_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.src.to_le_bytes());
        buf[12..16].copy_from_slice(&self.dst.to_le_bytes());
        buf[16..24].copy_from_slice(&self.ctx.to_le_bytes());
        buf[24..32].copy_from_slice(&self.tag.to_le_bytes());
        buf[32..40].copy_from_slice(&self.seq.to_le_bytes());
        buf[40..44].copy_from_slice(&self.wire_len.to_le_bytes());
        buf[44..48].copy_from_slice(&self.raw_len.to_le_bytes());
        buf[48..56].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    /// Decode a header from the first [`Self::WIRE_SIZE`] bytes of `bytes`.
    ///
    /// Validates `magic`, `version`, and that `wire_len`/`raw_len` are
    /// within [`MAX_FRAME`]. Trailing bytes beyond the header (the payload)
    /// are ignored here — slice them out yourself using `wire_len`.
    pub fn decode(bytes: &[u8]) -> Result<Self, NetError> {
        if bytes.len() < Self::WIRE_SIZE {
            return Err(NetError::MalformedFrame(format!(
                "header needs {} bytes, got {}",
                Self::WIRE_SIZE,
                bytes.len()
            )));
        }

        let u32_at = |range: std::ops::Range<usize>| -> u32 {
            let mut a = [0u8; 4];
            a.copy_from_slice(&bytes[range]);
            u32::from_le_bytes(a)
        };
        let u16_at = |range: std::ops::Range<usize>| -> u16 {
            let mut a = [0u8; 2];
            a.copy_from_slice(&bytes[range]);
            u16::from_le_bytes(a)
        };
        let u64_at = |range: std::ops::Range<usize>| -> u64 {
            let mut a = [0u8; 8];
            a.copy_from_slice(&bytes[range]);
            u64::from_le_bytes(a)
        };

        let magic = u32_at(0..4);
        if magic != MAGIC {
            return Err(NetError::BadMagic {
                expected: MAGIC,
                actual: magic,
            });
        }

        let version = u16_at(4..6);
        if version != VERSION {
            return Err(NetError::UnsupportedVersion(version));
        }

        let header = Self {
            magic,
            version,
            flags: u16_at(6..8),
            src: u32_at(8..12),
            dst: u32_at(12..16),
            ctx: u64_at(16..24),
            tag: u64_at(24..32),
            seq: u64_at(32..40),
            wire_len: u32_at(40..44),
            raw_len: u32_at(44..48),
            reserved: u64_at(48..56),
        };

        if header.wire_len as usize > MAX_FRAME || header.raw_len as usize > MAX_FRAME {
            return Err(NetError::FrameTooLarge {
                size: (header.wire_len.max(header.raw_len)) as usize,
                max: MAX_FRAME,
            });
        }

        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_size_is_56_bytes() {
        // This is the number downstream lanes should hardcode; pin it
        // against a literal, not just the constant it's derived from.
        assert_eq!(FrameHeader::WIRE_SIZE, 56);
        let h = FrameHeader::new(1, 2, 3, 4, 5, 10, 10, 0).expect("valid header");
        assert_eq!(h.encode().len(), FrameHeader::WIRE_SIZE);
    }

    #[test]
    fn magic_spells_nrs2_on_the_wire() {
        let h = FrameHeader::new(0, 0, 0, 0, 0, 0, 0, 0).expect("valid header");
        assert_eq!(&h.encode()[0..4], b"NRS2");
    }

    #[test]
    fn roundtrip_preserves_every_field() {
        let h = FrameHeader::new(
            7,
            9,
            0xdead_beef,
            0xcafe_babe_0000_0001,
            42,
            128,
            256,
            FLAG_COMPRESSED,
        )
        .expect("valid header");
        let bytes = h.encode();
        let decoded = FrameHeader::decode(&bytes).expect("decodes");
        assert_eq!(h, decoded);
        assert!(decoded.is_compressed());
    }

    #[test]
    fn uncompressed_flag_round_trips_false() {
        let h = FrameHeader::new(0, 1, 0, 0, 0, 64, 64, 0).expect("valid header");
        let decoded = FrameHeader::decode(&h.encode()).expect("decodes");
        assert!(!decoded.is_compressed());
    }

    #[test]
    fn reserved_defaults_to_zero_and_round_trips() {
        let h = FrameHeader::new(0, 1, 0, 0, 0, 0, 0, 0).expect("valid header");
        assert_eq!(h.reserved, 0);
        let decoded = FrameHeader::decode(&h.encode()).expect("decodes");
        assert_eq!(decoded.reserved, 0);
    }

    #[test]
    fn decode_accepts_nonzero_reserved_forward_compat() {
        let mut bytes = FrameHeader::new(0, 1, 0, 0, 0, 0, 0, 0)
            .expect("valid header")
            .encode();
        bytes[48..56].copy_from_slice(&42u64.to_le_bytes());
        let decoded = FrameHeader::decode(&bytes).expect("decodes despite nonzero reserved");
        assert_eq!(decoded.reserved, 42);
    }

    #[test]
    fn rejects_short_buffer() {
        let err = FrameHeader::decode(&[0u8; 10]);
        assert!(matches!(err, Err(NetError::MalformedFrame(_))));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = FrameHeader::new(0, 1, 0, 0, 0, 0, 0, 0)
            .expect("valid header")
            .encode();
        bytes[0] = !bytes[0];
        let err = FrameHeader::decode(&bytes);
        assert!(matches!(err, Err(NetError::BadMagic { .. })));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = FrameHeader::new(0, 1, 0, 0, 0, 0, 0, 0)
            .expect("valid header")
            .encode();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = FrameHeader::decode(&bytes);
        assert!(matches!(err, Err(NetError::UnsupportedVersion(99))));
    }

    #[test]
    fn rejects_oversized_frame_on_construction() {
        let too_big = (MAX_FRAME + 1) as u32;
        let err = FrameHeader::new(0, 1, 0, 0, 0, too_big, 0, 0);
        assert!(matches!(err, Err(NetError::FrameTooLarge { .. })));
    }

    #[test]
    fn rejects_oversized_frame_on_decode() {
        // Construct a structurally valid header, then hand-corrupt raw_len
        // past MAX_FRAME to prove decode() re-validates rather than
        // trusting the wire.
        let mut bytes = FrameHeader::new(0, 1, 0, 0, 0, 0, 0, 0)
            .expect("valid header")
            .encode();
        let too_big = (MAX_FRAME + 1) as u32;
        bytes[44..48].copy_from_slice(&too_big.to_le_bytes());
        let err = FrameHeader::decode(&bytes);
        assert!(matches!(err, Err(NetError::FrameTooLarge { .. })));
    }

    /// The frozen contract's two size constants, pinned against literals.
    ///
    /// `COMPRESS_THRESHOLD < MAX_FRAME` is a `const` assertion rather than a
    /// runtime one: both operands are compile-time constants, so a runtime
    /// `assert!` would be dead weight the compiler already knows the answer to
    /// (and clippy rightly flags it). As a `const` block it fails the *build*
    /// if anyone ever inverts them.
    #[test]
    fn compress_threshold_and_max_frame_are_sane() {
        const _: () = assert!(COMPRESS_THRESHOLD < MAX_FRAME);
        assert_eq!(COMPRESS_THRESHOLD, 64 * 1024);
        assert_eq!(MAX_FRAME, 256 * 1024 * 1024);
    }
}
