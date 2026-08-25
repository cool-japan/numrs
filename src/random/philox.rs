//! Philox4x64-10 counter-based bit generator
//!
//! ## A note on the name
//!
//! The task that produced this module asked for "Philox4x32-10 ... pin raw
//! output blocks against `np.random.Philox(key=...)` reference values
//! (exact u64s)". Those two requirements are mutually exclusive: NumPy's
//! `Philox` bit generator is **not** the 4x32-word member of the Random123
//! family. Its own docstring says so explicitly:
//!
//! > Container for the Philox (4x64) pseudo-random number generator.
//!
//! (verified directly against `numpy` 2.4.2 via `help(np.random.Philox)`).
//! `Philox4x32-10` uses four 32-bit words of counter/output and 32x32→64-bit
//! multiplications; NumPy's `Philox` uses four **64-bit** words and
//! 64x64→128-bit multiplications ("Philox4x64-10"). The two variants use
//! different round constants and produce entirely different output streams
//! from the same key/counter, so there is no way to satisfy "implement
//! 4x32" and "pin against `np.random.Philox`" simultaneously.
//!
//! This module implements **Philox4x64-10**, i.e. what NumPy actually calls
//! `Philox`, so that the explicit, testable requirement -- exact `u64`
//! parity with `np.random.Philox` -- is met. Every constant and the exact
//! seeding/counter/buffering behavior below have been verified bit-for-bit
//! against real NumPy 2.4.2 output (see this module's tests).
//!
//! ## Algorithm
//!
//! Per the Random123 paper ("Parallel Random Numbers: As Easy as 1, 2, 3",
//! Salmon, Moraes, Dror & Shaw, 2011) and NumPy's `philox.c`/`philox.pyx`:
//!
//! - State is a 256-bit counter (four `u64` words) and a 128-bit key (two
//!   `u64` words).
//! - Each call to [`next_u64`](Philox4x64BitGenerator::next_u64) that
//!   exhausts the internal 4-word output buffer first **increments** the
//!   256-bit counter (as a little-endian multi-word integer, carrying
//!   between words), then runs 10 rounds of the Philox mixing function on
//!   the new counter value with the key (rebumped by the two "Weyl"
//!   constants after each round except the last) to produce the next
//!   4-word output block.
//! - When seeded from a plain integer, the 128-bit key is derived from
//!   [`SeedSequence::generate_state_u64(2)`](super::seed_sequence::SeedSequence::generate_state_u64);
//!   the counter starts at zero. This is exactly NumPy's `Philox(seed=...)`
//!   behavior ("The input seed is processed by SeedSequence to generate the
//!   key. The counter is set to 0.").

use super::generator::{BitGenerator, SeedableBitGenerator};
use super::seed_sequence::SeedSequence;

/// Philox4x64 multiplier for counter word 0.
const PHILOX_M0: u64 = 0xD2E7_470E_E14C_6C93;
/// Philox4x64 multiplier for counter word 2.
const PHILOX_M1: u64 = 0xCA5A_8263_9512_1157;
/// Golden-ratio-derived Weyl constant bumping key word 0 between rounds.
const PHILOX_W0: u64 = 0x9E37_79B9_7F4A_7C15;
/// `sqrt(3)`-derived Weyl constant bumping key word 1 between rounds.
const PHILOX_W1: u64 = 0xBB67_AE85_84CA_A73B;

/// 64x64→128-bit multiply, returned as (low 64 bits, high 64 bits).
#[inline]
fn mulhilo64(a: u64, b: u64) -> (u64, u64) {
    let product = (a as u128) * (b as u128);
    (product as u64, (product >> 64) as u64)
}

/// One Philox4x64 round.
#[inline]
fn philox_round(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let (lo0, hi0) = mulhilo64(PHILOX_M0, ctr[0]);
    let (lo1, hi1) = mulhilo64(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

#[inline]
fn philox_bump_key(key: [u64; 2]) -> [u64; 2] {
    [
        key[0].wrapping_add(PHILOX_W0),
        key[1].wrapping_add(PHILOX_W1),
    ]
}

/// Run all 10 Philox4x64 rounds, producing one 4-word output block.
fn philox4x64_10(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let mut c = ctr;
    let mut k = key;
    for round in 0..10 {
        c = philox_round(c, k);
        if round < 9 {
            k = philox_bump_key(k);
        }
    }
    c
}

/// Increment a 256-bit little-endian counter (word 0 least significant),
/// carrying into the next word on overflow.
#[inline]
fn counter_increment(ctr: &mut [u64; 4]) {
    for word in ctr.iter_mut() {
        *word = word.wrapping_add(1);
        if *word != 0 {
            return;
        }
    }
}

/// Philox4x64-10: a counter-based bit generator matching NumPy's `Philox`.
///
/// Unlike PCG64/SFC64-style generators that mutate a small piece of state
/// on every draw, Philox computes each output block as a pure function of a
/// (key, counter) pair, which is what gives it easy, exact "jump ahead"
/// support (advancing the counter) and makes many independent streams
/// trivial to construct from distinct keys or counters. NumRS2 only needs
/// the sequential-draw behavior for the `Generator`/`BitGenerator`
/// abstraction, exposed the same way as every other bit generator in this
/// crate.
///
/// # Compatibility guarantee
///
/// [`Philox4x64BitGenerator::new`] and [`Philox4x64BitGenerator::from_key`]
/// reproduce `np.random.Philox(seed=...)` / `np.random.Philox(key=...)`
/// raw `u64` output bit-for-bit (verified against NumPy 2.4.2 in this
/// module's tests, across multiple seeds/keys and multiple internal buffer
/// refills). [`BitGenerator::next_f64`] uses the same
/// `(raw >> 11) as f64 * 2^-53` conversion NumPy itself uses to turn a raw
/// 64-bit draw into a double in `[0, 1)`; the higher-level distribution
/// methods on [`super::generator::Generator`] (`normal`, `uniform`, ...) are
/// implemented in terms of the `rand`-ecosystem distributions used
/// throughout this crate, and are therefore reproducible for a fixed key
/// but not NumPy-bit-identical.
pub struct Philox4x64BitGenerator {
    key: [u64; 2],
    ctr: [u64; 4],
    buffer: [u64; 4],
    /// Position of the next unconsumed word in `buffer`; `4` means the
    /// buffer is exhausted and the next `next_u64()` call must refill it.
    buffer_pos: usize,
}

impl Philox4x64BitGenerator {
    /// Create a generator seeded (via [`SeedSequence`]) from a plain
    /// integer, matching `np.random.Philox(seed=seed)`.
    pub fn new(seed: u64) -> Self {
        let seed_seq = SeedSequence::new(seed);
        Self::from_seed_sequence_impl(&seed_seq)
    }

    /// Create a generator with an explicit 128-bit key (as two `u64`
    /// words, least-significant first) and the counter at zero, matching
    /// `np.random.Philox(key=[key0, key1])`.
    pub fn from_key(key0: u64, key1: u64) -> Self {
        Self::from_key_and_counter([key0, key1], [0, 0, 0, 0])
    }

    /// Create a generator with an explicit key and initial counter value,
    /// matching `np.random.Philox(key=[...], counter=[...])`.
    pub fn from_key_and_counter(key: [u64; 2], counter: [u64; 4]) -> Self {
        Self {
            key,
            ctr: counter,
            buffer: [0, 0, 0, 0],
            // Force a refill on the first `next_u64()` call, exactly like a
            // freshly constructed `np.random.Philox`.
            buffer_pos: 4,
        }
    }

    fn from_seed_sequence_impl(seed_seq: &SeedSequence) -> Self {
        let words = seed_seq.generate_state_u64(2);
        Self::from_key(words[0], words[1])
    }

    /// The generator's 128-bit key, as two `u64` words (least-significant
    /// first).
    pub fn key(&self) -> [u64; 2] {
        self.key
    }

    /// The generator's current 256-bit counter, as four `u64` words
    /// (least-significant first).
    pub fn counter(&self) -> [u64; 4] {
        self.ctr
    }
}

impl BitGenerator for Philox4x64BitGenerator {
    fn next_u64(&mut self) -> u64 {
        if self.buffer_pos == 4 {
            counter_increment(&mut self.ctr);
            self.buffer = philox4x64_10(self.ctr, self.key);
            self.buffer_pos = 0;
        }
        let value = self.buffer[self.buffer_pos];
        self.buffer_pos += 1;
        value
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn seed(&mut self, seed: u64) {
        *self = Self::new(seed);
    }
}

impl SeedableBitGenerator for Philox4x64BitGenerator {
    fn from_seed_sequence(seed_seq: &SeedSequence) -> Self {
        Self::from_seed_sequence_impl(seed_seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw_n(gen: &mut Philox4x64BitGenerator, n: usize) -> Vec<u64> {
        (0..n).map(|_| gen.next_u64()).collect()
    }

    // ---- Reference values pinned against `np.random.Philox` -----------
    //
    // Computed with numpy 2.4.2:
    //     g = np.random.Philox(seed=<seed>)  # or key=[...]
    //     g.random_raw(<n>)

    #[test]
    fn matches_numpy_key_12345_0_two_blocks() {
        let mut gen = Philox4x64BitGenerator::from_key(12345, 0);
        assert_eq!(gen.key(), [12345, 0]);
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                11923609910150341984,
                14282716219641783572,
                14507188490975060125,
                2944039161201405073,
                2968871015012291328,
                15096419966541852992,
                16923256687495202939,
                1160419304018695661,
            ]
        );
        // After 8 draws (two full 4-word blocks), the counter has been
        // pre-incremented twice: 0 -> 1 (for the first block) -> 2 (for the
        // second), matching numpy's exposed `state['counter']`.
        assert_eq!(gen.counter(), [2, 0, 0, 0]);
    }

    #[test]
    fn matches_numpy_key_12345_0_explicit_counter_5() {
        // np.random.Philox(key=[12345, 0], counter=[5, 0, 0, 0]).random_raw(4)
        // Also verifies counter word order (word 0 = least significant):
        // this matches `np.random.Philox(key=[12345, 0], counter=5)` too.
        let mut gen = Philox4x64BitGenerator::from_key_and_counter([12345, 0], [5, 0, 0, 0]);
        assert_eq!(
            draw_n(&mut gen, 4),
            vec![
                14204178594271561098,
                4077206687122642885,
                13868212636456409208,
                8171055895019847646,
            ]
        );
    }

    #[test]
    fn matches_numpy_key_0_12345() {
        let mut gen = Philox4x64BitGenerator::from_key(0, 12345);
        assert_eq!(
            draw_n(&mut gen, 4),
            vec![
                16603019634450795966,
                2013504916108008274,
                8977841106562892152,
                3879301421665326984,
            ]
        );
    }

    #[test]
    fn matches_numpy_key_7_99999999999_three_blocks() {
        let mut gen = Philox4x64BitGenerator::from_key(7, 99999999999);
        assert_eq!(
            draw_n(&mut gen, 12),
            vec![
                8254373014095508315,
                2302241542758195920,
                2028493942077039283,
                11752752479999847065,
                16064001591349734671,
                2526155484687315525,
                8721105590713415716,
                16619633126342208717,
                3586411780284126987,
                13878104919006832679,
                17043517329545207428,
                13472755383390816889,
            ]
        );
    }

    #[test]
    fn matches_numpy_seed_42() {
        // key derivation via SeedSequence(42).generate_state(2, uint64)
        let mut gen = Philox4x64BitGenerator::new(42);
        assert_eq!(gen.key(), [11465652750463011511, 15382171918060459190]);
        assert_eq!(
            draw_n(&mut gen, 4),
            vec![
                1587852024645073290,
                2611271723512893552,
                4982337093617253890,
                16123152800351476682,
            ]
        );
    }

    #[test]
    fn matches_numpy_seed_0() {
        let mut gen = Philox4x64BitGenerator::new(0);
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                259491006799949737,
                4754966410622352325,
                8698845897610382596,
                1686395276220330909,
                18061843536446043542,
                4723914225006068263,
                17258640445484096837,
                3505852312317462091,
            ]
        );
    }

    #[test]
    fn matches_numpy_seed_1() {
        let mut gen = Philox4x64BitGenerator::new(1);
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                1232279569898196538,
                1457532264001425278,
                106569017797417483,
                14878344917644725055,
                4521404008232170583,
                5545736711149519115,
                6632677743886600724,
                8560958798690075040,
            ]
        );
    }

    #[test]
    fn matches_numpy_seed_u64_max() {
        let mut gen = Philox4x64BitGenerator::new(u64::MAX);
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                10924362251224890726,
                17368330293776168947,
                17111559942051582280,
                11921582585585661498,
                7322015467787760668,
                11741153101999238196,
                4294947573519283067,
                8333663531596248063,
            ]
        );
    }

    #[test]
    fn seeded_generator_is_deterministic() {
        let mut a = Philox4x64BitGenerator::new(2024);
        let mut b = Philox4x64BitGenerator::new(2024);
        assert_eq!(draw_n(&mut a, 16), draw_n(&mut b, 16));
    }

    #[test]
    fn different_keys_produce_different_streams() {
        let mut a = Philox4x64BitGenerator::from_key(1, 0);
        let mut b = Philox4x64BitGenerator::from_key(2, 0);
        assert_ne!(draw_n(&mut a, 4), draw_n(&mut b, 4));
    }

    #[test]
    fn seed_method_reseeds_deterministically() {
        let mut gen = Philox4x64BitGenerator::from_key(1, 1);
        gen.next_u64();
        gen.seed(42);
        let mut expected = Philox4x64BitGenerator::new(42);
        assert_eq!(draw_n(&mut gen, 4), draw_n(&mut expected, 4));
    }

    #[test]
    fn counter_increment_carries_across_words() {
        let mut ctr = [u64::MAX, 0, 0, 0];
        counter_increment(&mut ctr);
        assert_eq!(ctr, [0, 1, 0, 0]);

        let mut ctr_full_carry = [u64::MAX, u64::MAX, u64::MAX, 5];
        counter_increment(&mut ctr_full_carry);
        assert_eq!(ctr_full_carry, [0, 0, 0, 6]);
    }

    #[test]
    fn from_seed_sequence_matches_seeded_key() {
        let seq = SeedSequence::new(42);
        let mut from_seq = Philox4x64BitGenerator::from_seed_sequence(&seq);
        let mut from_seed = Philox4x64BitGenerator::new(42);
        assert_eq!(from_seq.key(), from_seed.key());
        assert_eq!(draw_n(&mut from_seq, 4), draw_n(&mut from_seed, 4));
    }
}
