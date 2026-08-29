//! SFC64 bit generator (Chris Doty-Humphrey's "Small Fast Chaotic" PRNG)
//!
//! This implements the exact variant NumPy exposes as `numpy.random.SFC64`:
//! a 256-bit state (three `u64` mixing words `a`, `b`, `c`, plus a `u64`
//! counter `w` that increments by one every draw, guaranteeing a minimum
//! cycle length of 2^64 regardless of how the chaotic part behaves), seeded
//! by drawing `a`, `b`, `c` from a [`SeedSequence`] and then discarding the
//! first 12 outputs to mix the seed thoroughly before the stream is used.
//!
//! Every constant and the exact seeding/warm-up behavior have been verified
//! bit-for-bit against real NumPy 2.4.2 output (see this module's tests).
//!
//! ## Algorithm
//!
//! Per NumPy's `sfc64.c` (`sfc64_next`/`sfc64_set_seed`):
//!
//! ```text
//! next():
//!     tmp = a + b + w
//!     a   = b ^ (b >> 11)
//!     b   = c + (c << 3)
//!     c   = rotl(c, 24) + tmp
//!     w  += 1
//!     return tmp
//! ```
//!
//! with `rotl` a 64-bit left rotation. Seeding draws `(a, b, c)` from
//! `SeedSequence::generate_state_u64(3)`, sets `w = 1`, then calls `next()`
//! twelve times and discards the results.

use super::generator::{BitGenerator, SeedableBitGenerator};
use super::seed_sequence::SeedSequence;

/// Number of warm-up draws discarded after seeding, per NumPy's
/// `sfc64_set_seed`.
const WARMUP_ROUNDS: usize = 12;

/// Chris Doty-Humphrey's SFC64 bit generator, matching NumPy's `SFC64`.
///
/// # Compatibility guarantee
///
/// [`SFC64BitGenerator::new`] reproduces `np.random.SFC64(seed)` raw `u64`
/// output bit-for-bit (verified against NumPy 2.4.2 in this module's
/// tests). As with [`super::philox::Philox4x64BitGenerator`],
/// [`BitGenerator::next_f64`] uses NumPy's own raw-to-double conversion, but
/// the higher-level distribution methods on [`super::generator::Generator`]
/// go through this crate's existing `rand`-ecosystem sampling and are
/// therefore reproducible-but-not-NumPy-bit-identical.
pub struct SFC64BitGenerator {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl SFC64BitGenerator {
    /// Create a generator seeded (via [`SeedSequence`]) from a plain
    /// integer, matching `np.random.SFC64(seed)`.
    pub fn new(seed: u64) -> Self {
        let seed_seq = SeedSequence::new(seed);
        Self::from_seed_sequence_impl(&seed_seq)
    }

    /// Create a generator directly from its three mixing words (before the
    /// warm-up mixing rounds are applied), with the counter starting at 1.
    /// This is the constructor [`SeedableBitGenerator::from_seed_sequence`]
    /// and [`SFC64BitGenerator::new`] both build on.
    pub fn from_state_words(a: u64, b: u64, c: u64) -> Self {
        let mut gen = Self {
            a,
            b,
            c,
            counter: 1,
        };
        for _ in 0..WARMUP_ROUNDS {
            gen.next_u64();
        }
        gen
    }

    fn from_seed_sequence_impl(seed_seq: &SeedSequence) -> Self {
        let words = seed_seq.generate_state_u64(3);
        Self::from_state_words(words[0], words[1], words[2])
    }

    /// The generator's current internal state, as `(a, b, c, counter)` --
    /// matching NumPy's exposed `SFC64(seed).state['state']['state']` array.
    pub fn state_words(&self) -> (u64, u64, u64, u64) {
        (self.a, self.b, self.c, self.counter)
    }
}

impl BitGenerator for SFC64BitGenerator {
    fn next_u64(&mut self) -> u64 {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(24).wrapping_add(tmp);
        self.counter = self.counter.wrapping_add(1);
        tmp
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

impl SeedableBitGenerator for SFC64BitGenerator {
    fn from_seed_sequence(seed_seq: &SeedSequence) -> Self {
        Self::from_seed_sequence_impl(seed_seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw_n(gen: &mut SFC64BitGenerator, n: usize) -> Vec<u64> {
        (0..n).map(|_| gen.next_u64()).collect()
    }

    // ---- Reference values pinned against `np.random.SFC64` ------------
    //
    // Computed with numpy 2.4.2:
    //     g = np.random.SFC64(<seed>)
    //     g.random_raw(8)
    //     g.state['state']['state']   # (a, b, c, counter) after seeding

    #[test]
    fn matches_numpy_seed_42() {
        let mut gen = SFC64BitGenerator::new(42);
        // State immediately after seeding (before any draw is consumed by
        // this check -- `state_words` doesn't advance the generator).
        assert_eq!(
            gen.state_words(),
            (
                9143715722600539226,
                631878879238184246,
                4804307118799948943,
                13
            )
        );
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                9775594601838723485,
                6977463094773878866,
                17439770048677797496,
                7768405669198076140,
                11828679036797625575,
                5968245995936705621,
                18446377427755072631,
                9778856443036923222,
            ]
        );
    }

    #[test]
    fn matches_numpy_seed_0() {
        let mut gen = SFC64BitGenerator::new(0);
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                10490465040999277362,
                4331856608414834465,
                7312684695965765022,
                1874867651408945186,
                7329937082660668956,
                11278147118872085440,
                13479525968569490213,
                15446970583328750443,
            ]
        );
        assert_eq!(
            gen.state_words(),
            (
                4285069076241942803,
                1497268133912585534,
                18388051688725505767,
                21
            )
        );
    }

    #[test]
    fn matches_numpy_seed_1() {
        let mut gen = SFC64BitGenerator::new(1);
        assert_eq!(
            draw_n(&mut gen, 8),
            vec![
                18365948275979584072,
                6864396556639111295,
                7917024265190753706,
                14104257147265189493,
                15367196606101136102,
                15285311301336988976,
                16167655303506643925,
                5283050751462028101,
            ]
        );
        assert_eq!(
            gen.state_words(),
            (
                9714687587817825645,
                12413695779364563919,
                8441037069557600362,
                21
            )
        );
    }

    #[test]
    fn seeded_generator_is_deterministic() {
        let mut a = SFC64BitGenerator::new(2024);
        let mut b = SFC64BitGenerator::new(2024);
        assert_eq!(draw_n(&mut a, 16), draw_n(&mut b, 16));
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = SFC64BitGenerator::new(1);
        let mut b = SFC64BitGenerator::new(2);
        assert_ne!(draw_n(&mut a, 4), draw_n(&mut b, 4));
    }

    #[test]
    fn seed_method_reseeds_deterministically() {
        let mut gen = SFC64BitGenerator::new(1);
        gen.next_u64();
        gen.seed(42);
        let mut expected = SFC64BitGenerator::new(42);
        assert_eq!(draw_n(&mut gen, 8), draw_n(&mut expected, 8));
    }

    #[test]
    fn from_seed_sequence_matches_new() {
        let seq = SeedSequence::new(42);
        let mut from_seq = SFC64BitGenerator::from_seed_sequence(&seq);
        let mut from_seed = SFC64BitGenerator::new(42);
        assert_eq!(draw_n(&mut from_seq, 8), draw_n(&mut from_seed, 8));
    }

    #[test]
    fn counter_increments_by_one_each_draw() {
        let mut gen = SFC64BitGenerator::new(7);
        let (_, _, _, w0) = gen.state_words();
        gen.next_u64();
        let (_, _, _, w1) = gen.state_words();
        assert_eq!(w1, w0.wrapping_add(1));
    }
}
