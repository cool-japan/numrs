//! `SeedSequence`: NumPy-compatible entropy pooling and spawning
//!
//! This module implements NumPy's `SeedSequence` algorithm exactly, as
//! documented in `numpy.random.bit_generator.SeedSequence` and verified
//! byte-for-byte against `numpy` 2.4.2 (see the pinned tests at the bottom
//! of this file).
//!
//! `SeedSequence` solves the problem of turning a small, human-supplied seed
//! (an integer, or a short sequence of integers) into a well-mixed pool of
//! entropy that can (a) seed a [`super::generator::BitGenerator`]'s internal
//! state with no obvious structure inherited from the input, and (b) be
//! split into many statistically independent child sequences via
//! [`SeedSequence::spawn`], which is the standard way to seed parallel
//! streams (one child per worker/thread) without risking overlapping
//! sequences.
//!
//! ## Algorithm
//!
//! The algorithm (originally described by Melissa O'Neill as an alternative
//! entropy-mixing scheme for `std::seed_seq`, and adopted by NumPy) is:
//!
//! 1. The input entropy (and, for spawned children, the `spawn_key` tuple)
//!    is coerced into an array of `u32` words.
//! 2. Those words are folded into a fixed-size pool (4 words / 128 bits by
//!    default) with a multiplicative hash-mixing step (`hashmix`) that
//!    guarantees every output pool word depends on every input word,
//!    followed by a second pass (`mix`) that mixes pool words with each
//!    other so that later input bits can influence earlier pool words too.
//! 3. [`SeedSequence::generate_state_u32`] / `_u64` deterministically expand
//!    the pool into as many further-hashed output words as a bit generator
//!    needs for its internal state, by cycling through the pool and
//!    applying a second, independently-parameterized `hashmix` pass.
//!
//! Constants (`INIT_A`, `MULT_A`, `INIT_B`, `MULT_B`, `MIX_MULT_L`,
//! `MIX_MULT_R`) and the pool size (4 words) are taken verbatim from
//! NumPy's implementation so that output is bit-for-bit identical.

use scirs2_core::random::prelude::*;

/// Number of `u32` words in the entropy pool. NumPy calls this
/// `DEFAULT_POOL_SIZE`; NumRS2 does not expose a way to change it (NumPy's
/// own docs say "there is very little to be gained by selecting another
/// value").
const POOL_SIZE: usize = 4;

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
/// `size_of::<u32>() * 8 / 2`
const XSHIFT: u32 = 16;

/// Convert a non-negative integer into little-endian base-2^32 digits, the
/// same representation NumPy's `_int_to_uint32_array` produces for a Python
/// int. `0` maps to `[0]` (a single zero word), matching NumPy exactly.
fn int_to_u32_words(mut n: u64) -> Vec<u32> {
    if n == 0 {
        return vec![0];
    }
    let mut words = Vec::with_capacity(2);
    while n > 0 {
        words.push((n & 0xFFFF_FFFF) as u32);
        n >>= 32;
    }
    words
}

/// The `hashmix` step shared by entropy-pool mixing (parameterized with
/// `INIT_A`/`MULT_A`) and state generation (parameterized with
/// `INIT_B`/`MULT_B`). Carries a running multiplier (`hash_const`) that
/// advances on every call, so calling `mix` twice with the same `value`
/// yields two different results.
struct HashMix {
    hash_const: u32,
    mult: u32,
}

impl HashMix {
    fn new(init: u32, mult: u32) -> Self {
        Self {
            hash_const: init,
            mult,
        }
    }

    fn mix(&mut self, value: u32) -> u32 {
        let mut v = value ^ self.hash_const;
        self.hash_const = self.hash_const.wrapping_mul(self.mult);
        v = v.wrapping_mul(self.hash_const);
        v ^= v >> XSHIFT;
        v
    }
}

/// The pool-to-pool mixing step ("mix" in NumPy's source): combines two
/// already hash-mixed pool words so that changes to a later pool word can
/// still influence an earlier one.
fn mix_words(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// NumPy-compatible entropy pool for seeding bit generators.
///
/// `SeedSequence` mixes a small seed into a well-distributed pool of
/// entropy, from which arbitrarily many further pseudo-random `u32`/`u64`
/// "state words" can be drawn via [`generate_state_u32`](Self::generate_state_u32)(Self::generate_state_u32)
/// / [`generate_state_u64`](Self::generate_state_u64)(Self::generate_state_u64) — exactly what a bit
/// generator needs to initialize its internal state without any visible
/// structure inherited from a small human-chosen seed (e.g. `0`, `1`, `42`).
///
/// Independent, non-overlapping streams for parallel work are obtained via
/// [`spawn`](Self::spawn), which deterministically derives `n` child
/// `SeedSequence`s that are statistically independent from each other and
/// from the parent (each child mixes in a distinct `spawn_key`).
///
/// # Compatibility guarantee
///
/// For a given `u64` (or `&[u64]`) entropy input, [`SeedSequence::new`] /
/// [`SeedSequence::from_u64_sequence`] and [`generate_state_u32`](Self::generate_state_u32)/
/// [`generate_state_u64`](Self::generate_state_u64)/[`spawn`] reproduce NumPy's
/// `numpy.random.SeedSequence(entropy)` bit-for-bit (verified against NumPy
/// 2.4.2 in this module's tests, including the full `u64` range, sequence
/// entropy, and multi-level `spawn`/`spawn.spawn` nesting).
///
/// [`spawn`]: Self::spawn
#[derive(Debug, Clone)]
pub struct SeedSequence {
    /// Coerced input entropy, as `u32` words (NumPy's `self.entropy`, after
    /// `_coerce_to_uint32_array`). Retained (rather than just folded into
    /// `pool`) because [`spawn`](Self::spawn) needs to re-mix it together
    /// with each child's `spawn_key`.
    run_entropy: Vec<u32>,
    /// This sequence's position in the tree of sequences produced by
    /// repeated `spawn` calls; `[]` for a root sequence.
    spawn_key: Vec<u32>,
    /// The mixed entropy pool (`POOL_SIZE` words).
    pool: Vec<u32>,
    /// Number of children already produced by `spawn`, so that repeated
    /// calls keep handing out fresh, non-overlapping spawn keys.
    n_children_spawned: u32,
}

impl SeedSequence {
    /// Create a root `SeedSequence` from a single non-negative integer seed.
    ///
    /// Equivalent to `numpy.random.SeedSequence(entropy)` for a Python
    /// `int` entropy value that fits in a `u64`.
    pub fn new(entropy: u64) -> Self {
        Self::from_entropy_words(int_to_u32_words(entropy))
    }

    /// Create a root `SeedSequence` from a sequence of non-negative integer
    /// entropy values.
    ///
    /// Equivalent to `numpy.random.SeedSequence(entropy)` for a Python
    /// array-like entropy value; each element is independently expanded
    /// into one or two `u32` words (matching NumPy's per-element
    /// `_int_to_uint32_array` coercion) and the results concatenated in
    /// order.
    pub fn from_u64_sequence(entropy: &[u64]) -> Self {
        let words = entropy.iter().flat_map(|&e| int_to_u32_words(e)).collect();
        Self::from_entropy_words(words)
    }

    /// Create a root `SeedSequence` from already-coerced `u32` entropy
    /// words (NumPy's internal representation after `_coerce_to_uint32_array`).
    pub fn from_entropy_words(words: Vec<u32>) -> Self {
        Self::from_parts(words, Vec::new())
    }

    /// Create a root `SeedSequence` seeded with fresh, unpredictable entropy
    /// pulled from this process's random source.
    ///
    /// Equivalent to `numpy.random.SeedSequence()` (entropy left as
    /// `None`), except NumRS2 draws its OS-backed entropy through
    /// `scirs2_core::random::thread_rng()` rather than `os.urandom`
    /// directly. Being non-deterministic by design, this constructor has no
    /// pinned reference values; use [`SeedSequence::new`] for reproducible
    /// seeding.
    pub fn from_os_entropy() -> Self {
        let mut rng = thread_rng();
        let words: Vec<u32> = (0..POOL_SIZE).map(|_| rng.random::<u32>()).collect();
        Self::from_entropy_words(words)
    }

    fn from_parts(run_entropy: Vec<u32>, spawn_key: Vec<u32>) -> Self {
        let mut seq = Self {
            run_entropy,
            spawn_key,
            pool: vec![0u32; POOL_SIZE],
            n_children_spawned: 0,
        };
        seq.mix_entropy();
        seq
    }

    /// NumPy's `get_assembled_entropy`: the run entropy, padded with zeros
    /// up to the pool size if a non-empty `spawn_key` needs to be appended
    /// (so a spawn key can never collide with a short run entropy array),
    /// followed by the spawn key words.
    fn assembled_entropy(&self) -> Vec<u32> {
        let mut run_entropy = self.run_entropy.clone();
        if !self.spawn_key.is_empty() && run_entropy.len() < POOL_SIZE {
            run_entropy.resize(POOL_SIZE, 0);
        }
        run_entropy.extend_from_slice(&self.spawn_key);
        run_entropy
    }

    /// NumPy's `mix_entropy`: fold the assembled entropy into `self.pool`.
    fn mix_entropy(&mut self) {
        let entropy = self.assembled_entropy();
        let mut hm = HashMix::new(INIT_A, MULT_A);

        // Seed the pool directly from the entropy (zero-padded if the pool
        // is larger than the entropy).
        for (i, slot) in self.pool.iter_mut().enumerate() {
            let value = entropy.get(i).copied().unwrap_or(0);
            *slot = hm.mix(value);
        }

        // Mix every pool word into every other pool word so that later
        // words can still affect earlier ones.
        let n = self.pool.len();
        for i_src in 0..n {
            for i_dst in 0..n {
                if i_src == i_dst {
                    continue;
                }
                let src_val = self.pool[i_src];
                let mixed_src = hm.mix(src_val);
                self.pool[i_dst] = mix_words(self.pool[i_dst], mixed_src);
            }
        }

        // Fold in any entropy words beyond the pool size. NumPy calls
        // `hash_mix_(entropy[i_src])` *inside* the `i_dst` loop, so despite
        // the input being the same `extra` value each time, the advancing
        // `hash_const` makes every one of the `n` calls produce a different
        // mixed value -- recomputing per pool slot (not hoisting one call
        // out of the inner loop) is required to match NumPy exactly.
        for &extra in entropy.iter().skip(n) {
            for slot in self.pool.iter_mut() {
                let mixed_src = hm.mix(extra);
                *slot = mix_words(*slot, mixed_src);
            }
        }
    }

    /// Draw `n_words` further-hashed `u32` state words from the pool.
    ///
    /// A `BitGenerator` should call this (or [`generate_state_u64`](Self::generate_state_u64)) in its
    /// constructor with whatever `n_words` its internal state needs.
    /// Successive words keep advancing an internal hash constant while
    /// cycling through the pool, so this is deterministic given the pool
    /// but not simply "the pool repeated".
    ///
    /// [`generate_state_u64`](Self::generate_state_u64): Self::generate_state_u64
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hm = HashMix::new(INIT_B, MULT_B);
        let pool_len = self.pool.len();
        (0..n_words)
            .map(|i| {
                let data_val = self.pool[i % pool_len];
                hm.mix(data_val)
            })
            .collect()
    }

    /// Draw `n_words` further-hashed `u64` state words from the pool.
    ///
    /// Equivalent to `generate_state(n_words, dtype=np.uint64)` in NumPy:
    /// internally draws `2 * n_words` `u32` words (a *continuation* of the
    /// same hash stream as [`generate_state_u32`](Self::generate_state_u32), not two independent
    /// draws) and pairs them up little-endian (`word[2i] | word[2i+1] << 32`).
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let words = self.generate_state_u32(n_words * 2);
        words
            .chunks_exact(2)
            .map(|pair| (pair[0] as u64) | ((pair[1] as u64) << 32))
            .collect()
    }

    /// Spawn `n_children` new, statistically independent child
    /// `SeedSequence`s.
    ///
    /// Each call hands out the next `n_children` spawn keys (tracked via an
    /// internal counter), so repeated calls on the same `SeedSequence` never
    /// reuse a spawn key and every spawned sequence — including children of
    /// children — is independent of every other one produced from the same
    /// root entropy.
    pub fn spawn(&mut self, n_children: usize) -> Vec<SeedSequence> {
        let mut children = Vec::with_capacity(n_children);
        for offset in 0..n_children {
            let child_index = self.n_children_spawned + offset as u32;
            let mut spawn_key = self.spawn_key.clone();
            spawn_key.push(child_index);
            children.push(SeedSequence::from_parts(
                self.run_entropy.clone(),
                spawn_key,
            ));
        }
        self.n_children_spawned += n_children as u32;
        children
    }

    /// The mixed entropy pool, exposed for debugging/testing parity with
    /// NumPy's `SeedSequence.pool` attribute.
    pub fn pool(&self) -> &[u32] {
        &self.pool
    }

    /// This sequence's spawn key (empty for a root sequence).
    pub fn spawn_key(&self) -> &[u32] {
        &self.spawn_key
    }

    /// Number of children already produced by [`spawn`](Self::spawn).
    pub fn n_children_spawned(&self) -> u32 {
        self.n_children_spawned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Reference values pinned against `np.random.SeedSequence` -----
    //
    // Computed with numpy 2.4.2:
    //     import numpy as np
    //     np.random.SeedSequence(<entropy>).generate_state(4)
    //
    // covering entropy = 0, small values, a value that needs exactly one
    // u32 word (2**31), and values that need two u32 words including the
    // full u64 range boundary (2**63+7, u64::MAX).

    #[test]
    fn generate_state_u32_matches_numpy_entropy_0() {
        let seq = SeedSequence::new(0);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![2968811710, 3677149159, 745650761, 2884920346]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_entropy_1() {
        let seq = SeedSequence::new(1);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![1835504127, 1731038949, 1320224556, 2330041505]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_entropy_42() {
        let seq = SeedSequence::new(42);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![3444837047, 2669555309, 2046530742, 3581440988]
        );
        // Pool contents are also pinned (numpy exposes `.pool`).
        assert_eq!(seq.pool(), &[1662858758, 128880814, 1875164712, 753753205]);
    }

    #[test]
    fn generate_state_u32_matches_numpy_entropy_12345() {
        let seq = SeedSequence::new(12345);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![2688385916, 3048105090, 4196366895, 3152189807]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_entropy_2_pow_31() {
        // Exactly one u32 word of entropy (2**31 < 2**32).
        let seq = SeedSequence::new(1u64 << 31);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![3155214650, 3649071964, 1060818499, 334526526]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_entropy_2_pow_63_plus_7() {
        // Needs two u32 words of entropy.
        let seq = SeedSequence::new((1u64 << 63) + 7);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![319677963, 4187467128, 3671185491, 765353098]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_entropy_u64_max() {
        let seq = SeedSequence::new(u64::MAX);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![2458692877, 2931597649, 2251873402, 295448644]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_sequence_entropy() {
        // np.random.SeedSequence([1, 2, 3]).generate_state(4)
        let seq = SeedSequence::from_u64_sequence(&[1, 2, 3]);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![3822189696, 3026158655, 540542919, 1119972918]
        );
    }

    #[test]
    fn generate_state_u32_matches_numpy_sequence_entropy_with_a_wide_element() {
        // Each sequence element is independently expanded into one or two
        // u32 words (NumPy's per-element `_int_to_uint32_array`), not just
        // truncated to one word each -- exercised here with an element
        // (2**40) that needs two words:
        //     np.random.SeedSequence([2**40, 5]).generate_state(4)
        // which NumPy computes identically to
        //     np.random.SeedSequence([0, 256, 5]).generate_state(4)
        // (2**40 == 0 | 256 << 32), confirming the flattened-word form this
        // constructor produces internally.
        let seq = SeedSequence::from_u64_sequence(&[1u64 << 40, 5]);
        assert_eq!(
            seq.generate_state_u32(4),
            vec![2527941005, 1857525667, 905782527, 2553882280]
        );
    }

    #[test]
    fn generate_state_u32_of_8_matches_numpy_entropy_42() {
        // np.random.SeedSequence(42).generate_state(8): a longer draw is
        // NOT simply the 4-word draw repeated -- the hash stream keeps
        // advancing.
        let seq = SeedSequence::new(42);
        assert_eq!(
            seq.generate_state_u32(8),
            vec![
                3444837047, 2669555309, 2046530742, 3581440988, 1691623607, 2099784219, 1184028159,
                862288241
            ]
        );
    }

    #[test]
    fn generate_state_u64_matches_numpy_entropy_42() {
        // np.random.SeedSequence(42).generate_state(2, np.uint64)
        let seq = SeedSequence::new(42);
        assert_eq!(
            seq.generate_state_u64(2),
            vec![11465652750463011511, 15382171918060459190]
        );
        // np.random.SeedSequence(42).generate_state(4, np.uint64)
        assert_eq!(
            seq.generate_state_u64(4),
            vec![
                11465652750463011511,
                15382171918060459190,
                9018504550953525431,
                3703499796004394495,
            ]
        );
    }

    #[test]
    fn spawn_matches_numpy_entropy_42() {
        // sq = np.random.SeedSequence(42); [c.generate_state(4) for c in sq.spawn(3)]
        let mut seq = SeedSequence::new(42);
        let children = seq.spawn(3);
        assert_eq!(children.len(), 3);
        assert_eq!(seq.n_children_spawned(), 3);

        assert_eq!(children[0].spawn_key(), &[0]);
        assert_eq!(
            children[0].generate_state_u32(4),
            vec![2684470948, 3757501821, 1691896351, 1126406280]
        );

        assert_eq!(children[1].spawn_key(), &[1]);
        assert_eq!(
            children[1].generate_state_u32(4),
            vec![4091952314, 31242083, 366899054, 1794014678]
        );

        assert_eq!(children[2].spawn_key(), &[2]);
        assert_eq!(
            children[2].generate_state_u32(4),
            vec![233227757, 2701265274, 3388095807, 2508111505]
        );
    }

    #[test]
    fn spawn_of_spawn_matches_numpy_entropy_42() {
        // gc = np.random.SeedSequence(42).spawn(3)[0].spawn(2)
        let mut seq = SeedSequence::new(42);
        let mut children = seq.spawn(3);
        let grandchildren = children[0].spawn(2);

        assert_eq!(grandchildren[0].spawn_key(), &[0, 0]);
        assert_eq!(
            grandchildren[0].generate_state_u32(4),
            vec![3142992634, 1861194734, 1430013548, 2319789260]
        );

        assert_eq!(grandchildren[1].spawn_key(), &[0, 1]);
        assert_eq!(
            grandchildren[1].generate_state_u32(4),
            vec![3908812709, 3341582407, 3454793571, 679120907]
        );
    }

    #[test]
    fn spawn_never_reuses_a_key_across_repeated_calls() {
        let mut seq = SeedSequence::new(7);
        let first = seq.spawn(2);
        let second = seq.spawn(2);
        assert_eq!(first[0].spawn_key(), &[0]);
        assert_eq!(first[1].spawn_key(), &[1]);
        assert_eq!(second[0].spawn_key(), &[2]);
        assert_eq!(second[1].spawn_key(), &[3]);
        // Distinct spawn keys must produce distinct state.
        assert_ne!(
            first[0].generate_state_u32(4),
            second[0].generate_state_u32(4)
        );
    }

    #[test]
    fn from_os_entropy_produces_varying_state() {
        // Not pinned (non-deterministic by design): just check two draws
        // differ, i.e. we are not accidentally returning a fixed pool.
        let a = SeedSequence::from_os_entropy();
        let b = SeedSequence::from_os_entropy();
        assert_ne!(a.generate_state_u32(4), b.generate_state_u32(4));
    }

    #[test]
    fn new_is_deterministic() {
        let a = SeedSequence::new(999);
        let b = SeedSequence::new(999);
        assert_eq!(a.generate_state_u32(8), b.generate_state_u32(8));
        assert_eq!(a.pool(), b.pool());
    }
}
