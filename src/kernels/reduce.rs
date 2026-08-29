//! Deterministic, dtype-dispatched reduction kernels.
//!
//! `sum`/`mean`/`sum_sq_dev` tier on slice length at
//! [`super::PARALLEL_MIN_LEN`]: a single-threaded
//! `scirs2-core::simd_ops::SimdUnifiedOps` call below it, a `rayon`-backed
//! pass over *fixed* [`super::PARALLEL_CHUNK`]-sized chunks (each reduced
//! independently, then folded together sequentially in chunk order) above
//! it. Because the chunk boundaries are fixed by `PARALLEL_CHUNK` alone --
//! never derived from the number of threads actually available -- the
//! partial-sum tree, and so the final bit pattern, is the same on every
//! machine regardless of how many cores evaluate it.
//!
//! That determinism guarantee is scoped to *within* one side of the
//! `PARALLEL_MIN_LEN` boundary, not across it: the single-chunk SIMD sum
//! used below the threshold and the multi-chunk fold-of-partial-sums used
//! above it are two different summation trees, so a result can differ in
//! its last bit right at that boundary. That is expected floating-point
//! reassociation, not a bug -- do not assert bit-exact equality between a
//! length-9,999 and a length-10,001 call.
//!
//! # `min`/`max`: `NaN` propagates, exactly like NumPy
//!
//! [`min_f64`]/[`max_f64`] and their `f32` twins implement **NumPy's
//! `np.min`/`np.max` rule: any `NaN` anywhere in the input makes the
//! result `NaN`** -- independent of how many `NaN`s there are, where they
//! sit, how long the slice is, which tier (serial or parallel) runs, and
//! which dtype is being reduced. f64 and f32 agree exactly. The
//! `NaN`-*ignoring* variants are `nanmin`/`nanmax`, which live in
//! [`crate::math::nan_handling`] and do not go through these kernels.
//!
//! These two kernels deliberately do **not** call
//! `scirs2_core::simd_ops::SimdUnifiedOps::simd_min_element`/
//! `simd_max_element`, which they previously wrapped. That upstream pair
//! was found to return a **wrong, finite value** -- neither the true
//! extremum nor `NaN` -- for some `NaN` placements: on the `scirs2-core`
//! version in this workspace, a 64-element `f64` slice shaped
//! `[5.0, 1.0 (x7), NaN, 1.0 (x55)]` (true maximum `5.0` at index 0, one
//! `NaN` at index 8) makes `simd_max_element` return `1.0`, silently
//! discarding the real maximum. The cause is lane poisoning in upstream's
//! vectorized fold: `MAXPD(a, b)` yields `b` whenever either operand is
//! `NaN`, so a lane whose accumulator goes `NaN` is simply overwritten by
//! the next chunk, taking the real maximum with it. **The exact set of bad
//! placements therefore depends on the SIMD lane width chosen at runtime**
//! -- the `NaN` has to share a lane with the maximum -- so it is also
//! length-dependent, and a placement that happens to miss the maximum's
//! lane returns the correct value by luck rather than by fix. Index 8 is
//! the witness used above precisely because it is lane-aligned for every
//! width the kernel can pick (2 for SSE2/aarch64 NEON, 4 for AVX2). The
//! defect reproduces with zero `numrs2` code in between -- see
//! `crate::stats::basic`'s
//! `simd_max_element_upstream_wrong_value_is_a_live_bug_not_just_new_nan_convention`
//! test, which calls the upstream function directly and pins the bad
//! value as a tripwire for an upstream fix. Silently dropping a real
//! extremum is not a `NaN` convention this crate can expose, so
//! `min`/`max` are built here out of plain comparisons instead.
//!
//! Both tier the same *shape* as `sum`/`sum_sq_dev`, for the same
//! determinism reason -- below the threshold a single sequential fold;
//! above it, one `(partial extremum, saw_nan)` pair per fixed
//! [`super::PARALLEL_CHUNK`]-sized chunk, folded back together
//! sequentially in chunk order -- but on a *different, higher* threshold,
//! [`MINMAX_PARALLEL_MIN_LEN`] rather than [`super::PARALLEL_MIN_LEN`].
//! An extremum is one compare per element with no multiply and no
//! carried accumulator, so `rayon`'s dispatch cost takes much longer to
//! amortize than `sum`'s or `var`'s; see that constant's docs for the
//! measurements (tiering `min`/`max` at `PARALLEL_MIN_LEN` measured
//! 0.47x-0.65x of the fold it replaced at `n = 10_000`).
//!
//! Comparison-based extrema are exactly reassociation-invariant, so
//! unlike `sum` these are bit-identical on *both* sides of their
//! threshold, not merely within one side.
//!
//! All kernels here use the crate's existing empty-input convention (`0`
//! for sum/mean/min/max, matching `stats::basic::Statistics`; `1` for
//! product, the multiplicative identity, matching `Iterator::product`).
//!
//! # `simd_variance`/`simd_std`: never use them
//!
//! `scirs2_core::simd_ops::SimdUnifiedOps::{simd_variance, simd_std}`
//! compute the *sample* variance/stddev (denominator `n - 1`, hardcoded)
//! and return `NaN` for `n < 2`. NumRS2's public variance/std API takes an
//! explicit `ddof` (0 for population, 1 for sample, or any other value),
//! which a hardcoded `n - 1` cannot express, and NumPy's `ddof=0` default
//! disagrees with it outright for `n >= 2`. Callers that need variance/std
//! MUST build it themselves as `sum_sq_dev / (n - ddof)` from
//! [`sum_sq_dev_f64`] (or the `f32` twin) -- **never** call
//! `simd_variance`/`simd_std` from `kernels/` or its callers.

use super::{PARALLEL_CHUNK, PARALLEL_MIN_LEN};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Slice length at or above which [`var_f64`]/[`var_f32`] run both of
/// their passes on the chunked parallel tier instead of the serial one.
///
/// The fused variance kernels make exactly **one** tier decision, taken
/// against this constant, and then run *both* passes (mean, then sum of
/// squared deviations) on the tier it picked -- never one pass per tier
/// decision. That is the whole point of them: `math::var`/`stats::basic`'s
/// `var` previously took a separate `PARALLEL_MIN_LEN` decision per pass,
/// so a length just past the threshold paid two independent `rayon`
/// dispatches for work that barely covers one, and measured ~0.65x of the
/// pre-kernel sequential two-pass code at `n = 10_000`.
///
/// # Why this value
///
/// Measured by this file's `probe_var_threshold_candidates` test (see its
/// doc comment for how to re-run it): optimized profile, alternating A/B
/// samples, *minimum* over samples -- the machine is shared with other
/// concurrent work, so a mean would measure the other tenants and a
/// minimum measures the code. Each candidate is the fused kernel below
/// with only this constant changed. The baseline is the pre-kernel
/// sequential two-pass body (one `fold` for the mean, one
/// `Iterator::sum` for the squared deviations) that the reductions lane
/// replaced -- the arm the reported regression was measured against.
/// Ratio is `baseline / candidate`; higher is better, and `< 1.0` is the
/// regression. Three independent runs:
///
/// | n | `PARALLEL_MIN_LEN` (10_000) | `2x` (20_000) | `4x` (40_000) |
/// |---|---|---|---|
/// | 10_000 | 0.90x, 0.80x, 0.72x | 1.39x, 1.39x, 1.39x | 1.39x, 1.39x, 1.39x |
/// | 20_000 | 0.91x, 0.89x, 0.83x | 1.03x, 0.86x, 0.86x | 1.39x, 1.39x, 1.39x |
/// | 40_000 | 1.57x, 1.52x, 1.51x | 1.54x, 1.62x, 1.53x | 1.52x, 1.56x, 1.53x |
/// | 100_000 | 2.58x, 2.83x, 2.79x | 2.72x, 2.84x, 2.76x | 2.55x, 2.72x, 2.77x |
/// | 1_000_000 | 4.54x, 6.65x, 5.92x | 4.38x, 6.09x, 6.96x | 5.09x, 6.39x, 6.97x |
///
/// Two rows discriminate, and both point the same way:
///
/// - `n = 10_000`: `PARALLEL_MIN_LEN` is the regression -- 0.72x-0.90x,
///   i.e. measurably *slower* than the sequential code it replaced --
///   while `2x` and `4x` both sit at 1.39x. The two `rayon` dispatches
///   cost more than the work they split at this size.
/// - `n = 20_000`: `2x` has just switched to the parallel tier and is
///   back at 0.86x-1.03x, no better than `PARALLEL_MIN_LEN`; `4x` is
///   still serial and still 1.39x. This row is why `4x` is chosen over
///   `2x`, and it is the row the original 10k/100k/1e6 question could
///   not see.
///
/// The remaining rows are **not** evidence for any candidate: from
/// `n = 40_000` up, every threshold is below `n`, so all three run
/// byte-identical code. Their spread there (e.g. 4.38x-6.97x at
/// `n = 1_000_000`, for one single implementation) is a direct read-out
/// of this shared machine's measurement noise, and is why nothing is
/// claimed from a difference smaller than it. What those rows *do* show
/// is that the parallel tier genuinely pays from `n = 40_000` on
/// (~1.5x and climbing), so the threshold is placed at the smallest size
/// where parallelism stops losing rather than being pushed out further.
/// Slice length at or above which [`min_f64`]/[`max_f64`] (and the `f32`
/// twins) switch from a single sequential pass to the chunked parallel
/// one.
///
/// **Not** [`PARALLEL_MIN_LEN`], which every other kernel here tiers on.
/// An extremum is the cheapest reduction in this file -- one compare per
/// element, no multiply, no accumulator dependency chain that has to be
/// carried across the whole slice -- so the fixed cost of handing chunks
/// to `rayon` takes far longer to amortize than it does for `sum` or
/// `var`. Measured, not assumed.
///
/// # Why this value
///
/// From this file's `probe_minmax_threshold_candidates` test (see its doc
/// comment to re-run it), same method as [`VAR_PARALLEL_MIN_LEN`]:
/// optimized profile, alternating A/B samples, minimum over samples.
/// Baseline is the pre-kernel seeded comparison fold
/// (`a.iter().skip(1).fold(a[0], ..)`) that `math::aggregation::max` used
/// before this lane; ratio is `baseline / candidate`, higher is better,
/// `< 1.0` is a regression. Three independent runs:
///
/// | n | `1x` (10_000) | `4x` (40_000) | `8x` (80_000) | `16x` (160_000) |
/// |---|---|---|---|---|
/// | 10_000 | 0.59x, 0.65x, 0.47x | 1.00x, 0.99x, 1.00x | 1.00x, 1.00x, 1.00x | 1.00x, 1.00x, 1.00x |
/// | 20_000 | 0.54x, 0.54x, 0.64x | 1.00x, 0.99x, 1.00x | 1.00x, 0.98x, 1.00x | 1.00x, 0.98x, 0.99x |
/// | 40_000 | 0.83x, 0.94x, 0.95x | 0.83x, 0.92x, 0.91x | 1.00x, 1.00x, 0.99x | 1.00x, 1.00x, 1.00x |
/// | 80_000 | 1.70x, 1.58x, 1.36x | 1.78x, 1.52x, 1.38x | 1.75x, 1.50x, 1.39x | 1.00x, 1.00x, 0.99x |
/// | 160_000 | 2.29x, 1.76x, 1.75x | 2.27x, 1.79x, 1.81x | 2.20x, 1.77x, 1.83x | 2.18x, 1.75x, 1.78x |
/// | 1_000_000 | 4.15x, 2.72x, 3.02x | 4.12x, 2.70x, 2.90x | 4.14x, 3.51x, 2.81x | 4.07x, 2.20x, 2.95x |
///
/// Reading it: a candidate scores ~1.00x wherever it is still on the
/// serial tier (the serial loop and the baseline fold are the same
/// computation, so they compile to the same speed), and departs from
/// 1.00x only once `n` reaches it.
///
/// - `1x` is a **0.47x-0.65x regression** at `n = 10_000` and 0.54x-0.64x
///   at `n = 20_000` -- i.e. two to twice-and-a-half times slower than
///   the fold it replaced. This is the same failure mode
///   [`VAR_PARALLEL_MIN_LEN`] documents, and it bites harder here because
///   there is even less work per element to hide the dispatch behind.
/// - `4x` still loses at `n = 40_000` (0.83x-0.92x): parallelism has not
///   started paying yet at that size.
/// - Parallelism first wins at `n = 80_000` (1.36x-1.78x), and `8x` is
///   the smallest candidate that is on the parallel tier there while
///   never having been a loss below it.
/// - `16x` is strictly worse than `8x`: identical everywhere except
///   `n = 80_000`, where it is still serial and forfeits that ~1.4x.
///
/// From `n = 160_000` up all four candidates run identical code, so that
/// spread (e.g. 2.20x-4.15x at `n = 1_000_000`) is this shared machine's
/// measurement noise and is why no claim above rests on a smaller gap.
pub(crate) const MINMAX_PARALLEL_MIN_LEN: usize = 8 * PARALLEL_MIN_LEN;

pub(crate) const VAR_PARALLEL_MIN_LEN: usize = 4 * PARALLEL_MIN_LEN;

macro_rules! define_reduce_kernels {
    ($ty:ty, $sum:ident, $mean:ident, $sum_sq_dev:ident, $var:ident, $min:ident, $max:ident, $prod:ident) => {
        /// Sum of all elements. `0` for an empty slice.
        pub(crate) fn $sum(a: &[$ty]) -> $ty {
            if a.is_empty() {
                return 0.0;
            }
            if a.len() < PARALLEL_MIN_LEN {
                <$ty as SimdUnifiedOps>::simd_sum(&ArrayView1::from(a))
            } else {
                // Fixed-size chunks: the mapping of chunks to threads is
                // parallel, but the chunk boundaries and the final
                // sequential fold are not, so the result is independent
                // of the thread count.
                a.par_chunks(PARALLEL_CHUNK)
                    .map(|chunk| <$ty as SimdUnifiedOps>::simd_sum(&ArrayView1::from(chunk)))
                    .collect::<Vec<$ty>>()
                    .into_iter()
                    .fold(0.0, |acc, x| acc + x)
            }
        }

        /// Arithmetic mean. `0` for an empty slice.
        pub(crate) fn $mean(a: &[$ty]) -> $ty {
            if a.is_empty() {
                return 0.0;
            }
            $sum(a) / (a.len() as $ty)
        }

        /// `sum((a[i] - mean)^2)`: the second pass of a two-pass variance
        /// computation. Callers compute `mean` themselves (first pass,
        /// e.g. via the matching `mean` kernel above) and pass it in.
        /// Tiers the same way `sum` does; see the module docs for why
        /// this exists instead of `simd_variance`.
        pub(crate) fn $sum_sq_dev(a: &[$ty], mean: $ty) -> $ty {
            if a.len() < PARALLEL_MIN_LEN {
                a.iter().map(|&x| (x - mean) * (x - mean)).sum()
            } else {
                // Same shape as `$sum`'s parallel branch, and for the same
                // reason: `.sum()` on the *outer* (per-chunk) iterator
                // would resolve to `rayon::iter::ParallelIterator::sum`,
                // a tree reduction whose split points depend on
                // `current_num_threads()`/work-stealing scheduling --
                // not the fixed, thread-count-independent chunk fold this
                // module promises. `collect` the per-chunk partial sums
                // (each already a plain, sequential `Iterator::sum` over
                // that chunk's *own* `.iter()`, so that inner sum is
                // fine) into a `Vec`, then fold them together
                // sequentially and in fixed chunk order.
                a.par_chunks(PARALLEL_CHUNK)
                    .map(|chunk| chunk.iter().map(|&x| (x - mean) * (x - mean)).sum::<$ty>())
                    .collect::<Vec<$ty>>()
                    .into_iter()
                    .fold(0.0, |acc, x| acc + x)
            }
        }

        /// Fused, `ddof`-aware variance: `sum((a[i] - mean)^2) / (n - ddof)`.
        ///
        /// Equivalent to `$sum_sq_dev(a, $mean(a)) / (n - ddof)` but with
        /// a **single** length-tier decision (against
        /// [`VAR_PARALLEL_MIN_LEN`], not [`PARALLEL_MIN_LEN`]) covering
        /// both passes, instead of one decision per pass -- see
        /// [`VAR_PARALLEL_MIN_LEN`]'s docs for the measurements that
        /// motivate it. Each pass keeps the exact summation tree its
        /// standalone kernel uses (per-chunk `simd_sum` for the mean, then
        /// per-chunk `Iterator::sum` of squared deviations, each folded
        /// back sequentially in fixed chunk order), so for any length on
        /// the same side of the threshold this is bit-for-bit what the two
        /// separate calls produced, and it inherits their
        /// thread-count independence unchanged.
        ///
        /// `ddof` is honored as NumPy honors it: the divisor is
        /// `n - ddof` (`0` -> population, `1` -> sample). `0` for an empty
        /// slice, matching the other kernels here. Callers must guard
        /// `n <= ddof` themselves (every in-crate caller does, with a
        /// domain-specific error message); this kernel simply divides by a
        /// non-positive divisor and returns `+inf`/`NaN` accordingly
        /// rather than inventing an error type it cannot describe.
        pub(crate) fn $var(a: &[$ty], ddof: usize) -> $ty {
            if a.is_empty() {
                return 0.0;
            }
            let n = a.len();
            // Computed in floating point, not as `n - ddof`, so an
            // unguarded `ddof >= n` caller gets `inf`/`NaN` rather than a
            // `usize` underflow panic in release-vs-debug-dependent ways.
            let divisor = (n as $ty) - (ddof as $ty);

            if n < VAR_PARALLEL_MIN_LEN {
                let mean = <$ty as SimdUnifiedOps>::simd_sum(&ArrayView1::from(a)) / (n as $ty);
                let ssd: $ty = a.iter().map(|&x| (x - mean) * (x - mean)).sum();
                ssd / divisor
            } else {
                let mean = a
                    .par_chunks(PARALLEL_CHUNK)
                    .map(|chunk| <$ty as SimdUnifiedOps>::simd_sum(&ArrayView1::from(chunk)))
                    .collect::<Vec<$ty>>()
                    .into_iter()
                    .fold(0.0, |acc, x| acc + x)
                    / (n as $ty);
                let ssd = a
                    .par_chunks(PARALLEL_CHUNK)
                    .map(|chunk| chunk.iter().map(|&x| (x - mean) * (x - mean)).sum::<$ty>())
                    .collect::<Vec<$ty>>()
                    .into_iter()
                    .fold(0.0, |acc, x| acc + x);
                ssd / divisor
            }
        }

        /// Minimum element, with NumPy `np.min` semantics: **any `NaN`
        /// anywhere in `a` makes the result `NaN`** (see the module docs
        /// for why this is a plain comparison fold rather than a
        /// `simd_min_element` call). `0` for an empty slice -- callers
        /// that need to distinguish "empty" from "the minimum happens to
        /// be zero" must check emptiness themselves (`stats::basic::ptp`
        /// errors on empty; `Array::min_optimized` returns `None`).
        pub(crate) fn $min(a: &[$ty]) -> $ty {
            /// One chunk's `(partial minimum, saw_nan)`.
            ///
            /// Branchless on purpose: `saw_nan |= ..` (bitwise `|=`, not
            /// short-circuiting `||`) and no early `return` on the first
            /// `NaN`. Both a short circuit and an early exit are
            /// data-dependent branches out of the loop body, and either
            /// one stops the autovectorizer -- which matters here because
            /// this loop replaced a call into `scirs2-core`'s hand-written
            /// SIMD kernel. The only branch left, `if x < acc`, is the
            /// ordinary min-reduction pattern LLVM already recognizes and
            /// lowers to a vector `minps`/`minpd` chain.
            fn chunk_min(chunk: &[$ty]) -> ($ty, bool) {
                let mut acc = <$ty>::INFINITY;
                let mut saw_nan = false;
                for &x in chunk {
                    saw_nan |= x.is_nan();
                    if x < acc {
                        acc = x;
                    }
                }
                (acc, saw_nan)
            }

            if a.is_empty() {
                return 0.0;
            }
            let (acc, saw_nan) = if a.len() < MINMAX_PARALLEL_MIN_LEN {
                chunk_min(a)
            } else {
                // Same fixed-chunk shape as `$sum`'s parallel branch: the
                // chunk boundaries and the final fold are thread-count
                // independent, so the answer is too.
                let partials = a
                    .par_chunks(PARALLEL_CHUNK)
                    .map(chunk_min)
                    .collect::<Vec<($ty, bool)>>();
                partials
                    .into_iter()
                    .fold((<$ty>::INFINITY, false), |(acc, nan), (x, chunk_nan)| {
                        (if x < acc { x } else { acc }, nan | chunk_nan)
                    })
            };
            if saw_nan {
                <$ty>::NAN
            } else {
                acc
            }
        }

        /// Maximum element, with NumPy `np.max` semantics: **any `NaN`
        /// anywhere in `a` makes the result `NaN`**. See [`$min`] above
        /// and the module docs. `0` for an empty slice.
        pub(crate) fn $max(a: &[$ty]) -> $ty {
            /// One chunk's `(partial maximum, saw_nan)`; branchless for
            /// the same autovectorization reason as `chunk_min` in
            /// [`$min`].
            fn chunk_max(chunk: &[$ty]) -> ($ty, bool) {
                let mut acc = <$ty>::NEG_INFINITY;
                let mut saw_nan = false;
                for &x in chunk {
                    saw_nan |= x.is_nan();
                    if x > acc {
                        acc = x;
                    }
                }
                (acc, saw_nan)
            }

            if a.is_empty() {
                return 0.0;
            }
            let (acc, saw_nan) = if a.len() < MINMAX_PARALLEL_MIN_LEN {
                chunk_max(a)
            } else {
                let partials = a
                    .par_chunks(PARALLEL_CHUNK)
                    .map(chunk_max)
                    .collect::<Vec<($ty, bool)>>();
                partials.into_iter().fold(
                    (<$ty>::NEG_INFINITY, false),
                    |(acc, nan), (x, chunk_nan)| (if x > acc { x } else { acc }, nan | chunk_nan),
                )
            };
            if saw_nan {
                <$ty>::NAN
            } else {
                acc
            }
        }

        /// Product of all elements, strictly left-to-right: no
        /// reassociation, no parallelism, no SIMD, so this always takes
        /// the same summation order regardless of slice length and
        /// matches a naive loop bit-for-bit. `1` for an empty slice
        /// (multiplicative identity, matches `Iterator::product`).
        pub(crate) fn $prod(a: &[$ty]) -> $ty {
            a.iter().fold(1.0, |acc, &x| acc * x)
        }
    };
}

define_reduce_kernels!(
    f64,
    sum_f64,
    mean_f64,
    sum_sq_dev_f64,
    var_f64,
    min_f64,
    max_f64,
    prod_f64
);
define_reduce_kernels!(
    f32,
    sum_f32,
    mean_f32,
    sum_sq_dev_f32,
    var_f32,
    min_f32,
    max_f32,
    prod_f32
);

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::parallel_ops::ThreadPoolBuilder;

    fn seq_f64(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 0.5 - 3.0).collect()
    }

    fn seq_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.25 - 1.0).collect()
    }

    // ---- sum ----

    #[test]
    fn sum_f64_matches_naive_below_threshold() {
        let a = seq_f64(100);
        let naive: f64 = a.iter().sum();
        assert!((sum_f64(&a) - naive).abs() < 1e-9);
    }

    #[test]
    fn sum_f64_matches_naive_above_threshold() {
        let a = seq_f64(20_000); // > PARALLEL_MIN_LEN
        let naive: f64 = a.iter().sum();
        let got = sum_f64(&a);
        assert!(
            (got - naive).abs() / naive.abs().max(1.0) < 1e-9,
            "got {got}, naive {naive}"
        );
    }

    #[test]
    fn sum_f64_empty_is_zero() {
        assert_eq!(sum_f64(&[]), 0.0);
    }

    #[test]
    fn sum_f32_matches_naive() {
        let a = seq_f32(500);
        let naive: f32 = a.iter().sum();
        assert!((sum_f32(&a) - naive).abs() < 1e-2);
    }

    #[test]
    fn sum_f32_empty_is_zero() {
        assert_eq!(sum_f32(&[]), 0.0);
    }

    // ---- mean ----

    #[test]
    fn mean_f64_matches_naive() {
        let a = seq_f64(1000);
        let naive: f64 = a.iter().sum::<f64>() / a.len() as f64;
        assert!((mean_f64(&a) - naive).abs() < 1e-9);
    }

    #[test]
    fn mean_f64_empty_is_zero() {
        assert_eq!(mean_f64(&[]), 0.0);
    }

    #[test]
    fn mean_f32_empty_is_zero() {
        assert_eq!(mean_f32(&[]), 0.0);
    }

    // ---- sum_sq_dev ----

    #[test]
    fn sum_sq_dev_f64_matches_naive_below_and_above_threshold() {
        for &n in &[10usize, 20_000] {
            let a = seq_f64(n);
            let mean = mean_f64(&a);
            let naive: f64 = a.iter().map(|&x| (x - mean) * (x - mean)).sum();
            let got = sum_sq_dev_f64(&a, mean);
            assert!(
                (got - naive).abs() / naive.abs().max(1.0) < 1e-9,
                "n={n}: got {got}, naive {naive}"
            );
        }
    }

    #[test]
    fn sum_sq_dev_f64_matches_ddof_variance() {
        // Population variance (ddof=0) of [1,2,3,4,5] is 2.0 (NumPy:
        // np.var([1,2,3,4,5]) == 2.0); sample variance (ddof=1) is 2.5.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = mean_f64(&a);
        let ssd = sum_sq_dev_f64(&a, mean);
        let n = a.len() as f64;
        assert!((ssd / (n - 0.0) - 2.0).abs() < 1e-12);
        assert!((ssd / (n - 1.0) - 2.5).abs() < 1e-12);
    }

    // ---- fused var ----

    /// The fused kernel must equal the two separate kernel calls it
    /// replaces, exactly, for every length on the same side of
    /// `VAR_PARALLEL_MIN_LEN` as the separate calls' own tier -- i.e.
    /// everywhere except the `[PARALLEL_MIN_LEN, VAR_PARALLEL_MIN_LEN)`
    /// window, where the fused kernel deliberately stays serial while the
    /// separate ones went parallel (a different, equally valid summation
    /// tree -- checked to a relative tolerance instead).
    #[test]
    fn var_f64_matches_the_two_separate_kernel_calls_it_fuses() {
        for &n in &[
            1usize,
            5,
            64,
            PARALLEL_MIN_LEN - 1,
            VAR_PARALLEL_MIN_LEN,
            VAR_PARALLEL_MIN_LEN + PARALLEL_CHUNK + 17,
            3 * PARALLEL_CHUNK + 17,
        ] {
            let a = seq_f64(n);
            let separate = sum_sq_dev_f64(&a, mean_f64(&a)) / (n as f64);
            let fused = var_f64(&a, 0);
            if !(PARALLEL_MIN_LEN..VAR_PARALLEL_MIN_LEN).contains(&n) {
                assert_eq!(fused.to_bits(), separate.to_bits(), "n={n}");
            } else {
                assert!(
                    (fused - separate).abs() <= 1e-9 * separate.abs().max(1.0),
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn var_f64_honors_ddof_like_numpy() {
        // np.var([1,2,3,4,5]) == 2.0 (ddof=0); np.var(.., ddof=1) == 2.5.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((var_f64(&a, 0) - 2.0).abs() < 1e-12);
        assert!((var_f64(&a, 1) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn var_empty_is_zero() {
        assert_eq!(var_f64(&[], 0), 0.0);
        assert_eq!(var_f32(&[], 0), 0.0);
    }

    #[test]
    fn var_f32_matches_the_two_separate_kernel_calls_it_fuses() {
        for &n in &[
            5usize,
            64,
            VAR_PARALLEL_MIN_LEN,
            VAR_PARALLEL_MIN_LEN + PARALLEL_CHUNK + 17,
        ] {
            let a = seq_f32(n);
            let separate = sum_sq_dev_f32(&a, mean_f32(&a)) / (n as f32);
            assert_eq!(var_f32(&a, 0).to_bits(), separate.to_bits(), "n={n}");
        }
    }

    /// Both passes of the fused kernel run on fixed chunks, so the result
    /// must not move with the thread count.
    #[test]
    fn var_bit_identical_under_1_vs_8_actual_threads() {
        // Past `VAR_PARALLEL_MIN_LEN` (so the parallel tier is what is under test) and not
        // an exact multiple of `PARALLEL_CHUNK` (so the uneven last chunk is exercised too).
        let a = seq_f64(VAR_PARALLEL_MIN_LEN + PARALLEL_CHUNK + 17);
        let run = |threads: usize| -> u64 {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap_or_else(|e| panic!("failed to build a {threads}-thread pool: {e}"));
            pool.install(|| var_f64(&a, 0).to_bits())
        };
        assert_eq!(run(1), run(8));
    }

    /// Threshold-selection probe behind [`MINMAX_PARALLEL_MIN_LEN`]'s documented table; the
    /// `min`/`max` twin of [`probe_var_threshold_candidates`] below, same method (see that
    /// test's doc comment) and same `#[ignore]` reasoning. `baseline` here is the pre-kernel
    /// comparison fold `math::aggregation::max` used before this lane.
    ///
    /// ```sh
    /// CARGO_INCREMENTAL=0 cargo nextest run --lib probe_minmax_threshold_candidates \
    ///     --run-ignored all --no-capture
    /// ```
    #[test]
    #[ignore = "measurement, not an assertion -- see this test's doc comment for how to run it"]
    fn probe_minmax_threshold_candidates() {
        use std::hint::black_box;
        use std::time::Instant;

        fn max_with_threshold(a: &[f64], threshold: usize) -> f64 {
            fn chunk_max(chunk: &[f64]) -> (f64, bool) {
                let mut acc = f64::NEG_INFINITY;
                let mut saw_nan = false;
                for &x in chunk {
                    saw_nan |= x.is_nan();
                    if x > acc {
                        acc = x;
                    }
                }
                (acc, saw_nan)
            }
            let (acc, saw_nan) = if a.len() < threshold {
                chunk_max(a)
            } else {
                a.par_chunks(PARALLEL_CHUNK)
                    .map(chunk_max)
                    .collect::<Vec<(f64, bool)>>()
                    .into_iter()
                    .fold((f64::NEG_INFINITY, false), |(acc, nan), (x, cn)| {
                        (if x > acc { x } else { acc }, nan | cn)
                    })
            };
            if saw_nan {
                f64::NAN
            } else {
                acc
            }
        }

        /// The pre-kernel body: a plain, seeded comparison fold.
        fn baseline(a: &[f64]) -> f64 {
            a.iter()
                .skip(1)
                .fold(a[0], |m, &x| if x > m { x } else { m })
        }

        fn min_time(reps: usize, mut f: impl FnMut() -> f64) -> f64 {
            let t0 = Instant::now();
            for _ in 0..reps {
                black_box(f());
            }
            t0.elapsed().as_secs_f64() / reps as f64
        }

        const SAMPLES: usize = 21;
        let candidates = [
            ("1x", PARALLEL_MIN_LEN),
            ("4x", 4 * PARALLEL_MIN_LEN),
            ("8x", 8 * PARALLEL_MIN_LEN),
            ("16x", 16 * PARALLEL_MIN_LEN),
        ];

        println!(
            "\nmin/max threshold probe (ratio = min(baseline) / min(candidate); higher is better)"
        );
        for &n in &[10_000usize, 20_000, 40_000, 80_000, 160_000, 1_000_000] {
            let a: Vec<f64> = (0..n)
                .map(|i| ((i.wrapping_mul(2_654_435_761)) % 100_000) as f64 * 0.001 - 50.0)
                .collect();
            let reps = (4_000_000 / n).max(3);
            print!("  n = {n:>9}:");
            for (label, threshold) in candidates {
                let mut best_base = f64::INFINITY;
                let mut best_cand = f64::INFINITY;
                for _ in 0..SAMPLES {
                    best_base = best_base.min(min_time(reps, || baseline(&a)));
                    best_cand = best_cand.min(min_time(reps, || max_with_threshold(&a, threshold)));
                }
                print!("   {label} = {:.2}x", best_base / best_cand);
            }
            println!();
        }
    }

    /// Threshold-selection probe behind [`VAR_PARALLEL_MIN_LEN`]'s documented table.
    ///
    /// `#[ignore]`d: it is a measurement, not an assertion -- a shared machine's load makes
    /// any threshold it could assert either flaky or meaningless. Run it explicitly to
    /// re-derive the table:
    ///
    /// ```sh
    /// CARGO_INCREMENTAL=0 cargo nextest run --lib probe_var_threshold_candidates \
    ///     --run-ignored all --no-capture
    /// ```
    ///
    /// Each candidate is the *same* code as [`var_f64`] with a different threshold constant,
    /// reimplemented locally so all three can be compared in one build. `baseline` is the
    /// pre-kernel sequential two-pass body (`fold` for the mean, `Iterator::sum` for the
    /// squared deviations) that the reductions lane replaced -- the arm the reported ~0.65x
    /// regression at `n = 10_000` was measured against. Samples alternate baseline/candidate
    /// and the *minimum* over samples is reported, never a mean: under concurrent load a mean
    /// measures the other tenants, a minimum measures the code.
    #[test]
    #[ignore = "measurement, not an assertion -- see this test's doc comment for how to run it"]
    fn probe_var_threshold_candidates() {
        use std::hint::black_box;
        use std::time::Instant;

        fn var_with_threshold(a: &[f64], ddof: usize, threshold: usize) -> f64 {
            let n = a.len();
            let divisor = (n as f64) - (ddof as f64);
            if n < threshold {
                let mean = <f64 as SimdUnifiedOps>::simd_sum(&ArrayView1::from(a)) / (n as f64);
                let ssd: f64 = a.iter().map(|&x| (x - mean) * (x - mean)).sum();
                ssd / divisor
            } else {
                let mean = a
                    .par_chunks(PARALLEL_CHUNK)
                    .map(|c| <f64 as SimdUnifiedOps>::simd_sum(&ArrayView1::from(c)))
                    .collect::<Vec<f64>>()
                    .into_iter()
                    .fold(0.0, |acc, x| acc + x)
                    / (n as f64);
                let ssd = a
                    .par_chunks(PARALLEL_CHUNK)
                    .map(|c| c.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>())
                    .collect::<Vec<f64>>()
                    .into_iter()
                    .fold(0.0, |acc, x| acc + x);
                ssd / divisor
            }
        }

        /// The pre-kernel body: one sequential pass for the mean, one for the deviations.
        fn baseline(a: &[f64], ddof: usize) -> f64 {
            let n = a.len();
            let mean = a.iter().fold(0.0_f64, |acc, &x| acc + x) / n as f64;
            let ssd: f64 = a.iter().map(|&x| (x - mean) * (x - mean)).sum();
            ssd / (n - ddof) as f64
        }

        /// Minimum wall time over `samples` runs of `reps` calls each.
        fn min_time(samples: usize, reps: usize, mut f: impl FnMut() -> f64) -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..samples {
                let t0 = Instant::now();
                for _ in 0..reps {
                    black_box(f());
                }
                let dt = t0.elapsed().as_secs_f64() / reps as f64;
                if dt < best {
                    best = dt;
                }
            }
            best
        }

        const SAMPLES: usize = 21;
        let candidates = [
            ("PARALLEL_MIN_LEN", PARALLEL_MIN_LEN),
            ("2x", 2 * PARALLEL_MIN_LEN),
            ("4x", 4 * PARALLEL_MIN_LEN),
        ];

        println!(
            "\nvar threshold probe (ratio = min(baseline) / min(candidate); higher is better)"
        );
        // 10k/100k/1e6 are the sizes the threshold question was posed at; 20k and 40k are
        // added because they are the candidate thresholds themselves -- the only other
        // places the three candidates can possibly differ.
        for &n in &[10_000usize, 20_000, 40_000, 100_000, 1_000_000] {
            let a: Vec<f64> = (0..n)
                .map(|i| ((i.wrapping_mul(2_654_435_761)) % 100_000) as f64 * 0.001 - 50.0)
                .collect();
            let reps = (2_000_000 / n).max(3);
            print!("  n = {n:>9}:");
            for (label, threshold) in candidates {
                // Alternate the two arms sample-for-sample so any drift in machine load
                // hits both equally.
                let mut best_base = f64::INFINITY;
                let mut best_cand = f64::INFINITY;
                for _ in 0..SAMPLES {
                    best_base = best_base.min(min_time(1, reps, || baseline(&a, 0)));
                    best_cand =
                        best_cand.min(min_time(1, reps, || var_with_threshold(&a, 0, threshold)));
                }
                print!("   {label} = {:.2}x", best_base / best_cand);
            }
            println!();
        }
    }

    // ---- min / max ----

    #[test]
    fn min_max_f64_match_naive() {
        let a = vec![5.0, -3.0, 8.0, 0.5, -10.0, 2.0];
        assert_eq!(min_f64(&a), -10.0);
        assert_eq!(max_f64(&a), 8.0);
    }

    #[test]
    fn min_max_f64_empty_is_zero() {
        assert_eq!(min_f64(&[]), 0.0);
        assert_eq!(max_f64(&[]), 0.0);
    }

    /// NumPy's rule, stated positively: every `NaN` placement propagates.
    ///
    /// Replaces the old `min_max_f64_nan_behavior_is_pinned_not_a_simple_rule`,
    /// which pinned the previous `simd_min_element`/`simd_max_element`
    /// wrapper's placement-dependent behavior (`[NaN, 1.0, 2.0] -> 1.0`,
    /// `[NaN] -> +inf`, ...). Those values are no longer produced by
    /// anything in this crate; see the module docs.
    #[test]
    fn min_max_f64_propagate_nan_like_numpy() {
        // np.min/np.max of any of these is nan.
        assert!(min_f64(&[1.0, f64::NAN, 3.0, -2.0, f64::NAN]).is_nan());
        assert!(max_f64(&[1.0, f64::NAN, 3.0, -2.0, f64::NAN]).is_nan());

        // First element.
        assert!(min_f64(&[f64::NAN, 1.0, 2.0]).is_nan());
        assert!(max_f64(&[f64::NAN, 1.0, 2.0]).is_nan());

        // Last element.
        assert!(min_f64(&[1.0, 2.0, f64::NAN]).is_nan());
        assert!(max_f64(&[1.0, 2.0, f64::NAN]).is_nan());

        // Sole element, and all elements.
        assert!(min_f64(&[f64::NAN]).is_nan());
        assert!(max_f64(&[f64::NAN]).is_nan());
        assert!(min_f64(&[f64::NAN, f64::NAN]).is_nan());
        assert!(max_f64(&[f64::NAN, f64::NAN]).is_nan());

        // Infinities are ordinary values, not NaNs: they must NOT be
        // confused with the `+inf`/`-inf` fold identity used internally.
        assert_eq!(min_f64(&[f64::INFINITY, 1.0]), 1.0);
        assert_eq!(max_f64(&[f64::INFINITY, 1.0]), f64::INFINITY);
        assert_eq!(min_f64(&[f64::NEG_INFINITY, 1.0]), f64::NEG_INFINITY);
        assert_eq!(max_f64(&[f64::NEG_INFINITY, 1.0]), 1.0);
    }

    /// The vector that proved the upstream `simd_max_element` defect: 64
    /// elements, true maximum `5.0` at index 0, a single `NaN` at index 10.
    /// The old kernel returned `1.0` here -- neither the true maximum nor
    /// `NaN`, a silently wrong *finite* value -- on the 2-lane vector path
    /// (SSE2, aarch64 NEON), which is where it was originally measured; on
    /// a 4-lane AVX2 path index 10 misses the maximum's lane and index 8 is
    /// the lane-aligned witness instead. These kernels do not vectorize the
    /// comparison at all, so *every* placement must be `NaN` regardless.
    #[test]
    fn min_max_f64_upstream_wrong_finite_value_vector_is_now_nan() {
        let mut data = vec![1.0f64; 64];
        data[0] = 5.0;
        data[10] = f64::NAN;
        assert!(
            max_f64(&data).is_nan(),
            "max_f64 returned {} for the vector that made the old \
             `simd_max_element` wrapper return a wrong finite 1.0",
            max_f64(&data)
        );
        assert!(min_f64(&data).is_nan());

        // The same defect was length-dependent (on the 2-lane path, len 63
        // poisoned fully where len 64 did not -- the boundary moves with the
        // lane width). Both must now be NaN.
        let mut short = vec![1.0f64; 63];
        short[0] = 5.0;
        short[10] = f64::NAN;
        assert!(max_f64(&short).is_nan());
        assert!(min_f64(&short).is_nan());

        // ... and position-dependent: index 0, 32 and 63 all behaved
        // differently upstream on the 2-lane path, and index 8 is the
        // placement that is lane-aligned with the maximum at every lane
        // width upstream can pick. All must be NaN here.
        for &pos in &[0usize, 8, 32, 63] {
            let mut d = vec![1.0f64; 64];
            d[0] = 5.0;
            d[pos] = f64::NAN;
            assert!(max_f64(&d).is_nan(), "NaN at index {pos}");
            assert!(min_f64(&d).is_nan(), "NaN at index {pos}");
        }
    }

    /// `NaN` at the first, a middle and the last position of a slice long
    /// enough to engage the *parallel* tier (so the `NaN` lands in the
    /// first, a middle and the last fixed chunk respectively).
    #[test]
    fn min_max_f64_nan_propagates_from_every_chunk_on_the_parallel_tier() {
        let n = MINMAX_PARALLEL_MIN_LEN + PARALLEL_CHUNK + 17;
        for &pos in &[0usize, PARALLEL_CHUNK + 5, 2 * PARALLEL_CHUNK, n - 1] {
            let mut data = seq_f64(n);
            data[pos] = f64::NAN;
            assert!(min_f64(&data).is_nan(), "NaN at index {pos} of {n}");
            assert!(max_f64(&data).is_nan(), "NaN at index {pos} of {n}");
        }
    }

    /// No-`NaN` correctness against a plain `Iterator` fold, at every
    /// length that straddles a tier or chunk boundary.
    #[test]
    fn min_max_no_nan_match_iterator_fold_at_tier_boundaries() {
        for &n in &[
            1usize,
            63,
            64,
            65,
            PARALLEL_MIN_LEN - 1,
            PARALLEL_MIN_LEN,
            MINMAX_PARALLEL_MIN_LEN - 1,
            MINMAX_PARALLEL_MIN_LEN,
            MINMAX_PARALLEL_MIN_LEN + PARALLEL_CHUNK + 17,
        ] {
            // Deliberately not a monotone ramp: the extremes sit in the
            // interior, so a kernel that only looked at the first or last
            // chunk would fail.
            let data: Vec<f64> = (0..n)
                .map(|i| (((i * 7919) % 10_007) as f64) * 0.125 - 600.0)
                .collect();
            let want_min = data.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
            let want_max = data
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            assert_eq!(min_f64(&data), want_min, "n={n}");
            assert_eq!(max_f64(&data), want_max, "n={n}");

            let data32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            let want_min32 = data32.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
            let want_max32 = data32
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
            assert_eq!(min_f32(&data32), want_min32, "n={n}");
            assert_eq!(max_f32(&data32), want_max32, "n={n}");
        }
    }

    /// The determinism promise, checked with real `rayon` pools rather
    /// than a pinned constant (same pattern as
    /// `tests/test_reduction_determinism.rs`). `-0.0` is deliberately kept
    /// out of the data: comparison-based extrema cannot distinguish it
    /// from `+0.0`, so which of the two bit patterns comes back would be
    /// an artifact of chunking, not a determinism defect.
    #[test]
    fn min_max_bit_identical_under_1_vs_8_actual_threads() {
        let n = MINMAX_PARALLEL_MIN_LEN + PARALLEL_CHUNK + 17;
        let data: Vec<f64> = (0..n)
            .map(|i| (((i * 7919) % 10_007) as f64) * 0.125 - 600.0)
            .collect();

        let run = |threads: usize| -> (u64, u64) {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap_or_else(|e| panic!("failed to build a {threads}-thread pool: {e}"));
            pool.install(|| (min_f64(&data).to_bits(), max_f64(&data).to_bits()))
        };

        assert_eq!(
            run(1),
            run(8),
            "min/max must be bit-for-bit identical under 1 vs 8 real rayon threads"
        );
    }

    #[test]
    fn min_max_f32_match_naive() {
        let a = vec![5.0f32, -3.0, 8.0, 0.5, -10.0, 2.0];
        assert_eq!(min_f32(&a), -10.0);
        assert_eq!(max_f32(&a), 8.0);
    }

    #[test]
    fn min_max_f32_empty_is_zero() {
        assert_eq!(min_f32(&[]), 0.0);
        assert_eq!(max_f32(&[]), 0.0);
    }

    /// `f32` now agrees with `f64` exactly -- the old
    /// `min_max_f32_nan_behavior_is_pinned_not_a_simple_rule` existed
    /// precisely because it did not (`[NaN, NaN]` used to give `+inf`/
    /// `-inf` for `f32` but `NaN` for `f64`). One rule, both dtypes.
    #[test]
    fn min_max_f32_propagate_nan_like_numpy() {
        assert!(min_f32(&[1.0, f32::NAN, 3.0, -2.0, f32::NAN]).is_nan());
        assert!(max_f32(&[1.0, f32::NAN, 3.0, -2.0, f32::NAN]).is_nan());

        assert!(min_f32(&[f32::NAN, 1.0, 2.0]).is_nan());
        assert!(max_f32(&[f32::NAN, 1.0, 2.0]).is_nan());

        assert!(min_f32(&[1.0, 2.0, f32::NAN]).is_nan());
        assert!(max_f32(&[1.0, 2.0, f32::NAN]).is_nan());

        assert!(min_f32(&[f32::NAN]).is_nan());
        assert!(max_f32(&[f32::NAN]).is_nan());

        assert!(min_f32(&[f32::NAN, f32::NAN]).is_nan());
        assert!(max_f32(&[f32::NAN, f32::NAN]).is_nan());
    }

    // ---- product ----

    #[test]
    fn prod_f64_matches_sequential_order() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(prod_f64(&a), 24.0);
    }

    #[test]
    fn prod_f64_empty_is_one() {
        assert_eq!(prod_f64(&[]), 1.0);
    }

    #[test]
    fn prod_f32_empty_is_one() {
        assert_eq!(prod_f32(&[]), 1.0);
    }
}
