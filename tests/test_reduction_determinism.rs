//! Determinism regression test for the parallel tier of `src/kernels/reduce.rs`.
//!
//! `kernels::reduce` is `pub(crate)`, so this integration test (a separate crate) can only
//! reach it indirectly, through the public API it now backs: `numrs2::math::{sum, mean, var,
//! std}` for `axis = None` on `f64`/`f32` arrays. Per `kernels::reduce`'s own module docs, the
//! parallel tier (engaged once a slice's length reaches `PARALLEL_MIN_LEN` == 10,000) folds
//! *fixed*-size chunks sequentially in a fixed order -- the mapping of chunks to worker threads
//! is parallel, but the chunk boundaries and the final fold are not, so the result must be
//! bit-for-bit identical no matter how many threads rayon actually has available.
//!
//! # Two complementary checks
//!
//! [`var_f64_bit_identical_under_1_vs_8_actual_threads`] and
//! [`var_f32_bit_identical_under_1_vs_8_actual_threads`] below are the primary determinism
//! proof: they build two real `rayon` thread pools (`scirs2_core::parallel_ops::ThreadPoolBuilder`
//! -- the crate's approved re-export, never a direct `rayon` dependency) sized 1 and 8, run the
//! *same* `numrs2::math::var` call inside each via `ThreadPool::install`, and assert the two
//! results are bit-for-bit equal -- entirely in one process, one `cargo nextest run`, no manual
//! shell re-invocation required, and no reliance on whatever `RAYON_NUM_THREADS` happens to be
//! set to in the environment that runs this test.
//!
//! The remaining pinned-value tests are a secondary, cheaper-to-check regression net: they pin
//! today's exact bit pattern for `sum`/`mean`/`var`/`std` at one thread count so any future
//! change to `PARALLEL_CHUNK`, `PARALLEL_MIN_LEN`, or the fold order is caught immediately (a
//! real reassociation change is expected to move these, not a bug) even without rebuilding a
//! second thread pool. They can still be run under an explicit `RAYON_NUM_THREADS` from the
//! shell to additionally confirm the *global* pool path agrees with the pin:
//!
//! ```sh
//! RAYON_NUM_THREADS=1 cargo nextest run --test test_reduction_determinism
//! RAYON_NUM_THREADS=8 cargo nextest run --test test_reduction_determinism
//! ```

use numrs2::array::Array;
use numrs2::math;
use scirs2_core::parallel_ops::ThreadPoolBuilder;

/// n = 50,000: well above `PARALLEL_MIN_LEN` (10,000), and not an exact multiple of
/// `PARALLEL_CHUNK` (8,192) -- 50,000 / 8,192 = 6 full chunks plus a 3,072-element partial
/// last chunk -- so the fold-of-partial-chunks path exercises an uneven final chunk too, not
/// just a round number of full ones.
const N: usize = 50_000;

fn data_f64() -> Vec<f64> {
    (0..N).map(|i| (i as f64) * 0.5 - 12345.0).collect()
}

fn data_f32() -> Vec<f32> {
    (0..N).map(|i| (i as f32) * 0.5 - 12345.0).collect()
}

/// Build a real, isolated `rayon` thread pool of exactly `n` threads and run `f` inside it via
/// `ThreadPool::install`, so the reduction underneath genuinely executes with that many workers
/// rather than whatever the ambient global pool happens to be sized to.
fn run_with_threads<R>(n: usize, f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    let pool = ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap_or_else(|e| panic!("failed to build a {n}-thread pool: {e}"));
    pool.install(f)
}

/// The primary determinism proof for item 7's "same `var` result under `RAYON_NUM_THREADS` 1 vs
/// 8": both thread counts are real, freshly-built `rayon` pools exercised in the same process
/// and the same test run, not a pinned constant that only means something if a human remembers
/// to re-invoke the binary under a second environment variable. `N` (50,000) clears
/// `PARALLEL_MIN_LEN` (10,000) and is not an exact multiple of `PARALLEL_CHUNK` (8,192), so the
/// fixed-chunk fold-of-partial-sums path (with an uneven last chunk) is what's actually being
/// compared across thread counts, not the single-threaded SIMD tier.
#[test]
fn var_f64_bit_identical_under_1_vs_8_actual_threads() {
    let arr = Array::from_vec(data_f64());

    let got_1 = run_with_threads(1, || {
        math::var(&arr, None, 0, false)
            .expect("var should succeed")
            .to_vec()[0]
    });
    let got_8 = run_with_threads(8, || {
        math::var(&arr, None, 0, false)
            .expect("var should succeed")
            .to_vec()[0]
    });

    assert_eq!(
        got_1.to_bits(),
        got_8.to_bits(),
        "var_f64 must be bit-for-bit identical under 1 vs 8 real rayon threads (the fixed-chunk \
         kernel's whole determinism guarantee) -- got {got_1} ({:#x}) under 1 thread, {got_8} \
         ({:#x}) under 8 threads",
        got_1.to_bits(),
        got_8.to_bits()
    );
}

/// `f32` companion: a distinct kernel instantiation (`kernels::reduce`'s macro-generated `_f32`
/// twins), not merely the same code re-run with a different scalar type.
#[test]
fn var_f32_bit_identical_under_1_vs_8_actual_threads() {
    let arr = Array::from_vec(data_f32());

    let got_1 = run_with_threads(1, || {
        math::var(&arr, None, 0, false)
            .expect("var should succeed")
            .to_vec()[0]
    });
    let got_8 = run_with_threads(8, || {
        math::var(&arr, None, 0, false)
            .expect("var should succeed")
            .to_vec()[0]
    });

    assert_eq!(
        got_1.to_bits(),
        got_8.to_bits(),
        "var_f32 must be bit-for-bit identical under 1 vs 8 real rayon threads -- got {got_1} \
         ({:#x}) under 1 thread, {got_8} ({:#x}) under 8 threads",
        got_1.to_bits(),
        got_8.to_bits()
    );
}

#[test]
fn sum_f64_is_deterministic_across_thread_counts() {
    let arr = Array::from_vec(data_f64());
    let got = math::sum(&arr, None, false)
        .expect("sum should succeed")
        .to_vec()[0];
    // Pinned from an actual run (RAYON_NUM_THREADS=8) of this exact code.
    assert_eq!(
        got.to_bits(),
        4710066088337997824u64,
        "sum_f64 bit pattern changed -- got {got} ({:#x}); re-run under both \
         RAYON_NUM_THREADS=1 and RAYON_NUM_THREADS=8 before updating this pin, to confirm \
         a genuine algorithm change rather than a thread-count-dependent regression",
        got.to_bits()
    );
}

#[test]
fn mean_f64_is_deterministic_across_thread_counts() {
    let arr = Array::from_vec(data_f64());
    let got = math::mean(&arr, None, false)
        .expect("mean should succeed")
        .to_vec()[0];
    assert_eq!(
        got.to_bits(),
        4639648798144987136u64,
        "mean_f64 bit pattern changed -- got {got} ({:#x})",
        got.to_bits()
    );
}

#[test]
fn var_f64_ddof0_is_deterministic_across_thread_counts() {
    let arr = Array::from_vec(data_f64());
    let got = math::var(&arr, None, 0, false)
        .expect("var should succeed")
        .to_vec()[0];
    assert_eq!(
        got.to_bits(),
        4722259316520779776u64,
        "var_f64 (ddof=0) bit pattern changed -- got {got} ({:#x})",
        got.to_bits()
    );
}

#[test]
fn std_f64_ddof0_is_deterministic_across_thread_counts() {
    let arr = Array::from_vec(data_f64());
    let got = math::std(&arr, None, 0, false)
        .expect("std should succeed")
        .to_vec()[0];
    assert_eq!(
        got.to_bits(),
        4664657056377925821u64,
        "std_f64 (ddof=0) bit pattern changed -- got {got} ({:#x})",
        got.to_bits()
    );
}

#[test]
fn sum_f32_is_deterministic_across_thread_counts() {
    let arr = Array::from_vec(data_f32());
    let got = math::sum(&arr, None, false)
        .expect("sum should succeed")
        .to_vec()[0];
    assert_eq!(
        got.to_bits(),
        1256988984u32,
        "sum_f32 bit pattern changed -- got {got} ({:#x})",
        got.to_bits()
    );
}

/// Sanity companion (not itself a determinism claim): confirms the pinned `f64` values above
/// actually match an independent, naive (non-chunked, non-parallel) computation, so a future
/// reader knows the pins are *correct*, not merely *stable*.
#[test]
fn pinned_values_match_naive_computation() {
    let data = data_f64();
    let naive_sum: f64 = data.iter().sum();
    let naive_mean = naive_sum / data.len() as f64;
    let naive_ssd: f64 = data
        .iter()
        .map(|&x| (x - naive_mean) * (x - naive_mean))
        .sum();
    let naive_var = naive_ssd / data.len() as f64;
    let naive_std = naive_var.sqrt();

    let arr = Array::from_vec(data);
    let got_sum = math::sum(&arr, None, false).expect("sum").to_vec()[0];
    let got_mean = math::mean(&arr, None, false).expect("mean").to_vec()[0];
    let got_var = math::var(&arr, None, 0, false).expect("var").to_vec()[0];
    let got_std = math::std(&arr, None, 0, false).expect("std").to_vec()[0];

    assert!((got_sum - naive_sum).abs() / naive_sum.abs() < 1e-9);
    assert!((got_mean - naive_mean).abs() / naive_mean.abs() < 1e-9);
    assert!((got_var - naive_var).abs() / naive_var.abs() < 1e-9);
    assert!((got_std - naive_std).abs() / naive_std.abs() < 1e-9);
}
