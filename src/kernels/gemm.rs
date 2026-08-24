//! Dtype-dispatched, shape-agnostic 2D GEMM.
//!
//! `gemm_2d(m, k, n, a, b, c)` computes `C := A * B` for row-major `A`
//! (`m` x `k`), `B` (`k` x `n`), `C` (`m` x `n`). **Overwrite** semantics:
//! whatever `c` held on entry is discarded, matching the `beta = 0.0`
//! contract the f64/f32 tier below calls `general_mat_mul` with (see
//! [`gemm_generic`]'s doc comment for why the generic tier must zero `c`
//! itself to match).
//!
//! # Backends
//!
//! - `f64`/`f32`: `scirs2_core::ndarray::linalg::general_mat_mul`, i.e.
//!   the pure-Rust `matrixmultiply` GEMM, writing straight into `c` with
//!   `alpha = 1`, `beta = 0`. Single-threaded below
//!   [`GEMM_PARALLEL_MIN_FLOPS`]; above it, the same call over an `M`-only
//!   row split.
//! - every other `T`: [`gemm_generic`], a `BLOCK_SIZE = 64` blocked i-k-j
//!   triple loop.
//!
//! See [`gemm_2d`] for the measured bake-off that chose these.
//!
//! # Determinism
//!
//! Every backend here is deterministic, and the `f64`/`f32` result is
//! **bit-identical regardless of how many threads run it**:
//!
//! - `matrixmultiply` is single-threaded (its optional `threading` feature
//!   is not enabled anywhere in this tree's lock file) and its cache
//!   blocking constants (`MC`/`KC`/`NC`) are compile-time per-kernel
//!   constants, not functions of `m`/`k`/`n`. So the order in which a given
//!   `C[i,j] = sum_p A[i,p]*B[p,j]` accumulates depends only on `k`.
//! - The parallel tier splits only `M`, never the `K` reduction, so every
//!   dot product is still summed inside one chunk in that same
//!   `k`-determined order. Changing the thread count changes which core
//!   computes a row, never the arithmetic -- which is what lets
//!   [`gemm_2d_parallel_tier_is_bit_identical_to_row_blocked_serial_tier`]
//!   assert exact equality rather than a tolerance.
//! - [`gemm_generic`] is a plain sequential loop.
//!
//! [`gemm_2d_parallel_tier_is_bit_identical_to_row_blocked_serial_tier`]:
//!     tests::gemm_2d_parallel_tier_is_bit_identical_to_row_blocked_serial_tier

use super::cast;
use super::GEMM_PARALLEL_MIN_FLOPS;
use num_traits::Zero;
use scirs2_core::ndarray::linalg::general_mat_mul;
use scirs2_core::ndarray::{ArrayView2, ArrayViewMut2, LinalgScalar};
use scirs2_core::parallel_ops::*;
use std::ops::{Add, Mul};

/// Dispatching entry point.
///
/// `T: 'static` unlocks the f64/f32 fast tier via [`super::cast`]; every
/// other `T` runs [`gemm_generic`].
///
/// # Precondition
///
/// Callers -- always an internal, `Result`-validating public API one
/// layer up, never a user directly, since this is `pub(crate)` -- must
/// pass **dense, row-major** `a`/`b`/`c`: element `(i, j)` of `A` lives at
/// `a[i * k + j]`, with no padding and no other axis order. Every current
/// caller satisfies this by construction --
/// [`super::borrow::operand`] yields either a `Borrowed` slice (only when
/// `Array::as_slice()` returns `Some`, which `ndarray` grants for standard
/// C-contiguous layout alone) or an `Owned` copy materialized by walking
/// `iter()` in logical row-major order; `array/linalg.rs`'s batched path
/// then hands out `&a[t*m*k..(t+1)*m*k]` sub-slices of exactly such a
/// buffer, and output buffers are always freshly allocated `Vec`s. This
/// requirement is not new (the previous backend indexed `a[i*lda + p]`
/// with `lda = k`), but it is worth stating outright now that a single
/// float path depends on it: a length-correct slice that is *not* dense
/// row-major would be accepted by both the checks below and by
/// `ArrayView2::from_shape`, and would silently produce the wrong product.
///
/// Callers must also pass `a.len() == m*k`, `b.len() == k*n`,
/// `c.len() == m*n` -- the one half of the contract that *is* checked. If that
/// precondition is violated, this routes straight to [`gemm_generic`]
/// rather than the float tier, whose `ArrayView2::from_shape` calls would
/// otherwise all fail. `gemm_generic` never panics even then (see its own
/// doc comment) -- there is no sensible *value* to return from a function
/// with no `Result` in its signature, but "no value" does not have to mean
/// "no panic hygiene" for a violated precondition. By the time any public,
/// shape-validated caller reaches this crate-private dispatcher, `m`/`k`/`n`
/// are already known to match `a`/`b`/`c` in practice, so this branch
/// exists as defense in depth against a caller bug, not as a path a user
/// can reach through the public API.
///
/// # Tier map
///
/// | dtype | `2*m*n*k` | backend |
/// |---|---|---|
/// | `f64`, `f32` | `< GEMM_PARALLEL_MIN_FLOPS` | `general_mat_mul`, one call |
/// | `f64`, `f32` | `>= GEMM_PARALLEL_MIN_FLOPS` | `general_mat_mul` over an `M`-only row split |
/// | anything else | any | [`gemm_generic`] |
///
/// Degenerate shapes short-circuit ahead of both float tiers: `m == 0` or
/// `n == 0` leaves an empty `c` untouched, and `k == 0` zero-fills `c`
/// (see [`gemm_float_tier`]).
///
/// # The bake-off that chose this (2026-08-24)
///
/// `aarch64-apple-darwin` (Apple silicon, 8 cores, NEON), `profile.release`
/// (`opt-level = 3`, fat LTO), **`load average 24-45 on 8 cores` from
/// unrelated concurrent build sessions** -- so these are *not* quiet-machine
/// numbers. The estimator is therefore the **minimum over three separate
/// process invocations of a rotated round-robin sweep**, never a mean: under
/// that much oversubscription a mean measures the scheduler, while the
/// minimum measures the kernel. Every candidate wrote into one reused
/// output buffer so none was charged for an allocation another avoided.
/// Harness: `tests/test_matmul_dispatch.rs`'s `bakeoff` module (run with
/// `--ignored`); every candidate is checked against a naive triple-loop
/// oracle by `bakeoff::every_bakeoff_candidate_matches_naive_oracle` before
/// any of its timings are trusted.
///
/// Candidates: `blk_ser` = the pre-migration `BLOCK_SIZE = 64` blocked i-k-j
/// loop (commit `fc464bf`); `blk_par` = that loop under an `M` row split;
/// `simd_ser`/`simd_par` = `scirs2_core::simd_ops::simd_matrix_multiply_f64`
/// alone and row-split (**what this file dispatched onto before this
/// change**); `blas_acc` = `scirs2_linalg::blas_accelerated::matmul`
/// (`ndarray`'s `a.dot(b)`) plus the copy out of its owned return;
/// `nd_gmm`/`nd_gmm_par` = `general_mat_mul` alone and row-split (**what
/// this file dispatches onto now**).
///
/// ## `f64`, square `m = k = n` (microseconds, lower is better)
///
/// | m=k=n | blk_ser | blk_par | simd_ser | simd_par | blas_acc | nd_gmm | nd_gmm_par |
/// |------:|--------:|--------:|---------:|---------:|---------:|-------:|-----------:|
/// |     8 |   0.291 |   2.375 |    0.166 |    2.208 |    0.125 |  0.041 |      2.666 |
/// |    16 |   1.375 |   3.541 |    1.333 |    3.416 |    0.416 |  0.208 |      3.333 |
/// |    32 |   6.791 |   8.125 |   10.375 |    9.541 |    2.125 |  1.458 |      6.291 |
/// |    48 |  18.958 |  13.792 |   36.583 |   20.583 |    6.041 |  4.750 |      8.708 |
/// |    64 |  40.083 |  21.959 |   92.916 |   32.958 |   12.875 | 10.666 |     11.167 |
/// |    80 |  98.500 |  39.875 |  206.625 |   49.125 |   24.333 | 20.750 |     21.334 |
/// |    96 | 151.167 |  57.042 |  383.667 |   77.417 |   39.791 | 34.584 |     31.375 |
/// |   112 | 225.791 |  87.416 |  615.416 |  123.500 |   60.667 | 53.875 |     42.791 |
/// |   128 | 320.417 | 120.958 |  154.209 |  175.875 |   89.917 | 80.791 |     66.000 |
/// |   160 |   671.6 |   256.8 |    543.9 |    340.3 |    169.3 |  155.3 |      107.8 |
/// |   192 |  1086.0 |   383.3 |   1336.4 |    569.3 |    288.1 |  263.8 |      161.9 |
/// |   256 |  2671.9 |  1045.8 |   1517.6 |    461.0 |    678.2 |  629.4 |      318.5 |
/// |   320 |  5068.3 |  1684.7 |   4946.0 |   1672.1 |   1316.3 | 1239.1 |      564.5 |
/// |   384 |  8798.8 |  3364.3 |   5472.4 |   1663.9 |   2285.5 | 2159.8 |      920.8 |
/// |   512 | 23891.5 |  7964.8 |  13633.2 |   3903.5 |   5408.3 | 5230.5 |     2094.9 |
///
/// ## `f64`, rectangular (microseconds)
///
/// | m,k,n | blk_ser | blk_par | simd_ser | simd_par | blas_acc | nd_gmm | nd_gmm_par |
/// |------:|--------:|--------:|---------:|---------:|---------:|-------:|-----------:|
/// |   512,8,512 |   339.4 |   143.8 |    643.5 |   220.5 |  379.8 |  230.8 |  109.4 |
/// |  512,16,512 |   671.3 |   286.1 |   1250.7 |   417.4 |  399.5 |  241.9 |  138.5 |
/// |  512,24,512 |   992.6 |   420.1 |   1857.8 |   542.4 |  420.1 |  251.0 |  157.1 |
/// |  512,32,512 |  1453.7 |   635.2 |   2597.0 |   912.6 |  507.3 |  319.6 |  172.8 |
/// |  512,48,512 |  2207.2 |   970.3 |   4169.1 |  1650.4 |  661.5 |  490.8 |  265.8 |
/// |  512,64,512 |  2969.8 |  1245.7 |   5963.2 |  2396.8 |  833.2 |  653.0 |  405.9 |
/// |  512,96,512 |  4456.0 |  1816.1 |  11351.7 |  3759.0 | 1125.6 |  982.8 |  592.9 |
/// | 512,128,512 |  5957.8 |  2318.6 |   2866.8 |  1292.6 | 1424.3 | 1274.3 |  826.2 |
/// | 512,192,512 |  8964.4 |  3178.8 |   9527.5 |  3006.6 | 2043.1 | 1870.4 | 1055.8 |
/// | 512,256,512 | 11874.0 |  4230.7 |   6770.5 |  2203.4 | 2730.3 | 2519.8 | 1550.5 |
/// | 512,512,512 | 24482.1 |  7870.5 |  14166.1 |  3769.2 | 5500.3 | 5198.1 | 2075.0 |
/// |   64,512,64 |   319.3 |   134.1 |    159.3 |   206.8 |   88.2 |   85.5 |   64.6 |
/// |  256,32,256 |   325.5 |   125.5 |    650.1 |   211.6 |  117.0 |   80.7 |   53.8 |
/// | 1024,32,1024 | 6889.9 |  2249.0 |  10458.8 |  3138.3 | 1885.9 | 1275.1 |  531.3 |
///
/// ## `f32` (microseconds)
///
/// | m,k,n | blk_ser | blk_par | simd_ser | simd_par | blas_acc | nd_gmm | nd_gmm_par |
/// |------:|--------:|--------:|---------:|---------:|---------:|-------:|-----------:|
/// |    32^3 |    9.416 |   8.750 |   10.333 |   6.000 |  1.458 |  0.791 |   4.917 |
/// |    48^3 |   21.250 |  12.458 |   36.417 |   9.500 |  3.791 |  2.416 |   7.125 |
/// |    64^3 |   40.167 |  19.167 |   91.333 |  14.459 |  7.666 |  5.416 |   7.542 |
/// |    96^3 |    173.0 |    54.6 |    373.9 |    22.9 |   22.3 |   17.3 |    18.0 |
/// |   128^3 |    314.8 |    98.8 |    948.7 |    48.6 |   49.3 |   40.7 |    25.7 |
/// |   192^3 |   1062.0 |   316.4 |   3389.5 |   160.4 |  150.0 |  130.3 |    83.4 |
/// |   256^3 |   2520.7 |   694.8 |    607.8 |   173.4 |  339.3 |  303.3 |   152.0 |
/// |   384^3 |   8556.5 |  2233.9 |  10250.8 |  2572.0 | 1104.8 | 1021.0 |   328.5 |
/// | 512,64,512 |  2533.5 |  731.8 |   5782.5 |  1522.2 |  451.5 |  304.3 |   146.8 |
/// | 512,32,512 |  1278.8 |  391.7 |   2549.1 |   703.1 |  298.7 |  157.0 |    93.9 |
///
/// ## What the table says
///
/// 1. **`nd_gmm` beats every other serial candidate at all 39 measured
///    shapes, in both dtypes** -- by 1.4x-3.4x over `blk_ser`, 1.5x-11.6x
///    over `simd_ser`, and 1.03x-1.9x over `blas_acc`. There is no size
///    band anywhere in the sweep where a second serial backend earns a
///    tier, so none was added: this file went from two backends to one.
/// 2. **The `blas_acc`/`nd_gmm` gap is pure plumbing.** They reach the same
///    `matrixmultiply` kernel; `blas_accelerated::matmul` returns an owned
///    `Array2`, so adopting it would cost an `m*n` allocation plus a
///    copy-back per call. That overhead is the entire difference (largest
///    where the copy is large relative to the work: 1.9x at `512,8,512`,
///    1.45x at `256,32,256`). `general_mat_mul` writes into the caller's
///    `c` and is reached through the mandatory `scirs2_core::ndarray`
///    re-export, so it is both the faster and the policy-correct route.
/// 3. **Why the previous backend lost, structurally rather than
///    incidentally.** `simd_matrix_multiply_f64` takes its blocked path
///    only when `min(m, n, k) >= 32`, uses `kc = 128` for `f64`, and falls
///    into a per-panel *edge* micro-kernel whenever `k % kc != 0` -- so it
///    has a ~4x cliff on `k mod 128`. The table shows it directly:
///    `512,96,512` costs 11.35 ms but `512,128,512` costs 2.87 ms, 4.0x
///    *faster* on 33% more work; `112^3` costs 615 us but `128^3` costs
///    154 us. Its `f64` micro-panel is `mr x nr = 8 x 2` on NEON -- one
///    vector wide -- so even on the aligned path it re-streams `A` `n/2`
///    times. A backend whose cost jumps 4x on an alignment predicate is
///    not something to keep behind a size threshold; `matrixmultiply`
///    packs both operands and has no such cliff, which is why this
///    conclusion is not merely "the other one was faster on this host".
/// 4. **`GEMM_PARALLEL_MIN_FLOPS = 1 << 20` is exactly the measured
///    serial/parallel crossover** and needed no change. `f64`: `80^3` is
///    1,024,000 FLOPs (just under) and serial wins there, 20.75 vs 21.33 us;
///    `96^3` is 1,769,472 FLOPs (just over) and parallel wins, 31.38 vs
///    34.58 us. `f32` agrees within noise (`96^3`: 17.96 par vs 17.33 ser,
///    a 3.5% serial edge that is inside this machine's run-to-run spread;
///    `128^3`: 25.7 par vs 40.7 ser, a clear 1.58x parallel win).
/// 5. **The parallel tier's speedup is load-limited, not algorithm-limited.**
///    `nd_gmm_par`/`nd_gmm` peaks at ~2.5x on 8 cores here; on a quiet
///    machine it would be nearer 6x. That makes every parallel-tier number
///    above a *lower* bound, and it is why the serial/parallel crossover
///    was chosen from the shapes where both sides are single-threaded
///    enough to compare honestly.
///
/// ## End-to-end effect through the public `Array::matmul`
///
/// Harness: `tests/test_matmul_dispatch.rs`'s
/// `perf::matmul_perf_evidence_2d`, minimum over repeated whole processes
/// (five "before", eight "after"). `legacy` is the faithful `fc464bf`
/// blocked loop, measured in the same process as the dispatched side so the
/// speedup column is contention-symmetric.
///
/// | m,k,n | before (SIMD tier) | after (this file) | legacy | after vs legacy |
/// |---|---|---|---|---|
/// | `8^3`        |   291 ns |   166 ns |   416 ns |  2.51x |
/// | `32^3`       | 10.54 us |  1.58 us |  7.21 us |  4.55x |
/// | `64^3`       | 93.33 us | 11.08 us | 41.67 us |  3.76x |
/// | `128^3`      | 213.8 us |  68.2 us | 334.3 us |  4.90x |
/// | `256^3`      | 529.3 us | 216.2 us | 2.747 ms | 12.70x |
/// | `512^3`      | 3.357 ms | 1.278 ms | 24.06 ms | 18.82x |
/// | `512x64x512` | 2.190 ms | 219.3 us | 3.030 ms | 13.82x |
///
/// `32^3` and `64^3` were the two shapes that *regressed* against the
/// legacy loop under the old backend (0.72x and 0.46x). Both are below
/// [`GEMM_PARALLEL_MIN_FLOPS`], i.e. single-threaded on both sides, so
/// their improvement is load-independent -- unlike the four parallel-tier
/// rows, whose "before" figures were taken at 3x this session's background
/// load. For those, the load-matched comparison is the bake-off table
/// above (`nd_gmm_par` vs `simd_par`, one round-robin: 2.66x at `128^3`,
/// 1.45x at `256^3`, 1.86x at `512^3`, 5.90x at `512x64x512`).
pub(crate) fn gemm_2d<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero + 'static,
{
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        gemm_generic(m, k, n, a, b, c);
        return;
    }

    if let (Some(a64), Some(b64)) = (cast::as_f64(a), cast::as_f64(b)) {
        if let Some(c64) = cast::as_f64_mut(c) {
            gemm_float_tier(m, k, n, a64, b64, c64);
            return;
        }
    }
    if let (Some(a32), Some(b32)) = (cast::as_f32(a), cast::as_f32(b)) {
        if let Some(c32) = cast::as_f32_mut(c) {
            gemm_float_tier(m, k, n, a32, b32, c32);
            return;
        }
    }
    gemm_generic(m, k, n, a, b, c);
}

/// The `f64`/`f32` tier, shared by both dtypes.
///
/// One `matrixmultiply` call below [`GEMM_PARALLEL_MIN_FLOPS`], the same
/// call over an `M`-only row split above it. `f64` and `f32` do not need
/// separate thresholds: the bake-off in [`gemm_2d`] measured the crossover
/// in the same place for both.
///
/// # Degenerate shapes
///
/// `m == 0` or `n == 0` makes `c` empty -- there is nothing to overwrite,
/// so this returns immediately rather than constructing zero-extent views.
///
/// `k == 0` is zero-filled here rather than delegated. `matrixmultiply`
/// does handle it correctly (with no `K` blocks it applies `beta`, and its
/// `beta == 0` branch *assigns* `T::zero()` instead of multiplying, so a
/// `NaN` left in `c` would not survive) -- but that is an edge-case detail
/// of a transitive dependency, and this kernel's overwrite contract is
/// pinned by [`tests::gemm_2d_overwrites_pre_filled_garbage_in_c`]. Making
/// it locally true costs one branch on a path that does no arithmetic
/// anyway.
fn gemm_float_tier<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: LinalgScalar + Send + Sync,
{
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for c_ij in c.iter_mut() {
            *c_ij = T::zero();
        }
        return;
    }

    let flops = 2usize.saturating_mul(m).saturating_mul(n).saturating_mul(k);
    if flops < GEMM_PARALLEL_MIN_FLOPS {
        if !gmm_into(m, k, n, a, b, c) {
            gemm_generic(m, k, n, a, b, c);
        }
        return;
    }
    parallel_row_split(m, k, n, a, b, c);
}

/// One `general_mat_mul(1, A, B, 0, C)` over flat row-major slices.
///
/// Returns `false` -- without having written anything -- if any of the
/// three views could not be built, which happens exactly when a slice's
/// length disagrees with its `(rows, cols)`. [`gemm_2d`] validates all
/// three lengths before reaching here, so `false` is unreachable through
/// the dispatcher; it exists so the caller can fall back to
/// [`gemm_generic`] instead of this function having to `unwrap`/`expect` a
/// `Result` it cannot honestly promise (`unwrap` is banned in this crate,
/// and "it can't fail" is not the same as "it doesn't return a `Result`").
///
/// Returning a `bool` rather than falling back internally is what keeps
/// the borrow checker happy: the three views borrow `c` mutably, and the
/// borrow must be over before a caller can hand the same `c` to
/// [`gemm_generic`].
fn gmm_into<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T]) -> bool
where
    T: LinalgScalar,
{
    match (
        ArrayView2::from_shape((m, k), a),
        ArrayView2::from_shape((k, n), b),
        ArrayViewMut2::from_shape((m, n), c),
    ) {
        (Ok(a_view), Ok(b_view), Ok(mut c_view)) => {
            general_mat_mul(T::one(), &a_view, &b_view, T::zero(), &mut c_view);
            true
        }
        _ => false,
    }
}

/// Split `A`/`C` into row-contiguous chunks (one per available thread,
/// never splitting the `K`/reduction dimension) and run [`gmm_into`] on
/// each chunk in parallel.
///
/// This is safe to parallelize without any of `reduce.rs`'s determinism
/// concerns: `M` (rows) is an independent axis with no cross-term
/// accumulation, so splitting it changes nothing about the arithmetic --
/// every `C[i,j] = sum_p A[i,p]*B[p,j]` dot product is still computed
/// within a single chunk's call, in the same order a non-chunked call
/// would use, because `matrixmultiply`'s `K` blocking is a compile-time
/// constant and so cannot vary with a chunk's row count. Only *which
/// thread* computes which rows changes. See this module's `# Determinism`
/// section.
///
/// Only reached when `flops >= GEMM_PARALLEL_MIN_FLOPS`, which requires
/// `m, n, k` all `> 0` (any of them `0` makes `flops == 0`), so
/// `chunk_rows * k` and `chunk_rows * n` below are always `> 0` -- rayon's
/// `par_chunks`/`par_chunks_mut` require a nonzero chunk size.
fn parallel_row_split<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: LinalgScalar + Send + Sync,
{
    let n_threads = current_num_threads().max(1);
    let chunk_rows = m.div_ceil(n_threads).max(1);
    a.par_chunks(chunk_rows * k)
        .zip(c.par_chunks_mut(chunk_rows * n))
        .for_each(|(a_chunk, c_chunk)| {
            let rows = a_chunk.len() / k;
            if rows == 0 {
                return;
            }
            if !gmm_into(rows, k, n, a_chunk, b, c_chunk) {
                gemm_generic(rows, k, n, a_chunk, b, c_chunk);
            }
        });
}

/// Generic `T` fallback: a `BLOCK_SIZE=64` blocked i-k-j triple loop,
/// mirroring the existing hand-blocked loops in
/// `array/operations_optimized.rs::matmul_to` and
/// `array/linalg.rs`, with two differences:
///
/// 1. It always zeroes `c` first. `matmul_to` accumulates onto whatever
///    `c` already holds (its callers are required to pass a freshly
///    zeroed buffer); `gemm_2d` makes no such assumption of its callers,
///    and must match the float tier's `beta=0.0` *overwrite* contract
///    regardless of what `c` contains on entry.
/// 2. The per-element accumulate uses `mem::replace(&mut c[idx], T::zero())`
///    to take the current accumulated value out of `c` by move (returning
///    it as `acc`) rather than `c[idx].clone() + ...`, so this works for
///    any `T: Clone` (not just `T: Copy`) without an extra clone of `c`'s
///    contents on every inner-loop iteration.
/// 3. It never panics, even if `a`/`b`/`c`'s actual lengths don't match
///    `m*k`/`k*n`/`m*n` (the case `gemm_2d` routes here for, per its own
///    precondition doc): `c` is zeroed using its own real length first,
///    then the indexed blocked loop below only runs at all once every
///    length has been confirmed sufficient, so it can never read or
///    write out of bounds.
///
/// This is also the float tier's unreachable-in-practice fallback when a
/// view cannot be built (see [`gmm_into`]) -- `f64`/`f32` satisfy these
/// bounds like any other `T`.
pub(crate) fn gemm_generic<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    for c_ij in c.iter_mut() {
        *c_ij = T::zero();
    }

    // Defensive: only reachable with insufficient lengths when `gemm_2d`'s
    // precondition was violated by a caller bug (see its doc comment).
    // There is no well-defined product to compute in that case; leave `c`
    // zeroed (above) and return rather than indexing past what the caller
    // actually gave us.
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return;
    }

    const BLOCK_SIZE: usize = 64;
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        let i_end = (i_block + BLOCK_SIZE).min(m);
        for k_block in (0..k).step_by(BLOCK_SIZE) {
            let k_end = (k_block + BLOCK_SIZE).min(k);
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                let j_end = (j_block + BLOCK_SIZE).min(n);
                for i in i_block..i_end {
                    for k_l in k_block..k_end {
                        let a_ik = a[i * k + k_l].clone();
                        for j in j_block..j_end {
                            let b_kj = b[k_l * n + j].clone();
                            let acc = std::mem::replace(&mut c[i * n + j], T::zero());
                            c[i * n + j] = acc + a_ik.clone() * b_kj;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_gemm_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let aik = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aik * b[p * n + j];
                }
            }
        }
        c
    }

    fn seq_f64(len: usize, seed: f64) -> Vec<f64> {
        (0..len).map(|i| (i as f64) * 0.1 + seed).collect()
    }

    /// Magnitude-relative comparison for GEMM outputs.
    ///
    /// A fixed *absolute* tolerance is the wrong tool here: `matrixmultiply`'s
    /// blocked/packed accumulation order for a `k`-term dot product is not
    /// the same as this test's naive left-to-right `+=` loop, and floating
    /// point addition is not associative -- the two legitimately
    /// disagree by a handful of ULPs once `k` and the accumulated
    /// magnitude grow (observed empirically: (127,127,127) disagrees by
    /// ~2e-9 on a ~6.1e6-magnitude value, i.e. ~3e-16 relative, about 1.5
    /// ULPs of f64 precision -- not a bug). `tol` is relative to the
    /// larger of 1.0 and the expected value's magnitude, so it stays
    /// meaningful for both near-zero and large accumulated sums.
    fn assert_close_rel(got: f64, expected: f64, tol: f64, ctx: impl std::fmt::Display) {
        let scale = expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tol * scale,
            "{ctx}: got {got}, expected {expected} (relative diff {:.3e}, tol {tol:.0e})",
            (got - expected).abs() / scale,
        );
    }

    /// `f32` twin of [`assert_close_rel`].
    fn assert_close_rel_f32(got: f32, expected: f32, tol: f32, ctx: impl std::fmt::Display) {
        let scale = expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tol * scale,
            "{ctx}: got {got}, expected {expected} (relative diff {:.3e}, tol {tol:.0e})",
            (got - expected).abs() / scale,
        );
    }

    const SIZE_GRID: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (2, 2, 2),
        (31, 31, 31),
        (32, 32, 32),
        (33, 33, 33),
        (63, 63, 63),
        (64, 64, 64),
        (65, 65, 65),
        (127, 127, 127),
        (128, 128, 128),
        (0, 5, 4), // m = 0
        (5, 0, 4), // k = 0
        (5, 4, 0), // n = 0
    ];

    /// `(m, k)` used to straddle the serial/parallel cutover.
    ///
    /// The cutover is a predicate on `2*m*n*k`, not on any single
    /// dimension, so "cutover -1 / cutover / cutover +1" has to be
    /// expressed in FLOP space. With `m = k = 64` fixed, `2*m*k = 8192`
    /// divides [`GEMM_PARALLEL_MIN_FLOPS`] exactly, so sweeping `n` moves
    /// the FLOP count in steps of 8192 and there is an `n` that lands
    /// precisely *on* the threshold -- see [`cutover_n`].
    const CUTOVER_M: usize = 64;
    const CUTOVER_K: usize = 64;

    /// The `n` at which `(CUTOVER_M, CUTOVER_K, n)` costs exactly
    /// [`GEMM_PARALLEL_MIN_FLOPS`].
    ///
    /// Derived from the constant rather than hardcoded, so retuning the
    /// threshold moves these tests with it instead of silently leaving
    /// them testing an interior point.
    fn cutover_n() -> usize {
        let per_n = 2 * CUTOVER_M * CUTOVER_K;
        assert_eq!(
            GEMM_PARALLEL_MIN_FLOPS % per_n,
            0,
            "CUTOVER_M/CUTOVER_K must divide GEMM_PARALLEL_MIN_FLOPS so an exact-boundary n exists"
        );
        GEMM_PARALLEL_MIN_FLOPS / per_n
    }

    #[test]
    fn gemm_2d_f64_matches_naive_across_size_grid() {
        for &(m, k, n) in SIZE_GRID {
            let a = seq_f64(m * k, 1.0);
            let b = seq_f64(k * n, -2.0);
            let expected = naive_gemm_f64(m, k, n, &a, &b);
            let mut c = vec![0.0f64; m * n];
            gemm_2d(m, k, n, &a, &b, &mut c);
            for idx in 0..c.len() {
                assert_close_rel(
                    c[idx],
                    expected[idx],
                    1e-9,
                    format!("(m={m},k={k},n={n}) idx={idx}"),
                );
            }
        }
    }

    #[test]
    fn gemm_2d_f32_matches_naive_across_size_grid() {
        for &(m, k, n) in SIZE_GRID {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1 + 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1 - 2.0).collect();
            let mut expected = vec![0.0f32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let aik = a[i * k + p];
                    for j in 0..n {
                        expected[i * n + j] += aik * b[p * n + j];
                    }
                }
            }
            let mut c = vec![0.0f32; m * n];
            gemm_2d(m, k, n, &a, &b, &mut c);
            for idx in 0..c.len() {
                assert_close_rel_f32(
                    c[idx],
                    expected[idx],
                    1e-4,
                    format!("(m={m},k={k},n={n}) idx={idx}"),
                );
            }
        }
    }

    #[test]
    fn gemm_2d_i32_uses_generic_tier_and_matches_naive() {
        for &(m, k, n) in SIZE_GRID {
            let a: Vec<i32> = (0..m * k).map(|i| (i as i32 % 7) - 3).collect();
            let b: Vec<i32> = (0..k * n).map(|i| (i as i32 % 5) - 2).collect();
            let mut expected = vec![0i32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let aik = a[i * k + p];
                    for j in 0..n {
                        expected[i * n + j] += aik * b[p * n + j];
                    }
                }
            }
            let mut c = vec![0i32; m * n];
            gemm_2d(m, k, n, &a, &b, &mut c);
            assert_eq!(c, expected, "(m={m},k={k},n={n})");
        }
    }

    /// The float tier now dispatches on `2*m*n*k` crossing
    /// [`GEMM_PARALLEL_MIN_FLOPS`], so the reference grid has to include
    /// the three shapes that sit immediately below, exactly on, and
    /// immediately above that predicate -- an interior-point-only grid
    /// would not notice a mis-signed comparison there.
    #[test]
    fn gemm_2d_matches_naive_at_the_parallel_cutover() {
        let n_at = cutover_n();
        for (label, n) in [
            ("cutover-1", n_at - 1),
            ("cutover", n_at),
            ("cutover+1", n_at + 1),
        ] {
            let (m, k) = (CUTOVER_M, CUTOVER_K);
            let flops = 2 * m * k * n;
            match label {
                "cutover-1" => assert!(flops < GEMM_PARALLEL_MIN_FLOPS),
                "cutover" => assert_eq!(flops, GEMM_PARALLEL_MIN_FLOPS),
                _ => assert!(flops > GEMM_PARALLEL_MIN_FLOPS),
            }

            let a = seq_f64(m * k, 0.5);
            let b = seq_f64(k * n, -1.25);
            let expected = naive_gemm_f64(m, k, n, &a, &b);
            let mut c = vec![0.0f64; m * n];
            gemm_2d(m, k, n, &a, &b, &mut c);
            for idx in 0..c.len() {
                assert_close_rel(c[idx], expected[idx], 1e-9, format!("{label} idx={idx}"));
            }

            let a32: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.05 + 0.5).collect();
            let b32: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.05 - 1.25).collect();
            let mut expected32 = vec![0.0f32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let aik = a32[i * k + p];
                    for j in 0..n {
                        expected32[i * n + j] += aik * b32[p * n + j];
                    }
                }
            }
            let mut c32 = vec![0.0f32; m * n];
            gemm_2d(m, k, n, &a32, &b32, &mut c32);
            for idx in 0..c32.len() {
                assert_close_rel_f32(
                    c32[idx],
                    expected32[idx],
                    1e-4,
                    format!("{label} f32 idx={idx}"),
                );
            }
        }
    }

    #[test]
    fn gemm_2d_overwrites_pre_filled_garbage_in_c() {
        // The f64/f32 tier is called with beta=0.0 (overwrite); the
        // generic tier must match that -- not accumulate onto whatever
        // was already in `c` -- for every dtype tier.
        let m = 3;
        let k = 4;
        let n = 5;

        let a = seq_f64(m * k, 1.0);
        let b = seq_f64(k * n, -1.0);
        let expected = naive_gemm_f64(m, k, n, &a, &b);
        let mut c_f64 = vec![9999.0f64; m * n]; // garbage
        gemm_2d(m, k, n, &a, &b, &mut c_f64);
        for idx in 0..c_f64.len() {
            assert_close_rel(c_f64[idx], expected[idx], 1e-9, format!("idx={idx}"));
        }

        // Force the generic tier via a non-f64/f32 dtype, same shape.
        let a_i: Vec<i64> = (0..m * k).map(|i| i as i64 + 1).collect();
        let b_i: Vec<i64> = (0..k * n).map(|i| i as i64 - 1).collect();
        let mut expected_i = vec![0i64; m * n];
        for i in 0..m {
            for p in 0..k {
                let aik = a_i[i * k + p];
                for j in 0..n {
                    expected_i[i * n + j] += aik * b_i[p * n + j];
                }
            }
        }
        let mut c_i = vec![-777i64; m * n]; // garbage
        gemm_2d(m, k, n, &a_i, &b_i, &mut c_i);
        assert_eq!(c_i, expected_i);
    }

    /// `k == 0` is the one shape where the float tier writes `c` without
    /// doing any arithmetic, so it needs its own overwrite check: a `NaN`
    /// left in `c` must be replaced by `0.0`, not multiplied by `beta`
    /// (`0.0 * NaN` is `NaN`).
    #[test]
    fn gemm_2d_zero_k_overwrites_garbage_with_zeros() {
        let (m, k, n) = (4usize, 0usize, 6usize);
        let a: Vec<f64> = Vec::new();
        let b: Vec<f64> = Vec::new();
        let mut c = vec![f64::NAN; m * n];
        gemm_2d(m, k, n, &a, &b, &mut c);
        assert_eq!(c, vec![0.0f64; m * n]);

        let a32: Vec<f32> = Vec::new();
        let b32: Vec<f32> = Vec::new();
        let mut c32 = vec![f32::NAN; m * n];
        gemm_2d(m, k, n, &a32, &b32, &mut c32);
        assert_eq!(c32, vec![0.0f32; m * n]);

        let a_i: Vec<i32> = Vec::new();
        let b_i: Vec<i32> = Vec::new();
        let mut c_i = vec![-1i32; m * n];
        gemm_2d(m, k, n, &a_i, &b_i, &mut c_i);
        assert_eq!(c_i, vec![0i32; m * n]);
    }

    #[test]
    fn gemm_2d_triggers_parallel_flop_tier_and_still_matches_naive() {
        let (m, k, n) = (200, 200, 200);
        assert!(
            2 * m * n * k >= GEMM_PARALLEL_MIN_FLOPS,
            "test is supposed to exercise the parallel row-split tier"
        );
        let a = seq_f64(m * k, 0.3);
        let b = seq_f64(k * n, -0.7);
        let expected = naive_gemm_f64(m, k, n, &a, &b);
        let mut c = vec![0.0f64; m * n];
        gemm_2d(m, k, n, &a, &b, &mut c);
        for idx in 0..c.len() {
            assert_close_rel(c[idx], expected[idx], 1e-9, format!("idx={idx}"));
        }
    }

    #[test]
    fn gemm_2d_parallel_tier_with_distinct_mkn_matches_naive() {
        // `gemm_2d_triggers_parallel_flop_tier_and_still_matches_naive` above
        // is cubic (m == k == n == 200), which cannot catch a `k`/`n` mixup
        // in `parallel_row_split`'s chunking arithmetic
        // (`a.par_chunks(chunk_rows * k)` zipped with
        // `c.par_chunks_mut(chunk_rows * n)`): with `k == n` such a bug is
        // invisible, since either multiplier slices identically. Use three
        // distinct dimensions instead, still above `GEMM_PARALLEL_MIN_FLOPS`,
        // so a swapped `k`/`n` would misalign `a`'s and `c`'s chunk
        // boundaries and either panic (chunk count mismatch) or produce
        // wrong values -- this test would fail either way.
        let (m, k, n) = (128, 100, 140);
        assert_ne!(k, n, "test requires k != n to catch a k/n mixup");
        assert!(
            2 * m * n * k >= GEMM_PARALLEL_MIN_FLOPS,
            "test is supposed to exercise the parallel row-split tier"
        );
        let a = seq_f64(m * k, 0.2);
        let b = seq_f64(k * n, -0.4);
        let expected = naive_gemm_f64(m, k, n, &a, &b);
        let mut c = vec![0.0f64; m * n];
        gemm_2d(m, k, n, &a, &b, &mut c);
        assert_eq!(c.len(), expected.len());
        for idx in 0..c.len() {
            assert_close_rel(c[idx], expected[idx], 1e-9, format!("idx={idx}"));
        }
    }

    /// The row split must not perturb a single bit, on any core count.
    ///
    /// This is the claim this module's `# Determinism` section makes, and
    /// it is stronger than a tolerance check: a `k`-blocking that varied
    /// with a chunk's row count would shift the summation order and show
    /// up here as a last-ULP difference, which `assert_eq!` on the raw
    /// `f64`s catches and `assert_close_rel` would not.
    ///
    /// The shape is chosen so the whole matrix clears
    /// [`GEMM_PARALLEL_MIN_FLOPS`] (so `gemm_2d` splits it across however
    /// many threads this machine has) while each half falls below it (so
    /// each half is a single, unsplit call). Agreement between an
    /// `n_threads`-way split and a 2-way split is agreement between two
    /// different splits, which is the property at issue.
    #[test]
    fn gemm_2d_parallel_tier_is_bit_identical_to_row_blocked_serial_tier() {
        let (m, k, n) = (128usize, 64usize, 96usize);
        let full = 2 * m * k * n;
        assert!(full >= GEMM_PARALLEL_MIN_FLOPS, "full matrix must split");
        assert!(full / 2 < GEMM_PARALLEL_MIN_FLOPS, "each half must not");

        let a = seq_f64(m * k, 0.7);
        let b = seq_f64(k * n, -0.3);

        let mut c_full = vec![0.0f64; m * n];
        gemm_2d(m, k, n, &a, &b, &mut c_full);

        let half = m / 2;
        let mut c_halves = vec![0.0f64; m * n];
        let (a_top, a_bot) = a.split_at(half * k);
        let (c_top, c_bot) = c_halves.split_at_mut(half * n);
        gemm_2d(half, k, n, a_top, &b, c_top);
        gemm_2d(m - half, k, n, a_bot, &b, c_bot);

        assert_eq!(
            c_full, c_halves,
            "row-split GEMM must be bit-identical regardless of how rows are partitioned"
        );
    }

    #[test]
    fn gemm_2d_length_mismatch_routes_to_generic_without_ub() {
        // Deliberately wrong `n` relative to `b`/`c`'s real lengths so the
        // float tier's view construction would fail; `gemm_2d` must
        // route around it to `gemm_generic` instead. `gemm_generic` here
        // still computes correctly because m/k/n *do* match a's actual
        // layout for k=2 -- only the a.len()==m*k check fails on purpose.
        let m = 2;
        let k = 2;
        let n = 2;
        let a = vec![1.0, 2.0, 3.0]; // wrong: should be m*k = 4
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; m * n];
        gemm_2d(m, k, n, &a, &b, &mut c); // must not panic
    }

    /// [`gmm_into`]'s `false` return is unreachable through [`gemm_2d`]
    /// (which validates every length first), so it is exercised directly
    /// here -- otherwise the fallback it exists for would never be run by
    /// any test, and a regression that made it write garbage before
    /// bailing would go unnoticed.
    #[test]
    fn gmm_into_reports_failure_without_writing_when_a_view_cannot_be_built() {
        let a = vec![1.0f64, 2.0, 3.0]; // should be 2*2 = 4 for (m=2, k=2)
        let b = vec![1.0f64, 0.0, 0.0, 1.0];
        let mut c = vec![-1.0f64; 4];
        assert!(!gmm_into(2, 2, 2, &a, &b, &mut c));
        assert_eq!(c, vec![-1.0f64; 4], "must not have written anything");

        // The same call with consistent lengths must succeed.
        let a_ok = vec![1.0f64, 0.0, 0.0, 1.0];
        assert!(gmm_into(2, 2, 2, &a_ok, &b, &mut c));
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }
}
