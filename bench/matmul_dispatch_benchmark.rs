//! Criterion harness for `Array::matmul`'s migration onto the crate's
//! dtype-dispatched GEMM kernel (`src/kernels/gemm.rs`).
//!
//! Run with: `cargo bench --bench matmul_dispatch_benchmark`
//! Quick pass: `cargo bench --bench matmul_dispatch_benchmark -- --quick`
//!
//! # What "before" means here
//!
//! `src/kernels/` is `pub(crate)`, so a bench -- which is a separate crate
//! -- cannot call `kernels::gemm::gemm_2d` directly. Everything below goes
//! through the public `Array::matmul`, which is the honest end-to-end
//! measurement anyway.
//!
//! The pre-migration implementations are *replicated locally* in this file
//! ([`legacy_matmul_2d_blocked`], [`legacy_batched_ixdyn`]) rather than
//! recovered by checking out the old tree, so both sides can be measured in
//! one process, on one machine, at one moment. Each is a faithful copy of
//! what `src/array/linalg.rs` actually contained at commit `fc464bf`
//! (`git show fc464bf:src/array/linalg.rs`), not a convenient
//! approximation:
//!
//! - **2-D**: a `BLOCK_SIZE = 64` blocked i-k-j triple loop over *flat*
//!   `as_slice()` operands. Note this is already a reasonably good loop --
//!   it is emphatically not a strawman naive `O(n^3)` version, and the
//!   speedups in M1 are measured against it, not against something worse.
//! - **N-D batched**: a per-element `IxDyn` `get`/`set` walk that rebuilt
//!   an index `Vec` and re-walked strides for every multiply-add. This one
//!   *is* slow, and its cost is the point: it is what the batched path
//!   really did.
//!
//! # Groups
//!
//! - `M1` -- 2-D dispatch, `f64` and `f32`, square `{8, 32, 64, 128, 256,
//!   512}` plus a rectangular `512x64x512` (a shape where `k` is small
//!   relative to `m`/`n`, so the blocked loop's `k`-blocking buys nothing
//!   -- and the shape that exposed the retired SIMD tier's `k`-alignment
//!   cliff most sharply).
//! - `M2` -- N-D batched, `batch` in `{1, 8, 64}` x panel `{16, 64}`.
//!   `batch = 1` isolates per-call overhead; `batch = 64` at panel 64
//!   clears `kernels::GEMM_PARALLEL_MIN_FLOPS` and so exercises the
//!   parallel batch split.
//! - `M3` -- the dispatched `f64` path against
//!   `scirs2_linalg::blas_accelerated::matmul` at `128^2` and `512^2`. Since
//!   the G3 cutover (below) both sides reach the *same* `matrixmultiply`
//!   kernel, so this group no longer asks "which kernel is faster" -- it
//!   measures what the dispatcher adds and avoids on top of it: the row
//!   split at `512^2`, and the `m*n` allocation plus copy-back that
//!   `blas_accelerated`'s owned-`Array2` return forces at both sizes.
//!
//! # G3 result: the SIMD tier was replaced outright (2026-08-24)
//!
//! The earlier revision of this header reported an unresolved `128^2`
//! /`512^2` criterion bake-off and concluded "no branch was added". That
//! conclusion has been superseded. It rested on criterion *means* taken at
//! load average 52-135 on 8 cores, where the same benchmark id measured
//! 6.14 ms, 26.50 ms and 39.27 ms across three consecutive repeats -- a
//! 6.4x spread that no mean can survive. Re-measured with a
//! minimum-over-rotated-round-robin estimator (harness:
//! `tests/test_matmul_dispatch.rs`'s `bakeoff` module, `--ignored`), across
//! 39 shapes and both float dtypes, the ordering is unambiguous and stable:
//!
//! | candidate | what it is | verdict |
//! |---|---|---|
//! | `nd_gmm` | `scirs2_core::ndarray::linalg::general_mat_mul` (i.e. `matrixmultiply`), writing into the caller's `C` | **fastest serial candidate at every measured shape** |
//! | `blas_acc` | `scirs2_linalg::blas_accelerated::matmul` -- same kernel, owned return | 1.03x-1.9x slower than `nd_gmm`; the gap is exactly its alloc + copy-back |
//! | `blk_ser` | the pre-migration `BLOCK_SIZE = 64` blocked loop | 1.4x-3.4x slower than `nd_gmm`; never wins anywhere |
//! | `simd_ser` | `scirs2_core::simd_ops::simd_matrix_multiply_*` (**the old tier**) | 1.5x-11.6x slower than `nd_gmm`; has a ~4x cliff on `k mod 128` |
//! | `blk_par`, `simd_par`, `nd_gmm_par` | the same three under an `M`-only row split | `nd_gmm_par` wins above `GEMM_PARALLEL_MIN_FLOPS` |
//!
//! So `kernels::gemm::gemm_2d`'s float tier now dispatches onto
//! `general_mat_mul` -- one call below `kernels::GEMM_PARALLEL_MIN_FLOPS`,
//! row-split above it -- and the `scirs2-core` SIMD tier is gone. The full
//! per-shape table, the structural reason `simd_matrix_multiply` lost (a
//! `kc = 128` edge-kernel cliff and an `8 x 2` NEON micro-panel), and the
//! evidence that `1 << 20` is exactly the serial/parallel crossover all
//! live in `src/kernels/gemm.rs`'s doc comment above `gemm_2d`, which is
//! the single definitive record.
//!
//! End-to-end effect through the public `Array::matmul`, min over repeated
//! process invocations of the `perf` module's alternating A/B harness (five
//! "before" at load 29-37, eight "after" at load 10-11),
//! `aarch64-apple-darwin`, release + fat LTO:
//!
//! | m,k,n | before (SIMD tier) | after (`general_mat_mul` tier) | vs legacy blocked |
//! |---|---|---|---|
//! | `8^3`       | 291 ns   | 166 ns   | 2.51x  |
//! | `32^3`      | 10.54 us | 1.58 us  | 4.55x  |
//! | `64^3`      | 93.33 us | 11.08 us | 3.76x  |
//! | `128^3`     | 213.8 us | 68.2 us  | 4.90x  |
//! | `256^3`     | 529.3 us | 216.2 us | 12.70x |
//! | `512^3`     | 3.357 ms | 1.278 ms | 18.82x |
//! | `512x64x512`| 2.190 ms | 219.3 us | 13.82x |
//!
//! The two rows that used to be outright regressions against the legacy
//! blocked loop -- `32^3` at 0.72x and `64^3` at 0.46x -- are now 4.55x and
//! 3.76x wins. Those two rows, and `8^3`, are single-threaded on both sides
//! and so are load-independent; the parallel-tier rows below them were
//! measured at a lower background load than the "before" session, and the
//! load-independent comparison for those is the `bakeoff` module in
//! `tests/test_matmul_dispatch.rs` (`nd_gmm_par` vs `simd_par`, same
//! round-robin, same load: 2.66x at `128^3`, 1.45x at `256^3`, 1.86x at
//! `512^3`, 5.90x at `512x64x512`).
//!
//! # Measurement caveat: none of this was taken on a quiet machine
//!
//! Load average ran 11-45 on 8 cores throughout (and 52-135 for the
//! superseded criterion numbers above), from unrelated build sessions on
//! the same host. That is why every number quoted here is a *minimum* over
//! repeated samples and, for the before/after table, over repeated whole
//! processes -- under oversubscription a mean measures the scheduler.
//! Criterion's own confidence intervals in this file's output should be
//! read with the same suspicion: prefer the `perf` and `bakeoff` modules in
//! `tests/test_matmul_dispatch.rs` for any number you intend to act on.
//! The parallel-tier figures in particular are *lower* bounds -- the row
//! split reached only ~2.5x on 8 cores under this load, against a ~6x
//! ceiling on an idle host.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use scirs2_core::ndarray::{Array2, IxDyn};
use std::hint::black_box;

// ---------------------------------------------------------------------
// Pre-migration reference implementations (the "before" side)
// ---------------------------------------------------------------------

/// Faithful copy of `Array::matmul_2d`'s body as of commit `fc464bf`: a
/// cache-blocked (`BLOCK_SIZE = 64`) i-k-j triple loop over flat operand
/// slices, with the same `to_vec()` fallback for non-contiguous inputs.
fn legacy_matmul_2d_blocked(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let mut c_data = vec![0.0f64; m * n];
    let owned_a;
    let a_data: &[f64] = match a.as_slice() {
        Some(slice) => slice,
        None => {
            owned_a = a.to_vec();
            &owned_a
        }
    };
    let owned_b;
    let b_data: &[f64] = match b.as_slice() {
        Some(slice) => slice,
        None => {
            owned_b = b.to_vec();
            &owned_b
        }
    };

    const BLOCK_SIZE: usize = 64;
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        for k_block in (0..k).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);

                for i in i_block..i_end {
                    for k_l in k_block..k_end {
                        let a_ik = a_data[i * k + k_l];
                        for j in j_block..j_end {
                            c_data[i * n + j] += a_ik * b_data[k_l * n + j];
                        }
                    }
                }
            }
        }
    }

    Array::from_vec(c_data).reshape(&[m, n])
}

/// `f32` twin of [`legacy_matmul_2d_blocked`].
fn legacy_matmul_2d_blocked_f32(a: &Array<f32>, b: &Array<f32>) -> Array<f32> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let mut c_data = vec![0.0f32; m * n];
    let a_data = a.to_vec();
    let b_data = b.to_vec();

    const BLOCK_SIZE: usize = 64;
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        for k_block in (0..k).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);

                for i in i_block..i_end {
                    for k_l in k_block..k_end {
                        let a_ik = a_data[i * k + k_l];
                        for j in j_block..j_end {
                            c_data[i * n + j] += a_ik * b_data[k_l * n + j];
                        }
                    }
                }
            }
        }
    }

    Array::from_vec(c_data).reshape(&[m, n])
}

/// Faithful copy of the N-D batched branch of `Array::matmul` as of commit
/// `fc464bf`: for every output element, rebuild an `IxDyn` index `Vec` and
/// look the operands up through it, then `set()` the result the same way.
///
/// Restricted here to the non-broadcast case (both operands already
/// `[batch, m, k]` / `[batch, k, n]`), which is what the M2 group measures;
/// the broadcasting logic that preceded this loop is unchanged by the
/// migration and so is not part of the comparison.
fn legacy_batched_ixdyn(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let batch_shape = &a_shape[..a_shape.len() - 2];
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];

    let mut output_shape = batch_shape.to_vec();
    output_shape.push(m);
    output_shape.push(n);
    let mut result = Array::<f64>::zeros(&output_shape);

    let batch_size: usize = batch_shape.iter().product();

    for batch_idx in 0..batch_size {
        let mut batch_indices = Vec::with_capacity(batch_shape.len());
        let mut temp = batch_idx;
        for &dim in batch_shape.iter().rev() {
            batch_indices.insert(0, temp % dim);
            temp /= dim;
        }

        let mut a_indices = batch_indices.clone();
        a_indices.push(0);
        a_indices.push(0);
        let mut b_indices = batch_indices.clone();
        b_indices.push(0);
        b_indices.push(0);

        for i in 0..m {
            let a_idx_pos = a_indices.len() - 2;
            a_indices[a_idx_pos] = i;

            for j in 0..n {
                let b_idx_pos = b_indices.len() - 1;
                b_indices[b_idx_pos] = j;

                let mut sum = 0.0f64;
                for l in 0..k {
                    let a_col_pos = a_indices.len() - 1;
                    a_indices[a_col_pos] = l;
                    let b_row_pos = b_indices.len() - 2;
                    b_indices[b_row_pos] = l;

                    let a_val = a
                        .array()
                        .get(IxDyn(&a_indices))
                        .expect("batched element access should succeed");
                    let b_val = b
                        .array()
                        .get(IxDyn(&b_indices))
                        .expect("batched element access should succeed");
                    sum += a_val * b_val;
                }

                let mut output_indices = batch_indices.clone();
                output_indices.push(i);
                output_indices.push(j);
                result
                    .set(&output_indices, sum)
                    .expect("batched output write should succeed");
            }
        }
    }

    result
}

// ---------------------------------------------------------------------
// Deterministic operand construction
// ---------------------------------------------------------------------

fn seq_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i as f64) * 0.125 - 3.0).collect()
}

fn seq_f32(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32) * 0.125 - 3.0).collect()
}

fn mat_f64(m: usize, n: usize) -> Array<f64> {
    Array::from_vec(seq_f64(m * n)).reshape(&[m, n])
}

fn mat_f32(m: usize, n: usize) -> Array<f32> {
    Array::from_vec(seq_f32(m * n)).reshape(&[m, n])
}

/// `2 * m * n * k`, the conventional GEMM FLOP count, used as criterion's
/// throughput unit so the reported rate is comparable across shapes.
fn gemm_flops(m: usize, k: usize, n: usize) -> u64 {
    2 * (m as u64) * (k as u64) * (n as u64)
}

// ---------------------------------------------------------------------
// M1: 2-D dispatch
// ---------------------------------------------------------------------

/// `(m, k, n)` shapes for M1: squares across the blocking and
/// serial/parallel tier boundaries, plus one rectangular shape with a
/// short `k`.
const M1_SHAPES: &[(usize, usize, usize)] = &[
    (8, 8, 8),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (512, 64, 512),
];

fn bench_m1_matmul_2d_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("M1/matmul_2d/f64");
    // The 512-cubed cases are ~268 MFLOP each; criterion's default 100
    // samples would spend minutes on them for no extra resolution.
    group.sample_size(10);

    for &(m, k, n) in M1_SHAPES {
        let a = mat_f64(m, k);
        let b = mat_f64(k, n);
        let label = format!("{m}x{k}x{n}");
        group.throughput(Throughput::Elements(gemm_flops(m, k, n)));

        group.bench_with_input(
            BenchmarkId::new("dispatched", &label),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).expect("matmul should succeed"))),
        );
        group.bench_with_input(
            BenchmarkId::new("legacy_blocked", &label),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(legacy_matmul_2d_blocked(a, b))),
        );
    }

    group.finish();
}

fn bench_m1_matmul_2d_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("M1/matmul_2d/f32");
    group.sample_size(10);

    for &(m, k, n) in M1_SHAPES {
        let a = mat_f32(m, k);
        let b = mat_f32(k, n);
        let label = format!("{m}x{k}x{n}");
        group.throughput(Throughput::Elements(gemm_flops(m, k, n)));

        group.bench_with_input(
            BenchmarkId::new("dispatched", &label),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).expect("matmul should succeed"))),
        );
        group.bench_with_input(
            BenchmarkId::new("legacy_blocked", &label),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(legacy_matmul_2d_blocked_f32(a, b))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------
// M2: N-D batched
// ---------------------------------------------------------------------

fn bench_m2_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("M2/matmul_batched/f64");
    group.sample_size(10);

    for &batch in &[1usize, 8, 64] {
        for &panel in &[16usize, 64] {
            let a = Array::from_vec(seq_f64(batch * panel * panel)).reshape(&[batch, panel, panel]);
            let b = Array::from_vec(seq_f64(batch * panel * panel)).reshape(&[batch, panel, panel]);
            let label = format!("batch{batch}/panel{panel}");
            group.throughput(Throughput::Elements(
                batch as u64 * gemm_flops(panel, panel, panel),
            ));

            group.bench_with_input(
                BenchmarkId::new("dispatched", &label),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| black_box(a.matmul(b).expect("matmul should succeed")))
                },
            );
            group.bench_with_input(
                BenchmarkId::new("legacy_ixdyn", &label),
                &(&a, &b),
                |bench, (a, b)| bench.iter(|| black_box(legacy_batched_ixdyn(a, b))),
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------
// M3: dispatcher overhead against scirs2-linalg's BLAS-accelerated matmul
//
// Post-G3 both sides run the same `matrixmultiply` kernel, so what is
// left on the scale is the plumbing: `Array::matmul` allocates its result
// and (above `GEMM_PARALLEL_MIN_FLOPS`) splits rows across threads, while
// `blas_accelerated::matmul` is single-threaded and returns an owned
// `Array2` its caller must then copy out of. `512^2` is above the split
// threshold and `128^2` is above it too, so both rows show the row split;
// the definitive backend comparison is the `bakeoff` module in
// `tests/test_matmul_dispatch.rs`, not this group.
// ---------------------------------------------------------------------

fn bench_m3_bakeoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("M3/bakeoff/f64");
    group.sample_size(10);

    for &size in &[128usize, 512] {
        let a = mat_f64(size, size);
        let b = mat_f64(size, size);

        // Same numbers, as plain `ndarray` 2-D arrays, for the
        // `scirs2-linalg` side. Built once, outside the timing loop, so
        // the comparison is kernel-vs-kernel and not conversion overhead.
        let a_nd = Array2::from_shape_vec((size, size), seq_f64(size * size))
            .expect("shape should match data length");
        let b_nd = Array2::from_shape_vec((size, size), seq_f64(size * size))
            .expect("shape should match data length");

        let label = format!("{size}x{size}");
        group.throughput(Throughput::Elements(gemm_flops(size, size, size)));

        group.bench_with_input(
            BenchmarkId::new("kernels_gemm", &label),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).expect("matmul should succeed"))),
        );
        group.bench_with_input(
            BenchmarkId::new("blas_accelerated", &label),
            &(&a_nd, &b_nd),
            |bench, (a_nd, b_nd)| {
                bench.iter(|| {
                    black_box(
                        scirs2_linalg::blas_accelerated::matmul(&a_nd.view(), &b_nd.view())
                            .expect("matmul should succeed"),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_m1_matmul_2d_f64,
    bench_m1_matmul_2d_f32,
    bench_m2_batched,
    bench_m3_bakeoff
);
criterion_main!(benches);
