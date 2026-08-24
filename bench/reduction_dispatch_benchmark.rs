//! Criterion harness for Lane W2-C (reductions), backing `sum`/`mean`/`var`/`std`/`min`/`max`
//! onto `src/kernels/reduce.rs`'s deterministic, dtype-dispatched kernels.
//!
//! `kernels` is `pub(crate)` and not reachable from this external bench crate, so every group
//! here measures through the *public* entry points that route into it: [`numrs2::math::sum`],
//! `mean`, `var`, `std`, `min`, `max` (all `axis = None`). "before" arms are local
//! reimplementations of the pre-conversion
//! function body, not obtained by stashing or checking out the working tree -- over two dozen
//! files are concurrently modified by other in-flight lanes, so a checkout would clobber
//! unrelated work.
//!
//! Both "before" and "after" arms take `&Array<f64>`/`&Array<f32>` (not a raw slice) and
//! extract the scalar result the same way (`Array::from_vec(vec![result]).to_vec()[0]`, exactly
//! mirroring how a real caller unwraps the `Result<Array<T>>` these functions return) so that
//! the *only* thing the comparison isolates is the one thing each lane item actually changed
//! (`to_vec()` -> `kernels::borrow::operand`, and for `sum`/`mean`/`var`/`std`, generic fold ->
//! kernel dispatch) -- not incidental differences in how the input is borrowed or the output is
//! wrapped. An earlier version of this file took `&[T]` directly for "before" and skipped the
//! output-`Array` wrap/unwrap round trip that the real (pre- and post-conversion) function
//! signature always pays; that made `min`/`max` look *slower* after their `operand()` swap,
//! which was an artifact of the unfair baseline, not a real regression -- confirmed by
//! rebuilding both arms symmetrically as done here.
//!
//! # `min`/`max` ARE kernel-dispatched now
//!
//! An earlier revision of this file recorded that `math::aggregation::max`/`min` were
//! deliberately left on their original fold, because `kernels::reduce`'s `max_f64`/`min_f64`
//! then wrapped `scirs2_core::simd_ops::SimdUnifiedOps::simd_max_element`/`simd_min_element`,
//! which returns a **wrong, finite value** for certain `NaN` placements (a real extremum
//! silently dropped, not merely an unusual `NaN` convention). That upstream defect is real and
//! still unfixed -- `src/stats/basic.rs`'s
//! `simd_max_element_upstream_wrong_value_is_a_live_bug_not_just_new_nan_convention` calls the
//! upstream function directly and pins the bad value as a tripwire. The resolution was to stop
//! calling it: `kernels::reduce`'s `min`/`max` are now plain comparison folds with NumPy's
//! `NaN`-propagation rule, and `math::aggregation::max`/`min` dispatch onto them for
//! `f64`/`f32`. The `min`/`max` groups below therefore measure a real conversion again, not a
//! withheld one.
//!
//! They also measure the *second* thing that conversion needed. Tiering `min`/`max` on
//! `PARALLEL_MIN_LEN` (10,000), as `sum` does, made them **2.2x slower** than the fold they
//! replaced at exactly `n = 10_000` (`after_f64` 15.1-20.7 us vs `before_f64` 6.7-7.0 us) --
//! an extremum is one compare per element with no multiply and no carried accumulator, so
//! `rayon`'s dispatch cost has far less work to hide behind than `sum`'s or `var`'s. They
//! tier on `kernels::reduce::MINMAX_PARALLEL_MIN_LEN` (80,000) instead, measured the same way
//! `VAR_PARALLEL_MIN_LEN` was; see that constant's doc comment.
//!
//! `max`, `f64`, three runs at `--sample-size 10` (criterion's median estimate, before -> after):
//!
//! | n | before | after | ratio |
//! |---|---|---|---|
//! | 10_000 | 7.27, 6.72, 6.70 us | 10.52, 5.40, 5.37 us | 0.69x, 1.24x, 1.25x |
//! | 80_000 | 52.4, 51.0, 52.4 us | 30.9, 32.6, 26.4 us | 1.70x, 1.56x, 1.98x |
//! | 1_000_000 | 690, 674, 697 us | 165, 160, 146 us | 4.18x, 4.21x, 4.77x |
//!
//! `n = 10_000` is now on the serial tier and lands at parity or better; the 0.69x in run 1 is
//! that run's `after` outlier (10.5 us against 5.4 us in the two clean runs on identical code),
//! not a tier effect -- the machine is shared, which is also why criterion's own interval there
//! spans 6.5-18.7 us.
//!
//! # `var`: the fused-kernel regression fix
//!
//! `bench_var_10k_threshold` is a focused re-measurement of the one size where the first
//! kernel conversion made `var` *slower* than the code it replaced. `var` needs two passes
//! (mean, then squared deviations); routing each through its own `PARALLEL_MIN_LEN` decision
//! meant an `n` just past the threshold paid two independent `rayon` dispatches for work that
//! barely covers one. `kernels::reduce::var_f64`/`var_f32` fuse the two passes under a single
//! decision taken against `VAR_PARALLEL_MIN_LEN` (= `4 * PARALLEL_MIN_LEN`); see that
//! constant's doc comment for the threshold-selection measurements.
//!
//! # `sum_axis1_2d`: the `Some(axis)` stride hoist, not kernel dispatch
//!
//! `bench_sum_axis` below measures a different item-4 change: `math::aggregation::sum`'s
//! `Some(axis)` branch used to compute a `strides` array and never use it, reading every
//! element through `array.get(&indices)` instead. `Array::get` (`src/indexing/core.rs`)
//! re-derives `self.shape()` -- which allocates a fresh `Vec<usize>` -- inside its own
//! per-dimension bounds-check loop on *every* call, so the old path paid `O(ndim)` allocations
//! per element, `O(ndim * rows * cols)` overall for a 2-D reduction. The hoisted version takes
//! one `kernels::borrow::operand` slice and one `strides` computation per call, then walks a
//! flat index by a constant `axis_stride` per step -- the same shape already used by
//! `cumsum_no_out`/`cumprod_no_out`'s axis branches. `before::sum_axis1_f64` reimplements the
//! exact old code (not obtained by checking out history, for the same concurrent-lanes reason
//! as above).
//!
//! Run: `cargo bench --bench reduction_dispatch_benchmark`
//! (add `-- --sample-size 10 --warm-up-time 0.3 --measurement-time 0.5` for a quicker
//! before/after read, which is how the numbers in this lane's report were captured).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::math;
use std::hint::black_box;

const SIZES: [usize; 4] = [64, 1_000, 10_000, 1_000_000];

/// `SIZES` plus 80,000 -- `kernels::reduce::MINMAX_PARALLEL_MIN_LEN`, where `min`/`max` switch
/// to their parallel tier. Used only by the `min`/`max` groups, which are the only ones tiered
/// on that constant rather than on `PARALLEL_MIN_LEN`.
const MINMAX_SIZES: [usize; 5] = [64, 1_000, 10_000, 80_000, 1_000_000];

fn f64_arr(n: usize) -> Array<f64> {
    Array::from_vec(
        (0..n)
            .map(|i| ((i.wrapping_mul(2_654_435_761)) % 100_000) as f64 * 0.001 - 50.0)
            .collect(),
    )
}

fn f32_arr(n: usize) -> Array<f32> {
    Array::from_vec(
        (0..n)
            .map(|i| ((i.wrapping_mul(2_654_435_761)) % 100_000) as f32 * 0.001 - 50.0)
            .collect(),
    )
}

/// Faithfully mirrors the pre-conversion function body's input (`array.to_vec()`) and output
/// (`Array::from_vec(vec![result])`, unwrapped by the caller via `.to_vec()[0]` exactly as
/// `after_*` unwraps `math::*`'s real `Result<Array<T>>`) shape, differing only in whichever
/// one thing this lane's item actually changed for that operation. Intentionally does NOT
/// reproduce the population/sample `simd_variance`/`simd_std` bug item 1 of this lane fixed
/// (see `kernels::reduce`'s "never use them" module docs) -- that was a correctness defect,
/// not a performance baseline worth measuring, and population-variance-via-fold (mirrored in
/// `before_var`/`before_std` below) was already this crate's own correct small-`n` behavior.
mod before {
    use super::Array;

    pub fn sum_f64(arr: &Array<f64>) -> f64 {
        let data = arr.to_vec();
        let v = data.iter().fold(0.0_f64, |acc, &x| acc + x);
        Array::from_vec(vec![v]).to_vec()[0]
    }
    pub fn sum_f32(arr: &Array<f32>) -> f32 {
        let data = arr.to_vec();
        let v = data.iter().fold(0.0_f32, |acc, &x| acc + x);
        Array::from_vec(vec![v]).to_vec()[0]
    }

    pub fn mean_f64(arr: &Array<f64>) -> f64 {
        let data = arr.to_vec();
        let sum = data.iter().fold(0.0_f64, |acc, &x| acc + x);
        let v = sum / data.len() as f64;
        Array::from_vec(vec![v]).to_vec()[0]
    }
    pub fn mean_f32(arr: &Array<f32>) -> f32 {
        let data = arr.to_vec();
        let sum = data.iter().fold(0.0_f32, |acc, &x| acc + x);
        let v = sum / data.len() as f32;
        Array::from_vec(vec![v]).to_vec()[0]
    }

    pub fn var_f64(arr: &Array<f64>, ddof: usize) -> f64 {
        let data = arr.to_vec();
        let n = data.len();
        let mean = data.iter().fold(0.0_f64, |acc, &x| acc + x) / n as f64;
        let ssd: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
        let v = ssd / (n - ddof) as f64;
        Array::from_vec(vec![v]).to_vec()[0]
    }
    pub fn var_f32(arr: &Array<f32>, ddof: usize) -> f32 {
        let data = arr.to_vec();
        let n = data.len();
        let mean = data.iter().fold(0.0_f32, |acc, &x| acc + x) / n as f32;
        let ssd: f32 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
        let v = ssd / (n - ddof) as f32;
        Array::from_vec(vec![v]).to_vec()[0]
    }

    pub fn std_f64(arr: &Array<f64>, ddof: usize) -> f64 {
        var_f64(arr, ddof).sqrt()
    }
    pub fn std_f32(arr: &Array<f32>, ddof: usize) -> f32 {
        var_f32(arr, ddof).sqrt()
    }

    pub fn min_f64(arr: &Array<f64>) -> f64 {
        let data = arr.to_vec();
        let v = data
            .iter()
            .skip(1)
            .fold(data[0], |m, &x| if x < m { x } else { m });
        Array::from_vec(vec![v]).to_vec()[0]
    }
    pub fn max_f64(arr: &Array<f64>) -> f64 {
        let data = arr.to_vec();
        let v = data
            .iter()
            .skip(1)
            .fold(data[0], |m, &x| if x > m { x } else { m });
        Array::from_vec(vec![v]).to_vec()[0]
    }
    pub fn min_f32(arr: &Array<f32>) -> f32 {
        let data = arr.to_vec();
        let v = data
            .iter()
            .skip(1)
            .fold(data[0], |m, &x| if x < m { x } else { m });
        Array::from_vec(vec![v]).to_vec()[0]
    }
    pub fn max_f32(arr: &Array<f32>) -> f32 {
        let data = arr.to_vec();
        let v = data
            .iter()
            .skip(1)
            .fold(data[0], |m, &x| if x > m { x } else { m });
        Array::from_vec(vec![v]).to_vec()[0]
    }

    /// Mirrors `math::aggregation::sum`'s pre-hoist `Some(axis)` branch exactly: strides were
    /// computed but never used, and every element was read through `Array::get(&indices)`
    /// inside the `axis_size` inner loop -- which itself re-derives `self.shape()` (a fresh
    /// `Vec` allocation) on every one of its own per-dimension bounds checks, so this pays
    /// `O(ndim)` allocations per element, `O(ndim * rows * cols)` overall. Not a hypothetical:
    /// this was the actual code in this file before the axis-branch hoist (see git history for
    /// this bench, and `math::aggregation::sum`'s doc comment for the hoisted replacement).
    pub fn sum_axis1_f64(arr: &Array<f64>, rows: usize, cols: usize) -> Vec<f64> {
        let mut result = vec![0.0f64; rows];
        for (r, out) in result.iter_mut().enumerate() {
            let mut indices = vec![0usize; 2];
            indices[0] = r;
            let mut sum = 0.0f64;
            for c in 0..cols {
                indices[1] = c;
                sum += arr.get(&indices).expect("index should be in bounds");
            }
            *out = sum;
        }
        result
    }
}

/// Today's public API, `axis = None`, unwrapped the same way `before::*` unwraps its own
/// `Array::from_vec(vec![result])`.
mod after {
    use super::{math, Array};

    pub fn sum_f64(arr: &Array<f64>) -> f64 {
        math::sum(arr, None, false)
            .expect("sum should succeed")
            .to_vec()[0]
    }
    pub fn sum_f32(arr: &Array<f32>) -> f32 {
        math::sum(arr, None, false)
            .expect("sum should succeed")
            .to_vec()[0]
    }
    pub fn mean_f64(arr: &Array<f64>) -> f64 {
        math::mean(arr, None, false)
            .expect("mean should succeed")
            .to_vec()[0]
    }
    pub fn mean_f32(arr: &Array<f32>) -> f32 {
        math::mean(arr, None, false)
            .expect("mean should succeed")
            .to_vec()[0]
    }
    pub fn var_f64(arr: &Array<f64>, ddof: usize) -> f64 {
        math::var(arr, None, ddof, false)
            .expect("var should succeed")
            .to_vec()[0]
    }
    pub fn var_f32(arr: &Array<f32>, ddof: usize) -> f32 {
        math::var(arr, None, ddof, false)
            .expect("var should succeed")
            .to_vec()[0]
    }
    pub fn std_f64(arr: &Array<f64>, ddof: usize) -> f64 {
        math::std(arr, None, ddof, false)
            .expect("std should succeed")
            .to_vec()[0]
    }
    pub fn std_f32(arr: &Array<f32>, ddof: usize) -> f32 {
        math::std(arr, None, ddof, false)
            .expect("std should succeed")
            .to_vec()[0]
    }
    pub fn min_f64(arr: &Array<f64>) -> f64 {
        math::min(arr, None, false)
            .expect("min should succeed")
            .to_vec()[0]
    }
    pub fn max_f64(arr: &Array<f64>) -> f64 {
        math::max(arr, None, false)
            .expect("max should succeed")
            .to_vec()[0]
    }
    pub fn min_f32(arr: &Array<f32>) -> f32 {
        math::min(arr, None, false)
            .expect("min should succeed")
            .to_vec()[0]
    }
    pub fn max_f32(arr: &Array<f32>) -> f32 {
        math::max(arr, None, false)
            .expect("max should succeed")
            .to_vec()[0]
    }

    /// Today's `math::aggregation::sum`, `Some(axis)`: the stride-hoisted, `operand()`-backed
    /// replacement for `before::sum_axis1_f64` above.
    pub fn sum_axis1_f64(arr: &Array<f64>) -> Vec<f64> {
        math::sum(arr, Some(1), false)
            .expect("sum should succeed")
            .to_vec()
    }
}

// ---------------------------------------------------------------------------------------
// Benchmark groups: one per operation, each covering f64/f32 x every size in `SIZES`.
// ---------------------------------------------------------------------------------------

macro_rules! bench_pair {
    ($group:expr, $n:expr, $label64:literal, $label32:literal, $arr64:expr, $arr32:expr, $before64:expr, $after64:expr, $before32:expr, $after32:expr) => {
        $group.bench_with_input(
            BenchmarkId::new(concat!("before_", $label64), $n),
            &$arr64,
            |b, a| b.iter(|| black_box($before64(black_box(a)))),
        );
        $group.bench_with_input(
            BenchmarkId::new(concat!("after_", $label64), $n),
            &$arr64,
            |b, a| b.iter(|| black_box($after64(black_box(a)))),
        );
        $group.bench_with_input(
            BenchmarkId::new(concat!("before_", $label32), $n),
            &$arr32,
            |b, a| b.iter(|| black_box($before32(black_box(a)))),
        );
        $group.bench_with_input(
            BenchmarkId::new(concat!("after_", $label32), $n),
            &$arr32,
            |b, a| b.iter(|| black_box($after32(black_box(a)))),
        );
    };
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/sum");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        let a32 = f32_arr(n);
        bench_pair!(
            group,
            n,
            "f64",
            "f32",
            a64,
            a32,
            before::sum_f64,
            after::sum_f64,
            before::sum_f32,
            after::sum_f32
        );
    }
    group.finish();
}

fn bench_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/mean");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        let a32 = f32_arr(n);
        bench_pair!(
            group,
            n,
            "f64",
            "f32",
            a64,
            a32,
            before::mean_f64,
            after::mean_f64,
            before::mean_f32,
            after::mean_f32
        );
    }
    group.finish();
}

fn bench_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/var");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        let a32 = f32_arr(n);
        group.bench_with_input(BenchmarkId::new("before_f64", n), &a64, |b, a| {
            b.iter(|| black_box(before::var_f64(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("after_f64", n), &a64, |b, a| {
            b.iter(|| black_box(after::var_f64(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("before_f32", n), &a32, |b, a| {
            b.iter(|| black_box(before::var_f32(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("after_f32", n), &a32, |b, a| {
            b.iter(|| black_box(after::var_f32(black_box(a), 0)))
        });
    }
    group.finish();
}

/// The regression that motivated `kernels::reduce`'s fused variance kernels, measured where it
/// actually lived: right at `PARALLEL_MIN_LEN`. `9_999` (one below the old per-pass threshold,
/// so both passes were already serial and no regression was possible), `10_000` (the regressing
/// size: both passes went parallel independently), `20_000` (still on the fused kernel's serial
/// tier) and `40_000` (`VAR_PARALLEL_MIN_LEN`, where the fused kernel itself switches to the
/// parallel tier). `before_f64` is the same pre-kernel sequential two-pass body the other groups
/// use; `after_f64` >= 0.95x of it at `n = 10_000` is the acceptance bar.
fn bench_var_10k_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/var_10k_threshold");
    for &n in &[9_999usize, 10_000, 20_000, 40_000] {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        group.bench_with_input(BenchmarkId::new("before_f64", n), &a64, |b, a| {
            b.iter(|| black_box(before::var_f64(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("after_f64", n), &a64, |b, a| {
            b.iter(|| black_box(after::var_f64(black_box(a), 0)))
        });
    }
    group.finish();
}

fn bench_std(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/std");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        let a32 = f32_arr(n);
        group.bench_with_input(BenchmarkId::new("before_f64", n), &a64, |b, a| {
            b.iter(|| black_box(before::std_f64(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("after_f64", n), &a64, |b, a| {
            b.iter(|| black_box(after::std_f64(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("before_f32", n), &a32, |b, a| {
            b.iter(|| black_box(before::std_f32(black_box(a), 0)))
        });
        group.bench_with_input(BenchmarkId::new("after_f32", n), &a32, |b, a| {
            b.iter(|| black_box(after::std_f32(black_box(a), 0)))
        });
    }
    group.finish();
}

/// `MINMAX_SIZES` rather than `SIZES`: it adds 80,000, the size
/// `kernels::reduce::MINMAX_PARALLEL_MIN_LEN` sits at, so the group covers both sides of the
/// tier boundary `min`/`max` actually switch on rather than only `SIZES`' 10,000 and
/// 1,000,000, which straddle it too widely to show where the crossover is.
fn bench_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/min");
    for &n in &MINMAX_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        let a32 = f32_arr(n);
        bench_pair!(
            group,
            n,
            "f64",
            "f32",
            a64,
            a32,
            before::min_f64,
            after::min_f64,
            before::min_f32,
            after::min_f32
        );
    }
    group.finish();
}

/// See [`bench_min`] for why this group uses `MINMAX_SIZES`.
fn bench_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/max");
    for &n in &MINMAX_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n);
        let a32 = f32_arr(n);
        bench_pair!(
            group,
            n,
            "f64",
            "f32",
            a64,
            a32,
            before::max_f64,
            after::max_f64,
            before::max_f32,
            after::max_f32
        );
    }
    group.finish();
}

/// `Some(axis)` path, not `axis = None`: this is the axis-branch stride hoist (item 4 of this
/// lane), not the kernel-dispatch conversion the groups above measure. Fixed `rows = 8`,
/// `cols = n / 8` for each `n` in [`SIZES`] (all exact divisions), reducing along `axis = 1`
/// (the contiguous, unit-stride axis -- the common "row sum" shape).
fn bench_sum_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_dispatch/sum_axis1_2d");
    let rows = 8usize;
    for &n in &SIZES {
        let cols = n / rows;
        group.throughput(Throughput::Elements(n as u64));
        let a64 = f64_arr(n).reshape(&[rows, cols]);
        group.bench_with_input(BenchmarkId::new("before_f64", n), &a64, |b, a| {
            b.iter(|| black_box(before::sum_axis1_f64(black_box(a), rows, cols)))
        });
        group.bench_with_input(BenchmarkId::new("after_f64", n), &a64, |b, a| {
            b.iter(|| black_box(after::sum_axis1_f64(black_box(a))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sum,
    bench_mean,
    bench_var,
    bench_var_10k_threshold,
    bench_std,
    bench_min,
    bench_max,
    bench_sum_axis
);
criterion_main!(benches);
