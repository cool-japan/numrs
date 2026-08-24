//! Criterion harness for the hot paths Lane W2-B (operators & comparisons)
//! migrated onto `src/kernels/elementwise.rs`'s dtype-dispatched
//! `binary_serial`/`binary_dispatch` kernels.
//!
//! `kernels` itself is `pub(crate)` and not reachable from this external
//! bench crate, so every group here measures through the *public* entry
//! points that route into it: [`numrs2::array::Array::add_broadcast`] /
//! `multiply_broadcast` (item 1, `array/operations.rs`) and
//! [`numrs2::comparisons::greater`] (item 3, `comparisons.rs`). "old"
//! variants either call the still-unchanged inherent `Array::add`/
//! `multiply` (plain `&ndarray::ArrayBase<_, IxDyn> + ..`, untouched by
//! this migration -- deliberately called via `Array::add(&a, &b)` path
//! syntax rather than `a.add(&b)` method-call syntax, and with
//! `std::ops::Add`/`Mul` never imported into this file, so method
//! resolution cannot instead pick the `Add`/`Mul` *trait* impls that now
//! delegate to `add_broadcast`/`multiply_broadcast`) or, for `greater`,
//! reproduce its pre-migration body verbatim from public API
//! (`broadcast_to`/`to_vec`/`clone`), since the real old body no longer
//! exists once `comparisons.rs` was migrated.
//!
//! Run with: `cargo bench --bench elementwise_dispatch_benchmark`
//! (add `-- --sample-size 10` for a quicker before/after read).
//!
//! # G4 gate-fix note: `equal_shape/add_*` at n=1e6, `--sample-size 10`
//!
//! A single `--sample-size 10` run of this exact `equal_shape` group's
//! `add_old_ndarray`/`add_new_broadcast` pair at n=1e6 flagged
//! `add_new_broadcast` as ~8.9% slower with non-overlapping 95% CIs
//! (227.01 µs [221.95, 232.12] vs. 247.13 µs [239.06, 258.07]; see the
//! Wave 2 gate log). Two findings from following up, and they don't
//! contradict each other:
//!
//! 1. That **8.9% magnitude does not reproduce**. 4 independent
//!    `--release` runs of a purpose-built alternating-A/B
//!    min-of-many-rounds probe (`array::operations::perf_probe::
//!    probe_add_broadcast_large_n_min_estimator` -- see that function's
//!    doc comment for the full methodology and per-run table), run
//!    minutes apart under this session's own heavy concurrent-agent
//!    build load (`uptime` load averages ranged ~10-40 across the runs),
//!    put `add_broadcast`/`Array::add`'s gap at n=1e6 at {1.006, 1.049,
//!    1.014, 1.009}x -- mean ~1.9% slower, never above ~4.9% in any
//!    single run. Non-overlapping criterion CIs from one `--sample-size
//!    10` run describe sampling variance *within* that run, not
//!    run-to-run variance from external load -- on a machine shared with
//!    other concurrent agents that CI does not mean what it would on a
//!    quiet machine, which is the shape of what happened here.
//! 2. A **smaller one-sided effect is real**, though: across 16 (n, run)
//!    combinations from 1e4 to 4e6, `add_broadcast` was never once
//!    measurably *faster* than `Array::add` at these sizes (unlike the
//!    small-`n` numbers in the `equal_shape` group below, where the
//!    migration is a clean win) -- but it traces to `add_broadcast`'s
//!    thin wrapping around the kernel (`broadcast_op`'s generic closure
//!    dispatch, `Array::from_vec_shape`), not to the kernel
//!    (`binary_serial`) itself: isolated from the `Array` wrap, the
//!    kernel tracks the old baseline in *both* directions across runs at
//!    every size, including 1e6.
//!
//! No dispatch code changed as a result of either finding -- finding 1
//! means there is no 8.9%-sized regression to fix, and finding 2's real
//! but small effect is outside `binary_serial` (this item's scope) and
//! too small on its own to justify restructuring `add_broadcast`. See the
//! probe's doc comment for the full root-cause elimination (mechanisms
//! (a)/(b)/(c) from the gate-fix brief, all checked and ruled out) and the
//! acceptance-criteria reading against this data.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::comparisons::greater;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::hint::black_box;

const SIZES: [usize; 5] = [64, 1_000, 10_000, 100_000, 1_000_000];

fn make_f64(n: usize, scale: f64, offset: f64) -> Array<f64> {
    Array::from_vec((0..n).map(|i| i as f64 * scale + offset).collect())
}

/// `add`/`multiply` on two *equal-shape* f64 arrays: old (`Array::add`/
/// `multiply`, unchanged, ndarray's own `IxDyn` operator) vs. new
/// (`Array::add_broadcast`/`multiply_broadcast`,
/// `kernels::elementwise::binary_serial` over a flat contiguous slice via
/// `kernels::borrow::operand`). `broadcast_op`'s equal-shape fast path
/// means neither side pays for `broadcast_to` here -- this isolates the
/// closure-body change alone.
fn bench_add_mul_equal_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_dispatch/equal_shape");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a = make_f64(n, 1.0, 0.0);
        let b = make_f64(n, 0.5, 1.0);

        group.bench_with_input(BenchmarkId::new("add_old_ndarray", n), &n, |bch, _| {
            bch.iter(|| black_box(Array::add(&a, &b)))
        });
        group.bench_with_input(BenchmarkId::new("add_new_broadcast", n), &n, |bch, _| {
            bch.iter(|| black_box(a.add_broadcast(&b).expect("equal shapes never fail")))
        });
        group.bench_with_input(BenchmarkId::new("mul_old_ndarray", n), &n, |bch, _| {
            bch.iter(|| black_box(Array::multiply(&a, &b)))
        });
        group.bench_with_input(BenchmarkId::new("mul_new_broadcast", n), &n, |bch, _| {
            bch.iter(|| black_box(a.multiply_broadcast(&b).expect("equal shapes never fail")))
        });
    }
    group.finish();
}

/// Same two ops, but `b` has a genuinely different, broadcastable shape
/// (a single element broadcasting up to `n`) -- exercises
/// `broadcast_op`'s `broadcast_to` + closure path on both operands.
/// ndarray's own `Add`/`Mul` impls also broadcast internally (via
/// `broadcast_with`, see `impl_ops.rs`), so `Array::add`/`multiply` are a
/// meaningful "old" baseline here too, not just in the equal-shape group.
fn bench_add_mul_broadcast_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_dispatch/broadcast_shape");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a = make_f64(n, 1.0, 0.0);
        let b = make_f64(1, 1.0, 3.0); // shape [1], broadcasts to [n]

        group.bench_with_input(BenchmarkId::new("add_old_ndarray", n), &n, |bch, _| {
            bch.iter(|| black_box(Array::add(&a, &b)))
        });
        group.bench_with_input(BenchmarkId::new("add_new_broadcast", n), &n, |bch, _| {
            bch.iter(|| black_box(a.add_broadcast(&b).expect("[n] and [1] always broadcast")))
        });
        group.bench_with_input(BenchmarkId::new("mul_old_ndarray", n), &n, |bch, _| {
            bch.iter(|| black_box(Array::multiply(&a, &b)))
        });
        group.bench_with_input(BenchmarkId::new("mul_new_broadcast", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(
                    a.multiply_broadcast(&b)
                        .expect("[n] and [1] always broadcast"),
                )
            })
        });
    }
    group.finish();
}

/// A/B: the flat zip-loop `add_broadcast`/`binary_serial` uses vs. calling
/// `scirs2_core`'s `f64::simd_add` (`SimdUnifiedOps`) directly on the same
/// two contiguous slices. `simd_add` returns a freshly allocated
/// `Array1<f64>` of its own -- the same single output allocation the zip
/// loop's `.collect()` already pays -- so this isolates "hand-rolled,
/// autovectorizable scalar loop" vs. "explicit SIMD intrinsics", not an
/// allocation-count difference. See `kernels::elementwise::binary_dispatch`
/// doc comment for the full write-up and numbers from this group.
fn bench_zip_loop_vs_simd_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_dispatch/zip_loop_vs_simd_add");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a = make_f64(n, 1.0, 0.0);
        let b = make_f64(n, 0.5, 1.0);
        let a_slice = a.as_slice().expect("from_vec is always contiguous");
        let b_slice = b.as_slice().expect("from_vec is always contiguous");

        group.bench_with_input(BenchmarkId::new("zip_loop", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(
                    a_slice
                        .iter()
                        .zip(b_slice.iter())
                        .map(|(&x, &y)| x + y)
                        .collect::<Vec<f64>>(),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("simd_add", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(f64::simd_add(
                    &ArrayView1::from(a_slice),
                    &ArrayView1::from(b_slice),
                ))
            })
        });
    }
    group.finish();
}

/// Faithful reproduction of `comparisons::greater`'s pre-migration body
/// (`a.clone()`/`b.clone()` in the equal-shape case, `to_vec()` on both
/// operands, zip+map+collect, `from_vec`+`reshape`) built here from public
/// API only, since the real old body no longer exists once
/// `comparisons.rs` was migrated onto `maybe_broadcast`/`operand`/
/// `binary_serial`.
fn greater_old_five_copies(a: &Array<f64>, b: &Array<f64>) -> Array<bool> {
    let broadcast_shape =
        Array::<f64>::broadcast_shape(&a.shape(), &b.shape()).expect("bench inputs broadcast");
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)
            .expect("bench inputs broadcast")
    } else {
        a.clone()
    };
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)
            .expect("bench inputs broadcast")
    } else {
        b.clone()
    };
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    let result: Vec<bool> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x > y)
        .collect();
    Array::from_vec(result).reshape(&broadcast_shape)
}

/// `comparisons::greater` at the lane's acceptance point (n=1e5,
/// equal-shape): old (`a.clone()`+`b.clone()`+`to_vec()`x2+`reshape` = 5
/// full copies) vs. new (`maybe_broadcast` `Cow::Borrowed` + zero-copy
/// `operand` + one zip pass).
fn bench_greater(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_dispatch/comparisons_greater");
    let n = 100_000usize;
    group.throughput(Throughput::Elements(n as u64));
    let a = make_f64(n, 1.0, 0.0);
    let b = make_f64(n, 1.0, -1.0);

    group.bench_function(BenchmarkId::new("greater_old_5_copies", n), |bch| {
        bch.iter(|| black_box(greater_old_five_copies(&a, &b)))
    });
    group.bench_function(BenchmarkId::new("greater_new_cow_operand", n), |bch| {
        bch.iter(|| black_box(greater(&a, &b).expect("equal shapes never fail")))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_add_mul_equal_shape,
    bench_add_mul_broadcast_shape,
    bench_zip_loop_vs_simd_add,
    bench_greater,
);
criterion_main!(benches);
