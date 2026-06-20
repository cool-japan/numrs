//! Allocation hot-path benchmark for NumRS2 (Tier C remediation)
//!
//! Targets the code paths that previously cloned an entire array via
//! `to_vec()` purely to read it. These benchmarks let us measure the
//! before/after effect of routing read-only access through the zero-copy
//! `Array::array().iter()` / `Array::as_slice()` accessors instead.
//!
//! Covered paths (each previously cloned its input(s) via `to_vec()`):
//! - `Array::map` — operations.rs; previously double-cloned (whole array + each element).
//! - `Array::zip_with` — operations.rs; cloned both operands.
//! - `Array::sum` — operations.rs; cloned in both the SIMD and the scalar fallback.
//! - `ufuncs::hypot` — ufuncs.rs; cloned both operands on the sub-`SIMD_THRESHOLD` fallback.
//! - `Array::simd_add` — simd.rs; cloned both operands via `to_ndarray_1d` on the SIMD path.
//!
//! Run with: `cargo bench --bench allocation_hotpath_benchmark`

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::simd::SimdOps;
use std::hint::black_box;

/// Sizes chosen to exercise both branches of the `SIMD_THRESHOLD == 64`
/// gate: 32 hits the scalar `to_vec()` fallback, the larger sizes hit the
/// SIMD path (which itself used to clone via `to_ndarray_1d`).
const SIZES: [usize; 4] = [32, 1_024, 16_384, 262_144];

fn make(n: usize) -> Array<f64> {
    Array::from_vec((0..n).map(|i| (i as f64) * 0.5 + 1.0).collect())
}

fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("hotpath/map");
    for &n in SIZES.iter() {
        let a = make(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &a, |b, a| {
            b.iter(|| black_box(a.map(|x| x * 2.0 + 1.0)))
        });
    }
    group.finish();
}

fn bench_zip_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("hotpath/zip_with");
    for &n in SIZES.iter() {
        let a = make(n);
        let b_arr = make(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(a.zip_with(&b_arr, |x, y| x * y).unwrap()))
        });
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("hotpath/sum");
    for &n in SIZES.iter() {
        let a = make(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &a, |b, a| {
            b.iter(|| black_box(a.sum()))
        });
    }
    group.finish();
}

fn bench_ufunc_hypot(c: &mut Criterion) {
    let mut group = c.benchmark_group("hotpath/ufunc_hypot");
    for &n in SIZES.iter() {
        let a = make(n);
        let b_arr = make(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(numrs2::ufuncs::hypot(&a, &b_arr).unwrap()))
        });
    }
    group.finish();
}

fn bench_simd_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("hotpath/simd_add");
    for &n in SIZES.iter() {
        let a = make(n);
        let b_arr = make(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(a.simd_add(&b_arr).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_map,
    bench_zip_with,
    bench_sum,
    bench_ufunc_hypot,
    bench_simd_add
);
criterion_main!(benches);
