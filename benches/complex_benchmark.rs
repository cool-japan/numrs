//! Complex Number Operations Benchmarks
//!
//! Benchmarks for complex array construction, unary operations,
//! polar conversion, and throughput comparison with real arithmetic.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::complex_ops::{absolute, angle, conj, from_polar, imag, real, to_complex};
use numrs2::prelude::*;
use scirs2_core::Complex;
use std::hint::black_box;

/// Benchmark conversion of f64 arrays to complex arrays
fn bench_complex_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_construction");

    for &n in [1_000usize, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(n as u64));

        let rng = random::default_rng();
        if let Ok(real_arr) = rng.random::<f64>(&[n]) {
            group.bench_with_input(BenchmarkId::new("to_complex", n), &n, |bencher, _| {
                bencher.iter(|| {
                    black_box(to_complex(&real_arr));
                });
            });
        }
    }

    group.finish();
}

/// Benchmark unary operations on complex arrays
fn bench_complex_unary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_unary_ops");

    for &n in [10_000usize, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(n as u64));

        let rng = random::default_rng();
        if let Ok(real_arr) = rng.random::<f64>(&[n]) {
            let complex_arr = to_complex(&real_arr);

            group.bench_with_input(BenchmarkId::new("conj", n), &n, |bencher, _| {
                bencher.iter(|| {
                    black_box(conj(&complex_arr));
                });
            });

            group.bench_with_input(BenchmarkId::new("absolute", n), &n, |bencher, _| {
                bencher.iter(|| {
                    black_box(absolute(&complex_arr));
                });
            });

            group.bench_with_input(BenchmarkId::new("angle_radians", n), &n, |bencher, _| {
                bencher.iter(|| {
                    black_box(angle(&complex_arr, false));
                });
            });

            group.bench_with_input(BenchmarkId::new("real", n), &n, |bencher, _| {
                bencher.iter(|| {
                    black_box(real(&complex_arr));
                });
            });

            group.bench_with_input(BenchmarkId::new("imag", n), &n, |bencher, _| {
                bencher.iter(|| {
                    black_box(imag(&complex_arr));
                });
            });
        }
    }

    group.finish();
}

/// Benchmark polar-to-complex conversion
fn bench_from_polar(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_polar");

    for &n in [10_000usize, 100_000].iter() {
        group.throughput(Throughput::Elements(n as u64));

        let rng = random::default_rng();
        if let (Ok(r_arr), Ok(theta_arr)) = (rng.random::<f64>(&[n]), rng.random::<f64>(&[n])) {
            group.bench_with_input(BenchmarkId::new("from_polar", n), &n, |bencher, _| {
                bencher.iter(|| {
                    if let Ok(result) = from_polar(&r_arr, &theta_arr, false) {
                        black_box(result);
                    }
                });
            });
        }
    }

    group.finish();
}

/// Throughput comparison: real f64 element-wise add vs complex element-wise add
///
/// Both operate on 1_000_000-element arrays to measure the cost of complex arithmetic
/// relative to real arithmetic.
fn bench_complex_vs_real_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_vs_real_throughput");
    let n = 1_000_000usize;
    group.throughput(Throughput::Elements(n as u64));

    let rng = random::default_rng();
    if let (Ok(a_real), Ok(b_real)) = (rng.random::<f64>(&[n]), rng.random::<f64>(&[n])) {
        // Real f64 element-wise addition
        group.bench_function("real_add_f64", |bencher| {
            bencher.iter(|| {
                black_box(a_real.add(&b_real));
            });
        });

        // Complex element-wise addition (Array<Complex<f64>>.add is non-Result)
        let a_complex = to_complex(&a_real);
        let b_complex = to_complex(&b_real);
        group.bench_function("complex_add", |bencher| {
            bencher.iter(|| {
                black_box(a_complex.add(&b_complex));
            });
        });
    }

    // Suppress unused-import warning for Complex (used for type annotation in doc)
    let _ = Complex::<f64>::new(0.0, 0.0);

    group.finish();
}

criterion_group!(
    complex_benches,
    bench_complex_construction,
    bench_complex_unary_ops,
    bench_from_polar,
    bench_complex_vs_real_throughput,
);
criterion_main!(complex_benches);
