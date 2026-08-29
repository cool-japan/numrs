//! Comprehensive Parallel Computing Benchmarks for NumRS2
//!
//! This benchmark suite tests parallel operations including:
//! - Parallel vs sequential operations
//! - Scaling with thread count
//! - Load balancing efficiency
//! - Parallel reduction operations
//! - Parallel map operations
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::prelude::*;
use std::hint::black_box;

/// Benchmark parallel vs sequential sum
fn bench_parallel_vs_sequential_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential_sum");

    for size in [1000, 10000, 100000, 1000000, 10000000].iter() {
        // Sequential sum
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    let vec = arr.to_vec();
                    let result: f64 = vec.iter().sum();
                    black_box(result);
                });
            }
        });

        // Parallel sum
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel vs sequential matrix multiplication
fn bench_parallel_vs_sequential_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential_matmul");

    for size in [50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s, s]), rng.random::<f64>(&[s, s])) {
                bencher.iter(|| {
                    if let Ok(result) = a.matmul(&b) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel reduction operations
fn bench_parallel_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_reductions");

    for size in [10000, 100000, 1000000, 10000000].iter() {
        // Parallel sum
        group.bench_with_input(BenchmarkId::new("sum", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Parallel product
        group.bench_with_input(BenchmarkId::new("prod", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(0.99, 1.01, &[s]) {
                bencher.iter(|| {
                    if let Ok(result) = prod(&arr, None, false, None) {
                        black_box(result);
                    }
                });
            }
        });

        // Parallel min
        group.bench_with_input(BenchmarkId::new("min", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = min(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Parallel max
        group.bench_with_input(BenchmarkId::new("max", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = max(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel map operations
fn bench_parallel_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_map");

    for size in [10000, 100000, 1000000, 10000000].iter() {
        // Element-wise square
        group.bench_with_input(BenchmarkId::new("square", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    let result = &arr * &arr;
                    black_box(result);
                });
            }
        });

        // Element-wise exponential
        group.bench_with_input(BenchmarkId::new("exp", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(-2.0, 2.0, &[s]) {
                bencher.iter(|| {
                    let result = exp(&arr);
                    black_box(result);
                });
            }
        });

        // Element-wise logarithm
        group.bench_with_input(BenchmarkId::new("log", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(0.1, 10.0, &[s]) {
                bencher.iter(|| {
                    let result = log(&arr);
                    black_box(result);
                });
            }
        });

        // Element-wise sine
        group.bench_with_input(BenchmarkId::new("sin", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(-std::f64::consts::PI, std::f64::consts::PI, &[s]) {
                bencher.iter(|| {
                    let result = numrs2::ufuncs::sin(&arr);
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark thread count scaling
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");

    let size = 1000000;

    // Test with different workloads
    group.bench_function("sum_1M", |bencher| {
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            bencher.iter(|| {
                if let Ok(result) = sum(&arr, None, false) {
                    black_box(result);
                }
            });
        }
    });

    group.bench_function("matmul_500x500", |bencher| {
        let rng = random::default_rng();
        if let (Ok(a), Ok(b)) = (
            rng.random::<f64>(&[500, 500]),
            rng.random::<f64>(&[500, 500]),
        ) {
            bencher.iter(|| {
                if let Ok(result) = a.matmul(&b) {
                    black_box(result);
                }
            });
        }
    });

    group.bench_function("element_wise_ops_1M", |bencher| {
        let rng = random::default_rng();
        if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[size]), rng.random::<f64>(&[size])) {
            bencher.iter(|| {
                let result = &a + &b;
                black_box(result);
            });
        }
    });

    group.finish();
}

/// Benchmark parallel overhead for small vs large arrays
fn bench_parallel_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_overhead");

    // Small array - overhead may dominate
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("sum", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("add", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = a.add(&b);
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel array operations
fn bench_parallel_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_array_operations");

    for size in [10000, 100000, 1000000].iter() {
        // Addition
        group.bench_with_input(BenchmarkId::new("add", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = a.add(&b);
                    black_box(result);
                });
            }
        });

        // Multiplication
        group.bench_with_input(BenchmarkId::new("mul", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a * &b;
                    black_box(result);
                });
            }
        });

        // Division
        group.bench_with_input(BenchmarkId::new("div", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.uniform::<f64>(0.1, 10.0, &[s])) {
                bencher.iter(|| {
                    let result = &a / &b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark load balancing with different workload distributions
fn bench_load_balancing(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_balancing");

    for size in [10000, 100000, 1000000].iter() {
        // Uniform workload
        group.bench_with_input(BenchmarkId::new("uniform", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Reduction which is inherently unbalanced
        group.bench_with_input(BenchmarkId::new("reduction", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = prod(&arr, None, false, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel matrix operations
fn bench_parallel_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_matrix_operations");

    for size in [100, 200, 500, 1000].iter() {
        // Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s, s]), rng.random::<f64>(&[s, s])) {
                bencher.iter(|| {
                    if let Ok(result) = a.matmul(&b) {
                        black_box(result);
                    }
                });
            }
        });

        // Matrix transpose (typically less parallel benefit)
        group.bench_with_input(BenchmarkId::new("transpose", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                bencher.iter(|| {
                    black_box(mat.transpose());
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel statistical operations
fn bench_parallel_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_statistics");

    for size in [10000, 100000, 1000000, 10000000].iter() {
        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = mean(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Variance
        group.bench_with_input(BenchmarkId::new("variance", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = var(&arr, None, 0, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = std(&arr, None, 0, false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark parallel FFT operations
fn bench_parallel_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_fft");

    for size in [1024, 4096, 16384, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("fft", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = FFT::fft(&signal) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("rfft", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = FFT::rfft(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_vs_sequential_sum,
    bench_parallel_vs_sequential_matmul,
    bench_parallel_reductions,
    bench_parallel_map,
    bench_thread_scaling,
    bench_parallel_overhead,
    bench_parallel_array_operations,
    bench_load_balancing,
    bench_parallel_matrix_operations,
    bench_parallel_statistics,
    bench_parallel_fft,
);

criterion_main!(benches);
