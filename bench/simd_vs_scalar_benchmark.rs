//! SIMD vs Scalar Performance Benchmark
//!
//! This benchmark suite compares SIMD-optimized operations against scalar implementations
//! to demonstrate the performance gains from vectorization. It covers:
//!
//! - Mathematical operations (exp, log, sin, cos, sqrt)
//! - Array operations (sum, product, mean)
//! - Element-wise operations (add, multiply, abs)
//! - Transcendental functions (trig, hyperbolic)
//!
//! Results help validate our SIMD optimizations and identify areas for improvement.

#![allow(deprecated)]
#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::prelude::*;
use numrs2::ufuncs;
use std::hint::black_box;

// =============================================================================
// CONFIGURATION
// =============================================================================

const SMALL_SIZE: usize = 64; // Below SIMD threshold
const MEDIUM_SIZE: usize = 512; // SIMD beneficial
const LARGE_SIZE: usize = 4096; // Maximum SIMD benefit
const HUGE_SIZE: usize = 32768; // Cache effects

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate random f64 array for testing
fn random_f64_array(size: usize) -> Array<f64> {
    let rng = random::default_rng();
    rng.uniform(-10.0, 10.0, &[size]).unwrap()
}

/// Generate random positive f64 array for log/sqrt testing
fn random_positive_f64_array(size: usize) -> Array<f64> {
    let rng = random::default_rng();
    rng.uniform(0.1, 100.0, &[size]).unwrap()
}

/// Scalar implementation of exp
fn scalar_exp(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.exp()).collect();
    Array::from_vec(result)
}

/// Scalar implementation of sin
fn scalar_sin(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.sin()).collect();
    Array::from_vec(result)
}

/// Scalar implementation of cos
fn scalar_cos(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.cos()).collect();
    Array::from_vec(result)
}

/// Scalar implementation of log
fn scalar_log(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.ln()).collect();
    Array::from_vec(result)
}

/// Scalar implementation of sqrt
fn scalar_sqrt(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.sqrt()).collect();
    Array::from_vec(result)
}

/// Scalar implementation of abs
fn scalar_abs(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
    Array::from_vec(result)
}

/// Scalar implementation of sum
fn scalar_sum(arr: &Array<f64>) -> f64 {
    arr.to_vec().iter().sum()
}

/// Scalar implementation of mean
fn scalar_mean(arr: &Array<f64>) -> f64 {
    scalar_sum(arr) / (arr.len() as f64)
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

/// Benchmark exponential function (SIMD vs Scalar)
fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("Exponential");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_f64_array(size);

        // SIMD implementation (uses avx2_enhanced if available)
        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(ufuncs::exp(data));
                black_box(result)
            });
        });

        // Scalar implementation for comparison
        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_exp(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark logarithm function (SIMD vs Scalar)
fn bench_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("Logarithm");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_positive_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(ufuncs::log(data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_log(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark sine function (SIMD vs Scalar)
fn bench_sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sine");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(ufuncs::sin(data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_sin(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark cosine function (SIMD vs Scalar)
fn bench_cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(ufuncs::cos(data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_cos(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark square root (SIMD vs Scalar)
fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("SquareRoot");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_positive_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(ufuncs::sqrt(data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_sqrt(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark absolute value (SIMD vs Scalar)
fn bench_abs(c: &mut Criterion) {
    let mut group = c.benchmark_group("AbsoluteValue");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(ufuncs::absolute(data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_abs(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark sum reduction (SIMD vs Scalar)
fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sum");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(data.sum());
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_sum(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark mean calculation (SIMD vs Scalar)
fn bench_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mean");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let data = random_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| {
                let result = black_box(data.mean());
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| {
                let result = scalar_mean(black_box(data));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise addition (SIMD vs Scalar)
fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Addition");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let a = random_f64_array(size);
        let b = random_f64_array(size);

        group.bench_with_input(
            BenchmarkId::new("SIMD", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = black_box(a.add(b));
                    black_box(result)
                });
            },
        );

        // Scalar implementation
        group.bench_with_input(
            BenchmarkId::new("Scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let a_data = a.to_vec();
                    let b_data = b.to_vec();
                    let result: Vec<f64> = a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(&x, &y)| x + y)
                        .collect();
                    black_box(Array::from_vec(result))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark element-wise multiplication (SIMD vs Scalar)
fn bench_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multiplication");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));

        let a = random_f64_array(size);
        let b = random_f64_array(size);

        group.bench_with_input(
            BenchmarkId::new("SIMD", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = black_box(a.multiply(b));
                    black_box(result)
                });
            },
        );

        // Scalar implementation
        group.bench_with_input(
            BenchmarkId::new("Scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let a_data = a.to_vec();
                    let b_data = b.to_vec();
                    let result: Vec<f64> = a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(&x, &y)| x * y)
                        .collect();
                    black_box(Array::from_vec(result))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CRITERION SETUP
// =============================================================================

criterion_group!(
    benches,
    bench_exp,
    bench_log,
    bench_sin,
    bench_cos,
    bench_sqrt,
    bench_abs,
    bench_sum,
    bench_mean,
    bench_add,
    bench_multiply
);

criterion_main!(benches);
