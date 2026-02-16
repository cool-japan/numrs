//! Comprehensive SIMD vs Scalar Benchmarks for NumRS2
//!
//! This benchmark suite compares SIMD and scalar performance including:
//! - SIMD vs scalar operations (addition, multiplication, etc.)
//! - AVX2 vs scalar (on x86_64)
//! - NEON vs scalar (on ARM)
//! - Threshold analysis (when SIMD becomes beneficial)
//! - Various data types (f32, f64)
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::prelude::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::hint::black_box;

/// Benchmark SIMD vs scalar addition
fn bench_simd_vs_scalar_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_addition");

    for size in [16, 64, 256, 1024, 4096, 16384, 65536].iter() {
        // Scalar addition
        group.bench_with_input(BenchmarkId::new("scalar_f64", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    // Force scalar path
                    let vec_a = a.to_vec();
                    let vec_b = b.to_vec();
                    let result: Vec<f64> = vec_a
                        .iter()
                        .zip(vec_b.iter())
                        .map(|(&x, &y)| x + y)
                        .collect();
                    black_box(result);
                });
            }
        });

        // SIMD addition (through operator overloading)
        group.bench_with_input(BenchmarkId::new("simd_f64", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a + &b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark SIMD vs scalar multiplication
fn bench_simd_vs_scalar_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_multiplication");

    for size in [16, 64, 256, 1024, 4096, 16384, 65536].iter() {
        // Scalar multiplication
        group.bench_with_input(BenchmarkId::new("scalar_f64", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let vec_a = a.to_vec();
                    let vec_b = b.to_vec();
                    let result: Vec<f64> = vec_a
                        .iter()
                        .zip(vec_b.iter())
                        .map(|(&x, &y)| x * y)
                        .collect();
                    black_box(result);
                });
            }
        });

        // SIMD multiplication
        group.bench_with_input(BenchmarkId::new("simd_f64", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a * &b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark SIMD vs scalar division
fn bench_simd_vs_scalar_division(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_division");

    for size in [16, 64, 256, 1024, 4096, 16384, 65536].iter() {
        // Scalar division
        group.bench_with_input(BenchmarkId::new("scalar_f64", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.uniform::<f64>(0.1, 10.0, &[s])) {
                bencher.iter(|| {
                    let vec_a = a.to_vec();
                    let vec_b = b.to_vec();
                    let result: Vec<f64> = vec_a
                        .iter()
                        .zip(vec_b.iter())
                        .map(|(&x, &y)| x / y)
                        .collect();
                    black_box(result);
                });
            }
        });

        // SIMD division
        group.bench_with_input(BenchmarkId::new("simd_f64", size), size, |bencher, &s| {
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

/// Benchmark SIMD reduction operations
fn bench_simd_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_reductions");

    for size in [16, 64, 256, 1024, 4096, 16384, 65536].iter() {
        // Scalar sum
        group.bench_with_input(BenchmarkId::new("scalar_sum", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    let vec = arr.to_vec();
                    let result: f64 = vec.iter().sum();
                    black_box(result);
                });
            }
        });

        // SIMD sum
        group.bench_with_input(BenchmarkId::new("simd_sum", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Scalar product
        group.bench_with_input(
            BenchmarkId::new("scalar_product", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.uniform::<f64>(0.99, 1.01, &[s]) {
                    bencher.iter(|| {
                        let vec = arr.to_vec();
                        let result: f64 = vec.iter().product();
                        black_box(result);
                    });
                }
            },
        );

        // SIMD product
        group.bench_with_input(
            BenchmarkId::new("simd_product", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.uniform::<f64>(0.99, 1.01, &[s]) {
                    bencher.iter(|| {
                        if let Ok(result) = prod(&arr, None, false, None) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD threshold analysis
fn bench_simd_threshold_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_threshold_analysis");

    // Test very small sizes where SIMD overhead may dominate
    for size in [4, 8, 16, 32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let vec_a = a.to_vec();
                    let vec_b = b.to_vec();
                    let result: Vec<f64> = vec_a
                        .iter()
                        .zip(vec_b.iter())
                        .map(|(&x, &y)| x + y)
                        .collect();
                    black_box(result);
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a + &b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark f32 vs f64 SIMD performance
fn bench_simd_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_data_types");

    for size in [256, 1024, 4096, 16384].iter() {
        // f32 operations
        group.bench_with_input(BenchmarkId::new("f32_add", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f32>(&[s]), rng.random::<f32>(&[s])) {
                bencher.iter(|| {
                    let result = &a + &b;
                    black_box(result);
                });
            }
        });

        // f64 operations
        group.bench_with_input(BenchmarkId::new("f64_add", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a + &b;
                    black_box(result);
                });
            }
        });

        // f32 multiplication
        group.bench_with_input(BenchmarkId::new("f32_mul", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f32>(&[s]), rng.random::<f32>(&[s])) {
                bencher.iter(|| {
                    let result = &a * &b;
                    black_box(result);
                });
            }
        });

        // f64 multiplication
        group.bench_with_input(BenchmarkId::new("f64_mul", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a * &b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark mathematical operations with SIMD
fn bench_simd_math_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_math_operations");

    for size in [256, 1024, 4096, 16384].iter() {
        // Scalar exp
        group.bench_with_input(BenchmarkId::new("scalar_exp", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(-2.0, 2.0, &[s]) {
                bencher.iter(|| {
                    let vec = arr.to_vec();
                    let result: Vec<f64> = vec.iter().map(|&x| x.exp()).collect();
                    black_box(result);
                });
            }
        });

        // SIMD exp
        group.bench_with_input(BenchmarkId::new("simd_exp", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(-2.0, 2.0, &[s]) {
                bencher.iter(|| {
                    let result = exp(&arr);
                    black_box(result);
                });
            }
        });

        // Scalar sqrt
        group.bench_with_input(
            BenchmarkId::new("scalar_sqrt", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.uniform::<f64>(0.1, 100.0, &[s]) {
                    bencher.iter(|| {
                        let vec = arr.to_vec();
                        let result: Vec<f64> = vec.iter().map(|&x| x.sqrt()).collect();
                        black_box(result);
                    });
                }
            },
        );

        // SIMD sqrt
        group.bench_with_input(BenchmarkId::new("simd_sqrt", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(0.1, 100.0, &[s]) {
                bencher.iter(|| {
                    let result = sqrt(&arr);
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark SIMD alignment effects
fn bench_simd_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_alignment");

    for size in [256, 1024, 4096].iter() {
        // Standard aligned data
        group.bench_with_input(BenchmarkId::new("aligned", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a + &b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_vs_scalar_addition,
    bench_simd_vs_scalar_multiplication,
    bench_simd_vs_scalar_division,
    bench_simd_reductions,
    bench_simd_threshold_analysis,
    bench_simd_data_types,
    bench_simd_math_operations,
    bench_simd_alignment,
);

criterion_main!(benches);
