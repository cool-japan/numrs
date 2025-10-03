//! Benchmarks for linear algebra operations in NumRS2
//!
//! This file contains benchmarks for various linear algebra operations
//! to track performance and identify bottlenecks.

#[macro_use]
extern crate criterion;
use criterion::{BenchmarkId, Criterion};

use numrs2::prelude::*;
use std::hint::black_box;

/// Benchmark matrix multiplication for different sizes
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bench, &size| {
            // Create random matrices of the specified size
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();
            let b = rng.random::<f64>(&[size, size]).unwrap();

            bench.iter(|| black_box(a.matmul(&b).unwrap()));
        });

        // Also benchmark SIMD matrix multiplication if available
        group.bench_with_input(BenchmarkId::new("simd_matmul", size), size, |b, &size| {
            // Create random matrices of the specified size
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();
            let b = rng.random::<f64>(&[size, size]).unwrap();

            // b.iter(|| black_box(simd_matmul(&a, &b).unwrap())); // simd_matmul not available
        });
    }

    group.finish();
}

/// Benchmark matrix inversion for different sizes
fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inverse");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("inverse", size), size, |b, &size| {
            // Create a random positive definite matrix (well-conditioned for inversion)
            let rng = random::default_rng();
            let tmp = rng.random::<f64>(&[size, size]).unwrap();
            let tmp_t = tmp.transpose();
            let a = tmp.matmul(&tmp_t).unwrap();

            // b.iter(|| black_box(inv(&a).unwrap())); // inv requires lapack feature
        });
    }

    group.finish();
}

/// Benchmark determinant calculation for different sizes
fn bench_determinant(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinant");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("det", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            // b.iter(|| black_box(det(&a).unwrap())); // det requires lapack feature
        });
    }

    group.finish();
}

/// Benchmark solving linear systems for different sizes
fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_linear_system");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("solve", size), size, |b, &size| {
            // Create a random positive definite matrix (well-conditioned)
            let rng = random::default_rng();
            let tmp = rng.random::<f64>(&[size, size]).unwrap();
            let tmp_t = tmp.transpose();
            let a = tmp.matmul(&tmp_t).unwrap();

            // Create a random right-hand side
            let x = rng.random::<f64>(&[size]).unwrap();

            // b.iter(|| black_box(solve(&a, &x).unwrap())); // solve requires lapack feature
        });
    }

    group.finish();
}

/// Benchmark eigendecomposition for different sizes
fn bench_eigendecomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigendecomposition");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("eigh", size), size, |b, &size| {
            // Create a random symmetric matrix
            let rng = random::default_rng();
            let tmp = rng.random::<f64>(&[size, size]).unwrap();
            let tmp_t = tmp.transpose();
            let a = tmp.add(&tmp_t).multiply_scalar(0.5);

            b.iter(|| black_box(eigh(&a, "lower").unwrap()));
        });
    }

    group.finish();
}

/// Benchmark SVD for different sizes
fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("svd", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            // b.iter(|| black_box(svd(&a).unwrap())); // svd requires lapack feature
        });
    }

    group.finish();
}

/// Benchmark QR decomposition for different sizes
fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_decomposition");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("qr", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            // b.iter(|| black_box(qr(&a).unwrap())); // qr requires lapack feature
        });
    }

    group.finish();
}

/// Benchmark Cholesky decomposition for different sizes
fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_decomposition");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("cholesky", size), size, |b, &size| {
            // Create a random positive definite matrix
            let rng = random::default_rng();
            let tmp = rng.random::<f64>(&[size, size]).unwrap();
            let tmp_t = tmp.transpose();
            let a = tmp.matmul(&tmp_t).unwrap();

            // b.iter(|| black_box(cholesky(&a, "lower").unwrap())); // cholesky requires lapack feature
        });
    }

    group.finish();
}

/// Benchmark LU decomposition for different sizes
fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_decomposition");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("lu", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            b.iter(|| black_box(lu(&a).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark matrix norm calculations for different sizes
fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_norm");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("frobenius_norm", size),
            size,
            |b, &size| {
                // Create a random matrix
                let rng = random::default_rng();
                let a = rng.random::<f64>(&[size, size]).unwrap();

                b.iter(|| black_box(norm(&a, Some(2.0)).unwrap()));
            },
        );

        group.bench_with_input(BenchmarkId::new("inf_norm", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            b.iter(|| black_box(norm(&a, Some(f64::INFINITY)).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark matrix rank calculation for different sizes
fn bench_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_rank");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("rank", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            b.iter(|| black_box(matrix_rank(&a, None).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark condition number calculation for different sizes
fn bench_condition_number(c: &mut Criterion) {
    let mut group = c.benchmark_group("condition_number");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("cond", size), size, |b, &size| {
            // Create a random matrix
            let rng = random::default_rng();
            let a = rng.random::<f64>(&[size, size]).unwrap();

            b.iter(|| black_box(condition_number(&a).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_inverse,
    bench_determinant,
    bench_solve,
    bench_eigendecomposition,
    bench_svd,
    bench_qr,
    bench_cholesky,
    bench_lu,
    bench_norm,
    bench_rank,
    bench_condition_number,
);
criterion_main!(benches);
