//! Comprehensive Linear Algebra Benchmarks for NumRS2
//!
//! This benchmark suite tests all major linear algebra operations including:
//! - Matrix multiplication (various sizes)
//! - Matrix decompositions (SVD, QR, LU, Cholesky)
//! - Matrix inverse and determinant
//! - Iterative solvers
//! - Eigenvalue decomposition
//! - Matrix norms and condition numbers
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::linalg::vector_ops::cross;
use numrs2::linalg_accelerated::{cholesky, det, eig, eigvals, inv, lu, qr, solve, svd};
use numrs2::prelude::*;
use scirs2_core::ndarray::Ix2;
use scirs2_linalg::compat::{cond, matrix_rank};
use std::hint::black_box;

/// Benchmark matrix multiplication for various sizes
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");

    // Test sizes from small to large
    for size in [10, 50, 100, 200, 500, 1000].iter() {
        // Throughput: 2*N^3 FLOPs for NxN matrix multiplication (multiply-adds)
        group.throughput(Throughput::Elements(
            (*size as u64) * (*size as u64) * (*size as u64) * 2,
        ));
        group.bench_with_input(
            BenchmarkId::new("square_matmul", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let (Ok(a), Ok(b)) = (rng.random::<f64>(&[s, s]), rng.random::<f64>(&[s, s])) {
                    bencher.iter(|| {
                        if let Ok(result) = a.matmul(&b) {
                            black_box(result);
                        }
                    });
                }
            },
        );

        // Benchmark rectangular matrices: [s, 2s] @ [2s, s] -> [s, s]: FLOPs = 2*s*s*(2s) = 4*s^3
        group.throughput(Throughput::Elements(
            (*size as u64) * (*size as u64) * (*size as u64) * 4,
        ));
        group.bench_with_input(
            BenchmarkId::new("rect_matmul", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let (Ok(a), Ok(b)) = (
                    rng.random::<f64>(&[s, s * 2]),
                    rng.random::<f64>(&[s * 2, s]),
                ) {
                    bencher.iter(|| {
                        if let Ok(result) = a.matmul(&b) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark matrix-vector multiplication
fn bench_matrix_vector_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_vector_multiplication");

    for size in [10, 50, 100, 200, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("matvec", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(mat), Ok(vec)) = (rng.random::<f64>(&[s, s]), rng.random::<f64>(&[s])) {
                b.iter(|| {
                    if let Ok(result) = mat.matmul(&vec) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark matrix transpose operations
fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    for size in [10, 50, 100, 200, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("square_transpose", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    black_box(mat.transpose());
                });
            }
        });

        // Benchmark rectangular transpose
        group.bench_with_input(BenchmarkId::new("rect_transpose", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s * 2]) {
                b.iter(|| {
                    black_box(mat.transpose());
                });
            }
        });
    }

    group.finish();
}

/// Benchmark matrix inverse calculation
fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inverse");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("inverse", size), size, |b, &s| {
            let rng = random::default_rng();
            // Create well-conditioned positive definite matrix
            if let Ok(tmp) = rng.random::<f64>(&[s, s]) {
                let tmp_t = tmp.transpose();
                if let Ok(mat) = tmp.matmul(&tmp_t) {
                    // Add identity to improve conditioning
                    let eye = Array::eye(s, s, 0);
                    let mat = mat.add(&eye);
                    b.iter(|| {
                        if let Ok(result) = inv(&mat) {
                            black_box(result);
                        }
                    });
                }
            }
        });
    }

    group.finish();
}

/// Benchmark determinant calculation
fn bench_determinant(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinant");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("det", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = det(&mat) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark matrix norms
fn bench_matrix_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_norms");

    for size in [10, 50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::new("frobenius_norm", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = norm(&mat, Some(2.0)) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("inf_norm", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = norm(&mat, Some(f64::INFINITY)) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("one_norm", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = norm(&mat, Some(1.0)) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark QR decomposition
fn bench_qr_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_decomposition");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("qr", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = qr(&mat) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Cholesky decomposition
fn bench_cholesky_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_decomposition");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("cholesky_lower", size), size, |b, &s| {
            let rng = random::default_rng();
            // Create positive definite matrix
            if let Ok(tmp) = rng.random::<f64>(&[s, s]) {
                let tmp_t = tmp.transpose();
                if let Ok(mat) = tmp.matmul(&tmp_t) {
                    // Add identity to ensure positive definiteness
                    let eye = Array::eye(s, s, 0);
                    let mat = mat.add(&eye);
                    b.iter(|| {
                        if let Ok(result) = cholesky(&mat) {
                            black_box(result);
                        }
                    });
                }
            }
        });

        group.bench_with_input(BenchmarkId::new("cholesky_upper", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(tmp) = rng.random::<f64>(&[s, s]) {
                let tmp_t = tmp.transpose();
                if let Ok(mat) = tmp.matmul(&tmp_t) {
                    let eye = Array::eye(s, s, 0);
                    let mat = mat.add(&eye);
                    b.iter(|| {
                        if let Ok(result) = cholesky(&mat) {
                            black_box(result);
                        }
                    });
                }
            }
        });
    }

    group.finish();
}

/// Benchmark SVD (Singular Value Decomposition)
fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("svd_square", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = svd(&mat, true) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("svd_rect", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s * 2]) {
                b.iter(|| {
                    if let Ok(result) = svd(&mat, true) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark LU decomposition
fn bench_lu_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_decomposition");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("lu", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = lu(&mat) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark eigenvalue decomposition
fn bench_eigendecomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigendecomposition");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("eig", size), size, |b, &s| {
            let rng = random::default_rng();
            // Create symmetric matrix for stable eigendecomposition
            if let Ok(tmp) = rng.random::<f64>(&[s, s]) {
                let tmp_t = tmp.transpose();
                let mat = tmp.add(&tmp_t);
                b.iter(|| {
                    if let Ok(result) = eig(&mat) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("eigvals", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(tmp) = rng.random::<f64>(&[s, s]) {
                let tmp_t = tmp.transpose();
                let mat = tmp.add(&tmp_t);
                b.iter(|| {
                    if let Ok(result) = eigvals(&mat) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark solving linear systems
fn bench_solve_linear_systems(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_linear_systems");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("solve", size), size, |b, &s| {
            let rng = random::default_rng();
            // Create well-conditioned system
            if let Ok(tmp) = rng.random::<f64>(&[s, s]) {
                let tmp_t = tmp.transpose();
                if let Ok(mat) = tmp.matmul(&tmp_t) {
                    let eye = Array::eye(s, s, 0);
                    let mat = mat.add(&eye);
                    if let Ok(rhs) = rng.random::<f64>(&[s]) {
                        b.iter(|| {
                            if let Ok(result) = solve(&mat, &rhs) {
                                black_box(result);
                            }
                        });
                    }
                }
            }
        });
    }

    group.finish();
}

/// Benchmark matrix rank calculation
fn bench_matrix_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_rank");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("rank", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(mat_view) = mat.array().view().into_dimensionality::<Ix2>() {
                        if let Ok(result) = matrix_rank(&mat_view, None) {
                            black_box(result);
                        }
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark condition number calculation
fn bench_condition_number(c: &mut Criterion) {
    let mut group = c.benchmark_group("condition_number");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("cond", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(mat_view) = mat.array().view().into_dimensionality::<Ix2>() {
                        if let Ok(result) = cond::<f64, _>(&mat_view) {
                            black_box(result);
                        }
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark matrix trace
fn bench_matrix_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_trace");

    for size in [10, 50, 100, 200, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("trace", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = trace(&mat) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark outer product
fn bench_outer_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("outer_product");

    for size in [10, 50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::new("outer", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(vec_a), Ok(vec_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                b.iter(|| {
                    if let Ok(result) = outer(&vec_a, &vec_b) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark inner/dot product
fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");

    for size in [10, 50, 100, 200, 500, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("dot", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(vec_a), Ok(vec_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                b.iter(|| {
                    if let Ok(result) = dot(&vec_a, &vec_b) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark cross product (3D vectors)
fn bench_cross_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_product");

    group.bench_function("cross_3d", |b| {
        let rng = random::default_rng();
        if let (Ok(vec_a), Ok(vec_b)) = (rng.random::<f64>(&[3]), rng.random::<f64>(&[3])) {
            b.iter(|| {
                if let Ok(result) = cross(&vec_a, &vec_b) {
                    black_box(result);
                }
            });
        }
    });

    group.finish();
}

/// Benchmark Kronecker product
fn bench_kronecker_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("kronecker_product");

    for size in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::new("kron", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(mat_a), Ok(mat_b)) = (rng.random::<f64>(&[s, s]), rng.random::<f64>(&[s, s]))
            {
                b.iter(|| {
                    if let Ok(result) = kron(&mat_a, &mat_b) {
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
    bench_matrix_multiplication,
    bench_matrix_vector_multiplication,
    bench_transpose,
    bench_inverse,
    bench_determinant,
    bench_matrix_norms,
    bench_qr_decomposition,
    bench_cholesky_decomposition,
    bench_svd,
    bench_lu_decomposition,
    bench_eigendecomposition,
    bench_solve_linear_systems,
    bench_matrix_rank,
    bench_condition_number,
    bench_matrix_trace,
    bench_outer_product,
    bench_inner_product,
    bench_cross_product,
    bench_kronecker_product,
);

criterion_main!(benches);
