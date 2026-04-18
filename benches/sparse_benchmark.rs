//! Sparse Matrix and Array Benchmarks for NumRS2
//!
//! This benchmark suite tests sparse matrix operations including:
//! - Sparse matrix creation from dense arrays at various sizes and densities
//! - Format conversion (CSR, CSC, DIA)
//! - Sparse matrix multiplication
//! - Transpose operations
//! - Sparse-to-dense conversion
//! - Sparse vs dense matmul comparison
//!
//! All benchmarks follow COOLJAPAN policies: no unwrap(), snake_case, no warnings.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::prelude::*;
use std::hint::black_box;

/// Build a dense Array<f64> of shape [n, n] with approximately `density_pct`% non-zero elements.
/// Generates a uniform random [0,1] array, then zeros out elements above the density threshold.
fn make_sparse_dense(n: usize, density_pct: u32) -> Option<Array<f64>> {
    let rng = random::default_rng();
    let raw = rng.random::<f64>(&[n, n]).ok()?;
    let threshold = density_pct as f64 / 100.0;
    let data: Vec<f64> = raw
        .to_vec()
        .into_iter()
        .map(|v| if v < threshold { v } else { 0.0 })
        .collect();
    let arr = Array::from_vec(data).reshape(&[n, n]);
    Some(arr)
}

/// Benchmark `SparseMatrix::from_array()` for various sizes and densities.
fn bench_sparse_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_creation");

    let sizes = [100_usize, 500, 1000];
    let densities: &[u32] = &[1, 10, 50];

    for &s in &sizes {
        for &density_pct in densities {
            // Prepare dense array outside the benchmark loop
            if let Some(dense) = make_sparse_dense(s, density_pct) {
                group.bench_with_input(
                    BenchmarkId::new(format!("from_dense_{}pct", density_pct), s),
                    &s,
                    |b, _| {
                        b.iter(|| {
                            if let Ok(sm) = SparseMatrix::from_array(&dense) {
                                black_box(sm);
                            }
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark format conversions (CSR, CSC, DIA) on a 500x500 sparse matrix at 10% density.
fn bench_sparse_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_format_conversion");

    let size = 500_usize;
    let density_pct = 10_u32;

    if let Some(dense) = make_sparse_dense(size, density_pct) {
        if let Ok(sm) = SparseMatrix::from_array(&dense) {
            // to_csr: clone per iteration since to_csr is idempotent
            group.bench_function("to_csr_500x500_10pct", |b| {
                b.iter(|| {
                    let mut clone = sm.clone();
                    let _ = clone.to_csr();
                    black_box(clone);
                });
            });

            // to_csc: clone per iteration
            group.bench_function("to_csc_500x500_10pct", |b| {
                b.iter(|| {
                    let mut clone = sm.clone();
                    let _ = clone.to_csc();
                    black_box(clone);
                });
            });

            // to_dia: clone per iteration
            group.bench_function("to_dia_500x500_10pct", |b| {
                b.iter(|| {
                    let mut clone = sm.clone();
                    let _ = clone.to_dia();
                    black_box(clone);
                });
            });
        }
    }

    group.finish();
}

/// Benchmark `SparseMatrix::matmul()` for NxN @ NxN at various sizes (10% density).
fn bench_sparse_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matmul");

    let sizes = [100_usize, 300, 500];
    let density_pct = 10_u32;

    for &s in &sizes {
        // Scale throughput by N^2 elements
        group.throughput(Throughput::Elements((s * s) as u64));

        if let (Some(dense_a), Some(dense_b)) = (
            make_sparse_dense(s, density_pct),
            make_sparse_dense(s, density_pct),
        ) {
            if let (Ok(sm_a), Ok(sm_b)) = (
                SparseMatrix::from_array(&dense_a),
                SparseMatrix::from_array(&dense_b),
            ) {
                group.bench_with_input(BenchmarkId::new("sparse_matmul_10pct", s), &s, |b, _| {
                    b.iter(|| {
                        if let Ok(result) = sm_a.matmul(&sm_b) {
                            black_box(result);
                        }
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark `SparseMatrix::transpose()` at various sizes (10% density).
fn bench_sparse_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_transpose");

    let sizes = [100_usize, 500, 1000];
    let density_pct = 10_u32;

    for &s in &sizes {
        if let Some(dense) = make_sparse_dense(s, density_pct) {
            if let Ok(sm) = SparseMatrix::from_array(&dense) {
                group.bench_with_input(BenchmarkId::new("transpose_10pct", s), &s, |b, _| {
                    b.iter(|| {
                        if let Ok(result) = sm.transpose() {
                            black_box(result);
                        }
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark `SparseMatrix::to_array()` (sparse→dense) at various sizes (10% density).
fn bench_sparse_to_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_to_dense");

    let sizes = [100_usize, 500];
    let density_pct = 10_u32;

    for &s in &sizes {
        if let Some(dense) = make_sparse_dense(s, density_pct) {
            if let Ok(sm) = SparseMatrix::from_array(&dense) {
                group.bench_with_input(BenchmarkId::new("to_dense_10pct", s), &s, |b, _| {
                    b.iter(|| {
                        black_box(sm.to_array());
                    });
                });
            }
        }
    }

    group.finish();
}

/// Comparison: sparse matmul vs dense matmul for 300x300 at 5% density.
fn bench_sparse_vs_dense_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense_matmul_300x300_5pct");

    let size = 300_usize;
    let density_pct = 5_u32;

    if let (Some(dense_a), Some(dense_b)) = (
        make_sparse_dense(size, density_pct),
        make_sparse_dense(size, density_pct),
    ) {
        // Dense matmul benchmark
        group.bench_function("dense_matmul", |b| {
            b.iter(|| {
                if let Ok(result) = dense_a.matmul(&dense_b) {
                    black_box(result);
                }
            });
        });

        // Sparse matmul benchmark
        if let (Ok(sm_a), Ok(sm_b)) = (
            SparseMatrix::from_array(&dense_a),
            SparseMatrix::from_array(&dense_b),
        ) {
            group.bench_function("sparse_matmul", |b| {
                b.iter(|| {
                    if let Ok(result) = sm_a.matmul(&sm_b) {
                        black_box(result);
                    }
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    sparse_benches,
    bench_sparse_creation,
    bench_sparse_format_conversion,
    bench_sparse_matmul,
    bench_sparse_transpose,
    bench_sparse_to_dense,
    bench_sparse_vs_dense_matmul,
);
criterion_main!(sparse_benches);
