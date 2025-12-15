//! Simple benchmark for NumRS2 core operations

#![allow(deprecated)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::array::Array;
use numrs2::stats::Statistics;
use std::hint::black_box;

/// Benchmark basic array operations
fn bench_array_basics(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_basics");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let arr = Array::from_vec(data);

        // Benchmark array creation
        group.bench_with_input(
            BenchmarkId::new("array_creation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
                    let arr = Array::from_vec(data);
                    black_box(arr)
                })
            },
        );

        // Benchmark statistics
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.mean();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.std();
                black_box(result)
            })
        });

        // Benchmark array access
        group.bench_with_input(BenchmarkId::new("array_access", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..size {
                    if let Ok(val) = arr.get(&[i]) {
                        sum += val;
                    }
                }
                black_box(sum)
            })
        });

        // Benchmark reshaping
        if size == 10000 {
            group.bench_with_input(BenchmarkId::new("reshape", size), &size, |b, _| {
                b.iter(|| {
                    let result = arr.reshape(&[100, 100]);
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

/// Benchmark matrix operations
fn bench_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_ops");

    let sizes = vec![100, 200, 500];

    for size in sizes {
        let data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let matrix = Array::from_vec(data).reshape(&[size, size]);

        // Benchmark matrix access patterns
        group.bench_with_input(
            BenchmarkId::new("sequential_access", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..size {
                        for j in 0..size {
                            if let Ok(val) = matrix.get(&[i, j]) {
                                sum += val;
                            }
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark transpose
        group.bench_with_input(BenchmarkId::new("transpose", size), &size, |b, _| {
            b.iter(|| {
                let result = matrix.transpose();
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_array_basics, bench_matrix_ops);
criterion_main!(benches);
