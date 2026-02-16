//! Benchmark for NumRS2 Expression Templates
//!
//! This benchmark measures the performance of:
//! - SharedArray operations vs regular Array
//! - SharedExpr lazy evaluation
//! - CSE (Common Subexpression Elimination)
//! - Memory access pattern optimization

#![allow(deprecated)]
#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::array::Array;
use numrs2::expr::{CachedExpr, ExprCache, SharedExpr, SharedExprBuilder};
use numrs2::memory_optimize::access_patterns::{
    cache_aware_binary_op, cache_aware_copy, cache_aware_transform, BlockedIterator,
};
use numrs2::shared_array::SharedArray;
use std::hint::black_box;

/// Benchmark SharedArray vs Array operations
fn bench_shared_array_vs_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_vs_regular_array");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // Regular Array creation
        group.bench_with_input(BenchmarkId::new("array_creation", size), &size, |b, _| {
            b.iter(|| {
                let arr = Array::from_vec(data.clone());
                black_box(arr)
            })
        });

        // SharedArray creation
        group.bench_with_input(
            BenchmarkId::new("shared_array_creation", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let arr: SharedArray<f64> = SharedArray::from_vec(data.clone());
                    black_box(arr)
                })
            },
        );

        // SharedArray clone (O(1) ref count increment)
        let shared = SharedArray::from_vec(data.clone());
        group.bench_with_input(
            BenchmarkId::new("shared_array_clone", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let cloned = shared.clone();
                    black_box(cloned)
                })
            },
        );

        // Regular Array clone (full copy)
        let regular = Array::from_vec(data.clone());
        group.bench_with_input(BenchmarkId::new("array_clone", size), &size, |b, _| {
            b.iter(|| {
                let cloned = regular.clone();
                black_box(cloned)
            })
        });
    }

    group.finish();
}

/// Benchmark SharedArray operator overloading
fn bench_operator_overloading(c: &mut Criterion) {
    let mut group = c.benchmark_group("operator_overloading");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data_a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();

        let shared_a: SharedArray<f64> = SharedArray::from_vec(data_a.clone());
        let shared_b: SharedArray<f64> = SharedArray::from_vec(data_b.clone());

        // SharedArray addition
        group.bench_with_input(BenchmarkId::new("shared_array_add", size), &size, |b, _| {
            b.iter(|| {
                let result = shared_a.clone() + shared_b.clone();
                black_box(result)
            })
        });

        // SharedArray multiplication
        group.bench_with_input(BenchmarkId::new("shared_array_mul", size), &size, |b, _| {
            b.iter(|| {
                let result = shared_a.clone() * shared_b.clone();
                black_box(result)
            })
        });

        // Chained operations
        group.bench_with_input(
            BenchmarkId::new("shared_array_chain", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = (shared_a.clone() + shared_b.clone()) * 2.0 - 1.0;
                    black_box(result)
                })
            },
        );

        // Scalar operations
        group.bench_with_input(
            BenchmarkId::new("shared_array_scalar_mul", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = shared_a.clone() * 2.0;
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SharedExpr lazy evaluation
fn bench_shared_expr(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_expr");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let shared: SharedArray<f64> = SharedArray::from_vec(data);

        // Expression building (no computation)
        group.bench_with_input(BenchmarkId::new("expr_build", size), &size, |b, _| {
            b.iter(|| {
                let expr = SharedExprBuilder::from_shared_array(shared.clone());
                let mapped = expr.map(|x| x * x);
                black_box(mapped)
            })
        });

        // Expression evaluation
        group.bench_with_input(BenchmarkId::new("expr_eval", size), &size, |b, _| {
            b.iter(|| {
                let expr = SharedExprBuilder::from_shared_array(shared.clone());
                let mapped = expr.map(|x| x * x);
                let result = mapped.eval();
                black_box(result)
            })
        });

        // Chained expression evaluation
        group.bench_with_input(BenchmarkId::new("expr_chain_eval", size), &size, |b, _| {
            b.iter(|| {
                let expr = SharedExprBuilder::from_shared_array(shared.clone());
                let mapped = expr.map(|x| x * x + 2.0 * x + 1.0);
                let result = mapped.eval();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark CSE (Common Subexpression Elimination)
fn bench_cse(c: &mut Criterion) {
    let mut group = c.benchmark_group("cse");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data_a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();

        let a: SharedArray<f64> = SharedArray::from_vec(data_a);
        let b: SharedArray<f64> = SharedArray::from_vec(data_b);

        // Without caching - compute sum twice
        group.bench_with_input(
            BenchmarkId::new("without_cache", size),
            &size,
            |b_iter, _| {
                b_iter.iter(|| {
                    let sum1 = a.clone() + b.clone();
                    let sum2 = a.clone() + b.clone();
                    black_box((sum1, sum2))
                })
            },
        );

        // With caching - compute once, reuse
        group.bench_with_input(BenchmarkId::new("with_cache", size), &size, |b_iter, _| {
            b_iter.iter(|| {
                let cache: ExprCache<f64> = ExprCache::new();
                let sum = a.clone() + b.clone();
                let sum_expr = SharedExprBuilder::from_shared_array(sum);
                let cached = CachedExpr::new(sum_expr.into_expr(), cache);
                let result1 = cached.eval();
                let result2 = cached.eval();
                black_box((result1, result2))
            })
        });
    }

    group.finish();
}

/// Benchmark memory access pattern optimization
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    let sizes = vec![10000, 100000, 1000000];

    for size in sizes {
        let src: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // Standard copy
        group.bench_with_input(BenchmarkId::new("standard_copy", size), &size, |b, _| {
            b.iter(|| {
                let mut dst = vec![0.0f64; size];
                dst.copy_from_slice(&src);
                black_box(dst)
            })
        });

        // Cache-aware copy
        group.bench_with_input(BenchmarkId::new("cache_aware_copy", size), &size, |b, _| {
            b.iter(|| {
                let mut dst = vec![0.0f64; size];
                cache_aware_copy(&src, &mut dst);
                black_box(dst)
            })
        });

        // Standard transform
        group.bench_with_input(
            BenchmarkId::new("standard_transform", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut dst = vec![0.0f64; size];
                    for (d, s) in dst.iter_mut().zip(src.iter()) {
                        *d = s * 2.0;
                    }
                    black_box(dst)
                })
            },
        );

        // Cache-aware transform
        group.bench_with_input(
            BenchmarkId::new("cache_aware_transform", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut dst = vec![0.0f64; size];
                    cache_aware_transform(&src, &mut dst, |x| x * 2.0);
                    black_box(dst)
                })
            },
        );

        // Standard binary op
        let src2: Vec<f64> = (0..size).map(|i| (i * 3) as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("standard_binary_op", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut dst = vec![0.0f64; size];
                    for i in 0..size {
                        dst[i] = src[i] + src2[i];
                    }
                    black_box(dst)
                })
            },
        );

        // Cache-aware binary op
        group.bench_with_input(
            BenchmarkId::new("cache_aware_binary_op", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut dst = vec![0.0f64; size];
                    cache_aware_binary_op(&src, &src2, &mut dst, |a, b| a + b);
                    black_box(dst)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark blocked iteration
fn bench_blocked_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocked_iteration");

    let sizes = vec![10000, 100000, 1000000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // Sequential iteration
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0;
                for &value in data.iter().take(size) {
                    sum += value;
                }
                black_box(sum)
            })
        });

        // Blocked iteration
        group.bench_with_input(BenchmarkId::new("blocked", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0;
                let block_iter = BlockedIterator::new(size, 64);
                for block in block_iter {
                    for &value in data[block.start..block.end].iter() {
                        sum += value;
                    }
                }
                black_box(sum)
            })
        });

        // Blocked with larger block size
        group.bench_with_input(BenchmarkId::new("blocked_large", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0;
                let block_iter = BlockedIterator::new(size, 256);
                for block in block_iter {
                    for &value in data[block.start..block.end].iter() {
                        sum += value;
                    }
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_shared_array_vs_array,
    bench_operator_overloading,
    bench_shared_expr,
    bench_cse,
    bench_memory_patterns,
    bench_blocked_iteration
);
criterion_main!(benches);
