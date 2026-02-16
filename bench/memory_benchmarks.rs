//! Comprehensive Memory and Cache Benchmarks for NumRS2
//!
//! This benchmark suite tests memory-related performance including:
//! - Memory allocation patterns
//! - Cache efficiency (row-major vs column-major)
//! - Memory bandwidth utilization
//! - Copy vs view operations
//! - In-place vs allocating operations
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::prelude::*;
use std::hint::black_box;

/// Benchmark memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        // Allocate 1D array
        group.bench_with_input(BenchmarkId::new("alloc_1d", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(arr) = rng.random::<f64>(&[s]) {
                    black_box(arr);
                }
            });
        });

        // Allocate 2D array
        let dim = (*size as f64).sqrt() as usize;
        if dim * dim == *size {
            group.bench_with_input(BenchmarkId::new("alloc_2d", size), size, |b, &_s| {
                let rng = random::default_rng();
                b.iter(|| {
                    if let Ok(arr) = rng.random::<f64>(&[dim, dim]) {
                        black_box(arr);
                    }
                });
            });
        }

        // Allocate and initialize with zeros
        group.bench_with_input(
            BenchmarkId::new("alloc_zeros", size),
            size,
            |bencher, &s| {
                bencher.iter(|| {
                    let arr: Array<f64> = Array::zeros(&[s]);
                    black_box(arr);
                });
            },
        );

        // Allocate and initialize with ones
        group.bench_with_input(BenchmarkId::new("alloc_ones", size), size, |bencher, &s| {
            bencher.iter(|| {
                let arr: Array<f64> = Array::ones(&[s]);
                black_box(arr);
            });
        });
    }

    group.finish();
}

/// Benchmark cache efficiency - row-major vs column-major access
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");

    for size in [100, 200, 500, 1000].iter() {
        // Row-major access (cache-friendly)
        group.bench_with_input(BenchmarkId::new("row_major_sum", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    let vec = mat.to_vec();
                    let mut sum = 0.0;
                    for i in 0..s {
                        for j in 0..s {
                            sum += vec[i * s + j];
                        }
                    }
                    black_box(sum);
                });
            }
        });

        // Column-major access (cache-unfriendly)
        group.bench_with_input(BenchmarkId::new("col_major_sum", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    let vec = mat.to_vec();
                    let mut sum = 0.0;
                    for j in 0..s {
                        for i in 0..s {
                            sum += vec[i * s + j];
                        }
                    }
                    black_box(sum);
                });
            }
        });

        // Transpose operation (affects cache access)
        group.bench_with_input(BenchmarkId::new("transpose", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(mat) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    black_box(mat.transpose());
                });
            }
        });
    }

    group.finish();
}

/// Benchmark memory bandwidth utilization
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");

    for size in [1000, 10000, 100000, 1000000, 10000000].iter() {
        // Single read (streaming read)
        group.bench_with_input(BenchmarkId::new("stream_read", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    let vec = arr.to_vec();
                    let mut sum = 0.0;
                    for &val in vec.iter() {
                        sum += val;
                    }
                    black_box(sum);
                });
            }
        });

        // Single write (streaming write)
        group.bench_with_input(BenchmarkId::new("stream_write", size), size, |b, &s| {
            b.iter(|| {
                let mut vec = vec![0.0; s];
                for (i, val) in vec.iter_mut().enumerate() {
                    *val = i as f64;
                }
                black_box(vec);
            });
        });

        // Copy (read + write)
        group.bench_with_input(BenchmarkId::new("copy", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    let result = arr.clone();
                    black_box(result);
                });
            }
        });

        // Triad (a[i] = b[i] + scalar * c[i]) - STREAM benchmark pattern
        group.bench_with_input(BenchmarkId::new("triad", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(b_arr), Ok(c_arr)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                let scalar = 2.5;
                b.iter(|| {
                    // Manual triad computation since SpecialArray lacks mul_scalar
                    let mut vec_c = c_arr.to_vec();
                    let vec_b = b_arr.to_vec();
                    for i in 0..s {
                        vec_c[i] = vec_b[i] + scalar * vec_c[i];
                    }
                    black_box(vec_c);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark copy vs view operations
fn bench_copy_vs_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_vs_view");

    for size in [1000, 10000, 100000].iter() {
        // Full copy
        group.bench_with_input(BenchmarkId::new("copy_full", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    let result = arr.clone();
                    black_box(result);
                });
            }
        });

        // Slice (view)
        group.bench_with_input(BenchmarkId::new("slice_view", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    // slice takes axis and index, not a range
                    // For 1D array, we can use to_vec and slice the vector instead
                    let vec = arr.to_vec();
                    let sliced = &vec[0..s / 2];
                    black_box(sliced);
                });
            }
        });

        // Reshape (view when possible)
        let dim = (*size as f64).sqrt() as usize;
        if dim * dim == *size {
            group.bench_with_input(
                BenchmarkId::new("reshape_view", size),
                size,
                |bencher, &_s| {
                    let rng = random::default_rng();
                    if let Ok(arr) = rng.random::<f64>(&[*size]) {
                        bencher.iter(|| {
                            // reshape returns Array directly, not Result
                            let result = arr.reshape(&[dim, dim]);
                            black_box(result);
                        });
                    }
                },
            );
        }
    }

    group.finish();
}

/// Benchmark in-place vs allocating operations
fn bench_inplace_vs_allocating(c: &mut Criterion) {
    let mut group = c.benchmark_group("inplace_vs_allocating");

    for size in [1000, 10000, 100000, 1000000].iter() {
        // Allocating addition (creates new array)
        group.bench_with_input(
            BenchmarkId::new("allocating_add", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                    bencher.iter(|| {
                        // add() returns Array directly, not Result
                        let result = a.add(&arr_b);
                        black_box(result);
                    });
                }
            },
        );

        // In-place addition (would modify original if API supported)
        // For now, measure the overhead of the operation itself
        group.bench_with_input(
            BenchmarkId::new("inplace_simulation", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                    bencher.iter(|| {
                        // Simulate in-place by modifying vector directly
                        let mut vec_a = a.to_vec();
                        let vec_b = arr_b.to_vec();
                        for i in 0..s {
                            vec_a[i] += vec_b[i];
                        }
                        black_box(vec_a);
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access_patterns");

    let size = 10000;

    // Sequential access
    group.bench_function("sequential", |b| {
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                let vec = arr.to_vec();
                let sum: f64 = vec.iter().sum();
                black_box(sum);
            });
        }
    });

    // Strided access (every 2nd element)
    group.bench_function("strided_2", |b| {
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                let vec = arr.to_vec();
                let mut sum = 0.0;
                for i in (0..size).step_by(2) {
                    sum += vec[i];
                }
                black_box(sum);
            });
        }
    });

    // Strided access (every 4th element)
    group.bench_function("strided_4", |b| {
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                let vec = arr.to_vec();
                let mut sum = 0.0;
                for i in (0..size).step_by(4) {
                    sum += vec[i];
                }
                black_box(sum);
            });
        }
    });

    // Random access
    group.bench_function("random", |b| {
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            // Pre-generate random indices
            let indices: Vec<usize> = (0..1000).map(|i| (i * 13) % size).collect();
            b.iter(|| {
                let vec = arr.to_vec();
                let mut sum = 0.0;
                for &idx in indices.iter() {
                    sum += vec[idx];
                }
                black_box(sum);
            });
        }
    });

    group.finish();
}

/// Benchmark cache line effects
fn bench_cache_line_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_line_effects");

    // Typical cache line size is 64 bytes = 8 f64 values
    let cache_line_floats = 8;

    // Access aligned to cache lines
    group.bench_function("cache_aligned", |b| {
        let size = 10000;
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                let vec = arr.to_vec();
                let mut sum = 0.0;
                for i in (0..size).step_by(cache_line_floats) {
                    sum += vec[i];
                }
                black_box(sum);
            });
        }
    });

    // Access with false sharing potential (multiple threads accessing nearby)
    group.bench_function("potential_false_sharing", |b| {
        let size = 10000;
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                let vec = arr.to_vec();
                // Simulate accessing adjacent elements (could cause false sharing in parallel)
                let mut sum1 = 0.0;
                let mut sum2 = 0.0;
                for i in 0..size / 2 {
                    sum1 += vec[i * 2];
                    sum2 += vec[i * 2 + 1];
                }
                black_box((sum1, sum2));
            });
        }
    });

    group.finish();
}

/// Benchmark memory allocation size effects
fn bench_allocation_size_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_size_effects");

    // Small allocations
    for size in [8, 16, 32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("small", size), size, |bencher, &s| {
            bencher.iter(|| {
                let arr: Array<f64> = Array::zeros(&[s]);
                black_box(arr);
            });
        });
    }

    // Medium allocations
    for size in [1024, 2048, 4096, 8192].iter() {
        group.bench_with_input(BenchmarkId::new("medium", size), size, |bencher, &s| {
            bencher.iter(|| {
                let arr: Array<f64> = Array::zeros(&[s]);
                black_box(arr);
            });
        });
    }

    // Large allocations
    for size in [65536, 131072, 262144, 524288].iter() {
        group.bench_with_input(BenchmarkId::new("large", size), size, |bencher, &s| {
            bencher.iter(|| {
                let arr: Array<f64> = Array::zeros(&[s]);
                black_box(arr);
            });
        });
    }

    group.finish();
}

/// Benchmark contiguous vs non-contiguous memory
fn bench_contiguous_vs_noncontiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_vs_noncontiguous");

    let size = 1000;

    // Contiguous memory access
    group.bench_function("contiguous_sum", |b| {
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size * size]) {
            b.iter(|| {
                if let Ok(result) = sum(&arr, None, false) {
                    black_box(result);
                }
            });
        }
    });

    // Non-contiguous (transposed matrix sum along rows)
    group.bench_function("noncontiguous_sum", |b| {
        let rng = random::default_rng();
        if let Ok(mat) = rng.random::<f64>(&[size, size]) {
            let mat_t = mat.transpose();
            b.iter(|| {
                // Sum along rows of transposed matrix (non-contiguous in original layout)
                if let Ok(result) = sum(&mat_t, Some(1), false) {
                    black_box(result);
                }
            });
        }
    });

    group.finish();
}

/// Benchmark memory prefetching effects
fn bench_prefetching_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetching_effects");

    // Sequential access (good for prefetching)
    group.bench_function("sequential_good_prefetch", |b| {
        let size = 100000;
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                let vec = arr.to_vec();
                let sum: f64 = vec.iter().sum();
                black_box(sum);
            });
        }
    });

    // Random access (poor for prefetching)
    group.bench_function("random_poor_prefetch", |b| {
        let size = 100000;
        let rng = random::default_rng();
        if let Ok(arr) = rng.random::<f64>(&[size]) {
            // Pre-generate random indices
            let indices: Vec<usize> = (0..10000).map(|i| (i * 7919) % size).collect();
            b.iter(|| {
                let vec = arr.to_vec();
                let mut sum = 0.0;
                for &idx in indices.iter() {
                    sum += vec[idx];
                }
                black_box(sum);
            });
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_allocation,
    bench_cache_efficiency,
    bench_memory_bandwidth,
    bench_copy_vs_view,
    bench_inplace_vs_allocating,
    bench_memory_access_patterns,
    bench_cache_line_effects,
    bench_allocation_size_effects,
    bench_contiguous_vs_noncontiguous,
    bench_prefetching_effects,
);

criterion_main!(benches);
