//! Parallel Algorithms Performance Benchmarks
//!
//! Comprehensive benchmarks for parallel algorithm implementations:
//! - Parallel map, reduce, filter, sort operations
//! - Thread scaling analysis (1, 2, 4, 8 threads)
//! - Work distribution and load balancing
//! - Sequential vs parallel comparison
//! - Strong and weak scaling efficiency

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::parallel::parallel_algorithms::{ParallelArrayOps, ParallelConfig};
use std::hint::black_box;

/// Array sizes for testing
const SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

/// Thread counts for scaling tests
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

/// Create parallel configuration with specified thread count
fn create_config(num_threads: usize) -> ParallelConfig {
    ParallelConfig {
        num_threads: Some(num_threads),
        parallel_threshold: 1000,
        block_size: 64,
        numa_aware: false,
        chunk_size: 256,
    }
}

/// Benchmark parallel map operation with different thread counts
fn bench_parallel_map_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_map_scaling");

    for size in [100_000, 1_000_000, 10_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let input: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        for &num_threads in THREAD_COUNTS.iter() {
            let config = create_config(num_threads);
            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
            let mut output = vec![0.0f64; *size];

            group.bench_with_input(
                BenchmarkId::new("threads", format!("{}t_{}", num_threads, size)),
                &num_threads,
                |bencher, _| {
                    bencher.iter(|| {
                        // Complex computation: sqrt + sin + cos
                        ops.parallel_map(&input, &mut output, |x| x.sqrt().sin() + x.cos())
                            .expect("parallel_map should succeed");
                        black_box(&output);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark parallel reduce operation
fn bench_parallel_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_reduce");

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        // Sum reduction
        for &num_threads in THREAD_COUNTS.iter() {
            let config = create_config(num_threads);
            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

            group.bench_with_input(
                BenchmarkId::new("sum", format!("{}t_{}", num_threads, size)),
                &num_threads,
                |bencher, _| {
                    bencher.iter(|| {
                        let result = ops
                            .parallel_reduce(&data, 0.0, |a, b| a + b)
                            .expect("parallel_reduce should succeed");
                        black_box(result);
                    });
                },
            );
        }

        // Product reduction (limited size to avoid overflow)
        if *size <= 100_000 {
            for &num_threads in THREAD_COUNTS.iter() {
                let config = create_config(num_threads);
                let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

                // Use small values to avoid overflow
                let small_data: Vec<f64> = (0..*size).map(|i| 1.0 + (i as f64) * 0.00001).collect();

                group.bench_with_input(
                    BenchmarkId::new("product", format!("{}t_{}", num_threads, size)),
                    &num_threads,
                    |bencher, _| {
                        bencher.iter(|| {
                            let result = ops
                                .parallel_reduce(&small_data, 1.0, |a, b| a * b)
                                .expect("parallel_reduce should succeed");
                            black_box(result);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark parallel filter with different selectivity rates
fn bench_parallel_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_filter");

    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<i32> = (0..*size).collect();

        // Different selectivity rates
        let selectivities = [
            ("10pct", 10), // 10% pass rate
            ("50pct", 50), // 50% pass rate
            ("90pct", 90), // 90% pass rate
        ];

        for (name, threshold) in selectivities.iter() {
            for &num_threads in THREAD_COUNTS.iter() {
                let config = create_config(num_threads);
                let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

                group.bench_with_input(
                    BenchmarkId::new(*name, format!("{}t_{}", num_threads, size)),
                    &num_threads,
                    |bencher, _| {
                        bencher.iter(|| {
                            let result = ops
                                .parallel_filter(&data, |&x| (x % 100) < *threshold)
                                .expect("parallel_filter should succeed");
                            black_box(result);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark parallel sort
fn bench_parallel_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_sort");
    group.sample_size(10); // Reduce sample size for expensive operations

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Test on different data patterns
        let patterns = [
            (
                "random",
                (0..*size).map(|i| i * 7919 % *size).collect::<Vec<_>>(),
            ),
            ("sorted", (0..*size).collect::<Vec<_>>()),
            ("reverse", (0..*size).rev().collect::<Vec<_>>()),
        ];

        for (pattern_name, pattern_data) in patterns.iter() {
            for &num_threads in THREAD_COUNTS.iter() {
                let config = create_config(num_threads);
                let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

                group.bench_with_input(
                    BenchmarkId::new(*pattern_name, format!("{}t_{}", num_threads, size)),
                    &num_threads,
                    |bencher, _| {
                        bencher.iter(|| {
                            let mut data = pattern_data.clone();
                            ops.parallel_sort(&mut data)
                                .expect("parallel_sort should succeed");
                            black_box(data);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark parallel map-reduce operation
fn bench_parallel_map_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_map_reduce");

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        for &num_threads in THREAD_COUNTS.iter() {
            let config = create_config(num_threads);
            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

            group.bench_with_input(
                BenchmarkId::new("sqrt_sum", format!("{}t_{}", num_threads, size)),
                &num_threads,
                |bencher, _| {
                    bencher.iter(|| {
                        // Map: compute sqrt, Reduce: sum
                        let result = ops
                            .parallel_map_reduce(&data, |x| x.sqrt(), |a, b| a + b, 0.0)
                            .expect("parallel_map_reduce should succeed");
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark parallel prefix sum (scan)
fn bench_parallel_prefix_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_prefix_sum");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        for &num_threads in THREAD_COUNTS.iter() {
            let config = create_config(num_threads);
            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
            let mut result = vec![0.0f64; *size];

            group.bench_with_input(
                BenchmarkId::new("prefix_sum", format!("{}t_{}", num_threads, size)),
                &num_threads,
                |bencher, _| {
                    bencher.iter(|| {
                        ops.parallel_prefix_sum(&data, &mut result)
                            .expect("parallel_prefix_sum should succeed");
                        black_box(&result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark strong scaling efficiency
/// Fixed problem size, vary thread count
fn bench_strong_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("strong_scaling");

    let size = 10_000_000;
    group.throughput(Throughput::Elements(size as u64));

    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    // Measure baseline (1 thread)
    let baseline_config = create_config(1);
    let baseline_ops =
        ParallelArrayOps::new(baseline_config).expect("Failed to create parallel ops");
    let mut baseline_output = vec![0.0f64; size];

    group.bench_function("baseline_1thread", |bencher| {
        bencher.iter(|| {
            baseline_ops
                .parallel_map(&data, &mut baseline_output, |x| x.sqrt().sin())
                .expect("parallel_map should succeed");
            black_box(&baseline_output);
        });
    });

    // Measure with multiple threads
    for &num_threads in &[2, 4, 8] {
        let config = create_config(num_threads);
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
        let mut output = vec![0.0f64; size];

        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |bencher, _| {
                bencher.iter(|| {
                    ops.parallel_map(&data, &mut output, |x| x.sqrt().sin())
                        .expect("parallel_map should succeed");
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark weak scaling efficiency
/// Problem size increases proportionally with thread count
fn bench_weak_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("weak_scaling");

    let base_size = 1_000_000; // Size per thread

    for &num_threads in THREAD_COUNTS.iter() {
        let size = base_size * num_threads;
        group.throughput(Throughput::Elements(size as u64));

        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let config = create_config(num_threads);
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
        let mut output = vec![0.0f64; size];

        group.bench_with_input(
            BenchmarkId::new("size_per_thread", format!("{}t_{}elem", num_threads, size)),
            &num_threads,
            |bencher, _| {
                bencher.iter(|| {
                    ops.parallel_map(&data, &mut output, |x| x.sqrt().sin())
                        .expect("parallel_map should succeed");
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark binary operations with different thread counts
fn bench_parallel_binary_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_binary_op");

    for size in [100_000, 1_000_000, 10_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64) * 2.0).collect();

        for &num_threads in THREAD_COUNTS.iter() {
            let config = create_config(num_threads);
            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
            let mut result = vec![0.0f64; *size];

            group.bench_with_input(
                BenchmarkId::new("add", format!("{}t_{}", num_threads, size)),
                &num_threads,
                |bencher, _| {
                    bencher.iter(|| {
                        ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y)
                            .expect("parallel_binary_op should succeed");
                        black_box(&result);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("mul", format!("{}t_{}", num_threads, size)),
                &num_threads,
                |bencher, _| {
                    bencher.iter(|| {
                        ops.parallel_binary_op(&a, &b, &mut result, |x, y| x * y)
                            .expect("parallel_binary_op should succeed");
                        black_box(&result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark work distribution with irregular workloads
fn bench_irregular_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("irregular_workload");

    let size = 100_000;
    group.throughput(Throughput::Elements(size as u64));

    // Create irregular workload: some elements require more computation
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    for &num_threads in THREAD_COUNTS.iter() {
        let config = create_config(num_threads);
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
        let mut output = vec![0.0f64; size];

        group.bench_with_input(
            BenchmarkId::new("variable_work", format!("{}t", num_threads)),
            &num_threads,
            |bencher, _| {
                bencher.iter(|| {
                    ops.parallel_map(&data, &mut output, |x| {
                        // Irregular work: more iterations for larger indices
                        let iterations = ((x % 100.0) as usize) + 1;
                        let mut result = x;
                        for _ in 0..iterations {
                            result = result.sqrt() + 0.1;
                        }
                        result
                    })
                    .expect("parallel_map should succeed");
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_map_scaling,
    bench_parallel_reduce,
    bench_parallel_filter,
    bench_parallel_sort,
    bench_parallel_map_reduce,
    bench_parallel_prefix_sum,
    bench_strong_scaling,
    bench_weak_scaling,
    bench_parallel_binary_op,
    bench_irregular_workload,
);
criterion_main!(benches);
