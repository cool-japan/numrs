//! Memory Optimization Performance Benchmarks
//!
//! Comprehensive benchmarks comparing memory-optimized operations against
//! standard implementations. Memory-optimized operations minimize allocations
//! through:
//! - Direct iteration (avoiding to_vec())
//! - In-place operations (reusing buffers)
//! - Buffer reuse patterns
//! - Stack allocation for small arrays

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use std::hint::black_box;

/// Array sizes for testing
const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000];

/// Benchmark sum: sum_optimized vs sum
/// sum_optimized uses direct iteration without to_vec() allocation
fn bench_sum_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_comparison");

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let array = Array::from_vec(data);

        // Standard sum (allocates via to_vec())
        group.bench_with_input(
            BenchmarkId::new("sum_standard", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.sum());
                    black_box(result);
                });
            },
        );

        // Optimized sum (no allocation, direct iteration)
        group.bench_with_input(
            BenchmarkId::new("sum_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.sum_optimized());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark product: product_optimized vs product
fn bench_product_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_comparison");

    for size in SIZES.iter().take(4) {
        // Limit size to avoid overflow
        group.throughput(Throughput::Elements(*size as u64));

        // Use small values to avoid overflow
        let data: Vec<f64> = (0..*size).map(|i| 1.0 + (i as f64) * 0.0001).collect();
        let array = Array::from_vec(data);

        // Standard product (allocates via to_vec())
        group.bench_with_input(
            BenchmarkId::new("product_standard", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.product());
                    black_box(result);
                });
            },
        );

        // Optimized product (no allocation)
        group.bench_with_input(
            BenchmarkId::new("product_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.product_optimized());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mean_optimized
/// Shows memory-efficient mean calculation with parallel support
fn bench_mean_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_optimized");

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let array = Array::from_vec(data);

        group.bench_with_input(
            BenchmarkId::new("mean_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.mean_optimized());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark variance_optimized and std_optimized
fn bench_statistical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_operations");

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let array = Array::from_vec(data);

        group.bench_with_input(
            BenchmarkId::new("variance_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.variance_optimized());
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.std_optimized());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark in-place operations: map_inplace vs map
/// map_inplace modifies array in place, map allocates new array
fn bench_inplace_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("inplace_operations");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Standard map (allocates new array)
        group.bench_with_input(
            BenchmarkId::new("map_allocating", size),
            size,
            |bencher, &size| {
                bencher.iter(|| {
                    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
                    let array = Array::from_vec(data);
                    let result = black_box(array.map(|x| x * 2.0 + 1.0));
                    black_box(result);
                });
            },
        );

        // In-place map (reuses existing array memory)
        group.bench_with_input(
            BenchmarkId::new("map_inplace", size),
            size,
            |bencher, &size| {
                bencher.iter(|| {
                    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
                    let mut array = Array::from_vec(data);
                    array.map_inplace(|x| x * 2.0 + 1.0);
                    black_box(&array);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark buffer reuse: map_to vs map
/// map_to writes to pre-allocated output buffer
fn bench_buffer_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_reuse");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let array = Array::from_vec(data.clone());

        // Standard map (allocates new array each time)
        group.bench_with_input(
            BenchmarkId::new("map_allocating", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.map(|x| x.sqrt()));
                    black_box(result);
                });
            },
        );

        // map_to (reuses pre-allocated buffer)
        group.bench_with_input(
            BenchmarkId::new("map_to_reuse", size),
            size,
            |bencher, _| {
                let mut output = Array::from_vec(vec![0.0; *size]);
                bencher.iter(|| {
                    array
                        .map_to(|x| x.sqrt(), &mut output)
                        .expect("map_to should succeed");
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch operations with buffer reuse
/// Shows cumulative benefit of avoiding allocations in loops
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let size = 10_000;
    let iterations = 10;
    group.throughput(Throughput::Elements((size * iterations) as u64));

    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let array = Array::from_vec(data);

    // Without buffer reuse (allocates each iteration)
    group.bench_function("batch_without_reuse", |bencher| {
        bencher.iter(|| {
            for _ in 0..iterations {
                let result = array.map(|x| x.sqrt() + 1.0);
                black_box(result);
            }
        });
    });

    // With buffer reuse (single allocation)
    group.bench_function("batch_with_reuse", |bencher| {
        bencher.iter(|| {
            let mut output = Array::from_vec(vec![0.0; size]);
            for _ in 0..iterations {
                array
                    .map_to(|x| x.sqrt() + 1.0, &mut output)
                    .expect("map_to should succeed");
                black_box(&output);
            }
        });
    });

    group.finish();
}

/// Benchmark min/max operations
fn bench_minmax_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_operations");

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| (i as f64).sin()).collect();
        let array = Array::from_vec(data);

        group.bench_with_input(
            BenchmarkId::new("min_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.min_optimized());
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("max_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.max_optimized());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark axis operations
fn bench_axis_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("axis_operations");

    for size in [100, 500, 1000].iter() {
        let shape = vec![*size, *size];
        group.throughput(Throughput::Elements((size * size) as u64));

        let data: Vec<f64> = (0..(size * size)).map(|i| i as f64).collect();
        let array = Array::from_vec(data).reshape(&shape);

        // Sum along axis 0
        group.bench_with_input(
            BenchmarkId::new("sum_axis0_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(
                        array
                            .sum_axis_optimized(0)
                            .expect("sum_axis should succeed"),
                    );
                    black_box(result);
                });
            },
        );

        // Sum along axis 1
        group.bench_with_input(
            BenchmarkId::new("sum_axis1_optimized", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(
                        array
                            .sum_axis_optimized(1)
                            .expect("sum_axis should succeed"),
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark allocation patterns - measure overhead of repeated operations
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_overhead");

    let size = 10_000;
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let array = Array::from_vec(data);

    // Many small operations with allocation
    group.bench_function("many_allocations", |bencher| {
        bencher.iter(|| {
            for _ in 0..100 {
                let result = array.sum();
                black_box(result);
            }
        });
    });

    // Many small operations without allocation
    group.bench_function("no_allocations", |bencher| {
        bencher.iter(|| {
            for _ in 0..100 {
                let result = array.sum_optimized();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark SIMD acceleration in optimized operations
fn bench_simd_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_acceleration");

    // Test with different sizes to show SIMD threshold (64 elements)
    for size in [32, 64, 128, 1024, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let array = Array::from_vec(data);

        group.bench_with_input(
            BenchmarkId::new("sum_optimized_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.sum_optimized());
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("variance_optimized_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = black_box(array.variance_optimized());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sum_comparison,
    bench_product_comparison,
    bench_mean_optimized,
    bench_statistical_operations,
    bench_inplace_operations,
    bench_buffer_reuse,
    bench_batch_operations,
    bench_minmax_operations,
    bench_axis_operations,
    bench_allocation_overhead,
    bench_simd_acceleration,
);
criterion_main!(benches);
