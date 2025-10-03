//! Production readiness benchmarks for NumRS2
//!
//! This benchmark suite tests critical performance characteristics that are
//! essential for production use, including memory efficiency, SIMD optimization,
//! parallel processing, and numerical stability.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::parallel::parallel_algorithms::ParallelArrayOps;
use numrs2::prelude::*;
use numrs2::stats::Statistics;
use std::hint::black_box;
use std::time::Duration;

/// Production readiness benchmarks
pub struct ProductionBenchmarks;

impl ProductionBenchmarks {
    /// Benchmark matrix multiplication scaling with size
    pub fn matrix_multiplication_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("matrix_multiplication_scaling");
        group.measurement_time(Duration::from_secs(5));

        for size in [64, 128, 256].iter() {
            group.bench_with_input(
                BenchmarkId::new("standard_matmul", size),
                size,
                |bench, &size| {
                    let a: Array<f64> = Array::ones(&[size, size]);
                    let b: Array<f64> = Array::ones(&[size, size]);
                    bench.iter(|| black_box(a.matmul(&b).unwrap()));
                },
            );
        }
        group.finish();
    }

    /// Benchmark element-wise operations performance
    pub fn elementwise_operations_performance(c: &mut Criterion) {
        let mut group = c.benchmark_group("elementwise_operations");
        group.measurement_time(Duration::from_secs(3));

        for size in [1000, 10000, 100000].iter() {
            group.bench_with_input(
                BenchmarkId::new("add_operation", size),
                size,
                |bench, &size| {
                    let a: Array<f32> = Array::ones(&[size]);
                    let b: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(a.add(&b)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("multiply_operation", size),
                size,
                |bench, &size| {
                    let a: Array<f32> = Array::ones(&[size]);
                    let b: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(a.multiply(&b)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("dot_product", size),
                size,
                |bench, &size| {
                    let a: Array<f32> = Array::ones(&[size]);
                    let b: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(a.dot(&b).unwrap()));
                },
            );
        }
        group.finish();
    }

    /// Benchmark parallel processing efficiency
    pub fn parallel_processing_efficiency(c: &mut Criterion) {
        let mut group = c.benchmark_group("parallel_processing");
        group.measurement_time(Duration::from_secs(3));

        for size in [10000, 100000].iter() {
            group.bench_with_input(
                BenchmarkId::new("sequential_sum", size),
                size,
                |bench, &size| {
                    let data: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(data.sum()));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("parallel_sum", size),
                size,
                |bench, &size| {
                    let data: Array<f32> = Array::ones(&[size]);
                    let parallel_ops = ParallelArrayOps::new(Default::default()).unwrap();
                    bench.iter(|| {
                        black_box(
                            parallel_ops
                                .parallel_reduce(&data.to_vec(), 0.0f32, |a, b| a + b)
                                .unwrap(),
                        )
                    });
                },
            );
        }
        group.finish();
    }

    /// Benchmark array creation and memory allocation
    pub fn array_creation_performance(c: &mut Criterion) {
        let mut group = c.benchmark_group("array_creation");
        group.measurement_time(Duration::from_secs(3));

        for size in [1000, 10000, 100000].iter() {
            group.bench_with_input(
                BenchmarkId::new("zeros_creation", size),
                size,
                |bench, &size| {
                    bench.iter(|| black_box(Array::<f32>::zeros(&[size])));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("ones_creation", size),
                size,
                |bench, &size| {
                    bench.iter(|| black_box(Array::<f32>::ones(&[size])));
                },
            );

            group.bench_with_input(BenchmarkId::new("from_vec", size), size, |bench, &size| {
                let data = vec![1.0f32; size];
                bench.iter(|| black_box(Array::from_vec(data.clone())));
            });
        }
        group.finish();
    }

    /// Benchmark linear algebra operations
    pub fn linalg_operations_performance(c: &mut Criterion) {
        let mut group = c.benchmark_group("linalg_operations");
        group.measurement_time(Duration::from_secs(5));

        for size in [32, 64, 128].iter() {
            group.bench_with_input(
                BenchmarkId::new("matrix_inverse", size),
                size,
                |bench, &size| {
                    // Create a well-conditioned matrix
                    let mut data = vec![0.0f32; size * size];
                    for i in 0..size {
                        for j in 0..size {
                            if i == j {
                                data[i * size + j] = 2.0;
                            } else if (i as isize - j as isize).abs() == 1 {
                                data[i * size + j] = -1.0;
                            }
                        }
                    }
                    let matrix = Array::from_vec(data).reshape(&[size, size]);

                    bench.iter(|| black_box(matrix.inv().unwrap()));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("solve_linear_system", size),
                size,
                |bench, &size| {
                    // Create a well-conditioned matrix
                    let mut a_data = vec![0.0f32; size * size];
                    for i in 0..size {
                        for j in 0..size {
                            if i == j {
                                a_data[i * size + j] = 2.0;
                            } else if (i as isize - j as isize).abs() == 1 {
                                a_data[i * size + j] = -1.0;
                            }
                        }
                    }
                    let a = Array::from_vec(a_data).reshape(&[size, size]);
                    let b: Array<f64> = Array::ones(&[size]);

                    // bench.iter(|| black_box(numrs2::linalg::solve(&a, &b).unwrap())); // solve requires lapack feature
                },
            );
        }
        group.finish();
    }

    /// Benchmark memory access patterns
    pub fn memory_access_patterns(c: &mut Criterion) {
        let mut group = c.benchmark_group("memory_access");
        group.measurement_time(Duration::from_secs(3));

        for size in [100, 500, 1000].iter() {
            // Row-major access (cache-friendly)
            group.bench_with_input(
                BenchmarkId::new("row_major_access", size),
                size,
                |bench, &size| {
                    let matrix: Array<f32> = Array::ones(&[size, size]);
                    bench.iter(|| {
                        let mut sum = 0.0f32;
                        for i in 0..size {
                            for j in 0..size {
                                sum += matrix.get(&[i, j]).unwrap();
                            }
                        }
                        black_box(sum)
                    });
                },
            );

            // Column-major access (cache-unfriendly)
            group.bench_with_input(
                BenchmarkId::new("column_major_access", size),
                size,
                |bench, &size| {
                    let matrix: Array<f32> = Array::ones(&[size, size]);
                    bench.iter(|| {
                        let mut sum = 0.0f32;
                        for j in 0..size {
                            for i in 0..size {
                                sum += matrix.get(&[i, j]).unwrap();
                            }
                        }
                        black_box(sum)
                    });
                },
            );
        }
        group.finish();
    }

    /// Benchmark statistical operations
    pub fn statistical_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("statistical_operations");
        group.measurement_time(Duration::from_secs(3));

        for size in [10000, 100000].iter() {
            group.bench_with_input(
                BenchmarkId::new("mean_calculation", size),
                size,
                |bench, &size| {
                    let data: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(data.mean()));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("variance_calculation", size),
                size,
                |bench, &size| {
                    let data: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(data.var()));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("sum_calculation", size),
                size,
                |bench, &size| {
                    let data: Array<f32> = Array::ones(&[size]);
                    bench.iter(|| black_box(data.sum()));
                },
            );
        }
        group.finish();
    }

    /// Benchmark broadcasting operations
    pub fn broadcasting_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("broadcasting");
        group.measurement_time(Duration::from_secs(3));

        for size in [100, 500, 1000].iter() {
            // Broadcasting operations commented out - not yet implemented
            // group.bench_with_input(
            //     BenchmarkId::new("matrix_scalar_add", size),
            //     size,
            //     |bench, &size| {
            //         let matrix: Array<f32> = Array::ones(&[size, size]);
            //         let scalar = 2.0f32;
            //         bench.iter(|| black_box(&matrix + scalar));
            //     },
            // );

            // group.bench_with_input(
            //     BenchmarkId::new("matrix_vector_add", size),
            //     size,
            //     |bench, &size| {
            //         let matrix: Array<f32> = Array::ones(&[size, size]);
            //         let vector: Array<f32> = Array::ones(&[size]);
            //         bench.iter(|| black_box(&matrix + &vector));
            //     },
            // );
        }
        group.finish();
    }

    /// Benchmark shape manipulation operations
    pub fn shape_manipulation_performance(c: &mut Criterion) {
        let mut group = c.benchmark_group("shape_manipulation");
        group.measurement_time(Duration::from_secs(3));

        for size in [1000, 10000].iter() {
            group.bench_with_input(
                BenchmarkId::new("reshape_operation", size),
                size,
                |bench, &size| {
                    let data: Array<f32> = Array::ones(&[size]);
                    let new_shape = if size == 1000 {
                        vec![20, 50]
                    } else {
                        vec![100, 100]
                    };
                    bench.iter(|| black_box(data.reshape(&new_shape)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("transpose_operation", size),
                size,
                |bench, &size| {
                    let sqrt_size = (size as f64).sqrt() as usize;
                    let matrix: Array<f32> = Array::ones(&[sqrt_size, sqrt_size]);
                    bench.iter(|| black_box(matrix.transpose()));
                },
            );
        }
        group.finish();
    }
}

// Define benchmark groups
criterion_group!(
    production_benches,
    ProductionBenchmarks::matrix_multiplication_scaling,
    ProductionBenchmarks::elementwise_operations_performance,
    ProductionBenchmarks::parallel_processing_efficiency,
    ProductionBenchmarks::array_creation_performance,
    ProductionBenchmarks::linalg_operations_performance,
    ProductionBenchmarks::memory_access_patterns,
    ProductionBenchmarks::statistical_operations,
    ProductionBenchmarks::broadcasting_operations,
    ProductionBenchmarks::shape_manipulation_performance
);

criterion_main!(production_benches);
