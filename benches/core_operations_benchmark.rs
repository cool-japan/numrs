//! Core array operations benchmarks for NumRS2
//!
//! This benchmark suite focuses on the fundamental array operations that form
//! the backbone of numerical computing, with detailed performance profiling.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::array_ops;
use numrs2::linalg::LinAlg;
use numrs2::memory_optimize::cache_layout::{
    calculate_optimal_block_size, optimize_layout, LayoutStrategy,
};
use numrs2::simd::SimdOps;
use numrs2::stats::Statistics;
use std::time::Duration;

/// Benchmark array indexing operations
fn bench_array_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_indexing");

    let sizes = vec![100, 1000, 10000];

    for size in sizes {
        let data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let arr = Array::from_vec(data).reshape(&[size, size]);

        // Sequential access pattern
        group.bench_with_input(
            BenchmarkId::new("sequential_access", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..size {
                        for j in 0..size {
                            sum += arr.get(&[i, j]).unwrap();
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Random access pattern
        group.bench_with_input(
            BenchmarkId::new("random_access", size),
            &size,
            |b, &size| {
                let indices: Vec<(usize, usize)> =
                    (0..1000).map(|i| (i % size, (i * 7) % size)).collect();
                b.iter(|| {
                    let mut sum = 0.0;
                    for &(i, j) in &indices {
                        sum += arr.get(&[i, j]).unwrap();
                    }
                    black_box(sum)
                })
            },
        );

        // Strided access pattern
        group.bench_with_input(
            BenchmarkId::new("strided_access", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut sum = 0.0;
                    let stride = 2;
                    for i in (0..size).step_by(stride) {
                        for j in (0..size).step_by(stride) {
                            sum += arr.get(&[i, j]).unwrap();
                        }
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark broadcasting operations
fn bench_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    let sizes = vec![100, 500, 1000];

    for size in sizes {
        // Vector-matrix broadcasting
        let matrix_data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let vector_data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        let matrix = Array::from_vec(matrix_data).reshape(&[size, size]);
        let vector = Array::from_vec(vector_data);

        group.bench_with_input(
            BenchmarkId::new("vector_matrix_add", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix + &vector;
                    black_box(result)
                })
            },
        );

        // Scalar-array broadcasting
        group.bench_with_input(BenchmarkId::new("scalar_array_add", size), &size, |b, _| {
            b.iter(|| {
                let result = matrix.add_scalar(2.5);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark element-wise operations with different data types
fn bench_element_wise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_operations");
    group.throughput(Throughput::Elements(100000));

    let size = 100000;

    // f32 operations
    let data_f32: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let arr_f32_1 = Array::from_vec(data_f32.clone());
    let arr_f32_2 = Array::from_vec(data_f32);

    group.bench_function("f32_add", |b| {
        b.iter(|| {
            let result = &arr_f32_1 + &arr_f32_2;
            black_box(result)
        })
    });

    group.bench_function("f32_multiply", |b| {
        b.iter(|| {
            let result = &arr_f32_1 * &arr_f32_2;
            black_box(result)
        })
    });

    // f64 operations
    let data_f64: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    let arr_f64_1 = Array::from_vec(data_f64.clone());
    let arr_f64_2 = Array::from_vec(data_f64);

    group.bench_function("f64_add", |b| {
        b.iter(|| {
            let result = &arr_f64_1 + &arr_f64_2;
            black_box(result)
        })
    });

    group.bench_function("f64_multiply", |b| {
        b.iter(|| {
            let result = &arr_f64_1 * &arr_f64_2;
            black_box(result)
        })
    });

    // i32 operations
    let data_i32: Vec<i32> = (0..size).map(|i| i as i32).collect();
    let arr_i32_1 = Array::from_vec(data_i32.clone());
    let arr_i32_2 = Array::from_vec(data_i32);

    group.bench_function("i32_add", |b| {
        b.iter(|| {
            let result = array_ops::add(&arr_i32_1, &arr_i32_2).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark memory layout optimizations
fn bench_cache_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_optimization");

    let sizes = vec![1000, 5000, 10000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // Benchmark layout optimization strategies
        group.bench_with_input(BenchmarkId::new("row_major_layout", size), &size, |b, _| {
            b.iter(|| {
                let mut arr_data_copy = data.clone();
                optimize_layout(&mut arr_data_copy, LayoutStrategy::RowMajor);
                black_box(arr_data_copy)
            })
        });

        group.bench_with_input(BenchmarkId::new("morton_layout", size), &size, |b, _| {
            b.iter(|| {
                let mut data_copy = data.clone();
                optimize_layout(&mut data_copy, LayoutStrategy::Morton);
                black_box(data_copy)
            })
        });

        group.bench_with_input(BenchmarkId::new("blocked_layout", size), &size, |b, _| {
            let block_size = calculate_optimal_block_size::<f64>();
            b.iter(|| {
                let mut data_copy = data.clone();
                optimize_layout(&mut data_copy, LayoutStrategy::Blocked(block_size));
                black_box(data_copy)
            })
        });
    }

    group.finish();
}

/// Benchmark reduction operations
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let arr = Array::from_vec(data);

        group.throughput(Throughput::Elements(size as u64));

        // Sum reduction
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.to_vec().iter().sum::<f64>();
                black_box(result)
            })
        });

        // Product reduction
        group.bench_with_input(BenchmarkId::new("product", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.to_vec().iter().product::<f64>();
                black_box(result)
            })
        });

        // Min/Max reductions
        group.bench_with_input(BenchmarkId::new("min", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.min();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("max", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.max();
                black_box(result)
            })
        });

        // Cumulative operations
        group.bench_with_input(BenchmarkId::new("cumsum", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::cumsum(&arr, None).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark axis-based operations
fn bench_axis_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("axis_operations");

    let sizes = vec![100, 500, 1000];

    for size in sizes {
        let data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let arr = Array::from_vec(data).reshape(&[size, size]);

        // Statistics operations
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

        group.bench_with_input(BenchmarkId::new("var", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.var();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark SIMD-optimized operations
fn bench_simd_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_performance");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        let data1: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
        let data2: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.1).collect();
        let arr1 = Array::from_vec(data1);
        let arr2 = Array::from_vec(data2);

        group.throughput(Throughput::Elements(size as u64));

        // SIMD addition
        group.bench_with_input(BenchmarkId::new("simd_add_f64", size), &size, |b, _| {
            b.iter(|| {
                let result = arr1.simd_add(&arr2).unwrap();
                black_box(result)
            })
        });

        // SIMD multiplication
        group.bench_with_input(
            BenchmarkId::new("simd_multiply_f64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = arr1.simd_mul(&arr2).unwrap();
                    black_box(result)
                })
            },
        );

        // SIMD dot product
        group.bench_with_input(BenchmarkId::new("simd_dot_f64", size), &size, |b, _| {
            b.iter(|| {
                let result = arr1.simd_dot(&arr2).unwrap();
                black_box(result)
            })
        });

        // FMA operations
        group.bench_with_input(BenchmarkId::new("simd_fma_f64", size), &size, |b, _| {
            b.iter(|| {
                let result = arr1.simd_fma(&arr2, &arr1).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark different array shapes and strides
fn bench_array_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_shapes");

    let total_elements = 100000;

    // Different shapes with same total elements
    let shapes = vec![
        vec![total_elements],  // 1D
        vec![316, 316],        // 2D square
        vec![100, 1000],       // 2D rectangular
        vec![100, 100, 10],    // 3D
        vec![50, 20, 100],     // 3D rectangular
        vec![10, 10, 10, 100], // 4D
    ];

    for shape in shapes {
        let data: Vec<f64> = (0..total_elements).map(|i| i as f64).collect();
        let arr = Array::from_vec(data).reshape(&shape);

        let shape_name = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");

        // Benchmark sum across different shapes
        group.bench_with_input(BenchmarkId::new("sum", &shape_name), &shape_name, |b, _| {
            b.iter(|| {
                let result = arr.to_vec().iter().sum::<f64>();
                black_box(result)
            })
        });

        // Benchmark element access pattern
        group.bench_with_input(
            BenchmarkId::new("element_access", &shape_name),
            &shape_name,
            |b, _| {
                let indices = generate_indices_for_shape(&shape, 1000);
                b.iter(|| {
                    let mut sum = 0.0;
                    for idx in &indices {
                        sum += arr.get(idx).unwrap();
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Helper function to generate valid indices for a given shape
fn generate_indices_for_shape(shape: &[usize], count: usize) -> Vec<Vec<usize>> {
    let mut indices = Vec::new();
    for i in 0..count {
        let mut idx = Vec::new();
        let mut remaining = i;
        for &dim_size in shape.iter().rev() {
            idx.insert(0, remaining % dim_size);
            remaining /= dim_size;
        }
        indices.push(idx);
    }
    indices
}

criterion_group! {
    name = core_operations_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(15))
        .sample_size(100);
    targets =
        bench_array_indexing,
        bench_broadcasting,
        bench_element_wise_operations,
        bench_cache_optimization,
        bench_reductions,
        bench_axis_operations,
        bench_simd_performance,
        bench_array_shapes
}

criterion_main!(core_operations_benches);
