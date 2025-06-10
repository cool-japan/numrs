//! Comprehensive benchmarks comparing NumRS2 performance against NumPy
//!
//! This benchmark suite provides detailed performance comparisons between NumRS2 and NumPy
//! for core array operations, demonstrating the performance characteristics of the Rust
//! implementation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::array::Array;
use numrs2::array_ops;
use numrs2::blas;
use numrs2::linalg;
use numrs2::math;
use numrs2::prelude::*;
use numrs2::random::distributions::*;
use numrs2::random::generator::RandomState;
use numrs2::stats;
use numrs2::unique::unique;
use std::time::Duration;

/// Benchmark configuration for array operations
struct BenchmarkConfig {
    pub name: &'static str,
    pub sizes: Vec<usize>,
    pub iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default",
            sizes: vec![100, 1000, 10000, 100000],
            iterations: 100,
        }
    }
}

/// Generate test data for benchmarks
fn generate_test_data_f64(size: usize) -> Vec<f64> {
    let mut rng = RandomState::new();
    (0..size).map(|_| rng.uniform(0.0, 1.0)).collect()
}

fn generate_test_data_i32(size: usize) -> Vec<i32> {
    let mut rng = RandomState::new();
    (0..size).map(|_| rng.uniform_int(0, 1000)).collect()
}

/// Benchmark basic array creation operations
fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        // Benchmark zeros creation
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &size| {
            b.iter(|| {
                let arr = Array::<f64>::zeros(&[size]);
                black_box(arr)
            })
        });

        // Benchmark ones creation
        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &size| {
            b.iter(|| {
                let arr = Array::<f64>::ones(&[size]);
                black_box(arr)
            })
        });

        // Benchmark from_vec creation
        group.bench_with_input(BenchmarkId::new("from_vec", size), &size, |b, &size| {
            let data = generate_test_data_f64(size);
            b.iter(|| {
                let arr = Array::from_vec(data.clone());
                black_box(arr)
            })
        });

        // Benchmark arange creation
        group.bench_with_input(BenchmarkId::new("arange", size), &size, |b, &size| {
            b.iter(|| {
                let arr = array_ops::arange(0.0, size as f64, 1.0).unwrap();
                black_box(arr)
            })
        });
    }

    group.finish();
}

/// Benchmark basic arithmetic operations
fn bench_arithmetic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic_operations");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data1 = generate_test_data_f64(size);
        let data2 = generate_test_data_f64(size);
        let arr1 = Array::from_vec(data1);
        let arr2 = Array::from_vec(data2);

        // Benchmark addition
        group.bench_with_input(BenchmarkId::new("add", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::add(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });

        // Benchmark subtraction
        group.bench_with_input(BenchmarkId::new("subtract", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::subtract(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });

        // Benchmark multiplication
        group.bench_with_input(BenchmarkId::new("multiply", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::multiply(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });

        // Benchmark division
        group.bench_with_input(BenchmarkId::new("divide", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::divide(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });

        // Benchmark scalar operations
        group.bench_with_input(BenchmarkId::new("add_scalar", size), &size, |b, _| {
            b.iter(|| {
                let result = arr1.add_scalar(2.5);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark mathematical functions
fn bench_mathematical_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("mathematical_functions");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data = generate_test_data_f64(size);
        let arr = Array::from_vec(data);

        // Benchmark sqrt
        group.bench_with_input(BenchmarkId::new("sqrt", size), &size, |b, _| {
            b.iter(|| {
                let result = math::sqrt(&arr).unwrap();
                black_box(result)
            })
        });

        // Benchmark exp
        group.bench_with_input(BenchmarkId::new("exp", size), &size, |b, _| {
            b.iter(|| {
                let result = math::exp(&arr).unwrap();
                black_box(result)
            })
        });

        // Benchmark log
        group.bench_with_input(BenchmarkId::new("log", size), &size, |b, _| {
            // Use positive values for log
            let positive_data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
            let positive_arr = Array::from_vec(positive_data);
            b.iter(|| {
                let result = math::log(&positive_arr).unwrap();
                black_box(result)
            })
        });

        // Benchmark sin
        group.bench_with_input(BenchmarkId::new("sin", size), &size, |b, _| {
            b.iter(|| {
                let result = math::sin(&arr).unwrap();
                black_box(result)
            })
        });

        // Benchmark cos
        group.bench_with_input(BenchmarkId::new("cos", size), &size, |b, _| {
            b.iter(|| {
                let result = math::cos(&arr).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark array manipulation operations
fn bench_array_manipulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_manipulation");

    let sizes = vec![100, 1000, 10000];

    for size in sizes {
        let data = generate_test_data_f64(size * size);
        let arr = Array::from_vec(data).reshape(&[size, size]);

        // Benchmark transpose
        group.bench_with_input(BenchmarkId::new("transpose", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.transpose();
                black_box(result)
            })
        });

        // Benchmark reshape
        group.bench_with_input(BenchmarkId::new("reshape", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.reshape(&[size / 2, size * 2]);
                black_box(result)
            })
        });

        // Benchmark flatten
        group.bench_with_input(BenchmarkId::new("flatten", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.flatten(None);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark statistical operations
fn bench_statistical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_operations");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data = generate_test_data_f64(size);
        let arr = Array::from_vec(data);

        // Benchmark sum
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::sum(&arr, None).unwrap();
                black_box(result)
            })
        });

        // Benchmark mean
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::mean(&arr, None).unwrap();
                black_box(result)
            })
        });

        // Benchmark std
        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::std(&arr, None, 0).unwrap();
                black_box(result)
            })
        });

        // Benchmark var
        group.bench_with_input(BenchmarkId::new("var", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::var(&arr, None, 0).unwrap();
                black_box(result)
            })
        });

        // Benchmark min/max
        group.bench_with_input(BenchmarkId::new("min", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::min(&arr, None).unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("max", size), &size, |b, _| {
            b.iter(|| {
                let result = stats::max(&arr, None).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark linear algebra operations
fn bench_linear_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_algebra");
    group.sample_size(20); // Fewer samples for expensive operations

    let sizes = vec![50, 100, 200, 500];

    for size in sizes {
        let data1 = generate_test_data_f64(size * size);
        let data2 = generate_test_data_f64(size * size);
        let mat1 = Array::from_vec(data1).reshape(&[size, size]);
        let mat2 = Array::from_vec(data2).reshape(&[size, size]);

        // Benchmark matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, _| {
            b.iter(|| {
                let result = linalg::matmul(&mat1, &mat2).unwrap();
                black_box(result)
            })
        });

        // Benchmark matrix inversion (for smaller matrices)
        if size <= 200 {
            group.bench_with_input(BenchmarkId::new("inv", size), &size, |b, _| {
                b.iter(|| {
                    let result = linalg::inv(&mat1).unwrap();
                    black_box(result)
                })
            });
        }

        // Benchmark determinant
        if size <= 200 {
            group.bench_with_input(BenchmarkId::new("det", size), &size, |b, _| {
                b.iter(|| {
                    let result = linalg::det(&mat1).unwrap();
                    black_box(result)
                })
            });
        }

        // Benchmark eigenvalues (for smaller matrices)
        if size <= 100 {
            group.bench_with_input(BenchmarkId::new("eig", size), &size, |b, _| {
                b.iter(|| {
                    let result = eig_general(&mat1).unwrap();
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

/// Benchmark sorting and searching operations
fn bench_sorting_searching(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_searching");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data = generate_test_data_i32(size);
        let arr = Array::from_vec(data);

        // Benchmark sort
        group.bench_with_input(BenchmarkId::new("sort", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::sort(&arr, None).unwrap();
                black_box(result)
            })
        });

        // Benchmark argsort
        group.bench_with_input(BenchmarkId::new("argsort", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::argsort(&arr, None).unwrap();
                black_box(result)
            })
        });

        // Benchmark unique
        group.bench_with_input(BenchmarkId::new("unique", size), &size, |b, _| {
            b.iter(|| {
                let result = unique(&arr, None).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark memory layout optimization
fn bench_memory_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_optimization");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data = generate_test_data_f64(size);
        let arr = Array::from_vec(data);

        // Benchmark contiguous check
        group.bench_with_input(BenchmarkId::new("is_c_contiguous", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.is_c_contiguous();
                black_box(result)
            })
        });

        // Benchmark layout conversion
        group.bench_with_input(BenchmarkId::new("to_c_layout", size), &size, |b, _| {
            b.iter(|| {
                let result = arr.to_c_layout();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark SIMD operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        let data1 = generate_test_data_f64(size);
        let data2 = generate_test_data_f64(size);
        let arr1 = Array::from_vec(data1);
        let arr2 = Array::from_vec(data2);

        // Benchmark SIMD addition (if available)
        group.bench_with_input(BenchmarkId::new("simd_add", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::add(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });

        // Benchmark SIMD multiplication
        group.bench_with_input(BenchmarkId::new("simd_multiply", size), &size, |b, _| {
            b.iter(|| {
                let result = array_ops::multiply(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });

        // Benchmark dot product
        group.bench_with_input(BenchmarkId::new("dot_product", size), &size, |b, _| {
            b.iter(|| {
                let result = blas::dot(&arr1, &arr2).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Comprehensive benchmark suite
criterion_group! {
    name = numpy_comparison_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets =
        bench_array_creation,
        bench_arithmetic_operations,
        bench_mathematical_functions,
        bench_array_manipulation,
        bench_statistical_operations,
        bench_linear_algebra,
        bench_sorting_searching,
        bench_memory_optimization,
        bench_simd_operations
}

criterion_main!(numpy_comparison_benches);
