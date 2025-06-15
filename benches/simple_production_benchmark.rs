//! Simple production readiness benchmarks for NumRS2

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use numrs2::prelude::*;
use std::time::Duration;

/// Test basic array operations performance
pub fn basic_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations");
    group.measurement_time(Duration::from_secs(3));
    
    for size in [1000usize, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("array_addition", size),
            size,
            |bench, &size| {
                let a: Array<f32> = Array::ones(&[size]);
                let b: Array<f32> = Array::ones(&[size]);
                bench.iter(|| {
                    black_box(a.add(&b))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("array_multiplication", size),
            size,
            |bench, &size| {
                let a: Array<f32> = Array::ones(&[size]);
                let b: Array<f32> = Array::ones(&[size]);
                bench.iter(|| {
                    black_box(a.multiply(&b))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            size,
            |bench, &size| {
                let a: Array<f32> = Array::ones(&[size]);
                let b: Array<f32> = Array::ones(&[size]);
                bench.iter(|| {
                    black_box(a.dot(&b).unwrap())
                });
            },
        );
    }
    group.finish();
}

/// Test matrix operations performance
pub fn matrix_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    group.measurement_time(Duration::from_secs(5));
    
    for size in [64usize, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::new("matrix_multiplication", size),
            size,
            |bench, &size| {
                let a: Array<f32> = Array::ones(&[size, size]);
                let b: Array<f32> = Array::ones(&[size, size]);
                bench.iter(|| {
                    black_box(a.matmul(&b).unwrap())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_transpose", size),
            size,
            |bench, &size| {
                let matrix: Array<f32> = Array::ones(&[size, size]);
                bench.iter(|| {
                    black_box(matrix.transpose())
                });
            },
        );
    }
    group.finish();
}

/// Test memory allocation performance
pub fn memory_allocation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.measurement_time(Duration::from_secs(3));
    
    for size in [1000usize, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("zeros_creation", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    black_box(Array::<f32>::zeros(&[size]))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("ones_creation", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    black_box(Array::<f32>::ones(&[size]))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("from_vec", size),
            size,
            |bench, &size| {
                let data = vec![1.0f32; size];
                bench.iter(|| {
                    black_box(Array::from_vec(data.clone()))
                });
            },
        );
    }
    group.finish();
}

/// Test statistical operations performance
pub fn statistical_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_operations");
    group.measurement_time(Duration::from_secs(3));
    
    for size in [10000usize, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("sum_calculation", size),
            size,
            |bench, &size| {
                let data: Array<f32> = Array::ones(&[size]);
                bench.iter(|| {
                    black_box(data.sum())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mean_calculation", size),
            size,
            |bench, &size| {
                let data: Array<f32> = Array::ones(&[size]);
                bench.iter(|| {
                    black_box(data.mean())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("variance_calculation", size),
            size,
            |bench, &size| {
                let data: Array<f32> = Array::ones(&[size]);
                bench.iter(|| {
                    black_box(data.var())
                });
            },
        );
    }
    group.finish();
}

/// Test linear algebra performance
pub fn linear_algebra_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_algebra");
    group.measurement_time(Duration::from_secs(5));
    
    for size in [32usize, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::new("matrix_inverse", size),
            size,
            |bench, &size| {
                // Create a well-conditioned tridiagonal matrix
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
                
                bench.iter(|| {
                    black_box(matrix.inv().unwrap())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("solve_linear_system", size),
            size,
            |bench, &size| {
                // Create a well-conditioned tridiagonal matrix
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
                let b: Array<f32> = Array::ones(&[size]);
                
                bench.iter(|| {
                    black_box(numrs2::linalg::solve(&a, &b).unwrap())
                });
            },
        );
    }
    group.finish();
}

// Define benchmark groups
criterion_group!(
    production_benches,
    basic_operations_benchmark,
    matrix_operations_benchmark,
    memory_allocation_benchmark,
    statistical_operations_benchmark,
    linear_algebra_benchmark
);

criterion_main!(production_benches);