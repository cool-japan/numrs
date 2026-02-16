//! Core Operations Benchmarking Suite for NumRS2
//!
//! This benchmark suite provides comprehensive performance measurements for:
//! - Array creation operations
//! - Element-wise operations
//! - Reduction operations
//! - Statistical operations
//! - Memory operations (reshape, transpose)
//!
//! Run with: `cargo bench --bench core_operations_benchmark`

#![allow(deprecated)]
#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::prelude::*;
use std::hint::black_box;

// Sample sizes for benchmarks
const SMALL_SIZE: usize = 1_000;
const MEDIUM_SIZE: usize = 10_000;
const LARGE_SIZE: usize = 100_000;
const XLARGE_SIZE: usize = 1_000_000;

const MATRIX_SMALL: usize = 100;
const MATRIX_MEDIUM: usize = 500;

// ============================================================================
// ARRAY CREATION BENCHMARKS
// ============================================================================

fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Array Creation");

    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, XLARGE_SIZE];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Zeros
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::zeros(&[size])))
        });

        // Ones
        group.bench_with_input(BenchmarkId::new("ones", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::ones(&[size])))
        });

        // Full (custom value)
        group.bench_with_input(BenchmarkId::new("full", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::full(&[size], std::f64::consts::PI)))
        });

        // From Vec
        group.bench_with_input(BenchmarkId::new("from_vec", size), size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|x| x as f64).collect();
            b.iter(|| black_box(Array::from_vec(data.clone())))
        });

        // Linspace (module function)
        if *size <= MEDIUM_SIZE {
            group.bench_with_input(BenchmarkId::new("linspace", size), size, |b, &size| {
                b.iter(|| black_box(linspace(0.0f64, 100.0f64, size)))
            });
        }

        // Arange (module function)
        if *size <= MEDIUM_SIZE {
            group.bench_with_input(BenchmarkId::new("arange", size), size, |b, &size| {
                b.iter(|| black_box(arange(0.0f64, size as f64, 1.0f64)))
            });
        }
    }

    group.finish();
}

fn bench_matrix_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Creation");

    let sizes = [MATRIX_SMALL, MATRIX_MEDIUM];

    for size in sizes.iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        // Zeros 2D
        group.bench_with_input(BenchmarkId::new("zeros_2d", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::zeros(&[size, size])))
        });

        // Ones 2D
        group.bench_with_input(BenchmarkId::new("ones_2d", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::ones(&[size, size])))
        });

        // Identity
        group.bench_with_input(BenchmarkId::new("identity", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::identity(size)))
        });

        // Eye
        group.bench_with_input(BenchmarkId::new("eye", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::eye(size, size, 0)))
        });
    }

    group.finish();
}

// ============================================================================
// ELEMENT-WISE OPERATIONS BENCHMARKS
// ============================================================================

fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Element-wise Operations");

    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Pre-create arrays for benchmarks
        let a_data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.01 + 1.0).collect();
        let b_data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.02 + 0.5).collect();
        let a = Array::from_vec(a_data);
        let b = Array::from_vec(b_data);

        // Addition
        group.bench_with_input(BenchmarkId::new("add", size), size, |bench, _| {
            bench.iter(|| black_box(&a + &b))
        });

        // Subtraction
        group.bench_with_input(BenchmarkId::new("sub", size), size, |bench, _| {
            bench.iter(|| black_box(&a - &b))
        });

        // Multiplication
        group.bench_with_input(BenchmarkId::new("mul", size), size, |bench, _| {
            bench.iter(|| black_box(&a * &b))
        });

        // Division
        group.bench_with_input(BenchmarkId::new("div", size), size, |bench, _| {
            bench.iter(|| black_box(&a / &b))
        });

        // Scalar operations using add_scalar method
        group.bench_with_input(BenchmarkId::new("scalar_add", size), size, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(2.5)))
        });
    }

    group.finish();
}

fn bench_math_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mathematical Functions");

    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Pre-create array with positive values for log/sqrt
        let data: Vec<f64> = (1..=*size).map(|x| x as f64).collect();
        let a = Array::from_vec(data);

        // Square root
        group.bench_with_input(BenchmarkId::new("sqrt", size), size, |bench, _| {
            bench.iter(|| black_box(a.sqrt()))
        });

        // Exponential
        let exp_data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.001).collect();
        let exp_arr = Array::from_vec(exp_data);
        group.bench_with_input(BenchmarkId::new("exp", size), size, |bench, _| {
            bench.iter(|| black_box(exp_arr.exp()))
        });

        // Natural logarithm
        group.bench_with_input(BenchmarkId::new("log", size), size, |bench, _| {
            bench.iter(|| black_box(a.log()))
        });

        // Sine
        let angle_data: Vec<f64> = (0..*size)
            .map(|x| (x as f64) * 0.01 * std::f64::consts::PI)
            .collect();
        let angles = Array::from_vec(angle_data);
        group.bench_with_input(BenchmarkId::new("sin", size), size, |bench, _| {
            bench.iter(|| black_box(angles.sin()))
        });

        // Cosine
        group.bench_with_input(BenchmarkId::new("cos", size), size, |bench, _| {
            bench.iter(|| black_box(angles.cos()))
        });

        // Absolute value
        let mixed_data: Vec<f64> = (0..*size)
            .map(|x| (x as f64) - (*size as f64) / 2.0)
            .collect();
        let mixed = Array::from_vec(mixed_data);
        group.bench_with_input(BenchmarkId::new("abs", size), size, |bench, _| {
            bench.iter(|| black_box(mixed.abs()))
        });
    }

    group.finish();
}

// ============================================================================
// REDUCTION OPERATIONS BENCHMARKS
// ============================================================================

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Reduction Operations");

    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, XLARGE_SIZE];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.001).collect();
        let a = Array::from_vec(data);

        // Sum
        group.bench_with_input(BenchmarkId::new("sum", size), size, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });

        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), size, |bench, _| {
            bench.iter(|| black_box(a.mean()))
        });

        // Variance
        group.bench_with_input(BenchmarkId::new("var", size), size, |bench, _| {
            bench.iter(|| black_box(a.var()))
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), size, |bench, _| {
            bench.iter(|| black_box(a.std()))
        });

        // Min
        group.bench_with_input(BenchmarkId::new("min", size), size, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });

        // Max
        group.bench_with_input(BenchmarkId::new("max", size), size, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
    }

    group.finish();
}

// ============================================================================
// ARRAY MANIPULATION BENCHMARKS
// ============================================================================

fn bench_array_manipulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Array Manipulation");

    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|x| x as f64).collect();
        let a = Array::from_vec(data);

        // Find a valid 2D reshape size
        let rows = (*size as f64).sqrt() as usize;
        let cols = size / rows;
        if rows * cols == *size && rows > 0 && cols > 0 {
            // Reshape
            group.bench_with_input(BenchmarkId::new("reshape", size), size, |bench, _| {
                bench.iter(|| black_box(a.reshape(&[rows, cols])))
            });

            // Transpose (2D)
            let matrix = a.reshape(&[rows, cols]);
            group.bench_with_input(BenchmarkId::new("transpose", size), size, |bench, _| {
                bench.iter(|| black_box(matrix.transpose()))
            });

            // Flatten
            group.bench_with_input(BenchmarkId::new("flatten", size), size, |bench, _| {
                bench.iter(|| black_box(matrix.flatten(None)))
            });
        }

        // Clone
        group.bench_with_input(BenchmarkId::new("clone", size), size, |bench, _| {
            bench.iter(|| black_box(a.clone()))
        });
    }

    group.finish();
}

// ============================================================================
// LINEAR ALGEBRA BENCHMARKS
// ============================================================================

fn bench_linear_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear Algebra");
    group.sample_size(20); // Reduce sample size for expensive operations

    let sizes = [50, 100, 200];

    for size in sizes.iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        // Create matrix
        let data: Vec<f64> = (0..elements)
            .map(|i| ((i as f64 * 0.7) % 10.0) + 0.1)
            .collect();
        let a = Array::from_vec(data).reshape(&[*size, *size]);

        // Create vectors for dot product
        let vec_data: Vec<f64> = (0..*size).map(|i| (i as f64) + 1.0).collect();
        let v = Array::from_vec(vec_data);

        // Matrix-Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bench, _| {
            bench.iter(|| black_box(a.matmul(&a)))
        });

        // Dot product (vector)
        group.bench_with_input(BenchmarkId::new("dot", size), size, |bench, _| {
            bench.iter(|| black_box(v.dot(&v)))
        });

        // Trace - use standalone function from linalg
        group.bench_with_input(BenchmarkId::new("trace", size), size, |bench, _| {
            bench.iter(|| black_box(numrs2::linalg::vector_ops::trace(&a)))
        });
    }

    group.finish();
}

// ============================================================================
// POLYNOMIAL OPERATIONS BENCHMARKS
// ============================================================================

fn bench_polynomial_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial Operations");

    // Polynomial evaluation
    let degrees = [5, 10, 20, 50];

    for degree in degrees.iter() {
        // Create polynomial coefficients
        let coeffs: Vec<f64> = (0..=*degree).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let poly = Polynomial::new(coeffs);

        // Evaluation at single point
        group.bench_with_input(
            BenchmarkId::new("evaluate_single", degree),
            degree,
            |bench, _| bench.iter(|| black_box(poly.evaluate(2.5))),
        );

        // Derivative
        group.bench_with_input(
            BenchmarkId::new("derivative", degree),
            degree,
            |bench, _| bench.iter(|| black_box(poly.derivative())),
        );

        // Integral
        group.bench_with_input(BenchmarkId::new("integral", degree), degree, |bench, _| {
            bench.iter(|| black_box(poly.integral()))
        });
    }

    // Polynomial arithmetic
    let sizes = [5, 10, 20];
    for size in sizes.iter() {
        let coeffs1: Vec<f64> = (0..*size).map(|i| (i as f64) + 1.0).collect();
        let coeffs2: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.5 + 0.5).collect();
        let p1 = Polynomial::new(coeffs1);
        let p2 = Polynomial::new(coeffs2);

        // Multiplication
        group.bench_with_input(BenchmarkId::new("multiply", size), size, |bench, _| {
            bench.iter(|| black_box(p1.clone() * p2.clone()))
        });

        // Addition
        group.bench_with_input(BenchmarkId::new("add", size), size, |bench, _| {
            bench.iter(|| black_box(p1.clone() + p2.clone()))
        });
    }

    group.finish();
}

// ============================================================================
// SCALABILITY BENCHMARKS
// ============================================================================

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalability");
    group.sample_size(10);

    // Test how operations scale with size
    let sizes: Vec<usize> = vec![1000, 5000, 10000, 50000, 100000, 500000, 1000000];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.001).collect();
        let a = Array::from_vec(data);

        // Sum scalability
        group.bench_with_input(BenchmarkId::new("sum_scale", size), size, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });

        // Element-wise ops scalability
        let b = a.add_scalar(1.0);
        group.bench_with_input(BenchmarkId::new("add_scale", size), size, |bench, _| {
            bench.iter(|| black_box(&a + &b))
        });

        // Memory copy scalability
        group.bench_with_input(BenchmarkId::new("clone_scale", size), size, |bench, _| {
            bench.iter(|| black_box(a.clone()))
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

criterion_group!(
    benches,
    bench_array_creation,
    bench_matrix_creation,
    bench_elementwise_ops,
    bench_math_functions,
    bench_reductions,
    bench_array_manipulation,
    bench_linear_algebra,
    bench_polynomial_ops,
    bench_scalability,
);

criterion_main!(benches);
