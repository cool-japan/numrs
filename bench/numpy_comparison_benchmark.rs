//! NumPy Comparison Benchmark Suite for NumRS2
//!
//! This benchmark suite provides performance measurements that can be directly
//! compared with equivalent NumPy operations. Results can be exported for
//! cross-comparison analysis.
//!
//! Run with: `cargo bench --bench numpy_comparison_benchmark`

#![allow(deprecated)]
#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::new_modules::polynomial::polyfit;
use numrs2::prelude::*;
use std::hint::black_box;
use std::time::Duration;

// Standard benchmark sizes matching common NumPy benchmarks
const SIZES: [usize; 4] = [1000, 10000, 100000, 1000000];
const MATRIX_SIZES: [usize; 4] = [32, 64, 128, 256];

// ============================================================================
// ARRAY CREATION - NumPy Equivalents
// ============================================================================

fn numpy_creation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_creation");
    group.measurement_time(Duration::from_secs(3));

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // np.zeros
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::zeros(&[size])))
        });

        // np.ones
        group.bench_with_input(BenchmarkId::new("ones", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::ones(&[size])))
        });

        // np.full
        group.bench_with_input(BenchmarkId::new("full", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::full(&[size], std::f64::consts::PI)))
        });

        // np.arange
        if *size <= 100000 {
            group.bench_with_input(BenchmarkId::new("arange", size), size, |b, &size| {
                b.iter(|| black_box(arange(0.0f64, size as f64, 1.0f64)))
            });
        }

        // np.linspace
        if *size <= 100000 {
            group.bench_with_input(BenchmarkId::new("linspace", size), size, |b, &size| {
                b.iter(|| black_box(linspace(0.0f64, 1.0f64, size)))
            });
        }
    }

    // Matrix creation
    for size in MATRIX_SIZES.iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        // np.eye
        group.bench_with_input(BenchmarkId::new("eye", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::eye(size, size, 0)))
        });

        // np.identity
        group.bench_with_input(BenchmarkId::new("identity", size), size, |b, &size| {
            b.iter(|| black_box(Array::<f64>::identity(size)))
        });
    }

    group.finish();
}

// ============================================================================
// ARRAY OPERATIONS - NumPy Equivalents
// ============================================================================

fn numpy_arithmetic_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_arithmetic");
    group.measurement_time(Duration::from_secs(3));

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Prepare arrays
        let a_data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.001 + 1.0).collect();
        let b_data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.002 + 0.5).collect();
        let a = Array::from_vec(a_data);
        let b = Array::from_vec(b_data);

        // np.add (a + b)
        group.bench_with_input(BenchmarkId::new("add", size), size, |bench, _| {
            bench.iter(|| black_box(&a + &b))
        });

        // np.subtract (a - b)
        group.bench_with_input(BenchmarkId::new("subtract", size), size, |bench, _| {
            bench.iter(|| black_box(&a - &b))
        });

        // np.multiply (a * b)
        group.bench_with_input(BenchmarkId::new("multiply", size), size, |bench, _| {
            bench.iter(|| black_box(&a * &b))
        });

        // np.divide (a / b)
        group.bench_with_input(BenchmarkId::new("divide", size), size, |bench, _| {
            bench.iter(|| black_box(&a / &b))
        });

        // a + scalar
        group.bench_with_input(BenchmarkId::new("add_scalar", size), size, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(2.5)))
        });
    }

    group.finish();
}

fn numpy_ufunc_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_ufuncs");
    group.measurement_time(Duration::from_secs(3));

    for size in SIZES.iter() {
        if *size > 100000 {
            continue; // Skip largest for expensive ops
        }

        group.throughput(Throughput::Elements(*size as u64));

        // Positive data for sqrt/log
        let pos_data: Vec<f64> = (1..=*size).map(|x| x as f64).collect();
        let pos = Array::from_vec(pos_data);

        // Small data for exp (avoid overflow)
        let small_data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.0001).collect();
        let small = Array::from_vec(small_data);

        // Angle data for trig functions
        let angle_data: Vec<f64> = (0..*size)
            .map(|x| (x as f64) * 0.001 * std::f64::consts::PI)
            .collect();
        let angles = Array::from_vec(angle_data);

        // np.sqrt
        group.bench_with_input(BenchmarkId::new("sqrt", size), size, |bench, _| {
            bench.iter(|| black_box(pos.sqrt()))
        });

        // np.exp
        group.bench_with_input(BenchmarkId::new("exp", size), size, |bench, _| {
            bench.iter(|| black_box(small.exp()))
        });

        // np.log
        group.bench_with_input(BenchmarkId::new("log", size), size, |bench, _| {
            bench.iter(|| black_box(pos.log()))
        });

        // np.sin
        group.bench_with_input(BenchmarkId::new("sin", size), size, |bench, _| {
            bench.iter(|| black_box(angles.sin()))
        });

        // np.cos
        group.bench_with_input(BenchmarkId::new("cos", size), size, |bench, _| {
            bench.iter(|| black_box(angles.cos()))
        });

        // np.abs
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
// REDUCTION OPERATIONS - NumPy Equivalents
// ============================================================================

fn numpy_reduction_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_reductions");
    group.measurement_time(Duration::from_secs(3));

    for size in SIZES.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|x| (x as f64) * 0.001).collect();
        let a = Array::from_vec(data);

        // np.sum
        group.bench_with_input(BenchmarkId::new("sum", size), size, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });

        // np.mean
        group.bench_with_input(BenchmarkId::new("mean", size), size, |bench, _| {
            bench.iter(|| black_box(a.mean()))
        });

        // np.std
        group.bench_with_input(BenchmarkId::new("std", size), size, |bench, _| {
            bench.iter(|| black_box(a.std()))
        });

        // np.var
        group.bench_with_input(BenchmarkId::new("var", size), size, |bench, _| {
            bench.iter(|| black_box(a.var()))
        });

        // np.min
        group.bench_with_input(BenchmarkId::new("min", size), size, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });

        // np.max
        group.bench_with_input(BenchmarkId::new("max", size), size, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
    }

    group.finish();
}

// ============================================================================
// ARRAY MANIPULATION - NumPy Equivalents
// ============================================================================

fn numpy_manipulation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_manipulation");
    group.measurement_time(Duration::from_secs(3));

    for size in SIZES.iter() {
        if *size > 100000 {
            continue;
        }

        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|x| x as f64).collect();
        let a = Array::from_vec(data);

        // np.reshape
        let side = (*size as f64).sqrt() as usize;
        if side * side == *size && side > 0 {
            group.bench_with_input(BenchmarkId::new("reshape", size), size, |bench, _| {
                bench.iter(|| black_box(a.reshape(&[side, side])))
            });

            // np.transpose
            let matrix = a.reshape(&[side, side]);
            group.bench_with_input(BenchmarkId::new("transpose", size), size, |bench, _| {
                bench.iter(|| black_box(matrix.transpose()))
            });

            // np.flatten
            group.bench_with_input(BenchmarkId::new("flatten", size), size, |bench, _| {
                bench.iter(|| black_box(matrix.flatten(None)))
            });
        }

        // np.copy
        group.bench_with_input(BenchmarkId::new("copy", size), size, |bench, _| {
            bench.iter(|| black_box(a.clone()))
        });
    }

    group.finish();
}

// ============================================================================
// LINEAR ALGEBRA - NumPy Equivalents
// ============================================================================

fn numpy_linalg_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_linalg");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    for size in MATRIX_SIZES.iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        // Create test matrix
        let data: Vec<f64> = (0..elements)
            .map(|i| ((i as f64 * 0.73) % 10.0) + 0.1)
            .collect();
        let a = Array::from_vec(data).reshape(&[*size, *size]);

        // Create vector
        let vec_data: Vec<f64> = (0..*size).map(|i| (i as f64) + 1.0).collect();
        let v = Array::from_vec(vec_data);

        // np.dot (matrix @ matrix)
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bench, _| {
            bench.iter(|| black_box(a.matmul(&a)))
        });

        // np.dot (vector . vector)
        group.bench_with_input(BenchmarkId::new("dot", size), size, |bench, _| {
            bench.iter(|| black_box(v.dot(&v)))
        });

        // np.trace - use standalone function from linalg
        group.bench_with_input(BenchmarkId::new("trace", size), size, |bench, _| {
            bench.iter(|| black_box(numrs2::linalg::vector_ops::trace(&a)))
        });
    }

    group.finish();
}

// ============================================================================
// POLYNOMIAL - NumPy Equivalents
// ============================================================================

fn numpy_polynomial_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("np_polynomial");
    group.measurement_time(Duration::from_secs(3));

    let degrees = [5, 10, 20, 50];

    for degree in degrees.iter() {
        // Create polynomial
        let coeffs: Vec<f64> = (0..=*degree).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let poly = Polynomial::new(coeffs);

        // np.polyval
        group.bench_with_input(
            BenchmarkId::new("polyval_single", degree),
            degree,
            |bench, _| bench.iter(|| black_box(poly.evaluate(2.5))),
        );

        // np.polyder
        group.bench_with_input(BenchmarkId::new("polyder", degree), degree, |bench, _| {
            bench.iter(|| black_box(poly.derivative()))
        });

        // np.polyint
        group.bench_with_input(BenchmarkId::new("polyint", degree), degree, |bench, _| {
            bench.iter(|| black_box(poly.integral()))
        });

        // np.polymul
        let p2_coeffs: Vec<f64> = (0..=*degree).map(|i| (i as f64) * 0.5).collect();
        let p2 = Polynomial::new(p2_coeffs);
        group.bench_with_input(BenchmarkId::new("polymul", degree), degree, |bench, _| {
            bench.iter(|| black_box(poly.clone() * p2.clone()))
        });

        // np.polyadd
        group.bench_with_input(BenchmarkId::new("polyadd", degree), degree, |bench, _| {
            bench.iter(|| black_box(poly.clone() + p2.clone()))
        });
    }

    // np.polyfit - use standalone polyfit function
    let fit_sizes = [10, 50, 100];
    for size in fit_sizes.iter() {
        let x_data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|x| x * x * 0.5 + x * 2.0 + 1.0 + (x * 0.1).sin())
            .collect();
        let x = Array::from_vec(x_data);
        let y = Array::from_vec(y_data);

        group.bench_with_input(BenchmarkId::new("polyfit_deg3", size), size, |bench, _| {
            bench.iter(|| black_box(polyfit(&x, &y, 3)))
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

criterion_group!(
    name = numpy_comparison;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1));
    targets =
        numpy_creation_benchmarks,
        numpy_arithmetic_benchmarks,
        numpy_ufunc_benchmarks,
        numpy_reduction_benchmarks,
        numpy_manipulation_benchmarks,
        numpy_linalg_benchmarks,
        numpy_polynomial_benchmarks,
);

criterion_main!(numpy_comparison);
