//! Benchmarks for special functions in NumRS2
//!
//! This file contains benchmarks for various special functions
//! to track performance and identify bottlenecks.

#[macro_use]
extern crate criterion;
use criterion::{BenchmarkId, Criterion};

use numrs2::prelude::*;
use std::hint::black_box;

/// Benchmark error functions
fn bench_error_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_functions");

    // Create arrays of different sizes
    for size in [100, 1000, 10000].iter() {
        // Create random input
        let rng = random::default_rng();
        let x = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark erf
        group.bench_with_input(BenchmarkId::new("erf", size), size, |b, _| {
            b.iter(|| black_box(erf(&x)))
        });

        // Benchmark erfc
        group.bench_with_input(BenchmarkId::new("erfc", size), size, |b, _| {
            b.iter(|| black_box(erfc(&x)))
        });

        // Create inputs in [-0.99, 0.99] for erfinv
        let erfinv_input = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(1.98)
            .add_scalar(-0.99);

        // Benchmark erfinv
        group.bench_with_input(BenchmarkId::new("erfinv", size), size, |b, _| {
            b.iter(|| black_box(erfinv(&erfinv_input)))
        });

        // Create inputs in [0.01, 1.99] for erfcinv
        let erfcinv_input = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(1.98)
            .add_scalar(0.01);

        // Benchmark erfcinv
        group.bench_with_input(BenchmarkId::new("erfcinv", size), size, |b, _| {
            b.iter(|| black_box(erfcinv(&erfcinv_input)))
        });
    }

    group.finish();
}

/// Benchmark gamma functions
fn bench_gamma_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_functions");

    // Create arrays of different sizes
    for size in [100, 1000, 10000].iter() {
        // Create random inputs [0.5, 10.5] for gamma
        let rng = random::default_rng();
        let x = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(10.0)
            .add_scalar(0.5);

        // Benchmark gamma
        group.bench_with_input(BenchmarkId::new("gamma", size), size, |b, _| {
            b.iter(|| black_box(gamma(&x)))
        });

        // Benchmark gammaln
        group.bench_with_input(BenchmarkId::new("gammaln", size), size, |b, _| {
            b.iter(|| black_box(gammaln(&x)))
        });

        // Benchmark digamma
        group.bench_with_input(BenchmarkId::new("digamma", size), size, |b, _| {
            b.iter(|| black_box(digamma(&x)))
        });

        // Create a and x for gammainc
        let a = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(5.0)
            .add_scalar(0.5);
        let x2 = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(10.0)
            .add_scalar(0.1);

        // Benchmark gammainc
        group.bench_with_input(BenchmarkId::new("gammainc", size), size, |b, _| {
            b.iter(|| black_box(gammainc(&a, &x2).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark Bessel functions
fn bench_bessel_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("bessel_functions");

    // Create arrays of different sizes
    for size in [100, 1000, 10000].iter() {
        // Create random inputs [0.1, 20.1] for Bessel functions
        let rng = random::default_rng();
        let x = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(20.0)
            .add_scalar(0.1);

        // Benchmark bessel_j for different orders
        for n in [0, 1, 2, 5, 10].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("bessel_j_{}", n), size),
                size,
                |b, _| b.iter(|| black_box(bessel_j(*n, &x))),
            );
        }

        // Create positive inputs [0.1, 20.1] for bessel_y
        let x_positive = rng
            .random::<f64>(&[*size])
            .unwrap()
            .multiply_scalar(20.0)
            .add_scalar(0.1);

        // Benchmark bessel_y for different orders
        for n in [0, 1, 2, 5].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("bessel_y_{}", n), size),
                size,
                |b, _| b.iter(|| black_box(bessel_y(*n, &x_positive))),
            );
        }

        // Benchmark bessel_i for different orders
        for n in [0, 1, 2, 5].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("bessel_i_{}", n), size),
                size,
                |b, _| b.iter(|| black_box(bessel_i(*n, &x))),
            );
        }

        // Benchmark bessel_k for different orders
        for n in [0, 1, 2, 5].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("bessel_k_{}", n), size),
                size,
                |b, _| b.iter(|| black_box(bessel_k(*n, &x_positive))),
            );
        }
    }

    group.finish();
}

/// Benchmark elliptic integrals
fn bench_elliptic_integrals(c: &mut Criterion) {
    let mut group = c.benchmark_group("elliptic_integrals");

    // Create arrays of different sizes
    for size in [100, 1000, 10000].iter() {
        // Create random inputs [0, 0.99] for elliptic integrals
        let rng = random::default_rng();
        let m = rng.random::<f64>(&[*size]).unwrap().multiply_scalar(0.99);

        // Benchmark ellipk
        group.bench_with_input(BenchmarkId::new("ellipk", size), size, |b, _| {
            b.iter(|| black_box(ellipk(&m)))
        });

        // Benchmark ellipe
        group.bench_with_input(BenchmarkId::new("ellipe", size), size, |b, _| {
            b.iter(|| black_box(ellipe(&m)))
        });
    }

    group.finish();
}

/// Benchmark vectorized vs. scalar special functions
fn bench_vectorized_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectorized_vs_scalar");

    // Test with an array of size 1000
    let size = 1000;
    let rng = random::default_rng();
    let x = rng
        .random::<f64>(&[size])
        .unwrap()
        .multiply_scalar(10.0)
        .add_scalar(0.1);

    // Vectorized erf
    group.bench_function("vectorized_erf", |b| b.iter(|| black_box(erf(&x))));

    // Scalar erf (element by element)
    group.bench_function("scalar_erf", |b| {
        b.iter(|| {
            let mut result = Array::<f64>::zeros(&[size]);
            let x_vec = x.to_vec();
            for i in 0..size {
                let val = numerics::error_functions::erf_scalar(x_vec[i]);
                result.set(&[i], val).unwrap();
            }
            black_box(result)
        })
    });

    // Vectorized gamma
    group.bench_function("vectorized_gamma", |b| b.iter(|| black_box(gamma(&x))));

    // Scalar gamma (element by element)
    group.bench_function("scalar_gamma", |b| {
        b.iter(|| {
            let mut result = Array::<f64>::zeros(&[size]);
            let x_vec = x.to_vec();
            for i in 0..size {
                let val = numerics::gamma_functions::gamma_scalar(x_vec[i]);
                result.set(&[i], val).unwrap();
            }
            black_box(result)
        })
    });

    group.finish();
}

// Create a dummy namespace for scalar implementations for benchmarking
#[allow(dead_code)]
mod numerics {
    pub mod error_functions {
        pub fn erf_scalar(x: f64) -> f64 {
            // Use a polynomial approximation for the error function
            // Based on Abramowitz and Stegun formula 7.1.26

            // Constants
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;

            // Save the sign of x
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();

            // A&S formula 7.1.26
            let t = 1.0 / (1.0 + p * x);
            let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

            sign * y
        }
    }

    pub mod gamma_functions {
        pub fn gamma_scalar(x: f64) -> f64 {
            // Using Lanczos approximation for the gamma function

            // Handle special cases
            if x == 0.0 {
                return f64::INFINITY;
            }
            if x < 0.0 {
                // Reflection formula
                let pi = std::f64::consts::PI;
                return pi / ((pi * x).sin() * gamma_scalar(1.0 - x));
            }

            let g = 7.0; // Lanczos parameter
            let coeffs = [
                0.99999999999980993,
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ];

            // Shift x by 1 to accommodate the algorithm
            let x = x - 1.0;

            // Calculate the approximation
            let mut sum = coeffs[0];
            for i in 1..coeffs.len() {
                sum = sum + coeffs[i] / (x + i as f64);
            }

            let t = x + g + 0.5;
            let sqrt_2pi = 2.506628274631000502415765284811; // sqrt(2*pi)

            sqrt_2pi * sum * t.powf(x + 0.5) * (-t).exp()
        }
    }
}

criterion_group!(
    benches,
    bench_error_functions,
    bench_gamma_functions,
    bench_bessel_functions,
    bench_elliptic_integrals,
    bench_vectorized_vs_scalar,
);
criterion_main!(benches);
