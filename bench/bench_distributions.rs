//! Benchmarks for random distributions in NumRS2
//!
//! This benchmark suite compares the performance of:
//! 1. NumRS2's native distribution implementations
//! 2. SciRS2 integration implementations (when available)
//!
//! Run with: `cargo bench --bench bench_distributions`
//! To include SciRS2: `cargo bench --bench bench_distributions --features scirs`

#![allow(deprecated)]
#![allow(clippy::result_large_err)]

#[macro_use]
extern crate criterion;
use criterion::{BenchmarkId, Criterion};
use numrs2::array::Array;
use numrs2::random::distributions::*;
use std::hint::black_box;

// Import the SciRS2 integration when available
#[cfg(feature = "scirs")]
use numrs2::interop::scirs_compat as sci;

fn bench_normal_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Normal Distributions");

    // Sample sizes to benchmark
    let sizes = [100, 1000, 10000];

    for size in sizes.iter() {
        // NumRS2 normal distribution
        group.bench_with_input(BenchmarkId::new("NumRS2 normal", size), size, |b, &size| {
            b.iter(|| black_box(normal(0.0, 1.0, &[size])));
        });

        // NumRS2 truncated normal distribution
        group.bench_with_input(
            BenchmarkId::new("NumRS2 truncated_normal", size),
            size,
            |b, &size| {
                b.iter(|| black_box(truncated_normal(0.0, 1.0, -2.0, 2.0, &[size])));
            },
        );

        // SciRS2 truncated normal distribution (when available)
        #[cfg(feature = "scirs")]
        group.bench_with_input(
            BenchmarkId::new("SciRS2 truncated_normal", size),
            size,
            |b, &size| {
                b.iter(|| black_box(sci::truncated_normal(0.0, 1.0, -2.0, 2.0, &[size])));
            },
        );
    }

    group.finish();
}

fn bench_advanced_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Advanced Distributions");

    // Sample sizes to benchmark
    let sizes = [100, 1000, 10000];

    for size in sizes.iter() {
        // Native NumRS2 chi-square
        group.bench_with_input(
            BenchmarkId::new("NumRS2 chisquare", size),
            size,
            |b, &size| {
                b.iter(|| black_box(chisquare(5.0, &[size])));
            },
        );

        // NumRS2 noncentral chi-square distribution (when available)
        #[cfg(not(feature = "scirs"))]
        group.bench_with_input(
            BenchmarkId::new("NumRS2 noncentral_chisquare", size),
            size,
            |b, &size| {
                b.iter(|| black_box(noncentral_chisquare(5.0, 2.0, &[size])));
            },
        );

        // SciRS2 noncentral chi-square distribution (when available)
        #[cfg(feature = "scirs")]
        group.bench_with_input(
            BenchmarkId::new("SciRS2 noncentral_chisquare", size),
            size,
            |b, &size| {
                b.iter(|| black_box(sci::noncentral_chisquare(5.0, 2.0, &[size])));
            },
        );

        // Native NumRS2 F distribution
        // TODO: f_dist is not yet implemented
        // group.bench_with_input(BenchmarkId::new("NumRS2 f", size), size, |b, &size| {
        //     b.iter(|| black_box(f_dist(5.0, 10.0, &[size])));
        // });

        // NumRS2 noncentral F distribution (when available)
        #[cfg(not(feature = "scirs"))]
        group.bench_with_input(
            BenchmarkId::new("NumRS2 noncentral_f", size),
            size,
            |b, &size| {
                b.iter(|| black_box(noncentral_f(5.0, 10.0, 2.0, &[size])));
            },
        );

        // SciRS2 noncentral F distribution (when available)
        #[cfg(feature = "scirs")]
        group.bench_with_input(
            BenchmarkId::new("SciRS2 noncentral_f", size),
            size,
            |b, &size| {
                b.iter(|| black_box(sci::noncentral_f(5.0, 10.0, 2.0, &[size])));
            },
        );
    }

    group.finish();
}

fn bench_circular_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Circular Distributions");

    // Sample sizes to benchmark
    let sizes = [100, 1000, 10000];

    for size in sizes.iter() {
        // NumRS2 uniform circular distribution
        group.bench_with_input(
            BenchmarkId::new("NumRS2 uniform circular", size),
            size,
            |b, &size| {
                b.iter(|| black_box(uniform(0.0, 2.0 * std::f64::consts::PI, &[size])));
            },
        );

        // NumRS2 von Mises distribution (when available)
        #[cfg(not(feature = "scirs"))]
        group.bench_with_input(
            BenchmarkId::new("NumRS2 vonmises", size),
            size,
            |b, &size| {
                b.iter(|| black_box(vonmises(0.0, 2.0, &[size])));
            },
        );

        // SciRS2 von Mises distribution (when available)
        #[cfg(feature = "scirs")]
        group.bench_with_input(
            BenchmarkId::new("SciRS2 vonmises", size),
            size,
            |b, &size| {
                b.iter(|| black_box(sci::vonmises(0.0, 2.0, &[size])));
            },
        );
    }

    group.finish();
}

fn bench_multivariate_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multivariate Distributions");

    // Sample sizes to benchmark
    let sizes = [10, 100, 1000];

    // Common setup for multivariate tests
    let mean = vec![0.0, 0.0];
    let cov_data = vec![1.0, 0.5, 0.5, 1.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

    // Create a rotation matrix for 45 degrees
    use std::f64::consts::FRAC_1_SQRT_2;
    let rotation_data = vec![FRAC_1_SQRT_2, FRAC_1_SQRT_2, -FRAC_1_SQRT_2, FRAC_1_SQRT_2];
    let rotation = Array::from_vec(rotation_data.clone()).reshape(&[2, 2]);

    for size in sizes.iter() {
        // Native NumRS2 multivariate normal
        group.bench_with_input(
            BenchmarkId::new("NumRS2 multivariate_normal", size),
            size,
            |b, &size| {
                b.iter(|| black_box(multivariate_normal(&mean, &cov, Some(&[size]))));
            },
        );

        // NumRS2 multivariate normal with Cholesky decomposition
        // TODO: multivariate_normal_cholesky is not yet implemented
        // group.bench_with_input(
        //     BenchmarkId::new("NumRS2 multivariate_normal_cholesky", size),
        //     size,
        //     |b, &size| {
        //         b.iter(|| black_box(multivariate_normal_cholesky(&mean, &cov, size)));
        //     },
        // );

        // SciRS2 multivariate normal with rotation (when available)
        #[cfg(feature = "scirs")]
        group.bench_with_input(
            BenchmarkId::new("SciRS2 multivariate_normal_with_rotation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    black_box(sci::multivariate_normal_with_rotation(
                        &mean,
                        &cov,
                        Some(&[size]),
                        Some(&rotation),
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_normal_distributions,
    bench_advanced_distributions,
    bench_circular_distributions,
    bench_multivariate_distributions
);
criterion_main!(benches);
