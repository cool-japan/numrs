//! Comprehensive Statistics Benchmarks for NumRS2
//!
//! This benchmark suite tests all major statistical operations including:
//! - Statistical functions (mean, variance, std, median, quantiles)
//! - Distribution sampling (all distributions in scirs2_stats)
//! - Correlation and covariance
//! - Histogram computation
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::prelude::*;
use numrs2::stats::distributions::*;
use std::hint::black_box;

/// Benchmark basic statistical functions
fn bench_basic_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_statistics");

    for size in [100, 1000, 10000, 100000].iter() {
        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = mean(&data, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Variance
        group.bench_with_input(BenchmarkId::new("variance", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = var(&data, None, 0, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = std(&data, None, 0, false) {
                        black_box(result);
                    }
                });
            }
        });

        // Median
        group.bench_with_input(BenchmarkId::new("median", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = median(&data, None, false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark quantile calculations
fn bench_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantiles");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("percentile_50", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                let q = Array::from_vec(vec![50.0]);
                b.iter(|| {
                    if let Ok(result) = percentile(&data, &q, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("quantile_0.25", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                let q = Array::from_vec(vec![0.25]);
                b.iter(|| {
                    if let Ok(result) = quantile(&data, &q, None) {
                        black_box(result);
                    }
                });
            }
        });

        // Multiple quantiles
        group.bench_with_input(
            BenchmarkId::new("quantiles_multiple", size),
            size,
            |b, &s| {
                let rng = random::default_rng();
                if let Ok(data) = rng.random::<f64>(&[s]) {
                    let q_vec = vec![0.25, 0.5, 0.75];
                    let quantiles = Array::from_vec(q_vec);
                    b.iter(|| {
                        if let Ok(result) = quantile(&data, &quantiles, None) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark correlation and covariance
fn bench_correlation_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_covariance");

    for size in [100, 1000, 10000].iter() {
        // Correlation between two vectors
        group.bench_with_input(BenchmarkId::new("corrcoef_2vec", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(x), Ok(y)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                b.iter(|| {
                    if let Ok(result) = corrcoef(&x, Some(&y), None) {
                        black_box(result);
                    }
                });
            }
        });

        // Covariance between two vectors
        group.bench_with_input(BenchmarkId::new("cov_2vec", size), size, |b, &s| {
            let rng = random::default_rng();
            if let (Ok(x), Ok(y)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                b.iter(|| {
                    if let Ok(result) = cov(&x, Some(&y), None, None, None) {
                        black_box(result);
                    }
                });
            }
        });

        // Covariance matrix
        group.bench_with_input(BenchmarkId::new("cov_matrix", size), size, |b, &s| {
            let rng = random::default_rng();
            // Create matrix with observations as rows
            if let Ok(data) = rng.random::<f64>(&[s, 10]) {
                b.iter(|| {
                    if let Ok(result) = cov(&data, None, None, None, None) {
                        black_box(result);
                    }
                });
            }
        });

        // Correlation matrix
        group.bench_with_input(BenchmarkId::new("corrcoef_matrix", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s, 10]) {
                b.iter(|| {
                    if let Ok(result) = corrcoef(&data, None, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark histogram computation
fn bench_histograms(c: &mut Criterion) {
    let mut group = c.benchmark_group("histograms");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("histogram_10bins", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = histogram(&data, 10, None, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("histogram_50bins", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = histogram(&data, 50, None, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(
            BenchmarkId::new("histogram_100bins", size),
            size,
            |b, &s| {
                let rng = random::default_rng();
                if let Ok(data) = rng.random::<f64>(&[s]) {
                    b.iter(|| {
                        if let Ok(result) = histogram(&data, 100, None, None) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark distribution sampling - Normal distribution
fn bench_normal_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("normal_standard", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.standard_normal::<f64>(&[s]) {
                    black_box(result);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("normal_custom", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.normal::<f64>(5.0, 2.0, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark distribution sampling - Uniform distribution
fn bench_uniform_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("uniform_0_1", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.random::<f64>(&[s]) {
                    black_box(result);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("uniform_custom", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.uniform::<f64>(-5.0, 5.0, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark distribution sampling - Exponential distribution
fn bench_exponential_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("exponential", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.exponential::<f64>(1.5, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark distribution sampling - Gamma distribution
fn bench_gamma_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("gamma", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.gamma::<f64>(2.0, 2.0, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark distribution sampling - Beta distribution
fn bench_beta_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("beta", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.beta::<f64>(2.0, 5.0, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark distribution sampling - Chi-squared distribution
fn bench_chisquare_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("chisquare_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("chisquare", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.chisquare::<f64>(5.0, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

// /// Benchmark distribution sampling - Student's t distribution
// /// DISABLED: Generator::t() method not yet implemented
// fn bench_t_distribution(c: &mut Criterion) {
//     let mut group = c.benchmark_group("t_distribution");
//
//     for size in [100, 1000, 10000, 100000].iter() {
//         group.bench_with_input(BenchmarkId::new("t_dist", size), size, |b, &s| {
//             let rng = random::default_rng();
//             b.iter(|| {
//                 if let Ok(result) = rng.t::<f64>(&[s], 10.0) {
//                     black_box(result);
//                 }
//             });
//         });
//     }
//
//     group.finish();
// }
//
// /// Benchmark distribution sampling - F distribution
// /// DISABLED: Generator::f() method not yet implemented
// fn bench_f_distribution(c: &mut Criterion) {
//     let mut group = c.benchmark_group("f_distribution");
//
//     for size in [100, 1000, 10000, 100000].iter() {
//         group.bench_with_input(BenchmarkId::new("f_dist", size), size, |b, &s| {
//             let rng = random::default_rng();
//             b.iter(|| {
//                 if let Ok(result) = rng.f::<f64>(&[s], 5.0, 10.0) {
//                     black_box(result);
//                 }
//             });
//         });
//     }
//
//     group.finish();
// }

/// Benchmark distribution sampling - Poisson distribution
fn bench_poisson_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("poisson", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.poisson::<f64>(5.0, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark distribution sampling - Binomial distribution
fn bench_binomial_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("binomial_distribution");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("binomial", size), size, |b, &s| {
            let rng = random::default_rng();
            b.iter(|| {
                if let Ok(result) = rng.binomial::<f64>(10, 0.5, &[s]) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark cumulative statistics
fn bench_cumulative_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("cumulative_statistics");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("cumsum", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = cumsum(&data, None, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("cumprod", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = cumprod(&data, None, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark statistical moments
fn bench_statistical_moments(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_moments");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("skewness", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = skew(&data, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("kurtosis", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(data) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = kurtosis(&data, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

// TODO: shuffle and choice are in random/legacy, not stats - need to fix imports
// /// Benchmark random sampling and shuffling
// fn bench_random_sampling(c: &mut Criterion) {
//     let mut group = c.benchmark_group("random_sampling");
//
//     for size in [100, 1000, 10000].iter() {
//         group.bench_with_input(BenchmarkId::new("shuffle", size), size, |b, &s| {
//             let rng = random::default_rng();
//             if let Ok(data) = rng.random::<f64>(&[s]) {
//                 b.iter(|| {
//                     if let Ok(result) = shuffle(&data) {
//                         black_box(result);
//                     }
//                 });
//             }
//         });
//
//         group.bench_with_input(BenchmarkId::new("choice", size), size, |b, &s| {
//             let rng = random::default_rng();
//             if let Ok(data) = rng.random::<f64>(&[s]) {
//                 b.iter(|| {
//                     if let Ok(result) = choice(&data, s / 10, None, None) {
//                         black_box(result);
//                     }
//                 });
//             }
//         });
//     }
//
//     group.finish();
// }

/// Benchmark distribution functions (PDF, CDF, PPF)
fn bench_distribution_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution_functions");

    // Beta distribution functions
    group.bench_function("beta_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = beta_pdf(black_box(0.5), black_box(2.0), black_box(3.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("beta_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = beta_cdf(black_box(0.5), black_box(2.0), black_box(3.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("beta_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = beta_ppf(black_box(0.75), black_box(2.0), black_box(3.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("beta_logpdf", |b| {
        b.iter(|| {
            if let Ok(result) = beta_logpdf(black_box(0.5), black_box(2.0), black_box(3.0)) {
                black_box(result);
            }
        });
    });

    // Gamma distribution functions
    group.bench_function("gamma_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = gamma_pdf(black_box(2.0), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("gamma_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = gamma_cdf(black_box(2.0), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("gamma_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = gamma_ppf(black_box(0.75), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("gamma_logpdf", |b| {
        b.iter(|| {
            if let Ok(result) = gamma_logpdf(black_box(2.0), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    // Student's t distribution functions
    group.bench_function("student_t_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = student_t_pdf(black_box(1.5), black_box(10.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("student_t_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = student_t_cdf(black_box(1.5), black_box(10.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("student_t_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = student_t_ppf(black_box(0.975), black_box(10.0)) {
                black_box(result);
            }
        });
    });

    // Cauchy distribution functions
    group.bench_function("cauchy_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = cauchy_pdf(black_box(1.0), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("cauchy_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = cauchy_cdf(black_box(1.0), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("cauchy_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = cauchy_ppf(black_box(0.75), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    // Laplace distribution functions
    group.bench_function("laplace_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = laplace_pdf(black_box(1.0), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("laplace_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = laplace_cdf(black_box(1.0), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("laplace_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = laplace_ppf(black_box(0.75), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    // Logistic distribution functions
    group.bench_function("logistic_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = logistic_pdf(black_box(1.0), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("logistic_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = logistic_cdf(black_box(1.0), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("logistic_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = logistic_ppf(black_box(0.75), black_box(0.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    // Pareto distribution functions
    group.bench_function("pareto_pdf", |b| {
        b.iter(|| {
            if let Ok(result) = pareto_pdf(black_box(2.0), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("pareto_cdf", |b| {
        b.iter(|| {
            if let Ok(result) = pareto_cdf(black_box(2.0), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.bench_function("pareto_ppf", |b| {
        b.iter(|| {
            if let Ok(result) = pareto_ppf(black_box(0.75), black_box(2.0), black_box(1.0)) {
                black_box(result);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_statistics,
    bench_quantiles,
    bench_correlation_covariance,
    bench_histograms,
    bench_normal_distribution,
    bench_uniform_distribution,
    bench_exponential_distribution,
    bench_gamma_distribution,
    bench_beta_distribution,
    bench_chisquare_distribution,
    // bench_t_distribution, // DISABLED: Generator::t() not yet implemented
    // bench_f_distribution, // DISABLED: Generator::f() not yet implemented
    bench_poisson_distribution,
    bench_binomial_distribution,
    bench_cumulative_statistics,
    bench_statistical_moments,
    // bench_random_sampling, // TODO: Fix imports for shuffle/choice
    bench_distribution_functions,
);

criterion_main!(benches);
