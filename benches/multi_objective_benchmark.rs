//! Multi-Objective Optimization Performance Benchmarks
//!
//! Comprehensive benchmarks for multi-objective optimization algorithms:
//! - NSGA-II: Non-dominated Sorting Genetic Algorithm II
//! - NSGA-III: Reference point-based NSGA for many objectives
//! - Quality metrics: Hypervolume, IGD, GD, Spacing, Spread
//! - Convergence analysis and scalability testing

#![allow(clippy::type_complexity)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::optimize::nsga2::{nsga2, NSGA2Config, QualityMetricsConfig};
use numrs2::optimize::nsga3::{nsga3, NSGA3Config};
use numrs2::optimize::test_problems::{TestProblem, DTLZ2, DTLZ3, ZDT1, ZDT2, ZDT3};
use std::hint::black_box;

/// Benchmark NSGA-II on ZDT1 with varying population sizes
fn bench_nsga2_zdt1_population(c: &mut Criterion) {
    let mut group = c.benchmark_group("nsga2_zdt1_population");
    group.sample_size(10); // Reduce sample size for expensive operations

    for pop_size in [50, 100, 200].iter() {
        let problem = ZDT1::new(30);
        let bounds = problem.bounds();

        group.bench_with_input(
            BenchmarkId::new("pop_size", pop_size),
            pop_size,
            |bencher, &pop_size| {
                bencher.iter(|| {
                    let config = NSGA2Config {
                        pop_size,
                        max_generations: 50,
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark NSGA-II on ZDT2 with varying number of generations
fn bench_nsga2_zdt2_generations(c: &mut Criterion) {
    let mut group = c.benchmark_group("nsga2_zdt2_generations");
    group.sample_size(10);

    for generations in [50, 100, 200].iter() {
        let problem = ZDT2::new(30);
        let bounds = problem.bounds();

        group.bench_with_input(
            BenchmarkId::new("generations", generations),
            generations,
            |bencher, &generations| {
                bencher.iter(|| {
                    let config = NSGA2Config {
                        pop_size: 100,
                        max_generations: generations,
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT2::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT2::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark NSGA-II on ZDT3 (disconnected Pareto front)
fn bench_nsga2_zdt3(c: &mut Criterion) {
    let mut group = c.benchmark_group("nsga2_zdt3");
    group.sample_size(10);

    let problem = ZDT3::new(30);
    let bounds = problem.bounds();

    group.bench_function("zdt3_100gen", |bencher| {
        bencher.iter(|| {
            let config = NSGA2Config {
                pop_size: 100,
                max_generations: 100,
                ..Default::default()
            };

            let objectives: Vec<_> = vec![
                |x: &[f64]| {
                    let problem = ZDT3::new(30);
                    problem.evaluate(x)[0]
                },
                |x: &[f64]| {
                    let problem = ZDT3::new(30);
                    problem.evaluate(x)[1]
                },
            ];

            black_box(nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"));
        });
    });

    group.finish();
}

/// Benchmark NSGA-III on DTLZ2 with varying objective counts
fn bench_nsga3_dtlz2_objectives(c: &mut Criterion) {
    let mut group = c.benchmark_group("nsga3_dtlz2_objectives");
    group.sample_size(10);

    for n_objectives in [3, 5, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("objectives", n_objectives),
            n_objectives,
            |bencher, &n_obj| {
                bencher.iter(|| {
                    let config = NSGA3Config {
                        pop_size: 100,
                        max_generations: 50,
                        n_divisions: 12,
                        ..Default::default()
                    };

                    let problem = DTLZ2::new(n_obj, n_obj + 9);
                    let bounds = problem.bounds();

                    // Create objective functions dynamically
                    let mut objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = Vec::new();
                    for i in 0..n_obj {
                        let obj_idx = i;
                        objectives.push(Box::new(move |x: &[f64]| {
                            let problem = DTLZ2::new(n_obj, n_obj + 9);
                            problem.evaluate(x)[obj_idx]
                        }));
                    }

                    let obj_refs: Vec<_> = objectives.iter().map(|f| f.as_ref()).collect();

                    black_box(
                        nsga3(&obj_refs, &bounds, Some(config)).expect("NSGA-III should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark NSGA-III on DTLZ3 with varying population sizes
fn bench_nsga3_dtlz3_population(c: &mut Criterion) {
    let mut group = c.benchmark_group("nsga3_dtlz3_population");
    group.sample_size(10);

    for pop_size in [100, 200, 400].iter() {
        let problem = DTLZ3::new(3, 12); // 3 objectives, 12 variables
        let bounds = problem.bounds();

        group.bench_with_input(
            BenchmarkId::new("pop_size", pop_size),
            pop_size,
            |bencher, &pop_size| {
                bencher.iter(|| {
                    let config = NSGA3Config {
                        pop_size,
                        max_generations: 50,
                        n_divisions: 12,
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = DTLZ3::new(3, 12);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = DTLZ3::new(3, 12);
                            problem.evaluate(x)[1]
                        },
                        |x: &[f64]| {
                            let problem = DTLZ3::new(3, 12);
                            problem.evaluate(x)[2]
                        },
                    ];

                    black_box(
                        nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hypervolume calculation with different Pareto front sizes
fn bench_hypervolume_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypervolume_calculation");

    for front_size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*front_size as u64));

        // Generate a synthetic Pareto front
        let pareto_front: Vec<Vec<f64>> = (0..*front_size)
            .map(|i| {
                let f1 = i as f64 / *front_size as f64;
                let f2 = 1.0 - f1.sqrt();
                vec![f1, f2]
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("front_size", front_size),
            front_size,
            |bencher, _| {
                bencher.iter(|| {
                    // Run NSGA-II with hypervolume calculation
                    let problem = ZDT1::new(30);
                    let bounds = problem.bounds();

                    let config = NSGA2Config {
                        pop_size: *front_size,
                        max_generations: 10, // Short run just to trigger metric calculation
                        quality_metrics_config: Some(QualityMetricsConfig {
                            calculate_spacing: false,
                            calculate_spread: false,
                            reference_front: Some(pareto_front.clone()),
                        }),
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark spacing metric calculation
fn bench_spacing_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("spacing_metric");

    for front_size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*front_size as u64));

        group.bench_with_input(
            BenchmarkId::new("front_size", front_size),
            front_size,
            |bencher, &front_size| {
                bencher.iter(|| {
                    let problem = ZDT1::new(30);
                    let bounds = problem.bounds();

                    let config = NSGA2Config {
                        pop_size: front_size,
                        max_generations: 10,
                        quality_metrics_config: Some(QualityMetricsConfig {
                            calculate_spacing: true,
                            calculate_spread: false,
                            reference_front: None,
                        }),
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark spread metric calculation
fn bench_spread_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("spread_metric");

    for front_size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*front_size as u64));

        group.bench_with_input(
            BenchmarkId::new("front_size", front_size),
            front_size,
            |bencher, &front_size| {
                bencher.iter(|| {
                    let problem = ZDT1::new(30);
                    let bounds = problem.bounds();

                    let config = NSGA2Config {
                        pop_size: front_size,
                        max_generations: 10,
                        quality_metrics_config: Some(QualityMetricsConfig {
                            calculate_spacing: false,
                            calculate_spread: true,
                            reference_front: None,
                        }),
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark IGD (Inverted Generational Distance) calculation
fn bench_igd_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("igd_metric");

    for front_size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*front_size as u64));

        // Generate true Pareto front
        let problem = ZDT1::new(30);
        let reference_front = problem.generate_pareto_front(*front_size);

        group.bench_with_input(
            BenchmarkId::new("front_size", front_size),
            front_size,
            |bencher, &front_size| {
                bencher.iter(|| {
                    let problem = ZDT1::new(30);
                    let bounds = problem.bounds();

                    let config = NSGA2Config {
                        pop_size: front_size,
                        max_generations: 10,
                        quality_metrics_config: Some(QualityMetricsConfig {
                            calculate_spacing: false,
                            calculate_spread: false,
                            reference_front: Some(reference_front.clone()),
                        }),
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GD (Generational Distance) calculation
fn bench_gd_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("gd_metric");

    for front_size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*front_size as u64));

        // Generate true Pareto front
        let problem = ZDT1::new(30);
        let reference_front = problem.generate_pareto_front(*front_size);

        group.bench_with_input(
            BenchmarkId::new("front_size", front_size),
            front_size,
            |bencher, &front_size| {
                bencher.iter(|| {
                    let problem = ZDT1::new(30);
                    let bounds = problem.bounds();

                    let config = NSGA2Config {
                        pop_size: front_size,
                        max_generations: 10,
                        quality_metrics_config: Some(QualityMetricsConfig {
                            calculate_spacing: false,
                            calculate_spread: false,
                            reference_front: Some(reference_front.clone()),
                        }),
                        ..Default::default()
                    };

                    let objectives: Vec<_> = vec![
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[0]
                        },
                        |x: &[f64]| {
                            let problem = ZDT1::new(30);
                            problem.evaluate(x)[1]
                        },
                    ];

                    black_box(
                        nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark convergence speed comparison
fn bench_convergence_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_speed");
    group.sample_size(10);

    let problem = ZDT1::new(30);
    let bounds = problem.bounds();

    // NSGA-II convergence per generation
    group.bench_function("nsga2_per_generation", |bencher| {
        bencher.iter(|| {
            let config = NSGA2Config {
                pop_size: 100,
                max_generations: 1, // Single generation
                ..Default::default()
            };

            let objectives: Vec<_> = vec![
                |x: &[f64]| {
                    let problem = ZDT1::new(30);
                    problem.evaluate(x)[0]
                },
                |x: &[f64]| {
                    let problem = ZDT1::new(30);
                    problem.evaluate(x)[1]
                },
            ];

            black_box(nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"));
        });
    });

    // NSGA-III convergence per generation
    group.bench_function("nsga3_per_generation", |bencher| {
        bencher.iter(|| {
            let config = NSGA3Config {
                pop_size: 92,
                max_generations: 1, // Single generation
                n_divisions: 12,
                ..Default::default()
            };

            let objectives: Vec<_> = vec![
                |x: &[f64]| {
                    let problem = ZDT1::new(30);
                    problem.evaluate(x)[0]
                },
                |x: &[f64]| {
                    let problem = ZDT1::new(30);
                    problem.evaluate(x)[1]
                },
            ];

            black_box(nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should succeed"));
        });
    });

    group.finish();
}

/// Benchmark NSGA-II vs NSGA-III on same problem
fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");
    group.sample_size(10);

    let problem = DTLZ2::new(3, 12); // 3 objectives
    let bounds = problem.bounds();

    group.bench_function("nsga2_3obj", |bencher| {
        bencher.iter(|| {
            let config = NSGA2Config {
                pop_size: 100,
                max_generations: 50,
                ..Default::default()
            };

            let objectives: Vec<_> = vec![
                |x: &[f64]| {
                    let problem = DTLZ2::new(3, 12);
                    problem.evaluate(x)[0]
                },
                |x: &[f64]| {
                    let problem = DTLZ2::new(3, 12);
                    problem.evaluate(x)[1]
                },
                |x: &[f64]| {
                    let problem = DTLZ2::new(3, 12);
                    problem.evaluate(x)[2]
                },
            ];

            black_box(nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed"));
        });
    });

    group.bench_function("nsga3_3obj", |bencher| {
        bencher.iter(|| {
            let config = NSGA3Config {
                pop_size: 92,
                max_generations: 50,
                n_divisions: 12,
                ..Default::default()
            };

            let objectives: Vec<_> = vec![
                |x: &[f64]| {
                    let problem = DTLZ2::new(3, 12);
                    problem.evaluate(x)[0]
                },
                |x: &[f64]| {
                    let problem = DTLZ2::new(3, 12);
                    problem.evaluate(x)[1]
                },
                |x: &[f64]| {
                    let problem = DTLZ2::new(3, 12);
                    problem.evaluate(x)[2]
                },
            ];

            black_box(nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should succeed"));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_nsga2_zdt1_population,
    bench_nsga2_zdt2_generations,
    bench_nsga2_zdt3,
    bench_nsga3_dtlz2_objectives,
    bench_nsga3_dtlz3_population,
    bench_hypervolume_calculation,
    bench_spacing_metric,
    bench_spread_metric,
    bench_igd_metric,
    bench_gd_metric,
    bench_convergence_speed,
    bench_algorithm_comparison,
);
criterion_main!(benches);
