//! Comprehensive Optimization Benchmarks for NumRS2
//!
//! This benchmark suite tests optimization algorithms including:
//! - BFGS and L-BFGS
//! - Conjugate gradient methods
//! - Trust region methods
//! - Genetic algorithms
//! - Particle swarm optimization
//! - Simulated annealing
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::optimize::{
    bfgs, conjugate_gradient_fr, conjugate_gradient_hs, conjugate_gradient_pr,
    differential_evolution, genetic_algorithm, lbfgs, particle_swarm, simulated_annealing,
    trust_region,
};
use numrs2::prelude::*;
use std::f64::consts::PI;
use std::hint::black_box;

/// Rosenbrock function for testing optimization
fn rosenbrock(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::INFINITY;
    }

    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let term1 = 100.0 * (x[i + 1] - x[i].powi(2)).powi(2);
        let term2 = (1.0 - x[i]).powi(2);
        sum += term1 + term2;
    }
    sum
}

/// Gradient of Rosenbrock function
fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];

    if n < 2 {
        return grad;
    }

    for i in 0..n - 1 {
        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i].powi(2)) - 2.0 * (1.0 - x[i]);
    }
    for i in 1..n {
        grad[i] += 200.0 * (x[i] - x[i - 1].powi(2));
    }

    grad
}

/// Sphere function for testing optimization
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Gradient of Sphere function
fn sphere_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| 2.0 * xi).collect()
}

/// Rastrigin function for testing optimization
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x
        .iter()
        .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
        .sum();
    10.0 * n + sum
}

/// Gradient of Rastrigin function
///
/// Mirrors `sphere_grad` above; unlike `sphere`, `rastrigin` is currently
/// only benchmarked through gradient-free methods (genetic_algorithm), so
/// this gradient has no caller yet. Kept for a future gradient-based
/// (bfgs/lbfgs/conjugate-gradient) Rastrigin benchmark group.
#[allow(dead_code)]
fn rastrigin_grad(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| 2.0 * xi + 20.0 * PI * (2.0 * PI * xi).sin())
        .collect()
}

/// Ackley function for testing optimization
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum();

    let term1 = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp();
    let term2 = -(sum_cos / n).exp();
    term1 + term2 + 20.0 + std::f64::consts::E
}

/// Gradient of Ackley function
///
/// Mirrors `sphere_grad` above; unlike `sphere`, `ackley` is currently only
/// benchmarked through gradient-free methods (particle_swarm), so this
/// gradient has no caller yet. Kept for a future gradient-based
/// (bfgs/lbfgs/conjugate-gradient) Ackley benchmark group.
#[allow(dead_code)]
fn ackley_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum();
    let sqrt_term = (sum_sq / n).sqrt();

    x.iter()
        .map(|&xi| {
            let term1 = 4.0 * (-0.2 * sqrt_term).exp() * xi / (n * sqrt_term);
            let term2 = 2.0 * PI * -(sum_cos / n).exp() * (2.0 * PI * xi).sin() / n;
            term1 + term2
        })
        .collect()
}

/// Hessian of Rosenbrock function (simplified - returns identity for benchmarking)
fn rosenbrock_hess(_x: &[f64]) -> Vec<Vec<f64>> {
    let n = _x.len();
    let mut hess = vec![vec![0.0; n]; n];
    for (i, row) in hess.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    hess
}

/// Hessian of Sphere function (constant diagonal)
fn sphere_hess(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];
    for (i, row) in hess.iter_mut().enumerate() {
        row[i] = 2.0;
    }
    hess
}

/// Benchmark BFGS optimization
fn bench_bfgs_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfgs_optimization");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("rosenbrock", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-2.0, 2.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = bfgs(rosenbrock, rosenbrock_grad, &x0_vec, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = bfgs(sphere, sphere_grad, &x0_vec, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark L-BFGS optimization
fn bench_lbfgs_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgs_optimization");

    for dims in [2, 5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::new("rosenbrock", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-2.0, 2.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = lbfgs(rosenbrock, rosenbrock_grad, &x0_vec, 10, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = lbfgs(sphere, sphere_grad, &x0_vec, 10, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Conjugate Gradient - Fletcher-Reeves
fn bench_conjugate_gradient_fr(c: &mut Criterion) {
    let mut group = c.benchmark_group("conjugate_gradient_fletcher_reeves");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("rosenbrock", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-2.0, 2.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) =
                        conjugate_gradient_fr(rosenbrock, rosenbrock_grad, &x0_vec, None)
                    {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = conjugate_gradient_fr(sphere, sphere_grad, &x0_vec, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Conjugate Gradient - Polak-Ribiere
fn bench_conjugate_gradient_pr(c: &mut Criterion) {
    let mut group = c.benchmark_group("conjugate_gradient_polak_ribiere");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("rosenbrock", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-2.0, 2.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) =
                        conjugate_gradient_pr(rosenbrock, rosenbrock_grad, &x0_vec, None)
                    {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Conjugate Gradient - Hestenes-Stiefel
fn bench_conjugate_gradient_hs(c: &mut Criterion) {
    let mut group = c.benchmark_group("conjugate_gradient_hestenes_stiefel");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("rosenbrock", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-2.0, 2.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) =
                        conjugate_gradient_hs(rosenbrock, rosenbrock_grad, &x0_vec, None)
                    {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Trust Region optimization
fn bench_trust_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("trust_region");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("rosenbrock", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-2.0, 2.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) =
                        trust_region(rosenbrock, rosenbrock_grad, rosenbrock_hess, &x0_vec, None)
                    {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) =
                        trust_region(sphere, sphere_grad, sphere_hess, &x0_vec, None)
                    {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Genetic Algorithm optimization
fn bench_genetic_algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_algorithm");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); d];
            b.iter(|| {
                if let Ok(result) = genetic_algorithm(sphere, &bounds, None) {
                    black_box(result);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("rastrigin", dims), dims, |b, &d| {
            let bounds: Vec<(f64, f64)> = vec![(-5.12, 5.12); d];
            b.iter(|| {
                if let Ok(result) = genetic_algorithm(rastrigin, &bounds, None) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark Particle Swarm Optimization
fn bench_particle_swarm_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_swarm_optimization");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); d];
            b.iter(|| {
                if let Ok(result) = particle_swarm(sphere, &bounds, None) {
                    black_box(result);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("ackley", dims), dims, |b, &d| {
            let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); d];
            b.iter(|| {
                if let Ok(result) = particle_swarm(ackley, &bounds, None) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark Simulated Annealing
fn bench_simulated_annealing(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulated_annealing");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = simulated_annealing(sphere, &x0_vec, None) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("rastrigin", dims), dims, |b, &d| {
            let rng = random::default_rng();
            if let Ok(x0) = rng.uniform::<f64>(-5.12, 5.12, &[d]) {
                let x0_vec = x0.to_vec();
                b.iter(|| {
                    if let Ok(result) = simulated_annealing(rastrigin, &x0_vec, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark Differential Evolution
fn bench_differential_evolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("differential_evolution");

    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("sphere", dims), dims, |b, &d| {
            let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); d];
            b.iter(|| {
                if let Ok(result) = differential_evolution(sphere, &bounds, None) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark comparison between optimization algorithms
fn bench_optimization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_comparison");

    let dims = 5;
    let rng = random::default_rng();

    // BFGS
    group.bench_function("bfgs_sphere_5d", |b| {
        if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[dims]) {
            let x0_vec = x0.to_vec();
            b.iter(|| {
                if let Ok(result) = bfgs(sphere, sphere_grad, &x0_vec, None) {
                    black_box(result);
                }
            });
        }
    });

    // L-BFGS
    group.bench_function("lbfgs_sphere_5d", |b| {
        if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[dims]) {
            let x0_vec = x0.to_vec();
            b.iter(|| {
                if let Ok(result) = lbfgs(sphere, sphere_grad, &x0_vec, 10, None) {
                    black_box(result);
                }
            });
        }
    });

    // CG Fletcher-Reeves
    group.bench_function("cg_fr_sphere_5d", |b| {
        if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[dims]) {
            let x0_vec = x0.to_vec();
            b.iter(|| {
                if let Ok(result) = conjugate_gradient_fr(sphere, sphere_grad, &x0_vec, None) {
                    black_box(result);
                }
            });
        }
    });

    // Trust Region
    group.bench_function("trust_region_sphere_5d", |b| {
        if let Ok(x0) = rng.uniform::<f64>(-5.0, 5.0, &[dims]) {
            let x0_vec = x0.to_vec();
            b.iter(|| {
                if let Ok(result) = trust_region(sphere, sphere_grad, sphere_hess, &x0_vec, None) {
                    black_box(result);
                }
            });
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bfgs_optimization,
    bench_lbfgs_optimization,
    bench_conjugate_gradient_fr,
    bench_conjugate_gradient_pr,
    bench_conjugate_gradient_hs,
    bench_trust_region,
    bench_genetic_algorithm,
    bench_particle_swarm_optimization,
    bench_simulated_annealing,
    bench_differential_evolution,
    bench_optimization_comparison,
);

criterion_main!(benches);
