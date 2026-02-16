//! Advanced Optimization Example for NumRS2
//!
//! This example demonstrates all available optimization algorithms:
//! - Gradient-based: BFGS, L-BFGS, Conjugate Gradient (FR, PR, HS)
//! - Derivative-free: Nelder-Mead, Powell's Method
//! - Global optimization: Particle Swarm (PSO), Simulated Annealing, Differential Evolution, Genetic Algorithm
//! - Constrained optimization: Projected Gradient, Penalty Method, Interior Point, SQP
//! - Least-squares: Levenberg-Marquardt, Trust Region
//!
//! Real-world examples:
//! - Portfolio optimization
//! - Parameter tuning for models
//! - Constraint handling
//! - Performance comparison
//!
//! Run with: cargo run --example advanced_optimization

#![allow(clippy::type_complexity)]

use numrs2::optimize::*;
use numrs2::prelude::*;
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== NumRS2 Advanced Optimization Examples ===\n");

    // Example 1: Gradient-Based Methods (BFGS, L-BFGS, Conjugate Gradient)
    example1_gradient_methods()?;

    // Example 2: Derivative-Free Methods (Nelder-Mead)
    example2_derivative_free()?;

    // Example 3: Global Optimization (PSO, SA, DE, GA)
    example3_global_optimization()?;

    // Example 4: Constrained Optimization
    example4_constrained_optimization()?;

    // Example 5: Real-World Application - Portfolio Optimization
    example5_portfolio_optimization()?;

    // Example 6: Parameter Tuning Comparison
    example6_algorithm_comparison()?;

    // Example 7: Least-Squares Problems
    example7_least_squares()?;

    println!("\n=== All Optimization Examples Completed Successfully! ===");
    Ok(())
}

/// Example 1: Gradient-Based Optimization Methods
fn example1_gradient_methods() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Gradient-Based Optimization");
    println!("======================================\n");

    // Test function: Rosenbrock function
    // f(x, y) = (1-x)² + 100(y-x²)²
    // Global minimum at (1, 1) with f(1,1) = 0
    let rosenbrock = |x: &[f64]| {
        let (x0, x1) = (x[0], x[1]);
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
    };

    let rosenbrock_grad = |x: &[f64]| {
        let (x0, x1) = (x[0], x[1]);
        vec![
            -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
            200.0 * (x1 - x0 * x0),
        ]
    };

    let x0 = vec![0.0, 0.0];

    // BFGS Method
    println!("1.1 BFGS (Broyden-Fletcher-Goldfarb-Shanno)");
    let start = Instant::now();
    let result = bfgs(rosenbrock, rosenbrock_grad, &x0, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Time: {:?}\n", duration);

    // L-BFGS Method
    println!("1.2 L-BFGS (Limited-memory BFGS)");
    let start = Instant::now();
    let result = lbfgs(rosenbrock, rosenbrock_grad, &x0, 10, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    // Conjugate Gradient - Fletcher-Reeves
    println!("1.3 Conjugate Gradient (Fletcher-Reeves)");
    let start = Instant::now();
    let result = conjugate_gradient_fr(rosenbrock, rosenbrock_grad, &x0, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    // Conjugate Gradient - Polak-Ribiere
    println!("1.4 Conjugate Gradient (Polak-Ribiere)");
    let start = Instant::now();
    let result = conjugate_gradient_pr(rosenbrock, rosenbrock_grad, &x0, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    println!("✓ Example 1 completed\n");
    Ok(())
}

/// Example 2: Derivative-Free Optimization
fn example2_derivative_free() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Derivative-Free Optimization");
    println!("=======================================\n");

    // Test function: Himmelblau's function
    // f(x,y) = (x²+y-11)² + (x+y²-7)²
    // Has four identical local minima
    let himmelblau = |x: &[f64]| {
        let (x0, x1) = (x[0], x[1]);
        (x0 * x0 + x1 - 11.0).powi(2) + (x0 + x1 * x1 - 7.0).powi(2)
    };

    println!("2.1 Nelder-Mead Simplex Method");
    let x0 = vec![0.0, 0.0];
    let start = Instant::now();

    let config = OptimizeConfig {
        max_iter: 2000,
        ftol: 1e-8,
        ..Default::default()
    };

    let result = nelder_mead(himmelblau, &x0, Some(config))?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}", duration);
    println!("  Note: Himmelblau's function has 4 minima at approximately:");
    println!("        (3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)\n");

    println!("✓ Example 2 completed\n");
    Ok(())
}

/// Example 3: Global Optimization Methods
fn example3_global_optimization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Global Optimization Methods");
    println!("======================================\n");

    // Rastrigin function - highly multimodal test function
    // Global minimum at origin with f(0,...,0) = 0
    let rastrigin = |x: &[f64]| {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    };

    // Bounds for optimization
    let bounds = (-5.12, 5.12);
    let dim = 2;

    // 3.1 Particle Swarm Optimization
    println!("3.1 Particle Swarm Optimization (PSO)");
    let start = Instant::now();

    let pso_config = PSOConfig {
        swarm_size: 30,
        max_iter: 100,
        inertia: InertiaStrategy::Constant(0.7),
        c1: 1.5,
        c2: 1.5,
        v_max_factor: 0.2,
        topology: Topology::Global,
        ftol: 1e-8,
    };

    let bounds_vec = vec![bounds; dim];
    let result = particle_swarm(rastrigin, &bounds_vec, Some(pso_config))?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    // 3.2 Simulated Annealing
    println!("3.2 Simulated Annealing");
    let start = Instant::now();

    let sa_config = SAConfig {
        initial_temp: 100.0,
        cooling_rate: 0.95,
        linear_delta: 0.1,
        max_iter: 1000,
        neighbor_sigma: 0.5,
        cooling_schedule: CoolingSchedule::Exponential,
        neighbor_strategy: NeighborStrategy::Gaussian,
        ftol: 1e-9,
        stagnation_limit: 500,
    };

    let x0 = vec![0.0; dim];
    let result = simulated_annealing(rastrigin, &x0, Some(sa_config))?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    // 3.3 Differential Evolution
    println!("3.3 Differential Evolution");
    let start = Instant::now();

    let de_config = DEConfig {
        pop_size: 50,
        max_generations: 100,
        mutation_factor: 0.8,
        crossover_rate: 0.9,
        strategy: DEMutationStrategy::Best1,
        crossover: DECrossoverType::Binomial,
        adaptive: false,
        ftol: 1e-8,
    };

    let bounds_vec = vec![bounds; dim];
    let result = differential_evolution(rastrigin, &bounds_vec, Some(de_config))?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    // 3.4 Genetic Algorithm
    println!("3.4 Genetic Algorithm");
    let start = Instant::now();

    let ga_config = GAConfig {
        pop_size: 50,
        max_generations: 100,
        selection: SelectionStrategy::Tournament(3),
        crossover: GACrossoverType::Uniform(0.5),
        crossover_rate: 0.8,
        mutation: GAMutationStrategy::Gaussian(0.1),
        mutation_rate: 0.1,
        elitism_count: 2,
        ftol: 1e-8,
    };

    let bounds_vec = vec![bounds; dim];
    let result = genetic_algorithm(rastrigin, &bounds_vec, Some(ga_config))?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    println!("✓ Example 3 completed\n");
    Ok(())
}

/// Example 4: Constrained Optimization
fn example4_constrained_optimization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Constrained Optimization");
    println!("====================================\n");

    // 4.1 Box-Constrained Optimization with Projected Gradient
    println!("4.1 Projected Gradient Method (Box Constraints)");

    // Minimize f(x,y) = (x-5)² + (y-5)² subject to 0 ≤ x,y ≤ 3
    let objective = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
    let gradient = |x: &[f64]| vec![2.0 * (x[0] - 5.0), 2.0 * (x[1] - 5.0)];

    let bounds = BoxConstraints::uniform(2, Some(0.0), Some(3.0));
    let x0 = vec![1.0, 1.0];

    let start = Instant::now();
    let result = projected_gradient(objective, gradient, &x0, &bounds, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Expected: x = [3.0, 3.0] (boundary of feasible region)");
    println!("  Time: {:?}\n", duration);

    // 4.2 Penalty Method with Equality Constraints
    println!("4.2 Penalty Method (Equality Constraint)");

    // Minimize f(x,y) = x² + y² subject to x + y = 1
    let objective = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let gradient = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
    let eq_constraint = |x: &[f64]| x[0] + x[1] - 1.0;
    let eq_constraints: Vec<&dyn Fn(&[f64]) -> f64> = vec![&eq_constraint];

    let x0 = vec![0.5, 0.5];
    let start = Instant::now();
    let result = penalty_method(
        objective,
        gradient,
        &eq_constraints,
        &[],
        &x0,
        1.0,
        10.0,
        None,
    )?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Expected: x ≈ [0.5, 0.5] (on constraint line)");
    println!(
        "  Constraint violation: {:.6}",
        eq_constraint(&result.x).abs()
    );
    println!("  Time: {:?}\n", duration);

    // 4.3 Interior Point Method
    println!("4.3 Interior Point Method");

    // Minimize f(x) = -x subject to 0 ≤ x ≤ 10
    // Reformulate as: minimize f(x) subject to -x ≤ 0 and x - 10 ≤ 0
    let objective = |x: &[f64]| -x[0];
    let gradient = |_x: &[f64]| vec![-1.0];
    let hessian = |_x: &[f64]| vec![vec![0.0]];

    // Inequality constraints g(x) <= 0
    let ineq1 = |x: &[f64]| -x[0]; // -x ≤ 0 => x ≥ 0
    let ineq2 = |x: &[f64]| x[0] - 10.0; // x - 10 ≤ 0 => x ≤ 10

    let ineq_constraints: Vec<&dyn Fn(&[f64]) -> f64> = vec![&ineq1, &ineq2];

    // Gradients of inequality constraints
    let grad_ineq1 = |_x: &[f64]| vec![-1.0];
    let grad_ineq2 = |_x: &[f64]| vec![1.0];
    let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_ineq1, &grad_ineq2];

    let x0 = vec![5.0]; // Strictly feasible point
    let ip_config = IPConfig {
        max_iter: 50,
        max_inner_iter: 20,
        mu_0: 1.0,
        mu_reduction: 0.1,
        mu_min: 1e-8,
        gtol: 1e-6,
        ctol: 1e-6,
        barrier: BarrierType::Logarithmic,
    };

    let start = Instant::now();
    let result = interior_point(
        objective,
        gradient,
        hessian,
        &[], // No equality constraints
        &[], // No equality constraint gradients
        &ineq_constraints,
        &grad_ineq,
        &x0,
        Some(ip_config),
    )?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Solution: x = [{:.6}]", result.x[0]);
    println!("  Function value: {:.10}", result.fun);
    println!("  Expected: x ≈ 10.0 (upper bound)");
    println!("  Time: {:?}\n", duration);

    println!("✓ Example 4 completed\n");
    Ok(())
}

/// Example 5: Portfolio Optimization
fn example5_portfolio_optimization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Portfolio Optimization");
    println!("==================================\n");

    // Portfolio optimization: minimize risk while achieving target return
    // Using mean-variance optimization (Markowitz)

    // Expected returns for 4 assets
    let expected_returns = vec![0.10, 0.12, 0.08, 0.15];

    // Covariance matrix (4x4) - symmetric
    #[rustfmt::skip]
    let covariance = [vec![0.04, 0.01, 0.00, 0.02],
        vec![0.01, 0.09, 0.01, 0.03],
        vec![0.00, 0.01, 0.01, 0.01],
        vec![0.02, 0.03, 0.01, 0.16]];

    let target_return = 0.11; // Target 11% return

    // Objective: minimize portfolio variance
    let portfolio_variance = |weights: &[f64]| {
        let mut variance = 0.0;
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                variance += weights[i] * weights[j] * covariance[i][j];
            }
        }
        variance
    };

    // Gradient of portfolio variance
    let portfolio_variance_grad = |weights: &[f64]| {
        let n = weights.len();
        let mut grad = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                grad[i] += 2.0 * weights[j] * covariance[i][j];
            }
        }
        grad
    };

    // Constraints:
    // 1. Sum of weights = 1 (fully invested)
    // 2. Portfolio return = target_return
    // 3. All weights >= 0 (no short selling)

    let sum_constraint = |w: &[f64]| w.iter().sum::<f64>() - 1.0;
    let return_constraint = |w: &[f64]| {
        w.iter()
            .zip(expected_returns.iter())
            .map(|(wi, ri)| wi * ri)
            .sum::<f64>()
            - target_return
    };

    let eq_constraints: Vec<&dyn Fn(&[f64]) -> f64> = vec![&sum_constraint, &return_constraint];
    let ineq_constraints_fn: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..4)
        .map(|i| Box::new(move |w: &[f64]| -w[i]) as Box<dyn Fn(&[f64]) -> f64>)
        .collect();
    let ineq_constraints: Vec<&dyn Fn(&[f64]) -> f64> =
        ineq_constraints_fn.iter().map(|f| f.as_ref()).collect();

    // Initial guess: equal weights
    let x0 = vec![0.25, 0.25, 0.25, 0.25];

    println!("Problem Setup:");
    println!("  Assets: 4");
    println!("  Expected returns: {:?}", expected_returns);
    println!("  Target return: {:.2}%", target_return * 100.0);
    println!("  Constraints: weights sum to 1, no short selling\n");

    let start = Instant::now();
    let result = penalty_method(
        portfolio_variance,
        portfolio_variance_grad,
        &eq_constraints,
        &ineq_constraints,
        &x0,
        10.0,
        10.0,
        None,
    )?;
    let duration = start.elapsed();

    println!("Optimization Result:");
    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!("  Optimal portfolio weights:");
    for (i, &w) in result.x.iter().enumerate() {
        println!("    Asset {}: {:.4} ({:.1}%)", i + 1, w, w * 100.0);
    }
    println!("  Portfolio variance: {:.6}", result.fun);
    println!("  Portfolio std dev (risk): {:.4}", result.fun.sqrt());

    let actual_return: f64 = result
        .x
        .iter()
        .zip(expected_returns.iter())
        .map(|(wi, ri)| wi * ri)
        .sum();
    println!("  Actual return: {:.2}%", actual_return * 100.0);
    println!("  Time: {:?}\n", duration);

    println!("✓ Example 5 completed\n");
    Ok(())
}

/// Example 6: Algorithm Performance Comparison
fn example6_algorithm_comparison() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 6: Algorithm Performance Comparison");
    println!("===========================================\n");

    // Sphere function: f(x) = Σ xi²
    let sphere = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
    let sphere_grad = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect();

    let dim = 10;
    let x0: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.5).collect();

    println!("Test Function: {}-dimensional Sphere", dim);
    println!("Global minimum: f(0,...,0) = 0");
    println!("Starting point: {:?}\n", &x0[..3.min(dim)]);

    let mut results = Vec::new();

    // Test BFGS
    let start = Instant::now();
    match bfgs(sphere, sphere_grad, &x0, None) {
        Ok(result) => {
            results.push(("BFGS", result.fun, result.nit, result.nfev, start.elapsed()));
        }
        Err(e) => println!("BFGS failed: {:?}", e),
    }

    // Test L-BFGS
    let start = Instant::now();
    match lbfgs(sphere, sphere_grad, &x0, 10, None) {
        Ok(result) => {
            results.push((
                "L-BFGS",
                result.fun,
                result.nit,
                result.nfev,
                start.elapsed(),
            ));
        }
        Err(e) => println!("L-BFGS failed: {:?}", e),
    }

    // Test Conjugate Gradient (FR)
    let start = Instant::now();
    match conjugate_gradient_fr(sphere, sphere_grad, &x0, None) {
        Ok(result) => {
            results.push((
                "CG-FR",
                result.fun,
                result.nit,
                result.nfev,
                start.elapsed(),
            ));
        }
        Err(e) => println!("CG-FR failed: {:?}", e),
    }

    // Test Nelder-Mead
    let start = Instant::now();
    let config = OptimizeConfig {
        max_iter: 1000,
        ..Default::default()
    };
    match nelder_mead(sphere, &x0, Some(config)) {
        Ok(result) => {
            results.push((
                "Nelder-Mead",
                result.fun,
                result.nit,
                result.nfev,
                start.elapsed(),
            ));
        }
        Err(e) => println!("Nelder-Mead failed: {:?}", e),
    }

    // Print comparison table
    println!("Performance Comparison:");
    println!(
        "{:<15} {:>15} {:>10} {:>10} {:>12}",
        "Algorithm", "Final f(x)", "Iters", "f-evals", "Time"
    );
    println!("{}", "-".repeat(65));

    for (name, fval, nit, nfev, time) in results {
        println!(
            "{:<15} {:>15.10} {:>10} {:>10} {:>12.3?}",
            name, fval, nit, nfev, time
        );
    }

    println!("\n✓ Example 6 completed\n");
    Ok(())
}

/// Example 7: Least-Squares Problems
fn example7_least_squares() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 7: Least-Squares Problems");
    println!("==================================\n");

    // 7.1 Linear Regression using Levenberg-Marquardt
    println!("7.1 Linear Regression (Levenberg-Marquardt)");

    // Data: y = 2.5x + 1.3 + noise
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data = vec![3.8, 6.3, 8.8, 11.3, 13.8, 16.3, 18.8, 21.3, 23.8, 26.3];

    // Residual function
    let residual = |params: &[f64]| -> Vec<f64> {
        let (slope, intercept) = (params[0], params[1]);
        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| slope * x + intercept - y)
            .collect()
    };

    let x0 = vec![1.0, 0.0];
    let start = Instant::now();
    let result = levenberg_marquardt(residual, &x0, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!(
        "  Fitted model: y = {:.4}x + {:.4}",
        result.x[0], result.x[1]
    );
    println!("  True model: y = 2.5x + 1.3");
    println!("  Residual sum of squares: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    // 7.2 Exponential Decay Fitting
    println!("7.2 Exponential Decay Fitting");

    // Data: y = 5 * exp(-0.5*x) + noise
    let x_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = [5.0, 3.0, 1.8, 1.1, 0.7, 0.4];

    let residual = |params: &[f64]| -> Vec<f64> {
        let (a, k) = (params[0], params[1]);
        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| a * (-k * x).exp() - y)
            .collect()
    };

    let x0 = vec![4.0, 0.4];
    let start = Instant::now();
    let result = levenberg_marquardt(residual, &x0, None)?;
    let duration = start.elapsed();

    println!(
        "  Status: {}",
        if result.success { "Success" } else { "Failed" }
    );
    println!(
        "  Fitted model: y = {:.4} * exp(-{:.4}*x)",
        result.x[0], result.x[1]
    );
    println!("  True model: y = 5.0 * exp(-0.5*x)");
    println!("  Residual sum of squares: {:.10}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Time: {:?}\n", duration);

    println!("✓ Example 7 completed\n");
    Ok(())
}
