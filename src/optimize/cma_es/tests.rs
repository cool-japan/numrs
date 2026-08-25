//! Tests for CMA-ES optimizer.

use super::*;

// -----------------------------------------------------------------------
// Test objective functions
// -----------------------------------------------------------------------

/// Sphere function: f(x) = sum(x_i^2)
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Rosenbrock function: f(x) = sum[100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
fn rosenbrock(x: &[f64]) -> f64 {
    let mut val = 0.0;
    for i in 0..(x.len() - 1) {
        val += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    val
}

/// Rastrigin function: f(x) = 10n + sum[x_i^2 - 10*cos(2*pi*x_i)]
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut val = 10.0 * n;
    for &xi in x {
        val += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    val
}

/// Ackley function (multimodal, useful for testing)
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
    let sum_cos: f64 = x
        .iter()
        .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
        / n;
    -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
}

// -----------------------------------------------------------------------
// Test 1: Sphere function 2D (trivial)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_sphere_2d() {
    let x0 = vec![5.0, -3.0];
    let config = CMAESConfig::new(2).with_sigma0(2.0).with_max_iter(2000);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed on sphere");

    assert!(
        result.fun < 1e-6,
        "Should find near-zero minimum for sphere, got fun = {}",
        result.fun
    );
    for (i, &xi) in result.x.iter().enumerate() {
        assert!(xi.abs() < 1e-3, "x[{}] should be near zero, got {}", i, xi);
    }
    assert!(result.success, "Should converge on sphere");
}

// -----------------------------------------------------------------------
// Test 2: Sphere function 5D (higher dimension)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_sphere_5d() {
    let x0 = vec![3.0, -2.0, 1.0, 4.0, -1.5];
    let config = CMAESConfig::new(5).with_sigma0(2.0).with_max_iter(5000);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed on 5D sphere");

    assert!(
        result.fun < 1e-4,
        "Should find near-zero minimum for 5D sphere, got fun = {}",
        result.fun
    );
}

// -----------------------------------------------------------------------
// Test 3: Sphere function 10D
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_sphere_10d() {
    let x0 = vec![2.0; 10];
    let config = CMAESConfig::new(10).with_sigma0(2.0).with_max_iter(10000);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed on 10D sphere");

    assert!(
        result.fun < 1e-2,
        "Should find good minimum for 10D sphere, got fun = {}",
        result.fun
    );
}

// -----------------------------------------------------------------------
// Test 4: Rosenbrock function 2D (ill-conditioned)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_rosenbrock_2d() {
    let x0 = vec![0.0, 0.0];
    let config = CMAESConfig::new(2)
        .with_sigma0(1.0)
        .with_max_iter(5000)
        .with_ftol(1e-10);

    let result = cma_es(rosenbrock, &x0, config).expect("CMA-ES should succeed on Rosenbrock");

    assert!(
        result.fun < 1e-2,
        "Should find good solution for Rosenbrock, got fun = {}",
        result.fun
    );
    assert!(
        (result.x[0] - 1.0).abs() < 0.1,
        "x[0] should be near 1.0, got {}",
        result.x[0]
    );
    assert!(
        (result.x[1] - 1.0).abs() < 0.1,
        "x[1] should be near 1.0, got {}",
        result.x[1]
    );
}

// -----------------------------------------------------------------------
// Test 5: Rastrigin function with IPOP restart (multimodal)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_rastrigin_with_restart() {
    let x0 = vec![3.0, -2.0];
    // `.with_seed(...)` is load-bearing: without it, `RngSource::create(None)`
    // falls back to `thread_rng()` (see `cma_es::types::RngSource`), and
    // Rastrigin is multimodal enough that different sampling paths across the
    // restarts can land in a worse basin, making `result.fun < 5.0` a
    // genuine per-run coin flip. This seed was checked to land comfortably
    // under the threshold (well under 5.0, not marginal), and the restart
    // path derives per-restart seeds deterministically from this base seed
    // (see `cma_es::algorithm`), so the whole run is reproducible.
    let config = CMAESConfig::new(2)
        .with_sigma0(2.0)
        .with_max_iter(1000)
        .with_restarts(5)
        .with_seed(20260825);

    let result = cma_es(rastrigin, &x0, config).expect("CMA-ES IPOP should succeed on Rastrigin");

    // Rastrigin global minimum is 0 at origin; with restarts we should get close
    assert!(
        result.fun < 5.0,
        "Should find reasonable solution for Rastrigin with restarts, got fun = {}",
        result.fun
    );
}

// -----------------------------------------------------------------------
// Test 6: Box constraints
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_box_constraints() {
    // Minimize sphere with constraints: 1 <= x_i <= 5
    let x0 = vec![3.0, 3.0];
    let bounds = vec![(1.0, 5.0), (1.0, 5.0)];
    let config = CMAESConfig::new(2)
        .with_sigma0(1.0)
        .with_bounds(bounds)
        .with_max_iter(2000);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed with box constraints");

    // The constrained minimum of sphere on [1,5]^2 is at (1, 1) with f = 2
    assert!(
        result.fun < 2.5,
        "Should find constrained minimum near 2.0, got fun = {}",
        result.fun
    );
    for (i, &xi) in result.x.iter().enumerate() {
        assert!(
            (0.9..=5.1).contains(&xi),
            "x[{}] = {} should be approximately within [1, 5]",
            i,
            xi
        );
    }
}

// -----------------------------------------------------------------------
// Test 7: Box constraints with tight bounds
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_tight_box_constraints() {
    // Minimize (x-5)^2 + (y-5)^2 with constraints 0 <= x,y <= 3
    let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
    let x0 = vec![1.5, 1.5];
    let bounds = vec![(0.0, 3.0), (0.0, 3.0)];
    let config = CMAESConfig::new(2)
        .with_sigma0(1.0)
        .with_bounds(bounds)
        .with_max_iter(2000);

    let result = cma_es(f, &x0, config).expect("CMA-ES should succeed with tight box constraints");

    // Constrained optimum is at (3, 3) with f = 8
    assert!(
        result.fun < 9.0,
        "Should find constrained minimum near 8.0, got fun = {}",
        result.fun
    );
}

// -----------------------------------------------------------------------
// Test 8: Convergence history is monotone non-increasing
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_convergence_history() {
    let x0 = vec![5.0, 5.0];
    let config = CMAESConfig::new(2).with_sigma0(2.0).with_max_iter(500);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed");

    assert!(
        !result.history.is_empty(),
        "Should have convergence history"
    );
    // History records best-so-far, so should be monotonically non-increasing
    for i in 1..result.history.len() {
        assert!(
            result.history[i] <= result.history[i - 1] + 1e-10,
            "Convergence history should be non-increasing at index {}",
            i
        );
    }
}

// -----------------------------------------------------------------------
// Test 9: IPOP restart mechanism
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_restart_mechanism() {
    let x0 = vec![5.0, 5.0];
    let config = CMAESConfig::new(2)
        .with_sigma0(0.001) // Very small sigma to force quick stagnation
        .with_max_iter(50) // Low gen limit per run
        .with_restarts(3)
        .with_ftol(1e-20); // Very tight tolerance

    let result = cma_es(sphere, &x0, config).expect("CMA-ES IPOP should produce a result");

    // The algorithm should complete (may or may not converge)
    assert!(
        result.nfev > 0,
        "Should have performed function evaluations"
    );
}

// -----------------------------------------------------------------------
// Test 10: Result structure completeness
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_result_structure() {
    let x0 = vec![2.0, 2.0];
    let config = CMAESConfig::new(2).with_sigma0(1.0).with_max_iter(100);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed");

    // Verify all result fields are populated and sensible
    assert_eq!(result.x.len(), 2);
    assert!(result.fun.is_finite());
    assert!(result.nit > 0);
    assert!(result.nfev > 0);
    assert!(!result.message.is_empty());
    assert!(result.final_sigma > 0.0);
    assert!(result.final_condition_number >= 1.0);
}

// -----------------------------------------------------------------------
// Test 11: Shifted sphere (non-origin optimum)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_shifted_sphere() {
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
    let x0 = vec![0.0, 0.0];
    let config = CMAESConfig::new(2).with_sigma0(2.0).with_max_iter(2000);

    let result = cma_es(f, &x0, config).expect("CMA-ES should succeed on shifted sphere");

    assert!(
        result.fun < 1e-4,
        "Should find minimum of shifted sphere, got fun = {}",
        result.fun
    );
    assert!(
        (result.x[0] - 2.0).abs() < 0.1,
        "x[0] should be near 2.0, got {}",
        result.x[0]
    );
    assert!(
        (result.x[1] - 3.0).abs() < 0.1,
        "x[1] should be near 3.0, got {}",
        result.x[1]
    );
}

// -----------------------------------------------------------------------
// Test 12: Ackley function (multimodal, moderate difficulty)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_ackley() {
    let x0 = vec![2.0, -2.0];
    // See the seed comment on `test_cma_es_rastrigin_with_restart`: without
    // `.with_seed(...)`, sampling falls back to `thread_rng()`, and Ackley's
    // many shallow local minima make convergence to `result.fun < 1.0`
    // sensitive to the sample path. Checked to land comfortably under the
    // threshold with this seed.
    let config = CMAESConfig::new(2)
        .with_sigma0(2.0)
        .with_max_iter(3000)
        .with_seed(20260825);

    let result = cma_es(ackley, &x0, config).expect("CMA-ES should succeed on Ackley");

    // Ackley minimum is 0 at origin
    assert!(
        result.fun < 1.0,
        "Should find good solution for Ackley, got fun = {}",
        result.fun
    );
}

// -----------------------------------------------------------------------
// Test 13: Eigendecomposition of identity matrix
// -----------------------------------------------------------------------

#[test]
fn test_eigendecomposition_identity() {
    use super::eigen::symmetric_eigendecomposition;

    let n = 3;
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }

    let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&mat, n)
        .expect("Eigendecomposition of identity should succeed");

    // All eigenvalues should be 1.0
    for &ev in &eigenvalues {
        assert!(
            (ev - 1.0).abs() < 1e-10,
            "Eigenvalue of identity should be 1.0, got {}",
            ev
        );
    }

    // Eigenvectors should form an orthogonal matrix
    for i in 0..n {
        let mut norm_sq = 0.0;
        for j in 0..n {
            norm_sq += eigenvectors[j * n + i] * eigenvectors[j * n + i];
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "Eigenvector {} should have unit norm",
            i
        );
    }
}

// -----------------------------------------------------------------------
// Test 14: Eigendecomposition of diagonal matrix
// -----------------------------------------------------------------------

#[test]
fn test_eigendecomposition_diagonal() {
    use super::eigen::symmetric_eigendecomposition;

    let n = 3;
    let mut mat = vec![0.0; n * n];
    mat[0] = 4.0;
    mat[4] = 9.0;
    mat[8] = 1.0;

    let (mut eigenvalues, _) = symmetric_eigendecomposition(&mat, n)
        .expect("Eigendecomposition of diagonal should succeed");

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    assert!(
        (eigenvalues[0] - 1.0).abs() < 1e-10,
        "Smallest eigenvalue should be 1.0, got {}",
        eigenvalues[0]
    );
    assert!(
        (eigenvalues[1] - 4.0).abs() < 1e-10,
        "Middle eigenvalue should be 4.0, got {}",
        eigenvalues[1]
    );
    assert!(
        (eigenvalues[2] - 9.0).abs() < 1e-10,
        "Largest eigenvalue should be 9.0, got {}",
        eigenvalues[2]
    );
}

// -----------------------------------------------------------------------
// Test 15: Eigendecomposition of general symmetric matrix
// -----------------------------------------------------------------------

#[test]
fn test_eigendecomposition_symmetric() {
    use super::eigen::symmetric_eigendecomposition;

    // A = [[2, 1], [1, 3]]
    // Eigenvalues: (5 +/- sqrt(5)) / 2
    let mat = vec![2.0, 1.0, 1.0, 3.0];
    let (mut eigenvalues, eigenvectors) = symmetric_eigendecomposition(&mat, 2)
        .expect("Eigendecomposition of symmetric matrix should succeed");

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let expected_1 = (5.0 - 5.0_f64.sqrt()) / 2.0;
    let expected_2 = (5.0 + 5.0_f64.sqrt()) / 2.0;

    assert!(
        (eigenvalues[0] - expected_1).abs() < 1e-8,
        "First eigenvalue should be {}, got {}",
        expected_1,
        eigenvalues[0]
    );
    assert!(
        (eigenvalues[1] - expected_2).abs() < 1e-8,
        "Second eigenvalue should be {}, got {}",
        expected_2,
        eigenvalues[1]
    );

    // Verify A * v = lambda * v for each eigenpair
    for col in 0..2 {
        let lambda = eigenvalues[col];
        let v = [eigenvectors[col], eigenvectors[2 + col]];
        let av0 = mat[0] * v[0] + mat[1] * v[1];
        let av1 = mat[2] * v[0] + mat[3] * v[1];
        let lv0 = lambda * v[0];
        let lv1 = lambda * v[1];

        assert!(
            (av0 - lv0).abs() < 1e-6,
            "A*v != lambda*v for eigenvalue {}: {} vs {}",
            lambda,
            av0,
            lv0
        );
        assert!(
            (av1 - lv1).abs() < 1e-6,
            "A*v != lambda*v for eigenvalue {}: {} vs {}",
            lambda,
            av1,
            lv1
        );
    }
}

// -----------------------------------------------------------------------
// Test 16: Default weights properties
// -----------------------------------------------------------------------

#[test]
fn test_default_weights() {
    use super::config::compute_default_weights;

    let weights = compute_default_weights(5, 10);
    assert_eq!(weights.len(), 5);

    // Weights should sum to 1
    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Weights should sum to 1.0, got {}",
        sum
    );

    // Weights should be monotonically decreasing
    for i in 1..weights.len() {
        assert!(
            weights[i] <= weights[i - 1],
            "Weights should be non-increasing"
        );
    }

    // All weights should be positive
    for &w in &weights {
        assert!(w > 0.0, "All weights should be positive");
    }
}

// -----------------------------------------------------------------------
// Test 17: Config builder pattern
// -----------------------------------------------------------------------

#[test]
fn test_config_builder() {
    let config = CMAESConfig::new(5)
        .with_sigma0(2.0)
        .with_max_iter(5000)
        .with_bounds(vec![(-10.0, 10.0); 5])
        .with_restarts(3)
        .with_ftol(1e-8)
        .with_xtol(1e-8)
        .with_lambda(20)
        .with_seed(42);

    assert_eq!(config.population_size, 20);
    assert!((config.sigma0 - 2.0).abs() < 1e-10);
    assert_eq!(config.max_iter, 5000);
    assert!(config.bounds.is_some());
    assert!(config.enable_restarts);
    assert_eq!(config.max_restarts, 3);
    assert_eq!(config.seed, Some(42));
}

// -----------------------------------------------------------------------
// Test 18: TerminationReason Display trait
// -----------------------------------------------------------------------

#[test]
fn test_termination_reason_display() {
    assert_eq!(
        format!("{}", TerminationReason::FunctionTolerance),
        "function value tolerance reached"
    );
    assert_eq!(
        format!("{}", TerminationReason::MaxGenerations),
        "maximum generations reached"
    );
    assert_eq!(
        format!("{}", TerminationReason::ConditionNumber),
        "covariance matrix condition number too large"
    );
    assert_eq!(
        format!("{}", TerminationReason::StepSizeDiverged),
        "step-size diverged"
    );
}

// -----------------------------------------------------------------------
// Test 19: Input validation (zero dimension)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_zero_dimension_error() {
    let x0: Vec<f64> = vec![];
    let config = CMAESConfig::default();
    let result = cma_es(sphere, &x0, config);
    assert!(result.is_err(), "Should fail with zero-dimensional input");
}

// -----------------------------------------------------------------------
// Test 20: Input validation (invalid sigma)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_invalid_sigma_error() {
    let x0 = vec![1.0, 2.0];
    let config = CMAESConfig::default().with_sigma0(-1.0);
    let result = cma_es(sphere, &x0, config);
    assert!(result.is_err(), "Should fail with negative sigma");
}

// -----------------------------------------------------------------------
// Test 21: Input validation (mismatched bounds)
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_mismatched_bounds_error() {
    let x0 = vec![1.0, 2.0];
    let bounds = vec![(0.0, 5.0)]; // Only 1 bound for 2D problem
    let config = CMAESConfig::default().with_bounds(bounds);
    let result = cma_es(sphere, &x0, config);
    assert!(
        result.is_err(),
        "Should fail with mismatched bounds dimensions"
    );
}

// -----------------------------------------------------------------------
// Test 22: Backward-compatible accessors
// -----------------------------------------------------------------------

#[test]
fn test_backward_compatible_accessors() {
    let x0 = vec![2.0, 2.0];
    let config = CMAESConfig::new(2).with_sigma0(1.0).with_max_iter(200);

    let result = cma_es(sphere, &x0, config).expect("CMA-ES should succeed");

    // Test backward-compatible accessors
    assert_eq!(result.x_best(), &result.x[..]);
    assert!((result.f_best() - result.fun).abs() < 1e-15);
    assert_eq!(result.generations(), result.nit);
    assert_eq!(result.function_evaluations(), result.nfev);
    assert_eq!(result.converged(), result.success);
    assert_eq!(result.convergence_history(), &result.history[..]);
}

// -----------------------------------------------------------------------
// Test 23: Seeded reproducibility
// -----------------------------------------------------------------------

#[test]
fn test_cma_es_seeded_reproducibility() {
    let x0 = vec![3.0, -2.0];
    let config1 = CMAESConfig::new(2)
        .with_sigma0(1.0)
        .with_max_iter(100)
        .with_seed(12345);

    let config2 = CMAESConfig::new(2)
        .with_sigma0(1.0)
        .with_max_iter(100)
        .with_seed(12345);

    let result1 = cma_es(sphere, &x0, config1).expect("CMA-ES run 1 should succeed");
    let result2 = cma_es(sphere, &x0, config2).expect("CMA-ES run 2 should succeed");

    // Same seed should produce same results
    assert!(
        (result1.fun - result2.fun).abs() < 1e-10,
        "Seeded runs should produce same fun: {} vs {}",
        result1.fun,
        result2.fun
    );
    for (i, (&a, &b)) in result1.x.iter().zip(result2.x.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "Seeded runs should produce same x[{}]: {} vs {}",
            i,
            a,
            b
        );
    }
}
