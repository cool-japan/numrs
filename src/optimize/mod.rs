//! Numerical optimization algorithms
//!
//! This module provides state-of-the-art optimization algorithms for finding
//! minima and maxima of scalar and vector-valued functions.
//!
//! # Available Methods
//!
//! ## Gradient-Based Methods
//! - **BFGS**: Quasi-Newton method with Hessian approximation
//! - **L-BFGS**: Limited-memory BFGS for large-scale problems
//! - **Conjugate Gradient**: Nonlinear conjugate gradient methods (Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel)
//!
//! ## Derivative-Free Methods
//! - **Nelder-Mead**: Simplex method for unconstrained optimization
//! - **Powell's Method**: Direction set method without derivatives
//!
//! ## Multi-Objective Methods
//! - **NSGA-II**: Non-dominated Sorting Genetic Algorithm II (2-3 objectives)
//! - **NSGA-III**: Non-dominated Sorting Genetic Algorithm III (3+ objectives, many-objective)
//! - **ZDT Test Suite**: Bi-objective benchmark problems (ZDT1, ZDT2, ZDT3)
//! - **DTLZ Test Suite**: Scalable many-objective benchmark problems (DTLZ1, DTLZ2, DTLZ3, DTLZ7)
//!
//! ## Line Search Methods
//! - **Wolfe conditions**: Strong Wolfe line search
//! - **Backtracking**: Simple backtracking line search
//! - **Golden section**: Exact line search for unimodal functions
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::optimize::*;
//!
//! // Minimize the Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
//! let f = |x: &[f64]| {
//!     let (x0, x1) = (x[0], x[1]);
//!     (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
//! };
//!
//! let grad = |x: &[f64]| {
//!     let (x0, x1) = (x[0], x[1]);
//!     vec![
//!         -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
//!         200.0 * (x1 - x0 * x0),
//!     ]
//! };
//!
//! let x0 = vec![0.0, 0.0]; // Initial guess
//! let result = bfgs(f, grad, &x0, None).expect("BFGS optimization should succeed");
//! assert!(result.success);
//! // Minimum at (1, 1)
//! ```

pub mod bayesian_opt;
pub mod cma_es;
mod conjugate_gradient;
mod constrained;
mod derivative_free;
pub mod differential_evolution;
pub mod genetic_algorithm;
mod gradient;
pub mod interior_point;
pub mod nsga2;
pub mod nsga3;
pub mod pso;
pub mod simulated_annealing;
pub mod sqp;
pub mod test_problems;
mod trust_region;

// Re-export all public items
pub use bayesian_opt::{
    bayesian_optimize, AcquisitionType, BayesOptConfig, BayesOptResult, GaussianProcess, KernelType,
};
pub use cma_es::{cma_es, CmaEsConfig, CmaEsResult, TerminationReason as CmaEsTermination};
pub use conjugate_gradient::{
    conjugate_gradient_fr, conjugate_gradient_hs, conjugate_gradient_pr, BetaComputation,
    ConjugateGradientMethod, FletcherReevesBeta, HesteneStiefelBeta, PolakRibiereBeta,
};
pub use constrained::{penalty_method, projected_gradient, BoxConstraints};
pub use derivative_free::nelder_mead;
pub use differential_evolution::{
    differential_evolution, CrossoverType as DECrossoverType, DEConfig,
    MutationStrategy as DEMutationStrategy,
};
pub use genetic_algorithm::{
    genetic_algorithm, CrossoverType as GACrossoverType, GAConfig,
    MutationStrategy as GAMutationStrategy, SelectionStrategy,
};
pub use gradient::{bfgs, check_gradient, lbfgs};
pub use interior_point::{interior_point, BarrierType, IPConfig};
pub use nsga2::{nsga2, Individual, NSGA2Config, NSGA2Result};
pub use nsga3::{nsga3, NSGA3Config, NSGA3Result, ReferencePoint};
pub use pso::{particle_swarm, InertiaStrategy, PSOConfig, Topology};
pub use simulated_annealing::{simulated_annealing, CoolingSchedule, NeighborStrategy, SAConfig};
pub use sqp::{sqp, SQPConfig};
pub use test_problems::{TestProblem, DTLZ1, DTLZ2, DTLZ3, DTLZ7, ZDT1, ZDT2, ZDT3};
pub use trust_region::{levenberg_marquardt, trust_region};

use num_traits::Float;

/// Configuration for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizeConfig<T: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance for gradient norm
    pub gtol: T,
    /// Convergence tolerance for function value change
    pub ftol: T,
    /// Convergence tolerance for parameter change
    pub xtol: T,
    /// Line search maximum iterations
    pub ls_max_iter: usize,
    /// Wolfe condition parameter c1 (sufficient decrease)
    pub c1: T,
    /// Wolfe condition parameter c2 (curvature)
    pub c2: T,
}

impl<T: Float> Default for OptimizeConfig<T> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            gtol: T::from(1e-5).expect("1e-5 should be representable in Float"),
            ftol: T::from(1e-9).expect("1e-9 should be representable in Float"),
            xtol: T::from(1e-9).expect("1e-9 should be representable in Float"),
            ls_max_iter: 20,
            c1: T::from(1e-4).expect("1e-4 should be representable in Float"),
            c2: T::from(0.9).expect("0.9 should be representable in Float"),
        }
    }
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizeResult<T: Float> {
    /// Optimal parameters found
    pub x: Vec<T>,
    /// Optimal function value
    pub fun: T,
    /// Gradient at optimum
    pub grad: Vec<T>,
    /// Number of iterations performed
    pub nit: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of gradient evaluations
    pub njev: usize,
    /// Whether optimization converged
    pub success: bool,
    /// Status message
    pub message: String,
}

// ============================================================================
// Utility Functions (used by multiple submodules)
// ============================================================================

/// Compute L2 norm of a vector
pub(crate) fn compute_norm<T: Float + std::iter::Sum>(v: &[T]) -> T {
    v.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Compute dot product of two vectors
pub(crate) fn dot_product<T: Float + std::iter::Sum>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bfgs_quadratic() {
        // Minimize f(x,y) = x^2 + y^2
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let result = bfgs(f, grad, &[3.0, 4.0], None).expect("BFGS should succeed for quadratic");
        assert!(result.success, "BFGS should converge for quadratic");
        assert!(result.fun < 1e-10, "Should find minimum at origin");
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        // Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        let f = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };
        let grad = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            vec![
                -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
                200.0 * (x1 - x0 * x0),
            ]
        };

        let result = bfgs(f, grad, &[0.0, 0.0], None).expect("BFGS should succeed for Rosenbrock");
        assert!(result.success, "BFGS should converge for Rosenbrock");
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lbfgs_quadratic() {
        // Minimize f(x,y) = x^2 + y^2
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let result =
            lbfgs(f, grad, &[3.0, 4.0], 5, None).expect("L-BFGS should succeed for quadratic");
        assert!(result.success, "L-BFGS should converge for quadratic");
        assert!(result.fun < 1e-10);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_lbfgs_higher_dimension() {
        // Minimize sum of squares in 5D
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let grad = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect();

        let x0 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result =
            lbfgs(f, grad, &x0, 5, None).expect("L-BFGS should succeed for higher dimension");

        assert!(result.success);
        for &xi in &result.x {
            assert_relative_eq!(xi, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_nelder_mead_quadratic() {
        // Minimize f(x,y) = (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        let result =
            nelder_mead(f, &[0.0, 0.0], None).expect("Nelder-Mead should succeed for quadratic");
        // Nelder-Mead should get reasonably close
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-2);
        assert!(result.fun < 0.01);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Rosenbrock is challenging for Nelder-Mead but it should make progress
        let f = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };

        let cfg = OptimizeConfig {
            max_iter: 2000, // More iterations for Nelder-Mead
            ftol: 1e-6,
            ..Default::default()
        };

        let result = nelder_mead(f, &[0.0, 0.0], Some(cfg))
            .expect("Nelder-Mead should succeed for Rosenbrock");
        // Should get reasonably close to (1, 1)
        assert!(result.fun < 0.1, "Should find good solution");
    }

    #[test]
    fn test_lbfgs_memory_limit() {
        // Test that L-BFGS respects memory limit
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let grad = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect();

        let x0 = vec![1.0; 10];
        let result = lbfgs(f, grad, &x0, 3, None).expect("L-BFGS should succeed with memory limit"); // Only 3 correction pairs

        assert!(result.success);
        for &xi in &result.x {
            assert_relative_eq!(xi, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_bfgs_beale_function() {
        // Beale's function: minimum at (3, 0.5) with f = 0
        let f = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            (1.5 - x0 + x0 * x1).powi(2)
                + (2.25 - x0 + x0 * x1 * x1).powi(2)
                + (2.625 - x0 + x0 * x1.powi(3)).powi(2)
        };

        let grad = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            let t1 = 1.5 - x0 + x0 * x1;
            let t2 = 2.25 - x0 + x0 * x1 * x1;
            let t3 = 2.625 - x0 + x0 * x1.powi(3);

            let df_dx0 = 2.0 * t1 * (-1.0 + x1)
                + 2.0 * t2 * (-1.0 + x1 * x1)
                + 2.0 * t3 * (-1.0 + x1.powi(3));
            let df_dx1 =
                2.0 * t1 * x0 + 2.0 * t2 * 2.0 * x0 * x1 + 2.0 * t3 * 3.0 * x0 * x1.powi(2);

            vec![df_dx0, df_dx1]
        };

        let result =
            bfgs(f, grad, &[1.0, 1.0], None).expect("BFGS should succeed for Beale function");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-3);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
    }

    // =========================================================================
    // Constrained Optimization Tests
    // =========================================================================

    #[test]
    fn test_box_constraints_projection() {
        let bounds = BoxConstraints {
            lower: vec![Some(0.0), Some(-1.0)],
            upper: vec![Some(5.0), Some(2.0)],
        };

        // Test projection
        let x = vec![-1.0, 3.0]; // Violates lower[0] and upper[1]
        let projected = bounds.project(&x);
        assert_relative_eq!(projected[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], 2.0, epsilon = 1e-10);

        // Test feasibility check
        assert!(!bounds.is_feasible(&x));
        assert!(bounds.is_feasible(&projected));
    }

    #[test]
    fn test_projected_gradient_simple() {
        // Minimize f(x,y) = (x-5)^2 + (y-5)^2 subject to 0 <= x,y <= 3
        let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 5.0), 2.0 * (x[1] - 5.0)];

        let bounds = BoxConstraints::uniform(2, Some(0.0), Some(3.0));
        let result = projected_gradient(f, grad, &[1.0, 1.0], &bounds, None)
            .expect("Projected gradient should succeed");

        assert!(result.success, "Projected gradient should converge");
        // Should find x = y = 3 (closest feasible point to unconstrained minimum at (5,5))
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-2);
    }

    #[test]
    fn test_projected_gradient_one_sided() {
        // Minimize f(x) = x^2 subject to x >= 2
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64]| vec![2.0 * x[0]];

        let bounds = BoxConstraints {
            lower: vec![Some(2.0)],
            upper: vec![None],
        };

        let result = projected_gradient(f, grad, &[3.0], &bounds, None)
            .expect("Projected gradient should succeed for one-sided bounds");
        assert!(result.success);
        // Should find x = 2.0 (boundary of feasible region)
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_method_equality() {
        // Minimize f(x,y) = x^2 + y^2 subject to x + y = 1
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let eq_const_fn = |x: &[f64]| x[0] + x[1] - 1.0;
        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&eq_const_fn];

        let result = penalty_method(f, grad, &eq_const, &[], &[0.5, 0.5], 1.0, 10.0, None)
            .expect("Penalty method should succeed for equality constraint");

        assert!(result.success, "Penalty method should converge");
        // Solution should be approximately x = y = 0.5 (on the constraint line)
        assert_relative_eq!(result.x[0] + result.x[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[0], 0.5, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_method_inequality() {
        // Minimize f(x) = (x-3)^2 subject to x <= 2
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 3.0)];
        let ineq_const_fn = |x: &[f64]| x[0] - 2.0; // x <= 2 means x - 2 <= 0
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&ineq_const_fn];

        // Use higher initial penalty for tighter constraint satisfaction
        let result = penalty_method(f, grad, &[], &ineq_const, &[1.0], 10.0, 10.0, None)
            .expect("Penalty method should succeed for inequality constraint");

        assert!(result.success);
        // Should find x ≈ 2.0 (boundary of feasible region)
        // Penalty methods give approximate solutions, allow some tolerance
        assert_relative_eq!(result.x[0], 2.0, epsilon = 0.2);
        assert!(
            result.x[0] <= 2.1,
            "Should not violate constraint significantly"
        );
    }

    #[test]
    fn test_penalty_method_mixed_constraints() {
        // Minimize f(x,y) = x^2 + y^2
        // Subject to: x + y = 1 (equality) and x >= 0, y >= 0 (inequality)
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let eq_const_fn = |x: &[f64]| x[0] + x[1] - 1.0;
        let ineq_const_fn1 = |x: &[f64]| -x[0]; // x >= 0
        let ineq_const_fn2 = |x: &[f64]| -x[1]; // y >= 0

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&eq_const_fn];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&ineq_const_fn1, &ineq_const_fn2];

        let result = penalty_method(
            f,
            grad,
            &eq_const,
            &ineq_const,
            &[0.5, 0.5],
            1.0,
            10.0,
            None,
        )
        .expect("Penalty method should succeed for mixed constraints");

        assert!(result.success);
        // Both should be approximately 0.5
        assert!(result.x[0] >= -1e-3, "x should be non-negative");
        assert!(result.x[1] >= -1e-3, "y should be non-negative");
        assert_relative_eq!(result.x[0] + result.x[1], 1.0, epsilon = 1e-3);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_bfgs_quadratic_convergence(
            a in -10.0f64..10.0,
            b in -10.0f64..10.0,
            x0 in -5.0f64..5.0,
            y0 in -5.0f64..5.0
        ) {
            // Any quadratic f(x,y) = (x-a)^2 + (y-b)^2 should converge to (a,b)
            let f = |x: &[f64]| (x[0] - a).powi(2) + (x[1] - b).powi(2);
            let grad = |x: &[f64]| vec![2.0 * (x[0] - a), 2.0 * (x[1] - b)];

            let result = bfgs(f, grad, &[x0, y0], None).expect("BFGS should succeed for quadratic");
            prop_assert!(result.success, "BFGS should always converge for quadratic");
            prop_assert!((result.x[0] - a).abs() < 1e-3);
            prop_assert!((result.x[1] - b).abs() < 1e-3);
            prop_assert!(result.fun < 1e-6);
        }

        #[test]
        fn prop_lbfgs_sphere_convergence(
            dim in 2usize..6,
            seed in 0usize..100
        ) {
            // Minimize sum of squares in varying dimensions
            let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
            let grad = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect();

            // Generate random starting point
            let x0: Vec<f64> = (0..dim).map(|i| ((i + seed) as f64 * 0.37) % 5.0 - 2.5).collect();

            let result = lbfgs(f, grad, &x0, 5, None).expect("L-BFGS should succeed for sphere function");
            prop_assert!(result.success, "L-BFGS should converge for sphere function");
            for &xi in &result.x {
                prop_assert!(xi.abs() < 1e-3, "All components should be near zero");
            }
        }

        #[test]
        fn prop_box_constraints_projection_properties(
            x in -10.0f64..10.0,
            y in -10.0f64..10.0,
            lb in 0.0f64..2.0,
            ub in 3.0f64..5.0
        ) {
            let bounds = BoxConstraints::uniform(2, Some(lb), Some(ub));
            let point = vec![x, y];
            let projected = bounds.project(&point);

            // Property 1: Projection should be feasible
            prop_assert!(bounds.is_feasible(&projected));

            // Property 2: Projection should be within bounds
            prop_assert!(projected[0] >= lb && projected[0] <= ub);
            prop_assert!(projected[1] >= lb && projected[1] <= ub);

            // Property 3: If original point is feasible, projection is identity
            if bounds.is_feasible(&point) {
                prop_assert!((projected[0] - point[0]).abs() < 1e-10);
                prop_assert!((projected[1] - point[1]).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_nelder_mead_local_improvement(
            x0 in -5.0f64..5.0,
            y0 in -5.0f64..5.0
        ) {
            // Nelder-Mead should always improve or maintain objective value
            let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
            let initial_val = f(&[x0, y0]);

            let cfg = OptimizeConfig {
                max_iter: 100,
                ..Default::default()
            };

            let result = nelder_mead(f, &[x0, y0], Some(cfg)).expect("Nelder-Mead should succeed for local improvement test");

            // Final value should be <= initial value (monotonic improvement)
            prop_assert!(result.fun <= initial_val + 1e-6);
        }

        #[test]
        fn prop_wolfe_conditions_satisfaction(
            a in 1.0f64..10.0,
            b in 1.0f64..10.0
        ) {
            // Test that Wolfe line search satisfies sufficient decrease
            let f = |x: &[f64]| a * x[0] * x[0] + b * x[1] * x[1];
            let grad = |x: &[f64]| vec![2.0 * a * x[0], 2.0 * b * x[1]];

            let x = vec![3.0, 4.0];
            let p = vec![-1.0, -1.0]; // Descent direction
            let f0 = f(&x);
            let g0 = grad(&x);

            let cfg = OptimizeConfig::default();
            let result = gradient::wolfe_line_search(&f, &grad, &x, &p, f0, &g0, &cfg);

            if let Ok((alpha, f_new, _)) = result {
                // Verify sufficient decrease (Armijo condition)
                let dg: f64 = g0.iter().zip(p.iter()).map(|(&gi, &pi)| gi * pi).sum();
                prop_assert!(f_new <= f0 + cfg.c1 * alpha * dg);
            }
        }
    }

    // =========================================================================
    // Trust Region Tests
    // =========================================================================

    #[test]
    fn test_trust_region_quadratic() {
        // Minimize f(x,y) = x^2 + y^2
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let hess = |_x: &[f64]| vec![vec![2.0, 0.0], vec![0.0, 2.0]];

        let result = trust_region(f, grad, hess, &[3.0, 4.0], None)
            .expect("Trust region should succeed for quadratic");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_trust_region_rosenbrock() {
        // Rosenbrock function with analytical gradient and Hessian
        let f = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };

        let grad = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            vec![
                -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
                200.0 * (x1 - x0 * x0),
            ]
        };

        let hess = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            let h11 = 2.0 - 400.0 * (x1 - x0 * x0) + 800.0 * x0 * x0;
            let h12 = -400.0 * x0;
            let h22 = 200.0;
            vec![vec![h11, h12], vec![h12, h22]]
        };

        let result = trust_region(f, grad, hess, &[0.0, 0.0], None)
            .expect("Trust region should succeed for Rosenbrock");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_levenberg_marquardt_linear() {
        // Fit linear model y = mx + c to data
        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = [1.0, 3.0, 5.0, 7.0, 9.0]; // Perfect line y = 2x + 1

        let residual = |params: &[f64]| -> Vec<f64> {
            let (m, c) = (params[0], params[1]);
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&xi, &yi)| m * xi + c - yi)
                .collect()
        };

        let result = levenberg_marquardt(residual, &[0.0, 0.0], None)
            .expect("Levenberg-Marquardt should succeed for linear fit");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-4); // Slope
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-4); // Intercept
    }

    #[test]
    fn test_levenberg_marquardt_exponential() {
        // Fit exponential decay: y = A * exp(-k*x)
        let x_data = [0.0, 1.0, 2.0, 3.0];
        let y_data = [2.0, 0.736, 0.271, 0.100]; // A=2, k=1

        let residual = |params: &[f64]| -> Vec<f64> {
            let (a, k) = (params[0], params[1]);
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&xi, &yi)| a * (-k * xi).exp() - yi)
                .collect()
        };

        let result = levenberg_marquardt(residual, &[1.5, 0.8], None)
            .expect("Levenberg-Marquardt should succeed for exponential fit");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-1);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-1);
    }

    #[test]
    fn test_check_gradient_accuracy() {
        // Verify gradient checker works
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 4.0 * x[1]];

        let x = vec![3.0, 4.0];
        let is_correct = check_gradient(&f, &grad, &x, 1e-6);
        assert!(is_correct, "Gradient should be verified as correct");
    }

    #[test]
    fn test_check_gradient_detects_error() {
        // Verify gradient checker detects incorrect gradient
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let wrong_grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]]; // Wrong coefficient!

        let x = vec![3.0, 4.0];
        let is_correct = check_gradient(&f, &wrong_grad, &x, 1e-3);
        assert!(!is_correct, "Gradient checker should detect error");
    }
}
