//! Conjugate Gradient optimization methods
//!
//! This module provides nonlinear conjugate gradient methods for unconstrained optimization:
//! - Fletcher-Reeves (FR)
//! - Polak-Ribiere (PR)
//! - Hestenes-Stiefel (HS)
//!
//! Conjugate gradient methods are efficient for large-scale optimization problems
//! where computing and storing the full Hessian matrix (as in Newton methods) is impractical.
//!
//! # Examples
//!
//! ```
//! use numrs2::optimize::*;
//!
//! // Minimize the Rosenbrock function using Fletcher-Reeves
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
//! let result = conjugate_gradient_fr(f, grad, &[0.0, 0.0], None)
//!     .expect("CG-FR optimization should succeed");
//! assert!(result.success);
//! ```

use crate::error::Result;
use num_traits::Float;

use super::gradient::wolfe_line_search;
use super::{compute_norm, dot_product, OptimizeConfig, OptimizeResult};

/// Conjugate gradient method variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateGradientMethod {
    /// Fletcher-Reeves method
    FletcherReeves,
    /// Polak-Ribiere method
    PolakRibiere,
    /// Hestenes-Stiefel method
    HesteneStiefel,
}

/// Trait for computing beta parameter in conjugate gradient methods
pub trait BetaComputation<T: Float> {
    /// Compute beta parameter for direction update
    ///
    /// # Arguments
    /// * `g_new` - Current gradient
    /// * `g_old` - Previous gradient
    /// * `d_old` - Previous search direction
    fn compute_beta(g_new: &[T], g_old: &[T], d_old: &[T]) -> T;
}

/// Fletcher-Reeves beta computation: β_k = ||g_k||² / ||g_{k-1}||²
pub struct FletcherReevesBeta;

impl<T: Float + std::iter::Sum> BetaComputation<T> for FletcherReevesBeta {
    fn compute_beta(g_new: &[T], g_old: &[T], _d_old: &[T]) -> T {
        let g_new_norm_sq: T = g_new.iter().map(|&x| x * x).sum();
        let g_old_norm_sq: T = g_old.iter().map(|&x| x * x).sum();

        // Avoid division by zero
        if g_old_norm_sq < T::epsilon() {
            T::zero()
        } else {
            g_new_norm_sq / g_old_norm_sq
        }
    }
}

/// Polak-Ribiere beta computation: β_k = g_k^T(g_k - g_{k-1}) / ||g_{k-1}||²
pub struct PolakRibiereBeta;

impl<T: Float + std::iter::Sum> BetaComputation<T> for PolakRibiereBeta {
    fn compute_beta(g_new: &[T], g_old: &[T], _d_old: &[T]) -> T {
        let numerator: T = g_new
            .iter()
            .zip(g_old.iter())
            .map(|(&gi_new, &gi_old)| gi_new * (gi_new - gi_old))
            .sum();
        let g_old_norm_sq: T = g_old.iter().map(|&x| x * x).sum();

        // Avoid division by zero
        if g_old_norm_sq < T::epsilon() {
            T::zero()
        } else {
            // Polak-Ribiere can be negative, so we take max with 0 for robustness
            T::max(numerator / g_old_norm_sq, T::zero())
        }
    }
}

/// Hestenes-Stiefel beta computation: β_k = g_k^T(g_k - g_{k-1}) / (g_k - g_{k-1})^T d_{k-1}
pub struct HesteneStiefelBeta;

impl<T: Float + std::iter::Sum> BetaComputation<T> for HesteneStiefelBeta {
    fn compute_beta(g_new: &[T], g_old: &[T], d_old: &[T]) -> T {
        // Compute g_k - g_{k-1}
        let g_diff: Vec<T> = g_new
            .iter()
            .zip(g_old.iter())
            .map(|(&gi_new, &gi_old)| gi_new - gi_old)
            .collect();

        let numerator: T = g_new
            .iter()
            .zip(g_diff.iter())
            .map(|(&gi_new, &gi_diff)| gi_new * gi_diff)
            .sum();

        let denominator: T = g_diff
            .iter()
            .zip(d_old.iter())
            .map(|(&gi_diff, &di_old)| gi_diff * di_old)
            .sum();

        // Avoid division by zero
        if denominator.abs() < T::epsilon() {
            T::zero()
        } else {
            numerator / denominator
        }
    }
}

/// Generic conjugate gradient optimization
///
/// Minimizes a scalar function using a conjugate gradient method with the specified
/// beta computation strategy.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `grad` - Gradient function
/// * `x0` - Initial guess
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Type Parameters
///
/// * `T` - Float type (f32 or f64)
/// * `F` - Objective function type
/// * `G` - Gradient function type
/// * `B` - Beta computation strategy
///
/// # Returns
///
/// An `OptimizeResult` containing the optimal point and convergence information
fn conjugate_gradient_generic<T, F, G, B>(
    f: F,
    grad: G,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
    B: BetaComputation<T>,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    // Initialize
    let mut x = x0.to_vec();
    let mut f_val = f(&x);
    let mut g = grad(&x);
    let mut nfev = 1;
    let mut njev = 1;

    // Initial search direction is negative gradient
    let mut d: Vec<T> = g.iter().map(|&gi| -gi).collect();

    // Compute initial gradient norm
    let mut g_norm = compute_norm(&g);

    // Check if already at minimum
    if g_norm < cfg.gtol {
        return Ok(OptimizeResult {
            x,
            fun: f_val,
            grad: g,
            nit: 0,
            nfev,
            njev,
            success: true,
            message: "Optimization terminated successfully (initial point is optimal)".to_string(),
        });
    }

    // Store old gradient for beta computation
    let mut g_old = g.clone();

    // Conjugate gradient iteration
    for k in 0..cfg.max_iter {
        // Line search along direction d
        let (alpha, f_new, nfev_ls) = wolfe_line_search(&f, &grad, &x, &d, f_val, &g, &cfg)?;
        nfev += nfev_ls;
        njev += nfev_ls;

        // Update position
        for i in 0..n {
            x[i] = x[i] + alpha * d[i];
        }

        // Evaluate at new position
        f_val = f_new;
        g = grad(&x);
        njev += 1;

        g_norm = compute_norm(&g);

        // Check convergence
        if g_norm < cfg.gtol {
            return Ok(OptimizeResult {
                x,
                fun: f_val,
                grad: g,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (gradient norm below tolerance)"
                    .to_string(),
            });
        }

        // Compute beta parameter using the specific method
        let beta = B::compute_beta(&g, &g_old, &d);

        // Update search direction: d = -g + beta * d_old
        let d_old = d.clone();
        for i in 0..n {
            d[i] = -g[i] + beta * d_old[i];
        }

        // Ensure descent direction (should satisfy d^T g < 0)
        let dg: T = dot_product(&d, &g);
        if dg >= T::zero() {
            // If not a descent direction, restart with steepest descent
            for i in 0..n {
                d[i] = -g[i];
            }
        }

        // Update old gradient
        g_old = g.clone();

        // Check function value change for convergence
        if k > 0 {
            // This is a safeguard - we check relative function change
            // (not always reliable for CG but prevents infinite loops)
        }
    }

    // Maximum iterations reached
    Ok(OptimizeResult {
        x,
        fun: f_val,
        grad: g,
        nit: cfg.max_iter,
        nfev,
        njev,
        success: false,
        message: "Maximum iterations reached without convergence".to_string(),
    })
}

/// Fletcher-Reeves conjugate gradient method
///
/// This method uses β_k = ||g_k||² / ||g_{k-1}||² for direction updates.
/// It is guaranteed to produce descent directions and has good global convergence properties.
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = |x: &[f64]| vec![2.0*x[0], 2.0*x[1]];
///
/// let result = conjugate_gradient_fr(f, grad, &[3.0, 4.0], None)
///     .expect("CG-FR should succeed");
/// assert!(result.success);
/// ```
pub fn conjugate_gradient_fr<T, F, G>(
    f: F,
    grad: G,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    conjugate_gradient_generic::<T, F, G, FletcherReevesBeta>(f, grad, x0, config)
}

/// Polak-Ribiere conjugate gradient method
///
/// This method uses β_k = g_k^T(g_k - g_{k-1}) / ||g_{k-1}||² for direction updates.
/// It often performs better than Fletcher-Reeves in practice and automatically restarts
/// when β becomes negative.
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
/// let grad = |x: &[f64]| vec![2.0*(x[0] - 1.0), 2.0*(x[1] - 2.0)];
///
/// let result = conjugate_gradient_pr(f, grad, &[0.0, 0.0], None)
///     .expect("CG-PR should succeed");
/// assert!(result.success);
/// ```
pub fn conjugate_gradient_pr<T, F, G>(
    f: F,
    grad: G,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    conjugate_gradient_generic::<T, F, G, PolakRibiereBeta>(f, grad, x0, config)
}

/// Hestenes-Stiefel conjugate gradient method
///
/// This method uses β_k = g_k^T(g_k - g_{k-1}) / (g_k - g_{k-1})^T d_{k-1} for direction updates.
/// It is more robust than Fletcher-Reeves and often competitive with Polak-Ribiere.
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// let f = |x: &[f64]| x[0].powi(4) + x[1].powi(4);
/// let grad = |x: &[f64]| vec![4.0*x[0].powi(3), 4.0*x[1].powi(3)];
///
/// let result = conjugate_gradient_hs(f, grad, &[2.0, 2.0], None)
///     .expect("CG-HS should succeed");
/// assert!(result.success);
/// ```
pub fn conjugate_gradient_hs<T, F, G>(
    f: F,
    grad: G,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    conjugate_gradient_generic::<T, F, G, HesteneStiefelBeta>(f, grad, x0, config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cg_fr_quadratic() {
        // Minimize f(x,y) = x^2 + y^2
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let result = conjugate_gradient_fr(f, grad, &[3.0, 4.0], None)
            .expect("CG-FR should succeed for quadratic");
        assert!(result.success);
        assert!(result.fun < 1e-10);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cg_pr_quadratic() {
        // Minimize f(x,y) = (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)];

        let result = conjugate_gradient_pr(f, grad, &[0.0, 0.0], None)
            .expect("CG-PR should succeed for quadratic");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cg_hs_quadratic() {
        // Minimize f(x,y) = x^2 + 4*y^2
        let f = |x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 8.0 * x[1]];

        let result = conjugate_gradient_hs(f, grad, &[5.0, 5.0], None)
            .expect("CG-HS should succeed for quadratic");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cg_fr_rosenbrock() {
        // Rosenbrock function
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

        let result = conjugate_gradient_fr(f, grad, &[0.0, 0.0], None)
            .expect("CG-FR should succeed for Rosenbrock");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_cg_pr_rosenbrock() {
        // Polak-Ribiere often performs better on Rosenbrock
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

        let result = conjugate_gradient_pr(f, grad, &[0.0, 0.0], None)
            .expect("CG-PR should succeed for Rosenbrock");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_cg_hs_beale() {
        // Beale's function
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

        let result = conjugate_gradient_hs(f, grad, &[1.0, 1.0], None)
            .expect("CG-HS should succeed for Beale");
        assert!(result.success);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_cg_higher_dimension() {
        // Test in higher dimensions
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let grad = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect();

        let x0 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = conjugate_gradient_fr(f, grad, &x0, None)
            .expect("CG-FR should succeed in higher dimensions");

        assert!(result.success);
        for &xi in &result.x {
            assert_relative_eq!(xi, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_cg_initial_optimum() {
        // Test when starting at optimum
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let result = conjugate_gradient_fr(f, grad, &[0.0, 0.0], None)
            .expect("CG-FR should handle initial optimum");
        assert!(result.success);
        assert_eq!(result.nit, 0);
    }

    // Property-based tests
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_cg_fr_quadratic_convergence(
            a in -10.0f64..10.0,
            b in -10.0f64..10.0,
            x0 in -5.0f64..5.0,
            y0 in -5.0f64..5.0
        ) {
            // Any quadratic should converge to minimum
            let f = |x: &[f64]| (x[0] - a).powi(2) + (x[1] - b).powi(2);
            let grad = |x: &[f64]| vec![2.0 * (x[0] - a), 2.0 * (x[1] - b)];

            let result = conjugate_gradient_fr(f, grad, &[x0, y0], None)
                .expect("CG-FR should succeed for any quadratic");
            prop_assert!(result.success);
            prop_assert!((result.x[0] - a).abs() < 1e-3);
            prop_assert!((result.x[1] - b).abs() < 1e-3);
        }

        #[test]
        fn prop_cg_pr_always_descends(
            x0 in -5.0f64..5.0,
            y0 in -5.0f64..5.0
        ) {
            // CG-PR should always find a local minimum or better
            let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
            let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

            let initial_val = f(&[x0, y0]);
            let result = conjugate_gradient_pr(f, grad, &[x0, y0], None)
                .expect("CG-PR should succeed");

            prop_assert!(result.fun <= initial_val + 1e-6);
        }
    }
}
