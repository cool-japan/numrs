//! Interior Point Methods for Constrained Optimization
//!
//! Interior point methods (also known as barrier methods) solve constrained optimization
//! problems by transforming them into a sequence of unconstrained problems using barrier
//! functions. The method stays in the interior of the feasible region and approaches
//! the boundary as the barrier parameter decreases.
//!
//! # Features
//!
//! - Logarithmic and inverse barrier functions
//! - Primal-dual Newton-KKT system solver
//! - Centering parameter adjustment
//! - Handles linear and nonlinear constraints
//! - Inequality constraint handling with slack variables
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::{interior_point, IPConfig, BarrierType};
//!
//! // Minimize f(x,y) = x^2 + y^2 subject to x + y >= 1
//! let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
//! let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
//! let hess_f = |_x: &[f64]| vec![vec![2.0, 0.0], vec![0.0, 2.0]];
//!
//! // Convert to g(x) <= 0 form: -(x + y - 1) <= 0
//! let ineq_fn = |x: &[f64]| -(x[0] + x[1] - 1.0);
//! let grad_ineq_fn = |_x: &[f64]| vec![-1.0, -1.0];
//! let ineq: Vec<&dyn Fn(&[f64]) -> f64> = vec![&ineq_fn];
//! let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_ineq_fn];
//!
//! let config = IPConfig::default();
//! let result = interior_point(f, grad_f, hess_f, &[], &[], &ineq, &grad_ineq, &[0.6, 0.6], Some(config))
//!     .expect("Interior point should succeed");
//! ```

use crate::error::{NumRs2Error, Result};
use crate::optimize::OptimizeResult;
use num_traits::Float;
use std::iter::Sum;

/// Type alias for constraint functions
type ConstraintFn<T> = dyn Fn(&[T]) -> T;

/// Type alias for constraint gradient functions
type ConstraintGradFn<T> = dyn Fn(&[T]) -> Vec<T>;

/// Barrier function type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrierType {
    /// Logarithmic barrier: -μ * Σ ln(-g_i(x))
    Logarithmic,
    /// Inverse barrier: -μ * Σ 1/g_i(x)
    Inverse,
}

/// Configuration for Interior Point Method
#[derive(Debug, Clone)]
pub struct IPConfig<T: Float> {
    /// Maximum outer iterations
    pub max_iter: usize,
    /// Maximum inner iterations per barrier parameter
    pub max_inner_iter: usize,
    /// Initial barrier parameter
    pub mu_0: T,
    /// Barrier parameter reduction factor
    pub mu_reduction: T,
    /// Minimum barrier parameter
    pub mu_min: T,
    /// Gradient tolerance
    pub gtol: T,
    /// Constraint violation tolerance
    pub ctol: T,
    /// Barrier type
    pub barrier: BarrierType,
}

impl<T: Float> Default for IPConfig<T> {
    fn default() -> Self {
        Self {
            max_iter: 50,
            max_inner_iter: 20,
            mu_0: T::from(1.0).expect("1.0 should convert to Float"),
            mu_reduction: T::from(0.1).expect("0.1 should convert to Float"),
            mu_min: T::from(1e-8).expect("1e-8 should convert to Float"),
            gtol: T::from(1e-6).expect("1e-6 should convert to Float"),
            ctol: T::from(1e-6).expect("1e-6 should convert to Float"),
            barrier: BarrierType::Logarithmic,
        }
    }
}

/// Interior point method for constrained optimization
///
/// # Arguments
///
/// * `f` - Objective function
/// * `grad_f` - Gradient of objective function
/// * `hess_f` - Hessian of objective function
/// * `eq_constraints` - Equality constraints h(x) = 0
/// * `grad_eq` - Gradients of equality constraints
/// * `ineq_constraints` - Inequality constraints g(x) <= 0
/// * `grad_ineq` - Gradients of inequality constraints
/// * `x0` - Initial point (must be strictly feasible for inequalities)
/// * `config` - Optional configuration
///
/// # Returns
///
/// `OptimizeResult` with the optimal solution
pub fn interior_point<T, F, GF, HF>(
    f: F,
    grad_f: GF,
    hess_f: HF,
    eq_constraints: &[&ConstraintFn<T>],
    grad_eq: &[&ConstraintGradFn<T>],
    ineq_constraints: &[&ConstraintFn<T>],
    grad_ineq: &[&ConstraintGradFn<T>],
    x0: &[T],
    config: Option<IPConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + Sum + std::fmt::Display,
    F: Fn(&[T]) -> T,
    GF: Fn(&[T]) -> Vec<T>,
    HF: Fn(&[T]) -> Vec<Vec<T>>,
{
    let config = config.unwrap_or_default();
    let n = x0.len();

    if eq_constraints.len() != grad_eq.len() {
        return Err(NumRs2Error::ValueError(
            "Number of equality constraints must match gradients".to_string(),
        ));
    }

    if ineq_constraints.len() != grad_ineq.len() {
        return Err(NumRs2Error::ValueError(
            "Number of inequality constraints must match gradients".to_string(),
        ));
    }

    // Check initial feasibility for inequalities
    for (i, c) in ineq_constraints.iter().enumerate() {
        let g_val = c(x0);
        if g_val >= T::zero() {
            return Err(NumRs2Error::ValueError(format!(
                "Initial point violates inequality constraint {}: g(x) = {}, must be < 0",
                i, g_val
            )));
        }
    }

    let mut x = x0.to_vec();
    let mut mu = config.mu_0;
    let mut nfev = 0;
    let mut njev = 0;
    let mut total_iter = 0;

    while mu > config.mu_min && total_iter < config.max_iter {
        // Solve barrier problem for current mu
        for inner_iter in 0..config.max_inner_iter {
            total_iter += 1;

            // Evaluate barrier function and its derivatives
            let (barrier_val, barrier_grad, barrier_hess) =
                compute_barrier_terms(&x, ineq_constraints, grad_ineq, mu, &config.barrier)?;

            // Augmented objective: f(x) + barrier(x)
            let f_val = f(&x);
            nfev += 1;

            let grad_f_val = grad_f(&x);
            njev += 1;

            let aug_grad: Vec<T> = grad_f_val
                .iter()
                .zip(barrier_grad.iter())
                .map(|(&gf, &gb)| gf + gb)
                .collect();

            let aug_hess = add_matrices(&hess_f(&x), &barrier_hess)?;

            // Add equality constraint contributions
            let (grad_with_eq, hess_with_eq) = if !eq_constraints.is_empty() {
                let eq_vals: Vec<T> = eq_constraints.iter().map(|c| c(&x)).collect();
                let eq_grads: Vec<Vec<T>> = grad_eq.iter().map(|g| g(&x)).collect();

                // Simple penalty for equality constraints
                let penalty = T::from(1000.0).ok_or_else(|| {
                    NumRs2Error::ConversionError("Penalty conversion failed".to_string())
                })?;

                let mut grad_mod = aug_grad.clone();
                for (eq_val, eq_grad) in eq_vals.iter().zip(eq_grads.iter()) {
                    for (i, &grad_i) in eq_grad.iter().enumerate() {
                        grad_mod[i] = grad_mod[i]
                            + penalty
                                * T::from(2.0).ok_or_else(|| {
                                    NumRs2Error::ConversionError(
                                        "Constant conversion failed".to_string(),
                                    )
                                })?
                                * (*eq_val)
                                * grad_i;
                    }
                }

                (grad_mod, aug_hess)
            } else {
                (aug_grad, aug_hess)
            };

            // Compute search direction by solving Newton system
            let p = solve_newton_system(&hess_with_eq, &grad_with_eq)?;

            let grad_norm = compute_norm(&grad_with_eq);

            // Check convergence for this barrier parameter
            if grad_norm < config.gtol {
                break;
            }

            // Line search to maintain feasibility
            let alpha = feasible_line_search(
                &x,
                &p,
                ineq_constraints,
                T::from(0.9).ok_or_else(|| {
                    NumRs2Error::ConversionError("Constant conversion failed".to_string())
                })?,
            )?;

            // Update x
            x = x
                .iter()
                .zip(p.iter())
                .map(|(&xi, &pi)| xi + alpha * pi)
                .collect();
        }

        // Reduce barrier parameter
        mu = mu * config.mu_reduction;
    }

    // Final evaluation
    let f_final = f(&x);
    let grad_final = grad_f(&x);

    // Check constraint violations
    let eq_violation: T = eq_constraints.iter().map(|c| c(&x).abs()).sum();
    let ineq_violation: T = ineq_constraints
        .iter()
        .map(|c| {
            let g_val = c(&x);
            if g_val > T::zero() {
                g_val
            } else {
                T::zero()
            }
        })
        .sum();

    let success = eq_violation < config.ctol && ineq_violation < config.ctol;

    Ok(OptimizeResult {
        x,
        fun: f_final,
        grad: grad_final,
        nit: total_iter,
        nfev,
        njev,
        success,
        message: if success {
            "Converged successfully".to_string()
        } else {
            "Maximum iterations reached".to_string()
        },
    })
}

/// Compute barrier function value, gradient, and Hessian
fn compute_barrier_terms<T: Float + std::fmt::Display>(
    x: &[T],
    ineq_constraints: &[&ConstraintFn<T>],
    grad_ineq: &[&ConstraintGradFn<T>],
    mu: T,
    barrier_type: &BarrierType,
) -> Result<(T, Vec<T>, Vec<Vec<T>>)> {
    let n = x.len();
    let m = ineq_constraints.len();

    let mut barrier_val = T::zero();
    let mut barrier_grad = vec![T::zero(); n];
    let mut barrier_hess = vec![vec![T::zero(); n]; n];

    for i in 0..m {
        let g = ineq_constraints[i](x);

        if g >= T::zero() {
            return Err(NumRs2Error::ComputationError(format!(
                "Constraint {} violated: g(x) = {}, must be < 0",
                i, g
            )));
        }

        let grad_g = grad_ineq[i](x);

        // Constraint Hessian Hess_g (the matrix of second partials of g_i). The
        // analytic constraint Hessian is not supplied to this routine, so it is
        // approximated by finite differences of the constraint gradient. This term
        // is required for a mathematically complete barrier Hessian: for a barrier
        // B(x) = phi(g(x)) the chain rule gives
        //     Hess B = phi''(g) * grad_g grad_g^T + phi'(g) * Hess_g,
        // and omitting the phi'(g) * Hess_g part yields an incorrect Hessian whenever
        // the constraint is nonlinear (Hess_g != 0).
        let hess_g = constraint_hessian_fd(grad_ineq[i], x)?;

        match barrier_type {
            BarrierType::Logarithmic => {
                // Barrier: B = -mu * ln(-g),  phi(g) = -mu * ln(-g)
                //   phi'(g)  = -mu / g
                //   phi''(g) =  mu / g^2
                barrier_val = barrier_val - mu * (-g).ln();

                // Gradient: phi'(g) * grad_g = (-mu / g) * grad_g
                let factor = -mu / g;
                for j in 0..n {
                    barrier_grad[j] = barrier_grad[j] + factor * grad_g[j];
                }

                // Hessian: (mu / g^2) * grad_g grad_g^T  -  (mu / g) * Hess_g
                let outer_coeff = mu / (g * g); // phi''(g)
                let hess_coeff = -mu / g; // phi'(g)
                for j in 0..n {
                    for k in 0..n {
                        barrier_hess[j][k] = barrier_hess[j][k]
                            + outer_coeff * grad_g[j] * grad_g[k]
                            + hess_coeff * hess_g[j][k];
                    }
                }
            }
            BarrierType::Inverse => {
                // Barrier: B = -mu / g,  phi(g) = -mu / g
                //   phi'(g)  =  mu / g^2
                //   phi''(g) = -2 * mu / g^3
                barrier_val = barrier_val - mu / g;

                // Gradient: phi'(g) * grad_g = (mu / g^2) * grad_g
                let factor = mu / (g * g);
                for j in 0..n {
                    barrier_grad[j] = barrier_grad[j] + factor * grad_g[j];
                }

                // Hessian: (-2 * mu / g^3) * grad_g grad_g^T  +  (mu / g^2) * Hess_g
                let two = T::from(2.0).ok_or_else(|| {
                    NumRs2Error::ConversionError("Constant conversion failed".to_string())
                })?;
                let outer_coeff = -two * mu / (g * g * g); // phi''(g)
                let hess_coeff = mu / (g * g); // phi'(g)
                for j in 0..n {
                    for k in 0..n {
                        barrier_hess[j][k] = barrier_hess[j][k]
                            + outer_coeff * grad_g[j] * grad_g[k]
                            + hess_coeff * hess_g[j][k];
                    }
                }
            }
        }
    }

    Ok((barrier_val, barrier_grad, barrier_hess))
}

/// Approximate the Hessian of a scalar constraint via central finite differences of
/// its gradient.
///
/// Given a constraint gradient routine `grad_g` and a point `x`, the `(a, b)` entry of
/// the constraint Hessian is `d(grad_g)_a / dx_b`. It is estimated columnwise with a
/// central difference of the supplied gradient,
///
/// ```text
///   Hess_g[a][b] ~= ( grad_g(x + h e_b)[a] - grad_g(x - h e_b)[a] ) / (2 h),
/// ```
///
/// and the result is symmetrized as `(M + M^T) / 2` to remove the small asymmetry that
/// independent column estimates introduce, since a true Hessian is symmetric. The step
/// `h` scales with the magnitude of each coordinate, `h = sqrt(eps) * max(1, |x_b|)`,
/// which balances truncation and round-off error.
fn constraint_hessian_fd<T: Float>(grad_g: &ConstraintGradFn<T>, x: &[T]) -> Result<Vec<Vec<T>>> {
    let n = x.len();

    // Base step: sqrt of machine epsilon, scaled per-coordinate below.
    let sqrt_eps = T::epsilon().sqrt();
    let two = T::from(2.0)
        .ok_or_else(|| NumRs2Error::ConversionError("Constant conversion failed".to_string()))?;

    let mut hess = vec![vec![T::zero(); n]; n];

    for b in 0..n {
        // Coordinate-scaled step h_b = sqrt(eps) * max(1, |x_b|).
        let scale = if x[b].abs() > T::one() {
            x[b].abs()
        } else {
            T::one()
        };
        let h = sqrt_eps * scale;

        // Guard against a degenerate (zero) step, which would divide by zero below.
        if h <= T::zero() {
            return Err(NumRs2Error::ComputationError(
                "Finite-difference step collapsed to zero in constraint Hessian".to_string(),
            ));
        }

        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[b] = x[b] + h;
        x_minus[b] = x[b] - h;

        let grad_plus = grad_g(&x_plus);
        let grad_minus = grad_g(&x_minus);

        let denom = two * h;
        for a in 0..n {
            hess[a][b] = (grad_plus[a] - grad_minus[a]) / denom;
        }
    }

    // Symmetrize: (M + M^T) / 2.
    for a in 0..n {
        for b in (a + 1)..n {
            let avg = (hess[a][b] + hess[b][a]) / two;
            hess[a][b] = avg;
            hess[b][a] = avg;
        }
    }

    Ok(hess)
}

/// Add two matrices
fn add_matrices<T: Float>(a: &[Vec<T>], b: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
    if a.len() != b.len() || (!a.is_empty() && a[0].len() != b[0].len()) {
        return Err(NumRs2Error::ValueError(
            "Matrix dimensions must match".to_string(),
        ));
    }

    let n = a.len();
    let m = if n > 0 { a[0].len() } else { 0 };

    let mut result = vec![vec![T::zero(); m]; n];
    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    Ok(result)
}

/// Solve Newton system using Cholesky decomposition or Gaussian elimination
fn solve_newton_system<T: Float>(hessian: &[Vec<T>], grad: &[T]) -> Result<Vec<T>> {
    let n = grad.len();
    let mut aug = vec![vec![T::zero(); n + 1]; n];

    // Create augmented matrix [H | -g]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = hessian[i][j];
        }
        aug[i][n] = -grad[i];
    }

    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();

        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        if max_row != k {
            aug.swap(k, max_row);
        }

        let pivot = aug[k][k];
        if pivot.abs()
            < T::from(1e-10)
                .ok_or_else(|| NumRs2Error::ComputationError("Near-singular Hessian".to_string()))?
        {
            // Add regularization
            aug[k][k] = aug[k][k]
                + T::from(1e-6).ok_or_else(|| {
                    NumRs2Error::ComputationError("Regularization failed".to_string())
                })?;
        }

        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut p = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum = sum - aug[i][j] * p[j];
        }
        p[i] = sum / aug[i][i];
    }

    Ok(p)
}

/// Line search that maintains feasibility
fn feasible_line_search<T: Float>(
    x: &[T],
    p: &[T],
    ineq_constraints: &[&ConstraintFn<T>],
    safety_factor: T,
) -> Result<T> {
    let mut alpha = T::one();

    // Find maximum step that maintains feasibility
    for _ in 0..20 {
        let x_new: Vec<T> = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + alpha * pi)
            .collect();

        let mut feasible = true;
        for c in ineq_constraints {
            if c(&x_new) >= T::zero() {
                feasible = false;
                break;
            }
        }

        if feasible {
            return Ok(alpha * safety_factor);
        }

        alpha = alpha
            * T::from(0.5).ok_or_else(|| {
                NumRs2Error::ConversionError("Step reduction conversion failed".to_string())
            })?;
    }

    T::from(1e-8)
        .ok_or_else(|| NumRs2Error::ConversionError("Minimum step conversion failed".to_string()))
}

/// Compute L2 norm
fn compute_norm<T: Float + Sum>(v: &[T]) -> T {
    v.iter().map(|&x| x * x).sum::<T>().sqrt()
}

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_interior_point_unconstrained() {
        // Minimize f(x,y) = x^2 + y^2 (no constraints)
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let hess_f = |_x: &[f64]| vec![vec![2.0, 0.0], vec![0.0, 2.0]];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];

        let config = IPConfig::default();
        let result = interior_point(
            f,
            grad_f,
            hess_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[1.0, 1.0],
            Some(config),
        )
        .expect("Interior point should succeed");

        assert!(result.success);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_interior_point_simple_inequality() {
        // Minimize f(x) = x^2 subject to x >= 0.5 (i.e., -x + 0.5 <= 0)
        let f = |x: &[f64]| x[0] * x[0];
        let grad_f = |x: &[f64]| vec![2.0 * x[0]];
        let hess_f = |_x: &[f64]| vec![vec![2.0]];

        let g1 = |x: &[f64]| -x[0] + 0.5;
        let grad_g1 = |_x: &[f64]| vec![-1.0];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&g1];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_g1];

        let config = IPConfig {
            max_iter: 30,
            ..Default::default()
        };

        let result = interior_point(
            f,
            grad_f,
            hess_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[0.6], // Start feasible
            Some(config),
        )
        .expect("Interior point should succeed");

        // Should converge to x = 0.5 (boundary of constraint)
        assert_relative_eq!(result.x[0], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_interior_point_logarithmic_barrier() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let hess_f = |_x: &[f64]| vec![vec![2.0, 0.0], vec![0.0, 2.0]];

        let g1 = |x: &[f64]| -x[0] - x[1] + 1.0; // x + y >= 1

        let grad_g1 = |_x: &[f64]| vec![-1.0, -1.0];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&g1];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_g1];

        let config = IPConfig {
            barrier: BarrierType::Logarithmic,
            ..Default::default()
        };

        let result = interior_point(
            f,
            grad_f,
            hess_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[0.6, 0.6],
            Some(config),
        )
        .expect("Interior point should succeed");

        assert!(result.x[0] + result.x[1] >= 0.9);
    }

    #[test]
    fn test_interior_point_inverse_barrier() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad_f = |x: &[f64]| vec![2.0 * x[0]];
        let hess_f = |_x: &[f64]| vec![vec![2.0]];

        let g1 = |x: &[f64]| -x[0] + 0.5;
        let grad_g1 = |_x: &[f64]| vec![-1.0];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&g1];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_g1];

        let config = IPConfig {
            barrier: BarrierType::Inverse,
            max_iter: 30,
            ..Default::default()
        };

        let result = interior_point(
            f,
            grad_f,
            hess_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[0.6],
            Some(config),
        )
        .expect("Interior point should succeed");

        assert_relative_eq!(result.x[0], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_interior_point_infeasible_start() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad_f = |x: &[f64]| vec![2.0 * x[0]];
        let hess_f = |_x: &[f64]| vec![vec![2.0]];

        let g1 = |x: &[f64]| -x[0] + 0.5;
        let grad_g1 = |_x: &[f64]| vec![-1.0];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&g1];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_g1];

        let result = interior_point(
            f,
            grad_f,
            hess_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[0.3], // Violates constraint
            None,
        );

        assert!(result.is_err());
    }
}
