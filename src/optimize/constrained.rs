//! Constrained optimization algorithms
//!
//! This module provides methods for constrained optimization:
//! - Box constraints (simple bounds)
//! - Projected gradient descent
//! - Penalty method for general constraints

use crate::error::{NumRs2Error, Result};
use num_traits::Float;

use super::gradient::bfgs;
use super::{compute_norm, OptimizeConfig, OptimizeResult};

/// Box constraints for optimization (simple bounds)
#[derive(Debug, Clone)]
pub struct BoxConstraints<T> {
    /// Lower bounds for each variable (None = -infinity)
    pub lower: Vec<Option<T>>,
    /// Upper bounds for each variable (None = +infinity)
    pub upper: Vec<Option<T>>,
}

impl<T: Float> BoxConstraints<T> {
    /// Create box constraints with same bounds for all variables
    pub fn uniform(n: usize, lower: Option<T>, upper: Option<T>) -> Self {
        Self {
            lower: vec![lower; n],
            upper: vec![upper; n],
        }
    }

    /// Project a point onto the feasible region
    pub fn project(&self, x: &[T]) -> Vec<T> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let mut val = xi;
                if let Some(lb) = self.lower[i] {
                    if val < lb {
                        val = lb;
                    }
                }
                if let Some(ub) = self.upper[i] {
                    if val > ub {
                        val = ub;
                    }
                }
                val
            })
            .collect()
    }

    /// Check if point is feasible
    pub fn is_feasible(&self, x: &[T]) -> bool {
        x.iter().enumerate().all(|(i, &xi)| {
            let lower_ok = self.lower[i].is_none_or(|lb| xi >= lb);
            let upper_ok = self.upper[i].is_none_or(|ub| xi <= ub);
            lower_ok && upper_ok
        })
    }
}

/// Projected gradient descent for box-constrained optimization
///
/// Minimizes f(x) subject to lower <= x <= upper using projected gradient descent.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `grad` - Gradient function
/// * `x0` - Initial guess (must be feasible)
/// * `constraints` - Box constraints
/// * `config` - Optional configuration
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// // Minimize f(x,y) = (x-5)^2 + (y-5)^2 subject to 0 <= x,y <= 3
/// let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
/// let grad = |x: &[f64]| vec![2.0 * (x[0] - 5.0), 2.0 * (x[1] - 5.0)];
///
/// let bounds = BoxConstraints::uniform(2, Some(0.0), Some(3.0));
/// let result = projected_gradient(f, grad, &[1.0, 1.0], &bounds, None).expect("Projected gradient optimization should succeed");
///
/// assert!(result.success);
/// // Should find x = y = 3 (closest feasible point to minimum at (5,5))
/// ```
pub fn projected_gradient<T, F, G>(
    f: F,
    grad: G,
    x0: &[T],
    constraints: &BoxConstraints<T>,
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    // Check initial point is feasible
    if !constraints.is_feasible(x0) {
        return Err(NumRs2Error::ValueError(
            "Initial point is not feasible".to_string(),
        ));
    }

    let mut x = x0.to_vec();
    let mut f_val = f(&x);
    let mut g = grad(&x);
    let mut nfev = 1;
    let mut njev = 1;

    // Check initial gradient
    let g_norm = compute_norm(&g);
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

    let mut alpha = T::from(0.01).expect("0.01 should be representable in Float"); // Initial step size

    // Projected gradient descent
    for k in 0..cfg.max_iter {
        // Compute projected gradient step: x_new = project(x - alpha * grad)
        let x_trial: Vec<T> = x
            .iter()
            .zip(g.iter())
            .map(|(&xi, &gi)| xi - alpha * gi)
            .collect();
        let x_new = constraints.project(&x_trial);

        let f_new = f(&x_new);
        nfev += 1;

        // Check for sufficient decrease (Armijo condition)
        let dx: Vec<T> = x_new
            .iter()
            .zip(x.iter())
            .map(|(&xi_new, &xi)| xi_new - xi)
            .collect();
        let grad_proj: T = g.iter().zip(dx.iter()).map(|(&gi, &dxi)| gi * dxi).sum();

        if f_new <= f_val + cfg.c1 * grad_proj {
            // Accept step
            let g_new = grad(&x_new);
            njev += 1;

            // Check convergence
            let dx_norm = compute_norm(&dx);
            let df = (f_new - f_val).abs();

            // Check projected gradient for convergence
            let x_pg_trial: Vec<T> = x_new
                .iter()
                .zip(g_new.iter())
                .map(|(&xi, &gi)| xi - gi)
                .collect();
            let x_pg = constraints.project(&x_pg_trial);
            let pg_norm = compute_norm(
                &x_pg
                    .iter()
                    .zip(x_new.iter())
                    .map(|(&xpg, &xi)| xpg - xi)
                    .collect::<Vec<_>>(),
            );

            if pg_norm < cfg.gtol {
                return Ok(OptimizeResult {
                    x: x_new,
                    fun: f_new,
                    grad: g_new,
                    nit: k + 1,
                    nfev,
                    njev,
                    success: true,
                    message: "Optimization terminated successfully (projected gradient converged)"
                        .to_string(),
                });
            }

            if dx_norm < cfg.xtol || df < cfg.ftol {
                return Ok(OptimizeResult {
                    x: x_new,
                    fun: f_new,
                    grad: g_new,
                    nit: k + 1,
                    nfev,
                    njev,
                    success: true,
                    message: "Optimization terminated successfully (parameters converged)"
                        .to_string(),
                });
            }

            // Update for next iteration
            x = x_new;
            f_val = f_new;
            g = g_new;

            // Increase step size slightly if making good progress
            alpha = alpha * T::from(1.05).expect("1.05 should be representable in Float");
        } else {
            // Reject step, decrease step size
            alpha = alpha * T::from(0.5).expect("0.5 should be representable in Float");

            if alpha < T::from(1e-12).expect("1e-12 should be representable in Float") {
                return Ok(OptimizeResult {
                    x,
                    fun: f_val,
                    grad: g,
                    nit: k + 1,
                    nfev,
                    njev,
                    success: false,
                    message: "Line search failed (step size too small)".to_string(),
                });
            }
        }
    }

    Ok(OptimizeResult {
        x,
        fun: f_val,
        grad: g,
        nit: cfg.max_iter,
        nfev,
        njev,
        success: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Penalty method for constrained optimization
///
/// Minimizes f(x) subject to constraints by adding penalty terms.
/// Converts constrained problem to a sequence of unconstrained problems.
///
/// # Arguments
///
/// * `f` - Objective function
/// * `grad` - Gradient function
/// * `equality_constraints` - Equality constraints c_eq(x) = 0
/// * `inequality_constraints` - Inequality constraints c_ineq(x) <= 0
/// * `x0` - Initial guess
/// * `penalty_factor` - Initial penalty parameter (e.g., 1.0)
/// * `penalty_increase` - Factor to increase penalty each iteration (e.g., 10.0)
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// // Minimize f(x,y) = x^2 + y^2 subject to x + y = 1
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = |x: &[f64]| vec![2.0*x[0], 2.0*x[1]];
/// let eq_const_fn = |x: &[f64]| x[0] + x[1] - 1.0;
/// let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&eq_const_fn];
///
/// let result = penalty_method(
///     f, grad, &eq_const, &[], &[0.5, 0.5], 1.0, 10.0, None
/// ).expect("Penalty method optimization should succeed");
/// assert!(result.success);
/// ```
#[allow(clippy::type_complexity)]
pub fn penalty_method<T, F, G>(
    f: F,
    grad: G,
    equality_constraints: &[&dyn Fn(&[T]) -> T],
    inequality_constraints: &[&dyn Fn(&[T]) -> T],
    x0: &[T],
    initial_penalty: T,
    penalty_increase: T,
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let mut cfg = config.unwrap_or_default();
    let mut mu = initial_penalty;
    let mut x = x0.to_vec();

    let mut total_nfev = 0;
    let mut total_njev = 0;
    let max_outer_iter = 20;

    for outer_iter in 0..max_outer_iter {
        // Create penalized objective function
        let f_penalized = |x_val: &[T]| {
            let mut val = f(x_val);

            // Add equality constraint penalties: mu * sum(c_eq^2)
            for c_eq in equality_constraints {
                let c_val = c_eq(x_val);
                val = val + mu * c_val * c_val;
            }

            // Add inequality constraint penalties: mu * sum(max(0, c_ineq)^2)
            for c_ineq in inequality_constraints {
                let c_val = c_ineq(x_val);
                if c_val > T::zero() {
                    val = val + mu * c_val * c_val;
                }
            }

            val
        };

        // Gradient of penalized function
        let grad_penalized = |x_val: &[T]| {
            let mut g_pen = grad(x_val);
            let n = x_val.len();
            let eps = T::from(1e-8).expect("1e-8 should be representable in Float");

            // Numerical gradient of penalty terms (could be analytical if provided)
            for c_eq in equality_constraints {
                let c_val = c_eq(x_val);
                for i in 0..n {
                    let mut x_plus = x_val.to_vec();
                    x_plus[i] = x_plus[i] + eps;
                    let c_plus = c_eq(&x_plus);
                    let dc_di = (c_plus - c_val) / eps;
                    g_pen[i] = g_pen[i]
                        + T::from(2.0).expect("2.0 should be representable in Float")
                            * mu
                            * c_val
                            * dc_di;
                }
            }

            for c_ineq in inequality_constraints {
                let c_val = c_ineq(x_val);
                if c_val > T::zero() {
                    for i in 0..n {
                        let mut x_plus = x_val.to_vec();
                        x_plus[i] = x_plus[i] + eps;
                        let c_plus = c_ineq(&x_plus);
                        let dc_di = (c_plus - c_val) / eps;
                        g_pen[i] = g_pen[i]
                            + T::from(2.0).expect("2.0 should be representable in Float")
                                * mu
                                * c_val
                                * dc_di;
                    }
                }
            }

            g_pen
        };

        // Solve unconstrained problem with current penalty
        cfg.max_iter = 100; // Limit iterations per penalty phase
        let result = bfgs(f_penalized, grad_penalized, &x, Some(cfg.clone()))?;

        x = result.x.clone();
        total_nfev += result.nfev;
        total_njev += result.njev;

        // Check constraint satisfaction
        let mut max_eq_violation = T::zero();
        for c_eq in equality_constraints {
            let c_val = c_eq(&x);
            max_eq_violation = max_eq_violation.max(c_val.abs());
        }

        let mut max_ineq_violation = T::zero();
        for c_ineq in inequality_constraints {
            let c_val = c_ineq(&x);
            max_ineq_violation = max_ineq_violation.max(c_val.max(T::zero()));
        }

        let constraint_tol = T::from(1e-6).expect("1e-6 should be representable in Float");
        if max_eq_violation < constraint_tol && max_ineq_violation < constraint_tol {
            return Ok(OptimizeResult {
                x: x.clone(),
                fun: f(&x),
                grad: grad(&x),
                nit: outer_iter + 1,
                nfev: total_nfev,
                njev: total_njev,
                success: true,
                message: "Optimization terminated successfully (constraints satisfied)".to_string(),
            });
        }

        // Increase penalty
        mu = mu * penalty_increase;
    }

    let final_f = f(&x);
    let final_g = grad(&x);

    Ok(OptimizeResult {
        x,
        fun: final_f,
        grad: final_g,
        nit: max_outer_iter,
        nfev: total_nfev,
        njev: total_njev,
        success: false,
        message: "Maximum penalty iterations reached".to_string(),
    })
}
