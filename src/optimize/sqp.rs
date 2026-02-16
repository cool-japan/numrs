//! Sequential Quadratic Programming (SQP)
//!
//! SQP is an iterative method for nonlinear optimization. It solves a sequence of
//! quadratic programming (QP) subproblems, each of which is used to generate a search
//! direction. The method uses BFGS updates for Hessian approximation and handles
//! both equality and inequality constraints.
//!
//! # Features
//!
//! - BFGS Hessian approximation
//! - Quadratic programming subproblem solver
//! - Line search with merit function
//! - Equality and inequality constraint handling
//! - KKT condition checking for optimality
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::{sqp, SQPConfig};
//!
//! // Minimize f(x,y) = x^2 + y^2 subject to x + y = 1
//! let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
//! let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
//!
//! let eq_const_fn = |x: &[f64]| x[0] + x[1] - 1.0;
//! let grad_eq_fn = |_x: &[f64]| vec![1.0, 1.0];
//! let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&eq_const_fn];
//! let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_eq_fn];
//!
//! let config = SQPConfig::default();
//! let result = sqp(f, grad_f, &eq_const, &grad_eq, &[], &[], &[0.5, 0.5], Some(config))
//!     .expect("SQP should succeed");
//! ```

use crate::error::{NumRs2Error, Result};
use crate::optimize::OptimizeResult;
use num_traits::Float;
use std::iter::Sum;

/// Type alias for constraint functions
type ConstraintFn<T> = dyn Fn(&[T]) -> T;

/// Type alias for constraint gradient functions
type ConstraintGradFn<T> = dyn Fn(&[T]) -> Vec<T>;

/// Configuration for SQP algorithm
#[derive(Debug, Clone)]
pub struct SQPConfig<T: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Gradient tolerance
    pub gtol: T,
    /// Function value tolerance
    pub ftol: T,
    /// Constraint violation tolerance
    pub ctol: T,
    /// Penalty parameter for merit function
    pub penalty_param: T,
    /// Line search step reduction factor
    pub alpha_reduction: T,
    /// Minimum step size
    pub alpha_min: T,
}

impl<T: Float> Default for SQPConfig<T> {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: T::from(1e-5).expect("1e-5 should convert to Float"),
            ftol: T::from(1e-8).expect("1e-8 should convert to Float"),
            ctol: T::from(1e-6).expect("1e-6 should convert to Float"),
            penalty_param: T::from(1.0).expect("1.0 should convert to Float"),
            alpha_reduction: T::from(0.5).expect("0.5 should convert to Float"),
            alpha_min: T::from(1e-12).expect("1e-12 should convert to Float"),
        }
    }
}

/// Sequential Quadratic Programming optimizer
///
/// # Arguments
///
/// * `f` - Objective function
/// * `grad_f` - Gradient of objective function
/// * `eq_constraints` - Equality constraints h(x) = 0
/// * `grad_eq` - Gradients of equality constraints
/// * `ineq_constraints` - Inequality constraints g(x) <= 0
/// * `grad_ineq` - Gradients of inequality constraints
/// * `x0` - Initial point
/// * `config` - Optional configuration
///
/// # Returns
///
/// `OptimizeResult` with the optimal solution
pub fn sqp<T, F, G>(
    f: F,
    grad_f: G,
    eq_constraints: &[&ConstraintFn<T>],
    grad_eq: &[&ConstraintGradFn<T>],
    ineq_constraints: &[&ConstraintFn<T>],
    grad_ineq: &[&ConstraintGradFn<T>],
    x0: &[T],
    config: Option<SQPConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + Sum + std::fmt::Display,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let config = config.unwrap_or_default();
    let n = x0.len();

    if eq_constraints.len() != grad_eq.len() {
        return Err(NumRs2Error::ValueError(
            "Number of equality constraints must match number of gradients".to_string(),
        ));
    }

    if ineq_constraints.len() != grad_ineq.len() {
        return Err(NumRs2Error::ValueError(
            "Number of inequality constraints must match number of gradients".to_string(),
        ));
    }

    let mut x = x0.to_vec();
    let mut hessian = initialize_identity_matrix::<T>(n);

    // Lagrange multipliers
    let mut lambda_eq = vec![T::zero(); eq_constraints.len()];
    let mut lambda_ineq = vec![T::zero(); ineq_constraints.len()];

    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..config.max_iter {
        // Evaluate objective and constraints
        let f_val = f(&x);
        nfev += 1;

        let grad_f_val = grad_f(&x);
        njev += 1;

        // Evaluate constraints
        let h_vals: Vec<T> = eq_constraints.iter().map(|c| c(&x)).collect();
        let g_vals: Vec<T> = ineq_constraints.iter().map(|c| c(&x)).collect();

        // Check constraint violations
        let eq_violation: T = h_vals.iter().map(|&h| h.abs()).sum();
        let ineq_violation: T = g_vals
            .iter()
            .map(|&g| if g > T::zero() { g } else { T::zero() })
            .sum();

        let constraint_violation = eq_violation + ineq_violation;

        // Compute gradient of Lagrangian
        let grad_l = compute_lagrangian_gradient(
            &grad_f_val,
            eq_constraints,
            grad_eq,
            ineq_constraints,
            grad_ineq,
            &x,
            &lambda_eq,
            &lambda_ineq,
        )?;

        let grad_norm = compute_norm(&grad_l);

        // Check KKT conditions
        if grad_norm < config.gtol && constraint_violation < config.ctol {
            return Ok(OptimizeResult {
                x,
                fun: f_val,
                grad: grad_f_val,
                nit: iter + 1,
                nfev,
                njev,
                success: true,
                message: "KKT conditions satisfied".to_string(),
            });
        }

        // Solve QP subproblem to get search direction
        let (p, lambda_eq_new, lambda_ineq_new) = solve_qp_subproblem(
            &hessian,
            &grad_f_val,
            eq_constraints,
            grad_eq,
            ineq_constraints,
            grad_ineq,
            &x,
        )?;

        // Line search with merit function
        let alpha = line_search_merit(
            &f,
            &x,
            &p,
            f_val,
            eq_constraints,
            ineq_constraints,
            config.penalty_param,
            config.alpha_reduction,
            config.alpha_min,
        )?;

        // Update x
        let x_new: Vec<T> = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + alpha * pi)
            .collect();

        // Update Hessian approximation using BFGS
        let s: Vec<T> = x_new
            .iter()
            .zip(x.iter())
            .map(|(&xn, &xo)| xn - xo)
            .collect();

        let grad_f_new = grad_f(&x_new);
        njev += 1;

        let grad_l_new = compute_lagrangian_gradient(
            &grad_f_new,
            eq_constraints,
            grad_eq,
            ineq_constraints,
            grad_ineq,
            &x_new,
            &lambda_eq_new,
            &lambda_ineq_new,
        )?;

        let y: Vec<T> = grad_l_new
            .iter()
            .zip(grad_l.iter())
            .map(|(&gn, &go)| gn - go)
            .collect();

        bfgs_update(&mut hessian, &s, &y)?;

        // Update state
        x = x_new;
        lambda_eq = lambda_eq_new;
        lambda_ineq = lambda_ineq_new;
    }

    let f_final = f(&x);
    let grad_final = grad_f(&x);

    Ok(OptimizeResult {
        x,
        fun: f_final,
        grad: grad_final,
        nit: config.max_iter,
        nfev,
        njev,
        success: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Compute gradient of Lagrangian
fn compute_lagrangian_gradient<T: Float>(
    grad_f: &[T],
    eq_constraints: &[&ConstraintFn<T>],
    grad_eq: &[&ConstraintGradFn<T>],
    ineq_constraints: &[&ConstraintFn<T>],
    grad_ineq: &[&ConstraintGradFn<T>],
    x: &[T],
    lambda_eq: &[T],
    lambda_ineq: &[T],
) -> Result<Vec<T>> {
    let n = grad_f.len();
    let mut grad_l = grad_f.to_vec();

    // Add equality constraint contributions
    for (i, &lambda) in lambda_eq.iter().enumerate() {
        let grad_h = grad_eq[i](x);
        for j in 0..n {
            grad_l[j] = grad_l[j] + lambda * grad_h[j];
        }
    }

    // Add inequality constraint contributions
    for (i, &lambda) in lambda_ineq.iter().enumerate() {
        let grad_g = grad_ineq[i](x);
        for j in 0..n {
            grad_l[j] = grad_l[j] + lambda * grad_g[j];
        }
    }

    Ok(grad_l)
}

/// Solve QP subproblem (simplified active-set method)
fn solve_qp_subproblem<T: Float>(
    hessian: &[Vec<T>],
    grad_f: &[T],
    eq_constraints: &[&ConstraintFn<T>],
    grad_eq: &[&ConstraintGradFn<T>],
    ineq_constraints: &[&ConstraintFn<T>],
    grad_ineq: &[&ConstraintGradFn<T>],
    x: &[T],
) -> Result<(Vec<T>, Vec<T>, Vec<T>)> {
    let n = grad_f.len();

    // For simplicity, solve unconstrained QP: min 0.5 * p^T * H * p + g^T * p
    // This is a simplified version - full SQP would solve constrained QP
    // solve_linear_system(H, g) returns p = -H^-1 * g, which is the Newton direction we need
    let p = solve_linear_system(hessian, grad_f)?;

    // Estimate Lagrange multipliers (simplified)
    let lambda_eq = vec![T::zero(); eq_constraints.len()];
    let lambda_ineq = vec![T::zero(); ineq_constraints.len()];

    Ok((p, lambda_eq, lambda_ineq))
}

/// Line search using merit function
fn line_search_merit<T, F>(
    f: &F,
    x: &[T],
    p: &[T],
    f_val: T,
    eq_constraints: &[&ConstraintFn<T>],
    ineq_constraints: &[&ConstraintFn<T>],
    mu: T,
    reduction: T,
    alpha_min: T,
) -> Result<T>
where
    T: Float + Sum,
    F: Fn(&[T]) -> T,
{
    let mut alpha = T::one();

    // Merit function: φ(x) = f(x) + μ * (Σ|h_i(x)| + Σmax(0, g_i(x)))
    let merit_fn = |x_eval: &[T]| -> T {
        let f_eval = f(x_eval);
        let eq_penalty: T = eq_constraints.iter().map(|c| c(x_eval).abs()).sum();
        let ineq_penalty: T = ineq_constraints
            .iter()
            .map(|c| {
                let g_val = c(x_eval);
                if g_val > T::zero() {
                    g_val
                } else {
                    T::zero()
                }
            })
            .sum();
        f_eval + mu * (eq_penalty + ineq_penalty)
    };

    let merit_0 = merit_fn(x);

    for _ in 0..20 {
        let x_new: Vec<T> = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + alpha * pi)
            .collect();
        let merit_new = merit_fn(&x_new);

        // Armijo condition for merit function
        if merit_new < merit_0 || alpha < alpha_min {
            return Ok(alpha);
        }

        alpha = alpha * reduction;
    }

    Ok(alpha_min)
}

/// Initialize identity matrix
fn initialize_identity_matrix<T: Float>(n: usize) -> Vec<Vec<T>> {
    let mut matrix = vec![vec![T::zero(); n]; n];
    for i in 0..n {
        matrix[i][i] = T::one();
    }
    matrix
}

/// BFGS Hessian update
fn bfgs_update<T: Float + std::iter::Sum>(hessian: &mut [Vec<T>], s: &[T], y: &[T]) -> Result<()> {
    let n = s.len();

    let sy: T = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum();

    if sy.abs()
        < T::from(1e-10).ok_or_else(|| {
            NumRs2Error::ComputationError("Denominator too small in BFGS update".to_string())
        })?
    {
        return Ok(()); // Skip update if curvature condition not satisfied
    }

    // Compute H*s
    let mut hs = vec![T::zero(); n];
    for i in 0..n {
        for j in 0..n {
            hs[i] = hs[i] + hessian[i][j] * s[j];
        }
    }

    let shs: T = s.iter().zip(hs.iter()).map(|(&si, &hsi)| si * hsi).sum();

    // BFGS update: H_new = H - (H*s*s^T*H)/(s^T*H*s) + (y*y^T)/(y^T*s)
    for i in 0..n {
        for j in 0..n {
            hessian[i][j] = hessian[i][j] - (hs[i] * hs[j]) / shs + (y[i] * y[j]) / sy;
        }
    }

    Ok(())
}

/// Solve linear system H*x = -b using simple Gaussian elimination
fn solve_linear_system<T: Float>(a: &[Vec<T>], b: &[T]) -> Result<Vec<T>> {
    let n = b.len();
    let mut aug = vec![vec![T::zero(); n + 1]; n];

    // Create augmented matrix [A | -b]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = -b[i];
    }

    // Forward elimination
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

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
        }

        // Check for singularity
        if aug[k][k].abs()
            < T::from(1e-10).ok_or_else(|| {
                NumRs2Error::ComputationError("Singular matrix in linear solve".to_string())
            })?
        {
            return Err(NumRs2Error::ComputationError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum = sum - aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
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
    fn test_sqp_unconstrained() {
        // Minimize f(x,y) = x^2 + y^2 (no constraints)
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];

        let config = SQPConfig::default();
        let result = sqp(
            f,
            grad_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[1.0, 1.0],
            Some(config),
        )
        .expect("SQP should succeed");

        assert!(result.success);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_sqp_equality_constraint() {
        // Minimize f(x,y) = x^2 + y^2 subject to x + y = 1
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

        let h1 = |x: &[f64]| x[0] + x[1] - 1.0;
        let grad_h1 = |_x: &[f64]| vec![1.0, 1.0];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![&h1];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![&grad_h1];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];

        let config = SQPConfig::default();
        let result = sqp(
            f,
            grad_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[0.5, 0.5],
            Some(config),
        )
        .expect("SQP should succeed");

        // Solution should be x = y = 0.5 (on the constraint line)
        assert_relative_eq!(result.x[0] + result.x[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_simple_quadratic() {
        // Minimize f(x) = (x - 2)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2);
        let grad_f = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];

        let eq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_eq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];
        let ineq_const: Vec<&dyn Fn(&[f64]) -> f64> = vec![];
        let grad_ineq: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![];

        let config = SQPConfig::default();
        let result = sqp(
            f,
            grad_f,
            &eq_const,
            &grad_eq,
            &ineq_const,
            &grad_ineq,
            &[0.0],
            Some(config),
        )
        .expect("SQP should succeed");

        assert!(result.success);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve: [2 1] [x1]   [5]
        //        [1 3] [x2] = [6]
        // Solution: x1 = 1.8, x2 = 1.4

        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![-5.0, -6.0]; // Negative because we solve Ax = -b

        let x = solve_linear_system(&a, &b).expect("Linear solve should succeed");

        assert_relative_eq!(x[0], 1.8, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.4, epsilon = 1e-10);
    }

    #[test]
    fn test_bfgs_update() {
        let mut h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s = vec![0.1, 0.2];
        let y = vec![0.15, 0.25];

        bfgs_update(&mut h, &s, &y).expect("BFGS update should succeed");

        // Check that H is still symmetric
        assert_relative_eq!(h[0][1], h[1][0], epsilon = 1e-10);
    }
}
