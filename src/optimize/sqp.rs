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

/// Solve the equality-constrained QP subproblem via the bordered KKT system.
///
/// The QP subproblem is
///
/// ```text
///   min_p   0.5 p^T H p + grad_f^T p
///   s.t.    A_active p + c_active = 0
/// ```
///
/// whose first-order (KKT) conditions form the symmetric saddle-point system
///
/// ```text
///   [ H        A_active^T ] [ p      ]   [ -grad_f   ]
///   [ A_active     0      ] [ lambda ] = [ -c_active ]
/// ```
///
/// Here `a_rows` holds the rows of `A_active` (the constraint Jacobian, one row per
/// active constraint, each of length `n`) and `c_vals` holds the constraint residuals
/// `c_active` evaluated at the current iterate. The returned tuple is `(p, lambda)`
/// where `lambda` has one entry per active constraint, in the same order as `a_rows`.
///
/// The bordered matrix is assembled explicitly and solved with the module-local
/// [`solve_linear_system`], which computes `M y = -rhs` for the supplied right-hand
/// side. Passing `rhs = [grad_f; c_active]` therefore yields the desired `[-grad_f; -c_active]`.
fn solve_kkt_system<T: Float>(
    hessian: &[Vec<T>],
    grad_f: &[T],
    a_rows: &[Vec<T>],
    c_vals: &[T],
) -> Result<(Vec<T>, Vec<T>)> {
    let n = grad_f.len();
    let m = a_rows.len();
    let dim = n + m;

    // Assemble the (n + m) x (n + m) bordered KKT matrix.
    let mut kkt = vec![vec![T::zero(); dim]; dim];

    // Top-left block: H.
    for i in 0..n {
        for j in 0..n {
            kkt[i][j] = hessian[i][j];
        }
    }

    // Off-diagonal blocks: A^T (top-right) and A (bottom-left).
    for (row, a_row) in a_rows.iter().enumerate() {
        for j in 0..n {
            // Bottom-left block A.
            kkt[n + row][j] = a_row[j];
            // Top-right block A^T.
            kkt[j][n + row] = a_row[j];
        }
    }
    // Bottom-right block is zero (already initialized).

    // Right-hand side vector b such that solve_linear_system yields M y = -b,
    // i.e. b = [grad_f; c_active] produces RHS [-grad_f; -c_active].
    let mut rhs = vec![T::zero(); dim];
    rhs[..n].copy_from_slice(&grad_f[..n]);
    for (row, &c) in c_vals.iter().enumerate() {
        rhs[n + row] = c;
    }

    let solution = solve_linear_system(&kkt, &rhs)?;

    let p = solution[..n].to_vec();
    let lambda = solution[n..].to_vec();

    Ok((p, lambda))
}

/// Solve the QP subproblem of SQP using a primal active-set strategy.
///
/// The subproblem minimizes the quadratic model `0.5 p^T H p + grad_f^T p` subject
/// to the *linearized* constraints at the current point `x`:
///
/// * equalities  `c_eq_i + grad_eq_i^T p = 0`  for every equality constraint,
/// * inequalities `g_i + grad_g_i^T p <= 0`     for every inequality constraint.
///
/// Equalities are always part of the working (active) set. The inequality working
/// set is determined iteratively:
///
/// 1. Solve the KKT system with the current working set (equalities plus the active
///    inequalities), treating active inequalities as equalities `grad_g_i^T p = -g_i`.
/// 2. Check the inequalities outside the working set; if any linearized constraint is
///    violated (`g_i + grad_g_i^T p > tol`), add the most-violated one to the set.
/// 3. Inspect the multipliers of the active inequalities. With the Lagrangian sign
///    convention `grad_f + Σ lambda_i grad_g_i` (see [`compute_lagrangian_gradient`]),
///    feasibility of the dual requires `lambda_i >= 0` for `g_i <= 0` constraints; any
///    active inequality whose multiplier turns negative is removed from the set.
/// 4. Repeat for a bounded number of iterations until the working set stabilizes.
///
/// Returns `(p, lambda_eq, lambda_ineq)` where `lambda_ineq` carries the recovered
/// multiplier for active inequalities and zero for inactive ones.
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
    let n_eq = eq_constraints.len();
    let n_ineq = ineq_constraints.len();

    // Equality constraint Jacobian rows A_eq and residuals c_eq at x.
    let eq_jac: Vec<Vec<T>> = grad_eq.iter().map(|g| g(x)).collect();
    let eq_vals: Vec<T> = eq_constraints.iter().map(|c| c(x)).collect();

    // Inequality constraint Jacobian rows A_ineq and values g at x.
    let ineq_jac: Vec<Vec<T>> = grad_ineq.iter().map(|g| g(x)).collect();
    let ineq_vals: Vec<T> = ineq_constraints.iter().map(|c| c(x)).collect();

    // Tolerance used both for linearized-constraint violation and multiplier sign.
    let tol = T::from(1e-10).ok_or_else(|| {
        NumRs2Error::ComputationError("Tolerance conversion failed in QP subproblem".to_string())
    })?;

    // Working set of inequality indices (initially empty: equality-only solution).
    let mut active: Vec<usize> = Vec::new();

    // Final results, populated by the loop below (deferred initialization avoids
    // dead-store assignments, in keeping with the no-warnings policy).
    let mut p: Vec<T>;
    let mut lambda_eq: Vec<T>;
    let mut lambda_ineq: Vec<T>;

    // Bound the number of active-set changes; each iteration adds or drops at most
    // one inequality, so 2 * n_ineq + 1 iterations suffice to stabilize plus a margin.
    let max_active_set_iter = 2 * n_ineq + 2;

    for _ in 0..max_active_set_iter {
        // Build the working-set Jacobian: equalities first, then active inequalities.
        let mut a_rows: Vec<Vec<T>> = Vec::with_capacity(n_eq + active.len());
        let mut c_vals: Vec<T> = Vec::with_capacity(n_eq + active.len());

        for row in 0..n_eq {
            a_rows.push(eq_jac[row].clone());
            c_vals.push(eq_vals[row]);
        }
        for &idx in &active {
            a_rows.push(ineq_jac[idx].clone());
            // Linearized active inequality treated as equality: grad_g^T p = -g.
            c_vals.push(ineq_vals[idx]);
        }

        // Solve the KKT system for the current working set.
        let (p_curr, lambda) = solve_kkt_system(hessian, grad_f, &a_rows, &c_vals)?;
        p = p_curr;

        // Split multipliers: first n_eq belong to equalities, the rest to the
        // active inequalities (in `active` order).
        lambda_eq = lambda[..n_eq].to_vec();
        lambda_ineq = vec![T::zero(); n_ineq];
        for (k, &idx) in active.iter().enumerate() {
            lambda_ineq[idx] = lambda[n_eq + k];
        }

        // Dual feasibility check: drop the active inequality with the most negative
        // multiplier (Lagrangian sign convention requires lambda >= 0 for g <= 0).
        let mut drop_pos: Option<usize> = None;
        let mut most_negative = -tol;
        for (k, &idx) in active.iter().enumerate() {
            let mult = lambda_ineq[idx];
            if mult < most_negative {
                most_negative = mult;
                drop_pos = Some(k);
            }
        }
        if let Some(k) = drop_pos {
            active.remove(k);
            continue;
        }

        // Primal feasibility check on inactive inequalities: find the most-violated
        // linearized constraint g_i + grad_g_i^T p > tol and add it to the set.
        let mut add_idx: Option<usize> = None;
        let mut worst_violation = tol;
        for idx in 0..n_ineq {
            if active.contains(&idx) {
                continue;
            }
            // Linearized constraint value g_i + grad_g_i^T p.
            let mut lin = ineq_vals[idx];
            for j in 0..n {
                lin = lin + ineq_jac[idx][j] * p[j];
            }
            if lin > worst_violation {
                worst_violation = lin;
                add_idx = Some(idx);
            }
        }
        if let Some(idx) = add_idx {
            active.push(idx);
            continue;
        }

        // No constraint to add and no multiplier to drop: optimal working set found.
        return Ok((p, lambda_eq, lambda_ineq));
    }

    // Active-set iteration did not stabilize within the bound; re-solve once with the
    // final working set so the returned direction and multipliers are mutually
    // consistent, then return them.
    let mut a_rows: Vec<Vec<T>> = Vec::with_capacity(n_eq + active.len());
    let mut c_vals: Vec<T> = Vec::with_capacity(n_eq + active.len());
    for row in 0..n_eq {
        a_rows.push(eq_jac[row].clone());
        c_vals.push(eq_vals[row]);
    }
    for &idx in &active {
        a_rows.push(ineq_jac[idx].clone());
        c_vals.push(ineq_vals[idx]);
    }
    let (p_final, lambda) = solve_kkt_system(hessian, grad_f, &a_rows, &c_vals)?;
    lambda_eq = lambda[..n_eq].to_vec();
    lambda_ineq = vec![T::zero(); n_ineq];
    for (k, &idx) in active.iter().enumerate() {
        lambda_ineq[idx] = lambda[n_eq + k];
    }

    Ok((p_final, lambda_eq, lambda_ineq))
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
