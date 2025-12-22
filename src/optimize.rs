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
//! - **Conjugate Gradient**: Nonlinear conjugate gradient method
//!
//! ## Derivative-Free Methods
//! - **Nelder-Mead**: Simplex method for unconstrained optimization
//! - **Powell's Method**: Direction set method without derivatives
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
//! let result = bfgs(f, grad, &x0, None).unwrap();
//! assert!(result.success);
//! // Minimum at (1, 1)
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
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
            gtol: T::from(1e-5).unwrap(),
            ftol: T::from(1e-9).unwrap(),
            xtol: T::from(1e-9).unwrap(),
            ls_max_iter: 20,
            c1: T::from(1e-4).unwrap(),
            c2: T::from(0.9).unwrap(),
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

/// BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton method
///
/// Minimizes a scalar function using gradient information and a quasi-Newton
/// approximation of the Hessian matrix.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `grad` - Gradient function
/// * `x0` - Initial guess
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
///
/// An `OptimizeResult` containing the optimal point and convergence information
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// // Minimize f(x,y) = x^2 + y^2
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = |x: &[f64]| vec![2.0*x[0], 2.0*x[1]];
///
/// let result = bfgs(f, grad, &[3.0, 4.0], None).unwrap();
/// assert!(result.success);
/// assert!(result.fun < 1e-10); // Should find minimum at (0,0)
/// ```
pub fn bfgs<T, F, G>(
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
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    // Initialize
    let mut x = x0.to_vec();
    let mut f_val = f(&x);
    let mut g = grad(&x);
    let mut nfev = 1;
    let mut njev = 1;

    // Initialize inverse Hessian approximation to identity
    let mut h_inv = vec![vec![T::zero(); n]; n];
    for i in 0..n {
        h_inv[i][i] = T::one();
    }

    // Compute initial gradient norm
    let g_norm = compute_norm(&g);

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

    // BFGS iteration
    for k in 0..cfg.max_iter {
        // Compute search direction: p = -H_inv * g
        let mut p = vec![T::zero(); n];
        for i in 0..n {
            for j in 0..n {
                p[i] = p[i] - h_inv[i][j] * g[j];
            }
        }

        // Line search along direction p
        let (alpha, f_new, nfev_ls) = wolfe_line_search(&f, &grad, &x, &p, f_val, &g, &cfg)?;
        nfev += nfev_ls;
        njev += nfev_ls; // Gradient evaluated in each line search step

        // Update x
        let x_new: Vec<T> = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + alpha * pi)
            .collect();

        // Compute new gradient
        let g_new = grad(&x_new);
        njev += 1;

        // Check convergence criteria
        let dx_norm = compute_norm(
            &x_new
                .iter()
                .zip(x.iter())
                .map(|(&xi_new, &xi)| xi_new - xi)
                .collect::<Vec<_>>(),
        );
        let df = (f_new - f_val).abs();
        let g_new_norm = compute_norm(&g_new);

        if g_new_norm < cfg.gtol {
            return Ok(OptimizeResult {
                x: x_new,
                fun: f_new,
                grad: g_new,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (gradient norm converged)"
                    .to_string(),
            });
        }

        if dx_norm < cfg.xtol {
            return Ok(OptimizeResult {
                x: x_new,
                fun: f_new,
                grad: g_new,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (parameter change converged)"
                    .to_string(),
            });
        }

        if df < cfg.ftol {
            return Ok(OptimizeResult {
                x: x_new,
                fun: f_new,
                grad: g_new,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (function value converged)"
                    .to_string(),
            });
        }

        // Compute s_k = x_{k+1} - x_k and y_k = g_{k+1} - g_k
        let s: Vec<T> = x_new
            .iter()
            .zip(x.iter())
            .map(|(&xi_new, &xi)| xi_new - xi)
            .collect();
        let y: Vec<T> = g_new
            .iter()
            .zip(g.iter())
            .map(|(&gi_new, &gi)| gi_new - gi)
            .collect();

        // Compute y^T * s (curvature condition)
        let ys: T = y.iter().zip(s.iter()).map(|(&yi, &si)| yi * si).sum();

        // Update inverse Hessian approximation using BFGS formula
        if ys > T::from(1e-14).unwrap() {
            // Compute H * y
            let mut hy = vec![T::zero(); n];
            for i in 0..n {
                for j in 0..n {
                    hy[i] = hy[i] + h_inv[i][j] * y[j];
                }
            }

            // Compute y^T * H * y
            let yhy: T = y.iter().zip(hy.iter()).map(|(&yi, &hyi)| yi * hyi).sum();

            // BFGS update: H_new = H + (1 + yHy/ys) * (s*s^T)/ys - (s*Hy^T + Hy*s^T)/ys
            for i in 0..n {
                for j in 0..n {
                    let term1 = (T::one() + yhy / ys) * s[i] * s[j] / ys;
                    let term2 = (s[i] * hy[j] + hy[i] * s[j]) / ys;
                    h_inv[i][j] = h_inv[i][j] + term1 - term2;
                }
            }
        }

        // Update for next iteration
        x = x_new;
        f_val = f_new;
        g = g_new;
    }

    // Max iterations reached
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

/// Wolfe line search for step size selection
///
/// Finds a step size alpha that satisfies both the Armijo (sufficient decrease)
/// and curvature (Wolfe) conditions.
fn wolfe_line_search<T, F, G>(
    f: &F,
    grad: &G,
    x: &[T],
    p: &[T],
    f0: T,
    g0: &[T],
    config: &OptimizeConfig<T>,
) -> Result<(T, T, usize)>
where
    T: Float + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let n = x.len();
    let mut alpha = T::one();
    let mut nfev = 0;

    // Compute directional derivative: g0^T * p
    let dg: T = g0.iter().zip(p.iter()).map(|(&gi, &pi)| gi * pi).sum();

    // Armijo condition check
    for _ in 0..config.ls_max_iter {
        // Compute x_new = x + alpha * p
        let x_new: Vec<T> = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + alpha * pi)
            .collect();

        let f_new = f(&x_new);
        nfev += 1;

        // Check Armijo (sufficient decrease) condition
        if f_new <= f0 + config.c1 * alpha * dg {
            // Check strong Wolfe curvature condition
            let g_new = grad(&x_new);
            let dg_new: T = g_new.iter().zip(p.iter()).map(|(&gi, &pi)| gi * pi).sum();

            if dg_new.abs() <= config.c2 * dg.abs() {
                return Ok((alpha, f_new, nfev + 1)); // +1 for gradient eval
            }
        }

        // Reduce step size
        alpha = alpha * T::from(0.5).unwrap();

        if alpha < T::from(1e-10).unwrap() {
            break;
        }
    }

    // If line search fails, return small step
    let alpha_min = T::from(1e-8).unwrap();
    let x_new: Vec<T> = x
        .iter()
        .zip(p.iter())
        .map(|(&xi, &pi)| xi + alpha_min * pi)
        .collect();
    let f_new = f(&x_new);

    Ok((alpha_min, f_new, nfev + 1))
}

/// L-BFGS (Limited-memory BFGS) optimization
///
/// Memory-efficient variant of BFGS that stores only a few recent update vectors
/// instead of the full inverse Hessian approximation.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `grad` - Gradient function
/// * `x0` - Initial guess
/// * `m` - Number of correction pairs to store (typically 5-20)
/// * `config` - Optional configuration
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = |x: &[f64]| vec![2.0*x[0], 2.0*x[1]];
///
/// let result = lbfgs(f, grad, &[3.0, 4.0], 10, None).unwrap();
/// assert!(result.success);
/// ```
pub fn lbfgs<T, F, G>(
    f: F,
    grad: G,
    x0: &[T],
    m: usize, // Number of correction pairs
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    if m == 0 {
        return Err(NumRs2Error::ValueError(
            "L-BFGS memory parameter m must be > 0".to_string(),
        ));
    }

    // Initialize
    let mut x = x0.to_vec();
    let mut f_val = f(&x);
    let mut g = grad(&x);
    let mut nfev = 1;
    let mut njev = 1;

    // Storage for L-BFGS: s and y vectors
    let mut s_history: Vec<Vec<T>> = Vec::with_capacity(m);
    let mut y_history: Vec<Vec<T>> = Vec::with_capacity(m);
    let mut rho_history: Vec<T> = Vec::with_capacity(m);

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

    // L-BFGS iteration
    for k in 0..cfg.max_iter {
        // Compute search direction using L-BFGS two-loop recursion
        let p = lbfgs_two_loop_recursion(&g, &s_history, &y_history, &rho_history);

        // Line search
        let (alpha, f_new, nfev_ls) = wolfe_line_search(&f, &grad, &x, &p, f_val, &g, &cfg)?;
        nfev += nfev_ls;
        njev += nfev_ls;

        // Update parameters
        let x_new: Vec<T> = x
            .iter()
            .zip(p.iter())
            .map(|(&xi, &pi)| xi + alpha * pi)
            .collect();

        // Compute new gradient
        let g_new = grad(&x_new);
        njev += 1;

        // Compute s and y
        let s: Vec<T> = x_new
            .iter()
            .zip(x.iter())
            .map(|(&xi_new, &xi)| xi_new - xi)
            .collect();
        let y: Vec<T> = g_new
            .iter()
            .zip(g.iter())
            .map(|(&gi_new, &gi)| gi_new - gi)
            .collect();

        // Compute rho = 1 / (y^T * s)
        let ys: T = y.iter().zip(s.iter()).map(|(&yi, &si)| yi * si).sum();

        if ys > T::from(1e-14).unwrap() {
            let rho = T::one() / ys;

            // Store in history (maintain max size m)
            if s_history.len() >= m {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
            rho_history.push(rho);
        }

        // Check convergence
        let g_new_norm = compute_norm(&g_new);
        let dx_norm = compute_norm(
            &x_new
                .iter()
                .zip(x.iter())
                .map(|(&xi_new, &xi)| xi_new - xi)
                .collect::<Vec<_>>(),
        );
        let df = (f_new - f_val).abs();

        if g_new_norm < cfg.gtol {
            return Ok(OptimizeResult {
                x: x_new,
                fun: f_new,
                grad: g_new,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (gradient converged)".to_string(),
            });
        }

        if dx_norm < cfg.xtol {
            return Ok(OptimizeResult {
                x: x_new,
                fun: f_new,
                grad: g_new,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (parameter converged)".to_string(),
            });
        }

        if df < cfg.ftol {
            return Ok(OptimizeResult {
                x: x_new,
                fun: f_new,
                grad: g_new,
                nit: k + 1,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (function value converged)"
                    .to_string(),
            });
        }

        // Update for next iteration
        x = x_new;
        f_val = f_new;
        g = g_new;
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
        message: "Maximum iterations reached".to_string(),
    })
}

/// L-BFGS two-loop recursion for computing search direction
///
/// Computes H_k * g using the stored correction pairs without forming H_k explicitly
fn lbfgs_two_loop_recursion<T: Float + std::iter::Sum>(
    g: &[T],
    s_history: &[Vec<T>],
    y_history: &[Vec<T>],
    rho_history: &[T],
) -> Vec<T> {
    let n = g.len();
    let m = s_history.len();

    if m == 0 {
        // No history: use steepest descent
        return g.iter().map(|&gi| -gi).collect();
    }

    let mut q = g.to_vec();
    let mut alpha = vec![T::zero(); m];

    // First loop (backward)
    for i in (0..m).rev() {
        alpha[i] = rho_history[i] * dot_product(&s_history[i], &q);
        for j in 0..n {
            q[j] = q[j] - alpha[i] * y_history[i][j];
        }
    }

    // Initialize H_0 = gamma * I where gamma = s^T*y / y^T*y
    let last_idx = m - 1;
    let sy: T = dot_product(&s_history[last_idx], &y_history[last_idx]);
    let yy: T = dot_product(&y_history[last_idx], &y_history[last_idx]);
    let gamma = if yy > T::from(1e-14).unwrap() {
        sy / yy
    } else {
        T::one()
    };

    // r = gamma * q
    let mut r: Vec<T> = q.iter().map(|&qi| gamma * qi).collect();

    // Second loop (forward)
    for i in 0..m {
        let beta = rho_history[i] * dot_product(&y_history[i], &r);
        for j in 0..n {
            r[j] = r[j] + (alpha[i] - beta) * s_history[i][j];
        }
    }

    // Return -r (search direction)
    r.iter().map(|&ri| -ri).collect()
}

/// Nelder-Mead simplex optimization (derivative-free)
///
/// Minimizes a scalar function without requiring gradient information.
/// Useful for non-smooth or noisy functions.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `x0` - Initial guess
/// * `config` - Optional configuration
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// // Minimize a non-smooth function
/// let f = |x: &[f64]| (x[0] - 2.0).abs() + (x[1] + 3.0).abs();
///
/// let result = nelder_mead(f, &[0.0, 0.0], None).unwrap();
/// assert!(result.success);
/// ```
pub fn nelder_mead<T, F>(
    f: F,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    // Nelder-Mead coefficients
    let alpha = T::one(); // Reflection
    let gamma = T::from(2.0).unwrap(); // Expansion
    let rho = T::from(0.5).unwrap(); // Contraction
    let sigma = T::from(0.5).unwrap(); // Shrink

    // Initialize simplex (n+1 vertices)
    let mut simplex: Vec<Vec<T>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    // Create initial simplex using perturbations
    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] = vertex[i] + T::from(0.05).unwrap();
        simplex.push(vertex);
    }

    // Evaluate function at all vertices
    let mut f_vals: Vec<T> = simplex.iter().map(|x| f(x)).collect();
    let mut nfev = n + 1;

    // Main iteration
    for iter in 0..cfg.max_iter {
        // Sort simplex by function values
        let mut indices: Vec<usize> = (0..n + 1).collect();
        indices.sort_by(|&i, &j| f_vals[i].partial_cmp(&f_vals[j]).unwrap());

        // Best, worst, and second-worst points
        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check convergence: range of function values
        let f_range = f_vals[worst_idx] - f_vals[best_idx];
        if f_range < cfg.ftol {
            return Ok(OptimizeResult {
                x: simplex[best_idx].clone(),
                fun: f_vals[best_idx],
                grad: vec![T::zero(); n], // No gradient in derivative-free method
                nit: iter,
                nfev,
                njev: 0,
                success: true,
                message: "Optimization terminated successfully (simplex converged)".to_string(),
            });
        }

        // Compute centroid of all points except worst
        let mut centroid = vec![T::zero(); n];
        for &idx in indices.iter().take(n) {
            for j in 0..n {
                centroid[j] = centroid[j] + simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] = centroid[j] / T::from(n).unwrap();
        }

        // Reflection: x_r = centroid + alpha * (centroid - x_worst)
        let x_r: Vec<T> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]))
            .collect();
        let f_r = f(&x_r);
        nfev += 1;

        if f_r < f_vals[best_idx] {
            // Expansion: try going further
            let x_e: Vec<T> = (0..n)
                .map(|j| centroid[j] + gamma * (x_r[j] - centroid[j]))
                .collect();
            let f_e = f(&x_e);
            nfev += 1;

            if f_e < f_r {
                simplex[worst_idx] = x_e;
                f_vals[worst_idx] = f_e;
            } else {
                simplex[worst_idx] = x_r;
                f_vals[worst_idx] = f_r;
            }
        } else if f_r < f_vals[second_worst_idx] {
            // Accept reflection
            simplex[worst_idx] = x_r;
            f_vals[worst_idx] = f_r;
        } else {
            // Contraction
            let (x_c, use_reflection) = if f_r < f_vals[worst_idx] {
                // Outside contraction
                let x_c: Vec<T> = (0..n)
                    .map(|j| centroid[j] + rho * (x_r[j] - centroid[j]))
                    .collect();
                (x_c, true)
            } else {
                // Inside contraction
                let x_c: Vec<T> = (0..n)
                    .map(|j| centroid[j] - rho * (simplex[worst_idx][j] - centroid[j]))
                    .collect();
                (x_c, false)
            };

            let f_c = f(&x_c);
            nfev += 1;

            if (use_reflection && f_c < f_r) || (!use_reflection && f_c < f_vals[worst_idx]) {
                simplex[worst_idx] = x_c;
                f_vals[worst_idx] = f_c;
            } else {
                // Shrink simplex toward best point
                for i in 1..=n {
                    let idx = indices[i];
                    for j in 0..n {
                        simplex[idx][j] =
                            simplex[best_idx][j] + sigma * (simplex[idx][j] - simplex[best_idx][j]);
                    }
                    f_vals[idx] = f(&simplex[idx]);
                    nfev += 1;
                }
            }
        }
    }

    // Find best point
    let best_idx = f_vals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    Ok(OptimizeResult {
        x: simplex[best_idx].clone(),
        fun: f_vals[best_idx],
        grad: vec![T::zero(); n],
        nit: cfg.max_iter,
        nfev,
        njev: 0,
        success: false,
        message: "Maximum iterations reached".to_string(),
    })
}

// ============================================================================
// Constrained Optimization
// ============================================================================

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
/// let result = projected_gradient(f, grad, &[1.0, 1.0], &bounds, None).unwrap();
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

    let mut alpha = T::from(0.01).unwrap(); // Initial step size

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
            let g_new_norm = compute_norm(&g_new);

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
            alpha = alpha * T::from(1.05).unwrap();
        } else {
            // Reject step, decrease step size
            alpha = alpha * T::from(0.5).unwrap();

            if alpha < T::from(1e-12).unwrap() {
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
/// ).unwrap();
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
            let eps = T::from(1e-8).unwrap();

            // Numerical gradient of penalty terms (could be analytical if provided)
            for c_eq in equality_constraints {
                let c_val = c_eq(x_val);
                for i in 0..n {
                    let mut x_plus = x_val.to_vec();
                    x_plus[i] = x_plus[i] + eps;
                    let c_plus = c_eq(&x_plus);
                    let dc_di = (c_plus - c_val) / eps;
                    g_pen[i] = g_pen[i] + T::from(2.0).unwrap() * mu * c_val * dc_di;
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
                        g_pen[i] = g_pen[i] + T::from(2.0).unwrap() * mu * c_val * dc_di;
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

        let constraint_tol = T::from(1e-6).unwrap();
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

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute L2 norm of a vector
fn compute_norm<T: Float + std::iter::Sum>(v: &[T]) -> T {
    v.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Compute dot product of two vectors
fn dot_product<T: Float + std::iter::Sum>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

// ============================================================================
// Trust Region Methods
// ============================================================================

/// Trust region optimization using dogleg method
///
/// Minimizes f(x) using a trust region approach with Hessian information.
/// More robust than line search methods when the quadratic model is poor.
///
/// # Arguments
///
/// * `f` - Objective function
/// * `grad` - Gradient function
/// * `hess` - Hessian function (returns matrix as `Vec<Vec<T>>`)
/// * `x0` - Initial guess
/// * `config` - Optional configuration
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = |x: &[f64]| vec![2.0*x[0], 2.0*x[1]];
/// let hess = |_x: &[f64]| vec![vec![2.0, 0.0], vec![0.0, 2.0]];
///
/// let result = trust_region(f, grad, hess, &[3.0, 4.0], None).unwrap();
/// assert!(result.success);
/// ```
pub fn trust_region<T, F, G, H>(
    f: F,
    grad: G,
    hess: H,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
    H: Fn(&[T]) -> Vec<Vec<T>>,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    let mut x = x0.to_vec();
    let mut f_val = f(&x);
    let mut g = grad(&x);
    let mut nfev = 1;
    let mut njev = 1;

    // Trust region parameters
    let mut delta = T::from(1.0).unwrap(); // Initial trust region radius
    let delta_max = T::from(10.0).unwrap();
    let eta = T::from(0.15).unwrap(); // Acceptance threshold

    for k in 0..cfg.max_iter {
        let g_norm = compute_norm(&g);

        // Check convergence
        if g_norm < cfg.gtol {
            return Ok(OptimizeResult {
                x,
                fun: f_val,
                grad: g,
                nit: k,
                nfev,
                njev,
                success: true,
                message: "Optimization terminated successfully (gradient converged)".to_string(),
            });
        }

        // Get Hessian
        let h = hess(&x);

        // Solve trust region subproblem using dogleg method
        let p = dogleg_step(&g, &h, delta);

        // Evaluate at trial point
        let x_new: Vec<T> = x.iter().zip(p.iter()).map(|(&xi, &pi)| xi + pi).collect();
        let f_new = f(&x_new);
        nfev += 1;

        // Compute actual vs predicted reduction
        let actual_reduction = f_val - f_new;

        // Predicted reduction from quadratic model
        let mut hess_p = vec![T::zero(); n];
        for i in 0..n {
            for j in 0..n {
                hess_p[i] = hess_p[i] + h[i][j] * p[j];
            }
        }
        let gp: T = g.iter().zip(p.iter()).map(|(&gi, &pi)| gi * pi).sum();
        let php: T = p
            .iter()
            .zip(hess_p.iter())
            .map(|(&pi, &hpi)| pi * hpi)
            .sum();
        let predicted_reduction = -(gp + T::from(0.5).unwrap() * php);

        // Compute ratio of actual to predicted reduction
        let rho = if predicted_reduction.abs() > T::from(1e-14).unwrap() {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };

        // Update trust region radius
        if rho < T::from(0.25).unwrap() {
            delta = delta * T::from(0.25).unwrap();
        } else if rho > T::from(0.75).unwrap() && compute_norm(&p) >= delta * T::from(0.99).unwrap()
        {
            delta = (delta * T::from(2.0).unwrap()).min(delta_max);
        }

        // Accept or reject step
        if rho > eta {
            x = x_new;
            f_val = f_new;
            g = grad(&x);
            njev += 1;
        }

        // Check for very small trust region
        if delta < T::from(1e-12).unwrap() {
            return Ok(OptimizeResult {
                x,
                fun: f_val,
                grad: g,
                nit: k + 1,
                nfev,
                njev,
                success: false,
                message: "Trust region became too small".to_string(),
            });
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

/// Dogleg step for trust region subproblem
///
/// Computes the step that approximately minimizes the quadratic model
/// m(p) = g^T*p + 0.5*p^T*H*p subject to ||p|| <= delta
fn dogleg_step<T: Float + std::iter::Sum>(g: &[T], h: &[Vec<T>], delta: T) -> Vec<T> {
    let n = g.len();

    // Compute Cauchy point (steepest descent direction)
    let mut hg = vec![T::zero(); n];
    for i in 0..n {
        for j in 0..n {
            hg[i] = hg[i] + h[i][j] * g[j];
        }
    }

    let gg: T = g.iter().map(|&gi| gi * gi).sum();
    let ghg: T = g.iter().zip(hg.iter()).map(|(&gi, &hgi)| gi * hgi).sum();

    // Cauchy step length
    let tau_c = if ghg > T::from(1e-14).unwrap() {
        gg / ghg
    } else {
        T::one()
    };

    let p_c: Vec<T> = g.iter().map(|&gi| -tau_c * gi).collect();
    let p_c_norm = compute_norm(&p_c);

    // If Cauchy point is outside trust region, return scaled version
    if p_c_norm >= delta {
        let scale = delta / p_c_norm;
        return p_c.iter().map(|&pi| pi * scale).collect();
    }

    // Compute Newton step (solve H*p = -g)
    let p_n = solve_linear_system(h, g);
    let p_n_norm = compute_norm(&p_n);

    // If Newton step is inside trust region, use it
    if p_n_norm <= delta {
        return p_n;
    }

    // Otherwise, find dogleg path between Cauchy and Newton
    // Solve ||p_c + tau*(p_n - p_c)|| = delta for tau in [0,1]
    let p_diff: Vec<T> = p_n
        .iter()
        .zip(p_c.iter())
        .map(|(&pni, &pci)| pni - pci)
        .collect();

    let a: T = p_diff.iter().map(|&di| di * di).sum();
    let b: T = T::from(2.0).unwrap()
        * p_c
            .iter()
            .zip(p_diff.iter())
            .map(|(&pci, &di)| pci * di)
            .sum::<T>();
    let c = p_c_norm * p_c_norm - delta * delta;

    let discriminant = b * b - T::from(4.0).unwrap() * a * c;
    let tau = if discriminant >= T::zero() && a > T::from(1e-14).unwrap() {
        (-b + discriminant.sqrt()) / (T::from(2.0).unwrap() * a)
    } else {
        T::one()
    };

    // Return dogleg point
    p_c.iter()
        .zip(p_diff.iter())
        .map(|(&pci, &di)| pci + tau * di)
        .collect()
}

/// Simple linear system solver for small dense systems (Gaussian elimination)
fn solve_linear_system<T: Float>(a: &[Vec<T>], b: &[T]) -> Vec<T> {
    let n = b.len();
    let mut aug = vec![vec![T::zero(); n + 1]; n];

    // Create augmented matrix [A | b]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = -b[i]; // Negative because solving H*p = -g
    }

    // Forward elimination with partial pivoting
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

        // Skip if pivot is too small (singular or ill-conditioned)
        if aug[k][k].abs() < T::from(1e-14).unwrap() {
            continue;
        }

        // Eliminate
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
        x[i] = if aug[i][i].abs() > T::from(1e-14).unwrap() {
            sum / aug[i][i]
        } else {
            T::zero()
        };
    }

    x
}

/// Levenberg-Marquardt algorithm for nonlinear least squares
///
/// Minimizes sum_i r_i(x)^2 where r is a vector-valued residual function.
/// Combines Gauss-Newton and gradient descent using adaptive damping.
///
/// # Arguments
///
/// * `residual` - Residual function r(x)
/// * `x0` - Initial guess
/// * `config` - Optional configuration
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// // Fit linear model y = mx + c
/// let x_data = vec![0.0, 1.0, 2.0, 3.0];
/// let y_data = vec![1.0, 3.0, 5.0, 7.0];
///
/// let residual = |params: &[f64]| -> Vec<f64> {
///     let (m, c) = (params[0], params[1]);
///     x_data.iter().zip(y_data.iter())
///         .map(|(&xi, &yi)| m * xi + c - yi)
///         .collect()
/// };
///
/// let result = levenberg_marquardt(residual, &[0.0, 0.0], None).unwrap();
/// assert!(result.success);
/// ```
pub fn levenberg_marquardt<T, R>(
    residual: R,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    R: Fn(&[T]) -> Vec<T>,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    let mut x = x0.to_vec();
    let mut r = residual(&x);
    let m = r.len(); // Number of residuals
    let mut f_val: T = r.iter().map(|&ri| ri * ri).sum();
    let mut nfev = 1;

    let mut lambda = T::from(0.01).unwrap(); // Damping parameter

    for k in 0..cfg.max_iter {
        // Compute Jacobian numerically
        let jac = numerical_jacobian(&residual, &x);
        let njev = 1;

        // Compute gradient: g = J^T * r
        let mut g = vec![T::zero(); n];
        for i in 0..n {
            for j in 0..m {
                g[i] = g[i] + jac[j][i] * r[j];
            }
        }

        let g_norm = compute_norm(&g);

        // Check convergence
        if g_norm < cfg.gtol {
            return Ok(OptimizeResult {
                x,
                fun: f_val,
                grad: g,
                nit: k,
                nfev,
                njev: k + 1,
                success: true,
                message: "Optimization terminated successfully".to_string(),
            });
        }

        // Compute J^T * J (approximate Hessian)
        let mut jtj = vec![vec![T::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..m {
                    jtj[i][j] = jtj[i][j] + jac[k][i] * jac[k][j];
                }
            }
        }

        // Add damping: (J^T*J + lambda*I) * p = -J^T*r
        for i in 0..n {
            jtj[i][i] = jtj[i][i] + lambda;
        }

        // Solve for step
        let p = solve_linear_system(&jtj, &g);

        // Trial point
        let x_new: Vec<T> = x.iter().zip(p.iter()).map(|(&xi, &pi)| xi + pi).collect();
        let r_new = residual(&x_new);
        let f_new: T = r_new.iter().map(|&ri| ri * ri).sum();
        nfev += 1;

        // Compute gain ratio
        let actual_reduction = f_val - f_new;

        let mut hp = vec![T::zero(); n];
        for i in 0..n {
            for j in 0..n {
                hp[i] = hp[i] + jtj[i][j] * p[j];
            }
        }
        let gp: T = g.iter().zip(p.iter()).map(|(&gi, &pi)| gi * pi).sum();
        let php: T = p.iter().zip(hp.iter()).map(|(&pi, &hpi)| pi * hpi).sum();
        let predicted_reduction = -(gp + T::from(0.5).unwrap() * php);

        let rho = if predicted_reduction.abs() > T::from(1e-14).unwrap() {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };

        // Update based on gain ratio
        if rho > T::from(0.25).unwrap() {
            // Good step, accept and possibly decrease damping
            x = x_new;
            r = r_new;
            f_val = f_new;
            lambda = lambda * T::from(0.1).unwrap().max(T::from(1e-7).unwrap());
        } else {
            // Poor step, increase damping
            lambda = lambda * T::from(10.0).unwrap().min(T::from(1e7).unwrap());
        }
    }

    let g = vec![T::zero(); n]; // Approximate
    Ok(OptimizeResult {
        x,
        fun: f_val,
        grad: g,
        nit: cfg.max_iter,
        nfev,
        njev: cfg.max_iter,
        success: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Compute numerical Jacobian using finite differences
fn numerical_jacobian<T: Float, R>(residual: &R, x: &[T]) -> Vec<Vec<T>>
where
    R: Fn(&[T]) -> Vec<T>,
{
    let n = x.len();
    let r0 = residual(x);
    let m = r0.len();
    let eps = T::from(1e-8).unwrap();

    let mut jac = vec![vec![T::zero(); n]; m];

    for j in 0..n {
        let mut x_plus = x.to_vec();
        x_plus[j] = x_plus[j] + eps;
        let r_plus = residual(&x_plus);

        for i in 0..m {
            jac[i][j] = (r_plus[i] - r0[i]) / eps;
        }
    }

    jac
}

/// Check gradient accuracy using finite differences
///
/// Verifies that the analytical gradient matches numerical approximation.
///
/// # Arguments
///
/// * `f` - Objective function
/// * `grad` - Gradient function to verify
/// * `x` - Point at which to check gradient
/// * `tol` - Tolerance for relative error
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = |x: &[f64]| vec![2.0*x[0], 2.0*x[1]];
///
/// assert!(check_gradient(&f, &grad, &[1.0, 2.0], 1e-6));
/// ```
pub fn check_gradient<T, F, G>(f: &F, grad: &G, x: &[T], tol: T) -> bool
where
    T: Float + std::iter::Sum,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let n = x.len();
    let eps = T::from(1e-8).unwrap();
    let g_analytical = grad(x);

    for i in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] = x_plus[i] + eps;
        x_minus[i] = x_minus[i] - eps;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        let g_numerical = (f_plus - f_minus) / (T::from(2.0).unwrap() * eps);

        let relative_error = if g_analytical[i].abs() > T::from(1e-10).unwrap() {
            ((g_analytical[i] - g_numerical) / g_analytical[i]).abs()
        } else {
            (g_analytical[i] - g_numerical).abs()
        };

        if relative_error > tol {
            return false;
        }
    }

    true
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

        let result = bfgs(f, grad, &[3.0, 4.0], None).unwrap();
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

        let result = bfgs(f, grad, &[0.0, 0.0], None).unwrap();
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

        let result = lbfgs(f, grad, &[3.0, 4.0], 5, None).unwrap();
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
        let result = lbfgs(f, grad, &x0, 5, None).unwrap();

        assert!(result.success);
        for &xi in &result.x {
            assert_relative_eq!(xi, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_nelder_mead_quadratic() {
        // Minimize f(x,y) = (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        let result = nelder_mead(f, &[0.0, 0.0], None).unwrap();
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

        let result = nelder_mead(f, &[0.0, 0.0], Some(cfg)).unwrap();
        // Should get reasonably close to (1, 1)
        assert!(result.fun < 0.1, "Should find good solution");
    }

    #[test]
    fn test_lbfgs_memory_limit() {
        // Test that L-BFGS respects memory limit
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let grad = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect();

        let x0 = vec![1.0; 10];
        let result = lbfgs(f, grad, &x0, 3, None).unwrap(); // Only 3 correction pairs

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

        let result = bfgs(f, grad, &[1.0, 1.0], None).unwrap();
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
        let result = projected_gradient(f, grad, &[1.0, 1.0], &bounds, None).unwrap();

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

        let result = projected_gradient(f, grad, &[3.0], &bounds, None).unwrap();
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

        let result = penalty_method(f, grad, &eq_const, &[], &[0.5, 0.5], 1.0, 10.0, None).unwrap();

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
        let result = penalty_method(f, grad, &[], &ineq_const, &[1.0], 10.0, 10.0, None).unwrap();

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
        .unwrap();

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

            let result = bfgs(f, grad, &[x0, y0], None).unwrap();
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

            let result = lbfgs(f, grad, &x0, 5, None).unwrap();
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

            let result = nelder_mead(f, &[x0, y0], Some(cfg)).unwrap();

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
            let result = wolfe_line_search(&f, &grad, &x, &p, f0, &g0, &cfg);

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

        let result = trust_region(f, grad, hess, &[3.0, 4.0], None).unwrap();
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

        let result = trust_region(f, grad, hess, &[0.0, 0.0], None).unwrap();
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

        let result = levenberg_marquardt(residual, &[0.0, 0.0], None).unwrap();
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

        let result = levenberg_marquardt(residual, &[1.5, 0.8], None).unwrap();
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
