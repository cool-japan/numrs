//! Gradient-based optimization algorithms
//!
//! This module provides gradient-based optimization methods including:
//! - BFGS quasi-Newton method
//! - L-BFGS limited-memory variant
//! - Wolfe line search
//! - Gradient verification utilities

use crate::error::{NumRs2Error, Result};
use num_traits::Float;

use super::{compute_norm, dot_product, OptimizeConfig, OptimizeResult};

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
/// let result = bfgs(f, grad, &[3.0, 4.0], None).expect("BFGS optimization should succeed");
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
        if ys > T::from(1e-14).expect("1e-14 should be representable in Float") {
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
pub fn wolfe_line_search<T, F, G>(
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
        alpha = alpha * T::from(0.5).expect("0.5 should be representable in Float");

        if alpha < T::from(1e-10).expect("1e-10 should be representable in Float") {
            break;
        }
    }

    // If line search fails, return small step
    let alpha_min = T::from(1e-8).expect("1e-8 should be representable in Float");
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
/// let result = lbfgs(f, grad, &[3.0, 4.0], 10, None).expect("L-BFGS optimization should succeed");
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

        if ys > T::from(1e-14).expect("1e-14 should be representable in Float") {
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
    let gamma = if yy > T::from(1e-14).expect("1e-14 should be representable in Float") {
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
    let eps = T::from(1e-8).expect("1e-8 should be representable in Float");
    let g_analytical = grad(x);

    for i in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] = x_plus[i] + eps;
        x_minus[i] = x_minus[i] - eps;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        let g_numerical = (f_plus - f_minus)
            / (T::from(2.0).expect("2.0 should be representable in Float") * eps);

        let relative_error = if g_analytical[i].abs()
            > T::from(1e-10).expect("1e-10 should be representable in Float")
        {
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
