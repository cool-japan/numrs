//! Trust region optimization algorithms
//!
//! This module provides trust region based optimization methods:
//! - Trust region with dogleg method
//! - Levenberg-Marquardt for nonlinear least squares
//! - Supporting utilities (linear solver, numerical Jacobian)

use crate::error::Result;
use num_traits::Float;

use super::{compute_norm, OptimizeConfig, OptimizeResult};

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
/// let result = trust_region(f, grad, hess, &[3.0, 4.0], None).expect("Trust region optimization should succeed");
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
    let mut delta = T::from(1.0).expect("1.0 should be representable in Float"); // Initial trust region radius
    let delta_max = T::from(10.0).expect("10.0 should be representable in Float");
    let eta = T::from(0.15).expect("0.15 should be representable in Float"); // Acceptance threshold

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
        let predicted_reduction =
            -(gp + T::from(0.5).expect("0.5 should be representable in Float") * php);

        // Compute ratio of actual to predicted reduction
        let rho = if predicted_reduction.abs()
            > T::from(1e-14).expect("1e-14 should be representable in Float")
        {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };

        // Update trust region radius
        if rho < T::from(0.25).expect("0.25 should be representable in Float") {
            delta = delta * T::from(0.25).expect("0.25 should be representable in Float");
        } else if rho > T::from(0.75).expect("0.75 should be representable in Float")
            && compute_norm(&p)
                >= delta * T::from(0.99).expect("0.99 should be representable in Float")
        {
            delta = (delta * T::from(2.0).expect("2.0 should be representable in Float"))
                .min(delta_max);
        }

        // Accept or reject step
        if rho > eta {
            x = x_new;
            f_val = f_new;
            g = grad(&x);
            njev += 1;
        }

        // Check for very small trust region
        if delta < T::from(1e-12).expect("1e-12 should be representable in Float") {
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
    let tau_c = if ghg > T::from(1e-14).expect("1e-14 should be representable in Float") {
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
    let b: T = T::from(2.0).expect("2.0 should be representable in Float")
        * p_c
            .iter()
            .zip(p_diff.iter())
            .map(|(&pci, &di)| pci * di)
            .sum::<T>();
    let c = p_c_norm * p_c_norm - delta * delta;

    let discriminant = b * b - T::from(4.0).expect("4.0 should be representable in Float") * a * c;
    let tau = if discriminant >= T::zero()
        && a > T::from(1e-14).expect("1e-14 should be representable in Float")
    {
        (-b + discriminant.sqrt())
            / (T::from(2.0).expect("2.0 should be representable in Float") * a)
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
        if aug[k][k].abs() < T::from(1e-14).expect("1e-14 should be representable in Float") {
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
        x[i] = if aug[i][i].abs() > T::from(1e-14).expect("1e-14 should be representable in Float")
        {
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
/// let result = levenberg_marquardt(residual, &[0.0, 0.0], None).expect("Levenberg-Marquardt optimization should succeed");
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

    let mut lambda = T::from(0.01).expect("0.01 should be representable in Float"); // Damping parameter

    for k in 0..cfg.max_iter {
        // Compute Jacobian numerically
        let jac = numerical_jacobian(&residual, &x);

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
                for l in 0..m {
                    jtj[i][j] = jtj[i][j] + jac[l][i] * jac[l][j];
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
        let predicted_reduction =
            -(gp + T::from(0.5).expect("0.5 should be representable in Float") * php);

        let rho = if predicted_reduction.abs()
            > T::from(1e-14).expect("1e-14 should be representable in Float")
        {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };

        // Update based on gain ratio
        if rho > T::from(0.25).expect("0.25 should be representable in Float") {
            // Good step, accept and possibly decrease damping
            x = x_new;
            r = r_new;
            f_val = f_new;
            lambda = lambda
                * T::from(0.1)
                    .expect("0.1 should be representable in Float")
                    .max(T::from(1e-7).expect("1e-7 should be representable in Float"));
        } else {
            // Poor step, increase damping
            lambda = lambda
                * T::from(10.0)
                    .expect("10.0 should be representable in Float")
                    .min(T::from(1e7).expect("1e7 should be representable in Float"));
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
    let eps = T::from(1e-8).expect("1e-8 should be representable in Float");

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
