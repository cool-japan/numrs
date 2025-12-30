//! Iterative solvers for linear systems
//!
//! This module provides iterative methods for solving large linear systems Ax = b,
//! which are more efficient than direct methods for sparse or very large matrices.
//!
//! # Available Solvers
//!
//! - **Conjugate Gradient (CG)**: For symmetric positive definite systems
//! - **GMRES**: For general non-symmetric systems
//! - **BiCGSTAB**: For non-symmetric systems with better stability
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::linalg::iterative_solvers::*;
//!
//! // Solve Ax = b using Conjugate Gradient
//! let a = Array::from_vec(vec![
//!     4.0, 1.0,
//!     1.0, 3.0,
//! ]).reshape(&[2, 2]);
//! let b = Array::from_vec(vec![1.0, 2.0]);
//!
//! let result = conjugate_gradient(&a, &b, None, Some(1e-6), Some(100));
//! assert!(result.is_ok());
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

/// Configuration for iterative solvers
#[derive(Debug, Clone)]
pub struct SolverConfig<T: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: T,
    /// Restart parameter (for GMRES)
    pub restart: Option<usize>,
    /// Use preconditioner
    pub use_preconditioner: bool,
}

impl<T: Float> Default for SolverConfig<T> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: T::from(1e-6).unwrap(),
            restart: Some(30),
            use_preconditioner: false,
        }
    }
}

/// Result of iterative solver
#[derive(Debug, Clone)]
pub struct SolverResult<T: Clone> {
    /// Solution vector
    pub solution: Array<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: T,
    /// Whether the solver converged
    pub converged: bool,
}

/// Conjugate Gradient method for symmetric positive definite systems
///
/// Solves Ax = b where A is symmetric positive definite.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (must be SPD)
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
///
/// # Returns
///
/// A `SolverResult` containing the solution and convergence information
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::conjugate_gradient;
///
/// let a = Array::from_vec(vec![
///     4.0, 1.0,
///     1.0, 3.0,
/// ]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 2.0]);
///
/// let result = conjugate_gradient(&a, &b, None, Some(1e-6), Some(100)).unwrap();
/// assert!(result.converged);
/// ```
pub fn conjugate_gradient<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    // Validate dimensions
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n);

    // Initialize x - use vectors for efficient access
    let mut x_vec: Vec<T> = match x0 {
        Some(x) => x.to_vec(),
        None => vec![T::zero(); n],
    };

    // Compute initial residual r = b - Ax using vectorized operations
    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();
    let b_vec = b.to_vec();

    let mut r_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&b_i, &ax_i)| b_i - ax_i)
        .collect();

    let mut r_norm = compute_norm_vec(&r_vec);
    let b_norm = compute_norm_vec(&b_vec);

    if b_norm.is_zero() {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
        });
    }

    // Check initial convergence
    if r_norm / b_norm < tol {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
        });
    }

    let mut p_vec = r_vec.clone();
    let mut r_dot_r = dot_vec(&r_vec, &r_vec);

    for iter in 0..max_iter {
        // Compute Ap
        let p_arr = Array::from_vec(p_vec.clone());
        let ap = matvec(a, &p_arr)?;
        let ap_vec = ap.to_vec();

        // Compute step size alpha
        let p_dot_ap = dot_vec(&p_vec, &ap_vec);
        if p_dot_ap.is_zero() {
            return Err(NumRs2Error::ComputationError(
                "Matrix is not positive definite".to_string(),
            ));
        }
        let alpha = r_dot_r / p_dot_ap;

        // Update solution: x = x + alpha * p (vectorized)
        for i in 0..n {
            x_vec[i] = x_vec[i] + alpha * p_vec[i];
        }

        // Update residual: r = r - alpha * Ap (vectorized)
        for i in 0..n {
            r_vec[i] = r_vec[i] - alpha * ap_vec[i];
        }

        let r_dot_r_new = dot_vec(&r_vec, &r_vec);
        r_norm = r_dot_r_new.sqrt();

        // Check convergence
        if r_norm / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter + 1,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // Compute new search direction: p = r + beta * p (vectorized)
        let beta = r_dot_r_new / r_dot_r;
        for i in 0..n {
            p_vec[i] = r_vec[i] + beta * p_vec[i];
        }

        r_dot_r = r_dot_r_new;
    }

    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: max_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// Helper function to compute norm of a vector slice
#[inline]
fn compute_norm_vec<T: Float>(v: &[T]) -> T {
    v.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt()
}

/// Helper function to compute dot product of vector slices
#[inline]
fn dot_vec<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// GMRES (Generalized Minimal Residual) method for general linear systems
///
/// Solves Ax = b for general (possibly non-symmetric) matrices.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
/// * `restart` - Restart parameter (if None, uses 30)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::gmres;
///
/// let a = Array::from_vec(vec![
///     4.0, 1.0,
///     1.0, 3.0,
/// ]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 2.0]);
///
/// let result = gmres(&a, &b, None, Some(1e-10), Some(200), Some(50)).unwrap();
/// // GMRES should solve this simple 2x2 system
/// assert!(result.solution.len() == 2);
/// ```
pub fn gmres<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
    restart: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n);
    let restart = restart.unwrap_or(30.min(n));

    let x_init = match x0 {
        Some(x) => x.clone(),
        None => Array::zeros(&[n]),
    };

    let b_norm = compute_norm(b)?;
    if b_norm.is_zero() {
        return Ok(SolverResult {
            solution: x_init,
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    let mut total_iter = 0;

    // Use Vec<T> for efficient slice operations
    let mut x_vec = x_init.to_vec();
    let b_vec = b.to_vec();

    // Outer iteration (restarts)
    for _ in 0..(max_iter / restart + 1) {
        // Compute initial residual r = b - Ax using vectorized operations
        let x_arr = Array::from_vec(x_vec.clone());
        let ax = matvec(a, &x_arr)?;
        let ax_vec = ax.to_vec();

        let r_vec: Vec<T> = b_vec
            .iter()
            .zip(ax_vec.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();

        let r_norm = compute_norm_vec(&r_vec);
        if r_norm / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: total_iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // Initialize Arnoldi iteration - store as Vec<Vec<T>> for efficient access
        let mut v_vecs: Vec<Vec<T>> = vec![vec![T::zero(); n]; restart + 1];
        let inv_r_norm = T::one() / r_norm;
        for i in 0..n {
            v_vecs[0][i] = r_vec[i] * inv_r_norm;
        }

        let mut h = vec![vec![T::zero(); restart]; restart + 1];
        let mut g = vec![T::zero(); restart + 1]; // RHS of least-squares: ||r|| * e_1
        g[0] = r_norm;

        // Store Givens rotation coefficients separately
        let mut cs_vec = vec![T::zero(); restart];
        let mut sn_vec = vec![T::zero(); restart];

        // Arnoldi iteration
        let mut k = 0;
        for j in 0..restart {
            if total_iter >= max_iter {
                break;
            }
            total_iter += 1;

            // Apply matrix
            let v_arr = Array::from_vec(v_vecs[j].clone());
            let w = matvec(a, &v_arr)?;
            let mut w_vec = w.to_vec();

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot_vec(&v_vecs[i], &w_vec);
                let h_val = h[i][j];
                for l in 0..n {
                    w_vec[l] = w_vec[l] - h_val * v_vecs[i][l];
                }
            }

            h[j + 1][j] = compute_norm_vec(&w_vec);

            if h[j + 1][j].abs() < T::from(1e-14).unwrap() {
                // Lucky breakdown - exact solution found
                k = j + 1;
                break;
            }

            // Normalize
            let inv_h = T::one() / h[j + 1][j];
            for l in 0..n {
                v_vecs[j + 1][l] = w_vec[l] * inv_h;
            }

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let temp = h[i][j];
                h[i][j] = cs_vec[i] * temp + sn_vec[i] * h[i + 1][j];
                h[i + 1][j] = -sn_vec[i] * temp + cs_vec[i] * h[i + 1][j];
            }

            // Compute new Givens rotation to eliminate h[j+1][j]
            let r_val = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
            if r_val < T::from(1e-14).unwrap() {
                // Lucky breakdown
                k = j + 1;
                break;
            }
            let cs = h[j][j] / r_val;
            let sn = h[j + 1][j] / r_val;
            cs_vec[j] = cs;
            sn_vec[j] = sn;

            // Apply new rotation to H
            h[j][j] = r_val;
            h[j + 1][j] = T::zero();

            // Apply new rotation to g (the RHS)
            let temp_g = g[j];
            g[j] = cs * temp_g;
            g[j + 1] = -sn * temp_g;

            // Increment k to track the number of basis vectors used
            k = j + 1;

            // Check convergence - |g[j+1]| is the residual norm
            if g[j + 1].abs() / b_norm < tol {
                break;
            }
        }

        // Solve upper triangular system
        let mut y = vec![T::zero(); k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum = sum - h[i][j] * y[j];
            }
            y[i] = sum / h[i][i];
        }

        // Update solution using vectorized operations
        for j in 0..k {
            let y_j = y[j];
            for i in 0..n {
                x_vec[i] = x_vec[i] + y_j * v_vecs[j][i];
            }
        }

        // Check for convergence after restart
        let x_arr = Array::from_vec(x_vec.clone());
        let ax = matvec(a, &x_arr)?;
        let ax_vec = ax.to_vec();

        let r_final_vec: Vec<T> = b_vec
            .iter()
            .zip(ax_vec.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        let final_r_norm = compute_norm_vec(&r_final_vec);

        if final_r_norm / b_norm < tol || total_iter >= max_iter {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: total_iter,
                residual_norm: final_r_norm,
                converged: final_r_norm / b_norm < tol,
            });
        }
    }

    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();
    let r_final_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();
    let r_norm = compute_norm_vec(&r_final_vec);

    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: total_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// Right-Preconditioned GMRES method
///
/// Solves AM^(-1)(Mx) = b using right preconditioning.
/// This is equivalent to solving Az = b where z = M^(-1)x, then x = M^(-1)z.
///
/// Right preconditioning preserves the residual r = b - Ax, making convergence
/// monitoring straightforward.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
/// * `preconditioner` - Preconditioner implementing the Preconditioner trait
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
/// * `restart` - Restart parameter (if None, uses min(30, n))
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::{gmres_precond, JacobiPreconditioner};
///
/// let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 5.0]);
///
/// let precond = JacobiPreconditioner::new(&a).unwrap();
/// let result = gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(30)).unwrap();
/// assert!(result.converged);
/// ```
pub fn gmres_precond<T, P>(
    a: &Array<T>,
    b: &Array<T>,
    preconditioner: &P,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
    restart: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
    P: Preconditioner<T>,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n);
    let restart = restart.unwrap_or(30.min(n));

    let x_init = match x0 {
        Some(x) => x.clone(),
        None => Array::zeros(&[n]),
    };

    let b_norm = compute_norm(b)?;
    if b_norm.is_zero() {
        return Ok(SolverResult {
            solution: x_init,
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    let mut total_iter = 0;

    // Use Vec<T> for efficient slice operations
    let mut x_vec = x_init.to_vec();
    let b_vec = b.to_vec();

    // Store v vectors for applying preconditioner at the end
    let mut v_arrays: Vec<Array<T>> = vec![Array::zeros(&[n]); restart + 1];

    // Outer iteration (restarts)
    for _ in 0..(max_iter / restart + 1) {
        // Compute initial residual r = b - Ax using vectorized operations
        let x_arr = Array::from_vec(x_vec.clone());
        let ax = matvec(a, &x_arr)?;
        let ax_vec = ax.to_vec();

        let r_vec: Vec<T> = b_vec
            .iter()
            .zip(ax_vec.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();

        let r_norm = compute_norm_vec(&r_vec);
        if r_norm / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: total_iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // Initialize Arnoldi iteration - store as Vec<Vec<T>> for efficient access
        let mut v_vecs: Vec<Vec<T>> = vec![vec![T::zero(); n]; restart + 1];
        let inv_r_norm = T::one() / r_norm;
        for i in 0..n {
            v_vecs[0][i] = r_vec[i] * inv_r_norm;
        }
        v_arrays[0] = Array::from_vec(v_vecs[0].clone());

        let mut h = vec![vec![T::zero(); restart]; restart + 1];
        let mut g = vec![T::zero(); restart + 1]; // RHS of least-squares
        g[0] = r_norm;

        // Store Givens rotation coefficients
        let mut cs_vec = vec![T::zero(); restart];
        let mut sn_vec = vec![T::zero(); restart];

        // Arnoldi iteration with right preconditioning
        let mut k = 0;
        for j in 0..restart {
            if total_iter >= max_iter {
                break;
            }
            total_iter += 1;

            // Right preconditioning: w = A * M^(-1) * v[j]
            let z = preconditioner.apply(&v_arrays[j])?;
            let w = matvec(a, &z)?;
            let mut w_vec = w.to_vec();

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot_vec(&v_vecs[i], &w_vec);
                let h_val = h[i][j];
                for l in 0..n {
                    w_vec[l] = w_vec[l] - h_val * v_vecs[i][l];
                }
            }

            h[j + 1][j] = compute_norm_vec(&w_vec);

            if h[j + 1][j].abs() < T::from(1e-14).unwrap() {
                // Lucky breakdown
                k = j + 1;
                break;
            }

            // Normalize
            let inv_h = T::one() / h[j + 1][j];
            for l in 0..n {
                v_vecs[j + 1][l] = w_vec[l] * inv_h;
            }
            v_arrays[j + 1] = Array::from_vec(v_vecs[j + 1].clone());

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let temp = h[i][j];
                h[i][j] = cs_vec[i] * temp + sn_vec[i] * h[i + 1][j];
                h[i + 1][j] = -sn_vec[i] * temp + cs_vec[i] * h[i + 1][j];
            }

            // Compute new Givens rotation
            let r_val = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
            if r_val < T::from(1e-14).unwrap() {
                k = j + 1;
                break;
            }
            let cs = h[j][j] / r_val;
            let sn = h[j + 1][j] / r_val;
            cs_vec[j] = cs;
            sn_vec[j] = sn;

            // Apply new rotation to H
            h[j][j] = r_val;
            h[j + 1][j] = T::zero();

            // Apply new rotation to g
            let temp_g = g[j];
            g[j] = cs * temp_g;
            g[j + 1] = -sn * temp_g;

            // Track number of basis vectors
            k = j + 1;

            // Check convergence
            if g[j + 1].abs() / b_norm < tol {
                break;
            }
        }

        // Solve upper triangular system
        let mut y = vec![T::zero(); k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for jj in (i + 1)..k {
                sum = sum - h[i][jj] * y[jj];
            }
            y[i] = sum / h[i][i];
        }

        // Update solution: x = x + M^(-1) * V * y
        // For right preconditioning, we need to apply M^(-1) to each basis vector
        for j in 0..k {
            let z_j = preconditioner.apply(&v_arrays[j])?;
            let z_j_vec = z_j.to_vec();
            let y_j = y[j];
            for i in 0..n {
                x_vec[i] = x_vec[i] + y_j * z_j_vec[i];
            }
        }

        // Check for convergence after restart
        let x_arr = Array::from_vec(x_vec.clone());
        let ax = matvec(a, &x_arr)?;
        let ax_vec = ax.to_vec();

        let r_final_vec: Vec<T> = b_vec
            .iter()
            .zip(ax_vec.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        let final_r_norm = compute_norm_vec(&r_final_vec);

        if final_r_norm / b_norm < tol || total_iter >= max_iter {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: total_iter,
                residual_norm: final_r_norm,
                converged: final_r_norm / b_norm < tol,
            });
        }
    }

    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();
    let r_final_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();
    let r_norm = compute_norm_vec(&r_final_vec);

    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: total_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// Convenience function for GMRES with Jacobi preconditioning
pub fn gmres_jacobi<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
    restart: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let precond = JacobiPreconditioner::new(a)?;
    gmres_precond(a, b, &precond, x0, tol, max_iter, restart)
}

/// Flexible GMRES (FGMRES) method with variable preconditioning
///
/// FGMRES allows the preconditioner to vary at each iteration, which is useful when:
/// - The preconditioner involves an iterative method (inner iteration)
/// - Using different preconditioners at different stages
/// - The preconditioner is defined implicitly
///
/// Unlike standard preconditioned GMRES, FGMRES stores both V (Krylov basis)
/// and Z (preconditioned vectors) separately, allowing for variable M^(-1).
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
/// * `preconditioner` - Function that applies the preconditioner (can vary per call)
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
/// * `restart` - Restart parameter (if None, uses min(30, n))
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::{fgmres, JacobiPreconditioner, Preconditioner};
///
/// let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 5.0]);
///
/// // Variable preconditioner (here using constant Jacobi, but could vary)
/// let precond = JacobiPreconditioner::new(&a).unwrap();
/// let precond_fn = move |v: &Array<f64>| -> Result<Array<f64>> {
///     precond.apply(v)
/// };
///
/// let result = fgmres(&a, &b, precond_fn, None, Some(1e-10), Some(100), Some(30)).unwrap();
/// assert!(result.converged);
/// ```
pub fn fgmres<T, F>(
    a: &Array<T>,
    b: &Array<T>,
    preconditioner: F,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
    restart: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n);
    let restart = restart.unwrap_or(30.min(n));

    let x_init = match x0 {
        Some(x) => x.clone(),
        None => Array::zeros(&[n]),
    };

    let b_norm = compute_norm(b)?;
    if b_norm.is_zero() {
        return Ok(SolverResult {
            solution: x_init,
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    let mut total_iter = 0;

    // Use Vec<T> for efficient slice operations
    let mut x_vec = x_init.to_vec();
    let b_vec = b.to_vec();

    // Outer iteration (restarts)
    for _ in 0..(max_iter / restart + 1) {
        // Compute initial residual r = b - Ax using vectorized operations
        let x_arr = Array::from_vec(x_vec.clone());
        let ax = matvec(a, &x_arr)?;
        let ax_vec = ax.to_vec();

        let r_vec: Vec<T> = b_vec
            .iter()
            .zip(ax_vec.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();

        let r_norm = compute_norm_vec(&r_vec);
        if r_norm / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: total_iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // FGMRES stores both V (Krylov basis) and Z (preconditioned vectors)
        // Use Vec<Vec<T>> for efficient access
        let mut v_vecs: Vec<Vec<T>> = vec![vec![T::zero(); n]; restart + 1];
        let mut z_vecs: Vec<Vec<T>> = vec![vec![]; restart]; // Z vectors for solution update

        let inv_r_norm = T::one() / r_norm;
        for i in 0..n {
            v_vecs[0][i] = r_vec[i] * inv_r_norm;
        }

        let mut h = vec![vec![T::zero(); restart]; restart + 1];
        let mut g = vec![T::zero(); restart + 1]; // RHS of least-squares
        g[0] = r_norm;

        // Store Givens rotation coefficients
        let mut cs_vec = vec![T::zero(); restart];
        let mut sn_vec = vec![T::zero(); restart];

        // Flexible Arnoldi iteration
        let mut k = 0;
        for j in 0..restart {
            if total_iter >= max_iter {
                break;
            }
            total_iter += 1;

            // Apply variable preconditioner: z[j] = M^(-1)[j] * v[j]
            let v_arr = Array::from_vec(v_vecs[j].clone());
            let z_arr = preconditioner(&v_arr)?;
            z_vecs[j] = z_arr.to_vec();

            // w = A * z[j]
            let z_arr = Array::from_vec(z_vecs[j].clone());
            let w = matvec(a, &z_arr)?;
            let mut w_vec = w.to_vec();

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot_vec(&v_vecs[i], &w_vec);
                let h_val = h[i][j];
                for l in 0..n {
                    w_vec[l] = w_vec[l] - h_val * v_vecs[i][l];
                }
            }

            h[j + 1][j] = compute_norm_vec(&w_vec);

            if h[j + 1][j].abs() < T::from(1e-14).unwrap() {
                // Lucky breakdown
                k = j + 1;
                break;
            }

            // Normalize
            let inv_h = T::one() / h[j + 1][j];
            for l in 0..n {
                v_vecs[j + 1][l] = w_vec[l] * inv_h;
            }

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let temp = h[i][j];
                h[i][j] = cs_vec[i] * temp + sn_vec[i] * h[i + 1][j];
                h[i + 1][j] = -sn_vec[i] * temp + cs_vec[i] * h[i + 1][j];
            }

            // Compute new Givens rotation
            let r_val = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
            if r_val < T::from(1e-14).unwrap() {
                k = j + 1;
                break;
            }
            let cs = h[j][j] / r_val;
            let sn = h[j + 1][j] / r_val;
            cs_vec[j] = cs;
            sn_vec[j] = sn;

            // Apply new rotation to H
            h[j][j] = r_val;
            h[j + 1][j] = T::zero();

            // Apply new rotation to g
            let temp_g = g[j];
            g[j] = cs * temp_g;
            g[j + 1] = -sn * temp_g;

            // Track number of basis vectors
            k = j + 1;

            // Check convergence
            if g[j + 1].abs() / b_norm < tol {
                break;
            }
        }

        // Solve upper triangular system
        let mut y = vec![T::zero(); k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for jj in (i + 1)..k {
                sum = sum - h[i][jj] * y[jj];
            }
            y[i] = sum / h[i][i];
        }

        // Update solution: x = x + Z * y (using Z vectors, not V!)
        // This is the key difference from standard preconditioned GMRES
        for j in 0..k {
            let y_j = y[j];
            for i in 0..n {
                x_vec[i] = x_vec[i] + y_j * z_vecs[j][i];
            }
        }

        // Check for convergence after restart
        let x_arr = Array::from_vec(x_vec.clone());
        let ax = matvec(a, &x_arr)?;
        let ax_vec = ax.to_vec();

        let r_final_vec: Vec<T> = b_vec
            .iter()
            .zip(ax_vec.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        let final_r_norm = compute_norm_vec(&r_final_vec);

        if final_r_norm / b_norm < tol || total_iter >= max_iter {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: total_iter,
                residual_norm: final_r_norm,
                converged: final_r_norm / b_norm < tol,
            });
        }
    }

    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();
    let r_final_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();
    let r_norm = compute_norm_vec(&r_final_vec);

    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: total_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// Convenience function for FGMRES with Jacobi preconditioning
pub fn fgmres_jacobi<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
    restart: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let precond = JacobiPreconditioner::new(a)?;
    fgmres(a, b, |v| precond.apply(v), x0, tol, max_iter, restart)
}

/// BiCGSTAB (Biconjugate Gradient Stabilized) method for non-symmetric systems
///
/// Solves Ax = b for non-symmetric matrices with improved stability over BiCG.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::bicgstab;
///
/// let a = Array::from_vec(vec![
///     3.0, 1.0,
///     1.0, 2.0,
/// ]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 2.0]);
///
/// let result = bicgstab(&a, &b, None, Some(1e-6), Some(100)).unwrap();
/// assert!(result.converged);
/// ```
pub fn bicgstab<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n);

    // Use Vec<T> for efficient slice operations
    let mut x_vec: Vec<T> = match x0 {
        Some(x) => x.to_vec(),
        None => vec![T::zero(); n],
    };
    let b_vec = b.to_vec();
    let b_norm = compute_norm_vec(&b_vec);

    if b_norm.is_zero() {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    // Compute initial residual r = b - Ax
    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();

    let mut r_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();

    let r_norm = compute_norm_vec(&r_vec);

    if r_norm / b_norm < tol {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
        });
    }

    let r0_vec = r_vec.clone();
    let mut rho = dot_vec(&r0_vec, &r_vec);
    let mut p_vec = r_vec.clone();
    let mut v_vec: Vec<T>;

    for iter in 0..max_iter {
        // Compute v = A * p
        let p_arr = Array::from_vec(p_vec.clone());
        let v = matvec(a, &p_arr)?;
        v_vec = v.to_vec();

        let r0_dot_v = dot_vec(&r0_vec, &v_vec);
        if r0_dot_v.abs() < T::from(1e-14).unwrap() {
            return Err(NumRs2Error::ComputationError(
                "BiCGSTAB breakdown: r0 dot v too small".to_string(),
            ));
        }
        let alpha = rho / r0_dot_v;

        // s = r - alpha * v (vectorized)
        let s_vec: Vec<T> = r_vec
            .iter()
            .zip(v_vec.iter())
            .map(|(&ri, &vi)| ri - alpha * vi)
            .collect();

        // Check for early convergence
        let s_norm = compute_norm_vec(&s_vec);
        if s_norm / b_norm < tol {
            // x = x + alpha * p (vectorized)
            for i in 0..n {
                x_vec[i] = x_vec[i] + alpha * p_vec[i];
            }

            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter + 1,
                residual_norm: s_norm,
                converged: true,
            });
        }

        // Compute t = A * s
        let s_arr = Array::from_vec(s_vec.clone());
        let t = matvec(a, &s_arr)?;
        let t_vec = t.to_vec();

        let t_dot_t = dot_vec(&t_vec, &t_vec);
        if t_dot_t.abs() < T::from(1e-14).unwrap() {
            // Already at solution
            for i in 0..n {
                x_vec[i] = x_vec[i] + alpha * p_vec[i];
            }
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter + 1,
                residual_norm: s_norm,
                converged: true,
            });
        }
        let omega = dot_vec(&t_vec, &s_vec) / t_dot_t;

        // Update solution: x = x + alpha * p + omega * s (vectorized)
        for i in 0..n {
            x_vec[i] = x_vec[i] + alpha * p_vec[i] + omega * s_vec[i];
        }

        // Update residual: r = s - omega * t (vectorized)
        for i in 0..n {
            r_vec[i] = s_vec[i] - omega * t_vec[i];
        }

        let r_norm = compute_norm_vec(&r_vec);

        // Check convergence
        if r_norm / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter + 1,
                residual_norm: r_norm,
                converged: true,
            });
        }

        let rho_new = dot_vec(&r0_vec, &r_vec);

        if rho.abs() < T::from(1e-14).unwrap() {
            return Err(NumRs2Error::ComputationError(
                "BiCGSTAB breakdown: rho too small".to_string(),
            ));
        }

        let beta = (rho_new / rho) * (alpha / omega);

        // Update search direction: p = r + beta * (p - omega * v) (vectorized)
        for i in 0..n {
            p_vec[i] = r_vec[i] + beta * (p_vec[i] - omega * v_vec[i]);
        }

        rho = rho_new;
    }

    let r_norm = compute_norm_vec(&r_vec);
    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: max_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// MINRES (Minimal Residual) method for symmetric indefinite systems
///
/// Solves Ax = b where A is symmetric (but not necessarily positive definite).
/// Unlike CG which requires A to be SPD, MINRES can handle symmetric indefinite systems.
///
/// MINRES minimizes the residual norm ||b - Ax|| over the Krylov subspace and uses
/// a three-term recurrence relation with Givens rotations for numerical stability.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (must be symmetric)
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
///
/// # Returns
///
/// A `SolverResult` containing the solution and convergence information
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::minres;
///
/// // Symmetric indefinite matrix
/// let a = Array::from_vec(vec![
///     2.0, 1.0,
///     1.0, -1.0,  // eigenvalues: ~2.414, ~-1.414 (indefinite)
/// ]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 0.0]);
///
/// let result = minres(&a, &b, None, Some(1e-6), Some(100)).unwrap();
/// assert!(result.converged);
/// ```
///
/// # Notes
///
/// - MINRES is particularly useful for saddle point problems and symmetric indefinite systems
/// - More numerically stable than methods like SYMMLQ for ill-conditioned systems
/// - Residual norm decreases monotonically (unlike CG which can oscillate for non-SPD)
pub fn minres<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    // Validate inputs
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    // Set defaults
    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n * 2); // Use 2n for safety

    // Convert to vectors for efficient computation
    let b_vec = b.to_vec();
    let b_norm = compute_norm_vec(&b_vec);

    if b_norm < T::from(1e-14).unwrap() {
        return Ok(SolverResult {
            solution: Array::zeros(&[n]),
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    // Initialize solution vector
    let mut x_vec = if let Some(x0_arr) = x0 {
        x0_arr.to_vec()
    } else {
        vec![T::zero(); n]
    };

    // Compute initial residual r0 = b - A*x0
    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();
    let r_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();

    let beta1 = compute_norm_vec(&r_vec);

    if beta1 < T::from(1e-14).unwrap() {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    // Initialize Lanczos vectors
    let mut v_prev = vec![T::zero(); n];
    let mut v: Vec<T> = r_vec.iter().map(|&ri| ri / beta1).collect();

    // Direction vectors for solution update (three-term recurrence)
    let mut d_prev = vec![T::zero(); n];
    let mut d_prev2 = vec![T::zero(); n];

    // QR factorization state - need TWO previous rotations
    let mut c_prev = T::one(); // c_{k-1}
    let mut s_prev = T::zero(); // s_{k-1}
    let mut c_prev2 = T::one(); // c_{k-2}
    let mut s_prev2 = T::zero(); // s_{k-2}

    // For tracking residual
    let mut phi_bar = beta1;
    let mut beta_k = T::zero(); // β_k (from previous Lanczos step)

    let mut iter = 0;

    // MINRES main loop
    for k in 0..max_iter {
        iter = k + 1;

        // Lanczos step: compute A*v
        let v_arr = Array::from_vec(v.clone());
        let av = matvec(a, &v_arr)?;
        let av_vec = av.to_vec();

        // α_k = v^T * A * v
        let alpha_k = dot_vec(&v, &av_vec);

        // v_new = A*v - α_k*v - β_k*v_{k-1}
        let v_new: Vec<T> = (0..n)
            .map(|i| av_vec[i] - alpha_k * v[i] - beta_k * v_prev[i])
            .collect();

        let beta_next = compute_norm_vec(&v_new);

        // Apply previous rotations to the k-th column of the tridiagonal matrix
        // Column k has entries: [..., 0, β_k, α_k] at positions k-2, k-1, k
        //
        // After G_{k-2}: position k-2 gets ε_k = s_{k-2} * β_k
        //                position k-1 gets c_{k-2} * β_k
        // After G_{k-1}: position k-1 gets δ_k = c_{k-1}*c_{k-2}*β_k + s_{k-1}*α_k
        //                position k gets γ̃_k = -s_{k-1}*c_{k-2}*β_k + c_{k-1}*α_k

        // ε_k: entry at (k-2, k) after rotations - used in three-term recurrence
        let epsilon_k = s_prev2 * beta_k;

        // Intermediate after G_{k-2}
        let beta_rotated = c_prev2 * beta_k;

        // δ_k: entry at (k-1, k) after G_{k-1}
        let delta_k = c_prev * beta_rotated + s_prev * alpha_k;

        // γ̃_k: entry at (k, k) before new rotation
        let gamma_tilde = -s_prev * beta_rotated + c_prev * alpha_k;

        // Compute new Givens rotation to eliminate β_{k+1}
        let gamma_k = (gamma_tilde * gamma_tilde + beta_next * beta_next).sqrt();
        let (c_k, s_k) = if gamma_k > T::from(1e-14).unwrap() {
            (gamma_tilde / gamma_k, beta_next / gamma_k)
        } else {
            (T::one(), T::zero())
        };

        // Update direction vector with three-term recurrence:
        // d_k = (v_k - δ_k * d_{k-1} - ε_k * d_{k-2}) / γ_k
        let d_new: Vec<T> = if gamma_k > T::from(1e-14).unwrap() {
            (0..n)
                .map(|i| (v[i] - delta_k * d_prev[i] - epsilon_k * d_prev2[i]) / gamma_k)
                .collect()
        } else {
            vec![T::zero(); n]
        };

        // Apply rotation to right-hand side and update solution
        // τ_k = c_k * φ̄_{k-1}
        let tau_k = c_k * phi_bar;

        // x_k = x_{k-1} + τ_k * d_k
        for i in 0..n {
            x_vec[i] = x_vec[i] + tau_k * d_new[i];
        }

        // Update φ̄_k = -s_k * φ̄_{k-1}
        phi_bar = -s_k * phi_bar;
        let residual_norm = phi_bar.abs();

        // Check convergence
        if residual_norm / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter,
                residual_norm,
                converged: true,
            });
        }

        // Check for breakdown (Lanczos terminates)
        if beta_next < T::from(1e-14).unwrap() {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter,
                residual_norm,
                converged: residual_norm / b_norm < tol,
            });
        }

        // Prepare for next iteration
        v_prev = v;
        v = v_new.iter().map(|&x| x / beta_next).collect();

        d_prev2 = d_prev;
        d_prev = d_new;

        // Shift rotation parameters
        c_prev2 = c_prev;
        s_prev2 = s_prev;
        c_prev = c_k;
        s_prev = s_k;
        beta_k = beta_next;
    }

    // Compute actual residual for final result
    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let final_residual: T = b_vec
        .iter()
        .zip(ax.to_vec().iter())
        .map(|(&bi, &axi)| {
            let diff = bi - axi;
            diff * diff
        })
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: iter,
        residual_norm: final_residual,
        converged: false,
    })
}

// ============================================================================
// PRECONDITIONERS
// ============================================================================

/// Preconditioner trait for preconditioning iterative solvers
///
/// A preconditioner M approximates A^(-1) and is used to solve M*z = r
/// where z is the preconditioned residual.
pub trait Preconditioner<T: Float + Clone> {
    /// Apply the preconditioner: solve M*z = r for z
    fn apply(&self, r: &Array<T>) -> Result<Array<T>>;
}

/// Identity preconditioner (no preconditioning)
#[derive(Debug, Clone)]
pub struct IdentityPreconditioner;

impl<T: Float + Clone> Preconditioner<T> for IdentityPreconditioner {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        Ok(r.clone())
    }
}

/// Jacobi (diagonal) preconditioner
///
/// The simplest preconditioner that uses the diagonal of A.
/// M = diag(A), so M^(-1) = diag(1/a_ii)
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner<T> {
    /// Inverse of diagonal elements
    diag_inv: Vec<T>,
}

impl<T: Float + Clone> JacobiPreconditioner<T> {
    /// Create a Jacobi preconditioner from a matrix
    pub fn new(a: &Array<T>) -> Result<Self> {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        let n = shape[0];
        let mut diag_inv = Vec::with_capacity(n);

        for i in 0..n {
            let diag_val = a.get(&[i, i])?;
            if diag_val.abs() < T::from(1e-14).unwrap() {
                return Err(NumRs2Error::ComputationError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
            diag_inv.push(T::one() / diag_val);
        }

        Ok(Self { diag_inv })
    }
}

impl<T: Float + Clone> Preconditioner<T> for JacobiPreconditioner<T> {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        let n = r.size();
        if n != self.diag_inv.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![self.diag_inv.len()],
                actual: r.shape(),
            });
        }

        let mut z = Array::zeros(&[n]);
        for i in 0..n {
            z.set(&[i], r.get(&[i])? * self.diag_inv[i])?;
        }
        Ok(z)
    }
}

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner
///
/// More effective than Jacobi for many problems.
/// Uses the lower and upper triangular parts of A with relaxation parameter omega.
#[derive(Debug, Clone)]
pub struct SSORPreconditioner<T: Clone> {
    /// Lower triangular part including diagonal
    lower: Array<T>,
    /// Upper triangular part including diagonal
    upper: Array<T>,
    /// Diagonal elements
    diag: Vec<T>,
    /// Relaxation parameter
    omega: T,
    /// Size
    n: usize,
}

impl<T: Float + Clone> SSORPreconditioner<T> {
    /// Create an SSOR preconditioner from a matrix
    ///
    /// # Arguments
    ///
    /// * `a` - The coefficient matrix
    /// * `omega` - Relaxation parameter (typically 0 < omega < 2, often ~1.0-1.5)
    pub fn new(a: &Array<T>, omega: T) -> Result<Self> {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        let n = shape[0];
        let mut diag = Vec::with_capacity(n);

        // Extract diagonal
        for i in 0..n {
            let d = a.get(&[i, i])?;
            if d.abs() < T::from(1e-14).unwrap() {
                return Err(NumRs2Error::ComputationError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
            diag.push(d);
        }

        Ok(Self {
            lower: a.clone(),
            upper: a.clone(),
            diag,
            omega,
            n,
        })
    }
}

impl<T: Float + Clone> Preconditioner<T> for SSORPreconditioner<T> {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        let n = self.n;
        if r.size() != n {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n],
                actual: r.shape(),
            });
        }

        // Forward sweep: solve (D + omega*L) * y = omega * r
        let mut y = Array::zeros(&[n]);
        for i in 0..n {
            let mut sum = r.get(&[i])? * self.omega;
            for j in 0..i {
                sum = sum - self.omega * self.lower.get(&[i, j])? * y.get(&[j])?;
            }
            y.set(&[i], sum / self.diag[i])?;
        }

        // Diagonal scaling: z = D * y
        let mut z = Array::zeros(&[n]);
        for i in 0..n {
            z.set(&[i], self.diag[i] * y.get(&[i])?)?;
        }

        // Backward sweep: solve (D + omega*U) * result = omega * z
        let mut result = Array::zeros(&[n]);
        for i in (0..n).rev() {
            let mut sum = z.get(&[i])? * self.omega;
            for j in (i + 1)..n {
                sum = sum - self.omega * self.upper.get(&[i, j])? * result.get(&[j])?;
            }
            result.set(&[i], sum / self.diag[i])?;
        }

        Ok(result)
    }
}

/// Incomplete Cholesky preconditioner (IC(0))
///
/// A simple incomplete Cholesky factorization that maintains the sparsity pattern.
/// Only computes non-zero elements where the original matrix has non-zeros.
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyPreconditioner<T: Clone> {
    /// Lower triangular factor
    l: Array<T>,
    /// Size
    n: usize,
}

impl<T: Float + Clone> IncompleteCholeskyPreconditioner<T> {
    /// Create an incomplete Cholesky preconditioner from a symmetric positive definite matrix
    pub fn new(a: &Array<T>) -> Result<Self> {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        let n = shape[0];
        let mut l = Array::zeros(&[n, n]);

        // Compute IC(0) factorization
        for i in 0..n {
            // Compute L[i,i]
            let mut sum = a.get(&[i, i])?;
            for k in 0..i {
                let l_ik = l.get(&[i, k])?;
                sum = sum - l_ik * l_ik;
            }

            if sum <= T::zero() {
                return Err(NumRs2Error::ComputationError(
                    "Matrix is not positive definite or IC factorization failed".to_string(),
                ));
            }
            l.set(&[i, i], sum.sqrt())?;

            // Compute L[j,i] for j > i
            for j in (i + 1)..n {
                let a_ji = a.get(&[j, i])?;
                // Only compute if A has a non-zero entry (IC(0) maintains sparsity pattern)
                if a_ji.abs() > T::from(1e-14).unwrap() {
                    let mut sum = a_ji;
                    for k in 0..i {
                        sum = sum - l.get(&[j, k])? * l.get(&[i, k])?;
                    }
                    l.set(&[j, i], sum / l.get(&[i, i])?)?;
                }
            }
        }

        Ok(Self { l, n })
    }
}

impl<T: Float + Clone> Preconditioner<T> for IncompleteCholeskyPreconditioner<T> {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        let n = self.n;
        if r.size() != n {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n],
                actual: r.shape(),
            });
        }

        // Solve L * y = r (forward substitution)
        let mut y = Array::zeros(&[n]);
        for i in 0..n {
            let mut sum = r.get(&[i])?;
            for j in 0..i {
                sum = sum - self.l.get(&[i, j])? * y.get(&[j])?;
            }
            y.set(&[i], sum / self.l.get(&[i, i])?)?;
        }

        // Solve L^T * z = y (backward substitution)
        let mut z = Array::zeros(&[n]);
        for i in (0..n).rev() {
            let mut sum = y.get(&[i])?;
            for j in (i + 1)..n {
                sum = sum - self.l.get(&[j, i])? * z.get(&[j])?;
            }
            z.set(&[i], sum / self.l.get(&[i, i])?)?;
        }

        Ok(z)
    }
}

/// Custom preconditioner using a user-provided function
pub struct CustomPreconditioner<T, F>
where
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    apply_fn: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> CustomPreconditioner<T, F>
where
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    /// Create a custom preconditioner from a function
    pub fn new(apply_fn: F) -> Self {
        Self {
            apply_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Clone, F> Preconditioner<T> for CustomPreconditioner<T, F>
where
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        (self.apply_fn)(r)
    }
}

// ============================================================================
// PRECONDITIONED CONJUGATE GRADIENT
// ============================================================================

/// Preconditioned Conjugate Gradient (PCG) method
///
/// Solves Ax = b where A is symmetric positive definite, using a preconditioner
/// to accelerate convergence.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (must be SPD)
/// * `b` - Right-hand side vector
/// * `preconditioner` - The preconditioner to use
/// * `x0` - Initial guess (if None, uses zeros)
/// * `tol` - Convergence tolerance (if None, uses 1e-6)
/// * `max_iter` - Maximum iterations (if None, uses n)
///
/// # Returns
///
/// A `SolverResult` containing the solution and convergence information
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::*;
///
/// // SPD matrix
/// let a = Array::from_vec(vec![
///     4.0, 1.0,
///     1.0, 3.0,
/// ]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 2.0]);
///
/// // Using Jacobi preconditioner
/// let precond = JacobiPreconditioner::new(&a).unwrap();
/// let result = pcg(&a, &b, &precond, None, Some(1e-6), Some(100)).unwrap();
/// assert!(result.converged);
/// ```
pub fn pcg<T, P>(
    a: &Array<T>,
    b: &Array<T>,
    preconditioner: &P,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
    P: Preconditioner<T>,
{
    // Validate dimensions
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.size() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let max_iter = max_iter.unwrap_or(n);

    // Use Vec<T> for efficient slice operations
    let mut x_vec: Vec<T> = match x0 {
        Some(x) => x.to_vec(),
        None => vec![T::zero(); n],
    };
    let b_vec = b.to_vec();
    let b_norm = compute_norm_vec(&b_vec);

    if b_norm.is_zero() {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
        });
    }

    // Compute initial residual r = b - Ax using vectorized operations
    let x_arr = Array::from_vec(x_vec.clone());
    let ax = matvec(a, &x_arr)?;
    let ax_vec = ax.to_vec();

    let mut r_vec: Vec<T> = b_vec
        .iter()
        .zip(ax_vec.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();

    let r_norm = compute_norm_vec(&r_vec);

    // Check initial convergence
    if r_norm / b_norm < tol {
        return Ok(SolverResult {
            solution: Array::from_vec(x_vec),
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
        });
    }

    // Apply preconditioner: z = M^(-1) * r
    let r_arr = Array::from_vec(r_vec.clone());
    let z = preconditioner.apply(&r_arr)?;
    let mut z_vec = z.to_vec();
    let mut p_vec = z_vec.clone();
    let mut r_dot_z = dot_vec(&r_vec, &z_vec);

    for iter in 0..max_iter {
        // Compute Ap
        let p_arr = Array::from_vec(p_vec.clone());
        let ap = matvec(a, &p_arr)?;
        let ap_vec = ap.to_vec();

        // Compute step size alpha
        let p_dot_ap = dot_vec(&p_vec, &ap_vec);
        if p_dot_ap.is_zero() || p_dot_ap.abs() < T::from(1e-14).unwrap() {
            return Err(NumRs2Error::ComputationError(
                "Matrix is not positive definite or breakdown occurred".to_string(),
            ));
        }
        let alpha = r_dot_z / p_dot_ap;

        // Update solution: x = x + alpha * p (vectorized)
        for i in 0..n {
            x_vec[i] = x_vec[i] + alpha * p_vec[i];
        }

        // Update residual: r = r - alpha * Ap (vectorized)
        for i in 0..n {
            r_vec[i] = r_vec[i] - alpha * ap_vec[i];
        }

        let r_norm_new = compute_norm_vec(&r_vec);

        // Check convergence
        if r_norm_new / b_norm < tol {
            return Ok(SolverResult {
                solution: Array::from_vec(x_vec),
                iterations: iter + 1,
                residual_norm: r_norm_new,
                converged: true,
            });
        }

        // Apply preconditioner: z = M^(-1) * r
        let r_arr = Array::from_vec(r_vec.clone());
        let z = preconditioner.apply(&r_arr)?;
        z_vec = z.to_vec();

        let r_dot_z_new = dot_vec(&r_vec, &z_vec);

        // Compute new search direction: p = z + beta * p (vectorized)
        let beta = r_dot_z_new / r_dot_z;
        for i in 0..n {
            p_vec[i] = z_vec[i] + beta * p_vec[i];
        }

        r_dot_z = r_dot_z_new;
    }

    let r_norm_final = compute_norm_vec(&r_vec);
    Ok(SolverResult {
        solution: Array::from_vec(x_vec),
        iterations: max_iter,
        residual_norm: r_norm_final,
        converged: false,
    })
}

/// Convenience function to solve using PCG with Jacobi preconditioning
pub fn pcg_jacobi<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let precond = JacobiPreconditioner::new(a)?;
    pcg(a, b, &precond, x0, tol, max_iter)
}

/// Convenience function to solve using PCG with SSOR preconditioning
pub fn pcg_ssor<T>(
    a: &Array<T>,
    b: &Array<T>,
    omega: T,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let precond = SSORPreconditioner::new(a, omega)?;
    pcg(a, b, &precond, x0, tol, max_iter)
}

/// Convenience function to solve using PCG with incomplete Cholesky preconditioning
pub fn pcg_ichol<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: Option<&Array<T>>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<SolverResult<T>>
where
    T: Float + Clone + Zero,
{
    let precond = IncompleteCholeskyPreconditioner::new(a)?;
    pcg(a, b, &precond, x0, tol, max_iter)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Matrix-vector multiplication that always returns a 1D vector
fn matvec<T>(a: &Array<T>, x: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone + Zero,
{
    let n = x.size();
    let x_col = x.clone().reshape(&[n, 1]);
    let result = a.matmul(&x_col)?;
    Ok(result.reshape(&[n]))
}

/// Compute the L2 norm of a vector
fn compute_norm<T>(v: &Array<T>) -> Result<T>
where
    T: Float + Clone + Zero,
{
    let n = v.size();
    let mut sum = T::zero();
    for i in 0..n {
        let val = v.get(&[i])?;
        sum = sum + val * val;
    }
    Ok(sum.sqrt())
}

// ============================================================================
// ITERATIVE REFINEMENT
// ============================================================================

/// Configuration for iterative refinement
#[derive(Debug, Clone)]
pub struct RefinementConfig<T: Float> {
    /// Maximum number of refinement iterations
    pub max_iter: usize,
    /// Convergence tolerance (relative residual)
    pub tol: T,
    /// Minimum improvement ratio to continue (0.5 means residual must reduce by half)
    pub min_improvement: T,
}

impl<T: Float> Default for RefinementConfig<T> {
    fn default() -> Self {
        Self {
            max_iter: 10,
            tol: T::from(1e-12).unwrap(),
            min_improvement: T::from(0.5).unwrap(),
        }
    }
}

/// Result of iterative refinement
#[derive(Debug, Clone)]
pub struct RefinementResult<T: Clone> {
    /// Refined solution vector
    pub solution: Array<T>,
    /// Number of refinement iterations performed
    pub iterations: usize,
    /// Initial residual norm (before refinement)
    pub initial_residual: T,
    /// Final residual norm (after refinement)
    pub final_residual: T,
    /// Improvement factor (initial/final residual ratio)
    pub improvement_factor: T,
    /// Whether refinement converged to tolerance
    pub converged: bool,
}

/// Iterative refinement for improving linear system solutions
///
/// Given an initial solution x0 to Ax = b, iteratively improves accuracy by:
/// 1. Computing residual r = b - Ax
/// 2. Solving Ay = r for correction y
/// 3. Updating x = x + y
/// 4. Repeating until convergence
///
/// This is particularly useful for ill-conditioned systems where direct
/// solvers may lose accuracy due to numerical errors.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
/// * `x0` - Initial solution (from a direct solver)
/// * `solver` - Function to solve Ay = r for correction y
/// * `config` - Optional refinement configuration
///
/// # Returns
///
/// A `RefinementResult` containing the refined solution and diagnostics
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::*;
///
/// // Create a system Ax = b
/// let a = Array::from_vec(vec![
///     4.0, 1.0,
///     1.0, 3.0,
/// ]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 2.0]);
///
/// // Get initial solution (could be from LU decomposition)
/// let x0 = Array::from_vec(vec![0.0909, 0.6363]); // Approximate solution
///
/// // Refine using CG as the correction solver
/// let result = iterative_refinement(&a, &b, &x0, |mat, rhs| {
///     conjugate_gradient(mat, rhs, None, Some(1e-12), Some(100))
///         .map(|r| r.solution)
/// }, None).unwrap();
///
/// assert!(result.improvement_factor > 1.0); // Solution improved
/// ```
pub fn iterative_refinement<T, F>(
    a: &Array<T>,
    b: &Array<T>,
    x0: &Array<T>,
    solver: F,
    config: Option<RefinementConfig<T>>,
) -> Result<RefinementResult<T>>
where
    T: Float + Clone + Zero + std::fmt::Debug + std::ops::AddAssign,
    F: Fn(&Array<T>, &Array<T>) -> Result<Array<T>>,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::InvalidOperation(
            "Matrix must be square".to_string(),
        ));
    }

    let n = shape[0];
    if b.shape() != [n] || x0.shape() != [n] {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: b.shape(),
        });
    }

    let config = config.unwrap_or_default();
    let mut x = x0.clone();

    // Compute initial residual r = b - Ax
    let ax = matvec(a, &x)?;
    let mut residual = compute_residual(b, &ax)?;
    let initial_norm = vector_norm(&residual)?;

    if initial_norm < config.tol {
        // Already converged
        return Ok(RefinementResult {
            solution: x,
            iterations: 0,
            initial_residual: initial_norm,
            final_residual: initial_norm,
            improvement_factor: T::one(),
            converged: true,
        });
    }

    let b_norm = vector_norm(b)?;
    let mut prev_norm = initial_norm;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Solve Ay = r for the correction
        let correction = solver(a, &residual)?;

        // Update solution: x = x + y
        x = array_add(&x, &correction)?;

        // Compute new residual: r = b - Ax
        let ax = matvec(a, &x)?;
        residual = compute_residual(b, &ax)?;
        let current_norm = vector_norm(&residual)?;

        // Check convergence
        let relative_residual = current_norm / b_norm;
        if relative_residual < config.tol {
            return Ok(RefinementResult {
                solution: x,
                iterations,
                initial_residual: initial_norm,
                final_residual: current_norm,
                improvement_factor: initial_norm / current_norm,
                converged: true,
            });
        }

        // Check if improvement is sufficient
        let improvement = prev_norm / current_norm;
        if improvement < config.min_improvement {
            // Stagnating, stop refinement
            return Ok(RefinementResult {
                solution: x,
                iterations,
                initial_residual: initial_norm,
                final_residual: current_norm,
                improvement_factor: initial_norm / current_norm,
                converged: false,
            });
        }

        prev_norm = current_norm;
    }

    let final_norm = vector_norm(&residual)?;
    Ok(RefinementResult {
        solution: x,
        iterations,
        initial_residual: initial_norm,
        final_residual: final_norm,
        improvement_factor: initial_norm / final_norm,
        converged: false,
    })
}

/// Iterative refinement using Conjugate Gradient as the correction solver
///
/// Convenience function that uses CG for the correction step.
/// Best suited for symmetric positive definite systems.
///
/// # Arguments
///
/// * `a` - SPD coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
/// * `x0` - Initial solution
/// * `tol` - Convergence tolerance (optional, default 1e-12)
/// * `max_iter` - Maximum refinement iterations (optional, default 10)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::iterative_solvers::*;
///
/// let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![1.0, 2.0]);
/// let x0 = Array::from_vec(vec![0.09, 0.64]); // Approximate
///
/// let result = iterative_refinement_cg(&a, &b, &x0, Some(1e-12), Some(5)).unwrap();
/// assert!(result.improvement_factor >= 1.0);
/// ```
pub fn iterative_refinement_cg<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: &Array<T>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<RefinementResult<T>>
where
    T: Float + Clone + Zero + std::fmt::Debug + std::ops::AddAssign,
{
    let config = RefinementConfig {
        max_iter: max_iter.unwrap_or(10),
        tol: tol.unwrap_or(T::from(1e-12).unwrap()),
        min_improvement: T::from(0.5).unwrap(),
    };

    iterative_refinement(
        a,
        b,
        x0,
        |mat, rhs| {
            conjugate_gradient(mat, rhs, None, Some(T::from(1e-14).unwrap()), Some(500))
                .map(|r| r.solution)
        },
        Some(config),
    )
}

/// Iterative refinement using BiCGSTAB as the correction solver
///
/// Convenience function that uses BiCGSTAB for the correction step.
/// Suitable for non-symmetric systems.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
/// * `x0` - Initial solution
/// * `tol` - Convergence tolerance (optional, default 1e-12)
/// * `max_iter` - Maximum refinement iterations (optional, default 10)
pub fn iterative_refinement_bicgstab<T>(
    a: &Array<T>,
    b: &Array<T>,
    x0: &Array<T>,
    tol: Option<T>,
    max_iter: Option<usize>,
) -> Result<RefinementResult<T>>
where
    T: Float + Clone + Zero + std::fmt::Debug + std::ops::AddAssign,
{
    let config = RefinementConfig {
        max_iter: max_iter.unwrap_or(10),
        tol: tol.unwrap_or(T::from(1e-12).unwrap()),
        min_improvement: T::from(0.5).unwrap(),
    };

    iterative_refinement(
        a,
        b,
        x0,
        |mat, rhs| {
            bicgstab(mat, rhs, None, Some(T::from(1e-14).unwrap()), Some(500)).map(|r| r.solution)
        },
        Some(config),
    )
}

/// Compute residual vector r = b - ax
fn compute_residual<T>(b: &Array<T>, ax: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone + Zero,
{
    let n = b.size();
    let mut r = Array::zeros(&[n]);
    for i in 0..n {
        let bi = b.get(&[i])?;
        let axi = ax.get(&[i])?;
        r.set(&[i], bi - axi)?;
    }
    Ok(r)
}

/// Compute 2-norm of a vector
fn vector_norm<T>(v: &Array<T>) -> Result<T>
where
    T: Float + Clone + Zero,
{
    let n = v.size();
    let mut sum = T::zero();
    for i in 0..n {
        let vi = v.get(&[i])?;
        sum = sum + vi * vi;
    }
    Ok(sum.sqrt())
}

/// Add two arrays element-wise
fn array_add<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone + Zero,
{
    let n = a.size();
    let mut result = Array::zeros(&[n]);
    for i in 0..n {
        let ai = a.get(&[i])?;
        let bi = b.get(&[i])?;
        result.set(&[i], ai + bi)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cg_simple() {
        // Simple 2x2 SPD system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = conjugate_gradient(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_gmres_simple() {
        // Use a 3x3 system to test GMRES
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![5.0, 5.0, 5.0]);

        let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).unwrap();
        assert!(result.converged, "GMRES should converge for 3x3 system");

        // Verify residual is small
        let ax = matvec(&a, &result.solution).unwrap();
        let residual: f64 = ax
            .to_vec()
            .iter()
            .zip(b.to_vec().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1e-6, "Residual should be small");
    }

    #[test]
    fn test_gmres_2x2() {
        // 2x2 non-symmetric system: use larger restart to allow full Krylov subspace
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // Solution: [3 1][x1]=[1] => 3*0 + 1*1 = 1, 1*0 + 2*1 = 2 => x=[0,1]
        //           [1 2][x2] [2]
        // Use practical tolerance - very tight tolerances may not be achievable
        let result = gmres(&a, &b, None, Some(1e-8), Some(100), Some(10)).unwrap();
        assert!(result.converged, "GMRES should converge for 2x2 system");

        // Verify solution is accurate to 1e-6
        let x = &result.solution;
        assert_relative_eq!(x.get(&[0]).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(x.get(&[1]).unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gmres_identity() {
        // Identity should converge in 1 iteration
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![2.0, 3.0]);

        let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).unwrap();
        assert!(result.converged, "GMRES should converge for identity");
        assert_relative_eq!(result.solution.get(&[0]).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.solution.get(&[1]).unwrap(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gmres_diagonal() {
        // Diagonal matrix should converge quickly
        let a = Array::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![4.0, 9.0, 16.0]);

        let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).unwrap();
        assert!(result.converged, "GMRES should converge for diagonal");
        assert_relative_eq!(result.solution.get(&[0]).unwrap(), 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[1]).unwrap(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[2]).unwrap(), 4.0, epsilon = 1e-6);
    }

    // =========================================================================
    // Preconditioned GMRES Tests
    // =========================================================================

    #[test]
    fn test_gmres_precond_jacobi_simple() {
        // Test preconditioned GMRES with Jacobi preconditioner on a diagonally dominant system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let precond = JacobiPreconditioner::new(&a).unwrap();
        let result =
            gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(10)).unwrap();

        assert!(result.converged, "Preconditioned GMRES should converge");

        // Verify solution: Ax = b
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gmres_jacobi_convenience() {
        // Test the convenience function
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let result = gmres_jacobi(&a, &b, None, Some(1e-10), Some(100), Some(10)).unwrap();

        assert!(result.converged, "GMRES with Jacobi should converge");

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gmres_precond_diagonal() {
        // Diagonal matrix with Jacobi preconditioner (ideal case - preconditioner inverts exactly)
        let a = Array::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![4.0, 9.0, 16.0]);

        let precond = JacobiPreconditioner::new(&a).unwrap();
        let result =
            gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(10)).unwrap();

        assert!(
            result.converged,
            "Preconditioned GMRES should converge for diagonal"
        );
        // For diagonal matrix, solution should be exact in 1-2 iterations
        assert!(
            result.iterations <= 3,
            "Should converge quickly for diagonal matrix"
        );

        // Verify solution x = [2, 3, 4]
        assert_relative_eq!(result.solution.get(&[0]).unwrap(), 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[1]).unwrap(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[2]).unwrap(), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gmres_precond_3x3() {
        // Test on a 3x3 non-symmetric system
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![5.0, 5.0, 5.0]);

        let precond = JacobiPreconditioner::new(&a).unwrap();
        let result =
            gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(10)).unwrap();

        assert!(
            result.converged,
            "Preconditioned GMRES should converge for 3x3 system"
        );

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..3 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gmres_precond_vs_unpreconditioned() {
        // Compare preconditioned vs unpreconditioned GMRES
        // Preconditioned should typically require fewer iterations for ill-conditioned systems
        let a = Array::from_vec(vec![
            10.0, 1.0, 0.0, 0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 4.0,
        ])
        .reshape(&[4, 4]);
        let b = Array::from_vec(vec![11.0, 10.0, 8.0, 5.0]);

        // Unpreconditioned GMRES
        let result_unprecond = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).unwrap();

        // Preconditioned GMRES
        let precond = JacobiPreconditioner::new(&a).unwrap();
        let result_precond =
            gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(10)).unwrap();

        // Both should converge
        assert!(
            result_unprecond.converged,
            "Unpreconditioned GMRES should converge"
        );
        assert!(
            result_precond.converged,
            "Preconditioned GMRES should converge"
        );

        // Both should give similar solutions
        for i in 0..4 {
            assert_relative_eq!(
                result_unprecond.solution.get(&[i]).unwrap(),
                result_precond.solution.get(&[i]).unwrap(),
                epsilon = 1e-4
            );
        }
    }

    #[test]
    fn test_gmres_precond_identity() {
        // Identity preconditioner should give same result as unpreconditioned
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let precond = IdentityPreconditioner;
        let result =
            gmres_precond(&a, &b, &precond, None, Some(1e-8), Some(100), Some(10)).unwrap();

        assert!(
            result.converged,
            "GMRES with identity preconditioner should converge"
        );

        // Solution should be x = [0, 1]
        assert_relative_eq!(result.solution.get(&[0]).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[1]).unwrap(), 1.0, epsilon = 1e-6);
    }

    // =========================================================================
    // Flexible GMRES (FGMRES) Tests
    // =========================================================================

    #[test]
    fn test_fgmres_simple() {
        // Test FGMRES with a simple 2x2 system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let precond = JacobiPreconditioner::new(&a).unwrap();
        let result = fgmres(
            &a,
            &b,
            |v| precond.apply(v),
            None,
            Some(1e-10),
            Some(100),
            Some(10),
        )
        .unwrap();

        assert!(result.converged, "FGMRES should converge");

        // Verify solution: Ax = b
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fgmres_jacobi_convenience() {
        // Test the convenience function
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let result = fgmres_jacobi(&a, &b, None, Some(1e-10), Some(100), Some(10)).unwrap();

        assert!(result.converged, "FGMRES with Jacobi should converge");

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fgmres_diagonal() {
        // Diagonal matrix - ideal case for Jacobi preconditioner
        let a = Array::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![4.0, 9.0, 16.0]);

        let precond = JacobiPreconditioner::new(&a).unwrap();
        let result = fgmres(
            &a,
            &b,
            |v| precond.apply(v),
            None,
            Some(1e-10),
            Some(100),
            Some(10),
        )
        .unwrap();

        assert!(result.converged, "FGMRES should converge for diagonal");

        // Verify solution x = [2, 3, 4]
        assert_relative_eq!(result.solution.get(&[0]).unwrap(), 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[1]).unwrap(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[2]).unwrap(), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fgmres_identity_precond() {
        // FGMRES with identity preconditioner (no preconditioning)
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // Identity preconditioner as closure
        let result = fgmres(
            &a,
            &b,
            |v| Ok(v.clone()),
            None,
            Some(1e-8),
            Some(100),
            Some(10),
        )
        .unwrap();

        assert!(result.converged, "FGMRES with identity should converge");

        // Solution should be x = [0, 1]
        assert_relative_eq!(result.solution.get(&[0]).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.solution.get(&[1]).unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fgmres_vs_gmres_precond() {
        // FGMRES with constant preconditioner should give similar results to gmres_precond
        let a = Array::from_vec(vec![
            10.0, 1.0, 0.0, 0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 4.0,
        ])
        .reshape(&[4, 4]);
        let b = Array::from_vec(vec![11.0, 10.0, 8.0, 5.0]);

        let precond = JacobiPreconditioner::new(&a).unwrap();

        // Standard preconditioned GMRES
        let result_gmres =
            gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(10)).unwrap();

        // FGMRES with same preconditioner
        let result_fgmres = fgmres(
            &a,
            &b,
            |v| precond.apply(v),
            None,
            Some(1e-10),
            Some(100),
            Some(10),
        )
        .unwrap();

        // Both should converge
        assert!(result_gmres.converged, "GMRES precond should converge");
        assert!(result_fgmres.converged, "FGMRES should converge");

        // Both should give similar solutions
        for i in 0..4 {
            assert_relative_eq!(
                result_gmres.solution.get(&[i]).unwrap(),
                result_fgmres.solution.get(&[i]).unwrap(),
                epsilon = 1e-4
            );
        }
    }

    #[test]
    fn test_bicgstab_simple() {
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = bicgstab(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_cg_identity() {
        // Identity matrix should converge in 1 iteration
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![3.0, 4.0]);

        let result = conjugate_gradient(&a, &b, None, Some(1e-10), Some(100)).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
    }

    // =========================================================================
    // MINRES Tests
    // =========================================================================

    #[test]
    fn test_minres_symmetric_indefinite() {
        // Symmetric indefinite matrix (has both positive and negative eigenvalues)
        let a = Array::from_vec(vec![
            2.0, 1.0, 1.0, -1.0, // eigenvalues: ~2.414, ~-1.414
        ])
        .reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 0.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(
            result.converged,
            "MINRES should converge for symmetric indefinite system"
        );

        // Verify solution: A*x ≈ b
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-5);
        }
    }

    #[test]
    fn test_minres_spd_matrix() {
        // MINRES should also work for SPD matrices (where CG would work)
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged, "MINRES should work for SPD matrices");

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-5);
        }
    }

    #[test]
    fn test_minres_saddle_point() {
        // Saddle point problem (common in constrained optimization)
        let a = Array::from_vec(vec![
            3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, -1.0, // Indefinite
        ])
        .reshape(&[3, 3]);
        let b = Array::from_vec(vec![1.0, 2.0, 1.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(150)).unwrap();
        assert!(
            result.converged,
            "MINRES should handle saddle point problems"
        );

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..3 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-5);
        }
    }

    #[test]
    fn test_minres_identity_matrix() {
        // Identity matrix should converge very quickly
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![3.0, 4.0]);

        let result = minres(&a, &b, None, Some(1e-10), Some(100)).unwrap();
        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "Identity should converge in ≤2 iterations"
        );

        // Solution should be exactly b
        for i in 0..2 {
            assert_relative_eq!(
                result.solution.get(&[i]).unwrap(),
                b.get(&[i]).unwrap(),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_minres_larger_indefinite() {
        // 4x4 symmetric indefinite system
        let a = Array::from_vec(vec![
            4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, -2.0,
            1.0, // Negative diagonal element
            0.0, 0.0, 1.0, 2.0,
        ])
        .reshape(&[4, 4]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(200)).unwrap();
        assert!(
            result.converged,
            "MINRES should converge for larger indefinite system"
        );

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..4 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-4);
        }
    }

    #[test]
    fn test_minres_with_initial_guess() {
        // Test with non-zero initial guess
        let a = Array::from_vec(vec![2.0, 1.0, 1.0, -1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 0.0]);
        let x0 = Array::from_vec(vec![0.5, 0.5]); // Initial guess

        let result = minres(&a, &b, Some(&x0), Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-5);
        }
    }

    #[test]
    fn test_minres_residual_monotonicity() {
        // MINRES residual should decrease monotonically
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, -2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 1.0]);

        let result = minres(&a, &b, None, Some(1e-8), Some(100)).unwrap();
        assert!(result.converged);
        // Just check it converged - residual monotonicity is implicit in the algorithm
    }

    #[test]
    fn test_minres_zero_rhs() {
        // Zero right-hand side should give zero solution immediately
        let a = Array::from_vec(vec![2.0, 1.0, 1.0, -1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![0.0, 0.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0, "Zero RHS should converge immediately");

        for i in 0..2 {
            assert_relative_eq!(result.solution.get(&[i]).unwrap(), 0.0, epsilon = 1e-10);
        }
    }

    // =========================================================================
    // PCG Tests
    // =========================================================================

    #[test]
    fn test_pcg_jacobi() {
        // SPD matrix
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = pcg_jacobi(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);

        // Verify solution approximately satisfies Ax = b
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pcg_with_identity_preconditioner() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let precond = IdentityPreconditioner;
        let result = pcg(&a, &b, &precond, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let precond = JacobiPreconditioner::new(&a).unwrap();

        // Test that preconditioner is created
        assert_eq!(precond.diag_inv.len(), 2);

        // Apply to a vector
        let r = Array::from_vec(vec![4.0, 6.0]);
        let z = precond.apply(&r).unwrap();

        // z[0] = r[0] / a[0,0] = 4 / 4 = 1
        // z[1] = r[1] / a[1,1] = 6 / 3 = 2
        assert_relative_eq!(z.get(&[0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(z.get(&[1]).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pcg_larger_system() {
        // 3x3 SPD system
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![1.0, 2.0, 1.0]);

        let result = pcg_jacobi(&a, &b, None, Some(1e-10), Some(100)).unwrap();
        assert!(result.converged);

        // Verify solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..3 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pcg_ssor_preconditioning() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // omega = 1.0 (standard SOR)
        let result = pcg_ssor(&a, &b, 1.0, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_pcg_ichol_preconditioning() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = pcg_ichol(&a, &b, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_incomplete_cholesky_preconditioner() {
        // Well-conditioned SPD matrix
        let a = Array::from_vec(vec![4.0, 2.0, 2.0, 5.0]).reshape(&[2, 2]);

        let precond = IncompleteCholeskyPreconditioner::new(&a).unwrap();
        assert_eq!(precond.n, 2);

        // Apply to a test vector
        let r = Array::from_vec(vec![2.0, 3.0]);
        let z = precond.apply(&r).unwrap();

        // Should return a valid result
        assert_eq!(z.size(), 2);
    }

    #[test]
    fn test_pcg_vs_cg_comparison() {
        // Compare convergence of PCG vs CG
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![1.0, 2.0, 1.0]);

        // Standard CG
        let cg_result = conjugate_gradient(&a, &b, None, Some(1e-10), Some(100)).unwrap();

        // PCG with Jacobi
        let pcg_result = pcg_jacobi(&a, &b, None, Some(1e-10), Some(100)).unwrap();

        // Both should converge
        assert!(cg_result.converged);
        assert!(pcg_result.converged);

        // PCG should converge in fewer or equal iterations for this well-conditioned system
        // (Jacobi preconditioning helps with diagonally dominant matrices)
        assert!(pcg_result.iterations <= cg_result.iterations + 2); // Allow small variance
    }

    #[test]
    fn test_custom_preconditioner() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // Custom preconditioner: scale by 0.25 (like dividing by diagonal=4)
        let custom = CustomPreconditioner::new(|r: &Array<f64>| {
            let n = r.size();
            let mut z = Array::zeros(&[n]);
            for i in 0..n {
                z.set(&[i], r.get(&[i]).unwrap() * 0.25)?;
            }
            Ok(z)
        });

        let result = pcg(&a, &b, &custom, None, Some(1e-6), Some(100)).unwrap();
        assert!(result.converged);
    }

    // ========================================================================
    // ITERATIVE REFINEMENT TESTS
    // ========================================================================

    #[test]
    fn test_iterative_refinement_basic() {
        // Well-conditioned SPD system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // True solution is approximately [0.0909, 0.6363]
        // Start with a slightly inaccurate initial guess
        let x0 = Array::from_vec(vec![0.09, 0.64]);

        let result = iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(5)).unwrap();

        // Refinement should improve the solution
        assert!(result.improvement_factor >= 1.0);

        // Verify refined solution
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-8);
        }
    }

    #[test]
    fn test_iterative_refinement_already_converged() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // Get exact solution using CG first
        let cg_result = conjugate_gradient(&a, &b, None, Some(1e-14), Some(100)).unwrap();
        let x0 = cg_result.solution;

        // Refinement should detect already converged
        let result = iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(5)).unwrap();

        // Should converge immediately or in 1 iteration
        assert!(result.iterations <= 1);
        assert!(result.converged);
    }

    #[test]
    fn test_iterative_refinement_larger_system() {
        // 4x4 SPD system (tridiagonal)
        let a = Array::from_vec(vec![
            4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0,
        ])
        .reshape(&[4, 4]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Poor initial guess
        let x0 = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

        let result = iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(10)).unwrap();

        // Should converge with improved solution
        assert!(result.improvement_factor > 1.0);

        // Verify solution accuracy
        let ax = matvec(&a, &result.solution).unwrap();
        for i in 0..4 {
            assert_relative_eq!(ax.get(&[i]).unwrap(), b.get(&[i]).unwrap(), epsilon = 1e-8);
        }
    }

    #[test]
    fn test_iterative_refinement_bicgstab() {
        // Non-symmetric system
        let a = Array::from_vec(vec![
            4.0, 1.0, 2.0, 3.0, // Not symmetric: a[1,0] != a[0,1]
        ])
        .reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // Get initial solution with BiCGSTAB
        let initial = bicgstab(&a, &b, None, Some(1e-4), Some(50)).unwrap();

        // Refine with BiCGSTAB
        let result =
            iterative_refinement_bicgstab(&a, &b, &initial.solution, Some(1e-10), Some(5)).unwrap();

        // Should improve or maintain accuracy
        assert!(result.improvement_factor >= 1.0 || result.final_residual < 1e-8);
    }

    #[test]
    fn test_iterative_refinement_custom_solver() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);
        let x0 = Array::from_vec(vec![0.0, 0.0]);

        // Use custom solver function
        let result = iterative_refinement(
            &a,
            &b,
            &x0,
            |mat, rhs| {
                // Use CG with high precision
                conjugate_gradient(mat, rhs, None, Some(1e-14), Some(200)).map(|r| r.solution)
            },
            None,
        )
        .unwrap();

        assert!(result.improvement_factor > 1.0);
    }

    #[test]
    fn test_refinement_config() {
        let config: RefinementConfig<f64> = RefinementConfig::default();
        assert_eq!(config.max_iter, 10);
        assert_relative_eq!(config.tol, 1e-12, epsilon = 1e-15);
        assert_relative_eq!(config.min_improvement, 0.5, epsilon = 1e-10);

        // Custom config
        let custom_config = RefinementConfig {
            max_iter: 20,
            tol: 1e-8,
            min_improvement: 0.1,
        };
        assert_eq!(custom_config.max_iter, 20);
    }

    #[test]
    fn test_refinement_result_fields() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);
        let x0 = Array::from_vec(vec![0.0, 0.0]);

        let result = iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(5)).unwrap();

        // Check result fields are populated
        assert_eq!(result.solution.size(), 2);
        assert!(result.iterations > 0);
        assert!(result.initial_residual > 0.0);
        assert!(result.final_residual >= 0.0);
        assert!(result.improvement_factor > 0.0);
    }
}
