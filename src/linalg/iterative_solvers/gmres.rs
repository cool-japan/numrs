//! GMRES (Generalized Minimal Residual) methods
//!
//! This module provides GMRES and its variants for solving general (non-symmetric)
//! linear systems.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

use super::core::{compute_norm, compute_norm_vec, dot_vec, matvec, SolverResult};
use super::preconditioners::{JacobiPreconditioner, Preconditioner};

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
/// let result = gmres(&a, &b, None, Some(1e-10), Some(200), Some(50)).expect("GMRES should converge");
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
    T: Float + Clone + Zero + 'static,
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

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap_or(T::epsilon()));
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

            if h[j + 1][j].abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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
            if r_val < T::from(1e-14).unwrap_or(T::epsilon()) {
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
/// let precond = JacobiPreconditioner::new(&a).expect("Jacobi preconditioner creation should succeed");
/// let result = gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(30)).expect("Preconditioned GMRES should converge");
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
    T: Float + Clone + Zero + 'static,
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

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap_or(T::epsilon()));
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

            if h[j + 1][j].abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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
            if r_val < T::from(1e-14).unwrap_or(T::epsilon()) {
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
    T: Float + Clone + Zero + 'static,
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
/// let precond = JacobiPreconditioner::new(&a).expect("Jacobi preconditioner creation should succeed");
/// let precond_fn = move |v: &Array<f64>| -> Result<Array<f64>> {
///     precond.apply(v)
/// };
///
/// let result = fgmres(&a, &b, precond_fn, None, Some(1e-10), Some(100), Some(30)).expect("FGMRES should converge");
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
    T: Float + Clone + Zero + 'static,
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

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap_or(T::epsilon()));
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

            if h[j + 1][j].abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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
            if r_val < T::from(1e-14).unwrap_or(T::epsilon()) {
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
    T: Float + Clone + Zero + 'static,
{
    let precond = JacobiPreconditioner::new(a)?;
    fgmres(a, b, |v| precond.apply(v), x0, tol, max_iter, restart)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::iterative_solvers::preconditioners::IdentityPreconditioner;
    use approx::assert_relative_eq;

    #[test]
    fn test_gmres_simple() {
        // Use a 3x3 system to test GMRES
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![5.0, 5.0, 5.0]);

        let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).expect("Should solve");
        assert!(result.converged, "GMRES should converge for 3x3 system");

        // Verify residual is small
        let ax = matvec(&a, &result.solution).expect("matvec should work");
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
        let result = gmres(&a, &b, None, Some(1e-8), Some(100), Some(10)).expect("Should solve");
        assert!(result.converged, "GMRES should converge for 2x2 system");

        // Verify solution is accurate to 1e-6
        let x = &result.solution;
        assert_relative_eq!(x.get(&[0]).expect("valid"), 0.0, epsilon = 1e-6);
        assert_relative_eq!(x.get(&[1]).expect("valid"), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gmres_identity() {
        // Identity should converge in 1 iteration
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![2.0, 3.0]);

        let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).expect("Should solve");
        assert!(result.converged, "GMRES should converge for identity");
        assert_relative_eq!(
            result.solution.get(&[0]).expect("valid"),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.solution.get(&[1]).expect("valid"),
            3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gmres_diagonal() {
        // Diagonal matrix should converge quickly
        let a = Array::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![4.0, 9.0, 16.0]);

        let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10)).expect("Should solve");
        assert!(result.converged, "GMRES should converge for diagonal");
        assert_relative_eq!(
            result.solution.get(&[0]).expect("valid"),
            2.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            result.solution.get(&[1]).expect("valid"),
            3.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            result.solution.get(&[2]).expect("valid"),
            4.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_gmres_precond_jacobi_simple() {
        // Test preconditioned GMRES with Jacobi preconditioner on a diagonally dominant system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let precond = JacobiPreconditioner::new(&a).expect("Should create");
        let result = gmres_precond(&a, &b, &precond, None, Some(1e-10), Some(100), Some(10))
            .expect("Should solve");

        assert!(result.converged, "Preconditioned GMRES should converge");

        // Verify solution: Ax = b
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..2 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_gmres_jacobi_convenience() {
        // Test the convenience function
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let result =
            gmres_jacobi(&a, &b, None, Some(1e-10), Some(100), Some(10)).expect("Should solve");

        assert!(result.converged, "GMRES with Jacobi should converge");

        // Verify solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..2 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_gmres_precond_identity() {
        // Identity preconditioner should give same result as unpreconditioned
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let precond = IdentityPreconditioner;
        let result = gmres_precond(&a, &b, &precond, None, Some(1e-8), Some(100), Some(10))
            .expect("Should solve");

        assert!(
            result.converged,
            "GMRES with identity preconditioner should converge"
        );

        // Solution should be x = [0, 1]
        assert_relative_eq!(
            result.solution.get(&[0]).expect("valid"),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            result.solution.get(&[1]).expect("valid"),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_fgmres_simple() {
        // Test FGMRES with a simple 2x2 system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let precond = JacobiPreconditioner::new(&a).expect("Should create");
        let result = fgmres(
            &a,
            &b,
            |v| precond.apply(v),
            None,
            Some(1e-10),
            Some(100),
            Some(10),
        )
        .expect("Should solve");

        assert!(result.converged, "FGMRES should converge");

        // Verify solution: Ax = b
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..2 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_fgmres_jacobi_convenience() {
        // Test the convenience function
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let result =
            fgmres_jacobi(&a, &b, None, Some(1e-10), Some(100), Some(10)).expect("Should solve");

        assert!(result.converged, "FGMRES with Jacobi should converge");

        // Verify solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..2 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-6
            );
        }
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
        .expect("Should solve");

        assert!(result.converged, "FGMRES with identity should converge");

        // Solution should be x = [0, 1]
        assert_relative_eq!(
            result.solution.get(&[0]).expect("valid"),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            result.solution.get(&[1]).expect("valid"),
            1.0,
            epsilon = 1e-6
        );
    }
}
