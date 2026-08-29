//! Conjugate Gradient methods
//!
//! This module provides the Conjugate Gradient (CG) method and its preconditioned
//! variant (PCG) for solving symmetric positive definite linear systems.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

use super::core::{compute_norm_vec, dot_vec, matvec, SolverResult};
use super::preconditioners::{
    IncompleteCholeskyPreconditioner, JacobiPreconditioner, Preconditioner, SSORPreconditioner,
};

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
/// let result = conjugate_gradient(&a, &b, None, Some(1e-6), Some(100)).expect("CG should converge for SPD matrix");
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
    T: Float + Clone + Zero + 'static,
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

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap_or(T::epsilon()));
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
/// let precond = JacobiPreconditioner::new(&a).expect("Jacobi preconditioner creation should succeed");
/// let result = pcg(&a, &b, &precond, None, Some(1e-6), Some(100)).expect("PCG should converge for SPD matrix");
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
    T: Float + Clone + Zero + 'static,
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

    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap_or(T::epsilon()));
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
        if p_dot_ap.is_zero() || p_dot_ap.abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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
    T: Float + Clone + Zero + 'static,
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
    T: Float + Clone + Zero + 'static,
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
    T: Float + Clone + Zero + 'static,
{
    let precond = IncompleteCholeskyPreconditioner::new(a)?;
    pcg(a, b, &precond, x0, tol, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::iterative_solvers::preconditioners::IdentityPreconditioner;
    use approx::assert_relative_eq;

    #[test]
    fn test_cg_simple() {
        // Simple 2x2 SPD system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = conjugate_gradient(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_cg_identity() {
        // Identity matrix should converge in 1 iteration
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![3.0, 4.0]);

        let result =
            conjugate_gradient(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_pcg_jacobi() {
        // SPD matrix
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = pcg_jacobi(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);

        // Verify solution approximately satisfies Ax = b
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..2 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn test_pcg_with_identity_preconditioner() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let precond = IdentityPreconditioner;
        let result = pcg(&a, &b, &precond, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
    }

    #[test]
    fn test_pcg_larger_system() {
        // 3x3 SPD system
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![1.0, 2.0, 1.0]);

        let result = pcg_jacobi(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);

        // Verify solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..3 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn test_pcg_ssor_preconditioning() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // omega = 1.0 (standard SOR)
        let result = pcg_ssor(&a, &b, 1.0, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
    }

    #[test]
    fn test_pcg_ichol_preconditioning() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = pcg_ichol(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
    }

    #[test]
    fn test_pcg_vs_cg_comparison() {
        // Compare convergence of PCG vs CG
        let a = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![1.0, 2.0, 1.0]);

        // Standard CG
        let cg_result =
            conjugate_gradient(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");

        // PCG with Jacobi
        let pcg_result = pcg_jacobi(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");

        // Both should converge
        assert!(cg_result.converged);
        assert!(pcg_result.converged);

        // PCG should converge in fewer or equal iterations for this well-conditioned system
        // (Jacobi preconditioning helps with diagonally dominant matrices)
        assert!(pcg_result.iterations <= cg_result.iterations + 2); // Allow small variance
    }
}
