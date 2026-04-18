//! MINRES (Minimal Residual) method
//!
//! This module provides the MINRES method for solving symmetric indefinite
//! linear systems.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

use super::core::{compute_norm_vec, dot_vec, matvec, SolverResult};

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
/// let result = minres(&a, &b, None, Some(1e-6), Some(100)).expect("valid minres solver params");
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
    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap_or(T::epsilon()));
    let max_iter = max_iter.unwrap_or(n * 2); // Use 2n for safety

    // Convert to vectors for efficient computation
    let b_vec = b.to_vec();
    let b_norm = compute_norm_vec(&b_vec);

    if b_norm < T::from(1e-14).unwrap_or(T::epsilon()) {
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

    if beta1 < T::from(1e-14).unwrap_or(T::epsilon()) {
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
    let mut beta_k = T::zero(); // beta_k (from previous Lanczos step)

    let mut iter = 0;

    // MINRES main loop
    for k in 0..max_iter {
        iter = k + 1;

        // Lanczos step: compute A*v
        let v_arr = Array::from_vec(v.clone());
        let av = matvec(a, &v_arr)?;
        let av_vec = av.to_vec();

        // alpha_k = v^T * A * v
        let alpha_k = dot_vec(&v, &av_vec);

        // v_new = A*v - alpha_k*v - beta_k*v_{k-1}
        let v_new: Vec<T> = (0..n)
            .map(|i| av_vec[i] - alpha_k * v[i] - beta_k * v_prev[i])
            .collect();

        let beta_next = compute_norm_vec(&v_new);

        // Apply previous rotations to the k-th column of the tridiagonal matrix
        // Column k has entries: [..., 0, beta_k, alpha_k] at positions k-2, k-1, k
        //
        // After G_{k-2}: position k-2 gets epsilon_k = s_{k-2} * beta_k
        //                position k-1 gets c_{k-2} * beta_k
        // After G_{k-1}: position k-1 gets delta_k = c_{k-1}*c_{k-2}*beta_k + s_{k-1}*alpha_k
        //                position k gets gamma_tilde_k = -s_{k-1}*c_{k-2}*beta_k + c_{k-1}*alpha_k

        // epsilon_k: entry at (k-2, k) after rotations - used in three-term recurrence
        let epsilon_k = s_prev2 * beta_k;

        // Intermediate after G_{k-2}
        let beta_rotated = c_prev2 * beta_k;

        // delta_k: entry at (k-1, k) after G_{k-1}
        let delta_k = c_prev * beta_rotated + s_prev * alpha_k;

        // gamma_tilde_k: entry at (k, k) before new rotation
        let gamma_tilde = -s_prev * beta_rotated + c_prev * alpha_k;

        // Compute new Givens rotation to eliminate beta_{k+1}
        let gamma_k = (gamma_tilde * gamma_tilde + beta_next * beta_next).sqrt();
        let (c_k, s_k) = if gamma_k > T::from(1e-14).unwrap_or(T::epsilon()) {
            (gamma_tilde / gamma_k, beta_next / gamma_k)
        } else {
            (T::one(), T::zero())
        };

        // Update direction vector with three-term recurrence:
        // d_k = (v_k - delta_k * d_{k-1} - epsilon_k * d_{k-2}) / gamma_k
        let d_new: Vec<T> = if gamma_k > T::from(1e-14).unwrap_or(T::epsilon()) {
            (0..n)
                .map(|i| (v[i] - delta_k * d_prev[i] - epsilon_k * d_prev2[i]) / gamma_k)
                .collect()
        } else {
            vec![T::zero(); n]
        };

        // Apply rotation to right-hand side and update solution
        // tau_k = c_k * phi_bar_{k-1}
        let tau_k = c_k * phi_bar;

        // x_k = x_{k-1} + tau_k * d_k
        for i in 0..n {
            x_vec[i] = x_vec[i] + tau_k * d_new[i];
        }

        // Update phi_bar_k = -s_k * phi_bar_{k-1}
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
        if beta_next < T::from(1e-14).unwrap_or(T::epsilon()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_minres_symmetric_indefinite() {
        // Symmetric indefinite matrix (has both positive and negative eigenvalues)
        let a = Array::from_vec(vec![
            2.0, 1.0, 1.0, -1.0, // eigenvalues: ~2.414, ~-1.414
        ])
        .reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 0.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(
            result.converged,
            "MINRES should converge for symmetric indefinite system"
        );

        // Verify solution: A*x ~ b
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
    fn test_minres_spd_matrix() {
        // MINRES should also work for SPD matrices (where CG would work)
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged, "MINRES should work for SPD matrices");

        // Verify solution
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
    fn test_minres_saddle_point() {
        // Saddle point problem (common in constrained optimization)
        let a = Array::from_vec(vec![
            3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, -1.0, // Indefinite
        ])
        .reshape(&[3, 3]);
        let b = Array::from_vec(vec![1.0, 2.0, 1.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(150)).expect("Should solve");
        assert!(
            result.converged,
            "MINRES should handle saddle point problems"
        );

        // Verify solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..3 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn test_minres_identity_matrix() {
        // Identity matrix should converge very quickly
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![3.0, 4.0]);

        let result = minres(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "Identity should converge in <= 2 iterations"
        );

        // Solution should be exactly b
        for i in 0..2 {
            assert_relative_eq!(
                result.solution.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
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

        let result = minres(&a, &b, None, Some(1e-6), Some(200)).expect("Should solve");
        assert!(
            result.converged,
            "MINRES should converge for larger indefinite system"
        );

        // Verify solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..4 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-4
            );
        }
    }

    #[test]
    fn test_minres_with_initial_guess() {
        // Test with non-zero initial guess
        let a = Array::from_vec(vec![2.0, 1.0, 1.0, -1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 0.0]);
        let x0 = Array::from_vec(vec![0.5, 0.5]); // Initial guess

        let result = minres(&a, &b, Some(&x0), Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);

        // Verify solution
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
    fn test_minres_zero_rhs() {
        // Zero right-hand side should give zero solution immediately
        let a = Array::from_vec(vec![2.0, 1.0, 1.0, -1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![0.0, 0.0]);

        let result = minres(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
        assert_eq!(result.iterations, 0, "Zero RHS should converge immediately");

        for i in 0..2 {
            assert_relative_eq!(
                result.solution.get(&[i]).expect("valid"),
                0.0,
                epsilon = 1e-10
            );
        }
    }
}
