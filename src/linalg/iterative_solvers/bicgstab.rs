//! BiCGSTAB (Biconjugate Gradient Stabilized) method
//!
//! This module provides the BiCGSTAB method for solving non-symmetric
//! linear systems.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

use super::core::{compute_norm_vec, dot_vec, matvec, SolverResult};

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
/// let result = bicgstab(&a, &b, None, Some(1e-6), Some(100)).expect("valid bicgstab solver params");
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
        if r0_dot_v.abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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
        if t_dot_t.abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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

        if rho.abs() < T::from(1e-14).unwrap_or(T::epsilon()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bicgstab_simple() {
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = bicgstab(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
    }

    #[test]
    fn test_bicgstab_identity() {
        // Identity matrix should converge quickly
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![3.0, 4.0]);

        let result = bicgstab(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);

        // Solution should be exactly b
        for i in 0..2 {
            assert_relative_eq!(
                result.solution.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn test_bicgstab_diagonal() {
        // Diagonal matrix
        let a = Array::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).reshape(&[3, 3]);
        let b = Array::from_vec(vec![4.0, 9.0, 16.0]);

        let result = bicgstab(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);

        // Verify solution x = [2, 3, 4]
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
    fn test_bicgstab_non_symmetric() {
        // Non-symmetric matrix
        let a = Array::from_vec(vec![4.0, 1.0, 2.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);

        let result = bicgstab(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);

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
    fn test_bicgstab_larger_system() {
        // 4x4 system
        let a = Array::from_vec(vec![
            4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0,
        ])
        .reshape(&[4, 4]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = bicgstab(&a, &b, None, Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);

        // Verify solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..4 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_bicgstab_zero_rhs() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![0.0, 0.0]);

        let result = bicgstab(&a, &b, None, Some(1e-6), Some(100)).expect("Should solve");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_bicgstab_with_initial_guess() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 5.0]);
        let x0 = Array::from_vec(vec![1.0, 1.0]); // Close to solution

        let result = bicgstab(&a, &b, Some(&x0), Some(1e-10), Some(100)).expect("Should solve");
        assert!(result.converged);

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
}
