//! Iterative Refinement for improving linear system solutions
//!
//! This module provides iterative refinement techniques that can improve
//! the accuracy of solutions obtained from direct or iterative solvers.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

use super::bicgstab::bicgstab;
use super::cg::conjugate_gradient;
use super::core::matvec;

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
            tol: T::from(1e-12).unwrap_or(T::epsilon()),
            min_improvement: T::from(0.5).unwrap_or(T::one() / (T::one() + T::one())),
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
/// }, None).expect("iterative refinement should succeed");
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
    T: Float + Clone + Zero + std::fmt::Debug + std::ops::AddAssign + 'static,
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
/// let result = iterative_refinement_cg(&a, &b, &x0, Some(1e-12), Some(5)).expect("iterative refinement should succeed");
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
    T: Float + Clone + Zero + std::fmt::Debug + std::ops::AddAssign + 'static,
{
    let config = RefinementConfig {
        max_iter: max_iter.unwrap_or(10),
        tol: tol.unwrap_or(T::from(1e-12).unwrap_or(T::epsilon())),
        min_improvement: T::from(0.5).unwrap_or(T::one() / (T::one() + T::one())),
    };

    iterative_refinement(
        a,
        b,
        x0,
        |mat, rhs| {
            conjugate_gradient(
                mat,
                rhs,
                None,
                Some(T::from(1e-14).unwrap_or(T::epsilon())),
                Some(500),
            )
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
    T: Float + Clone + Zero + std::fmt::Debug + std::ops::AddAssign + 'static,
{
    let config = RefinementConfig {
        max_iter: max_iter.unwrap_or(10),
        tol: tol.unwrap_or(T::from(1e-12).unwrap_or(T::epsilon())),
        min_improvement: T::from(0.5).unwrap_or(T::one() / (T::one() + T::one())),
    };

    iterative_refinement(
        a,
        b,
        x0,
        |mat, rhs| {
            bicgstab(
                mat,
                rhs,
                None,
                Some(T::from(1e-14).unwrap_or(T::epsilon())),
                Some(500),
            )
            .map(|r| r.solution)
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
    let out = r.array_mut();
    for i in 0..n {
        let bi = b.get(&[i])?;
        let axi = ax.get(&[i])?;
        out[[i]] = bi - axi;
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
    let out = result.array_mut();
    for i in 0..n {
        let ai = a.get(&[i])?;
        let bi = b.get(&[i])?;
        out[[i]] = ai + bi;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::iterative_solvers::core::matvec;
    use approx::assert_relative_eq;

    #[test]
    fn test_iterative_refinement_basic() {
        // Well-conditioned SPD system
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // True solution is approximately [0.0909, 0.6363]
        // Start with a slightly inaccurate initial guess
        let x0 = Array::from_vec(vec![0.09, 0.64]);

        let result =
            iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(5)).expect("Should refine");

        // Refinement should improve the solution
        assert!(result.improvement_factor >= 1.0);

        // Verify refined solution
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..2 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn test_iterative_refinement_already_converged() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        // Get exact solution using CG first
        let cg_result =
            conjugate_gradient(&a, &b, None, Some(1e-14), Some(100)).expect("Should solve");
        let x0 = cg_result.solution;

        // Refinement should detect already converged
        let result =
            iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(5)).expect("Should refine");

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

        let result =
            iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(10)).expect("Should refine");

        // Should converge with improved solution
        assert!(result.improvement_factor > 1.0);

        // Verify solution accuracy
        let ax = matvec(&a, &result.solution).expect("matvec should work");
        for i in 0..4 {
            assert_relative_eq!(
                ax.get(&[i]).expect("valid"),
                b.get(&[i]).expect("valid"),
                epsilon = 1e-8
            );
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
        let initial = bicgstab(&a, &b, None, Some(1e-4), Some(50)).expect("Should solve");

        // Refine with BiCGSTAB
        let result = iterative_refinement_bicgstab(&a, &b, &initial.solution, Some(1e-10), Some(5))
            .expect("Should refine");

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
        .expect("Should refine");

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

        let result =
            iterative_refinement_cg(&a, &b, &x0, Some(1e-10), Some(5)).expect("Should refine");

        // Check result fields are populated
        assert_eq!(result.solution.size(), 2);
        assert!(result.iterations > 0);
        assert!(result.initial_residual > 0.0);
        assert!(result.final_residual >= 0.0);
        assert!(result.improvement_factor > 0.0);
    }
}
