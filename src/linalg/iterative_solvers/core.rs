//! Core types and utilities for iterative solvers
//!
//! This module provides common types and helper functions used across
//! all iterative solver implementations.

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
            tol: T::from(1e-6).expect("1e-6 is a valid f64 constant"),
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

/// Helper function to compute norm of a vector slice
#[inline]
pub fn compute_norm_vec<T: Float>(v: &[T]) -> T {
    v.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt()
}

/// Helper function to compute dot product of vector slices
#[inline]
pub fn dot_vec<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// Matrix-vector multiplication that always returns a 1D vector
pub fn matvec<T>(a: &Array<T>, x: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone + Zero,
{
    let n = x.size();
    let x_col = x.clone().reshape(&[n, 1]);
    let result = a.matmul(&x_col)?;
    Ok(result.reshape(&[n]))
}

/// Compute the L2 norm of a vector
pub fn compute_norm<T>(v: &Array<T>) -> Result<T>
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

/// Validate that the matrix is square and compatible with vector
pub fn validate_system<T>(a: &Array<T>, b: &Array<T>) -> Result<usize>
where
    T: Float + Clone,
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

    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_config_default() {
        let config: SolverConfig<f64> = SolverConfig::default();
        assert_eq!(config.max_iter, 1000);
        assert!((config.tol - 1e-6).abs() < 1e-12);
        assert_eq!(config.restart, Some(30));
        assert!(!config.use_preconditioner);
    }

    #[test]
    fn test_compute_norm_vec() {
        let v = vec![3.0, 4.0];
        let norm = compute_norm_vec(&v);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_vec() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_vec(&a, &b);
        assert!((dot - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_system() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![1.0, 2.0]);
        let n = validate_system(&a, &b).expect("Should validate");
        assert_eq!(n, 2);
    }

    #[test]
    fn test_validate_system_non_square() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0, 2.0]);
        let result = validate_system(&a, &b);
        assert!(result.is_err());
    }
}
