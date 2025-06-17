//! SciRS2 compatibility module for advanced statistical distributions
//!
//! This module provides compatibility functions for advanced statistical distributions
//! that are available when SciRS2 integration is enabled. When SciRS2 is not available,
//! it provides implementations using NumRS2's built-in advanced distributions.
//!
//! This approach ensures compatibility while maintaining functionality.

#![allow(unexpected_cfgs)]
#![allow(unused_imports)]

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use std::fmt::{Debug, Display};

/// Adapter function for SciRS2's noncentral chi-square distribution
///
/// This function creates a noncentral chi-square distribution using SciRS2
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `df` - Degrees of freedom (> 0)
/// * `nonc` - Non-centrality parameter (≥ 0)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// * A NumRS2 array with samples from the noncentral chi-square distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
pub fn noncentral_chisquare<T>(df: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Use NumRS2's built-in noncentral chi-square distribution
    crate::random::distributions::noncentral_chisquare(df, nonc, shape)
}

/// Adapter function for SciRS2's noncentral F distribution  
pub fn noncentral_f<T>(dfnum: T, dfden: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Use NumRS2's built-in noncentral F distribution
    crate::random::distributions::noncentral_f(dfnum, dfden, nonc, shape)
}

/// Linear system solver using SciRS2
#[cfg(feature = "lapack")]
pub fn solve_linear_system<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display + ndarray_linalg::Lapack,
{
    // Use NumRS2's built-in linear algebra solve function
    crate::linalg::solve(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_noncentral_chisquare() {
        let samples = noncentral_chisquare(2.0f64, 1.0f64, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // All samples should be positive
        for val in samples.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_noncentral_f() {
        let samples = noncentral_f(2.0f64, 3.0f64, 1.0f64, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // All samples should be positive
        for val in samples.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[cfg(feature = "lapack")]
    #[test]
    fn test_solve_linear_system() {
        // Test a simple 2x2 system
        // System: 2x + y = 5, x + 3y = 6
        // Solution: x = 2, y = 1

        let a_data = vec![2.0f64, 1.0f64, 1.0f64, 3.0f64];
        let a = Array::from_vec(a_data).reshape(&[2, 2]);

        let b_data = vec![5.0f64, 6.0f64];
        let b = Array::from_vec(b_data);

        let x = solve_linear_system(&a, &b).unwrap();

        // Check shape
        assert_eq!(x.shape(), vec![2]);

        // Verify the solution by checking A*x = b rather than exact values
        // (since different solvers may have different numerical behavior)
        let ax = a.matmul(&x.reshape(&[2, 1])).unwrap().reshape(&[2]);
        let ax_data = ax.to_vec();
        let b_data = b.to_vec();

        assert_relative_eq!(ax_data[0], b_data[0], epsilon = 1e-10);
        assert_relative_eq!(ax_data[1], b_data[1], epsilon = 1e-10);
    }
}
