//! Matrix operation functions including determinant and matrix power calculations.
//!
//! This module provides essential matrix operations that are commonly used
//! in linear algebra computations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Compute the determinant of a matrix
///
/// # Arguments
/// * `a` - Input square matrix for which to compute the determinant
///
/// # Returns
/// * `Result<T>` - The determinant value if successful, error otherwise
///
/// # Errors
/// * `NumRs2Error::DimensionMismatch` - If the input is not a square matrix
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let det_val = det(&a).unwrap();
/// assert_eq!(det_val, -2.0);
/// ```
pub fn det<T: Float + Clone + Debug + ndarray_linalg::Lapack>(a: &Array<T>) -> Result<T> {
    a.det()
}

/// Compute the matrix power (A raised to power n)
///
/// # Arguments
/// * `a` - Input square matrix to raise to power n
/// * `n` - The power to raise the matrix to (can be positive, negative, or zero)
///
/// # Returns
/// * `Result<Array<T>>` - The matrix raised to power n if successful, error otherwise
///
/// # Errors
/// * `NumRs2Error::DimensionMismatch` - If the input is not a square matrix
/// * Matrix inversion errors if n is negative and matrix is singular
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![2.0, 0.0, 0.0, 2.0]).reshape(&[2, 2]);
/// let a_squared = matrix_power(&a, 2).unwrap();
/// let expected = Array::from_vec(vec![4.0, 0.0, 0.0, 4.0]).reshape(&[2, 2]);
/// // Result should be [[4, 0], [0, 4]]
/// ```
///
/// # Special cases
/// * `n = 0`: Returns the identity matrix of the same size
/// * `n = 1`: Returns a copy of the original matrix  
/// * `n = -1`: Returns the matrix inverse
/// * `n > 1`: Computes A * A * ... * A (n times)
/// * `n < -1`: Computes (A^-1)^|n|
pub fn matrix_power<T: Float + Clone + Debug + ndarray_linalg::Lapack>(
    a: &Array<T>,
    n: i32,
) -> Result<Array<T>> {
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "matrix_power requires a square matrix".to_string(),
        ));
    }

    let size = shape[0];

    // Handle special cases
    if n == 0 {
        // Return identity matrix
        return Ok(Array::identity(size));
    }

    if n == 1 {
        // Return a copy of the original matrix
        return Ok(a.clone());
    }

    if n == -1 {
        // Return the inverse
        return a.inv();
    }

    // For higher powers, we should implement a more efficient algorithm
    // using binary exponentiation. For simplicity, we'll use a direct approach
    // for now.

    if n > 0 {
        let mut result = a.clone();
        for _ in 1..n {
            result = result.matmul(a)?;
        }
        Ok(result)
    } else {
        // For negative powers, compute the inverse first
        let inv = a.inv()?;
        let abs_n = (-n) as u32;

        let mut result = inv.clone();
        for _ in 1..abs_n {
            result = result.matmul(&inv)?;
        }
        Ok(result)
    }
}
