//! Matrix operation functions including determinant and matrix power calculations.
//!
//! This module provides essential matrix operations that are commonly used
//! in linear algebra computations.

#[allow(unused_imports)] // Used conditionally based on features
use crate::array::Array;
#[allow(unused_imports)] // Used conditionally based on features
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used conditionally based on features
use num_traits::Float;
#[allow(unused_imports)] // Used conditionally based on features
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
/// let det_val = det(&a).expect("determinant computation should succeed for square matrix");
/// assert_eq!(det_val, -2.0);
/// ```
#[cfg(feature = "lapack")]
pub fn det<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
>(
    a: &Array<T>,
) -> Result<T> {
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
/// let a_squared = matrix_power(&a, 2).expect("matrix power should succeed for square matrix");
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
#[cfg(feature = "lapack")]
pub fn matrix_power<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
>(
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

    // For |n| > 1: binary exponentiation (exponentiation by squaring), an
    // O(log|n|) sequence of matmuls instead of the O(|n|) repeated-multiply
    // loop this replaces. Negative powers invert first, then raise that
    // inverse to the positive `|n|`; squaring itself doesn't care about
    // sign, so both directions share `binary_pow`.
    let base = if n > 0 { a.clone() } else { a.inv()? };
    binary_pow(&base, n.unsigned_abs(), size)
}

/// Exponentiation by squaring: `base^exponent` in `O(log exponent)`
/// matmuls. Only ever called by [`matrix_power`] above with `exponent >= 2`
/// (its `0`/`1`/`-1` cases are handled before this point), but the loop
/// below is self-contained and correct for any `exponent`, including `0`
/// (returns the identity untouched) and `1` (one multiply, no squaring).
#[cfg(feature = "lapack")]
fn binary_pow<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
>(
    base: &Array<T>,
    mut exponent: u32,
    size: usize,
) -> Result<Array<T>> {
    let mut result = Array::identity(size);
    let mut power_of_base = base.clone();
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.matmul(&power_of_base)?;
        }
        exponent >>= 1;
        if exponent > 0 {
            power_of_base = power_of_base.matmul(&power_of_base)?;
        }
    }
    Ok(result)
}
