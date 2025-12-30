//! Linear system solving functions
//!
//! This module contains functions for solving linear systems, computing matrix inverses,
//! and computing pseudoinverses.

#[allow(unused_imports)] // Used conditionally based on features
use crate::array::Array;
#[allow(unused_imports)] // Used conditionally based on features
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used conditionally based on features
use num_traits::Float;
#[allow(unused_imports)] // Used conditionally based on features
use std::fmt::Debug;

/// Solve a linear system Ax = b
///
/// This function solves the linear system Ax = b, where A is a square matrix and b is a vector.
/// It dispatches to the appropriate implementation based on available features.
///
/// # Arguments
///
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
///
/// # Returns
///
/// * The solution vector x
///
/// # Errors
///
/// Returns an error if the matrix is singular or if the dimensions do not match.
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn solve<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>> {
    a.solve(b)
}

/// Solve a linear system Ax = b
#[cfg(not(feature = "matrix_decomp"))]
pub fn solve<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>> {
    a.solve(b)
}

/// Compute the inverse of a matrix
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn inv<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
) -> Result<Array<T>> {
    a.inv()
}

/// Compute the inverse of a matrix
#[cfg(not(feature = "matrix_decomp"))]
pub fn inv<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
) -> Result<Array<T>> {
    a.inv()
}

/// Compute the Moore-Penrose pseudoinverse of a matrix
///
/// The Moore-Penrose pseudoinverse is a generalization of the inverse matrix
/// that exists for any matrix, even non-square or singular matrices.
/// It is computed using SVD decomposition.
///
/// # Parameters
///
/// * `a` - The input matrix
/// * `rcond` - Cutoff for small singular values. Singular values smaller
///   than rcond * largest_singular_value are set to zero.
///   Default is 1e-15.
///
/// # Returns
///
/// The pseudoinverse of the input matrix
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn pinv<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
    rcond: Option<T>,
) -> Result<Array<T>> {
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "pinv requires a 2D matrix".to_string(),
        ));
    }

    // Perform SVD: A = U * S * V^T
    let (u, s, vt) = crate::linalg::decomposition::svd(a)?;

    // Get the cutoff value for singular values
    let rcond_val =
        rcond.unwrap_or_else(|| T::from(1e-15).expect("Failed to convert default rcond value"));

    // Find the maximum singular value to determine cutoff
    let max_singular_val = s
        .array()
        .fold(T::zero(), |max, &val| if val > max { val } else { max });
    let cutoff = max_singular_val * rcond_val;

    // Invert the non-zero singular values
    let s_data = s.to_vec();
    let mut s_inv_data = Vec::with_capacity(s_data.len());

    for &val in &s_data {
        if val > cutoff {
            s_inv_data.push(T::one() / val);
        } else {
            s_inv_data.push(T::zero());
        }
    }

    // Create vectors for s_inv needed for diagonal matrix
    let s_inv_vec = s_inv_data.clone();

    // For a matrix A of shape (m, n), u has shape (m, k), s has shape (k),
    // and vt has shape (k, n) where k = min(m, n)
    let m = shape[0];
    let n = shape[1];
    let k = std::cmp::min(m, n);

    // Construct the pseudoinverse using the formula: A^+ = V * S^+ * U^T
    // Where S^+ is a diagonal matrix with 1/s_i if s_i > cutoff, and 0 otherwise

    // 1. Create a diagonal matrix from s_inv
    let mut s_inv_diag = Array::zeros(&[k, k]);
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        s_inv_diag.set(&[i, i], s_inv_vec[i])?;
    }

    // 2. Compute V * S^+
    let v = vt.transpose();
    let vs_inv = v.matmul(&s_inv_diag)?;

    // 3. Compute (V * S^+) * U^T
    let ut = u.transpose();
    let pinv_result = vs_inv.matmul(&ut)?;

    Ok(pinv_result)
}
