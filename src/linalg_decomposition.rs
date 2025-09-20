//! Matrix decomposition functions
//!
//! This module contains matrix decomposition functions extracted from the main linalg module
//! for better organization and modularity. Includes:
//!
//! - QR decomposition (`qr`)
//! - Cholesky decomposition (`cholesky`)
//! - Eigenvalue decomposition (`eig`)
//! - Singular value decomposition (`svd`)
//! - Matrix rank computation (`matrix_rank`)
//!
//! All functions support both feature-gated (with `matrix_decomp`) and non-feature-gated versions
//! to maintain compatibility across different build configurations.

#[allow(unused_imports)] // Used conditionally based on features
use crate::array::Array;
#[allow(unused_imports)] // Used conditionally based on features
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used conditionally based on features
use num_traits::Float;
#[allow(unused_imports)] // Used conditionally based on features
use std::fmt::Debug;

/// Compute the rank of a matrix
///
/// Computes the rank of a matrix using singular value decomposition.
/// The rank is determined by counting the number of singular values
/// that are greater than a specified tolerance.
///
/// # Arguments
///
/// * `a` - Input 2D matrix
/// * `tol` - Tolerance for determining rank. If None, uses default tolerance
///   based on machine precision and matrix size
///
/// # Returns
///
/// The rank of the matrix as a usize
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::matrix_rank;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let rank = matrix_rank(&a, None).unwrap();
/// assert_eq!(rank, 2);
/// ```
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn matrix_rank<T: Float + Clone + Debug + ndarray_linalg::Lapack>(
    a: &Array<T>,
    tol: Option<T>,
) -> Result<usize>
where
    <T as ndarray_linalg::Scalar>::Real: Clone + num_traits::Float,
{
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "matrix_rank requires a 2D matrix".to_string(),
        ));
    }

    // Compute SVD to get singular values using the proper implementation
    let (_, s, _) = crate::new_modules::matrix_decomp::svd(a)?;

    // Get the tolerance
    let tol_val = match tol {
        Some(t) => {
            // Convert user tolerance to Real type
            <<T as ndarray_linalg::Scalar>::Real as num_traits::NumCast>::from(t)
                .unwrap_or_else(<<T as ndarray_linalg::Scalar>::Real as num_traits::Zero>::zero)
        }
        None => {
            // Default is max(M, N) * eps * max(S)
            let m = shape[0];
            let n = shape[1];
            let max_dim = std::cmp::max(m, n);
            let eps = <<T as ndarray_linalg::Scalar>::Real as num_traits::Float>::epsilon();
            let s_data = s.to_vec();
            let max_s = s_data.iter().fold(
                <<T as ndarray_linalg::Scalar>::Real as num_traits::Zero>::zero(),
                |max, &val| if val > max { val } else { max },
            );

            <<T as ndarray_linalg::Scalar>::Real as num_traits::NumCast>::from(max_dim).unwrap()
                * eps
                * max_s
        }
    };

    // Count singular values larger than tolerance
    let s_data = s.to_vec();
    let rank = s_data.iter().filter(|&&val| val > tol_val).count();

    Ok(rank)
}

/// Compute the QR decomposition of a matrix
///
/// Performs QR decomposition of a matrix A such that A = Q * R, where
/// Q is an orthogonal matrix and R is an upper triangular matrix.
///
/// # Arguments
///
/// * `a` - Input matrix to decompose
///
/// # Returns
///
/// A tuple (Q, R) where Q is orthogonal and R is upper triangular
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::qr;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let (q, r) = qr(&a).unwrap();
/// ```
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn qr<T: Float + Clone + Debug + ndarray_linalg::Lapack>(
    a: &Array<T>,
) -> Result<(Array<T>, Array<T>)> {
    // Use the proper implementation from new_modules
    crate::new_modules::matrix_decomp::qr(a)
}

/// Compute the QR decomposition of a matrix
///
/// Performs QR decomposition of a matrix A such that A = Q * R, where
/// Q is an orthogonal matrix and R is an upper triangular matrix.
///
/// # Arguments
///
/// * `a` - Input matrix to decompose
///
/// # Returns
///
/// A tuple (Q, R) where Q is orthogonal and R is upper triangular
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::qr;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let (q, r) = qr(&a).unwrap();
/// ```
#[cfg(not(feature = "matrix_decomp"))]
pub fn qr<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
) -> Result<(Array<T>, Array<T>)> {
    a.qr()
}

/// Compute the Cholesky decomposition of a matrix
///
/// Performs Cholesky decomposition of a positive definite matrix A such that
/// A = L * L^T, where L is a lower triangular matrix.
///
/// # Arguments
///
/// * `a` - Input positive definite matrix
///
/// # Returns
///
/// The lower triangular matrix L
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::cholesky;
///
/// // Create a positive definite matrix
/// let a = Array::from_vec(vec![4.0, 2.0, 2.0, 5.0]).reshape(&[2, 2]);
/// let l = cholesky(&a).unwrap();
/// ```
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn cholesky<T: Float + Clone + Debug + ndarray_linalg::Lapack>(
    a: &Array<T>,
) -> Result<Array<T>> {
    // Use the proper implementation from new_modules
    crate::new_modules::matrix_decomp::cholesky(a)
}

/// Compute the Cholesky decomposition of a matrix
///
/// Performs Cholesky decomposition of a positive definite matrix A such that
/// A = L * L^T, where L is a lower triangular matrix.
///
/// # Arguments
///
/// * `a` - Input positive definite matrix
///
/// # Returns
///
/// The lower triangular matrix L
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::cholesky;
///
/// // Create a positive definite matrix
/// let a = Array::from_vec(vec![4.0, 2.0, 2.0, 5.0]).reshape(&[2, 2]);
/// let l = cholesky(&a).unwrap();
/// ```
#[cfg(not(feature = "matrix_decomp"))]
pub fn cholesky<
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
    a.cholesky()
}

/// Compute the eigenvalues and eigenvectors of a square matrix
///
/// Computes the eigenvalues and eigenvectors of a square matrix.
/// Optionally sorts the results by eigenvalue magnitude.
///
/// # Parameters
///
/// * `a` - The input square matrix
/// * `sort` - Sort eigenvalues and eigenvectors by eigenvalue magnitude.
///   Options: "asc" (ascending), "desc" (descending), or None (no sorting)
///
/// # Returns
///
/// A tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
/// and eigenvectors is a 2D array with eigenvectors as columns
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::eig;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 2.0, 1.0]).reshape(&[2, 2]);
/// let (eigenvals, eigenvecs) = eig(&a, None).unwrap();
/// ```
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn eig<T: Float + Clone + Debug + ndarray_linalg::Lapack>(
    a: &Array<T>,
    sort: Option<&str>,
) -> Result<(Array<T>, Array<T>)> {
    // Get the eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = a.eig()?;

    // Return if no sorting is requested
    if sort.is_none() {
        return Ok((eigenvalues, eigenvectors));
    }

    let sort_option = sort.unwrap();

    if sort_option != "asc" && sort_option != "desc" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Invalid sort option: {}. Must be 'asc', 'desc', or None",
            sort_option
        )));
    }

    // Get the shape of the eigenvectors matrix
    let evec_shape = eigenvectors.shape();

    // Convert eigenvalues to vector for sorting
    let evals_data = eigenvalues.to_vec();
    let n = evals_data.len();

    // Create indices for sorting
    let mut indices: Vec<usize> = (0..n).collect();

    // Sort indices by eigenvalue magnitude
    indices.sort_by(|&i, &j| {
        let a_abs = num_traits::Float::abs(evals_data[i]);
        let b_abs = num_traits::Float::abs(evals_data[j]);

        if sort_option == "asc" {
            a_abs.partial_cmp(&b_abs).unwrap()
        } else {
            b_abs.partial_cmp(&a_abs).unwrap()
        }
    });

    // Create sorted eigenvalues array
    let mut sorted_evals = Vec::with_capacity(n);
    for &idx in &indices {
        sorted_evals.push(evals_data[idx]);
    }

    // Create sorted eigenvectors array
    let evecs_data = eigenvectors.to_vec();
    let eigvec_size = evec_shape[0];
    let mut sorted_evecs = Vec::with_capacity(evecs_data.len());

    for &idx in &indices {
        // Extract the eigenvector column
        for i in 0..eigvec_size {
            let evec_idx = i * n + idx;
            sorted_evecs.push(evecs_data[evec_idx]);
        }
    }

    // Convert to Array objects
    let sorted_eigenvalues = Array::from_vec(sorted_evals);
    let sorted_eigenvectors = Array::from_vec(sorted_evecs).reshape(&evec_shape);

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

/// Compute the eigenvalues and eigenvectors of a square matrix
///
/// Computes the eigenvalues and eigenvectors of a square matrix.
/// Optionally sorts the results by eigenvalue magnitude.
///
/// # Parameters
///
/// * `a` - The input square matrix
/// * `sort` - Sort eigenvalues and eigenvectors by eigenvalue magnitude.
///   Options: "asc" (ascending), "desc" (descending), or None (no sorting)
///
/// # Returns
///
/// A tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
/// and eigenvectors is a 2D array with eigenvectors as columns
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::eig;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 2.0, 1.0]).reshape(&[2, 2]);
/// let (eigenvals, eigenvecs) = eig(&a, None).unwrap();
/// ```
#[cfg(not(feature = "matrix_decomp"))]
pub fn eig<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
    sort: Option<&str>,
) -> Result<(Array<T>, Array<T>)> {
    // Get the eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = a.eig()?;

    // Return if no sorting is requested
    if sort.is_none() {
        return Ok((eigenvalues, eigenvectors));
    }

    let sort_option = sort.unwrap();

    if sort_option != "asc" && sort_option != "desc" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Invalid sort option: {}. Must be 'asc', 'desc', or None",
            sort_option
        )));
    }

    // Get the shape of the eigenvectors matrix
    let evec_shape = eigenvectors.shape();

    // Convert eigenvalues to vector for sorting
    let evals_data = eigenvalues.to_vec();
    let n = evals_data.len();

    // Create indices for sorting
    let mut indices: Vec<usize> = (0..n).collect();

    // Sort indices by eigenvalue magnitude
    indices.sort_by(|&i, &j| {
        let a_abs = num_traits::Float::abs(evals_data[i]);
        let b_abs = num_traits::Float::abs(evals_data[j]);

        if sort_option == "asc" {
            a_abs.partial_cmp(&b_abs).unwrap()
        } else {
            b_abs.partial_cmp(&a_abs).unwrap()
        }
    });

    // Create sorted eigenvalues array
    let mut sorted_evals = Vec::with_capacity(n);
    for &idx in &indices {
        sorted_evals.push(evals_data[idx]);
    }

    // Create sorted eigenvectors array
    let evecs_data = eigenvectors.to_vec();
    let eigvec_size = evec_shape[0];
    let mut sorted_evecs = Vec::with_capacity(evecs_data.len());

    for &idx in &indices {
        // Extract the eigenvector column
        for i in 0..eigvec_size {
            let evec_idx = i * n + idx;
            sorted_evecs.push(evecs_data[evec_idx]);
        }
    }

    // Convert to Array objects
    let sorted_eigenvalues = Array::from_vec(sorted_evals);
    let sorted_eigenvectors = Array::from_vec(sorted_evecs).reshape(&evec_shape);

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

/// Compute the singular value decomposition of a matrix
///
/// Performs singular value decomposition of a matrix A such that
/// A = U * S * V^T, where U and V are orthogonal matrices and S is
/// a diagonal matrix containing the singular values.
///
/// # Arguments
///
/// * `a` - Input matrix to decompose
///
/// # Returns
///
/// A tuple (U, S, V^T) where:
/// - U: Left singular vectors (orthogonal matrix)
/// - S: Singular values (1D array)
/// - V^T: Right singular vectors transposed (orthogonal matrix)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::svd;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let (u, s, vt) = svd(&a).unwrap();
/// ```
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub fn svd<T: Float + Clone + Debug + ndarray_linalg::Lapack>(
    a: &Array<T>,
) -> Result<(Array<T>, Array<T>, Array<T>)>
where
    <T as ndarray_linalg::Scalar>::Real: Float + Clone + Debug,
{
    // Use the proper implementation from new_modules
    let (u, s_vec, vt) = crate::new_modules::matrix_decomp::svd(a)?;

    // Convert singular values vector to diagonal matrix
    let m = u.shape()[0];
    let n = vt.shape()[0];
    let k = s_vec.len();
    let mut s = Array::zeros(&[m, n]);
    for i in 0..k.min(m).min(n) {
        let val = s_vec.get(&[i])?;
        // Convert Real to T using NumCast which works for f32/f64
        if let Some(t_val) = num_traits::NumCast::from(val) {
            s.set(&[i, i], t_val)?;
        }
    }

    Ok((u, s, vt))
}

/// Compute the singular value decomposition of a matrix
///
/// Performs singular value decomposition of a matrix A such that
/// A = U * S * V^T, where U and V are orthogonal matrices and S is
/// a diagonal matrix containing the singular values.
///
/// # Arguments
///
/// * `a` - Input matrix to decompose
///
/// # Returns
///
/// A tuple (U, S, V^T) where:
/// - U: Left singular vectors (orthogonal matrix)
/// - S: Singular values (1D array)
/// - V^T: Right singular vectors transposed (orthogonal matrix)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::svd;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let (u, s, vt) = svd(&a).unwrap();
/// ```
#[cfg(not(feature = "matrix_decomp"))]
pub fn svd<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
) -> Result<(Array<T>, Array<T>, Array<T>)> {
    a.svd()
}

/// Compute the rank of a matrix
///
/// Computes the rank of a matrix using singular value decomposition.
/// The rank is determined by counting the number of singular values
/// that are greater than a specified tolerance.
///
/// # Arguments
///
/// * `a` - Input 2D matrix
/// * `tol` - Tolerance for determining rank. If None, uses default tolerance
///   based on machine precision and matrix size
///
/// # Returns
///
/// The rank of the matrix as a usize
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::decomposition::matrix_rank;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let rank = matrix_rank(&a, None).unwrap();
/// assert_eq!(rank, 2);
/// ```
#[cfg(not(feature = "matrix_decomp"))]
pub fn matrix_rank<
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
>(
    a: &Array<T>,
    tol: Option<T>,
) -> Result<usize> {
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "matrix_rank requires a 2D matrix".to_string(),
        ));
    }

    // Compute SVD to get singular values using the non-feature-gated implementation
    let (_, s, _) = svd(a)?;

    // Get the tolerance
    let tol_val = match tol {
        Some(t) => t,
        None => {
            // Default is max(M, N) * eps * max(S)
            let m = shape[0];
            let n = shape[1];
            let max_dim = if m > n { m } else { n };
            let eps = T::epsilon();

            // Find max singular value
            let s_data = s.to_vec();
            let max_s = s_data
                .iter()
                .fold(T::zero(), |max, &val| if val > max { val } else { max });

            T::from(max_dim).unwrap() * eps * max_s
        }
    };

    // Count singular values larger than tolerance
    let s_data = s.to_vec();
    let rank = s_data.iter().filter(|&&val| val > tol_val).count();

    Ok(rank)
}
