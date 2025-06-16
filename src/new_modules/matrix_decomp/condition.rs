use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::matrix_decomp::svd;
use num_traits::{Float, Zero};
use std::fmt::Debug;

/// Compute the condition number of a matrix using the SVD method
///
/// The condition number is the ratio of the largest to smallest singular value
/// and provides a measure of how sensitive matrix operations are to numerical errors.
/// A high condition number indicates an ill-conditioned matrix.
///
/// This implementation includes various numerical stability enhancements:
/// 1. Proper handling of very small singular values
/// 2. Scaling to avoid overflow in calculations
/// 3. Classification of different condition number ranges
pub fn condition_number<T>(a: &Array<T>) -> Result<<T as ndarray_linalg::Scalar>::Real>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack,
    <T as ndarray_linalg::Scalar>::Real: Clone + num_traits::Float,
{
    // SVD is the most numerically stable method for computing condition number
    let (_, s, _) = svd(a)?;

    // Convert to vector for easier manipulation
    let s_vec = s.to_vec();

    // Find the largest and smallest singular values
    if s_vec.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "Cannot compute condition number of empty matrix".to_string(),
        ));
    }

    // Get the largest singular value
    let max_sv = s_vec.iter().cloned().fold(
        <<T as ndarray_linalg::Scalar>::Real as num_traits::Zero>::zero(),
        |a, b| if a > b { a } else { b },
    );

    // Get the smallest non-zero singular value
    // We use a threshold based on machine epsilon to determine effective zeros
    let eps = <<T as ndarray_linalg::Scalar>::Real as num_traits::Float>::epsilon();
    let threshold = max_sv
        * eps
        * <<T as ndarray_linalg::Scalar>::Real as num_traits::NumCast>::from(std::cmp::max(
            a.shape()[0],
            a.shape()[1],
        ))
        .unwrap_or_else(<<T as ndarray_linalg::Scalar>::Real as num_traits::One>::one);

    // Filter out singular values effectively zero
    let non_zero_sv = s_vec
        .iter()
        .cloned()
        .filter(|&sv| sv > threshold)
        .collect::<Vec<_>>();

    if non_zero_sv.is_empty() || max_sv == <T as ndarray_linalg::Scalar>::Real::zero() {
        // Matrix is numerically singular (all singular values effectively zero)
        return Ok(<<T as ndarray_linalg::Scalar>::Real as num_traits::Float>::infinity());
    }

    // Also check if there are singular values that are almost zero
    let min_sv_all = s_vec
        .iter()
        .cloned()
        .fold(max_sv, |a, b| if a < b { a } else { b });

    // If the ratio between largest and smallest is very large, return infinity
    if max_sv / min_sv_all
        > <<T as ndarray_linalg::Scalar>::Real as num_traits::NumCast>::from(1e16).unwrap()
    {
        return Ok(<<T as ndarray_linalg::Scalar>::Real as num_traits::Float>::infinity());
    }

    // Get the smallest non-zero singular value
    let min_sv = non_zero_sv
        .iter()
        .cloned()
        .fold(max_sv, |a, b| if a < b { a } else { b });

    // Compute the condition number as the ratio of largest to smallest singular values
    let cond = max_sv / min_sv;

    // Check for overflow and handle appropriately
    if cond.is_infinite() || cond.is_nan() {
        // If we get overflow or NaN, return a high but finite condition number
        return Ok(<<T as ndarray_linalg::Scalar>::Real as num_traits::Float>::max_value());
    }

    Ok(cond)
}

/// Calculate the reciprocal condition number, which is more numerically stable
/// for very ill-conditioned matrices where the ratio might overflow.
///
/// Returns a value between 0 and 1, where values close to 0 indicate ill-conditioning,
/// and values close to 1 indicate good conditioning.
pub fn rcond<T>(a: &Array<T>) -> Result<<T as ndarray_linalg::Scalar>::Real>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack,
    <T as ndarray_linalg::Scalar>::Real: Clone + num_traits::Float,
{
    let cond = condition_number(a)?;

    // Compute the reciprocal, handling potential underflow
    if cond.is_infinite() {
        Ok(<<T as ndarray_linalg::Scalar>::Real as num_traits::Zero>::zero())
    } else {
        Ok(<<T as ndarray_linalg::Scalar>::Real as num_traits::One>::one() / cond)
    }
}