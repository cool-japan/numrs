//! Tensor operations for Array
//! Includes Kronecker product, tensor dot product, and other tensor operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Compute the Kronecker product of two arrays
///
/// The Kronecker product is a matrix operation that takes two matrices A (m×n) and B (p×q)
/// and produces a matrix of size (mp)×(nq). Each element A[i,j] is multiplied by the entire
/// matrix B and placed at the appropriate block position in the result.
///
/// # Arguments
/// * `a` - First input array (must be 2D)
/// * `b` - Second input array (must be 2D)
///
/// # Returns
/// * `Result<Array<T>>` - The Kronecker product of the two input arrays
///
/// # Errors
/// * `DimensionMismatch` - If either input is not a 2D array
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensor_ops::kron;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
/// let result = kron(&a, &b).unwrap();
/// // Result is a 4×4 matrix
/// ```
pub fn kron<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    // Check that both inputs are 2D arrays
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "kron requires two 2D arrays".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Output shape is (a_rows * b_rows, a_cols * b_cols)
    let out_shape = [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]];
    let mut result = Array::zeros(&out_shape);

    // Extract the data
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data = result.array_mut().as_slice_mut().unwrap();

    // Compute Kronecker product
    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            let a_idx = i * a_shape[1] + j;
            let a_val = a_data[a_idx];

            // For each element in A, multiply by entire B matrix
            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    let b_idx = k * b_shape[1] + l;
                    let b_val = b_data[b_idx];

                    // Position in result array
                    let row = i * b_shape[0] + k;
                    let col = j * b_shape[1] + l;
                    let result_idx = row * out_shape[1] + col;

                    result_data[result_idx] = a_val * b_val;
                }
            }
        }
    }

    Ok(result)
}

/// Compute tensor dot product of two arrays along specified axes
///
/// The tensor dot product contracts specified axes of two tensors. It generalizes
/// matrix multiplication to higher-dimensional arrays by summing over specified axes.
///
/// # Arguments
/// * `a` - First input array
/// * `b` - Second input array  
/// * `axes` - Array of axes to contract (must have exactly 2 elements)
///
/// # Returns
/// * `Result<Array<T>>` - The tensor dot product result
///
/// # Errors
/// * `InvalidOperation` - If axes array doesn't have exactly 2 elements
/// * `DimensionMismatch` - If input arrays are not 2D or axes are out of bounds
/// * `ShapeMismatch` - If the contracted dimensions don't match
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensor_ops::tensordot;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
/// let result = tensordot(&a, &b, &[1, 0]).unwrap(); // Contract axis 1 of a with axis 0 of b
/// ```
pub fn tensordot<T: Float + Clone + Debug>(
    a: &Array<T>,
    b: &Array<T>,
    axes: &[usize],
) -> Result<Array<T>> {
    // Simplified version for 2 axes
    if axes.len() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "This implementation of tensordot only supports 2 axes".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    let a_axis = axes[0];
    let b_axis = axes[1];

    if a_axis >= a_shape.len() || b_axis >= b_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Axis out of bounds".to_string(),
        ));
    }

    // Check that contracted dimensions match
    if a_shape[a_axis] != b_shape[b_axis] {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![a_shape[a_axis]],
            actual: vec![b_shape[b_axis]],
        });
    }

    // For simplicity, this implementation only handles 2D arrays
    // A complete implementation would handle arbitrary dimensions
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "This implementation of tensordot only supports 2D arrays".to_string(),
        ));
    }

    // When contracting along axis 1 of A and axis 0 of B,
    // this becomes a matrix multiplication (if dimensions match)
    if a_axis == 1 && b_axis == 0 {
        return a.matmul(b);
    }

    // If contracting along axis 0 of A and axis 1 of B,
    // transpose B first, then do matrix multiplication
    if a_axis == 0 && b_axis == 1 {
        let b_trans = b.transpose();
        let result = a.transpose().matmul(&b_trans)?;
        return Ok(result.transpose());
    }

    // Handle other cases (more complex tensor contractions)
    Err(NumRs2Error::InvalidOperation(
        "This axis combination is not implemented in this version".to_string(),
    ))
}
