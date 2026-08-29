//! Triangle and append operations
//!
//! This module provides functions for:
//! - Extracting lower and upper triangular matrices
//! - Appending values to arrays

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

/// Lower triangle of an array.
///
/// Return a copy of an array with elements above the k-th diagonal zeroed.
///
/// # Arguments
///
/// * `array` - Input array
/// * `k` - Diagonal above which to zero elements. k = 0 (the default) is the main diagonal,
///   k < 0 is below it and k > 0 is above.
///
/// # Returns
///
/// Lower triangle of array, of same shape and data-type as input array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let lower = tril(&a, 0).expect("operation should succeed");
/// assert_eq!(lower.to_vec(), vec![1, 0, 0, 4, 5, 0, 7, 8, 9]);
///
/// // With k=1 (include first diagonal above main)
/// let lower = tril(&a, 1).expect("operation should succeed");
/// assert_eq!(lower.to_vec(), vec![1, 2, 0, 4, 5, 6, 7, 8, 9]);
///
/// // With k=-1 (exclude main diagonal)
/// let lower = tril(&a, -1).expect("operation should succeed");
/// assert_eq!(lower.to_vec(), vec![0, 0, 0, 4, 0, 0, 7, 8, 0]);
/// ```
pub fn tril<T: Clone + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 2 {
        // For 1D arrays, return a copy
        return Ok(array.clone());
    }

    let n_rows = shape[ndim - 2];
    let n_cols = shape[ndim - 1];

    // Create a copy of the array
    let mut result = array.clone();
    let result_data = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    // Calculate the number of 2D matrices in the array
    let n_matrices: usize = shape[..ndim - 2].iter().product();
    let matrix_size = n_rows * n_cols;

    // Zero out elements above the k-th diagonal for each matrix
    for m in 0..n_matrices {
        let matrix_offset = m * matrix_size;

        for i in 0..n_rows {
            for j in 0..n_cols {
                // Check if element is above the k-th diagonal
                if (j as isize) > (i as isize + k) {
                    let idx = matrix_offset + i * n_cols + j;
                    result_data[idx] = T::zero();
                }
            }
        }
    }

    Ok(result)
}

/// Upper triangle of an array.
///
/// Return a copy of an array with elements below the k-th diagonal zeroed.
///
/// # Arguments
///
/// * `array` - Input array
/// * `k` - Diagonal below which to zero elements. k = 0 (the default) is the main diagonal,
///   k < 0 is below it and k > 0 is above.
///
/// # Returns
///
/// Upper triangle of array, of same shape and data-type as input array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let upper = triu(&a, 0).expect("operation should succeed");
/// assert_eq!(upper.to_vec(), vec![1, 2, 3, 0, 5, 6, 0, 0, 9]);
///
/// // With k=1 (exclude main diagonal)
/// let upper = triu(&a, 1).expect("operation should succeed");
/// assert_eq!(upper.to_vec(), vec![0, 2, 3, 0, 0, 6, 0, 0, 0]);
///
/// // With k=-1 (include first diagonal below main)
/// let upper = triu(&a, -1).expect("operation should succeed");
/// assert_eq!(upper.to_vec(), vec![1, 2, 3, 4, 5, 6, 0, 8, 9]);
/// ```
pub fn triu<T: Clone + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 2 {
        // For 1D arrays, return a copy
        return Ok(array.clone());
    }

    let n_rows = shape[ndim - 2];
    let n_cols = shape[ndim - 1];

    // Create a copy of the array
    let mut result = array.clone();
    let result_data = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    // Calculate the number of 2D matrices in the array
    let n_matrices: usize = shape[..ndim - 2].iter().product();
    let matrix_size = n_rows * n_cols;

    // Zero out elements below the k-th diagonal for each matrix
    for m in 0..n_matrices {
        let matrix_offset = m * matrix_size;

        for i in 0..n_rows {
            for j in 0..n_cols {
                // Check if element is below the k-th diagonal
                if (j as isize) < (i as isize + k) {
                    let idx = matrix_offset + i * n_cols + j;
                    result_data[idx] = T::zero();
                }
            }
        }
    }

    Ok(result)
}

/// Append values to the end of an array
///
/// # Arguments
/// * `array` - Input array
/// * `values` - Values to append to the array
/// * `axis` - The axis along which values are appended. If None, both arrays are flattened before appending.
///
/// # Returns
/// * `Result<Array<T>>` - A copy of array with values appended to axis
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::manipulation::append;
///
/// // Append to 1D array
/// let arr = Array::from_vec(vec![1, 2, 3]);
/// let result = append(&arr, &[4, 5], None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
///
/// // Append to 2D array along axis 0
/// let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let values = Array::from_vec(vec![5, 6]).reshape(&[1, 2]);
/// let result = append(&arr, &values.to_vec(), Some(0)).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![3, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // Append to 2D array along axis 1
/// let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let values = vec![5, 6];
/// let result = append(&arr, &values, Some(1)).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![2, 3]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 3, 4, 6]);
/// ```
pub fn append<T: Clone + Zero>(
    array: &Array<T>,
    values: &[T],
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Flatten both arrays and concatenate
            let mut result_data = array.to_vec();
            result_data.extend_from_slice(values);
            Ok(Array::from_vec(result_data))
        }
        Some(ax) => {
            let shape = array.shape();

            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            // For appending along an axis, values must be reshaped to match
            // the shape of array except along the concatenation axis
            let axis_size = shape[ax];
            let mut values_shape = shape.clone();

            // Calculate the expected size for values
            let expected_values_size: usize = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .product();

            // Check if values can be reshaped appropriately
            if !values.len().is_multiple_of(expected_values_size) {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![expected_values_size],
                    actual: vec![values.len()],
                });
            }

            // Calculate new size along the axis
            let values_axis_size = values.len() / expected_values_size;
            values_shape[ax] = values_axis_size;

            // Create values array with proper shape
            let values_array = Array::from_vec_shape(values.to_vec(), &values_shape)?;

            // Update result shape
            let mut result_shape = shape.clone();
            result_shape[ax] = axis_size + values_axis_size;

            // Calculate sizes for iteration
            let pre_axis_size: usize = shape[..ax].iter().product();
            let post_axis_size: usize = shape[ax + 1..].iter().product();
            let total_size = pre_axis_size * result_shape[ax] * post_axis_size;

            // Allocate result array
            let mut result_data = Vec::with_capacity(total_size);

            // Copy data
            let array_data = array.to_vec();
            let values_data = values_array.to_vec();

            for pre in 0..pre_axis_size {
                // Copy from original array
                for i in 0..axis_size {
                    for post in 0..post_axis_size {
                        let idx = pre * axis_size * post_axis_size + i * post_axis_size + post;
                        result_data.push(array_data[idx].clone());
                    }
                }

                // Copy from values array
                for i in 0..values_axis_size {
                    for post in 0..post_axis_size {
                        let idx =
                            pre * values_axis_size * post_axis_size + i * post_axis_size + post;
                        result_data.push(values_data[idx].clone());
                    }
                }
            }

            Ok(Array::from_vec_shape(result_data, &result_shape)?)
        }
    }
}
