//! Array transformation operations
//!
//! This module provides functions for transforming array shapes and layouts.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use scirs2_core::ndarray::IxDyn;
use std::cmp;

/// Extract a diagonal or construct a diagonal array
///
/// # Parameters
///
/// * `array` - Input array
/// * `k` - Offset of the diagonal from the main diagonal.
///   A positive value means the diagonal is above the main diagonal.
///   A negative value means the diagonal is below the main diagonal.
///   The default is 0 (the main diagonal).
///
/// # Returns
///
/// * If `array` is 1D, returns a 2D array with `array` on the `k`-th diagonal.
/// * If `array` is 2D, returns a 1D array of the diagonal elements along the `k`-th diagonal.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a diagonal matrix from a 1D array
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let diag_mat = diag(&a, Some(0)).expect("diag should succeed for 1D array");
/// assert_eq!(diag_mat.shape(), vec![3, 3]);
/// assert_eq!(diag_mat.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
///
/// // Extract the main diagonal from a 2D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let diag_vec = diag(&b, Some(0)).expect("diag should succeed for 2D array main diagonal");
/// assert_eq!(diag_vec.shape(), vec![3]);
/// assert_eq!(diag_vec.to_vec(), vec![1, 5, 9]);
///
/// // Extract a super-diagonal (k=1)
/// let super_diag = diag(&b, Some(1)).expect("diag should succeed for super-diagonal");
/// assert_eq!(super_diag.shape(), vec![2]);
/// assert_eq!(super_diag.to_vec(), vec![2, 6]);
///
/// // Extract a sub-diagonal (k=-1)
/// let sub_diag = diag(&b, Some(-1)).expect("diag should succeed for sub-diagonal");
/// assert_eq!(sub_diag.shape(), vec![2]);
/// assert_eq!(sub_diag.to_vec(), vec![4, 8]);
/// ```
pub fn diag<T: Clone + Zero>(array: &Array<T>, k: impl Into<Option<isize>>) -> Result<Array<T>> {
    let k = k.into().unwrap_or(0);
    let ndim = array.ndim();

    match ndim {
        1 => {
            // Create a 2D array with the 1D array on the k-th diagonal
            let size = array.size();
            let diag_size = size + k.unsigned_abs();

            // Create a square zero array
            let result = Array::zeros(&[diag_size, diag_size]);
            let mut result_vec = result.to_vec();

            // Place the 1D array on the k-th diagonal
            let array_vec = array.to_vec();

            #[allow(clippy::needless_range_loop)]
            for i in 0..size {
                let row: usize;
                let col: usize;

                if k >= 0 {
                    row = i;
                    col = i + k as usize;
                } else {
                    row = i + (-k) as usize;
                    col = i;
                }

                if row < diag_size && col < diag_size {
                    let idx = row * diag_size + col;
                    result_vec[idx] = array_vec[i].clone();
                }
            }

            Ok(Array::from_vec(result_vec).reshape(&[diag_size, diag_size]))
        }
        2 => {
            // Extract the k-th diagonal from a 2D array
            let shape = array.shape();

            if shape.len() != 2 {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Expected a 2D array, got shape {:?}",
                    shape
                )));
            }

            let rows = shape[0];
            let cols = shape[1];

            // Calculate the length of the resulting diagonal
            let diag_len = if k >= 0 {
                cmp::min(rows, cols.saturating_sub(k as usize))
            } else {
                cmp::min(rows.saturating_sub((-k) as usize), cols)
            };

            if diag_len == 0 {
                return Ok(Array::zeros(&[0]));
            }

            let mut result = Vec::with_capacity(diag_len);
            let array_vec = array.to_vec();

            for i in 0..diag_len {
                let row: usize;
                let col: usize;

                if k >= 0 {
                    row = i;
                    col = i + k as usize;
                } else {
                    row = i + (-k) as usize;
                    col = i;
                }

                if row < rows && col < cols {
                    let idx = row * cols + col;
                    result.push(array_vec[idx].clone());
                }
            }

            Ok(Array::from_vec(result))
        }
        _ => Err(NumRs2Error::InvalidOperation(format!(
            "Input must be 1D or 2D array, got {}D array",
            ndim
        ))),
    }
}

/// Return a specified diagonal of an array
///
/// # Parameters
///
/// * `array` - Input array
/// * `offset` - Offset of the diagonal from the main diagonal.
///   A positive value means the diagonal is above the main diagonal.
///   A negative value means the diagonal is below the main diagonal.
///   The default is 0 (the main diagonal).
/// * `axis1` - First axis of the 2D subarray from which the diagonal should be taken.
///   Default is 0.
/// * `axis2` - Second axis of the 2D subarray from which the diagonal should be taken.
///   Default is 1.
///
/// # Returns
///
/// * A view of the specified diagonal.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Extract the main diagonal from a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let diag = diagonal(&a, Some(0), None, None).expect("diagonal extraction should succeed");
/// assert_eq!(diag.shape(), vec![3]);
/// assert_eq!(diag.to_vec(), vec![1, 5, 9]);
///
/// // Extract a super-diagonal (offset=1)
/// let super_diag = diagonal(&a, Some(1), None, None).expect("super-diagonal extraction should succeed");
/// assert_eq!(super_diag.shape(), vec![2]);
/// assert_eq!(super_diag.to_vec(), vec![2, 6]);
///
/// // Extract a sub-diagonal (offset=-1)
/// let sub_diag = diagonal(&a, Some(-1), None, None).expect("sub-diagonal extraction should succeed");
/// assert_eq!(sub_diag.shape(), vec![2]);
/// assert_eq!(sub_diag.to_vec(), vec![4, 8]);
///
/// // Extract the diagonal from a 3D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[2, 2, 3]);
/// let diag = diagonal(&b, Some(0), Some(1), Some(2)).expect("3D diagonal extraction should succeed");
/// assert_eq!(diag.shape(), vec![2, 2]);
/// assert_eq!(diag.to_vec(), vec![1, 5, 7, 11]);
/// ```
pub fn diagonal<T: Clone + num_traits::Zero>(
    array: &Array<T>,
    offset: impl Into<Option<isize>>,
    axis1: impl Into<Option<usize>>,
    axis2: impl Into<Option<usize>>,
) -> Result<Array<T>> {
    let offset = offset.into().unwrap_or(0);
    let axis1 = axis1.into().unwrap_or(0);
    let axis2 = axis2.into().unwrap_or(1);

    let ndim = array.ndim();

    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Array must be at least 2D, got {}D array",
            ndim
        )));
    }

    if axis1 == axis2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "axis1 and axis2 cannot be the same: {}",
            axis1
        )));
    }

    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes ({}, {}) out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
    }

    // Get the lengths of the two axes
    let shape = array.shape();
    let axis1_len = shape[axis1];
    let axis2_len = shape[axis2];

    // Calculate the length of the resulting diagonal
    let diag_len = if offset >= 0 {
        cmp::min(axis1_len, axis2_len.saturating_sub(offset as usize))
    } else {
        cmp::min(axis1_len.saturating_sub((-offset) as usize), axis2_len)
    };

    if diag_len == 0 {
        // Create a result array with the same shape as the input array,
        // but with the two specified axes replaced by a single dimension of length 0
        let mut result_shape = Vec::with_capacity(ndim - 1);
        for (i, &dim) in shape.iter().enumerate() {
            if i != axis1 && i != axis2 {
                result_shape.push(dim);
            }
        }
        result_shape.push(0);

        return Ok(Array::zeros(&result_shape));
    }

    // Prepare the result shape
    let mut result_shape = Vec::with_capacity(ndim - 1);
    for (i, &dim) in shape.iter().enumerate() {
        if i != axis1 && i != axis2 {
            result_shape.push(dim);
        }
    }
    result_shape.push(diag_len);

    // Calculate the total size of the result array
    let result_size: usize = result_shape.iter().product();

    // Create the result array
    let mut result_vec = Vec::with_capacity(result_size);

    // Extract the diagonal values
    let array_vec = array.to_vec();

    // Calculate the strides for each dimension
    let mut strides = Vec::with_capacity(ndim);
    let mut stride = 1;
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();

    // Extract the diagonal elements
    let axis1_stride = strides[axis1];
    let axis2_stride = strides[axis2];

    // Helper function to calculate index without axis1 and axis2
    let calc_base_index = |indices: &[usize]| -> usize {
        let mut base_idx = 0;
        let mut _dst_idx = 0;

        for (src_idx, &dim) in indices.iter().enumerate() {
            if src_idx != axis1 && src_idx != axis2 {
                base_idx += dim * strides[src_idx];
                _dst_idx += 1;
            }
        }

        base_idx
    };

    // Pre-allocate indices array to avoid reallocating in each iteration
    let mut indices = vec![0; ndim];

    // Helper function to increment indices
    let increment_indices = |indices: &mut [usize], shape: &[usize], axis1, axis2| {
        for i in (0..indices.len()).rev() {
            if i != axis1 && i != axis2 {
                indices[i] += 1;
                if indices[i] < shape[i] {
                    return true;
                }
                indices[i] = 0;
            }
        }
        false
    };

    // Number of elements to process (excluding the diagonal axes)
    let mut outer_elements = 1;
    for (i, &dim) in shape.iter().enumerate() {
        if i != axis1 && i != axis2 {
            outer_elements *= dim;
        }
    }

    // Process each combination of indices (except for axis1 and axis2)
    for _ in 0..outer_elements {
        let base_idx = calc_base_index(&indices);

        // Extract the diagonal at this position
        for i in 0..diag_len {
            let row: usize;
            let col: usize;

            if offset >= 0 {
                row = i;
                col = i + offset as usize;
            } else {
                row = i + (-offset) as usize;
                col = i;
            }

            if row < axis1_len && col < axis2_len {
                let idx = base_idx + row * axis1_stride + col * axis2_stride;
                result_vec.push(array_vec[idx].clone());
            }
        }

        // Increment the indices (except for axis1 and axis2)
        increment_indices(&mut indices, &shape, axis1, axis2);
    }

    Ok(Array::from_vec(result_vec).reshape(&result_shape))
}

/// Roll the specified axis to a new position
///
/// # Parameters
///
/// * `array` - The input array
/// * `axis` - The axis to roll
/// * `start` - The new position (destination)
///
/// # Returns
///
/// A new array with the axes rearranged
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
///     .reshape(&[2, 2, 3]);
///
/// // Roll axis 2 to position 0
/// let b = rollaxis(&a, 2, 0).expect("rollaxis should succeed for valid axes");
/// assert_eq!(b.shape(), vec![3, 2, 2]);
/// ```
pub fn rollaxis<T: Clone + Zero>(array: &Array<T>, axis: usize, start: usize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if axis >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis, ndim
        )));
    }

    if start > ndim {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Start position {} exceeds array dimensions {}",
            start, ndim
        )));
    }

    if axis == start || (axis == ndim - 1 && start == ndim) {
        // No change needed
        return Ok(array.clone());
    }

    // Create a new axis order
    let mut axes: Vec<usize> = (0..ndim).collect();

    // Remove the rolled axis
    let rolled_axis = axes.remove(axis);

    // Insert it at the new position
    axes.insert(if start <= axis { start } else { start - 1 }, rolled_axis);

    // Create a new array with the axes in the new order
    // Using a simplified transposes approach for now
    // This is a simplified implementation that's not efficient for large arrays
    // For a proper implementation, we would use ndarray's permute_axes functionality
    let source_shape = array.shape().to_vec();
    let mut target_shape = Vec::with_capacity(ndim);

    for &ax in &axes {
        target_shape.push(source_shape[ax]);
    }

    let mut result_data = vec![T::zero(); array.size()];

    // Iterate through all elements of the array
    let source_size = array.size();
    let source_array = array.array();

    for i in 0..source_size {
        // Convert flat index to multi-dimensional indices
        let mut source_indices = vec![0; ndim];
        let mut remainder = i;
        for j in (0..ndim).rev() {
            source_indices[j] = remainder % source_shape[j];
            remainder /= source_shape[j];
        }

        // Map indices to the new order
        let mut target_indices = vec![0; ndim];
        for (j, &ax) in axes.iter().enumerate() {
            target_indices[j] = source_indices[ax];
        }

        // Calculate flat index in the target array
        let mut target_flat_index = 0;
        let mut multiplier = 1;
        for j in (0..ndim).rev() {
            target_flat_index += target_indices[j] * multiplier;
            multiplier *= target_shape[j];
        }

        // Copy the value
        if let Some(slice) = source_array.as_slice() {
            result_data[target_flat_index] = slice[i].clone();
        } else {
            return Err(NumRs2Error::InvalidOperation(
                "Failed to get array slice".into(),
            ));
        }
    }

    // Create the result array
    let result = Array::from_vec(result_data).reshape(&target_shape);
    Ok(result)
}

/// Helper function to transpose two specific axes
#[allow(dead_code)]
fn array_transpose<T: Clone + Zero>(
    array: &Array<T>,
    axis1: usize,
    axis2: usize,
) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes ({}, {}) out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
    }

    if axis1 == axis2 {
        return Ok(array.clone());
    }

    // Create a new array to hold the transposed data
    let mut transposed_shape = shape.clone();
    transposed_shape.swap(axis1, axis2);

    let mut result = Array::zeros(&transposed_shape);

    // Total number of elements to process
    let total_size = array.size();

    // Process each position in the array
    for i in 0..total_size {
        // Calculate the indices for the current element
        let mut indices = Vec::with_capacity(ndim);
        let mut temp = i;

        for j in (0..ndim).rev() {
            indices.insert(0, temp % shape[j]);
            temp /= shape[j];
        }

        // Create transposed indices
        let mut trans_indices = indices.clone();
        trans_indices.swap(axis1, axis2);

        // Copy the element
        if let Some(value) = array.array().get(IxDyn(&indices)) {
            if result.set(&trans_indices, value.clone()).is_err() {
                return Err(NumRs2Error::InvalidOperation(
                    "Failed to set transposed value".into(),
                ));
            }
        }
    }

    Ok(result)
}

/// Interchange two axes of an array.
///
/// # Parameters
///
/// * `array` - The array to transform
/// * `axis1` - The first axis to swap
/// * `axis2` - The second axis to swap
///
/// # Returns
///
/// A view of the array with the axes swapped
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let b = swapaxes(&a, 0, 1).expect("swapaxes should succeed for valid axes");
/// assert_eq!(b.shape(), vec![3, 2]);
/// ```
pub fn swapaxes<T: Clone>(array: &Array<T>, axis1: usize, axis2: usize) -> Result<Array<T>> {
    let ndim = array.ndim();

    // Check if the axes are valid
    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes {} and {} are out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
    }

    // If axes are the same, return a view of the original array
    if axis1 == axis2 {
        return Ok(array.clone());
    }

    // Create the new shape and permutation array
    let mut permutation = Vec::with_capacity(ndim);
    for i in 0..ndim {
        if i == axis1 {
            permutation.push(axis2);
        } else if i == axis2 {
            permutation.push(axis1);
        } else {
            permutation.push(i);
        }
    }

    // Transpose according to the permutation
    let mut result = array.clone();

    // Permute the axes
    for i in 0..ndim {
        if permutation[i] != i {
            // Find where the i-th axis should go
            let j = permutation[i];

            // Swap axes i and j in the result
            result = result.transpose_axis(i, j);

            // Update the permutation to reflect the swap
            permutation.swap(i, j);
        }
    }

    Ok(result)
}

/// Move the axes of an array to new positions.
///
/// # Parameters
///
/// * `array` - The array to transform
/// * `source` - The original positions of the axes to move
/// * `destination` - The destination positions of the axes to move
///
/// # Returns
///
/// A view of the array with axes moved
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8]).reshape(&[2, 2, 2]);
/// let b = moveaxis(&a, &[0], &[2]).expect("moveaxis should succeed for valid axes");
/// assert_eq!(b.shape(), vec![2, 2, 2]);
/// ```
pub fn moveaxis<T: Clone>(
    array: &Array<T>,
    source: &[usize],
    destination: &[usize],
) -> Result<Array<T>> {
    let ndim = array.ndim();

    // Check if the source and destination arrays have the same length
    if source.len() != destination.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Source and destination arrays must have the same length, got {} and {}",
            source.len(),
            destination.len()
        )));
    }

    // Check if the axes are valid
    for &axis in source.iter().chain(destination.iter()) {
        if axis >= ndim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} is out of bounds for array of dimension {}",
                axis, ndim
            )));
        }
    }

    // Create an array to track the new positions of the axes
    let mut perm = Vec::with_capacity(ndim);
    for i in 0..ndim {
        perm.push(i);
    }

    // Move the axes to their destination positions
    for (&src, &dst) in source.iter().zip(destination.iter()) {
        // Remove the source axis
        let src_axis = perm.remove(src);

        // Insert it at the destination position
        if dst < perm.len() {
            perm.insert(dst, src_axis);
        } else {
            perm.push(src_axis);
        }
    }

    // Permute the axes according to perm
    let mut result = array.clone();

    // Apply the permutation
    for i in 0..ndim {
        if perm[i] != i {
            // Find where the i-th axis should go
            if let Some(j) = perm.iter().position(|&p| p == i) {
                // Swap axes i and j in the result
                result = result.transpose_axis(i, j);

                // Update the permutation to reflect the swap
                perm.swap(i, j);
            }
        }
    }

    Ok(result)
}

/// Ensure the input is at least 1-D.
///
/// # Parameters
///
/// * `arys` - One or more arrays to convert
///
/// # Returns
///
/// A tuple of arrays that are at least 1-D
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = atleast_1d(&[&a]).expect("atleast_1d should succeed for valid array");
/// assert_eq!(b[0].shape(), vec![3]);
/// ```
pub fn atleast_1d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    let mut result = Vec::with_capacity(arys.len());

    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 1-D
            if let Ok(scalar_value) = array.get(&[]) {
                result.push(Array::from_vec(vec![scalar_value]).reshape(&[1]));
            } else {
                return Err(NumRs2Error::InvalidOperation(
                    "Failed to get scalar value".into(),
                ));
            }
        } else {
            // Already at least 1-D, add a view of the array
            result.push(array.clone());
        }
    }

    Ok(result)
}

/// Ensure the input is at least 2-D.
///
/// # Parameters
///
/// * `arys` - One or more arrays to convert
///
/// # Returns
///
/// A tuple of arrays that are at least 2-D
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = atleast_2d(&[&a]).expect("atleast_2d should succeed for valid array");
/// assert_eq!(b[0].shape(), vec![1, 3]);
/// ```
pub fn atleast_2d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    let mut result = Vec::with_capacity(arys.len());

    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 2-D
            if let Ok(scalar_value) = array.get(&[]) {
                result.push(Array::from_vec(vec![scalar_value]).reshape(&[1, 1]));
            } else {
                return Err(NumRs2Error::InvalidOperation(
                    "Failed to get scalar value".into(),
                ));
            }
        } else if array.ndim() == 1 {
            // 1-D, reshape to 2-D
            let data = array.to_vec();
            let new_shape = vec![1, data.len()];
            result.push(Array::from_vec(data).reshape(&new_shape));
        } else {
            // Already at least 2-D, add a view of the array
            result.push(array.clone());
        }
    }

    Ok(result)
}

/// Ensure the input is at least 3-D.
///
/// # Parameters
///
/// * `arys` - One or more arrays to convert
///
/// # Returns
///
/// A tuple of arrays that are at least 3-D
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = atleast_3d(&[&a]).expect("atleast_3d should succeed for valid array");
/// assert_eq!(b[0].shape(), vec![1, 3, 1]);
/// ```
pub fn atleast_3d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    let mut result = Vec::with_capacity(arys.len());

    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 3-D
            if let Ok(scalar_value) = array.get(&[]) {
                result.push(Array::from_vec(vec![scalar_value]).reshape(&[1, 1, 1]));
            } else {
                return Err(NumRs2Error::InvalidOperation(
                    "Failed to get scalar value".into(),
                ));
            }
        } else if array.ndim() == 1 {
            // 1-D, reshape to 3-D
            let data = array.to_vec();
            let new_shape = vec![1, data.len(), 1];
            result.push(Array::from_vec(data).reshape(&new_shape));
        } else if array.ndim() == 2 {
            // 2-D, reshape to 3-D
            let data = array.to_vec();
            let shape = array.shape();
            let new_shape = vec![shape[0], shape[1], 1];
            result.push(Array::from_vec(data).reshape(&new_shape));
        } else {
            // Already at least 3-D, add a view of the array
            result.push(array.clone());
        }
    }

    Ok(result)
}

/// Broadcast any number of arrays to a common shape
///
/// # Parameters
///
/// * `arrays` - A slice of arrays to broadcast
///
/// # Returns
///
/// A vector of arrays all broadcast to the same shape
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
/// let b = Array::from_vec(vec![10, 20, 30]).reshape(&[3, 1]);
/// let broadcasts = broadcast_arrays(&[&a, &b]).expect("broadcast_arrays should succeed for compatible shapes");
/// assert_eq!(broadcasts[0].shape(), vec![3, 3]);
/// assert_eq!(broadcasts[1].shape(), vec![3, 3]);
/// ```
pub fn broadcast_arrays<T: Clone>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    // Calculate the broadcast shape
    let mut broadcast_shape = arrays[0].shape();

    for arr in arrays.iter().skip(1) {
        broadcast_shape = Array::<T>::broadcast_shape(&broadcast_shape, &arr.shape())?;
    }

    // Broadcast each array to the common shape
    let mut result = Vec::with_capacity(arrays.len());

    for &arr in arrays {
        let broadcasted = arr.broadcast_to(&broadcast_shape)?;
        result.push(broadcasted);
    }

    Ok(result)
}

/// Broadcast an array to the given shape without copying the data
///
/// # Parameters
///
/// * `array` - The array to broadcast
/// * `shape` - The target shape
///
/// # Returns
///
/// A new array with the specified shape sharing the underlying data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 1x3 array (already 2D)
/// let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
///
/// // Broadcast to 2x3
/// let b = broadcast_to(&a, &[2, 3]).expect("broadcast_to should succeed for compatible shape");
/// assert_eq!(b.shape(), vec![2, 3]);
/// ```
pub fn broadcast_to<T: Clone>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>> {
    array.broadcast_to(shape)
}
