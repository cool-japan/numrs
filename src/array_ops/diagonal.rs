use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
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
/// let diag_mat = diag(&a, Some(0)).unwrap();
/// assert_eq!(diag_mat.shape(), vec![3, 3]);
/// assert_eq!(diag_mat.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
///
/// // Extract the main diagonal from a 2D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let diag_vec = diag(&b, Some(0)).unwrap();
/// assert_eq!(diag_vec.shape(), vec![3]);
/// assert_eq!(diag_vec.to_vec(), vec![1, 5, 9]);
///
/// // Extract a super-diagonal (k=1)
/// let super_diag = diag(&b, Some(1)).unwrap();
/// assert_eq!(super_diag.shape(), vec![2]);
/// assert_eq!(super_diag.to_vec(), vec![2, 6]);
///
/// // Extract a sub-diagonal (k=-1)
/// let sub_diag = diag(&b, Some(-1)).unwrap();
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
/// let diag = diagonal(&a, Some(0), None, None).unwrap();
/// assert_eq!(diag.shape(), vec![3]);
/// assert_eq!(diag.to_vec(), vec![1, 5, 9]);
///
/// // Extract a super-diagonal (offset=1)
/// let super_diag = diagonal(&a, Some(1), None, None).unwrap();
/// assert_eq!(super_diag.shape(), vec![2]);
/// assert_eq!(super_diag.to_vec(), vec![2, 6]);
///
/// // Extract a sub-diagonal (offset=-1)
/// let sub_diag = diagonal(&a, Some(-1), None, None).unwrap();
/// assert_eq!(sub_diag.shape(), vec![2]);
/// assert_eq!(sub_diag.to_vec(), vec![4, 8]);
///
/// // Extract the diagonal from a 3D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[2, 2, 3]);
/// let diag = diagonal(&b, Some(0), Some(1), Some(2)).unwrap();
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

/// Fill the main diagonal of the given array of any dimensionality.
///
/// For an array `a` with `a.ndim >= 2`, the diagonal is the list of
/// locations with indices `a[i, ..., i]` all identical. This function
/// modifies the input array in-place, it does not return a value.
///
/// # Arguments
///
/// * `array` - Array whose diagonal is to be filled, it gets modified in-place
/// * `val` - Value to be written on the diagonal
/// * `wrap` - For tall matrices in NumPy version 1.13, the diagonal "wraps" after N columns.
///   This behavior is deprecated, but you can specify wrap=true to continue using it.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Fill diagonal of a 3x3 matrix
/// let mut a = Array::zeros(&[3, 3]);
/// fill_diagonal(&mut a, 5).unwrap();
/// assert_eq!(a.to_vec(), vec![5, 0, 0, 0, 5, 0, 0, 0, 5]);
///
/// // Fill diagonal of a 4x3 matrix
/// let mut b = Array::zeros(&[4, 3]);
/// fill_diagonal(&mut b, 7).unwrap();
/// assert_eq!(b.to_vec(), vec![7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0]);
///
/// // Fill diagonal of a 3D array
/// let mut c = Array::zeros(&[3, 3, 3]);
/// fill_diagonal(&mut c, 1).unwrap();
/// // Only elements where all indices are equal get filled
/// let expected = vec![
///     1, 0, 0, 0, 0, 0, 0, 0, 0,  // c[0,:,:]
///     0, 0, 0, 0, 1, 0, 0, 0, 0,  // c[1,:,:]
///     0, 0, 0, 0, 0, 0, 0, 0, 1   // c[2,:,:]
/// ];
/// assert_eq!(c.to_vec(), expected);
/// ```
pub fn fill_diagonal<T: Clone>(array: &mut Array<T>, val: T, wrap: bool) -> Result<()> {
    let ndim = array.ndim();

    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "Array must be at least 2D".to_string(),
        ));
    }

    let shape = array.shape();

    // For 2D arrays
    if ndim == 2 {
        let n_rows = shape[0];
        let n_cols = shape[1];
        let diag_len = if wrap {
            n_rows
        } else {
            std::cmp::min(n_rows, n_cols)
        };

        let array_data = array
            .array_mut()
            .as_slice_mut()
            .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

        for i in 0..diag_len {
            let col = if wrap { i % n_cols } else { i };
            if col < n_cols {
                let idx = i * n_cols + col;
                array_data[idx] = val.clone();
            }
        }
    } else {
        // For N-dimensional arrays where N > 2
        // Fill positions where all indices are equal
        let min_dim = shape.iter().min().copied().unwrap_or(0);

        let array_data = array
            .array_mut()
            .as_slice_mut()
            .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

        // Calculate strides
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Fill diagonal elements where all indices are equal
        for i in 0..min_dim {
            let mut idx = 0;
            for &stride in &strides {
                idx += i * stride;
            }
            array_data[idx] = val.clone();
        }
    }

    Ok(())
}
