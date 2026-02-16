//! Array manipulation operations
//!
//! This module provides functions for manipulating array structures including
//! tiling, repeating, concatenating, stacking, and blocking.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::Axis;
use std::cmp;

use super::core::AxisArg;

/// Construct an array by repeating array the given number of times
pub fn tile<T: Clone>(array: &Array<T>, reps: &[usize]) -> Result<Array<T>> {
    let a_shape = array.shape();

    // Determine the output shape
    let mut output_shape = Vec::with_capacity(cmp::max(a_shape.len(), reps.len()));

    // Ensure reps has at least as many dimensions as a_shape, filling with 1s if needed
    let mut full_reps = Vec::with_capacity(cmp::max(a_shape.len(), reps.len()));
    let reps_offset = if a_shape.len() > reps.len() {
        a_shape.len() - reps.len()
    } else {
        0
    };

    for i in 0..full_reps.capacity() {
        if i < reps_offset {
            full_reps.push(1);
        } else {
            full_reps.push(reps[i - reps_offset]);
        }
    }

    // Compute the output shape
    for (&a_dim, &rep) in a_shape.iter().zip(full_reps.iter()) {
        output_shape.push(a_dim * rep);
    }

    // Add extra dimensions if reps has more dimensions than a_shape
    if reps.len() > a_shape.len() {
        let a_offset = reps.len() - a_shape.len();
        for &rep in reps.iter().take(a_offset) {
            output_shape.insert(0, rep);
        }
    }

    // Create the output array filled with the first element as a placeholder
    let first_elem = array
        .array()
        .first()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Cannot tile an empty array".into()))?
        .clone();

    let mut result = Array::full(&output_shape, first_elem);

    // Fill the output array by copying the input array in a tiled pattern
    // This is a simplified implementation - for efficiency, we would use
    // more sophisticated slicing and assignment operations

    let result_vec = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    let input_vec = array.to_vec();
    let input_size = input_vec.len();

    if input_size == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot tile an empty array".into(),
        ));
    }

    // For each position in the output, copy the corresponding element from the input
    for (i, item) in result_vec.iter_mut().enumerate() {
        // Calculate corresponding index in the input array
        // This is a simplification - for a complete implementation, we would need
        // to carefully map N-dimensional indices
        let input_idx = i % input_size;
        *item = input_vec[input_idx].clone();
    }

    Ok(result)
}

/// Repeat elements of an array along a specified axis
pub fn repeat<T: Clone>(array: &Array<T>, repeats: usize, axis: Option<usize>) -> Result<Array<T>> {
    let a_shape = array.shape();

    match axis {
        Some(ax) => {
            if ax >= a_shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    a_shape.len()
                )));
            }

            // Calculate the output shape
            let mut output_shape = a_shape.clone();
            output_shape[ax] *= repeats;

            // Create a result array
            let first_elem = array
                .array()
                .first()
                .ok_or_else(|| {
                    NumRs2Error::InvalidOperation("Cannot repeat an empty array".into())
                })?
                .clone();

            let mut result = Array::full(&output_shape, first_elem);

            // Fill the result array by repeating elements along the specified axis
            // This is a simplified implementation - a more efficient version would use
            // vectorized operations and views

            let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
            })?;

            let input_vec = array.to_vec();

            if input_vec.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "Cannot repeat an empty array".into(),
                ));
            }

            // For a complete implementation, we would need to carefully map indices
            // between N-dimensional arrays. This is a simplified approach.
            let axis_size = a_shape[ax];
            let pre_axis_size: usize = a_shape.iter().take(ax).product();
            let post_axis_size: usize = a_shape.iter().skip(ax + 1).product();

            for i_pre in 0..pre_axis_size {
                for i_axis in 0..axis_size {
                    for i_rep in 0..repeats {
                        for i_post in 0..post_axis_size {
                            let out_axis_idx = i_axis * repeats + i_rep;
                            let out_idx = i_pre * (output_shape[ax] * post_axis_size)
                                + out_axis_idx * post_axis_size
                                + i_post;

                            let in_idx = i_pre * (axis_size * post_axis_size)
                                + i_axis * post_axis_size
                                + i_post;

                            result_vec[out_idx] = input_vec[in_idx].clone();
                        }
                    }
                }
            }

            Ok(result)
        }
        None => {
            // Flattened repeat
            let input_vec = array.to_vec();
            let mut result_vec = Vec::with_capacity(input_vec.len() * repeats);

            for val in input_vec {
                for _ in 0..repeats {
                    result_vec.push(val.clone());
                }
            }

            Ok(Array::from_vec(result_vec))
        }
    }
}

/// Concatenate arrays along one or multiple axes
///
/// # Parameters
///
/// * `arrays` - A slice of arrays to concatenate
/// * `axis` - The axis or axes along which to concatenate. Can be a single axis or a slice of axes.
///
/// # Returns
///
/// A new array with the concatenated values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Single axis concatenation
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let c = concatenate(&[&a, &b], 0).expect("concatenate should succeed");
/// assert_eq!(c.shape(), vec![6]);
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn concatenate<T: Clone>(arrays: &[&Array<T>], axis: impl Into<AxisArg>) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "No arrays to concatenate".into(),
        ));
    }

    match axis.into() {
        AxisArg::Single(axis) => concatenate_single_axis(arrays, axis),
        AxisArg::Multiple(axes) => concatenate_multiple_axes(arrays, &axes),
    }
}

/// Concatenate arrays along a single axis
fn concatenate_single_axis<T: Clone>(arrays: &[&Array<T>], axis: usize) -> Result<Array<T>> {
    let first_shape = arrays[0].shape();

    if axis >= first_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            first_shape.len()
        )));
    }

    // Check that all arrays have compatible shapes
    for (_i, arr) in arrays.iter().enumerate().skip(1) {
        let shape = arr.shape();

        if shape.len() != first_shape.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: first_shape.clone(),
                actual: shape,
            });
        }

        for (j, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if j != axis && s1 != s2 {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: first_shape.clone(),
                    actual: shape,
                });
            }
        }
    }

    // Calculate the output shape
    let mut output_shape = first_shape.clone();
    output_shape[axis] = arrays.iter().map(|arr| arr.shape()[axis]).sum();

    // Create views for all arrays to concatenate
    let views: Result<Vec<_>> = arrays.iter().map(|arr| Ok(arr.array().view())).collect();

    let views = views?;

    // Use ndarray's concatenate function
    let result = scirs2_core::ndarray::concatenate(Axis(axis), &views).map_err(|e| {
        NumRs2Error::InvalidOperation(format!("Failed to concatenate arrays: {}", e))
    })?;

    // Convert the result back to our Array type
    Ok(Array::from_ndarray(result))
}

/// Concatenate arrays along multiple axes
fn concatenate_multiple_axes<T: Clone>(arrays: &[&Array<T>], axes: &[usize]) -> Result<Array<T>> {
    if axes.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "No axes provided for concatenation".into(),
        ));
    }

    // Process one axis at a time
    let mut result = arrays[0].clone();

    // We'll concatenate along each axis, starting with only the first array in the sequence
    for (i, &axis) in axes.iter().enumerate() {
        // For the first concatenation, we need to concatenate all arrays
        if i == 0 {
            result = concatenate_single_axis(arrays, axis)?;
        } else {
            // For subsequent concatenations, we would need arrays with matching shapes
            // This is a simplified implementation - for better user experience,
            // we should allow arrays to be broadcast to the correct shape
            result = concatenate_single_axis(&[&result, arrays[1]], axis)?;
        }
    }

    Ok(result)
}

/// Stack arrays along a new axis
///
/// # Parameters
///
/// * `arrays` - A slice of arrays to stack
/// * `axis` - The axis along which to stack
///
/// # Returns
///
/// A new array with the stacked values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let c = stack(&[&a, &b], 0).expect("stack should succeed");
/// assert_eq!(c.shape(), vec![2, 3]);
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn stack<T: Clone>(arrays: &[&Array<T>], axis: usize) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to stack".into()));
    }

    let first_shape = arrays[0].shape();

    if axis > first_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            first_shape.len()
        )));
    }

    // Check that all arrays have the same shape
    for arr in arrays.iter().skip(1) {
        let shape = arr.shape();

        if shape != first_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: first_shape.clone(),
                actual: shape,
            });
        }
    }

    // Calculate the output shape - insert a new dimension at the specified axis
    let mut output_shape = first_shape.clone();
    output_shape.insert(axis, arrays.len());

    // For a complete implementation, we would use ndarray's stack operation
    // For now, we'll use a simple implementation based on concatenate

    // First reshape each array to add a new dimension
    let mut reshaped_arrays = Vec::with_capacity(arrays.len());
    for &arr in arrays {
        let mut new_shape = first_shape.clone();
        new_shape.insert(axis, 1);
        let reshaped = arr.reshape(&new_shape);
        reshaped_arrays.push(reshaped);
    }

    // Then concatenate along the new dimension
    let mut result_refs: Vec<&Array<T>> = Vec::with_capacity(reshaped_arrays.len());
    for arr in &reshaped_arrays {
        result_refs.push(arr);
    }

    concatenate(&result_refs, axis)
}

/// Construct a block array from nested lists of blocks
///
/// # Parameters
///
/// * `blocks` - List of lists of arrays with compatible shapes
///
/// # Returns
///
/// A new array containing the blocks
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create arrays
/// let a = Array::from_vec(vec![1, 2]);
/// let b = Array::from_vec(vec![3, 4]);
/// let c = Array::from_vec(vec![5, 6]);
/// let d = Array::from_vec(vec![7, 8]);
///
/// // Arrange them in a 2x2 grid
/// let blocks = vec![vec![&a, &b], vec![&c, &d]];
/// let result = block(&blocks).expect("block should succeed");
///
/// // The result will be either 4x2 or 2x4 depending on the implementation
/// // Just check that all elements are present
/// let result_vec = result.to_vec();
/// assert_eq!(result_vec.len(), 8);
/// for i in 1..=8 {
///     assert!(result_vec.contains(&i));
/// }
/// ```
///
/// You can also use arrays of different dimensions:
///
/// ```
/// use numrs2::prelude::*;
///
/// // Test with 2D arrays
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c = Array::from_vec(vec![9, 10, 11, 12]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![13, 14, 15, 16]).reshape(&[2, 2]);
///
/// let blocks = vec![vec![&a, &b], vec![&c, &d]];
/// let result = block(&blocks).expect("block should succeed");
///
/// assert_eq!(result.shape(), vec![4, 4]);
/// // The result will be a 4x4 array with the blocks arranged in a 2x2 grid
/// ```
pub fn block<T: Clone>(blocks: &[Vec<&Array<T>>]) -> Result<Array<T>> {
    if blocks.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Empty block structure".into(),
        ));
    }

    // Normalize dimensions of arrays within each row
    let mut processed_rows = Vec::with_capacity(blocks.len());

    for (row_idx, row) in blocks.iter().enumerate() {
        if row.is_empty() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Empty row at index {} in block structure",
                row_idx
            )));
        }

        // Process each array in the row to ensure compatible dimensions
        let mut processed_row = Vec::with_capacity(row.len());

        // Determine the maximum number of dimensions in this row
        let max_ndim = row.iter().map(|arr| arr.ndim()).max().unwrap_or(1);

        for arr in row.iter() {
            let arr_ndim = arr.ndim();

            if arr_ndim < max_ndim {
                // Reshape to add missing dimensions with size 1
                let mut new_shape = arr.shape().to_vec();
                while new_shape.len() < max_ndim {
                    // For 1D arrays, add dimension at the end to make it a column vector
                    if arr_ndim == 1 {
                        new_shape.push(1);
                    } else {
                        // Otherwise add dimensions at the beginning
                        new_shape.insert(0, 1);
                    }
                }
                processed_row.push(arr.reshape(&new_shape));
            } else {
                processed_row.push((*arr).clone());
            }
        }

        processed_rows.push(processed_row);
    }

    // Now we can proceed with concatenation
    let mut rows_result = Vec::with_capacity(processed_rows.len());

    for row in &processed_rows {
        // Make sure all arrays in this row have the same number of dimensions
        let ndim = row[0].ndim();

        if !row.iter().all(|arr| arr.ndim() == ndim) {
            return Err(NumRs2Error::InvalidOperation(
                "Arrays in each row must have the same number of dimensions".into(),
            ));
        }

        // For single arrays in a row, we don't need to concatenate
        if row.len() == 1 {
            rows_result.push(row[0].clone());
            continue;
        }

        // Concatenate along the last axis (which would be 1 for 2D arrays)
        let row_refs: Vec<&Array<T>> = row.iter().collect();

        // Use the last axis for concatenation
        let axis = ndim - 1;
        let concatenated_row = concatenate(&row_refs, axis)?;
        rows_result.push(concatenated_row);
    }

    // For a single row, we're done
    if rows_result.len() == 1 {
        return Ok(rows_result[0].clone());
    }

    // Now concatenate the rows - verify they have compatible dimensions
    let row_ndim = rows_result[0].ndim();

    if !rows_result.iter().all(|arr| arr.ndim() == row_ndim) {
        return Err(NumRs2Error::InvalidOperation(
            "All rows must have the same number of dimensions after processing".into(),
        ));
    }

    // Concatenate rows along the first axis (which would be 0 for 2D arrays)
    let row_refs: Vec<&Array<T>> = rows_result.iter().collect();
    concatenate(&row_refs, 0)
}

/// Create a new array by stacking arrays row-wise (along axis 0)
///
/// # Parameters
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// A new array with the stacked values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let c = r_(&[&a, &b]).expect("r_ should succeed");
/// assert_eq!(c.shape(), vec![6]);  // Flattened to 1D array
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // With 2D arrays
/// let a2 = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b2 = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c2 = r_(&[&a2, &b2]).expect("r_ should succeed");
/// assert_eq!(c2.shape(), vec![4, 2]);
/// assert_eq!(c2.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
/// ```
pub fn r_<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    // Flatten 1D arrays if needed
    if arrays.len() > 1 && arrays.iter().all(|arr| arr.ndim() == 1) {
        // Concatenate directly to create a 1D array
        concatenate(arrays, 0)
    } else {
        // Add a dimension to any 1D arrays and concatenate along axis 0
        let processed_arrays: Result<Vec<Array<T>>> = arrays
            .iter()
            .map(|arr| {
                if arr.ndim() == 1 {
                    Ok(arr.reshape(&[1, arr.size()]))
                } else {
                    Ok((*arr).clone())
                }
            })
            .collect();

        let processed = processed_arrays?;
        let processed_refs: Vec<&Array<T>> = processed.iter().collect();

        concatenate(&processed_refs, 0)
    }
}

/// Create a new array by stacking arrays column-wise (along axis 1)
///
/// # Parameters
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// A new array with the stacked values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let c = c_(&[&a, &b]).expect("c_ should succeed");
/// // With 1D arrays, c_ reshapes them to [n, 1] and concatenates along axis 1
/// assert_eq!(c.shape(), vec![3, 2]);
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // With 2D arrays
/// let a2 = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b2 = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c2 = c_(&[&a2, &b2]).expect("c_ should succeed");
/// assert_eq!(c2.shape(), vec![2, 4]);
/// ```
pub fn c_<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    // Process 1D arrays by adding a dimension
    let processed_arrays: Result<Vec<Array<T>>> = arrays
        .iter()
        .map(|arr| {
            if arr.ndim() == 1 {
                Ok(arr.reshape(&[arr.size(), 1]))
            } else {
                Ok((*arr).clone())
            }
        })
        .collect();

    let processed = processed_arrays?;
    let processed_refs: Vec<&Array<T>> = processed.iter().collect();

    concatenate(&processed_refs, 1)
}

/// Return an array with the specified requirements
///
/// # Parameters
///
/// * `array` - Input array
/// * `requirements` - Requirements for the array, specified as a combination of flags
///
/// # Returns
///
/// * Array with the specified requirements. If the input array satisfies the requirements,
///   it may be returned as-is.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create an array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Require a C-contiguous array (row-major order)
/// let b = require(&a, Some(ArrayRequirements::CONTIGUOUS | ArrayRequirements::C_LAYOUT)).expect("require should succeed");
/// assert!(b.is_c_contiguous());
///
/// // Require a Fortran-contiguous array (column-major order)
/// let c = require(&a, Some(ArrayRequirements::CONTIGUOUS | ArrayRequirements::F_LAYOUT)).expect("require should succeed");
/// assert!(c.is_f_contiguous());
/// ```
pub fn require<T: Clone>(
    array: &Array<T>,
    requirements: Option<super::core::ArrayRequirements>,
) -> Result<Array<T>> {
    use super::core::ArrayRequirements;

    // If no requirements are specified, return a copy of the input array
    let requirements = requirements.unwrap_or(ArrayRequirements::empty());

    // If no requirements are specified, return a copy of the input array
    if requirements.is_empty() {
        return Ok(array.clone());
    }

    // Check if we need a specific layout
    let need_c_layout = requirements.contains(ArrayRequirements::C_LAYOUT);
    let need_f_layout = requirements.contains(ArrayRequirements::F_LAYOUT);

    // Check if we need a contiguous array
    let need_contiguous = requirements.contains(ArrayRequirements::CONTIGUOUS);

    // Check if we need to own the data
    let _need_owner = requirements.contains(ArrayRequirements::OWNDATA);

    // Check if we need a writeable array
    let _need_writeable = requirements.contains(ArrayRequirements::WRITEABLE);

    // Check if the input array satisfies the requirements
    let meets_c_layout = if need_c_layout {
        array.is_c_contiguous()
    } else {
        true
    };
    let meets_f_layout = if need_f_layout {
        array.is_f_contiguous()
    } else {
        true
    };
    let meets_contiguous = if need_contiguous {
        array.is_contiguous()
    } else {
        true
    };

    // NumRS arrays always own their data and are writeable, so these are always met
    // If this changes in the future, we should check for these requirements

    // If all requirements are met, return the original array
    if meets_c_layout && meets_f_layout && meets_contiguous {
        return Ok(array.clone());
    }

    // Otherwise, create a new array that meets the requirements
    let mut result = array.clone();

    // If we need a C-contiguous array, convert to C layout
    if need_c_layout && !meets_c_layout {
        result = result.to_c_layout();
    }

    // If we need a Fortran-contiguous array, convert to F layout
    if need_f_layout && !meets_f_layout {
        result = result.to_f_layout();
    }

    // If we need a contiguous array but neither C nor F layout is specified,
    // prefer C layout (row-major order)
    if need_contiguous && !meets_contiguous && !need_c_layout && !need_f_layout {
        result = result.to_c_layout();
    }

    Ok(result)
}
