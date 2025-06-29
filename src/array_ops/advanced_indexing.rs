//! Advanced indexing operations for Arrays
//!
//! This module provides advanced indexing functionality similar to NumPy's
//! advanced indexing capabilities, including compress, extract, place, put,
//! and other sophisticated indexing operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

/// Return selected slices of an array along given axis
///
/// # Arguments
/// * `array` - Input array
/// * `condition` - 1-D array of booleans corresponding to indices to select
/// * `axis` - Axis along which to take slices. If None, work on flattened array
///
/// # Returns
/// * `Result<Array<T>>` - Array with selected slices
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::compress;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let condition = Array::from_vec(vec![true, false, true, false, true]);
/// let compressed = compress(&arr, &condition, None).unwrap();
/// // Returns [1, 3, 5]
///
/// let arr2d = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let cond = Array::from_vec(vec![true, false, true]);
/// let compressed = compress(&arr2d, &cond, Some(1)).unwrap();
/// // Returns [[1, 3], [4, 6]] (columns 0 and 2)
/// ```
pub fn compress<T: Clone + Zero>(
    array: &Array<T>,
    condition: &Array<bool>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Work on flattened array
            let flat = array.to_vec();
            let cond_flat = condition.to_vec();

            if flat.len() != cond_flat.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![flat.len()],
                    actual: vec![cond_flat.len()],
                });
            }

            let compressed: Vec<T> = flat
                .into_iter()
                .zip(cond_flat)
                .filter_map(|(val, cond)| if cond { Some(val) } else { None })
                .collect();

            Ok(Array::from_vec(compressed))
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

            let cond_vec = condition.to_vec();
            if cond_vec.len() != shape[ax] {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![shape[ax]],
                    actual: vec![cond_vec.len()],
                });
            }

            // Determine which indices to keep
            let indices: Vec<usize> = cond_vec
                .into_iter()
                .enumerate()
                .filter_map(|(i, cond)| if cond { Some(i) } else { None })
                .collect();

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = indices.len();

            if indices.is_empty() {
                return Ok(Array::zeros(&new_shape));
            }

            // Extract selected slices
            let mut result_data = Vec::with_capacity(new_shape.iter().product());

            // Helper to iterate through all indices
            let mut current_indices = vec![0; shape.len()];
            let total_elements: usize = shape.iter().product();

            for _ in 0..total_elements {
                // Check if current index along axis is in our selection
                if indices.contains(&current_indices[ax]) {
                    let value = array.get(&current_indices)?;
                    result_data.push(value);
                }

                // Increment indices
                let mut carry = true;
                for dim in (0..shape.len()).rev() {
                    if carry {
                        current_indices[dim] += 1;
                        carry = current_indices[dim] >= shape[dim];
                        if carry {
                            current_indices[dim] = 0;
                        }
                    }
                }
            }

            Ok(Array::from_vec(result_data).reshape(&new_shape))
        }
    }
}

/// Return the elements of an array that satisfy some condition
///
/// This is equivalent to `array[condition]` in NumPy where condition is a boolean array.
///
/// # Arguments
/// * `array` - Input array
/// * `condition` - Boolean array with same shape as `array`
///
/// # Returns
/// * `Result<Array<T>>` - 1-D array with elements where condition is True
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::extract;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let condition = Array::from_vec(vec![true, false, true, false, true, false]).reshape(&[2, 3]);
/// let extracted = extract(&arr, &condition).unwrap();
/// // Returns [1, 3, 5]
/// ```
pub fn extract<T: Clone>(array: &Array<T>, condition: &Array<bool>) -> Result<Array<T>> {
    if array.shape() != condition.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: condition.shape(),
        });
    }

    let data = array.to_vec();
    let cond_data = condition.to_vec();

    let extracted: Vec<T> = data
        .into_iter()
        .zip(cond_data)
        .filter_map(|(val, cond)| if cond { Some(val) } else { None })
        .collect();

    Ok(Array::from_vec(extracted))
}

/// Place values into array at specified indices
///
/// # Arguments
/// * `array` - Array to modify (modified in-place)
/// * `mask` - Boolean array indicating where to place values
/// * `values` - Values to place (will be repeated if necessary)
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::place;
///
/// let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let mask = Array::from_vec(vec![false, true, false, true, false]);
/// place(&mut arr, &mask, &[10, 20]).unwrap();
/// // arr is now [1, 10, 3, 20, 5]
/// ```
pub fn place<T: Clone>(array: &mut Array<T>, mask: &Array<bool>, values: &[T]) -> Result<()> {
    if array.shape() != mask.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: mask.shape(),
        });
    }

    if values.is_empty() {
        return Err(NumRs2Error::ValueError(
            "values array cannot be empty".to_string(),
        ));
    }

    let mask_data = mask.to_vec();
    let num_true = mask_data.iter().filter(|&&x| x).count();

    if num_true == 0 {
        return Ok(()); // Nothing to place
    }

    // Get mutable slice
    let array_data = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    let mut value_idx = 0;
    for (i, &is_true) in mask_data.iter().enumerate() {
        if is_true {
            array_data[i] = values[value_idx % values.len()].clone();
            value_idx += 1;
        }
    }

    Ok(())
}

/// Replaces specified elements of array with given values
///
/// # Arguments
/// * `array` - Array to modify (modified in-place)
/// * `indices` - 1-D array of indices
/// * `values` - Values to put at those indices
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::put;
///
/// let mut arr = Array::from_vec(vec![0, 0, 0, 0, 0]);
/// let indices = Array::from_vec(vec![0, 2, 4]);
/// put(&mut arr, &indices, &[10, 20, 30]).unwrap();
/// // arr is now [10, 0, 20, 0, 30]
/// ```
pub fn put<T: Clone>(array: &mut Array<T>, indices: &Array<usize>, values: &[T]) -> Result<()> {
    if values.is_empty() {
        return Err(NumRs2Error::ValueError(
            "values array cannot be empty".to_string(),
        ));
    }

    let indices_vec = indices.to_vec();
    let array_len = array.size();

    // Validate indices
    for &idx in &indices_vec {
        if idx >= array_len {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "index {} is out of bounds for array of size {}",
                idx, array_len
            )));
        }
    }

    // Get mutable slice
    let array_data = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    for (i, &idx) in indices_vec.iter().enumerate() {
        array_data[idx] = values[i % values.len()].clone();
    }

    Ok(())
}

/// Put values into array using a mask
///
/// # Arguments
/// * `array` - Array to modify (modified in-place)
/// * `mask` - Boolean mask array  
/// * `values` - Array of values to put where mask is True
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::putmask;
///
/// let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let mask = Array::from_vec(vec![false, true, false, true, false]);
/// let values = Array::from_vec(vec![10, 20]);
/// putmask(&mut arr, &mask, &values).unwrap();
/// // arr is now [1, 10, 3, 20, 5]
/// ```
pub fn putmask<T: Clone>(
    array: &mut Array<T>,
    mask: &Array<bool>,
    values: &Array<T>,
) -> Result<()> {
    place(array, mask, &values.to_vec())
}

/// Take values from array along an axis using indices
///
/// # Arguments
/// * `array` - Input array
/// * `indices` - Array of indices to take
/// * `axis` - Axis along which to take values
///
/// # Returns
/// * `Result<Array<T>>` - Array with values taken along the specified axis
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::take_along_axis;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let indices = Array::from_vec(vec![0, 2, 1, 0]).reshape(&[2, 2]);
/// let result = take_along_axis(&arr, &indices, 1).unwrap();
/// // For row 0: takes elements at indices [0, 2] -> [1, 3]
/// // For row 1: takes elements at indices [1, 0] -> [5, 4]
/// // Result is [[1, 3], [5, 4]]
/// ```
pub fn take_along_axis<T: Clone + Zero>(
    array: &Array<T>,
    indices: &Array<usize>,
    axis: usize,
) -> Result<Array<T>> {
    let arr_shape = array.shape();
    let ind_shape = indices.shape();

    if axis >= arr_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            arr_shape.len()
        )));
    }

    // Check that shapes match except along the specified axis
    if arr_shape.len() != ind_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "array and indices must have same number of dimensions, got {} and {}",
            arr_shape.len(),
            ind_shape.len()
        )));
    }

    for (i, (&arr_dim, &ind_dim)) in arr_shape.iter().zip(ind_shape.iter()).enumerate() {
        if i != axis && arr_dim != ind_dim {
            return Err(NumRs2Error::ShapeMismatch {
                expected: arr_shape.clone(),
                actual: ind_shape.clone(),
            });
        }
    }

    // Create result array with same shape as indices
    let mut result_data = Vec::with_capacity(indices.size());

    // Iterate through all positions in indices array
    let mut current_pos = vec![0; ind_shape.len()];
    let total_elements = indices.size();

    for _ in 0..total_elements {
        // Get the index value at current position
        let idx = indices.get(&current_pos)?;

        // Check bounds
        if idx >= arr_shape[axis] {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "index {} is out of bounds for axis {} with size {}",
                idx, axis, arr_shape[axis]
            )));
        }

        // Build position in source array
        let mut source_pos = current_pos.clone();
        source_pos[axis] = idx;

        // Get value and add to result
        let value = array.get(&source_pos)?;
        result_data.push(value);

        // Increment position
        let mut carry = true;
        for dim in (0..ind_shape.len()).rev() {
            if carry {
                current_pos[dim] += 1;
                carry = current_pos[dim] >= ind_shape[dim];
                if carry {
                    current_pos[dim] = 0;
                }
            }
        }
    }

    Ok(Array::from_vec(result_data).reshape(&ind_shape))
}

/// Apply a function to 1-D slices along the given axis
///
/// # Arguments
/// * `func` - Function to apply to each 1-D slice
/// * `array` - Input array
/// * `axis` - Axis along which array is sliced
///
/// # Returns
/// * `Result<Array<U>>` - Array with function applied along axis
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::apply_along_axis;
///
/// // Sum along axis
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
/// let result = apply_along_axis(
///     |slice: &Array<f64>| -> f64 { slice.sum() },
///     &arr,
///     1
/// ).unwrap();
/// // Sums each row: [6.0, 15.0]
/// ```
pub fn apply_along_axis<T, U, F>(func: F, array: &Array<T>, axis: usize) -> Result<Array<U>>
where
    T: Clone + Zero,
    U: Clone + Zero,
    F: Fn(&Array<T>) -> U,
{
    let shape = array.shape();

    if axis >= shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            shape.len()
        )));
    }

    // Calculate output shape (remove the axis dimension)
    let mut out_shape = shape.clone();
    out_shape.remove(axis);

    if out_shape.is_empty() {
        // If only one dimension, return scalar as 1-element array
        let result = func(array);
        return Ok(Array::from_vec(vec![result]));
    }

    let mut result_data = Vec::new();

    // Number of slices to process
    let n_slices: usize = out_shape.iter().product();

    for slice_idx in 0..n_slices {
        // Convert linear index to multi-dimensional position
        let mut slice_pos = vec![0; out_shape.len()];
        let mut temp = slice_idx;
        for i in (0..out_shape.len()).rev() {
            slice_pos[i] = temp % out_shape[i];
            temp /= out_shape[i];
        }

        // Extract 1-D slice along the axis
        let mut slice_data = Vec::with_capacity(shape[axis]);

        for i in 0..shape[axis] {
            // Build full position in original array
            let mut full_pos = Vec::with_capacity(shape.len());
            let mut slice_dim = 0;

            for dim in 0..shape.len() {
                if dim == axis {
                    full_pos.push(i);
                } else {
                    full_pos.push(slice_pos[slice_dim]);
                    slice_dim += 1;
                }
            }

            slice_data.push(array.get(&full_pos)?);
        }

        // Apply function to slice
        let slice_array = Array::from_vec(slice_data);
        let result = func(&slice_array);
        result_data.push(result);
    }

    Ok(Array::from_vec(result_data).reshape(&out_shape))
}

/// Apply a function over multiple axes
///
/// # Arguments
/// * `func` - Function that takes an array and returns a scalar
/// * `array` - Input array
/// * `axes` - Axes over which to apply the function
///
/// # Returns
/// * `Result<Array<T>>` - Result array with specified axes removed
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::apply_over_axes;
///
/// // Sum over multiple axes
/// let arr = Array::from_vec(vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0
/// ]).reshape(&[2, 2, 2]);
///
/// let result = apply_over_axes(
///     |a: &Array<f64>| -> Result<Array<f64>> { Ok(Array::from_vec(vec![a.sum()])) },
///     &arr,
///     &[1, 2]
/// ).unwrap();
/// // Sums over axes 1 and 2, leaving axis 0: [10.0, 26.0]
/// ```
pub fn apply_over_axes<T, F>(func: F, array: &Array<T>, axes: &[usize]) -> Result<Array<T>>
where
    T: Clone + Zero,
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    let mut result = array.clone();

    // Sort axes in descending order to handle removal correctly
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));

    for &axis in &sorted_axes {
        if axis >= result.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "axis {} is out of bounds for array of dimension {}",
                axis,
                result.ndim()
            )));
        }

        // Apply function along this axis
        let temp = func(&result)?;

        // The function should reduce the dimension along the axis
        if temp.ndim() != result.ndim() - 1 {
            return Err(NumRs2Error::InvalidOperation(
                "Function must reduce dimension by 1".to_string(),
            ));
        }

        result = temp;
    }

    Ok(result)
}
