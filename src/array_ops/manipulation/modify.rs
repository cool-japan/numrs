//! Array modification operations
//!
//! This module provides functions for modifying array contents:
//! - Deleting elements from arrays
//! - Inserting values into arrays
//! - Trimming zeros from arrays
//! - Extracting elements by condition
//! - Placing values by condition
//! - Compressing arrays

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

/// Delete sub-arrays along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `indices` - Indicate which sub-arrays to remove (can be slice, integer, or array of integers)
/// * `axis` - The axis along which to delete. If None, array is flattened before operation
///
/// # Returns
///
/// A copy of array with the elements specified by indices removed
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Delete element at index 1 from 1D array
/// let a = Array::from_vec(vec![0, 1, 2, 3, 4]);
/// let result = delete(&a, &[1], None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![0, 2, 3, 4]);
///
/// // Delete multiple elements
/// let result = delete(&a, &[1, 3], None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![0, 2, 4]);
///
/// // Delete from 2D array along axis 0
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[3, 2]);
/// let result = delete(&b, &[1], Some(0)).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![2, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 6]);
/// ```
pub fn delete<T: Clone + Zero>(
    array: &Array<T>,
    indices: &[usize],
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        Some(ax) => {
            // Delete along specified axis
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Check if indices are valid
            for &idx in indices {
                if idx >= axis_size {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Index {} out of bounds for axis {} with size {}",
                        idx, ax, axis_size
                    )));
                }
            }

            // Create a sorted, unique list of indices to delete
            let mut delete_indices = indices.to_vec();
            delete_indices.sort_unstable();
            delete_indices.dedup();

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = axis_size - delete_indices.len();

            if new_shape[ax] == 0 {
                // Result would be empty along this axis
                return Ok(Array::zeros(&new_shape));
            }

            // Create result array
            let mut result_data = Vec::with_capacity(new_shape.iter().product());

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through all positions
            let total_size: usize = shape.iter().product();
            for i in 0..total_size {
                // Convert flat index to multi-dimensional indices
                let mut indices_arr = vec![0; shape.len()];
                let mut temp = i;
                for j in 0..shape.len() {
                    indices_arr[j] = temp / strides[j];
                    temp %= strides[j];
                }

                // Check if this position should be deleted
                let axis_pos = indices_arr[ax];
                if !delete_indices.contains(&axis_pos) {
                    result_data.push(array.get(&indices_arr)?);
                }
            }

            Ok(Array::from_vec_shape(result_data, &new_shape)?)
        }
        None => {
            // Flatten array and delete from flattened version
            let flat = array.to_vec();
            let flat_size = flat.len();

            // Check if indices are valid
            for &idx in indices {
                if idx >= flat_size {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Index {} out of bounds for flattened array with size {}",
                        idx, flat_size
                    )));
                }
            }

            // Create a sorted, unique list of indices to delete
            let mut delete_indices = indices.to_vec();
            delete_indices.sort_unstable();
            delete_indices.dedup();

            // Create result by including only non-deleted elements
            let mut result_data = Vec::with_capacity(flat_size - delete_indices.len());
            let mut del_idx = 0;

            for (i, val) in flat.iter().enumerate() {
                if del_idx < delete_indices.len() && i == delete_indices[del_idx] {
                    del_idx += 1;
                } else {
                    result_data.push(val.clone());
                }
            }

            Ok(Array::from_vec(result_data))
        }
    }
}

/// Insert values along the given axis before the given indices
///
/// # Parameters
///
/// * `array` - Input array
/// * `indices` - Indices before which values is inserted (single index or slice)
/// * `values` - Values to insert into array
/// * `axis` - Axis along which to insert values. If None, array is flattened first
///
/// # Returns
///
/// A copy of array with values inserted
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::manipulation::insert;
///
/// // Insert single value into 1D array
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = insert(&a, &[1], &[99], None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 99, 2, 3]);
///
/// // Insert multiple values
/// let result = insert(&a, &[1, 2], &[99, 100], None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 99, 2, 100, 3]);
///
/// // Insert into 2D array along axis
/// let b = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let values = vec![5, 6];
/// let result = insert(&b, &[1], &values, Some(0)).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![4, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 5, 6, 6, 3, 4]);
/// ```
pub fn insert<T: Clone + Zero>(
    array: &Array<T>,
    indices: &[usize],
    values: &[T],
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        Some(ax) => {
            // Insert along specified axis
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Sort indices for proper insertion
            let mut sorted_indices: Vec<(usize, usize)> = indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| (idx, i))
                .collect();
            sorted_indices.sort_by_key(|&(idx, _)| idx);

            // Calculate how many elements we're inserting along the axis
            let values_per_insertion = if values.len() == 1 {
                // Single value to be inserted at all positions
                1
            } else if values.len() == indices.len() {
                // One value per index
                1
            } else {
                // Values should be a multiple of indices.len()
                if !values.len().is_multiple_of(indices.len()) {
                    return Err(NumRs2Error::InvalidOperation(
                        "Values length must be 1, equal to indices length, or a multiple of indices length".into()
                    ));
                }
                values.len() / indices.len()
            };

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = axis_size + indices.len() * values_per_insertion;

            // Calculate total size of sub-array perpendicular to axis
            let mut sub_size = 1;
            for (i, &dim) in shape.iter().enumerate() {
                if i != ax {
                    sub_size *= dim;
                }
            }

            // Build result array
            let mut result_data = Vec::with_capacity(new_shape.iter().product());

            // Process each position along the axis
            let mut src_pos = 0;
            let mut insert_idx = 0;

            for _new_pos in 0..new_shape[ax] {
                // Check if we should insert values at this position
                let mut should_insert = false;
                let mut which_insert = 0;

                if insert_idx < sorted_indices.len() {
                    let (idx, orig_order) = sorted_indices[insert_idx];
                    if src_pos == idx {
                        should_insert = true;
                        which_insert = orig_order;
                    }
                }

                if should_insert {
                    // Insert values
                    for val_idx in 0..values_per_insertion {
                        // Copy the sub-array structure but with inserted values
                        for _sub_idx in 0..sub_size {
                            let value_idx = if values.len() == 1 {
                                0
                            } else if values.len() == indices.len() {
                                which_insert
                            } else {
                                which_insert * values_per_insertion + val_idx
                            };

                            result_data.push(values[value_idx].clone());
                        }
                    }
                    insert_idx += 1;
                } else if src_pos < axis_size {
                    // Copy from original array
                    // Calculate strides for copying
                    let mut indices_arr = vec![0; shape.len()];

                    for sub_idx in 0..sub_size {
                        // Convert sub_idx to multi-dimensional indices
                        let mut temp = sub_idx;
                        for i in (0..shape.len()).rev() {
                            if i == ax {
                                indices_arr[i] = src_pos;
                            } else {
                                let dim = shape[i];
                                if i < shape.len() - 1 {
                                    indices_arr[i] = temp % dim;
                                    temp /= dim;
                                } else {
                                    indices_arr[i] = temp;
                                }
                            }
                        }

                        result_data.push(array.get(&indices_arr)?);
                    }
                    src_pos += 1;
                }
            }

            Ok(Array::from_vec_shape(result_data, &new_shape)?)
        }
        None => {
            // Flatten array and insert into flattened version
            let flat = array.to_vec();
            let flat_size = flat.len();

            if indices.len() != values.len() && values.len() != 1 {
                return Err(NumRs2Error::InvalidOperation(
                    "For flat insertion, values must have length 1 or match indices length".into(),
                ));
            }

            // Create pairs of (index, value) and sort by index
            let mut insertions: Vec<(usize, T)> = Vec::new();
            for (i, &idx) in indices.iter().enumerate() {
                let val = if values.len() == 1 {
                    values[0].clone()
                } else {
                    values[i].clone()
                };
                insertions.push((idx, val));
            }
            insertions.sort_by_key(|&(idx, _)| idx);

            // Build result
            let mut result_data = Vec::with_capacity(flat_size + insertions.len());
            let mut orig_idx = 0;
            let mut insert_idx = 0;

            for _pos in 0..flat_size + insertions.len() {
                if insert_idx < insertions.len() && insertions[insert_idx].0 == orig_idx {
                    result_data.push(insertions[insert_idx].1.clone());
                    insert_idx += 1;
                } else if orig_idx < flat_size {
                    result_data.push(flat[orig_idx].clone());
                    orig_idx += 1;
                }
            }

            // Append any remaining insertions at the end
            while insert_idx < insertions.len() {
                result_data.push(insertions[insert_idx].1.clone());
                insert_idx += 1;
            }

            Ok(Array::from_vec(result_data))
        }
    }
}

/// Trim the leading and/or trailing zeros from a 1-D array
///
/// # Parameters
///
/// * `array` - Input array
/// * `trim` - A string with 'f' representing trim from front and 'b' to trim from back.
///   Default is 'fb', trim zeros from both front and back of the array.
///
/// # Returns
///
/// 1-D array with trimmed zeros
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Trim zeros from both ends
/// let a = Array::from_vec(vec![0, 0, 0, 1, 2, 3, 0, 0, 0]);
/// let result = trim_zeros(&a, None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 2, 3]);
///
/// // Trim only from front
/// let result = trim_zeros(&a, Some("f")).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 0, 0, 0]);
///
/// // Trim only from back
/// let result = trim_zeros(&a, Some("b")).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![0, 0, 0, 1, 2, 3]);
/// ```
pub fn trim_zeros<T>(array: &Array<T>, trim: Option<&str>) -> Result<Array<T>>
where
    T: Clone + Zero + PartialEq,
{
    // Ensure array is 1-D
    if array.ndim() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "trim_zeros requires a 1-D array".into(),
        ));
    }

    let data = array.to_vec();
    if data.is_empty() {
        return Ok(array.clone());
    }

    let trim_str = trim.unwrap_or("fb");
    let trim_front = trim_str.contains('f');
    let trim_back = trim_str.contains('b');

    let mut start = 0;
    let mut end = data.len();

    // Find the first non-zero element from front
    if trim_front {
        for (i, val) in data.iter().enumerate() {
            if !val.is_zero() {
                start = i;
                break;
            }
        }
        // If all elements are zero
        if start == 0 && data[0].is_zero() {
            let all_zero = data.iter().all(|x| x.is_zero());
            if all_zero {
                return Ok(Array::from_vec(vec![]));
            }
        }
    }

    // Find the first non-zero element from back
    if trim_back {
        for (i, val) in data.iter().enumerate().rev() {
            if !val.is_zero() {
                end = i + 1;
                break;
            }
        }
    }

    // If start >= end, all remaining elements are zeros
    if start >= end {
        return Ok(Array::from_vec(vec![]));
    }

    // Create the trimmed array
    let trimmed_data: Vec<T> = data[start..end].to_vec();
    Ok(Array::from_vec(trimmed_data))
}

/// Extract elements from an array that satisfy a condition
///
/// # Parameters
///
/// * `array` - Input array
/// * `condition` - Boolean array with same shape as `array`
///
/// # Returns
///
/// A 1-D array containing the elements from `array` where `condition` is true
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Extract positive values
/// let a = Array::from_vec(vec![-1, 2, -3, 4, -5]);
/// let condition = Array::from_vec(vec![false, true, false, true, false]);
/// let result = extract(&a, &condition).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![2, 4]);
///
/// // Extract from 2D array
/// let b = Array::from_vec(vec![1, -2, 3, -4, 5, -6]).reshape(&[2, 3]);
/// let cond = Array::from_vec(vec![true, false, true, false, true, false]).reshape(&[2, 3]);
/// let result = extract(&b, &cond).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 3, 5]);
/// ```
pub fn extract<T: Clone>(array: &Array<T>, condition: &Array<bool>) -> Result<Array<T>> {
    // Check that arrays have the same shape
    if array.shape() != condition.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Array and condition must have the same shape, got {:?} and {:?}",
            array.shape(),
            condition.shape()
        )));
    }

    // Flatten both arrays
    let array_flat = array.to_vec();
    let condition_flat = condition.to_vec();

    // Extract elements where condition is true
    let mut result_data = Vec::new();
    for (val, &cond) in array_flat.iter().zip(condition_flat.iter()) {
        if cond {
            result_data.push(val.clone());
        }
    }

    Ok(Array::from_vec(result_data))
}

/// Place values into an array according to a condition
///
/// # Parameters
///
/// * `array` - Array to be modified (modified in-place)
/// * `mask` - Boolean array with same shape as `array`
/// * `values` - Values to place where mask is true
///
/// # Returns
///
/// The modified array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Replace negative values with 0
/// let mut a = Array::from_vec(vec![-1, 2, -3, 4, -5]);
/// let mask = Array::from_vec(vec![true, false, true, false, true]);
/// place(&mut a, &mask, &[0, 0, 0]).expect("operation should succeed");
/// assert_eq!(a.to_vec(), vec![0, 2, 0, 4, 0]);
///
/// // Place values in 2D array
/// let mut b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let mask = Array::from_vec(vec![false, true, false, true, false, true]).reshape(&[2, 3]);
/// place(&mut b, &mask, &[10, 20, 30]).expect("operation should succeed");
/// assert_eq!(b.to_vec(), vec![1, 10, 3, 20, 5, 30]);
/// ```
pub fn place<T: Clone>(array: &mut Array<T>, mask: &Array<bool>, values: &[T]) -> Result<()> {
    // Check that arrays have the same shape
    if array.shape() != mask.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Array and mask must have the same shape, got {:?} and {:?}",
            array.shape(),
            mask.shape()
        )));
    }

    // Count true values in mask
    let mask_flat = mask.to_vec();
    let true_count = mask_flat.iter().filter(|&&x| x).count();

    // Check that we have the right number of values
    if values.len() != true_count {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Number of values ({}) must match number of true elements in mask ({})",
            values.len(),
            true_count
        )));
    }

    // Get mutable slice of array data
    let array_slice = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    // Place values where mask is true
    let mut value_idx = 0;
    for (i, &mask_val) in mask_flat.iter().enumerate() {
        if mask_val {
            array_slice[i] = values[value_idx].clone();
            value_idx += 1;
        }
    }

    Ok(())
}

/// Replace specified elements of an array with given values
///
/// # Parameters
///
/// * `array` - Array to be modified (modified in-place)
/// * `indices` - Target indices, can be flattened or multi-dimensional
/// * `values` - Values to place at target indices
///
/// # Returns
///
/// The modified array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Put values at specific indices in 1D array
/// let mut a = Array::from_vec(vec![0, 0, 0, 0, 0]);
/// put(&mut a, &[1, 3], &[10, 30]).expect("operation should succeed");
/// assert_eq!(a.to_vec(), vec![0, 10, 0, 30, 0]);
///
/// // Put with repeating values
/// let mut b = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// put(&mut b, &[0, 2, 4], &[100]).expect("operation should succeed");
/// assert_eq!(b.to_vec(), vec![100, 2, 100, 4, 100]);
/// ```
pub fn put<T: Clone>(array: &mut Array<T>, indices: &[usize], values: &[T]) -> Result<()> {
    if values.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Values array cannot be empty".into(),
        ));
    }

    // Get mutable slice of array data
    let array_slice = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;
    let array_size = array_slice.len();

    // Check that all indices are valid
    for &idx in indices {
        if idx >= array_size {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Index {} out of bounds for array with size {}",
                idx, array_size
            )));
        }
    }

    // Place values at indices, cycling through values if necessary
    for (i, &idx) in indices.iter().enumerate() {
        array_slice[idx] = values[i % values.len()].clone();
    }

    Ok(())
}

/// Select slices from an array along a given axis
///
/// # Parameters
///
/// * `array` - Array from which to select slices
/// * `condition` - 1-D boolean array that selects which slices to return
/// * `axis` - The axis along which to take slices. If None, array is flattened
///
/// # Returns
///
/// A new array with slices taken along the specified axis where condition is true
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Compress 1D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let condition = Array::from_vec(vec![true, false, true, false, true]);
/// let result = compress(&a, &condition, None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 3, 5]);
///
/// // Compress 2D array along axis 0
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[3, 2]);
/// let cond = Array::from_vec(vec![true, false, true]);
/// let result = compress(&b, &cond, Some(0)).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![2, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 6]);
/// ```
pub fn compress<T: Clone + Zero>(
    array: &Array<T>,
    condition: &Array<bool>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    // Ensure condition is 1-D
    if condition.ndim() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "Condition must be a 1-D array".into(),
        ));
    }

    match axis {
        Some(ax) => {
            // Compress along specified axis
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Check condition length matches axis size
            if condition.size() != axis_size {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Condition length {} doesn't match axis {} size {}",
                    condition.size(),
                    ax,
                    axis_size
                )));
            }

            // Get indices where condition is true
            let condition_vec = condition.to_vec();
            let selected_indices: Vec<usize> = condition_vec
                .iter()
                .enumerate()
                .filter_map(|(i, &val)| if val { Some(i) } else { None })
                .collect();

            if selected_indices.is_empty() {
                // Return empty array with appropriate shape
                let mut new_shape = shape.clone();
                new_shape[ax] = 0;
                return Array::from_vec_shape(vec![], &new_shape);
            }

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = selected_indices.len();

            // Calculate strides for indexing
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Collect selected slices
            let total_size: usize = new_shape.iter().product();
            let mut result_data = Vec::with_capacity(total_size);

            for i in 0..total_size {
                // Convert flat index to multi-dimensional indices
                let mut indices_arr = vec![0; shape.len()];
                let mut temp = i;
                for j in 0..new_shape.len() {
                    indices_arr[j] = temp / strides[j];
                    temp %= strides[j];
                }

                // Map the index along the compressed axis
                if ax < indices_arr.len() && indices_arr[ax] < selected_indices.len() {
                    indices_arr[ax] = selected_indices[indices_arr[ax]];
                    result_data.push(array.get(&indices_arr)?);
                }
            }

            Ok(Array::from_vec_shape(result_data, &new_shape)?)
        }
        None => {
            // Flatten array and compress
            extract(array, condition)
        }
    }
}
