//! Take and put operations for arrays
//!
//! This module provides free functions for:
//! - `ix_()` - Generate index arrays for fancy indexing
//! - `put()` - Set array values using indices
//! - `putmask()` - Set array elements using a mask
//! - `take()` - Take elements from an array
//! - `take_along_axis()` - Take values along an axis
//! - `put_along_axis()` - Put values along an axis
//! - `extract()` - Extract elements satisfying a condition

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::IxDyn;

/// Generate index arrays for fancy indexing
///
/// # Parameters
///
/// * `arrays` - A list of arrays to generate index grids from
///
/// # Returns
///
/// A list of N arrays, where N is the number of input arrays. The i-th output array is an array that
/// can be used for indexing the i-th dimension of an array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create arrays for indices
/// let a = Array::from_vec(vec![0, 1, 2]);
/// let b = Array::from_vec(vec![3, 4, 5]);
///
/// // Generate index arrays
/// let indices = ix_(&[&a, &b]).expect("ix_ should succeed with valid arrays");
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].shape(), vec![3, 1]);
/// assert_eq!(indices[1].shape(), vec![1, 3]);
///
/// // Can be used for fancy indexing
/// let data = Array::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(&[3, 3]);
/// // Would select elements at positions (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), etc.
/// ```
pub fn ix_<T: Clone>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    let n = arrays.len();
    let mut result = Vec::with_capacity(n);

    for (i, array) in arrays.iter().enumerate() {
        // Create a shape with 1s for all dimensions except the i-th
        let mut shape = vec![1; n];
        shape[i] = array.size();

        // Create a reshaped copy of the array
        let reshaped = array.reshape(&shape);
        result.push(reshaped);
    }

    Ok(result)
}

/// Set array values using indices
///
/// # Parameters
///
/// * `array` - Array to modify
/// * `indices` - Indices where values should be put
/// * `values` - Values to put at the indices
/// * `mode` - How out-of-bounds indices are handled ('raise', 'wrap', 'clip')
///
/// # Returns
///
/// Result<()> indicating success or error
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::indexing::put;
///
/// // Create an array
/// let mut a: Array<i32> = Array::zeros(&[5]);
///
/// // Set values at specific indices
/// let indices = Array::from_vec(vec![0, 2, 4]);
/// let values = Array::from_vec(vec![10, 20, 30]);
///
/// put(&mut a, &indices, &values, None).expect("put failed");
/// assert_eq!(a.to_vec(), vec![10, 0, 20, 0, 30]);
///
/// // Test with wrap mode
/// let mut b: Array<i32> = Array::zeros(&[3]);
/// let indices = Array::from_vec(vec![0, 1, 2, 3, 4, 5]);
/// let values = Array::from_vec(vec![10, 20, 30, 40, 50, 60]);
///
/// put(&mut b, &indices, &values, Some("wrap")).expect("put failed");
/// // Indices 3,4,5 wrap around to 0,1,2
/// assert_eq!(b.to_vec(), vec![40, 50, 60]);
/// ```
pub fn put<T: Clone + ToString>(
    array: &mut Array<T>,
    indices: &Array<T>,
    values: &Array<T>,
    mode: Option<&str>,
) -> Result<()> {
    let indices_slice = indices.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("indices array should be contiguous".to_string())
    })?;

    // Validate indices are integers
    for i in 0..indices.size() {
        if indices_slice[i].to_string().parse::<isize>().is_err() {
            return Err(NumRs2Error::InvalidOperation(
                "Indices must be integers".to_string(),
            ));
        }
    }

    // Check that values has at least as many elements as indices
    let n_indices = indices.size();
    let n_values = values.size();

    if n_values < n_indices {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Not enough values ({}) to put at all indices ({})",
            n_values, n_indices
        )));
    }

    let array_size = array.size();
    let handle_mode = mode.unwrap_or("raise");

    let values_slice = values.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("values array should be contiguous".to_string())
    })?;

    // Process each index
    for i in 0..n_indices {
        // Get the index value
        let idx_value = indices_slice[i]
            .to_string()
            .parse::<isize>()
            .map_err(|_| NumRs2Error::InvalidOperation("index should be parseable".to_string()))?;

        // Apply the mode
        let idx = match handle_mode {
            "raise" => {
                if idx_value < 0 || idx_value >= array_size as isize {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "Index {} is out of bounds for array with size {}",
                        idx_value, array_size
                    )));
                }
                idx_value as usize
            }
            "wrap" => {
                // Wrap around
                (((idx_value % array_size as isize) + array_size as isize) % array_size as isize)
                    as usize
            }
            "clip" => {
                // Clip to bounds
                if idx_value < 0 {
                    0
                } else if idx_value >= array_size as isize {
                    array_size - 1
                } else {
                    idx_value as usize
                }
            }
            _ => {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Invalid mode: {}. Must be one of 'raise', 'wrap', or 'clip'",
                    handle_mode
                )));
            }
        };

        // Compute the multi-dimensional index
        let shape = array.shape();
        let ndim = shape.len();

        let mut multi_idx = Vec::with_capacity(ndim);
        let mut temp = idx;

        for j in (0..ndim).rev() {
            if j == 0 {
                multi_idx.insert(0, temp);
            } else {
                let prod: usize = shape[1..=j].iter().product();
                multi_idx.insert(0, temp / prod);
                temp %= prod;
            }
        }

        // Get the value to put
        let value = values_slice[i % n_values].clone();

        // Set the value
        array.set(&multi_idx, value)?;
    }

    Ok(())
}

/// Set array elements using a mask array
///
/// # Parameters
///
/// * `array` - Array to modify
/// * `mask` - Boolean mask array of same shape as array
/// * `values` - Values to set at positions where mask is True
///
/// # Returns
///
/// Result<()> indicating success or error
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a simple array
/// let mut a = Array::from_vec(vec![1, 2, 3, 4, 5]);
///
/// // Create a mask (select even indices)
/// let mask = Array::from_vec(vec![false, true, false, true, false]);
///
/// // Set values at masked positions
/// let values = Array::from_vec(vec![20, 40]);
///
/// putmask(&mut a, &mask, &values).expect("putmask should succeed with valid inputs");
/// assert_eq!(a.to_vec(), vec![1, 20, 3, 40, 5]);
/// ```
pub fn putmask<T: Clone + ToString, U: Clone + ToString>(
    array: &mut Array<T>,
    mask: &Array<U>,
    values: &Array<T>,
) -> Result<()> {
    // Check shapes
    if array.shape() != mask.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: mask.shape(),
        });
    }

    let mask_slice = mask.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("mask array should be contiguous".to_string())
    })?;

    // Check if mask contains boolean values
    for i in 0..mask.size() {
        let val_str = mask_slice[i].to_string();
        if val_str != "true" && val_str != "false" {
            return Err(NumRs2Error::InvalidOperation(
                "Mask must contain boolean values".to_string(),
            ));
        }
    }

    // Count true values in mask to check against values size
    let true_count = mask
        .to_vec()
        .iter()
        .filter(|x| x.to_string() == "true")
        .count();

    let n_values = values.size();

    if n_values == 0 && true_count > 0 {
        return Err(NumRs2Error::InvalidOperation(
            "No values provided to fill masked elements".to_string(),
        ));
    }

    let values_slice = values.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("values array should be contiguous".to_string())
    })?;

    // Process each element
    let mut value_idx = 0;

    for i in 0..array.size() {
        let mask_val = mask_slice[i].to_string() == "true";

        if mask_val {
            // Calculate the multi-dimensional index
            let shape = array.shape();
            let ndim = shape.len();

            let mut multi_idx = Vec::with_capacity(ndim);
            let mut temp = i;

            for j in (0..ndim).rev() {
                if j == 0 {
                    multi_idx.insert(0, temp);
                } else {
                    let prod: usize = shape[1..=j].iter().product();
                    multi_idx.insert(0, temp / prod);
                    temp %= prod;
                }
            }

            // Get the value to put (cycling if necessary)
            let value = values_slice[value_idx % n_values].clone();

            // Set the value
            array.set(&multi_idx, value)?;

            value_idx += 1;
        }
    }

    Ok(())
}

/// Take elements from array along axis using indices
///
/// # Parameters
///
/// * `array` - Input array
/// * `indices` - Array of indices to take elements from
/// * `axis` - Optional axis along which to take elements. If None, array is flattened first
/// * `mode` - How out-of-bounds indices are handled ('raise', 'wrap', 'clip')
///
/// # Returns
///
/// A new array with elements taken from the input array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Take elements from a flattened array
/// let a = Array::from_vec(vec![10, 20, 30, 40, 50]);
/// let indices = Array::from_vec(vec![0, 2, 4]);
/// let result = take(&a, &indices, None, None).expect("take should succeed with valid indices");
/// assert_eq!(result.to_vec(), vec![10, 30, 50]);
///
/// // Take elements along a specific axis
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let indices = Array::from_vec(vec![0, 2]);
/// let result = take(&b, &indices, Some(1), None).expect("take should succeed along axis");
/// assert_eq!(result.shape(), vec![2, 2]);
/// assert_eq!(result.to_vec(), vec![1, 3, 4, 6]);
///
/// // Test with wrap mode
/// let c = Array::from_vec(vec![10, 20, 30]);
/// let indices = Array::from_vec(vec![0, 1, 2, 3, 4, 5]);
/// let result = take(&c, &indices, None, Some("wrap")).expect("take with wrap mode should succeed");
/// assert_eq!(result.to_vec(), vec![10, 20, 30, 10, 20, 30]);
/// ```
pub fn take<T: Clone + ToString + num_traits::Zero>(
    array: &Array<T>,
    indices: &Array<T>,
    axis: Option<usize>,
    mode: Option<&str>,
) -> Result<Array<T>> {
    let indices_slice = indices.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("indices array should be contiguous".to_string())
    })?;

    // Validate indices are integers
    for i in 0..indices.size() {
        if indices_slice[i].to_string().parse::<isize>().is_err() {
            return Err(NumRs2Error::InvalidOperation(
                "Indices must be integers".to_string(),
            ));
        }
    }

    let handle_mode = mode.unwrap_or("raise");
    let indices_vec: Vec<isize> = indices
        .to_vec()
        .iter()
        .map(|x| {
            x.to_string()
                .parse::<isize>()
                .map_err(|_| NumRs2Error::InvalidOperation("index should be parseable".to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    match axis {
        None => {
            // Flatten the array and take elements
            let flat_data = array.to_vec();
            let array_size = flat_data.len();
            let mut result_data = Vec::with_capacity(indices_vec.len());

            for &idx_value in &indices_vec {
                // Apply the mode
                let idx = match handle_mode {
                    "raise" => {
                        if idx_value < 0 || idx_value >= array_size as isize {
                            return Err(NumRs2Error::IndexOutOfBounds(format!(
                                "Index {} is out of bounds for array with size {}",
                                idx_value, array_size
                            )));
                        }
                        idx_value as usize
                    }
                    "wrap" => {
                        (((idx_value % array_size as isize) + array_size as isize)
                            % array_size as isize) as usize
                    }
                    "clip" => {
                        if idx_value < 0 {
                            0
                        } else if idx_value >= array_size as isize {
                            array_size - 1
                        } else {
                            idx_value as usize
                        }
                    }
                    _ => {
                        return Err(NumRs2Error::InvalidOperation(format!(
                            "Invalid mode: {}. Must be one of 'raise', 'wrap', or 'clip'",
                            handle_mode
                        )));
                    }
                };

                result_data.push(flat_data[idx].clone());
            }

            Ok(Array::from_vec(result_data))
        }
        Some(ax) => {
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Create output shape
            let mut out_shape = shape.clone();
            out_shape[ax] = indices_vec.len();

            // Validate and process indices
            let processed_indices: Result<Vec<usize>> = indices_vec
                .iter()
                .map(|&idx_value| match handle_mode {
                    "raise" => {
                        if idx_value < 0 || idx_value >= axis_size as isize {
                            return Err(NumRs2Error::IndexOutOfBounds(format!(
                                "Index {} is out of bounds for axis with size {}",
                                idx_value, axis_size
                            )));
                        }
                        Ok(idx_value as usize)
                    }
                    "wrap" => Ok((((idx_value % axis_size as isize) + axis_size as isize)
                        % axis_size as isize) as usize),
                    "clip" => {
                        if idx_value < 0 {
                            Ok(0)
                        } else if idx_value >= axis_size as isize {
                            Ok(axis_size - 1)
                        } else {
                            Ok(idx_value as usize)
                        }
                    }
                    _ => Err(NumRs2Error::InvalidOperation(format!(
                        "Invalid mode: {}. Must be one of 'raise', 'wrap', or 'clip'",
                        handle_mode
                    ))),
                })
                .collect();

            let processed_indices = processed_indices?;

            let mut result_data = Vec::new();

            // Build the result by iterating through all positions and
            // selecting the specified indices along the given axis
            let total_elements = out_shape.iter().product::<usize>();

            for result_idx in 0..total_elements {
                // Convert linear result index to multi-dimensional coordinates
                let mut coords = vec![0; array.ndim()];
                let mut remaining = result_idx;

                // Calculate coordinates in the output array
                for i in (0..array.ndim()).rev() {
                    let size = out_shape[i];
                    coords[i] = remaining % size;
                    remaining /= size;
                }

                // Map the coordinate along the selected axis to the original array
                let original_axis_coord = processed_indices[coords[ax]];

                // Build coordinates for the original array
                let mut orig_coords = coords.clone();
                orig_coords[ax] = original_axis_coord;

                // Convert to linear index in original array
                let mut orig_linear_idx = 0;
                let mut stride = 1;
                for i in (0..array.ndim()).rev() {
                    orig_linear_idx += orig_coords[i] * stride;
                    stride *= shape[i];
                }

                result_data.push(array.to_vec()[orig_linear_idx].clone());
            }

            // Reshape to the correct output shape
            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Take values from array by matching 1D indices along axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `indices` - Array of indices with shape compatible with array
/// * `axis` - Axis along which to take values
///
/// # Returns
///
/// A new array with values taken according to the indices
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Take values along axis 1
/// let a = Array::from_vec(vec![10, 20, 30, 40, 50, 60]).reshape(&[2, 3]);
/// let indices = Array::from_vec(vec![2, 0, 1, 1]).reshape(&[2, 2]);
/// let result = take_along_axis(&a, &indices, 1).expect("take_along_axis should succeed");
/// assert_eq!(result.shape(), vec![2, 2]);
/// assert_eq!(result.to_vec(), vec![30, 10, 50, 50]);
/// ```
pub fn take_along_axis<T: Clone + ToString>(
    array: &Array<T>,
    indices: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis,
            array.ndim()
        )));
    }

    let indices_slice = indices.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("indices array should be contiguous".to_string())
    })?;

    // Validate indices are integers
    for i in 0..indices.size() {
        if indices_slice[i].to_string().parse::<isize>().is_err() {
            return Err(NumRs2Error::InvalidOperation(
                "Indices must be integers".to_string(),
            ));
        }
    }

    let array_shape = array.shape();
    let indices_shape = indices.shape();
    let axis_size = array_shape[axis];

    // Check shape compatibility (all dimensions except axis must match)
    for (i, (&a_dim, &i_dim)) in array_shape.iter().zip(indices_shape.iter()).enumerate() {
        if i != axis && a_dim != i_dim {
            return Err(NumRs2Error::ShapeMismatch {
                expected: array_shape.clone(),
                actual: indices_shape.clone(),
            });
        }
    }

    let result_shape = indices_shape.clone();
    let mut result_data = Vec::with_capacity(indices.size());

    // Process each element in the indices array
    let indices_data: Vec<isize> = indices
        .to_vec()
        .iter()
        .map(|x| {
            x.to_string()
                .parse::<isize>()
                .map_err(|_| NumRs2Error::InvalidOperation("index should be parseable".to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    for (flat_idx, &idx_value) in indices_data.iter().enumerate() {
        // Validate index
        if idx_value < 0 || idx_value >= axis_size as isize {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index {} is out of bounds for axis with size {}",
                idx_value, axis_size
            )));
        }

        // Convert flat index to multi-dimensional index
        let mut multi_idx = Vec::with_capacity(indices_shape.len());
        let mut temp = flat_idx;

        for &dim in indices_shape.iter().rev() {
            multi_idx.insert(0, temp % dim);
            temp /= dim;
        }

        // Modify the index at the specified axis
        multi_idx[axis] = idx_value as usize;

        // Get the value from the array
        let value = array
            .array()
            .get(IxDyn(&multi_idx))
            .ok_or_else(|| {
                NumRs2Error::IndexOutOfBounds(
                    "multi_idx should be valid as index was validated".to_string(),
                )
            })?
            .clone();
        result_data.push(value);
    }

    Ok(Array::from_vec(result_data).reshape(&result_shape))
}

/// Put values into array by matching 1D indices along axis
///
/// # Parameters
///
/// * `array` - Array to modify
/// * `indices` - Array of indices with shape compatible with array
/// * `values` - Values to put at the specified indices
/// * `axis` - Axis along which to put values
///
/// # Returns
///
/// Result<()> indicating success or error
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Put values along axis 1
/// let mut a = Array::zeros(&[2, 3]);
/// let indices = Array::from_vec(vec![2, 0, 1, 1]).reshape(&[2, 2]);
/// let values = Array::from_vec(vec![10, 20, 30, 40]).reshape(&[2, 2]);
/// put_along_axis(&mut a, &indices, &values, 1).expect("put_along_axis should succeed");
/// // a[0, 2] = 10, a[0, 0] = 20, a[1, 1] = 30, a[1, 1] = 40 (overwrites to 40)
/// ```
pub fn put_along_axis<T: Clone + ToString>(
    array: &mut Array<T>,
    indices: &Array<T>,
    values: &Array<T>,
    axis: usize,
) -> Result<()> {
    if axis >= array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis,
            array.ndim()
        )));
    }

    let indices_slice = indices.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("indices array should be contiguous".to_string())
    })?;

    // Validate indices are integers
    for i in 0..indices.size() {
        if indices_slice[i].to_string().parse::<isize>().is_err() {
            return Err(NumRs2Error::InvalidOperation(
                "Indices must be integers".to_string(),
            ));
        }
    }

    let array_shape = array.shape();
    let indices_shape = indices.shape();
    let values_shape = values.shape();
    let axis_size = array_shape[axis];

    // Check shape compatibility
    if indices_shape != values_shape {
        return Err(NumRs2Error::ShapeMismatch {
            expected: indices_shape.clone(),
            actual: values_shape.clone(),
        });
    }

    // Check shape compatibility (all dimensions except axis must match or be broadcastable)
    for (i, (&a_dim, &i_dim)) in array_shape.iter().zip(indices_shape.iter()).enumerate() {
        if i != axis && a_dim != i_dim {
            return Err(NumRs2Error::ShapeMismatch {
                expected: array_shape.clone(),
                actual: indices_shape.clone(),
            });
        }
    }

    // Process each element in the indices array
    let indices_data: Vec<isize> = indices
        .to_vec()
        .iter()
        .map(|x| {
            x.to_string()
                .parse::<isize>()
                .map_err(|_| NumRs2Error::InvalidOperation("index should be parseable".to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let values_data = values.to_vec();

    for (flat_idx, &idx_value) in indices_data.iter().enumerate() {
        // Validate index
        if idx_value < 0 || idx_value >= axis_size as isize {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index {} is out of bounds for axis with size {}",
                idx_value, axis_size
            )));
        }

        // Convert flat index to multi-dimensional index
        let mut multi_idx = Vec::with_capacity(indices_shape.len());
        let mut temp = flat_idx;

        for &dim in indices_shape.iter().rev() {
            multi_idx.insert(0, temp % dim);
            temp /= dim;
        }

        // Modify the index at the specified axis
        multi_idx[axis] = idx_value as usize;

        // Set the value in the array
        array.set(&multi_idx, values_data[flat_idx].clone())?;
    }

    Ok(())
}

/// Return the elements of an array that satisfy some condition
///
/// # Parameters
///
/// * `array` - Input array
/// * `condition` - Array of boolean values with the same shape as array
///
/// # Returns
///
/// A 1-D array containing the elements that satisfy the condition
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Extract elements greater than 5
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
/// let condition = a.map(|x| x > 5);
/// let result = extract(&a, &condition).expect("extract should succeed with valid condition");
/// assert_eq!(result.to_vec(), vec![6, 7, 8, 9, 10]);
///
/// // With a 2D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let cond = b.map(|x| x % 2 == 0);
/// let result = extract(&b, &cond).expect("extract should succeed with 2D array");
/// assert_eq!(result.to_vec(), vec![2, 4, 6]);
/// ```
pub fn extract<T: Clone + ToString, U: Clone + ToString>(
    array: &Array<T>,
    condition: &Array<U>,
) -> Result<Array<T>> {
    // Check that shapes match
    if array.shape() != condition.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: condition.shape(),
        });
    }

    let cond_slice = condition.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation("condition array should be contiguous".to_string())
    })?;

    // Check if condition contains boolean values
    for i in 0..condition.size() {
        let val_str = cond_slice[i].to_string();
        if val_str != "true" && val_str != "false" {
            return Err(NumRs2Error::InvalidOperation(
                "Condition must contain boolean values".to_string(),
            ));
        }
    }

    // Extract elements where condition is true
    let array_data = array.to_vec();
    let condition_data: Vec<bool> = condition
        .to_vec()
        .iter()
        .map(|x| x.to_string() == "true")
        .collect();

    let mut result = Vec::new();
    for (i, &cond) in condition_data.iter().enumerate() {
        if cond {
            result.push(array_data[i].clone());
        }
    }

    Ok(Array::from_vec(result))
}
