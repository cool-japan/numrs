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
/// ]);
/// let arr = arr.reshape(&[2, 2, 2]);
///
/// let result = apply_over_axes(
///     |a: &Array<f64>| -> Result<Array<f64>> { a.sum_axis(0) },
///     &arr,
///     &[1]
/// ).unwrap();
/// // Sums over axis 1, reducing dimension by 1
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

/// Take elements from array using integer array indices (fancy indexing)
///
/// This implements NumPy-style fancy indexing where an array of integers
/// is used to select elements from the input array. The result has the
/// same shape as the indices array.
///
/// # Arguments
/// * `array` - Input array
/// * `indices` - Integer array of indices to take
/// * `axis` - Optional axis along which to take. If None, array is flattened
///
/// # Returns
/// * `Result<Array<T>>` - Array with selected elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::take;
///
/// // 1-D fancy indexing
/// let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
/// let indices = Array::from_vec(vec![0, 2, 4, 1]);
/// let result = take(&arr, &indices, None).unwrap();
/// // Returns [10, 30, 50, 20]
///
/// // 2-D fancy indexing along axis
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let indices = Array::from_vec(vec![2, 0, 1]);
/// let result = take(&arr, &indices, Some(1)).unwrap();
/// // Takes columns [2, 0, 1] from each row
/// ```
pub fn take<T: Clone + Zero>(
    array: &Array<T>,
    indices: &Array<usize>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Flatten array and take elements
            let flat = array.to_vec();
            let idx_vec = indices.to_vec();

            // Validate indices
            for &idx in &idx_vec {
                if idx >= flat.len() {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "index {} is out of bounds for flattened array of size {}",
                        idx,
                        flat.len()
                    )));
                }
            }

            // Take elements
            let result: Vec<T> = idx_vec.iter().map(|&idx| flat[idx].clone()).collect();

            // Result has same shape as indices
            Ok(Array::from_vec(result).reshape(&indices.shape()))
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

            let idx_vec = indices.to_vec();

            // Validate indices
            for &idx in &idx_vec {
                if idx >= shape[ax] {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, ax, shape[ax]
                    )));
                }
            }

            // Calculate result shape
            let mut result_shape = shape.clone();
            result_shape[ax] = idx_vec.len();

            let mut result_data = Vec::with_capacity(result_shape.iter().product());

            // Iterate through all positions
            let mut current_pos = vec![0; shape.len()];
            let total_out: usize = result_shape.iter().product();

            for _ in 0..total_out {
                // Map position in result to position in source
                let mut source_pos = current_pos.clone();
                source_pos[ax] = idx_vec[current_pos[ax]];

                let value = array.get(&source_pos)?;
                result_data.push(value);

                // Increment position
                let mut carry = true;
                for dim in (0..result_shape.len()).rev() {
                    if carry {
                        current_pos[dim] += 1;
                        carry = current_pos[dim] >= result_shape[dim];
                        if carry {
                            current_pos[dim] = 0;
                        }
                    }
                }
            }

            Ok(Array::from_vec(result_data).reshape(&result_shape))
        }
    }
}

/// Multi-dimensional fancy indexing using coordinate arrays
///
/// Select elements at specific coordinates specified by arrays of indices.
/// This is equivalent to `array[indices[0], indices[1], ...]` in NumPy.
///
/// # Arguments
/// * `array` - Input array
/// * `indices` - Slice of index arrays, one per dimension
///
/// # Returns
/// * `Result<Array<T>>` - 1-D array with selected elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::fancy_index;
///
/// let arr = Array::from_vec(vec![
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9
/// ]).reshape(&[3, 3]);
///
/// // Select elements at (0,0), (1,1), (2,2) - the diagonal
/// let row_idx = Array::from_vec(vec![0, 1, 2]);
/// let col_idx = Array::from_vec(vec![0, 1, 2]);
/// let result = fancy_index(&arr, &[row_idx, col_idx]).unwrap();
/// // Returns [1, 5, 9]
///
/// // Select elements at (0,2), (2,0)
/// let row_idx = Array::from_vec(vec![0, 2]);
/// let col_idx = Array::from_vec(vec![2, 0]);
/// let result = fancy_index(&arr, &[row_idx, col_idx]).unwrap();
/// // Returns [3, 7]
/// ```
pub fn fancy_index<T: Clone + Zero>(
    array: &Array<T>,
    indices: &[Array<usize>],
) -> Result<Array<T>> {
    let shape = array.shape();

    if indices.is_empty() {
        return Err(NumRs2Error::ValueError(
            "indices array cannot be empty".to_string(),
        ));
    }

    if indices.len() != shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "number of index arrays ({}) must match array dimensions ({})",
            indices.len(),
            shape.len()
        )));
    }

    // All index arrays must have the same shape
    let idx_shape = indices[0].shape();
    for idx_arr in &indices[1..] {
        if idx_arr.shape() != idx_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: idx_shape.clone(),
                actual: idx_arr.shape(),
            });
        }
    }

    let num_elements = indices[0].size();
    let mut result_data = Vec::with_capacity(num_elements);

    // Convert all index arrays to vectors
    let idx_vecs: Vec<Vec<usize>> = indices.iter().map(|arr| arr.to_vec()).collect();

    // For each element position
    for i in 0..num_elements {
        // Build coordinate from index arrays
        let mut coord = Vec::with_capacity(shape.len());
        for (dim, idx_vec) in idx_vecs.iter().enumerate() {
            let idx = idx_vec[i];

            // Validate index
            if idx >= shape[dim] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "index {} is out of bounds for dimension {} with size {}",
                    idx, dim, shape[dim]
                )));
            }

            coord.push(idx);
        }

        // Get element at coordinate
        let value = array.get(&coord)?;
        result_data.push(value);
    }

    // Result has same shape as the index arrays
    Ok(Array::from_vec(result_data).reshape(&idx_shape))
}

/// Boolean indexing convenience method
///
/// Extract elements from array using a boolean mask. This is a convenience
/// wrapper around `extract()` that returns a flattened array.
///
/// # Arguments
/// * `array` - Input array
/// * `mask` - Boolean mask with same shape as array
///
/// # Returns
/// * `Result<Array<T>>` - 1-D array with elements where mask is true
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::boolean_index;
///
/// let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
/// let mask = Array::from_vec(vec![true, false, true, false, true]);
/// let result = boolean_index(&arr, &mask).unwrap();
/// // Returns [10, 30, 50]
///
/// // Works with comparisons
/// let arr = Array::from_vec(vec![1, 5, 3, 8, 2]);
/// let mask = arr.map(|x| x > 3);  // [false, true, false, true, false]
/// let result = boolean_index(&arr, &mask).unwrap();
/// // Returns [5, 8]
/// ```
pub fn boolean_index<T: Clone>(array: &Array<T>, mask: &Array<bool>) -> Result<Array<T>> {
    extract(array, mask)
}

/// Choose elements from arrays based on conditions
///
/// Given a list of conditions and a list of choices, return an array drawn
/// from elements in choices, based on conditions. This is similar to NumPy's
/// `select()` function.
///
/// # Arguments
/// * `conditions` - List of boolean arrays. Each must have same shape
/// * `choices` - List of arrays to choose from. Each must match conditions shape
/// * `default` - Default value when no condition is met
///
/// # Returns
/// * `Result<Array<T>>` - Array with elements chosen based on conditions
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::select;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
///
/// // Condition 1: x < 3 -> return x * 10
/// let cond1 = arr.map(|x| x < 3);
/// let choice1 = arr.map(|x| x * 10);
///
/// // Condition 2: x >= 3 -> return x * 100
/// let cond2 = arr.map(|x| x >= 3);
/// let choice2 = arr.map(|x| x * 100);
///
/// let result = select(&[cond1, cond2], &[choice1, choice2], 0).unwrap();
/// // Returns [10, 20, 300, 400, 500]
/// ```
pub fn select<T: Clone>(
    conditions: &[Array<bool>],
    choices: &[Array<T>],
    default: T,
) -> Result<Array<T>> {
    if conditions.is_empty() {
        return Err(NumRs2Error::ValueError(
            "conditions array cannot be empty".to_string(),
        ));
    }

    if conditions.len() != choices.len() {
        return Err(NumRs2Error::ValueError(format!(
            "number of conditions ({}) must match number of choices ({})",
            conditions.len(),
            choices.len()
        )));
    }

    // All conditions and choices must have same shape
    let shape = conditions[0].shape();
    for cond in &conditions[1..] {
        if cond.shape() != shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.clone(),
                actual: cond.shape(),
            });
        }
    }

    for choice in choices {
        if choice.shape() != shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.clone(),
                actual: choice.shape(),
            });
        }
    }

    // Convert to vectors for easier access
    let cond_vecs: Vec<Vec<bool>> = conditions.iter().map(|c| c.to_vec()).collect();
    let choice_vecs: Vec<Vec<T>> = choices.iter().map(|c| c.to_vec()).collect();

    let num_elements = conditions[0].size();
    let mut result_data = Vec::with_capacity(num_elements);

    // For each element position
    for i in 0..num_elements {
        let mut selected = false;

        // Check conditions in order
        for (cond_vec, choice_vec) in cond_vecs.iter().zip(choice_vecs.iter()) {
            if cond_vec[i] {
                result_data.push(choice_vec[i].clone());
                selected = true;
                break;
            }
        }

        // If no condition matched, use default
        if !selected {
            result_data.push(default.clone());
        }
    }

    Ok(Array::from_vec(result_data).reshape(&shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_1d() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
        let indices = Array::from_vec(vec![0, 2, 4, 1]);

        let result = take(&arr, &indices, None).unwrap();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.to_vec(), vec![10, 30, 50, 20]);
    }

    #[test]
    fn test_take_2d_no_axis() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let indices = Array::from_vec(vec![0, 3, 5]);

        let result = take(&arr, &indices, None).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![1, 4, 6]);
    }

    #[test]
    fn test_take_along_axis() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let indices = Array::from_vec(vec![2, 0, 1]);

        let result = take(&arr, &indices, Some(1)).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: [1, 2, 3] -> indices [2, 0, 1] -> [3, 1, 2]
        // Row 1: [4, 5, 6] -> indices [2, 0, 1] -> [6, 4, 5]
        assert_eq!(result.to_vec(), vec![3, 1, 2, 6, 4, 5]);
    }

    #[test]
    fn test_take_out_of_bounds() {
        let arr = Array::from_vec(vec![1, 2, 3]);
        let indices = Array::from_vec(vec![0, 5]); // 5 is out of bounds

        let result = take(&arr, &indices, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fancy_index_diagonal() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

        let row_idx = Array::from_vec(vec![0, 1, 2]);
        let col_idx = Array::from_vec(vec![0, 1, 2]);

        let result = fancy_index(&arr, &[row_idx, col_idx]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![1, 5, 9]);
    }

    #[test]
    fn test_fancy_index_arbitrary_coords() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

        let row_idx = Array::from_vec(vec![0, 2, 1]);
        let col_idx = Array::from_vec(vec![2, 0, 1]);

        let result = fancy_index(&arr, &[row_idx, col_idx]).unwrap();

        assert_eq!(result.shape(), &[3]);
        // (0,2) -> 3, (2,0) -> 7, (1,1) -> 5
        assert_eq!(result.to_vec(), vec![3, 7, 5]);
    }

    #[test]
    fn test_fancy_index_2d_indices() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50, 60]).reshape(&[2, 3]);

        let row_idx = Array::from_vec(vec![0, 1, 0, 1]).reshape(&[2, 2]);
        let col_idx = Array::from_vec(vec![0, 1, 2, 2]).reshape(&[2, 2]);

        let result = fancy_index(&arr, &[row_idx, col_idx]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        // (0,0)->10, (1,1)->50, (0,2)->30, (1,2)->60
        assert_eq!(result.to_vec(), vec![10, 50, 30, 60]);
    }

    #[test]
    fn test_fancy_index_mismatched_shapes() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);

        let row_idx = Array::from_vec(vec![0, 1]);
        let col_idx = Array::from_vec(vec![0, 1, 2]); // Different shape

        let result = fancy_index(&arr, &[row_idx, col_idx]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fancy_index_out_of_bounds() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);

        let row_idx = Array::from_vec(vec![0, 3]); // 3 is out of bounds
        let col_idx = Array::from_vec(vec![0, 0]);

        let result = fancy_index(&arr, &[row_idx, col_idx]);
        assert!(result.is_err());
    }

    #[test]
    fn test_boolean_index_simple() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
        let mask = Array::from_vec(vec![true, false, true, false, true]);

        let result = boolean_index(&arr, &mask).unwrap();

        assert_eq!(result.to_vec(), vec![10, 30, 50]);
    }

    #[test]
    fn test_boolean_index_with_comparison() {
        let arr = Array::from_vec(vec![1, 5, 3, 8, 2]);
        let mask = arr.map(|x| x > 3);

        let result = boolean_index(&arr, &mask).unwrap();

        assert_eq!(result.to_vec(), vec![5, 8]);
    }

    #[test]
    fn test_boolean_index_2d() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let mask = Array::from_vec(vec![true, false, true, false, true, false]).reshape(&[2, 3]);

        let result = boolean_index(&arr, &mask).unwrap();

        assert_eq!(result.to_vec(), vec![1, 3, 5]);
    }

    #[test]
    fn test_boolean_index_all_false() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![false; 5]);

        let result = boolean_index(&arr, &mask).unwrap();

        assert_eq!(result.size(), 0);
    }

    #[test]
    fn test_boolean_index_all_true() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![true; 5]);

        let result = boolean_index(&arr, &mask).unwrap();

        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_select_simple() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);

        let cond1 = arr.map(|x| x < 3);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = arr.map(|x| x >= 3);
        let choice2 = arr.map(|x| x * 100);

        let result = select(&[cond1, cond2], &[choice1, choice2], 0).unwrap();

        assert_eq!(result.to_vec(), vec![10, 20, 300, 400, 500]);
    }

    #[test]
    fn test_select_with_default() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);

        // Only one condition that matches some elements
        let cond = arr.map(|x| x > 3);
        let choice = arr.map(|x| x * 10);

        let result = select(&[cond], &[choice], -1).unwrap();

        // Elements > 3 get multiplied by 10, others get -1
        assert_eq!(result.to_vec(), vec![-1, -1, -1, 40, 50]);
    }

    #[test]
    fn test_select_multiple_conditions() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);

        let cond1 = arr.map(|x| x == 1);
        let choice1 = Array::from_vec(vec![100, 100, 100, 100, 100]);

        let cond2 = arr.map(|x| x == 3);
        let choice2 = Array::from_vec(vec![300, 300, 300, 300, 300]);

        let cond3 = arr.map(|x| x == 5);
        let choice3 = Array::from_vec(vec![500, 500, 500, 500, 500]);

        let result = select(&[cond1, cond2, cond3], &[choice1, choice2, choice3], 0).unwrap();

        assert_eq!(result.to_vec(), vec![100, 0, 300, 0, 500]);
    }

    #[test]
    fn test_select_2d() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);

        let cond1 = arr.map(|x| x < 3);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = arr.map(|x| x >= 3);
        let choice2 = arr.map(|x| x * 100);

        let result = select(&[cond1, cond2], &[choice1, choice2], 0).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![10, 20, 300, 400]);
    }

    #[test]
    fn test_select_mismatched_lengths() {
        let arr = Array::from_vec(vec![1, 2, 3]);

        let cond1 = arr.map(|x| x < 2);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = arr.map(|x| x >= 2);
        let choice2 = arr.map(|x| x * 100);

        // 2 conditions but 3 choices
        let choice3 = arr.map(|x| x * 1000);

        let result = select(&[cond1, cond2], &[choice1, choice2, choice3], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_mismatched_shapes() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);

        let cond1 = arr.map(|x| x < 3);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = Array::from_vec(vec![true, false]); // Wrong shape
        let choice2 = arr.map(|x| x * 100);

        let result = select(&[cond1, cond2], &[choice1, choice2], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_combined_indexing_take_and_boolean() {
        // First use take to reorder, then boolean mask
        let arr = Array::from_vec(vec![5, 2, 8, 1, 9, 3]);
        let indices = Array::from_vec(vec![0, 2, 4]); // [5, 8, 9]

        let reordered = take(&arr, &indices, None).unwrap();
        let mask = reordered.map(|x| x > 7); // [false, true, true]

        let result = boolean_index(&reordered, &mask).unwrap();

        assert_eq!(result.to_vec(), vec![8, 9]);
    }

    #[test]
    fn test_take_empty_indices() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let indices: Array<usize> = Array::from_vec(vec![]);

        let result = take(&arr, &indices, None).unwrap();

        assert_eq!(result.size(), 0);
    }

    #[test]
    fn test_take_repeated_indices() {
        let arr = Array::from_vec(vec![10, 20, 30]);
        let indices = Array::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let result = take(&arr, &indices, None).unwrap();

        assert_eq!(result.to_vec(), vec![10, 10, 20, 20, 30, 30]);
    }
}
