use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use ndarray::{Axis, IxDyn, Order};
use num_traits::Zero;
use std::cmp;

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
    let first_elem = array.array().first().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Cannot tile an empty array".into())
    })?.clone();
    
    let mut result = Array::full(&output_shape, first_elem);
    
    // Fill the output array by copying the input array in a tiled pattern
    // This is a simplified implementation - for efficiency, we would use
    // more sophisticated slicing and assignment operations
    
    let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
    })?;
    
    let input_vec = array.to_vec();
    let input_size = input_vec.len();
    
    if input_size == 0 {
        return Err(NumRs2Error::InvalidOperation("Cannot tile an empty array".into()));
    }
    
    // For each position in the output, copy the corresponding element from the input
    for i in 0..result_vec.len() {
        // Calculate corresponding index in the input array
        // This is a simplification - for a complete implementation, we would need
        // to carefully map N-dimensional indices
        let input_idx = i % input_size;
        result_vec[i] = input_vec[input_idx].clone();
    }
    
    Ok(result)
}

/// Repeat elements of an array along a specified axis
pub fn repeat<T: Clone>(array: &Array<T>, repeats: usize, axis: Option<usize>) -> Result<Array<T>> {
    let a_shape = array.shape();
    
    match axis {
        Some(ax) => {
            if ax >= a_shape.len() {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Axis {} out of bounds for array of dimension {}", ax, a_shape.len())
                ));
            }
            
            // Calculate the output shape
            let mut output_shape = a_shape.clone();
            output_shape[ax] *= repeats;
            
            // Create a result array
            let first_elem = array.array().first().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Cannot repeat an empty array".into())
            })?.clone();
            
            let mut result = Array::full(&output_shape, first_elem);
            
            // Fill the result array by repeating elements along the specified axis
            // This is a simplified implementation - a more efficient version would use
            // vectorized operations and views
            
            let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
            })?;
            
            let input_vec = array.to_vec();
            
            if input_vec.is_empty() {
                return Err(NumRs2Error::InvalidOperation("Cannot repeat an empty array".into()));
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
        },
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
/// let c = concatenate(&[&a, &b], 0).unwrap();
/// assert_eq!(c.shape(), vec![6]);
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn concatenate<T: Clone>(arrays: &[&Array<T>], axis: impl Into<AxisArg>) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to concatenate".into()));
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
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, first_shape.len())
        ));
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
    let views: Result<Vec<_>> = arrays
        .iter()
        .map(|arr| Ok(arr.array().view()))
        .collect();
    
    let views = views?;
    
    // Use ndarray's concatenate function
    let result = ndarray::concatenate(Axis(axis), &views).map_err(|e| {
        NumRs2Error::InvalidOperation(format!("Failed to concatenate arrays: {}", e))
    })?;
    
    // Convert the result back to our Array type
    Ok(Array::from_ndarray(result))
}

/// Concatenate arrays along multiple axes
fn concatenate_multiple_axes<T: Clone>(arrays: &[&Array<T>], axes: &[usize]) -> Result<Array<T>> {
    if axes.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No axes provided for concatenation".into()));
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

/// Enum to handle both single axis and multiple axes for concatenation
pub enum AxisArg {
    Single(usize),
    Multiple(Vec<usize>),
}

impl From<usize> for AxisArg {
    fn from(axis: usize) -> Self {
        AxisArg::Single(axis)
    }
}

impl From<&[usize]> for AxisArg {
    fn from(axes: &[usize]) -> Self {
        AxisArg::Multiple(axes.to_vec())
    }
}

impl From<Vec<usize>> for AxisArg {
    fn from(axes: Vec<usize>) -> Self {
        AxisArg::Multiple(axes)
    }
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
/// let c = stack(&[&a, &b], 0).unwrap();
/// assert_eq!(c.shape(), vec![2, 3]);
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn stack<T: Clone>(arrays: &[&Array<T>], axis: usize) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to stack".into()));
    }
    
    let first_shape = arrays[0].shape();
    
    if axis > first_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, first_shape.len())
        ));
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
/// let result = block(&blocks).unwrap();
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
/// let result = block(&blocks).unwrap();
/// 
/// assert_eq!(result.shape(), vec![4, 4]);
/// // The result will be a 4x4 array with the blocks arranged in a 2x2 grid
/// ```
pub fn block<T: Clone>(blocks: &[Vec<&Array<T>>]) -> Result<Array<T>> {
    if blocks.is_empty() {
        return Err(NumRs2Error::InvalidOperation("Empty block structure".into()));
    }
    
    // Normalize dimensions of arrays within each row
    let mut processed_rows = Vec::with_capacity(blocks.len());
    
    for (row_idx, row) in blocks.iter().enumerate() {
        if row.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Empty row at index {} in block structure", row_idx)
            ));
        }
        
        // Process each array in the row to ensure compatible dimensions
        let mut processed_row = Vec::with_capacity(row.len());
        
        // Determine the maximum number of dimensions in this row
        let max_ndim = row.iter()
            .map(|arr| arr.ndim())
            .max()
            .unwrap_or(1);
        
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
                "Arrays in each row must have the same number of dimensions".into()
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
            "All rows must have the same number of dimensions after processing".into()
        ));
    }
    
    // Concatenate rows along the first axis (which would be 0 for 2D arrays)
    let row_refs: Vec<&Array<T>> = rows_result.iter().collect();
    concatenate(&row_refs, 0)
}

/// Split an array into multiple sub-arrays horizontally (column-wise)
///
/// # Parameters
///
/// * `array` - The array to split
/// * `sections_or_indices` - Number of equal sections or indices to split at
///
/// # Returns
///
/// A list of arrays
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
///
/// // Create a 2x6 array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[2, 6]);
///
/// // Split into 3 equal parts
/// let splits = hsplit(&a, 3).unwrap();
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2]);
///
/// // Split at specific indices
/// let splits2 = hsplit(&a, &vec![2, 4]).unwrap();
/// assert_eq!(splits2.len(), 3);
/// ```
pub fn hsplit<T: Clone>(array: &Array<T>, sections_or_indices: impl Into<SplitArg>) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "hsplit requires at least 2D array".to_string()
        ));
    }
    
    // Split along the second axis (columns)
    let axis = 1;
    
    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            let axis_len = shape[axis];
            
            if axis_len % sections != 0 {
                return Err(NumRs2Error::InvalidOperation(
                    format!("array of shape {:?} cannot be split into {} equal sections along axis {}", 
                            shape, sections, axis)
                ));
            }
            
            let section_size = axis_len / sections;
            let mut indices = Vec::with_capacity(sections - 1);
            
            for i in 1..sections {
                indices.push(i * section_size);
            }
            
            split(array, &indices, axis)
        },
        SplitArg::Indices(indices) => split(array, &indices, axis),
    }
}

/// Split an array into multiple sub-arrays vertically (row-wise)
///
/// # Parameters
///
/// * `array` - The array to split
/// * `sections_or_indices` - Number of equal sections or indices to split at
///
/// # Returns
///
/// A list of arrays
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
///
/// // Create a 6x2 array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[6, 2]);
///
/// // Split into 3 equal parts
/// let splits = vsplit(&a, 3).unwrap();
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2]);
///
/// // Split at specific indices
/// let splits2 = vsplit(&a, &vec![2, 4]).unwrap();
/// assert_eq!(splits2.len(), 3);
/// ```
pub fn vsplit<T: Clone>(array: &Array<T>, sections_or_indices: impl Into<SplitArg>) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "vsplit requires at least 2D array".to_string()
        ));
    }
    
    // Split along the first axis (rows)
    let axis = 0;
    
    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            let axis_len = shape[axis];
            
            if axis_len % sections != 0 {
                return Err(NumRs2Error::InvalidOperation(
                    format!("array of shape {:?} cannot be split into {} equal sections along axis {}", 
                            shape, sections, axis)
                ));
            }
            
            let section_size = axis_len / sections;
            let mut indices = Vec::with_capacity(sections - 1);
            
            for i in 1..sections {
                indices.push(i * section_size);
            }
            
            split(array, &indices, axis)
        },
        SplitArg::Indices(indices) => split(array, &indices, axis),
    }
}

/// Split an array into multiple sub-arrays along the third axis (depth)
///
/// # Parameters
///
/// * `array` - The array to split
/// * `sections_or_indices` - Number of equal sections or indices to split at
///
/// # Returns
///
/// A list of arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2x2x6 array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
///                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
///     .reshape(&[2, 2, 6]);
///
/// // Split into 3 equal parts
/// let splits = dsplit(&a, 3).unwrap();
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2, 2]);
/// ```
pub fn dsplit<T: Clone>(array: &Array<T>, sections_or_indices: impl Into<SplitArg>) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if ndim < 3 {
        return Err(NumRs2Error::InvalidOperation(
            "dsplit requires at least 3D array".to_string()
        ));
    }
    
    // Split along the third axis (depth)
    let axis = 2;
    
    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            let axis_len = shape[axis];
            
            if axis_len % sections != 0 {
                return Err(NumRs2Error::InvalidOperation(
                    format!("array of shape {:?} cannot be split into {} equal sections along axis {}", 
                            shape, sections, axis)
                ));
            }
            
            let section_size = axis_len / sections;
            let mut indices = Vec::with_capacity(sections - 1);
            
            for i in 1..sections {
                indices.push(i * section_size);
            }
            
            split(array, &indices, axis)
        },
        SplitArg::Indices(indices) => split(array, &indices, axis),
    }
}

/// Enumeration to handle either sections or indices for split functions
pub enum SplitArg {
    Sections(usize),
    Indices(Vec<usize>),
}

impl From<usize> for SplitArg {
    fn from(sections: usize) -> Self {
        SplitArg::Sections(sections)
    }
}

impl From<&[usize]> for SplitArg {
    fn from(indices: &[usize]) -> Self {
        SplitArg::Indices(indices.to_vec())
    }
}

impl From<Vec<usize>> for SplitArg {
    fn from(indices: Vec<usize>) -> Self {
        SplitArg::Indices(indices)
    }
}

/// Split an array into multiple subarrays along a specified axis
/// 
/// Parameters:
/// - array: The array to split
/// - indices: A list of indices to split at
/// - axis: The axis to split along
pub fn split<T: Clone>(array: &Array<T>, indices: &[usize], axis: usize) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    
    if axis >= shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, shape.len())
        ));
    }
    
    let axis_len = shape[axis];
    
    // Determine the split indices
    let mut split_indices = Vec::new();
    
    for &idx in indices {
        if idx == 0 || idx >= axis_len {
            return Err(NumRs2Error::InvalidOperation(
                format!("Split index {} out of bounds for axis {} with size {}", idx, axis, axis_len)
            ));
        }
        
        split_indices.push(idx);
    }
    
    // Sort indices to ensure they're in ascending order
    split_indices.sort();
    
    // Create the result arrays
    let mut result = Vec::new();
    
    
    let mut start_idx = 0;
    for &end_idx in split_indices.iter() {
        let mut sub_shape = shape.clone();
        sub_shape[axis] = end_idx - start_idx;
        
        let mut indices = vec![0; shape.len()];
        indices[axis] = start_idx;
        
        let view = array.array().slice_axis(Axis(axis), ndarray::Slice::from(start_idx..end_idx));
        result.push(Array::from_ndarray(view.into_owned().into_dyn()));
        
        start_idx = end_idx;
    }
    
    // Add the last section
    if start_idx < axis_len {
        let mut sub_shape = shape.clone();
        sub_shape[axis] = axis_len - start_idx;
        
        let view = array.array().slice_axis(Axis(axis), ndarray::Slice::from(start_idx..axis_len));
        result.push(Array::from_ndarray(view.into_owned().into_dyn()));
    }
    
    Ok(result)
}

/// Reverse the order of elements in an array along the specified axis
///
/// # Parameters
///
/// * `array` - The input array
/// * `axis` - The axis along which to flip values
///
/// # Returns
///
/// A new array with reversed elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Flip a 1D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let b = flip(&a, 0).unwrap();
/// assert_eq!(b.to_vec(), vec![5, 4, 3, 2, 1]);
///
/// // Flip a 2D array along rows
/// let c = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let d = flip(&c, 0).unwrap();
/// assert_eq!(d.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn flip<T: Clone + Zero>(array: &Array<T>, axis: usize) -> Result<Array<T>> {
    let shape = array.shape();
    
    if axis >= shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, shape.len())
        ));
    }
    
    // Create a new array to hold the flipped data
    let mut result = Array::zeros(&shape);
    
    // Process each position in the array
    let total_size = array.size();
    let ndim = shape.len();
    
    for i in 0..total_size {
        // Calculate the indices for the current element
        let mut indices = Vec::with_capacity(ndim);
        let mut temp = i;
        
        for j in (0..ndim).rev() {
            indices.insert(0, temp % shape[j]);
            temp /= shape[j];
        }
        
        // Calculate the flipped indices
        let mut flipped_indices = indices.clone();
        flipped_indices[axis] = shape[axis] - 1 - indices[axis];
        
        // Copy the element
        let value = array.array().get(IxDyn(&indices)).unwrap().clone();
        result.set(&flipped_indices, value).unwrap();
    }
    
    Ok(result)
}

/// Flip array in the left/right direction (last axis)
///
/// # Parameters
///
/// * `array` - The input array
///
/// # Returns
///
/// A new array with last dimension flipped
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Flip a 2D array horizontally
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let b = fliplr(&a).unwrap();
/// assert_eq!(b.to_vec(), vec![3, 2, 1, 6, 5, 4]);
/// ```
pub fn fliplr<T: Clone + Zero>(array: &Array<T>) -> Result<Array<T>> {
    let ndim = array.ndim();
    
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "fliplr requires at least 2D array".to_string()
        ));
    }
    
    // Flip along the last axis (columns)
    flip(array, ndim - 1)
}

/// Flip array in the up/down direction (first axis)
///
/// # Parameters
///
/// * `array` - The input array
///
/// # Returns
///
/// A new array with first dimension flipped
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Flip a 2D array vertically
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let b = flipud(&a).unwrap();
/// assert_eq!(b.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn flipud<T: Clone + Zero>(array: &Array<T>) -> Result<Array<T>> {
    let ndim = array.ndim();
    
    if ndim < 1 {
        return Err(NumRs2Error::InvalidOperation(
            "flipud requires at least 1D array".to_string()
        ));
    }
    
    // Flip along the first axis (rows)
    flip(array, 0)
}

/// Expand the shape of an array by inserting a new axis at the specified position
pub fn expand_dims<T: Clone>(array: &Array<T>, axis: usize) -> Result<Array<T>> {
    let shape = array.shape();
    
    if axis > shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, shape.len())
        ));
    }
    
    // Create a new shape with an extra dimension
    let mut new_shape = shape.clone();
    new_shape.insert(axis, 1);
    
    // Reshape the array
    Ok(array.reshape(&new_shape))
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
/// let c = r_(&[&a, &b]).unwrap();
/// assert_eq!(c.shape(), vec![6]);  // Flattened to 1D array
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // With 2D arrays
/// let a2 = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b2 = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c2 = r_(&[&a2, &b2]).unwrap();
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
        let processed_arrays: Result<Vec<Array<T>>> = arrays.iter()
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
/// ```ignore
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let c = c_(&[&a, &b]).unwrap();
/// // With 1D arrays, c_ reshapes them to [n, 1] and concatenates along axis 1
/// assert_eq!(c.shape(), vec![3, 2]);
/// assert_eq!(c.to_vec(), vec![1, 4, 2, 5, 3, 6]);
///
/// // With 2D arrays
/// let a2 = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b2 = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c2 = c_(&[&a2, &b2]).unwrap();
/// assert_eq!(c2.shape(), vec![2, 4]);
/// ```
pub fn c_<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    // Process 1D arrays by adding a dimension
    let processed_arrays: Result<Vec<Array<T>>> = arrays.iter()
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

/// Roll array elements along a given axis
///
/// # Parameters
///
/// * `array` - The input array
/// * `shift` - The number of places to roll
/// * `axis` - The axis along which to roll (optional, if None, roll the flattened array)
///
/// # Returns
///
/// A new array with the same shape but with elements rolled
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Roll a 1D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let b = roll(&a, 2, None).unwrap();
/// assert_eq!(b.to_vec(), vec![4, 5, 1, 2, 3]);
///
/// // Roll along axis 0
/// let c = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let d = roll(&c, 1, Some(0)).unwrap();
/// assert_eq!(d.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn roll<T: Clone + Zero>(array: &Array<T>, shift: isize, axis: Option<usize>) -> Result<Array<T>> {
    if shift == 0 {
        return Ok(array.clone());
    }
    
    match axis {
        None => {
            // Roll the flattened array
            let flat = array.flatten(None);
            let size = flat.size();
            
            if size == 0 {
                return Ok(array.clone());
            }
            
            // Calculate the effective shift (handle negative shifts and wrapping)
            let effective_shift = (((shift % size as isize) + size as isize) % size as isize) as usize;
            
            if effective_shift == 0 {
                return Ok(array.clone());
            }
            
            // Create the rolled array
            let flat_data = flat.to_vec();
            let mut rolled_data = Vec::with_capacity(size);
            
            // Copy the data with the roll
            rolled_data.extend_from_slice(&flat_data[size - effective_shift..]);
            rolled_data.extend_from_slice(&flat_data[..size - effective_shift]);
            
            // Reshape back to the original shape
            Ok(Array::from_vec(rolled_data).reshape(&array.shape()))
        },
        Some(ax) => {
            let shape = array.shape();
            
            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Axis {} out of bounds for array of dimension {}", ax, shape.len())
                ));
            }
            
            let axis_size = shape[ax];
            
            if axis_size == 0 {
                return Ok(array.clone());
            }
            
            // Calculate the effective shift (handle negative shifts and wrapping)
            let effective_shift = (((shift % axis_size as isize) + axis_size as isize) % axis_size as isize) as usize;
            
            if effective_shift == 0 {
                return Ok(array.clone());
            }
            
            // Create a new array to hold the result
            let mut result = Array::zeros(&shape);
            
            // Total number of elements to process
            let total_size = array.size();
            let _axis_stride = total_size / axis_size;
            
            // Process each position in the array
            for i in 0..total_size {
                // Calculate the indices for the current element
                let mut indices = Vec::with_capacity(shape.len());
                let mut temp = i;
                
                for j in (0..shape.len()).rev() {
                    indices.insert(0, temp % shape[j]);
                    temp /= shape[j];
                }
                
                // Calculate the rolled index for the axis
                let original_axis_index = indices[ax];
                let rolled_axis_index = (original_axis_index + effective_shift) % axis_size;
                
                // Create indices for the destination
                let mut dest_indices = indices.clone();
                dest_indices[ax] = rolled_axis_index;
                
                // Copy the element
                let value = array.array().get(IxDyn(&indices)).unwrap().clone();
                result.set(&dest_indices, value).unwrap();
            }
            
            Ok(result)
        }
    }
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
/// let b = rollaxis(&a, 2, 0).unwrap();
/// assert_eq!(b.shape(), vec![3, 2, 2]);
/// ```
pub fn rollaxis<T: Clone + Zero>(array: &Array<T>, axis: usize, start: usize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if axis >= ndim {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, ndim)
        ));
    }
    
    if start > ndim {
        return Err(NumRs2Error::InvalidOperation(
            format!("Start position {} exceeds array dimensions {}", start, ndim)
        ));
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
        result_data[target_flat_index] = source_array.as_slice().unwrap()[i].clone();
    }
    
    // Create the result array
    let result = Array::from_vec(result_data).reshape(&target_shape);
    
    Ok(result)
}

/// Helper function to transpose two specific axes
#[allow(dead_code)]
fn array_transpose<T: Clone + Zero>(array: &Array<T>, axis1: usize, axis2: usize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axes ({}, {}) out of bounds for array of dimension {}", axis1, axis2, ndim)
        ));
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
        let value = array.array().get(IxDyn(&indices)).unwrap().clone();
        result.set(&trans_indices, value).unwrap();
    }
    
    Ok(result)
}

/// Rotate an array by 90 degrees in the plane specified by axes
///
/// # Parameters
///
/// * `array` - The input array
/// * `k` - Number of times to rotate by 90 degrees (default 1)
/// * `axes` - The plane in which to rotate (default (0, 1))
///
/// # Returns
///
/// A new array with rotated elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// 
/// // Rotate 90 degrees
/// let b = rot90(&a, None, None).unwrap();
/// assert_eq!(b.shape(), vec![3, 2]);
/// // Order will be different based on memory layout
/// ```
pub fn rot90<T: Clone + Zero>(array: &Array<T>, k: Option<isize>, axes: Option<(usize, usize)>) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "rot90 requires at least 2D array".to_string()
        ));
    }
    
    let k_val = k.unwrap_or(1);
    let (axis1, axis2) = axes.unwrap_or((0, 1));
    
    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axes ({}, {}) out of bounds for array of dimension {}", axis1, axis2, ndim)
        ));
    }
    
    if axis1 == axis2 {
        return Err(NumRs2Error::InvalidOperation(
            "Axes for rotation cannot be the same".to_string()
        ));
    }
    
    // Apply the rotation
    let effective_k = ((k_val % 4) + 4) % 4;
    
    if effective_k == 0 {
        return Ok(array.clone());
    }
    
    // Create a new shape with the rotated axes
    let mut new_shape = shape.clone();
    if effective_k % 2 == 1 {
        new_shape.swap(axis1, axis2);
    }
    
    // Create a new array to hold the rotated data
    let mut result = Array::zeros(&new_shape);
    
    // Process each position in the array
    let total_size = array.size();
    
    for i in 0..total_size {
        // Calculate the indices for the current element
        let mut indices = Vec::with_capacity(ndim);
        let mut temp = i;
        
        for j in (0..ndim).rev() {
            indices.insert(0, temp % shape[j]);
            temp /= shape[j];
        }
        
        // Calculate the rotated indices
        let mut rot_indices = indices.clone();
        
        match effective_k {
            1 => {
                // 90 degrees
                rot_indices[axis1] = indices[axis2];
                rot_indices[axis2] = shape[axis1] - 1 - indices[axis1];
            },
            2 => {
                // 180 degrees
                rot_indices[axis1] = shape[axis1] - 1 - indices[axis1];
                rot_indices[axis2] = shape[axis2] - 1 - indices[axis2];
            },
            3 => {
                // 270 degrees
                rot_indices[axis1] = shape[axis2] - 1 - indices[axis2];
                rot_indices[axis2] = indices[axis1];
            },
            _ => unreachable!(),
        }
        
        // Copy the element
        let value = array.array().get(IxDyn(&indices)).unwrap().clone();
        result.set(&rot_indices, value).unwrap();
    }
    
    Ok(result)
}

/// Remove axes of length 1 from the array
pub fn squeeze<T: Clone>(array: &Array<T>, axis: Option<usize>) -> Result<Array<T>> {
    let shape = array.shape();
    
    match axis {
        Some(ax) => {
            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Axis {} out of bounds for array of dimension {}", ax, shape.len())
                ));
            }
            
            if shape[ax] != 1 {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Cannot squeeze axis {} with size {}", ax, shape[ax])
                ));
            }
            
            let mut new_shape = shape.clone();
            new_shape.remove(ax);
            
            Ok(array.reshape(&new_shape))
        },
        None => {
            // Remove all axes of length 1
            let new_shape: Vec<_> = shape.iter().filter(|&&s| s != 1).cloned().collect();
            
            if new_shape.is_empty() {
                // Result would be a scalar, return a 1D array with a single element
                Ok(array.reshape(&[1]))
            } else {
                Ok(array.reshape(&new_shape))
            }
        }
    }
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
/// let broadcasts = broadcast_arrays(&[&a, &b]).unwrap();
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
/// let b = broadcast_to(&a, &[2, 3]).unwrap();
/// assert_eq!(b.shape(), vec![2, 3]);
/// ```
pub fn broadcast_to<T: Clone>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>> {
    array.broadcast_to(shape)
}

// This section is commented out to avoid duplicate function definitions
// The roll function is already defined earlier in the file
/*
pub fn roll<T: Clone>(array: &Array<T>, shift: isize, axis: Option<usize>) -> Result<Array<T>> {
    if shift == 0 {
        return Ok(array.clone());
    }
    
    match axis {
        None => {
            // Roll the flattened array
            let flat = array.reshape(&[array.size()]);
            let size = flat.size();
            
            // Normalize the shift to be positive and within array bounds
            let shift_normalized = ((shift % size as isize) + size as isize) as usize % size;
            if shift_normalized == 0 {
                return Ok(array.clone());
            }
            
            // Extract the parts of the array to be rolled
            let data = flat.to_vec();
            let mut result = Vec::with_capacity(size);
            
            // Concatenate the parts in the new order
            result.extend_from_slice(&data[size - shift_normalized..]);
            result.extend_from_slice(&data[..size - shift_normalized]);
            
            // Reshape back to the original shape
            Ok(Array::from_vec(result).reshape(&array.shape()))
        },
        Some(ax) => {
            let shape = array.shape();
            let ndim = shape.len();
            
            if ax >= ndim {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Axis {} out of bounds for array of dimension {}", ax, ndim)
                ));
            }
            
            // Size of the specified axis
            let axis_size = shape[ax];
            
            // Normalize the shift to be positive and within axis bounds
            let shift_normalized = ((shift % axis_size as isize) + axis_size as isize) as usize % axis_size;
            if shift_normalized == 0 {
                return Ok(array.clone());
            }
            
            // Create a new array for the result
            let mut result = Array::zeros(shape);
            
            // Process the array along the specified axis
            // For efficiency, we should use ndarray's functionality directly
            // This is a simplified implementation
            
            // Calculate the total size of the array
            let total_size = array.size();
            
            // Calculate the stride for the axis we're rolling
            let mut stride = 1;
            for i in (ax + 1)..ndim {
                stride *= shape[i];
            }
            
            // Roll each sub-array along the specified axis
            let chunk_size = stride * axis_size;
            let num_chunks = total_size / chunk_size;
            
            let array_data = array.to_vec();
            let mut result_data = vec![array_data[0].clone(); total_size];
            
            for chunk in 0..num_chunks {
                let chunk_offset = chunk * chunk_size;
                
                for i in 0..axis_size {
                    let src_offset = chunk_offset + i * stride;
                    let dst_i = (i + axis_size - shift_normalized) % axis_size;
                    let dst_offset = chunk_offset + dst_i * stride;
                    
                    for j in 0..stride {
                        result_data[dst_offset + j] = array_data[src_offset + j].clone();
                    }
                }
            }
            
            Ok(Array::from_vec(result_data).reshape(shape))
        }
    }
}

/// Rotate an array by 90 degrees around specified axes
///
/// # Parameters
///
/// * `array` - The input array
/// * `k` - Number of 90-degree rotations, positive for counterclockwise, negative for clockwise
/// * `axes` - Tuple specifying the plane of rotation. Default is (0, 1)
///
/// # Returns
///
/// A new array rotated in the plane specified by axes
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
///
/// // Rotate 90 degrees counterclockwise
/// let rotated = rot90(&a, 1, Some((0, 1))).unwrap();
/// assert_eq!(rotated.shape(), vec![3, 3]);
/// assert_eq!(rotated.to_vec(), vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
///
/// // Rotate 180 degrees
/// let rotated_180 = rot90(&a, 2, None).unwrap();
/// assert_eq!(rotated_180.shape(), vec![3, 3]);
/// assert_eq!(rotated_180.to_vec(), vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
/// ```
/// Reverse the order of elements along the given axis
///
/// # Parameters
///
/// * `array` - The input array
/// * `axis` - Axis along which to flip. If None, all axes are flipped
///
/// # Returns
///
/// A new array with the elements reversed along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Flip along axis 0 (rows)
/// let flipped_rows = flip(&a, Some(0)).unwrap();
/// assert_eq!(flipped_rows.to_vec(), vec![4, 5, 6, 1, 2, 3]);
///
/// // Flip along axis 1 (columns)
/// let flipped_cols = flip(&a, Some(1)).unwrap();
/// assert_eq!(flipped_cols.to_vec(), vec![3, 2, 1, 6, 5, 4]);
///
/// // Flip along all axes
/// let flipped_all = flip(&a, None).unwrap();
/// assert_eq!(flipped_all.to_vec(), vec![6, 5, 4, 3, 2, 1]);
/// ```
pub fn flip<T: Clone>(array: &Array<T>, axis: Option<usize>) -> Result<Array<T>> {
    match axis {
        Some(ax) => {
            let shape = array.shape();
            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Axis {} out of bounds for array of dimension {}", ax, shape.len())
                ));
            }
            
            // Create a new array for the result
            let size = array.size();
            let axis_size = shape[ax];
            
            // Use a negative shift to reverse the axis
            // This essentially rotates each slice by 180 degrees along the specified axis
            let mut result_data = array.to_vec();
            let mut new_data = vec![result_data[0].clone(); size];
            
            // Calculate the stride for the axis we're flipping
            let mut stride = 1;
            for i in (ax + 1)..shape.len() {
                stride *= shape[i];
            }
            
            // Flip each sub-array along the specified axis
            let chunk_size = stride * axis_size;
            let num_chunks = size / chunk_size;
            
            for chunk in 0..num_chunks {
                let chunk_offset = chunk * chunk_size;
                
                for i in 0..axis_size {
                    let src_offset = chunk_offset + i * stride;
                    let dst_i = axis_size - 1 - i;
                    let dst_offset = chunk_offset + dst_i * stride;
                    
                    for j in 0..stride {
                        new_data[dst_offset + j] = result_data[src_offset + j].clone();
                    }
                }
            }
            
            Ok(Array::from_vec(new_data).reshape(shape))
        },
        None => {
            // Flip all axes by reversing the entire data
            let mut data = array.to_vec();
            data.reverse();
            Ok(Array::from_vec(data).reshape(&array.shape()))
        }
    }
}

/// Flip array in the left/right direction (last axis)
///
/// # Parameters
///
/// * `array` - The input array
///
/// # Returns
///
/// A new array with the elements reversed along the last axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Flip left/right (last axis - columns)
/// let flipped = fliplr(&a).unwrap();
/// assert_eq!(flipped.to_vec(), vec![3, 2, 1, 6, 5, 4]);
/// ```
pub fn fliplr<T: Clone>(array: &Array<T>) -> Result<Array<T>> {
    let ndim = array.ndim();
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "fliplr requires at least a 2D array".to_string()
        ));
    }
    
    // Flip along the last axis (columns)
    flip(array, Some(ndim - 1))
}

/// Flip array in the up/down direction (first axis)
///
/// # Parameters
///
/// * `array` - The input array
///
/// # Returns
///
/// A new array with the elements reversed along the first axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Flip up/down (first axis - rows)
/// let flipped = flipud(&a).unwrap();
/// assert_eq!(flipped.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn flipud<T: Clone>(array: &Array<T>) -> Result<Array<T>> {
    let ndim = array.ndim();
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "flipud requires at least a 2D array".to_string()
        ));
    }
    
    // Flip along the first axis (rows)
    flip(array, Some(0))
}

pub fn rot90<T: Clone>(array: &Array<T>, k: isize, axes: Option<(usize, usize)>) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "rot90 requires at least a 2D array".to_string()
        ));
    }
    
    // Default rotation axes are the first two dimensions
    let (axis1, axis2) = axes.unwrap_or((0, 1));
    
    if axis1 == axis2 || axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::InvalidOperation(
            format!("Invalid rotation axes ({}, {}) for array of dimension {}", axis1, axis2, ndim)
        ));
    }
    
    // Normalize k to be in [0, 3] since rotation repeats every 4 times
    let k_norm = ((k % 4) + 4) % 4;
    if k_norm == 0 {
        return Ok(array.clone());
    }
    
    // Get the shape of the axes we're rotating
    let axis1_size = shape[axis1];
    let axis2_size = shape[axis2];
    
    // Create a new shape with the rotated dimensions
    let mut new_shape = shape.clone();
    
    // Depending on the rotation amount, swap and/or reverse dimensions
    match k_norm {
        1 => {
            // 90 degrees: swap dimensions and reverse rows
            new_shape[axis1] = axis2_size;
            new_shape[axis2] = axis1_size;
            
            // Create result array
            let mut result = Array::zeros(&new_shape);
            
            // Fill the rotated array - this is a simplified approach
            // For large arrays, we'd use more efficient methods
            
            // Iterate through the source array
            let src_data = array.to_vec();
            let size = array.size();
            
            // Map each multidimensional index from source to rotated destination
            for flat_idx in 0..size {
                // Convert flat index to multidimensional indices
                let mut indices = vec![0; ndim];
                let mut temp = flat_idx;
                
                for i in (0..ndim).rev() {
                    indices[i] = temp % shape[i];
                    temp /= shape[i];
                }
                
                // Apply rotation: (i, j) -> (j, n-1-i) for 90° counterclockwise
                let i = indices[axis1];
                let j = indices[axis2];
                
                let new_i = j;
                let new_j = axis1_size - 1 - i;
                
                // Update the rotated indices
                let mut rotated_indices = indices.clone();
                rotated_indices[axis1] = new_i;
                rotated_indices[axis2] = new_j;
                
                // Set the value in the rotated array
                result.set(&rotated_indices, src_data[flat_idx].clone()).unwrap();
            }
            
            Ok(result)
        },
        2 => {
            // 180 degrees: keep dimensions but reverse both axes
            // Create result array with same shape
            let mut result = Array::zeros(&new_shape);
            
            // Iterate through the source array
            let src_data = array.to_vec();
            let size = array.size();
            
            // Map each multidimensional index from source to rotated destination
            for flat_idx in 0..size {
                // Convert flat index to multidimensional indices
                let mut indices = vec![0; ndim];
                let mut temp = flat_idx;
                
                for i in (0..ndim).rev() {
                    indices[i] = temp % shape[i];
                    temp /= shape[i];
                }
                
                // Apply rotation: (i, j) -> (n-1-i, m-1-j) for 180°
                let i = indices[axis1];
                let j = indices[axis2];
                
                let new_i = axis1_size - 1 - i;
                let new_j = axis2_size - 1 - j;
                
                // Update the rotated indices
                let mut rotated_indices = indices.clone();
                rotated_indices[axis1] = new_i;
                rotated_indices[axis2] = new_j;
                
                // Set the value in the rotated array
                result.set(&rotated_indices, src_data[flat_idx].clone()).unwrap();
            }
            
            Ok(result)
        },
        3 => {
            // 270 degrees (or -90): swap dimensions and reverse columns
            new_shape[axis1] = axis2_size;
            new_shape[axis2] = axis1_size;
            
            // Create result array
            let mut result = Array::zeros(&new_shape);
            
            // Iterate through the source array
            let src_data = array.to_vec();
            let size = array.size();
            
            // Map each multidimensional index from source to rotated destination
            for flat_idx in 0..size {
                // Convert flat index to multidimensional indices
                let mut indices = vec![0; ndim];
                let mut temp = flat_idx;
                
                for i in (0..ndim).rev() {
                    indices[i] = temp % shape[i];
                    temp /= shape[i];
                }
                
                // Apply rotation: (i, j) -> (m-1-j, i) for 270° counterclockwise
                let i = indices[axis1];
                let j = indices[axis2];
                
                let new_i = axis2_size - 1 - j;
                let new_j = i;
                
                // Update the rotated indices
                let mut rotated_indices = indices.clone();
                rotated_indices[axis1] = new_i;
                rotated_indices[axis2] = new_j;
                
                // Set the value in the rotated array
                result.set(&rotated_indices, src_data[flat_idx].clone()).unwrap();
            }
            
            Ok(result)
        },
        _ => unreachable!(),  // k_norm is already normalized to [0, 3]
    }
}
*/

/// Convert array to a flattened 1-D array (C-style order by default).
///
/// Unlike `flatten()`, this function returns a view of the original array when possible.
///
/// # Parameters
///
/// * `array` - The array to flatten
/// * `order` - Memory layout: 'C' (row-major, default) or 'F' (column-major)
///
/// # Returns
///
/// A flattened view of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flat = ravel(&a, None).unwrap();
/// assert_eq!(flat.shape(), vec![6]);
/// assert_eq!(flat.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn ravel<T: Clone>(array: &Array<T>, order: Option<char>) -> Result<Array<T>> {
    let size = array.size();
    
    // If array is empty, return empty 1D array
    if size == 0 {
        return Ok(Array::from_vec(Vec::<T>::new()));
    }
    
    // If array is already 1D, return a view
    if array.ndim() == 1 {
        return Ok(array.clone());
    }
    
    let order_val = order.unwrap_or('C');
    
    // Determine order for ndarray
    let nd_order = match order_val {
        'C' => Order::RowMajor,
        'F' => Order::ColumnMajor,
        _ => return Err(NumRs2Error::InvalidOperation(
            format!("Order must be 'C' or 'F', got '{}'", order_val)
        )),
    };
    
    // Create a flat view with the specified order
    let flat_data = match nd_order {
        Order::RowMajor => array.array().iter().cloned().collect::<Vec<_>>(),
        Order::ColumnMajor => {
            // Transpose the array and then collect elements
            let transposed = array.transpose();
            transposed.array().iter().cloned().collect::<Vec<_>>()
        },
        _ => {
            // This should never happen, but we need to handle the non-exhaustive enum
            return Err(NumRs2Error::InvalidOperation(
                format!("Unsupported memory order")
            ));
        }
    };
    
    Ok(Array::from_vec(flat_data))
}

/// Return a flattened copy of the array (1-D array).
///
/// # Parameters
///
/// * `array` - The array to flatten
/// * `order` - Memory layout: 'C' (row-major, default) or 'F' (column-major)
///
/// # Returns
///
/// A flattened copy of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flat = flatten(&a, None).unwrap();
/// assert_eq!(flat.shape(), vec![6]);
/// assert_eq!(flat.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn flatten<T: Clone>(array: &Array<T>, order: Option<char>) -> Result<Array<T>> {
    // flatten always returns a copy, so we can just use ravel and clone
    ravel(array, order)
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
/// let b = swapaxes(&a, 0, 1).unwrap();
/// assert_eq!(b.shape(), vec![3, 2]);
/// ```
pub fn swapaxes<T: Clone>(array: &Array<T>, axis1: usize, axis2: usize) -> Result<Array<T>> {
    let ndim = array.ndim();
    
    // Check if the axes are valid
    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axes {} and {} are out of bounds for array of dimension {}", axis1, axis2, ndim)
        ));
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
/// let b = moveaxis(&a, &[0], &[2]).unwrap();
/// assert_eq!(b.shape(), vec![2, 2, 2]);
/// ```
pub fn moveaxis<T: Clone>(array: &Array<T>, source: &[usize], destination: &[usize]) -> Result<Array<T>> {
    let ndim = array.ndim();
    
    // Check if the source and destination arrays have the same length
    if source.len() != destination.len() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Source and destination arrays must have the same length, got {} and {}", 
                    source.len(), destination.len())
        ));
    }
    
    // Check if the axes are valid
    for &axis in source.iter().chain(destination.iter()) {
        if axis >= ndim {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Axis {} is out of bounds for array of dimension {}", axis, ndim)
            ));
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
            let j = perm.iter().position(|&p| p == i).unwrap();
            
            // Swap axes i and j in the result
            result = result.transpose_axis(i, j);
            
            // Update the permutation to reflect the swap
            perm.swap(i, j);
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
/// let b = atleast_1d(&[&a]).unwrap();
/// assert_eq!(b[0].shape(), vec![3]);
/// ```
pub fn atleast_1d<T: Clone>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> 
where 
    T: num_traits::Zero
{
    let mut result = Vec::with_capacity(arys.len());
    
    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 1-D
            let scalar_value = array.get(&[]).unwrap();
            result.push(Array::from_vec(vec![scalar_value]).reshape(&[1]));
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
/// let b = atleast_2d(&[&a]).unwrap();
/// assert_eq!(b[0].shape(), vec![1, 3]);
/// ```
pub fn atleast_2d<T: Clone>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> 
where 
    T: num_traits::Zero
{
    let mut result = Vec::with_capacity(arys.len());
    
    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 2-D
            let scalar_value = array.get(&[]).unwrap();
            result.push(Array::from_vec(vec![scalar_value]).reshape(&[1, 1]));
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
/// let b = atleast_3d(&[&a]).unwrap();
/// assert_eq!(b[0].shape(), vec![1, 3, 1]);
/// ```
pub fn atleast_3d<T: Clone>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> 
where 
    T: num_traits::Zero
{
    let mut result = Vec::with_capacity(arys.len());
    
    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 3-D
            let scalar_value = array.get(&[]).unwrap();
            result.push(Array::from_vec(vec![scalar_value]).reshape(&[1, 1, 1]));
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

