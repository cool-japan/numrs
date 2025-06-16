use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use ndarray::Axis;

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

/// Join a sequence of arrays along an existing axis
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
    let result = ndarray::concatenate(Axis(axis), &views).map_err(|e| {
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

/// Join a sequence of arrays along a new axis
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
    let axis = 0;
    concatenate(&row_refs, axis)
}

/// Translate slice objects to concatenation along the first axis
pub fn r_<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    concatenate(arrays, 0)
}

/// Translate slice objects to concatenation along the second axis  
pub fn c_<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to concatenate".into()));
    }

    // Check if all arrays are 1D
    let all_1d = arrays.iter().all(|arr| arr.ndim() == 1);
    
    if all_1d {
        // For 1D arrays, convert them to column vectors and then concatenate along axis 1
        let mut column_vectors = Vec::with_capacity(arrays.len());
        for &arr in arrays {
            let shape = arr.shape();
            let new_shape = vec![shape[0], 1]; // Convert [n] to [n, 1]
            column_vectors.push(arr.reshape(&new_shape));
        }
        
        // Create references for concatenation
        let column_refs: Vec<&Array<T>> = column_vectors.iter().collect();
        concatenate(&column_refs, 1)
    } else {
        // For arrays with 2+ dimensions, concatenate directly along axis 1
        concatenate(arrays, 1)
    }
}
