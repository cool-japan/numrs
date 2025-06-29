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
        return Err(NumRs2Error::InvalidOperation(
            "No arrays to concatenate".into(),
        ));
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

/// Stack arrays in sequence vertically (row wise).
///
/// This is equivalent to concatenation along the first axis after 1-D arrays
/// of shape (N,) have been reshaped to (1,N). Rebuilds arrays divided by vsplit.
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// The array formed by stacking the given arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let stacked = vstack(&[&a, &b]).unwrap();
/// assert_eq!(stacked.shape(), vec![2, 3]);
/// assert_eq!(stacked.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// let c = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let stacked = vstack(&[&c, &d]).unwrap();
/// assert_eq!(stacked.shape(), vec![4, 2]);
/// ```
pub fn vstack<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to stack".into()));
    }

    // Convert 1D arrays to 2D arrays with shape (1, N)
    let mut reshaped_arrays = Vec::with_capacity(arrays.len());
    for &arr in arrays {
        if arr.ndim() == 1 {
            let shape = arr.shape();
            reshaped_arrays.push(arr.reshape(&[1, shape[0]]));
        } else {
            reshaped_arrays.push(arr.clone());
        }
    }

    // Create references for concatenation
    let array_refs: Vec<&Array<T>> = reshaped_arrays.iter().collect();
    concatenate(&array_refs, 0)
}

/// Stack arrays in sequence horizontally (column wise).
///
/// This is equivalent to concatenation along the second axis, except for 1-D
/// arrays where it concatenates along the first axis.
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// The array formed by stacking the given arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let stacked = hstack(&[&a, &b]).unwrap();
/// assert_eq!(stacked.shape(), vec![6]);
/// assert_eq!(stacked.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// let c = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let stacked = hstack(&[&c, &d]).unwrap();
/// assert_eq!(stacked.shape(), vec![2, 4]);
/// ```
pub fn hstack<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to stack".into()));
    }

    // Check if all arrays are 1D
    let all_1d = arrays.iter().all(|arr| arr.ndim() == 1);

    if all_1d {
        // For 1D arrays, concatenate along axis 0
        concatenate(arrays, 0)
    } else {
        // For arrays with 2+ dimensions, concatenate along axis 1
        concatenate(arrays, 1)
    }
}

/// Stack arrays in sequence depth wise (along third axis).
///
/// This is equivalent to concatenation along the third axis after 2-D arrays
/// of shape (M,N) have been reshaped to (M,N,1) and 1-D arrays of shape (N,)
/// have been reshaped to (1,N,1).
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// The array formed by stacking the given arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let stacked = dstack(&[&a, &b]).unwrap();
/// assert_eq!(stacked.shape(), vec![1, 3, 2]);
///
/// let c = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let stacked = dstack(&[&c, &d]).unwrap();
/// assert_eq!(stacked.shape(), vec![2, 2, 2]);
/// ```
pub fn dstack<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to stack".into()));
    }

    // Reshape arrays to have at least 3 dimensions
    let mut reshaped_arrays = Vec::with_capacity(arrays.len());
    for &arr in arrays {
        let shape = arr.shape();
        let reshaped = match arr.ndim() {
            1 => {
                // 1D array: reshape from (N,) to (1, N, 1)
                arr.reshape(&[1, shape[0], 1])
            }
            2 => {
                // 2D array: reshape from (M, N) to (M, N, 1)
                arr.reshape(&[shape[0], shape[1], 1])
            }
            _ => {
                // 3D+ array: use as is
                arr.clone()
            }
        };
        reshaped_arrays.push(reshaped);
    }

    // Create references for concatenation
    let array_refs: Vec<&Array<T>> = reshaped_arrays.iter().collect();
    concatenate(&array_refs, 2)
}

/// Stack arrays as rows.
///
/// This is an alias for vstack. Stack arrays in sequence vertically (row wise).
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// The array formed by stacking the given arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let stacked = row_stack(&[&a, &b]).unwrap();
/// assert_eq!(stacked.shape(), vec![2, 3]);
/// assert_eq!(stacked.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn row_stack<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    vstack(arrays)
}

/// Stack 1-D arrays as columns into a 2-D array.
///
/// Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array.
/// 2-D arrays are stacked as-is, similar to hstack.
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to stack
///
/// # Returns
///
/// 2-D array formed by stacking the given arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Stack 1-D arrays as columns
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let stacked = column_stack(&[&a, &b]).unwrap();
/// assert_eq!(stacked.shape(), vec![3, 2]);
/// assert_eq!(stacked.to_vec(), vec![1, 4, 2, 5, 3, 6]);
///
/// // 2-D arrays are stacked horizontally
/// let c = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let stacked = column_stack(&[&c, &d]).unwrap();
/// assert_eq!(stacked.shape(), vec![2, 4]);
/// assert_eq!(stacked.to_vec(), vec![1, 2, 5, 6, 3, 4, 7, 8]);
/// ```
pub fn column_stack<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation("No arrays to stack".into()));
    }

    // Check if all arrays are 1D
    let all_1d = arrays.iter().all(|arr| arr.ndim() == 1);

    if all_1d {
        // For 1D arrays, reshape them to column vectors and then concatenate horizontally
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
        // For arrays with 2+ dimensions, use hstack
        hstack(arrays)
    }
}

/// Interpret an object as a matrix and build a matrix from string representations
///
/// This function emulates NumPy's `bmat` functionality, allowing you to create
/// matrices from string representations like "A B; C D" or nested arrays.
///
/// # Arguments
///
/// * `obj` - String description of the matrix or nested array structure
///
/// # Returns
///
/// A 2D array built from the specified blocks
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::joining::bmat_from_string;
///
/// // Create a 2x2 block matrix from string description
/// // "A B; C D" where A, B, C, D are variable names in scope
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c = Array::from_vec(vec![9, 10, 11, 12]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![13, 14, 15, 16]).reshape(&[2, 2]);
///
/// // The result would be a 4x4 matrix with these blocks
/// ```
pub fn bmat_from_string<T: Clone>(_description: &str) -> Result<Array<T>> {
    // This is a simplified implementation - in practice, this would need
    // a more sophisticated parser to handle variable references
    Err(NumRs2Error::InvalidOperation(
        "String-based bmat not yet implemented - use bmat_from_arrays instead".to_string(),
    ))
}

/// Create a matrix from an array-like object or nested sequence of arrays
///
/// This function creates a matrix by treating each sub-array as a block.
/// It's similar to `block()` but specifically designed for 2D matrix construction.
///
/// # Arguments
///
/// * `obj` - Nested array structure representing the matrix blocks
///
/// # Returns
///
/// A 2D array built from the specified blocks
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::joining::bmat_from_arrays;
///
/// // Create individual matrices
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c = Array::from_vec(vec![9, 10, 11, 12]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![13, 14, 15, 16]).reshape(&[2, 2]);
///
/// // Create a 2x2 block matrix: [A B]
/// //                            [C D]
/// let blocks = vec![
///     vec![&a, &b],
///     vec![&c, &d],
/// ];
/// let result = bmat_from_arrays(&blocks).unwrap();
///
/// // Result is a 4x4 matrix:
/// // [ 1  2  5  6]
/// // [ 3  4  7  8]
/// // [ 9 10 13 14]
/// // [11 12 15 16]
/// assert_eq!(result.shape(), vec![4, 4]);
/// ```
pub fn bmat_from_arrays<T: Clone>(obj: &[Vec<&Array<T>>]) -> Result<Array<T>> {
    if obj.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Empty block matrix specification".to_string(),
        ));
    }

    // Ensure all arrays in the matrix are 2D
    for (row_idx, row) in obj.iter().enumerate() {
        if row.is_empty() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Empty row {} in block matrix",
                row_idx
            )));
        }

        for (col_idx, &arr) in row.iter().enumerate() {
            match arr.ndim() {
                0 => {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Scalar at position ({}, {}) - bmat requires 2D arrays",
                        row_idx, col_idx
                    )));
                }
                1 => {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "1D array at position ({}, {}) - bmat requires 2D arrays",
                        row_idx, col_idx
                    )));
                }
                2 => {} // Good - this is what we expect
                _ => {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "{}D array at position ({}, {}) - bmat requires 2D arrays",
                        arr.ndim(),
                        row_idx,
                        col_idx
                    )));
                }
            }
        }
    }

    // Verify that all rows have the same number of columns
    let num_cols = obj[0].len();
    for (row_idx, row) in obj.iter().enumerate() {
        if row.len() != num_cols {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Row {} has {} blocks, expected {}",
                row_idx,
                row.len(),
                num_cols
            )));
        }
    }

    // Verify that blocks in each row have compatible heights
    // and blocks in each column have compatible widths
    for row_idx in 0..obj.len() {
        let row = &obj[row_idx];
        let expected_height = row[0].shape()[0];

        for (col_idx, &arr) in row.iter().enumerate() {
            let height = arr.shape()[0];
            if height != expected_height {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Block at ({}, {}) has height {}, but row {} requires height {}",
                    row_idx, col_idx, height, row_idx, expected_height
                )));
            }
        }
    }

    for col_idx in 0..num_cols {
        let expected_width = obj[0][col_idx].shape()[1];

        for (row_idx, row) in obj.iter().enumerate() {
            let width = row[col_idx].shape()[1];
            if width != expected_width {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Block at ({}, {}) has width {}, but column {} requires width {}",
                    row_idx, col_idx, width, col_idx, expected_width
                )));
            }
        }
    }

    // Now we can safely concatenate
    // First, concatenate each row horizontally
    let mut concatenated_rows = Vec::with_capacity(obj.len());

    for row in obj.iter() {
        if row.len() == 1 {
            // Single block in this row
            concatenated_rows.push(row[0].clone());
        } else {
            // Multiple blocks - concatenate horizontally
            let concatenated_row = concatenate(row, 1)?;
            concatenated_rows.push(concatenated_row);
        }
    }

    // Then concatenate all rows vertically
    if concatenated_rows.len() == 1 {
        Ok(concatenated_rows[0].clone())
    } else {
        let row_refs: Vec<&Array<T>> = concatenated_rows.iter().collect();
        concatenate(&row_refs, 0)
    }
}

/// Convenient alias for bmat_from_arrays
///
/// This provides the main `bmat` functionality for creating block matrices.
///
/// # Arguments
///
/// * `obj` - Nested array structure representing the matrix blocks
///
/// # Returns
///
/// A 2D array built from the specified blocks
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::joining::bmat;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c = Array::from_vec(vec![9, 10, 11, 12]).reshape(&[2, 2]);
/// let d = Array::from_vec(vec![13, 14, 15, 16]).reshape(&[2, 2]);
///
/// let blocks = vec![
///     vec![&a, &b],
///     vec![&c, &d],
/// ];
/// let result = bmat(&blocks).unwrap();
/// assert_eq!(result.shape(), vec![4, 4]);
/// ```
pub fn bmat<T: Clone>(obj: &[Vec<&Array<T>>]) -> Result<Array<T>> {
    bmat_from_arrays(obj)
}
