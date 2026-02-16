//! Index utility functions
//!
//! This module provides utility functions for index manipulation:
//! - `indices_grid()` - Create index arrays for n-dimensional grids
//! - `mask_indices()` - Get indices satisfying a condition
//! - `ravel_multi_index()` - Convert multi-dimensional indices to flat indices
//! - `unravel_index()` - Convert flat indices to multi-dimensional indices
//! - Triangle and diagonal index functions

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

/// Create index arrays for the nth dimension in an n-dimensional grid from shape
///
/// # Parameters
///
/// * `shape` - The shape of the grid
///
/// # Returns
///
/// A vector of arrays containing indices for each dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create index arrays for a 2D grid
/// let indices = indices_grid::<usize>(&[3, 2]).expect("indices_grid should succeed");
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].shape(), vec![3, 1]);  // Column vector of row indices
/// assert_eq!(indices[1].shape(), vec![1, 2]);  // Row vector of column indices
/// ```
pub fn indices_grid<T: Clone + num_traits::Zero + num_traits::One + num_traits::NumCast>(
    shape: &[usize],
) -> Result<Vec<Array<T>>> {
    if shape.is_empty() {
        return Ok(vec![]);
    }

    let mut result = Vec::with_capacity(shape.len());

    for (i, &dim) in shape.iter().enumerate() {
        // Create a shape with 1s except at position i
        let mut index_shape = vec![1; shape.len()];
        index_shape[i] = dim;

        // Create the index array
        let mut index_data = Vec::with_capacity(dim);
        for j in 0..dim {
            index_data.push(T::from(j).ok_or_else(|| {
                NumRs2Error::InvalidOperation("usize should be castable to T".to_string())
            })?);
        }

        let index_array = Array::from_vec(index_data).reshape(&index_shape);
        result.push(index_array);
    }

    Ok(result)
}

/// Return the indices to access array elements that satisfy the given condition
///
/// # Parameters
///
/// * `shape` - Shape of the output array
/// * `mask_fn` - Function which given indices returns a boolean mask
///
/// # Returns
///
/// An array of indices that satisfy the condition
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create indices for the upper triangle of a 3x3 array
/// let indices = mask_indices(&[3, 3], |i| i[0] <= i[1]).expect("mask_indices should succeed");
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].to_vec(), vec![0, 0, 0, 1, 1, 2]);
/// assert_eq!(indices[1].to_vec(), vec![0, 1, 2, 1, 2, 2]);
/// ```
pub fn mask_indices<F>(shape: &[usize], mask_fn: F) -> Result<Vec<Array<usize>>>
where
    F: Fn(&[usize]) -> bool,
{
    if shape.is_empty() {
        return Ok(vec![]);
    }

    // Calculate the total number of elements
    let total_elements: usize = shape.iter().product();

    // Create arrays to store indices for each dimension
    let mut indices_vec: Vec<Vec<usize>> = vec![Vec::new(); shape.len()];

    // Iterate through all possible indices
    let mut indices = vec![0; shape.len()];
    for _ in 0..total_elements {
        // Check the mask function
        if mask_fn(&indices) {
            // Store the indices
            for (dim, &idx) in indices.iter().enumerate() {
                indices_vec[dim].push(idx);
            }
        }

        // Increment indices
        let mut carry = true;
        for dim in (0..shape.len()).rev() {
            if carry {
                indices[dim] += 1;
                carry = indices[dim] >= shape[dim];
                if carry {
                    indices[dim] = 0;
                }
            }
        }
    }

    // Convert to Array objects
    let result = indices_vec.into_iter().map(Array::from_vec).collect();

    Ok(result)
}

/// Converts a tuple of index arrays into an array of flat indices
///
/// Converts a tuple of coordinate arrays to an array of flat indices, applying
/// boundary modes to the multi-dimensional index and returning an error if
/// necessary.
///
/// # Arguments
///
/// * `multi_index` - A vector of arrays, where each array represents indices for one dimension
/// * `dims` - The shape of the array into which the indices will index
/// * `mode` - How to handle out-of-bounds indices: "raise", "wrap", or "clip"
///
/// # Returns
///
/// An array of flat indices corresponding to the multi-dimensional indices
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert 2D indices to flat indices for a 3x4 array
/// let row_indices = Array::from_vec(vec![0, 1, 2, 2]);
/// let col_indices = Array::from_vec(vec![0, 1, 2, 3]);
/// let flat = ravel_multi_index(&[row_indices, col_indices], &[3, 4], "raise").expect("ravel_multi_index should succeed");
/// assert_eq!(flat.to_vec(), vec![0, 5, 10, 11]);
///
/// // With clipping mode
/// let row_indices = Array::from_vec(vec![0, 1, 4]); // 4 is out of bounds
/// let col_indices = Array::from_vec(vec![0, 1, 2]);
/// let flat = ravel_multi_index(&[row_indices, col_indices], &[3, 4], "clip").expect("ravel_multi_index with clip should succeed");
/// assert_eq!(flat.to_vec(), vec![0, 5, 10]); // 4 is clipped to 2
/// ```
pub fn ravel_multi_index(
    multi_index: &[Array<usize>],
    dims: &[usize],
    mode: &str,
) -> Result<Array<usize>> {
    if multi_index.len() != dims.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Number of index arrays ({}) does not match number of dimensions ({})",
            multi_index.len(),
            dims.len()
        )));
    }

    // Check that all index arrays have the same shape
    if multi_index.is_empty() {
        return Ok(Array::from_vec(vec![]));
    }

    let shape = multi_index[0].shape();
    for (_i, arr) in multi_index.iter().enumerate().skip(1) {
        if arr.shape() != shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.clone(),
                actual: arr.shape(),
            });
        }
    }

    // Calculate strides for each dimension
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Convert multi-dimensional indices to flat indices
    let size = multi_index[0].size();
    let mut flat_indices = vec![0; size];

    for i in 0..size {
        let mut flat_idx = 0;

        for (dim_idx, (indices_arr, &dim_size)) in multi_index.iter().zip(dims.iter()).enumerate() {
            let idx = indices_arr.to_vec()[i];

            // Handle boundary conditions
            let bounded_idx = match mode {
                "raise" => {
                    if idx >= dim_size {
                        return Err(NumRs2Error::IndexOutOfBounds(format!(
                            "Index {} is out of bounds for dimension {} with size {}",
                            idx, dim_idx, dim_size
                        )));
                    }
                    idx
                }
                "wrap" => idx % dim_size,
                "clip" => idx.min(dim_size - 1),
                _ => {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Invalid mode '{}'. Must be 'raise', 'wrap', or 'clip'",
                        mode
                    )));
                }
            };

            flat_idx += bounded_idx * strides[dim_idx];
        }

        flat_indices[i] = flat_idx;
    }

    // Create result array with same shape as input index arrays
    if multi_index[0].ndim() == 1 {
        Ok(Array::from_vec(flat_indices))
    } else {
        Ok(Array::from_vec(flat_indices).reshape(&shape))
    }
}

/// Converts a flat index or array of flat indices into a tuple of coordinate arrays
///
/// # Arguments
///
/// * `indices` - Flat indices to convert
/// * `dims` - The shape of the array
///
/// # Returns
///
/// A vector of arrays, where each array contains the indices for one dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert flat indices to 2D coordinates for a 3x4 array
/// let flat = Array::from_vec(vec![0, 5, 10, 11]);
/// let coords = unravel_index(&flat, &[3, 4]).expect("unravel_index should succeed");
/// assert_eq!(coords[0].to_vec(), vec![0, 1, 2, 2]); // row indices
/// assert_eq!(coords[1].to_vec(), vec![0, 1, 2, 3]); // column indices
///
/// // Single index
/// let flat = Array::from_vec(vec![7]);
/// let coords = unravel_index(&flat, &[3, 4]).expect("unravel_index should succeed");
/// assert_eq!(coords[0].to_vec(), vec![1]); // row index
/// assert_eq!(coords[1].to_vec(), vec![3]); // column index
/// ```
pub fn unravel_index(indices: &Array<usize>, dims: &[usize]) -> Result<Vec<Array<usize>>> {
    if dims.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot unravel indices for empty dimensions".to_string(),
        ));
    }

    // Calculate the total size
    let total_size: usize = dims.iter().product();

    // Check that all indices are within bounds
    for &idx in indices.to_vec().iter() {
        if idx >= total_size {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index {} is out of bounds for array of size {}",
                idx, total_size
            )));
        }
    }

    // Calculate strides for each dimension
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Convert flat indices to multi-dimensional indices
    let flat_indices = indices.to_vec();
    let mut multi_indices = vec![vec![0; flat_indices.len()]; dims.len()];

    for (i, &flat_idx) in flat_indices.iter().enumerate() {
        let mut remainder = flat_idx;

        for (dim_idx, &stride) in strides.iter().enumerate() {
            multi_indices[dim_idx][i] = remainder / stride;
            remainder %= stride;
        }
    }

    // Create arrays for each dimension
    let shape = indices.shape();
    let result: Vec<Array<usize>> = multi_indices
        .into_iter()
        .map(|indices| {
            if shape.len() == 1 {
                Array::from_vec(indices)
            } else {
                Array::from_vec(indices).reshape(&shape)
            }
        })
        .collect();

    Ok(result)
}

/// Return the indices for the lower-triangle of an array.
///
/// # Arguments
///
/// * `n` - The row dimension of the square array
/// * `k` - Diagonal offset (default is 0, main diagonal)
/// * `m` - The column dimension of the array. If None, defaults to `n`
///
/// # Returns
///
/// A tuple of arrays (row indices, column indices)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Get indices for 3x3 lower triangle
/// let (rows, cols) = tril_indices(3, 0, None).expect("tril_indices should succeed");
/// assert_eq!(rows.to_vec(), vec![0, 1, 1, 2, 2, 2]);
/// assert_eq!(cols.to_vec(), vec![0, 0, 1, 0, 1, 2]);
///
/// // With k=1 (include first diagonal above main)
/// let (rows, cols) = tril_indices(3, 1, None).expect("tril_indices with k=1 should succeed");
/// assert_eq!(rows.to_vec(), vec![0, 0, 1, 1, 1, 2, 2, 2]);
/// assert_eq!(cols.to_vec(), vec![0, 1, 0, 1, 2, 0, 1, 2]);
/// ```
pub fn tril_indices(n: usize, k: isize, m: Option<usize>) -> Result<(Array<usize>, Array<usize>)> {
    let m = m.unwrap_or(n);

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..n {
        for j in 0..m {
            // Check if element is on or below the k-th diagonal
            if (j as isize) <= (i as isize + k) {
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    Ok((Array::from_vec(row_indices), Array::from_vec(col_indices)))
}

/// Return the indices for the upper-triangle of an array.
///
/// # Arguments
///
/// * `n` - The row dimension of the square array
/// * `k` - Diagonal offset (default is 0, main diagonal)
/// * `m` - The column dimension of the array. If None, defaults to `n`
///
/// # Returns
///
/// A tuple of arrays (row indices, column indices)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Get indices for 3x3 upper triangle
/// let (rows, cols) = triu_indices(3, 0, None).expect("triu_indices should succeed");
/// assert_eq!(rows.to_vec(), vec![0, 0, 0, 1, 1, 2]);
/// assert_eq!(cols.to_vec(), vec![0, 1, 2, 1, 2, 2]);
///
/// // With k=1 (exclude main diagonal)
/// let (rows, cols) = triu_indices(3, 1, None).expect("triu_indices with k=1 should succeed");
/// assert_eq!(rows.to_vec(), vec![0, 0, 1]);
/// assert_eq!(cols.to_vec(), vec![1, 2, 2]);
/// ```
pub fn triu_indices(n: usize, k: isize, m: Option<usize>) -> Result<(Array<usize>, Array<usize>)> {
    let m = m.unwrap_or(n);

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..n {
        for j in 0..m {
            // Check if element is on or above the k-th diagonal
            if (j as isize) >= (i as isize + k) {
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    Ok((Array::from_vec(row_indices), Array::from_vec(col_indices)))
}

/// Return the indices to access the main diagonal of an n-dimensional array.
///
/// # Arguments
///
/// * `n` - The size of the arrays for which the returned indices can be used
/// * `ndim` - The number of dimensions the arrays have (default is 2)
///
/// # Returns
///
/// A tuple of arrays of indices that can be used to access the main diagonal
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::indexing::diag_indices;
///
/// // Get diagonal indices for a 3x3 array
/// let indices = diag_indices(3, Some(2)).expect("diag_indices failed");
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].to_vec(), vec![0, 1, 2]);
/// assert_eq!(indices[1].to_vec(), vec![0, 1, 2]);
///
/// // For 3D array (3x3x3)
/// let indices = diag_indices(3, Some(3)).expect("diag_indices failed");
/// assert_eq!(indices.len(), 3);
/// for dim_indices in &indices {
///     assert_eq!(dim_indices.to_vec(), vec![0, 1, 2]);
/// }
/// ```
pub fn diag_indices(n: usize, ndim: Option<usize>) -> Result<Vec<Array<usize>>> {
    let ndim = ndim.unwrap_or(2);

    if ndim == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Number of dimensions must be at least 1".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(ndim);
    let diagonal_indices: Vec<usize> = (0..n).collect();

    for _dim in 0..ndim {
        result.push(Array::from_vec(diagonal_indices.clone()));
    }

    Ok(result)
}

/// Return the indices to access the main diagonal of an array.
///
/// This is equivalent to `diag_indices(min(arr.shape()), arr.ndim())` but more convenient.
///
/// # Arguments
///
/// * `arr` - Input array
///
/// # Returns
///
/// A tuple of arrays of indices that can be used to access the main diagonal
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::indexing::diag_indices_from;
///
/// // Get diagonal indices from a 3x4 array
/// let a: Array<f64> = Array::zeros(&[3, 4]);
/// let indices = diag_indices_from(&a).expect("diag_indices_from failed");
/// assert_eq!(indices[0].to_vec(), vec![0, 1, 2]);
/// assert_eq!(indices[1].to_vec(), vec![0, 1, 2]);
///
/// // With a 3D array
/// let b: Array<f64> = Array::zeros(&[3, 3, 3]);
/// let indices = diag_indices_from(&b).expect("diag_indices_from failed");
/// assert_eq!(indices.len(), 3);
/// for dim_indices in &indices {
///     assert_eq!(dim_indices.to_vec(), vec![0, 1, 2]);
/// }
/// ```
pub fn diag_indices_from<T: Clone>(arr: &Array<T>) -> Result<Vec<Array<usize>>> {
    let shape = arr.shape();
    let ndim = arr.ndim();

    if ndim == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Array must have at least 1 dimension".to_string(),
        ));
    }

    // Find the minimum dimension size
    let min_dim = shape.iter().min().copied().unwrap_or(0);

    diag_indices(min_dim, Some(ndim))
}

/// Return the indices for the lower-triangle of an array from an existing array.
///
/// # Arguments
///
/// * `arr` - Input array
/// * `k` - Diagonal offset (default is 0, main diagonal)
///
/// # Returns
///
/// A tuple of arrays (row indices, column indices)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::indexing::tril_indices_from;
///
/// // Get lower triangle indices from a 3x3 array
/// let a: Array<f64> = Array::zeros(&[3, 3]);
/// let (rows, cols) = tril_indices_from(&a, Some(0)).expect("tril_indices_from failed");
/// assert_eq!(rows.to_vec(), vec![0, 1, 1, 2, 2, 2]);
/// assert_eq!(cols.to_vec(), vec![0, 0, 1, 0, 1, 2]);
///
/// // With k=1 (include first diagonal above main)
/// let (rows, cols) = tril_indices_from(&a, Some(1)).expect("tril_indices_from failed");
/// assert_eq!(rows.to_vec(), vec![0, 0, 1, 1, 1, 2, 2, 2]);
/// assert_eq!(cols.to_vec(), vec![0, 1, 0, 1, 2, 0, 1, 2]);
/// ```
pub fn tril_indices_from<T: Clone>(
    arr: &Array<T>,
    k: Option<isize>,
) -> Result<(Array<usize>, Array<usize>)> {
    let shape = arr.shape();

    if shape.len() < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "Array must be at least 2-dimensional".to_string(),
        ));
    }

    let n = shape[shape.len() - 2]; // second to last dimension
    let m = shape[shape.len() - 1]; // last dimension
    let k = k.unwrap_or(0);

    tril_indices(n, k, Some(m))
}

/// Return the indices for the upper-triangle of an array from an existing array.
///
/// # Arguments
///
/// * `arr` - Input array
/// * `k` - Diagonal offset (default is 0, main diagonal)
///
/// # Returns
///
/// A tuple of arrays (row indices, column indices)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::indexing::triu_indices_from;
///
/// // Get upper triangle indices from a 3x3 array
/// let a: Array<f64> = Array::zeros(&[3, 3]);
/// let (rows, cols) = triu_indices_from(&a, Some(0)).expect("triu_indices_from failed");
/// assert_eq!(rows.to_vec(), vec![0, 0, 0, 1, 1, 2]);
/// assert_eq!(cols.to_vec(), vec![0, 1, 2, 1, 2, 2]);
///
/// // With k=1 (exclude main diagonal)
/// let (rows, cols) = triu_indices_from(&a, Some(1)).expect("triu_indices_from failed");
/// assert_eq!(rows.to_vec(), vec![0, 0, 1]);
/// assert_eq!(cols.to_vec(), vec![1, 2, 2]);
/// ```
pub fn triu_indices_from<T: Clone>(
    arr: &Array<T>,
    k: Option<isize>,
) -> Result<(Array<usize>, Array<usize>)> {
    let shape = arr.shape();

    if shape.len() < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "Array must be at least 2-dimensional".to_string(),
        ));
    }

    let n = shape[shape.len() - 2]; // second to last dimension
    let m = shape[shape.len() - 1]; // last dimension
    let k = k.unwrap_or(0);

    triu_indices(n, k, Some(m))
}
