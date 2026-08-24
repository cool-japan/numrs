//! Array concatenation helper functions
//!
//! This module provides convenience functions for array concatenation:
//! - `r_concatenate` - Concatenate arrays along the first axis (np.r_)
//! - `c_concatenate` - Concatenate arrays along columns (np.c_)
//! - `ix_` - Create an open mesh from input arrays (np.ix_)

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

/// Array concatenation helper (equivalent to np.r_)
///
/// Concatenate arrays along the first axis. This provides convenient syntax
/// for concatenating arrays similar to NumPy's r_ indexing.
///
/// # Parameters
///
/// * `arrays` - Arrays to concatenate
///
/// # Returns
///
/// Concatenated array along the first axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::r_concatenate;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let result = r_concatenate(&[&a, &b]).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // 2D arrays
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let result = r_concatenate(&[&a, &b]).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![4, 2]);
/// ```
pub fn r_concatenate<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot concatenate empty array list".to_string(),
        ));
    }

    // Use the existing concatenate function along axis 0
    crate::array_ops::concatenate(arrays, 0)
}

/// Array concatenation helper along columns (equivalent to np.c_)
///
/// Concatenate arrays along the second axis (columns). For 1D arrays,
/// this stacks them as columns.
///
/// # Parameters
///
/// * `arrays` - Arrays to concatenate along columns
///
/// # Returns
///
/// Concatenated array along the second axis (columns)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::c_concatenate;
///
/// // 1D arrays become columns
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let result = c_concatenate(&[&a, &b]).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![3, 2]);
/// // Note: the specific order depends on the internal implementation
///
/// // 2D arrays
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let result = c_concatenate(&[&a, &b]).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![2, 4]);
/// ```
pub fn c_concatenate<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot concatenate empty array list".to_string(),
        ));
    }

    // Convert 1D arrays to column vectors
    let mut column_arrays = Vec::new();
    for &arr in arrays {
        if arr.ndim() == 1 {
            // Reshape 1D array to column vector
            let reshaped = arr.reshape(&[arr.len(), 1]);
            column_arrays.push(reshaped);
        } else {
            column_arrays.push(arr.clone());
        }
    }

    // Get references for concatenation
    let column_refs: Vec<&Array<T>> = column_arrays.iter().collect();

    // Use the existing concatenate function along axis 1
    crate::array_ops::concatenate(&column_refs, 1)
}

/// Create an open mesh from input arrays (equivalent to np.ix_)
///
/// Construct an open mesh from multiple sequences. This function takes N 1-D sequences
/// and returns N N-D arrays. These arrays can be used for vectorized evaluation of
/// N-D scalar/vector fields over N-D grids.
///
/// # Parameters
///
/// * `sequences` - 1-D arrays representing coordinates along each axis
///
/// # Returns
///
/// Vector of N-D arrays where each array has values only along its respective axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::ix_;
///
/// let a = Array::from_vec(vec![0, 1, 2]);
/// let b = Array::from_vec(vec![3, 4]);
/// let indices = ix_(&[&a, &b]).expect("operation should succeed");
///
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].shape(), vec![3, 1]);  // Values for first dimension
/// assert_eq!(indices[1].shape(), vec![1, 2]);  // Values for second dimension
///
/// // Can be used for advanced indexing
/// let x = Array::from_vec(vec![10, 11, 12]);
/// let y = Array::from_vec(vec![20, 30]);
/// let grid_indices = ix_(&[&x, &y]).expect("operation should succeed");
/// ```
pub fn ix_<T: Clone>(sequences: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if sequences.is_empty() {
        return Ok(vec![]);
    }

    // Verify all inputs are 1D
    for (i, seq) in sequences.iter().enumerate() {
        if seq.ndim() != 1 {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Input array {} must be 1-D, got {}-D",
                i,
                seq.ndim()
            )));
        }
    }

    let ndim = sequences.len();
    let mut result = Vec::with_capacity(ndim);

    // Create mesh arrays where each array has values only along its axis
    for (axis_idx, &seq) in sequences.iter().enumerate() {
        let mut shape = vec![1; ndim];
        shape[axis_idx] = seq.len();

        let reshaped = seq.reshape(&shape);
        result.push(reshaped);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_concatenate() {
        // Test 1D arrays
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![4, 5, 6]);
        let result = r_concatenate(&[&a, &b]).expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(result.shape(), vec![6]);

        // Test 2D arrays
        let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
        let result = r_concatenate(&[&a, &b]).expect("operation should succeed");
        assert_eq!(result.shape(), vec![4, 2]);
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        // Test error on empty input
        let result = r_concatenate::<i32>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_c_concatenate() {
        // Test 1D arrays (become columns)
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![4, 5, 6]);
        let result = c_concatenate(&[&a, &b]).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3, 2]);
        // Row-major (C-order) flattening of the (3,2) result: each row is [a_i, b_i],
        // matching NumPy: np.c_[[1,2,3],[4,5,6]].flatten() == [1,4,2,5,3,6]
        assert_eq!(result.to_vec(), vec![1, 4, 2, 5, 3, 6]);

        // Test 2D arrays
        let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
        let result = c_concatenate(&[&a, &b]).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2, 4]);
        // NumPy: np.c_[[[1,2],[3,4]], [[5,6],[7,8]]].flatten() == [1,2,5,6,3,4,7,8]
        assert_eq!(result.to_vec(), vec![1, 2, 5, 6, 3, 4, 7, 8]);

        // Test error on empty input
        let result = c_concatenate::<i32>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ix_() {
        // Test 2D case
        let a = Array::from_vec(vec![0, 1, 2]);
        let b = Array::from_vec(vec![3, 4]);
        let indices = ix_(&[&a, &b]).expect("operation should succeed");

        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].shape(), vec![3, 1]);
        assert_eq!(indices[1].shape(), vec![1, 2]);

        // Check values
        assert_eq!(indices[0].to_vec(), vec![0, 1, 2]);
        assert_eq!(indices[1].to_vec(), vec![3, 4]);

        // Test 3D case
        let x = Array::from_vec(vec![10, 11]);
        let y = Array::from_vec(vec![20, 30]);
        let z = Array::from_vec(vec![100]);
        let indices_3d = ix_(&[&x, &y, &z]).expect("operation should succeed");

        assert_eq!(indices_3d.len(), 3);
        assert_eq!(indices_3d[0].shape(), vec![2, 1, 1]);
        assert_eq!(indices_3d[1].shape(), vec![1, 2, 1]);
        assert_eq!(indices_3d[2].shape(), vec![1, 1, 1]);

        // Test empty input
        let empty_indices = ix_::<i32>(&[]).expect("operation should succeed");
        assert_eq!(empty_indices.len(), 0);

        // Test error on non-1D input
        let bad_array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let result = ix_(&[&bad_array]);
        assert!(result.is_err());
    }
}
