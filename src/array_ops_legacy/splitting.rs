//! Array splitting operations
//!
//! This module provides functions for splitting arrays into multiple parts.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::Axis;

use super::core::{NumericExt, SplitArg};

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
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2x6 array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[2, 6]);
///
/// // Split into 3 equal parts
/// let splits = hsplit(&a, 3).expect("hsplit should succeed for valid 2D array");
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2]);
///
/// // Split at specific indices
/// let splits2 = hsplit(&a, vec![2, 4]).expect("hsplit should succeed for valid indices");
/// assert_eq!(splits2.len(), 3);
/// ```
pub fn hsplit<T: Clone>(
    array: &Array<T>,
    sections_or_indices: impl Into<SplitArg>,
) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "hsplit requires at least 2D array".to_string(),
        ));
    }

    // Split along the second axis (columns)
    let axis = 1;

    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            let axis_len = shape[axis];

            if !axis_len.is_multiple_of(sections) {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "array of shape {:?} cannot be split into {} equal sections along axis {}",
                    shape, sections, axis
                )));
            }

            let section_size = axis_len / sections;
            let mut indices = Vec::with_capacity(sections - 1);

            for i in 1..sections {
                indices.push(i * section_size);
            }

            split(array, &indices, axis)
        }
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
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 6x2 array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[6, 2]);
///
/// // Split into 3 equal parts
/// let splits = vsplit(&a, 3).expect("vsplit should succeed for valid 2D array");
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2]);
///
/// // Split at specific indices
/// let splits2 = vsplit(&a, vec![2, 4]).expect("vsplit should succeed for valid indices");
/// assert_eq!(splits2.len(), 3);
/// ```
pub fn vsplit<T: Clone>(
    array: &Array<T>,
    sections_or_indices: impl Into<SplitArg>,
) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "vsplit requires at least 2D array".to_string(),
        ));
    }

    // Split along the first axis (rows)
    let axis = 0;

    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            let axis_len = shape[axis];

            if !axis_len.is_multiple_of(sections) {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "array of shape {:?} cannot be split into {} equal sections along axis {}",
                    shape, sections, axis
                )));
            }

            let section_size = axis_len / sections;
            let mut indices = Vec::with_capacity(sections - 1);

            for i in 1..sections {
                indices.push(i * section_size);
            }

            split(array, &indices, axis)
        }
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
/// let splits = dsplit(&a, 3).expect("dsplit should succeed for valid 3D array");
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2, 2]);
/// ```
pub fn dsplit<T: Clone>(
    array: &Array<T>,
    sections_or_indices: impl Into<SplitArg>,
) -> Result<Vec<Array<T>>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 3 {
        return Err(NumRs2Error::InvalidOperation(
            "dsplit requires at least 3D array".to_string(),
        ));
    }

    // Split along the third axis (depth)
    let axis = 2;

    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            let axis_len = shape[axis];

            if !axis_len.is_multiple_of(sections) {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "array of shape {:?} cannot be split into {} equal sections along axis {}",
                    shape, sections, axis
                )));
            }

            let section_size = axis_len / sections;
            let mut indices = Vec::with_capacity(sections - 1);

            for i in 1..sections {
                indices.push(i * section_size);
            }

            split(array, &indices, axis)
        }
        SplitArg::Indices(indices) => split(array, &indices, axis),
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
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            shape.len()
        )));
    }

    let axis_len = shape[axis];

    // Determine the split indices
    let mut split_indices = Vec::new();

    for &idx in indices {
        if idx == 0 || idx >= axis_len {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Split index {} out of bounds for axis {} with size {}",
                idx, axis, axis_len
            )));
        }

        split_indices.push(idx);
    }

    // Sort indices to ensure they're in ascending order
    split_indices.sort();

    // Create the result arrays
    let mut result = Vec::new();

    let mut start_idx = 0;
    for &end_idx in split_indices.iter() {
        let view = array.array().slice_axis(
            Axis(axis),
            scirs2_core::ndarray::Slice::from(start_idx..end_idx),
        );
        result.push(Array::from_ndarray(view.into_owned().into_dyn()));

        start_idx = end_idx;
    }

    // Add the last section
    if start_idx < axis_len {
        let view = array.array().slice_axis(
            Axis(axis),
            scirs2_core::ndarray::Slice::from(start_idx..axis_len),
        );
        result.push(Array::from_ndarray(view.into_owned().into_dyn()));
    }

    Ok(result)
}
