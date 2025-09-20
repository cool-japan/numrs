use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use ndarray::Axis;

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
        let mut sub_shape = shape.clone();
        sub_shape[axis] = end_idx - start_idx;

        let mut indices = vec![0; shape.len()];
        indices[axis] = start_idx;

        let view = array
            .array()
            .slice_axis(Axis(axis), ndarray::Slice::from(start_idx..end_idx));
        result.push(Array::from_ndarray(view.into_owned().into_dyn()));

        start_idx = end_idx;
    }

    // Add the last section
    if start_idx < axis_len {
        let mut sub_shape = shape.clone();
        sub_shape[axis] = axis_len - start_idx;

        let view = array
            .array()
            .slice_axis(Axis(axis), ndarray::Slice::from(start_idx..axis_len));
        result.push(Array::from_ndarray(view.into_owned().into_dyn()));
    }

    Ok(result)
}

/// Split an array into multiple sub-arrays horizontally (column-wise)
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

/// Split an array into multiple sub-arrays along an axis (allows unequal splits)
///
/// Similar to `split`, but if the array cannot be divided into equal sections,
/// it will distribute the remainder across the first sections.
///
/// # Parameters
///
/// * `array` - The array to split
/// * `sections_or_indices` - Either the number of sections to split into, or explicit indices
/// * `axis` - The axis along which to split
///
/// # Returns
///
/// A vector of sub-arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::splitting::array_split;
///
/// // Split a 10-element array into 3 unequal parts
/// let a = Array::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// let parts = array_split(&a, 3, 0).unwrap();
/// assert_eq!(parts[0].to_vec(), vec![0, 1, 2, 3]);  // 4 elements
/// assert_eq!(parts[1].to_vec(), vec![4, 5, 6]);     // 3 elements
/// assert_eq!(parts[2].to_vec(), vec![7, 8, 9]);     // 3 elements
///
/// // Split using explicit indices
/// let parts = array_split(&a, vec![3, 5], 0).unwrap();
/// assert_eq!(parts[0].to_vec(), vec![0, 1, 2]);
/// assert_eq!(parts[1].to_vec(), vec![3, 4]);
/// assert_eq!(parts[2].to_vec(), vec![5, 6, 7, 8, 9]);
/// ```
pub fn array_split<T: Clone>(
    array: &Array<T>,
    sections_or_indices: impl Into<SplitArg>,
    axis: usize,
) -> Result<Vec<Array<T>>> {
    let shape = array.shape();

    if axis >= shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            shape.len()
        )));
    }

    let axis_len = shape[axis];

    match sections_or_indices.into() {
        SplitArg::Sections(sections) => {
            if sections == 0 {
                return Err(NumRs2Error::InvalidOperation(
                    "Number of sections must be greater than 0".to_string(),
                ));
            }

            if sections > axis_len {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Cannot split array of length {} into {} sections",
                    axis_len, sections
                )));
            }

            // Calculate sizes for each section
            let base_size = axis_len / sections;
            let remainder = axis_len % sections;

            let mut indices = Vec::with_capacity(sections - 1);
            let mut current_idx = 0;

            for i in 0..sections - 1 {
                // Distribute the remainder across the first sections
                let section_size = if i < remainder {
                    base_size + 1
                } else {
                    base_size
                };
                current_idx += section_size;
                indices.push(current_idx);
            }

            split(array, &indices, axis)
        }
        SplitArg::Indices(indices) => split(array, &indices, axis),
    }
}
