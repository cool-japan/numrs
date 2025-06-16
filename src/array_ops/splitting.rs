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

            if axis_len % sections != 0 {
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

            if axis_len % sections != 0 {
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

            if axis_len % sections != 0 {
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
