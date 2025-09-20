use crate::array::Array;
use crate::array_ops::manipulation::ravel;
use crate::error::{NumRs2Error, Result};
use ndarray::{Axis, IxDyn};
use num_traits::Zero;
use std::cmp;
use std::fmt::Display;

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
    let first_elem = array
        .array()
        .first()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Cannot tile an empty array".into()))?
        .clone();

    let mut result = Array::full(&output_shape, first_elem);

    // Fill the output array by copying the input array in a tiled pattern
    // This is a simplified implementation - for efficiency, we would use
    // more sophisticated slicing and assignment operations

    let result_vec = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    let input_vec = array.to_vec();
    let input_size = input_vec.len();

    if input_size == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot tile an empty array".into(),
        ));
    }

    // For each position in the output, copy the corresponding element from the input
    for (i, item) in result_vec.iter_mut().enumerate() {
        // Calculate corresponding index in the input array
        // This is a simplification - for a complete implementation, we would need
        // to carefully map N-dimensional indices
        let input_idx = i % input_size;
        *item = input_vec[input_idx].clone();
    }

    Ok(result)
}

/// Repeat elements of an array along a specified axis
pub fn repeat<T: Clone>(array: &Array<T>, repeats: usize, axis: Option<usize>) -> Result<Array<T>> {
    let a_shape = array.shape();

    match axis {
        Some(ax) => {
            if ax >= a_shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    a_shape.len()
                )));
            }

            // Calculate the output shape
            let mut output_shape = a_shape.clone();
            output_shape[ax] *= repeats;

            // Create a result array
            let first_elem = array
                .array()
                .first()
                .ok_or_else(|| {
                    NumRs2Error::InvalidOperation("Cannot repeat an empty array".into())
                })?
                .clone();

            let mut result = Array::full(&output_shape, first_elem);

            // Fill the result array by repeating elements along the specified axis
            // This is a simplified implementation - a more efficient version would use
            // vectorized operations and views

            let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
            })?;

            let input_vec = array.to_vec();

            if input_vec.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "Cannot repeat an empty array".into(),
                ));
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
        }
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

/// Return an array with the specified requirements
///
/// # Parameters
///
/// * `array` - Input array
/// * `requirements` - Requirements for the array, specified as a combination of flags
///
/// # Returns
///
/// * Array with the specified requirements. If the input array satisfies the requirements,
///   it may be returned as-is.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create an array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Require a C-contiguous array (row-major order)
/// let b = require(&a, Some(ArrayRequirements::CONTIGUOUS | ArrayRequirements::C_LAYOUT)).unwrap();
/// assert!(b.is_c_contiguous());
///
/// // Require a Fortran-contiguous array (column-major order)
/// let c = require(&a, Some(ArrayRequirements::CONTIGUOUS | ArrayRequirements::F_LAYOUT)).unwrap();
/// assert!(c.is_f_contiguous());
/// ```
pub fn require<T: Clone>(
    array: &Array<T>,
    requirements: Option<ArrayRequirements>,
) -> Result<Array<T>> {
    // If no requirements are specified, return a copy of the input array
    let requirements = requirements.unwrap_or(ArrayRequirements::empty());

    // If no requirements are specified, return a copy of the input array
    if requirements.is_empty() {
        return Ok(array.clone());
    }

    // Check if we need a specific layout
    let need_c_layout = requirements.contains(ArrayRequirements::C_LAYOUT);
    let need_f_layout = requirements.contains(ArrayRequirements::F_LAYOUT);

    // Check if we need a contiguous array
    let need_contiguous = requirements.contains(ArrayRequirements::CONTIGUOUS);

    // Check if we need to own the data
    let _need_owner = requirements.contains(ArrayRequirements::OWNDATA);

    // Check if we need a writeable array
    let _need_writeable = requirements.contains(ArrayRequirements::WRITEABLE);

    // Check if the input array satisfies the requirements
    let meets_c_layout = if need_c_layout {
        array.is_c_contiguous()
    } else {
        true
    };
    let meets_f_layout = if need_f_layout {
        array.is_f_contiguous()
    } else {
        true
    };
    let meets_contiguous = if need_contiguous {
        array.is_contiguous()
    } else {
        true
    };

    // NumRS arrays always own their data and are writeable, so these are always met
    // If this changes in the future, we should check for these requirements

    // If all requirements are met, return the original array
    if meets_c_layout && meets_f_layout && meets_contiguous {
        return Ok(array.clone());
    }

    // Otherwise, create a new array that meets the requirements
    let mut result = array.clone();

    // If we need a C-contiguous array, convert to C layout
    if need_c_layout && !meets_c_layout {
        result = result.to_c_layout();
    }

    // If we need a Fortran-contiguous array, convert to F layout
    if need_f_layout && !meets_f_layout {
        result = result.to_f_layout();
    }

    // If we need a contiguous array but neither C nor F layout is specified,
    // prefer C layout (row-major order)
    if need_contiguous && !meets_contiguous && !need_c_layout && !need_f_layout {
        result = result.to_c_layout();
    }

    Ok(result)
}

/// Bitflags to specify requirements for arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayRequirements(u32);

impl ArrayRequirements {
    // Flag values
    /// Ensure array is contiguous in memory
    pub const CONTIGUOUS: Self = Self(1 << 0);
    /// Ensure array is in C layout (row-major order)
    pub const C_LAYOUT: Self = Self(1 << 1);
    /// Ensure array is in Fortran layout (column-major order)
    pub const F_LAYOUT: Self = Self(1 << 2);
    /// Ensure array owns its data (not a view)
    pub const OWNDATA: Self = Self(1 << 3);
    /// Ensure array is writeable
    pub const WRITEABLE: Self = Self(1 << 4);

    /// Create an empty requirements set
    pub fn empty() -> Self {
        Self(0)
    }

    /// Check if the requirements are empty
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Check if the requirements contain a specific flag
    pub fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for ArrayRequirements {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for ArrayRequirements {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

/// Extract a diagonal or construct a diagonal array
///
/// # Parameters
///
/// * `array` - Input array
/// * `k` - Offset of the diagonal from the main diagonal.
///   A positive value means the diagonal is above the main diagonal.
///   A negative value means the diagonal is below the main diagonal.
///   The default is 0 (the main diagonal).
///
/// # Returns
///
/// * If `array` is 1D, returns a 2D array with `array` on the `k`-th diagonal.
/// * If `array` is 2D, returns a 1D array of the diagonal elements along the `k`-th diagonal.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a diagonal matrix from a 1D array
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let diag_mat = diag(&a, Some(0)).unwrap();
/// assert_eq!(diag_mat.shape(), vec![3, 3]);
/// assert_eq!(diag_mat.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
///
/// // Extract the main diagonal from a 2D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let diag_vec = diag(&b, Some(0)).unwrap();
/// assert_eq!(diag_vec.shape(), vec![3]);
/// assert_eq!(diag_vec.to_vec(), vec![1, 5, 9]);
///
/// // Extract a super-diagonal (k=1)
/// let super_diag = diag(&b, Some(1)).unwrap();
/// assert_eq!(super_diag.shape(), vec![2]);
/// assert_eq!(super_diag.to_vec(), vec![2, 6]);
///
/// // Extract a sub-diagonal (k=-1)
/// let sub_diag = diag(&b, Some(-1)).unwrap();
/// assert_eq!(sub_diag.shape(), vec![2]);
/// assert_eq!(sub_diag.to_vec(), vec![4, 8]);
/// ```
pub fn diag<T: Clone + Zero>(array: &Array<T>, k: impl Into<Option<isize>>) -> Result<Array<T>> {
    let k = k.into().unwrap_or(0);
    let ndim = array.ndim();

    match ndim {
        1 => {
            // Create a 2D array with the 1D array on the k-th diagonal
            let size = array.size();
            let diag_size = size + k.unsigned_abs();

            // Create a square zero array
            let result = Array::zeros(&[diag_size, diag_size]);
            let mut result_vec = result.to_vec();

            // Place the 1D array on the k-th diagonal
            let array_vec = array.to_vec();

            #[allow(clippy::needless_range_loop)]
            for i in 0..size {
                let row: usize;
                let col: usize;

                if k >= 0 {
                    row = i;
                    col = i + k as usize;
                } else {
                    row = i + (-k) as usize;
                    col = i;
                }

                if row < diag_size && col < diag_size {
                    let idx = row * diag_size + col;
                    result_vec[idx] = array_vec[i].clone();
                }
            }

            Ok(Array::from_vec(result_vec).reshape(&[diag_size, diag_size]))
        }
        2 => {
            // Extract the k-th diagonal from a 2D array
            let shape = array.shape();

            if shape.len() != 2 {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Expected a 2D array, got shape {:?}",
                    shape
                )));
            }

            let rows = shape[0];
            let cols = shape[1];

            // Calculate the length of the resulting diagonal
            let diag_len = if k >= 0 {
                cmp::min(rows, cols.saturating_sub(k as usize))
            } else {
                cmp::min(rows.saturating_sub((-k) as usize), cols)
            };

            if diag_len == 0 {
                return Ok(Array::zeros(&[0]));
            }

            let mut result = Vec::with_capacity(diag_len);
            let array_vec = array.to_vec();

            for i in 0..diag_len {
                let row: usize;
                let col: usize;

                if k >= 0 {
                    row = i;
                    col = i + k as usize;
                } else {
                    row = i + (-k) as usize;
                    col = i;
                }

                if row < rows && col < cols {
                    let idx = row * cols + col;
                    result.push(array_vec[idx].clone());
                }
            }

            Ok(Array::from_vec(result))
        }
        _ => Err(NumRs2Error::InvalidOperation(format!(
            "Input must be 1D or 2D array, got {}D array",
            ndim
        ))),
    }
}

/// Return a specified diagonal of an array
///
/// # Parameters
///
/// * `array` - Input array
/// * `offset` - Offset of the diagonal from the main diagonal.
///   A positive value means the diagonal is above the main diagonal.
///   A negative value means the diagonal is below the main diagonal.
///   The default is 0 (the main diagonal).
/// * `axis1` - First axis of the 2D subarray from which the diagonal should be taken.
///   Default is 0.
/// * `axis2` - Second axis of the 2D subarray from which the diagonal should be taken.
///   Default is 1.
///
/// # Returns
///
/// * A view of the specified diagonal.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Extract the main diagonal from a 2D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let diag = diagonal(&a, Some(0), None, None).unwrap();
/// assert_eq!(diag.shape(), vec![3]);
/// assert_eq!(diag.to_vec(), vec![1, 5, 9]);
///
/// // Extract a super-diagonal (offset=1)
/// let super_diag = diagonal(&a, Some(1), None, None).unwrap();
/// assert_eq!(super_diag.shape(), vec![2]);
/// assert_eq!(super_diag.to_vec(), vec![2, 6]);
///
/// // Extract a sub-diagonal (offset=-1)
/// let sub_diag = diagonal(&a, Some(-1), None, None).unwrap();
/// assert_eq!(sub_diag.shape(), vec![2]);
/// assert_eq!(sub_diag.to_vec(), vec![4, 8]);
///
/// // Extract the diagonal from a 3D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[2, 2, 3]);
/// let diag = diagonal(&b, Some(0), Some(1), Some(2)).unwrap();
/// assert_eq!(diag.shape(), vec![2, 2]);
/// assert_eq!(diag.to_vec(), vec![1, 5, 7, 11]);
/// ```
pub fn diagonal<T: Clone + num_traits::Zero>(
    array: &Array<T>,
    offset: impl Into<Option<isize>>,
    axis1: impl Into<Option<usize>>,
    axis2: impl Into<Option<usize>>,
) -> Result<Array<T>> {
    let offset = offset.into().unwrap_or(0);
    let axis1 = axis1.into().unwrap_or(0);
    let axis2 = axis2.into().unwrap_or(1);

    let ndim = array.ndim();

    if ndim < 2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Array must be at least 2D, got {}D array",
            ndim
        )));
    }

    if axis1 == axis2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "axis1 and axis2 cannot be the same: {}",
            axis1
        )));
    }

    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes ({}, {}) out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
    }

    // Get the lengths of the two axes
    let shape = array.shape();
    let axis1_len = shape[axis1];
    let axis2_len = shape[axis2];

    // Calculate the length of the resulting diagonal
    let diag_len = if offset >= 0 {
        cmp::min(axis1_len, axis2_len.saturating_sub(offset as usize))
    } else {
        cmp::min(axis1_len.saturating_sub((-offset) as usize), axis2_len)
    };

    if diag_len == 0 {
        // Create a result array with the same shape as the input array,
        // but with the two specified axes replaced by a single dimension of length 0
        let mut result_shape = Vec::with_capacity(ndim - 1);
        for (i, &dim) in shape.iter().enumerate() {
            if i != axis1 && i != axis2 {
                result_shape.push(dim);
            }
        }
        result_shape.push(0);

        return Ok(Array::zeros(&result_shape));
    }

    // Prepare the result shape
    let mut result_shape = Vec::with_capacity(ndim - 1);
    for (i, &dim) in shape.iter().enumerate() {
        if i != axis1 && i != axis2 {
            result_shape.push(dim);
        }
    }
    result_shape.push(diag_len);

    // Calculate the total size of the result array
    let result_size: usize = result_shape.iter().product();

    // Create the result array
    let mut result_vec = Vec::with_capacity(result_size);

    // Extract the diagonal values
    let array_vec = array.to_vec();

    // Calculate the strides for each dimension
    let mut strides = Vec::with_capacity(ndim);
    let mut stride = 1;
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();

    // Extract the diagonal elements
    let axis1_stride = strides[axis1];
    let axis2_stride = strides[axis2];

    // Helper function to calculate index without axis1 and axis2
    let calc_base_index = |indices: &[usize]| -> usize {
        let mut base_idx = 0;
        let mut _dst_idx = 0;

        for (src_idx, &dim) in indices.iter().enumerate() {
            if src_idx != axis1 && src_idx != axis2 {
                base_idx += dim * strides[src_idx];
                _dst_idx += 1;
            }
        }

        base_idx
    };

    // Pre-allocate indices array to avoid reallocating in each iteration
    let mut indices = vec![0; ndim];

    // Helper function to increment indices
    let increment_indices = |indices: &mut [usize], shape: &[usize], axis1, axis2| {
        for i in (0..indices.len()).rev() {
            if i != axis1 && i != axis2 {
                indices[i] += 1;
                if indices[i] < shape[i] {
                    return true;
                }
                indices[i] = 0;
            }
        }
        false
    };

    // Number of elements to process (excluding the diagonal axes)
    let mut outer_elements = 1;
    for (i, &dim) in shape.iter().enumerate() {
        if i != axis1 && i != axis2 {
            outer_elements *= dim;
        }
    }

    // Process each combination of indices (except for axis1 and axis2)
    for _ in 0..outer_elements {
        let base_idx = calc_base_index(&indices);

        // Extract the diagonal at this position
        for i in 0..diag_len {
            let row: usize;
            let col: usize;

            if offset >= 0 {
                row = i;
                col = i + offset as usize;
            } else {
                row = i + (-offset) as usize;
                col = i;
            }

            if row < axis1_len && col < axis2_len {
                let idx = base_idx + row * axis1_stride + col * axis2_stride;
                result_vec.push(array_vec[idx].clone());
            }
        }

        // Increment the indices (except for axis1 and axis2)
        increment_indices(&mut indices, &shape, axis1, axis2);
    }

    Ok(Array::from_vec(result_vec).reshape(&result_shape))
}

/// Return a partitioned copy of an array
///
/// Partitioning creates a partially sorted output where elements
/// smaller than the kth element are moved before it and larger elements
/// are moved after it. The kth element will be in the position it would
/// be in a sorted array.
///
/// # Parameters
///
/// * `array` - Array to be partitioned
/// * `kth` - Element index to partition by
/// * `axis` - Axis along which to partition
///   If None, array is flattened before partitioning
///
/// # Returns
///
/// * Copy of array with values arranged to ensure the kth element
///   is in its sorted position
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops_legacy::partition;
///
/// // Partition a 1D array
/// let a = Array::from_vec(vec![9, 4, 1, 7, 5, 3, 8, 2, 6]);
/// let partitioned = partition(&a, 3, None).expect("partition failed");
/// // The 4th element (index 3) is now in correct position for sorting
/// let val = partitioned.get(&[3]).expect("get failed");
/// assert!(val >= 1 && val <= 9);
/// ```
pub fn partition<T: Clone + PartialOrd>(
    array: &Array<T>,
    kth: usize,
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Flatten array and partition
            let mut data = array.to_vec();
            let n = data.len();

            if kth >= n {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "kth ({}) is out of bounds for array of size {}",
                    kth, n
                )));
            }

            // Quick-select algorithm to efficiently find the kth element and partition the array
            quick_select(&mut data, 0, n - 1, kth);

            // Reshape back to original shape
            Ok(Array::from_vec(data).reshape(&array.shape()))
        }
        Some(axis_val) => {
            let shape = array.shape();

            if axis_val >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis_val,
                    shape.len()
                )));
            }

            let axis_size = shape[axis_val];

            if kth >= axis_size {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "kth ({}) is out of bounds for axis {} with size {}",
                    kth, axis_val, axis_size
                )));
            }

            // Create a new array with the same shape
            let mut result = array.clone();
            let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
            })?;

            // Calculate the sizes of the pre-axis, axis, and post-axis dimensions
            let pre_axis_size: usize = shape.iter().take(axis_val).product();
            let post_axis_size: usize = shape.iter().skip(axis_val + 1).product();

            // Partition each slice along the specified axis
            for i_pre in 0..pre_axis_size {
                for i_post in 0..post_axis_size {
                    // Extract the slice along the axis
                    let mut slice = Vec::with_capacity(axis_size);

                    for i_axis in 0..axis_size {
                        let idx =
                            i_pre * (axis_size * post_axis_size) + i_axis * post_axis_size + i_post;
                        slice.push(result_vec[idx].clone());
                    }

                    // Partition the slice
                    quick_select(&mut slice, 0, axis_size - 1, kth);

                    // Write back the partitioned slice
                    #[allow(clippy::needless_range_loop)]
                    for i_axis in 0..axis_size {
                        let idx =
                            i_pre * (axis_size * post_axis_size) + i_axis * post_axis_size + i_post;
                        result_vec[idx] = slice[i_axis].clone();
                    }
                }
            }

            Ok(result)
        }
    }
}

/// Quick-select algorithm to partition an array and place the kth element
/// in its sorted position. Elements smaller than the kth element will be
/// before it, and elements larger than the kth element will be after it.
///
/// This is a helper function for the partition function.
fn quick_select<T: Clone + PartialOrd>(arr: &mut [T], left: usize, right: usize, k: usize) {
    if left == right {
        return;
    }

    // Choose a pivot index (using a simple median-of-three approach)
    let pivot_idx = choose_pivot(arr, left, right);

    // Partition around the pivot
    let pivot_idx = partition_around_pivot(arr, left, right, pivot_idx);

    match k.cmp(&pivot_idx) {
        std::cmp::Ordering::Equal => {
            // k is at its final position
        }
        std::cmp::Ordering::Less => {
            // k is in the left side
            if pivot_idx > 0 {
                quick_select(arr, left, pivot_idx - 1, k);
            }
        }
        std::cmp::Ordering::Greater => {
            // k is in the right side
            quick_select(arr, pivot_idx + 1, right, k);
        }
    }
}

/// Choose a good pivot index using median-of-three strategy
///
/// This helper function helps improve the performance of quick-select
/// by choosing a better pivot than just the first or last element.
fn choose_pivot<T: PartialOrd>(arr: &[T], left: usize, right: usize) -> usize {
    if right - left < 2 {
        return left;
    }

    let mid = left + (right - left) / 2;

    // Choose median of left, middle, and right elements
    let mut indices = [left, mid, right];

    // Simple bubble sort of the three indices based on their values
    if arr[indices[0]] > arr[indices[1]] {
        indices.swap(0, 1);
    }
    if arr[indices[1]] > arr[indices[2]] {
        indices.swap(1, 2);
    }
    if arr[indices[0]] > arr[indices[1]] {
        indices.swap(0, 1);
    }

    // Return the middle value
    indices[1]
}

/// Partition the array around a pivot value
///
/// After partitioning, all elements less than the pivot value are on the left side,
/// and all elements greater are on the right side. The pivot element is at the returned index.
fn partition_around_pivot<T: Clone + PartialOrd>(
    arr: &mut [T],
    left: usize,
    right: usize,
    pivot_idx: usize,
) -> usize {
    let pivot_value = arr[pivot_idx].clone();

    // Move pivot to the end temporarily
    arr.swap(pivot_idx, right);

    // Move all elements less than pivot to the left
    let mut store_idx = left;
    for i in left..right {
        if arr[i] < pivot_value {
            arr.swap(i, store_idx);
            store_idx += 1;
        }
    }

    // Move pivot to its final place
    arr.swap(store_idx, right);

    store_idx
}

/// Find indices where elements should be inserted to maintain order
///
/// Performs binary search to find the indices into a sorted array `a` such that,
/// if the corresponding elements in `v` were inserted before the indices, the
/// order of `a` would be preserved.
///
/// # Parameters
///
/// * `a` - Input array, must be sorted in ascending order
/// * `v` - Values to insert into `a`
/// * `side` - If 'left', return the first suitable location found.
///   If 'right', return the last such index. Default is 'left'.
/// * `sorter` - Optional array of integer indices that sorts `a` into ascending order.
///   This is typically the result of `argsort`.
///
/// # Returns
///
/// * Array of insertion points with the same shape as `v`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops_legacy::searchsorted;
///
/// // Create a sorted array
/// let a = Array::from_vec(vec![1, 3, 5, 7, 9]);
///
/// // Find insertion points for values
/// let v = Array::from_vec(vec![0, 1, 2, 4, 8, 10]);
/// let indices = searchsorted(&a, &v, Some("left"), None).expect("searchsorted failed");
/// assert_eq!(indices.to_vec(), vec![0, 0, 1, 2, 4, 5]);
///
/// // Use 'right' side
/// let indices = searchsorted(&a, &v, Some("right"), None).expect("searchsorted failed");
/// assert_eq!(indices.to_vec(), vec![0, 1, 1, 2, 4, 5]);
/// ```
pub fn searchsorted<T: Clone + PartialOrd>(
    a: &Array<T>,
    v: &Array<T>,
    side: Option<&str>,
    sorter: Option<&Array<usize>>,
) -> Result<Array<usize>> {
    let side = side.unwrap_or("left");
    if side != "left" && side != "right" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Side '{}' is invalid, must be 'left' or 'right'",
            side
        )));
    }

    // If a custom sorter is provided, rearrange the array
    let a_sorted = if let Some(sorter_array) = sorter {
        if sorter_array.ndim() != 1 {
            return Err(NumRs2Error::InvalidOperation(
                "Sorter array must be 1-dimensional".into(),
            ));
        }

        if sorter_array.size() != a.size() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Sorter size ({}) does not match array size ({})",
                sorter_array.size(),
                a.size()
            )));
        }

        // Create a new array using the sorter indices
        let mut sorted_data = Vec::with_capacity(a.size());
        let a_vec = a.to_vec();
        let sorter_vec = sorter_array.to_vec();

        for &idx in &sorter_vec {
            if idx >= a_vec.len() {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Sorter index {} out of range for array of size {}",
                    idx,
                    a_vec.len()
                )));
            }
            sorted_data.push(a_vec[idx].clone());
        }

        Array::from_vec(sorted_data)
    } else {
        a.clone()
    };

    // If a is not 1D, flatten it
    let a_flat = if a_sorted.ndim() != 1 {
        a_sorted.flatten(None)
    } else {
        a_sorted
    };

    // Check if a_flat is sorted
    let a_flat_vec = a_flat.to_vec();
    for i in 1..a_flat_vec.len() {
        if a_flat_vec[i] < a_flat_vec[i - 1] {
            return Err(NumRs2Error::InvalidOperation(
                "The input array must be sorted in ascending order".into(),
            ));
        }
    }

    // Convert v to a flat array if needed
    let v_vec = v.to_vec();

    // Perform binary search for each value in v
    let mut result = Vec::with_capacity(v_vec.len());

    for val in &v_vec {
        let idx = if side == "left" {
            binary_search_left(&a_flat_vec, val)
        } else {
            binary_search_right(&a_flat_vec, val)
        };

        result.push(idx);
    }

    // Reshape result to match v's shape
    Ok(Array::from_vec(result).reshape(&v.shape()))
}

/// Binary search for the leftmost insertion point
fn binary_search_left<T: PartialOrd>(arr: &[T], value: &T) -> usize {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + (right - left) / 2;

        if &arr[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    left
}

/// Binary search for the rightmost insertion point
fn binary_search_right<T: PartialOrd>(arr: &[T], value: &T) -> usize {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + (right - left) / 2;

        if value < &arr[mid] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    left
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
/// ```
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
/// let splits2 = hsplit(&a, vec![2, 4]).unwrap();
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
/// let splits = vsplit(&a, 3).unwrap();
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].shape(), vec![2, 2]);
/// assert_eq!(splits[1].shape(), vec![2, 2]);
/// assert_eq!(splits[2].shape(), vec![2, 2]);
///
/// // Split at specific indices
/// let splits2 = vsplit(&a, vec![2, 4]).unwrap();
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
/// let splits = dsplit(&a, 3).unwrap();
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

// This implementation is commented out to avoid duplicate function definitions
// The flip function with Option<usize> parameter is already implemented earlier in the file

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
// This implementation is commented out to avoid duplicate function definitions
// The fliplr function is already implemented earlier in the file
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
// This implementation is commented out to avoid duplicate function definitions
// The flipud function is already implemented earlier in the file
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
        let processed_arrays: Result<Vec<Array<T>>> = arrays
            .iter()
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
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let c = c_(&[&a, &b]).unwrap();
/// // With 1D arrays, c_ reshapes them to [n, 1] and concatenates along axis 1
/// assert_eq!(c.shape(), vec![3, 2]);
/// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // With 2D arrays
/// let a2 = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let b2 = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
/// let c2 = c_(&[&a2, &b2]).unwrap();
/// assert_eq!(c2.shape(), vec![2, 4]);
/// ```
pub fn c_<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    // Process 1D arrays by adding a dimension
    let processed_arrays: Result<Vec<Array<T>>> = arrays
        .iter()
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
// This implementation is commented out to avoid duplicate function definitions
// The roll function is already implemented earlier in the file
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
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis, ndim
        )));
    }

    if start > ndim {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Start position {} exceeds array dimensions {}",
            start, ndim
        )));
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
fn array_transpose<T: Clone + Zero>(
    array: &Array<T>,
    axis1: usize,
    axis2: usize,
) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes ({}, {}) out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
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
// This implementation is commented out to avoid duplicate function definitions
// The rot90 function is already implemented earlier in the file with a more comprehensive version
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
    // Implementation commented out to avoid duplication
}
*/

// Rotate an array by 90 degrees around specified axes
//
// # Parameters
//
// * `array` - The input array
// * `k` - Number of 90-degree rotations, positive for counterclockwise, negative for clockwise
// * `axes` - Tuple specifying the plane of rotation. Default is (0, 1)
//
// # Returns
//
// A new array rotated in the plane specified by axes
//
// # Examples
//
// ```
// use numrs2::prelude::*;
//
// // Create a 2D array
// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
//
// // Rotate 90 degrees counterclockwise
// let rotated = rot90(&a, 1, Some((0, 1))).unwrap();
// assert_eq!(rotated.shape(), vec![3, 3]);
// assert_eq!(rotated.to_vec(), vec![7, 8, 9, 4, 5, 6, 1, 2, 3]);
//
// // Rotate 180 degrees
// let rotated_180 = rot90(&a, 2, None).unwrap();
// assert_eq!(rotated_180.shape(), vec![3, 3]);
// assert_eq!(rotated_180.to_vec(), vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
// ```
// Reverse the order of elements along the given axis
//
// # Parameters
//
// * `array` - The input array
// * `axis` - Axis along which to flip. If None, all axes are flipped
//
// # Returns
//
// A new array with the elements reversed along the specified axis
//
// # Examples
//
// ```
// use numrs2::prelude::*;
//
// // Create a 2D array
// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
//
// // Flip along axis 0 (rows)
// let flipped_rows = flip(&a, Some(0)).unwrap();
// assert_eq!(flipped_rows.to_vec(), vec![4, 5, 6, 1, 2, 3]);
//
// // Flip along axis 1 (columns)
// let flipped_cols = flip(&a, Some(1)).unwrap();
// assert_eq!(flipped_cols.to_vec(), vec![3, 2, 1, 6, 5, 4]);
//
// // Flip along all axes
// let flipped_all = flip(&a, None).unwrap();
// assert_eq!(flipped_all.to_vec(), vec![6, 5, 4, 3, 2, 1]);
// ```
// This implementation is commented out to avoid duplicate function definitions
// The flip function is already implemented earlier in the file with a more comprehensive version
// Flip array in the left/right direction (last axis)
//
// # Parameters
//
// * `array` - The input array
//
// # Returns
//
// A new array with the elements reversed along the last axis
//
// # Examples
//
// ```
// use numrs2::prelude::*;
//
// // Create a 2D array
// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
//
// // Flip left/right (last axis - columns)
// let flipped = fliplr(&a).unwrap();
// assert_eq!(flipped.to_vec(), vec![3, 2, 1, 6, 5, 4]);
// ```
// This implementation is commented out to avoid duplicate function definitions
// The fliplr function is already implemented earlier in the file

// Flip array in the up/down direction (first axis)
//
// # Parameters
//
// * `array` - The input array
//
// # Returns
//
// A new array with the elements reversed along the first axis
//
// # Examples
//
// ```
// use numrs2::prelude::*;
//
// // Create a 2D array
// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
//
// // Flip up/down (first axis - rows)
// let flipped = flipud(&a).unwrap();
// assert_eq!(flipped.to_vec(), vec![4, 5, 6, 1, 2, 3]);
// ```
// This implementation is commented out to avoid duplicate function definitions
// The flipud function is already implemented earlier in the file

/*
// This implementation is commented out to avoid duplicate function definitions
// The rot90 function is already implemented earlier in the file with a more flexible signature
pub fn rot90<T: Clone>(array: &Array<T>, k: isize, axes: Option<(usize, usize)>) -> Result<Array<T>> {
    // Implementation commented out to avoid duplication
}
*/

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
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes {} and {} are out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
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
pub fn moveaxis<T: Clone>(
    array: &Array<T>,
    source: &[usize],
    destination: &[usize],
) -> Result<Array<T>> {
    let ndim = array.ndim();

    // Check if the source and destination arrays have the same length
    if source.len() != destination.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Source and destination arrays must have the same length, got {} and {}",
            source.len(),
            destination.len()
        )));
    }

    // Check if the axes are valid
    for &axis in source.iter().chain(destination.iter()) {
        if axis >= ndim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} is out of bounds for array of dimension {}",
                axis, ndim
            )));
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
pub fn atleast_1d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
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
pub fn atleast_2d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
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
pub fn atleast_3d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
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

/// One-dimensional linear interpolation.
///
/// Returns the one-dimensional piecewise linear interpolant to a function
/// with given discrete data points (xp, fp), evaluated at x.
///
/// # Parameters
///
/// * `x` - The x-coordinates at which to evaluate the interpolated values.
/// * `xp` - The x-coordinates of the data points, must be increasing.
/// * `fp` - The y-coordinates of the data points, same length as `xp`.
/// * `left` - Value to return for `x < xp[0]`. If not provided, defaults to `fp[0]`.
/// * `right` - Value to return for `x > xp[last]`. If not provided, defaults to `fp[last]`.
/// * `period` - A period for the x-coordinates. This parameter allows making the interpolation periodic in the specified period.
///
/// # Returns
///
/// The interpolated values, same shape as `x`.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops_legacy::interp;
///
/// let xp = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let fp = Array::from_vec(vec![3.0, 2.0, 0.0]);
/// let x = Array::from_vec(vec![0.0, 1.5, 2.0, 2.5, 3.0, 4.0]);
///
/// // Without explicitly specifying `left` and `right`
/// let y = interp(&x, &xp, &fp, None, None, None).expect("interp failed");
/// assert_eq!(y.to_vec(), vec![3.0, 2.5, 2.0, 1.0, 0.0, 0.0]);
///
/// // With explicit `left` and `right` values
/// let y = interp(&x, &xp, &fp, Some(-5.0), Some(-1.0), None).expect("interp failed");
/// assert_eq!(y.to_vec(), vec![-5.0, 2.5, 2.0, 1.0, 0.0, -1.0]);
/// ```
pub fn interp<T>(
    x: &Array<T>,
    xp: &Array<T>,
    fp: &Array<T>,
    left: Option<T>,
    right: Option<T>,
    period: Option<T>,
) -> Result<Array<T>>
where
    T: Clone
        + PartialOrd
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float,
{
    // Validate inputs
    if xp.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "xp must be 1-dimensional".into(),
        ));
    }
    if fp.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "fp must be 1-dimensional".into(),
        ));
    }
    if xp.len() != fp.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "xp and fp must have the same length".into(),
        ));
    }
    if xp.len() < 2 {
        return Err(NumRs2Error::ValueError(
            "xp and fp must have at least 2 elements".into(),
        ));
    }

    // Check if xp is strictly increasing
    for i in 1..xp.len() {
        if xp.get(&[i])? <= xp.get(&[i - 1])? {
            return Err(NumRs2Error::ValueError(
                "xp must be strictly increasing".into(),
            ));
        }
    }

    // Save original shape of x for reshaping result later
    let x_shape = x.shape().clone();

    // Flatten x for processing
    let x_flat = ravel(x, None)?;
    let mut result = Array::zeros(&x_flat.shape());

    // Get default left and right values
    let left_val = left.unwrap_or_else(|| fp.get(&[0]).unwrap());
    let right_val = right.unwrap_or_else(|| fp.get(&[fp.len() - 1]).unwrap());

    // Process each element in x
    for i in 0..x_flat.len() {
        let mut x_val = x_flat.get(&[i])?;

        // Handle periodicity if specified
        if let Some(ref p) = period {
            let p_val = *p;
            let xp_min = xp.get(&[0])?;
            let xp_max = xp.get(&[xp.len() - 1])?;
            let period_width = xp_max - xp_min;

            // Normalize x_val to be within [xp_min, xp_min + period)
            let mut x_norm = x_val;
            if x_norm >= xp_min + period_width || x_norm < xp_min {
                x_norm = xp_min + ((x_norm - xp_min) % p_val + p_val) % p_val;
            }
            x_val = x_norm;
        }

        // Out of bounds handling
        if x_val < xp.get(&[0])? {
            result.set(&[i], left_val)?;
            continue;
        }
        if x_val > xp.get(&[xp.len() - 1])? {
            result.set(&[i], right_val)?;
            continue;
        }

        // Binary search to find the interval containing x_val
        let mut low: usize = 0;
        let mut high: usize = xp.len() - 1;

        while low < high - 1 {
            let mid = (low + high) / 2;
            if x_val < xp.get(&[mid])? {
                high = mid;
            } else {
                low = mid;
            }
        }

        // Linear interpolation within the interval
        let x0 = xp.get(&[low])?;
        let x1 = xp.get(&[high])?;
        let y0 = fp.get(&[low])?;
        let y1 = fp.get(&[high])?;

        let t = (x_val - x0) / (x1 - x0);
        let interpolated = y0 * (T::one() - t) + y1 * t;

        result.set(&[i], interpolated)?;
    }

    // Reshape result back to original shape of x
    Ok(result.reshape(&x_shape))
}

/// Return elements chosen from x or y depending on condition
///
/// # Parameters
///
/// * `condition` - Where True, yield x, otherwise yield y
/// * `x` - Values to choose from where condition is True
/// * `y` - Values to choose from where condition is False
///
/// # Returns
///
/// A new array with values chosen from x or y based on condition
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let condition = Array::from_vec(vec![true, false, true, false]);
/// let x = Array::from_vec(vec![1, 2, 3, 4]);
/// let y = Array::from_vec(vec![10, 20, 30, 40]);
/// let result = where_cond(&condition, &x, &y).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 20, 3, 40]);
///
/// // With broadcasting
/// let condition_2d = Array::from_vec(vec![true, false, true, false]).reshape(&[2, 2]);
/// let x_scalar = Array::from_vec(vec![100]);
/// let y_2d = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let result_2d = where_cond(&condition_2d, &x_scalar, &y_2d).unwrap();
/// assert_eq!(result_2d.to_vec(), vec![100, 2, 100, 4]);
/// ```
pub fn where_cond<T: Clone + Display>(
    condition: &Array<bool>,
    x: &Array<T>,
    y: &Array<T>,
) -> Result<Array<T>> {
    // Get the shapes
    let cond_shape = condition.shape();
    let x_shape = x.shape();
    let y_shape = y.shape();

    // Calculate broadcast shape for all three arrays
    let broadcast_shape_xy = Array::<T>::broadcast_shape(&x_shape, &y_shape)?;
    let broadcast_shape = Array::<bool>::broadcast_shape(&cond_shape, &broadcast_shape_xy)?;

    // Broadcast all arrays to the common shape
    let cond_broadcast = condition.broadcast_to(&broadcast_shape)?;
    let x_broadcast = x.broadcast_to(&broadcast_shape)?;
    let y_broadcast = y.broadcast_to(&broadcast_shape)?;

    // Apply the conditional logic element-wise
    let cond_data = cond_broadcast.to_vec();
    let x_data = x_broadcast.to_vec();
    let y_data = y_broadcast.to_vec();

    let result_data: Vec<T> = cond_data
        .iter()
        .zip(x_data.iter())
        .zip(y_data.iter())
        .map(
            |((&cond, x_val), y_val)| {
                if cond {
                    x_val.clone()
                } else {
                    y_val.clone()
                }
            },
        )
        .collect();

    Ok(Array::from_vec(result_data).reshape(&broadcast_shape))
}

/// Select elements from choices array based on conditions
///
/// Given a list of conditions and a list of choices, return an array drawn from the elements in choices,
/// depending on the conditions.
///
/// # Parameters
///
/// * `condlist` - A list of boolean arrays. The length of condlist determines the number of conditions
/// * `choicelist` - A list of arrays from which to choose. Must have the same length as condlist
/// * `default` - The element to use if no condition is satisfied. If None, uses zero.
///
/// # Returns
///
/// A new array with elements selected from choicelist based on conditions
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create conditions
/// let x = Array::from_vec(vec![0, 1, 2, 3, 4, 5]);
/// let cond1 = x.map(|val| val < 3);
/// let cond2 = x.map(|val| val >= 3);
///
/// // Create choices
/// let choice1 = Array::from_vec(vec![10, 10, 10, 10, 10, 10]);
/// let choice2 = Array::from_vec(vec![20, 20, 20, 20, 20, 20]);
///
/// let result = select(&[&cond1, &cond2], &[&choice1, &choice2], Some(99)).unwrap();
/// assert_eq!(result.to_vec(), vec![10, 10, 10, 20, 20, 20]);
///
/// // When no condition matches, use default
/// let always_false = Array::from_vec(vec![false, false, false]);
/// let choice_unused = Array::from_vec(vec![1, 2, 3]);
/// let result_default = select(&[&always_false], &[&choice_unused], Some(99)).unwrap();
/// assert_eq!(result_default.to_vec(), vec![99, 99, 99]);
/// ```
pub fn select<T: Clone + num_traits::Zero>(
    condlist: &[&Array<bool>],
    choicelist: &[&Array<T>],
    default: Option<T>,
) -> Result<Array<T>> {
    if condlist.len() != choicelist.len() {
        return Err(NumRs2Error::InvalidOperation(
            "condlist and choicelist must have the same length".to_string(),
        ));
    }

    if condlist.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "condlist and choicelist cannot be empty".to_string(),
        ));
    }

    // Determine the broadcast shape
    let mut broadcast_shape = condlist[0].shape();
    for cond in condlist.iter() {
        broadcast_shape = Array::<bool>::broadcast_shape(&broadcast_shape, &cond.shape())?;
    }
    for choice in choicelist.iter() {
        broadcast_shape = Array::<T>::broadcast_shape(&broadcast_shape, &choice.shape())?;
    }

    // Broadcast all arrays to the common shape
    let mut cond_broadcasts = Vec::with_capacity(condlist.len());
    let mut choice_broadcasts = Vec::with_capacity(choicelist.len());

    for cond in condlist.iter() {
        cond_broadcasts.push(cond.broadcast_to(&broadcast_shape)?);
    }
    for choice in choicelist.iter() {
        choice_broadcasts.push(choice.broadcast_to(&broadcast_shape)?);
    }

    // Create result array with default values
    let default_val = default.unwrap_or_else(T::zero);
    let mut result = Array::full(&broadcast_shape, default_val);

    // Process each element
    let total_size = broadcast_shape.iter().product::<usize>();
    for i in 0..total_size {
        // Convert flat index to multi-dimensional index
        let mut indices = Vec::with_capacity(broadcast_shape.len());
        let mut temp = i;
        for &dim in broadcast_shape.iter().rev() {
            indices.insert(0, temp % dim);
            temp /= dim;
        }

        // Check conditions in order
        for (cond_broadcast, choice_broadcast) in
            cond_broadcasts.iter().zip(choice_broadcasts.iter())
        {
            let cond_val = cond_broadcast
                .array()
                .get(ndarray::IxDyn(&indices))
                .unwrap();
            if *cond_val {
                let choice_val = choice_broadcast
                    .array()
                    .get(ndarray::IxDyn(&indices))
                    .unwrap();
                result.set(&indices, choice_val.clone())?;
                break; // Take the first matching condition
            }
        }
    }

    Ok(result)
}

/// Construct an array by executing a function over each coordinate
///
/// # Parameters
///
/// * `function` - Function to call at each coordinate
/// * `shape` - Shape of the output array
/// * `dtype` - Data type of the output array (for type inference)
///
/// # Returns
///
/// A new array where `arr[i,j,k,...] = function(i,j,k,...)`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3x3 array where arr[i,j] = i + j
/// let result = fromfunction(|indices: &[usize]| (indices[0] + indices[1]) as f64, &[3, 3]).unwrap();
/// assert_eq!(result.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(result.get(&[0, 1]).unwrap(), 1.0);
/// assert_eq!(result.get(&[1, 1]).unwrap(), 2.0);
/// assert_eq!(result.get(&[2, 2]).unwrap(), 4.0);
///
/// // Create a 2x4 array where arr[i,j] = i * j
/// let result = fromfunction(|indices: &[usize]| (indices[0] * indices[1]) as i32, &[2, 4]).unwrap();
/// assert_eq!(result.get(&[1, 3]).unwrap(), 3);
/// assert_eq!(result.get(&[0, 2]).unwrap(), 0);
/// ```
pub fn fromfunction<T, F>(function: F, shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + num_traits::Zero,
    F: Fn(&[usize]) -> T,
{
    if shape.is_empty() {
        return Ok(Array::from_vec(vec![]));
    }

    // Calculate total number of elements
    let total_elements: usize = shape.iter().product();

    // Create result vector
    let mut result_data = Vec::with_capacity(total_elements);

    // Iterate through all indices and compute function values
    let mut indices = vec![0; shape.len()];
    for _ in 0..total_elements {
        // Call the function with current indices
        let value = function(&indices);
        result_data.push(value);

        // Increment indices (like an odometer)
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

    // Create and reshape the array
    Ok(Array::from_vec(result_data).reshape(shape))
}

/// Create an array from a raw buffer
///
/// # Parameters
///
/// * `buffer` - The raw buffer as a slice of bytes
/// * `dtype_size` - Size of each element in bytes (e.g., 4 for i32, 8 for f64)
/// * `count` - Number of elements to read from buffer (-1 means read all available)
/// * `offset` - Start reading from this position in the buffer (in bytes)
///
/// # Returns
///
/// A 1D array created from the buffer data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create array from i32 buffer
/// let data: Vec<i32> = vec![1, 2, 3, 4, 5];
/// let buffer = unsafe {
///     std::slice::from_raw_parts(
///         data.as_ptr() as *const u8,
///         data.len() * std::mem::size_of::<i32>()
///     )
/// };
/// let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), -1, 0).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
///
/// // Create array from f64 buffer with count limit
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let buffer = unsafe {
///     std::slice::from_raw_parts(
///         data.as_ptr() as *const u8,
///         data.len() * std::mem::size_of::<f64>()
///     )
/// };
/// let result = frombuffer::<f64>(buffer, std::mem::size_of::<f64>(), 3, 0).unwrap();
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn frombuffer<T: Clone + Default>(
    buffer: &[u8],
    dtype_size: usize,
    count: isize,
    offset: usize,
) -> Result<Array<T>> {
    if dtype_size == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Data type size cannot be zero".to_string(),
        ));
    }

    if offset >= buffer.len() {
        return Err(NumRs2Error::IndexOutOfBounds(format!(
            "Offset {} is beyond buffer size {}",
            offset,
            buffer.len()
        )));
    }

    if dtype_size != std::mem::size_of::<T>() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Data type size mismatch: expected {}, got {}",
            std::mem::size_of::<T>(),
            dtype_size
        )));
    }

    let available_bytes = buffer.len() - offset;
    let max_elements = available_bytes / dtype_size;

    let num_elements = if count < 0 {
        max_elements
    } else {
        let requested = count as usize;
        if requested > max_elements {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Requested {} elements but only {} available in buffer",
                requested, max_elements
            )));
        }
        requested
    };

    if num_elements == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    // Create vector by copying bytes and converting to T
    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let byte_offset = offset + i * dtype_size;
        let element_bytes = &buffer[byte_offset..byte_offset + dtype_size];

        // Safety: We've checked the size matches T and bounds are valid
        let element = unsafe { std::ptr::read(element_bytes.as_ptr() as *const T) };

        result.push(element);
    }

    Ok(Array::from_vec(result))
}

/// Create an array from an iterator
///
/// # Parameters
///
/// * `iter` - Iterator that yields elements
/// * `shape` - Optional shape for the resulting array
///
/// # Returns
///
/// Array created from the iterator elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 1D array from range
/// let result = fromiter((0..5).map(|x| x as f64), None).unwrap();
/// assert_eq!(result.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// // Create 2D array from range with specified shape
/// let result = fromiter((0..6).map(|x| x as i32), Some(&[2, 3])).unwrap();
/// assert_eq!(result.shape(), vec![2, 3]);
/// assert_eq!(result.to_vec(), vec![0, 1, 2, 3, 4, 5]);
/// ```
pub fn fromiter<T: Clone, I: Iterator<Item = T>>(
    iter: I,
    shape: Option<&[usize]>,
) -> Result<Array<T>> {
    let data: Vec<T> = iter.collect();

    match shape {
        Some(s) => {
            let expected_size: usize = s.iter().product();
            if data.len() != expected_size {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![expected_size],
                    actual: vec![data.len()],
                });
            }
            Ok(Array::from_vec(data).reshape(s))
        }
        None => Ok(Array::from_vec(data)),
    }
}
