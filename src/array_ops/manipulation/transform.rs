//! Array transformation operations
//!
//! This module provides functions for transforming array shapes and orientations:
//! - Rolling elements along axes
//! - Flipping arrays along axes
//! - Rotating arrays by 90 degrees
//! - Expanding and squeezing dimensions
//! - Flattening arrays

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::Order;

/// Roll array elements along a specified axis
///
/// # Parameters
///
/// * `array` - Array to roll
/// * `shift` - The number of places by which elements are shifted
/// * `axis` - The axis along which elements are shifted. If `None`, the array is flattened,
///   then shifted, and finally reshaped to the original shape.
///
/// # Returns
///
/// * Array with the same shape as input, but with elements rolled along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Roll elements by 2 in 1D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let rolled = roll(&a, 2, None).expect("operation should succeed");
/// assert_eq!(rolled.to_vec(), vec![4, 5, 1, 2, 3]);
///
/// // Roll elements by -1 along axis 0 in 2D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let rolled = roll(&b, -1, Some(0)).expect("operation should succeed");
/// assert_eq!(rolled.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn roll<T: Clone + Send + Sync>(
    array: &Array<T>,
    shift: isize,
    axis: Option<usize>,
) -> Result<Array<T>> {
    use scirs2_core::parallel_ops::*;

    const PARALLEL_THRESHOLD: usize = 10000;

    // If array is empty, return a copy
    if array.size() == 0 {
        return Ok(array.clone());
    }

    let shape = array.shape();

    match axis {
        Some(ax) => {
            // Validate axis
            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            // Get the size of the specified axis
            let axis_size = shape[ax];

            // Handle case where axis size is 0 or 1 (no rolling needed)
            if axis_size <= 1 {
                return Ok(array.clone());
            }

            // Convert shift to a positive value within range [0, axis_size)
            let shift_mod =
                ((shift % axis_size as isize) + axis_size as isize) % axis_size as isize;

            // No need to roll if the shift is 0
            if shift_mod == 0 {
                return Ok(array.clone());
            }

            // Calculate the sizes of the pre-axis, axis, and post-axis dimensions
            let pre_axis_size: usize = shape.iter().take(ax).product();
            let post_axis_size: usize = shape.iter().skip(ax + 1).product();

            // Create a flat copy of the array data
            let array_vec = array.to_vec();
            let total_size = array_vec.len();

            // Use parallel processing for large arrays
            if is_parallel_enabled() && total_size >= PARALLEL_THRESHOLD {
                // Parallel roll - iterate over all output indices in parallel
                let result_vec: Vec<T> = (0..total_size)
                    .into_par_iter()
                    .map(|dst_idx| {
                        // Decompose dst_idx into (i_pre, dst_axis_idx, i_post)
                        let pre_stride = axis_size * post_axis_size;
                        let i_pre = dst_idx / pre_stride;
                        let remainder = dst_idx % pre_stride;
                        let dst_axis_idx = remainder / post_axis_size;
                        let i_post = remainder % post_axis_size;

                        // Reverse the shift to find source axis index
                        let src_axis_idx =
                            (dst_axis_idx + axis_size - shift_mod as usize) % axis_size;
                        let src_idx = i_pre * pre_stride + src_axis_idx * post_axis_size + i_post;

                        array_vec[src_idx].clone()
                    })
                    .collect();

                Ok(Array::from_vec(result_vec).reshape(&shape))
            } else {
                // Sequential roll for small arrays - optimized with pre-computed indices
                let first_elem = array
                    .array()
                    .first()
                    .ok_or_else(|| NumRs2Error::InvalidOperation("Array is empty".into()))?
                    .clone();

                let mut result = Array::full(&shape, first_elem);
                let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                    NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
                })?;

                // Roll with optimized loop order for better cache locality
                let axis_stride = post_axis_size;
                let pre_stride = axis_size * post_axis_size;

                for i_pre in 0..pre_axis_size {
                    let base_pre = i_pre * pre_stride;
                    for i_axis in 0..axis_size {
                        let dst_axis_idx = (i_axis + shift_mod as usize) % axis_size;
                        let src_base = base_pre + i_axis * axis_stride;
                        let dst_base = base_pre + dst_axis_idx * axis_stride;

                        // Copy entire post_axis chunk - more cache friendly
                        result_vec[dst_base..(post_axis_size + dst_base)]
                            .clone_from_slice(&array_vec[src_base..(post_axis_size + src_base)]);
                    }
                }

                Ok(result)
            }
        }
        None => {
            // Flatten the array, roll, and then reshape back
            let array_vec = array.to_vec();
            let size = array_vec.len();

            // No need to roll if size is 0 or 1
            if size <= 1 {
                return Ok(array.clone());
            }

            // Convert shift to a positive value within range [0, size)
            let shift_mod = ((shift % size as isize) + size as isize) % size as isize;
            let shift_usize = shift_mod as usize;

            // No need to roll if the shift is 0
            if shift_mod == 0 {
                return Ok(array.clone());
            }

            // Use rotate-based approach for efficiency
            // For small arrays, use a simple copy approach
            // For large arrays, use parallel processing
            if is_parallel_enabled() && size >= PARALLEL_THRESHOLD {
                // Parallel roll
                let result_vec: Vec<T> = (0..size)
                    .into_par_iter()
                    .map(|i| {
                        let src_idx = (i + size - shift_usize) % size;
                        array_vec[src_idx].clone()
                    })
                    .collect();

                let result_array = Array::from_vec(result_vec);
                Ok(result_array.reshape(&shape))
            } else {
                // Sequential roll using efficient block copy
                let mut result_vec = vec![array_vec[0].clone(); size];

                // Copy the two contiguous blocks
                // [0..shift_mod] comes from [size-shift_mod..size]
                // [shift_mod..size] comes from [0..size-shift_mod]
                for i in 0..size {
                    let dst_idx = (i + shift_usize) % size;
                    result_vec[dst_idx] = array_vec[i].clone();
                }

                let result_array = Array::from_vec(result_vec);
                Ok(result_array.reshape(&shape))
            }
        }
    }
}

/// Reverse the order of elements in an array along the specified axis
///
/// # Parameters
///
/// * `array` - Array to flip
/// * `axis` - The axis along which to flip. If `None`, all axes are flipped.
///
/// # Returns
///
/// * Array with the same shape as input, but with elements flipped along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Flip a 1D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let flipped = flip(&a, None).expect("operation should succeed");
/// assert_eq!(flipped.to_vec(), vec![5, 4, 3, 2, 1]);
///
/// // Flip a 2D array along axis 0
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flipped = flip(&b, Some(0)).expect("operation should succeed");
/// assert_eq!(flipped.to_vec(), vec![4, 5, 6, 1, 2, 3]);
///
/// // Flip a 2D array along axis 1
/// let c = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flipped = flip(&c, Some(1)).expect("operation should succeed");
/// assert_eq!(flipped.to_vec(), vec![3, 2, 1, 6, 5, 4]);
/// ```
pub fn flip<T: Clone>(array: &Array<T>, axis: Option<usize>) -> Result<Array<T>> {
    // If array is empty, return a copy
    if array.size() == 0 {
        return Ok(array.clone());
    }

    let shape = array.shape();

    match axis {
        Some(ax) => {
            // Validate axis
            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            // Get the size of the specified axis
            let axis_size = shape[ax];

            // Handle case where axis size is 0 or 1 (no flipping needed)
            if axis_size <= 1 {
                return Ok(array.clone());
            }

            // Create a result array filled with the first element as a placeholder
            let first_elem = array
                .array()
                .first()
                .ok_or_else(|| NumRs2Error::InvalidOperation("Array is empty".into()))?
                .clone();

            let mut result = Array::full(&shape, first_elem);

            // Calculate the sizes of the pre-axis, axis, and post-axis dimensions
            let pre_axis_size: usize = shape.iter().take(ax).product();
            let post_axis_size: usize = shape.iter().skip(ax + 1).product();

            // Create a flat copy of the array data
            let array_vec = array.to_vec();
            let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
            })?;

            // Flip the array elements along the specified axis
            for i_pre in 0..pre_axis_size {
                for i_axis in 0..axis_size {
                    for i_post in 0..post_axis_size {
                        // Calculate source index
                        let src_axis_idx = i_axis;
                        let src_idx = i_pre * (axis_size * post_axis_size)
                            + src_axis_idx * post_axis_size
                            + i_post;

                        // Calculate destination index with flip (reversing along the axis)
                        let dst_axis_idx = axis_size - 1 - i_axis;
                        let dst_idx = i_pre * (axis_size * post_axis_size)
                            + dst_axis_idx * post_axis_size
                            + i_post;

                        result_vec[dst_idx] = array_vec[src_idx].clone();
                    }
                }
            }

            Ok(result)
        }
        None => {
            // Flip along all axes by recursively flipping along each axis
            let mut result = array.clone();

            for ax in 0..shape.len() {
                result = flip(&result, Some(ax))?;
            }

            Ok(result)
        }
    }
}

/// Flip array in the up/down direction (along axis 0)
///
/// # Parameters
///
/// * `array` - Array to flip
///
/// # Returns
///
/// * Array with the same shape as input, but with elements flipped along axis 0
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Flip a 2D array in the up/down direction
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flipped = flipud(&a).expect("operation should succeed");
/// assert_eq!(flipped.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn flipud<T: Clone>(array: &Array<T>) -> Result<Array<T>> {
    // Ensure array is at least 1D
    if array.ndim() == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Input must be at least 1-dimensional".into(),
        ));
    }

    // Flip along axis 0
    flip(array, Some(0))
}

/// Flip array in the left/right direction (along axis 1)
///
/// # Parameters
///
/// * `array` - Array to flip
///
/// # Returns
///
/// * Array with the same shape as input, but with elements flipped along axis 1
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Flip a 2D array in the left/right direction
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flipped = fliplr(&a).expect("operation should succeed");
/// assert_eq!(flipped.to_vec(), vec![3, 2, 1, 6, 5, 4]);
/// ```
pub fn fliplr<T: Clone>(array: &Array<T>) -> Result<Array<T>> {
    // Ensure array is at least 2D
    if array.ndim() < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "Input must be at least 2-dimensional".into(),
        ));
    }

    // Flip along axis 1
    flip(array, Some(1))
}

/// Rotate an array by 90 degrees in the plane specified by axes
///
/// # Parameters
///
/// * `array` - Array to rotate
/// * `k` - Number of times to rotate by 90 degrees. Default is 1.
/// * `axes` - The plane to rotate in. By default, the rotation is in the first two axes.
///
/// # Returns
///
/// * Rotated array with the same shape as input, with axes transposed and flipped appropriately
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2x3 array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Rotate once by 90 degrees (counterclockwise)
/// let rotated = rot90(&a, 1, None).expect("operation should succeed");
/// assert_eq!(rotated.shape(), vec![3, 2]);
/// assert_eq!(rotated.to_vec(), vec![5, 6, 3, 4, 1, 2]);
///
/// // Rotate twice by 90 degrees (180 degrees)
/// let rotated = rot90(&a, 2, None).expect("operation should succeed");
/// assert_eq!(rotated.shape(), vec![2, 3]);
/// assert_eq!(rotated.to_vec(), vec![6, 5, 4, 3, 2, 1]);
/// ```
pub fn rot90<T: Clone>(
    array: &Array<T>,
    k: impl Into<Option<i32>>,
    axes: impl Into<Option<(usize, usize)>>,
) -> Result<Array<T>> {
    // Get the number of rotations and the rotation plane
    let k = k.into().unwrap_or(1);
    let axes = axes.into().unwrap_or((0, 1));

    let ndim = array.ndim();

    // Validate axes
    if axes.0 >= ndim || axes.1 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes ({}, {}) out of bounds for array of dimension {}",
            axes.0, axes.1, ndim
        )));
    }

    if axes.0 == axes.1 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Axes ({}, {}) must be different",
            axes.0, axes.1
        )));
    }

    // Normalize k to be in range [0, 3]
    let k = ((k % 4) + 4) % 4;

    // If k is 0, return a copy of the input array
    if k == 0 {
        return Ok(array.clone());
    }

    // Create a view of the array
    let mut result = array.clone();

    // If k is 2, we can just flip along both axes
    if k == 2 {
        result = flip(&result, Some(axes.0))?;
        result = flip(&result, Some(axes.1))?;
        return Ok(result);
    }

    // For k=1 or k=3, we need to transpose and flip

    // Use the Array::transpose_axis implementation to transpose the array
    result = result.transpose_axis(axes.0, axes.1);

    // Then flip along the appropriate axis
    if k == 1 {
        // For 90 degrees counterclockwise, flip along the second axis
        result = flip(&result, Some(axes.0))?;
    } else if k == 3 {
        // For 270 degrees counterclockwise (90 clockwise), flip along the first axis
        result = flip(&result, Some(axes.1))?;
    }

    Ok(result)
}

/// Expand the shape of an array by inserting a new axis at the specified position
///
/// # Parameters
///
/// * `array` - The input array
/// * `axis` - Position in the expanded array where the new axis is placed
///
/// # Returns
///
/// Array with an additional dimension of size 1 inserted at the specified position
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Expand a 1D array to 2D
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let expanded = expand_dims(&a, 0).expect("operation should succeed");
/// assert_eq!(expanded.shape(), vec![1, 3]);
///
/// // Insert axis at position 1
/// let expanded = expand_dims(&a, 1).expect("operation should succeed");
/// assert_eq!(expanded.shape(), vec![3, 1]);
/// ```
pub fn expand_dims<T: Clone>(array: &Array<T>, axis: usize) -> Result<Array<T>> {
    let shape = array.shape();

    if axis > shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            shape.len()
        )));
    }

    // Create a new shape with an extra dimension
    let mut new_shape = shape.clone();
    new_shape.insert(axis, 1);

    // Reshape the array
    Ok(array.reshape(&new_shape))
}

/// Remove axes of length 1 from the array
///
/// # Parameters
///
/// * `array` - The input array
/// * `axis` - Axis to squeeze. If `None`, all axes of length 1 are removed.
///
/// # Returns
///
/// Array with axes of length 1 removed
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Remove all axes of length 1
/// let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3, 1]);
/// let squeezed = squeeze(&a, None).expect("operation should succeed");
/// assert_eq!(squeezed.shape(), vec![3]);
///
/// // Remove specific axis
/// let b = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
/// let squeezed = squeeze(&b, Some(0)).expect("operation should succeed");
/// assert_eq!(squeezed.shape(), vec![3]);
/// ```
pub fn squeeze<T: Clone>(array: &Array<T>, axis: Option<usize>) -> Result<Array<T>> {
    let shape = array.shape();

    match axis {
        Some(ax) => {
            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            if shape[ax] != 1 {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Cannot squeeze axis {} with size {}",
                    ax, shape[ax]
                )));
            }

            let mut new_shape = shape.clone();
            new_shape.remove(ax);

            Ok(array.reshape(&new_shape))
        }
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
/// let flat = ravel(&a, None).expect("operation should succeed");
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
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Order must be 'C' or 'F', got '{}'",
                order_val
            )))
        }
    };

    // Create a flat view with the specified order
    let flat_data = match nd_order {
        Order::RowMajor => array.array().iter().cloned().collect::<Vec<_>>(),
        Order::ColumnMajor => {
            // Transpose the array and then collect elements
            let transposed = array.transpose();
            transposed.array().iter().cloned().collect::<Vec<_>>()
        }
        _ => {
            // This should never happen, but we need to handle the non-exhaustive enum
            return Err(NumRs2Error::InvalidOperation(
                "Unsupported memory order".to_string(),
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
/// let flat = flatten(&a, None).expect("operation should succeed");
/// assert_eq!(flat.shape(), vec![6]);
/// assert_eq!(flat.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn flatten<T: Clone>(array: &Array<T>, order: Option<char>) -> Result<Array<T>> {
    // flatten always returns a copy, so we can just use ravel and clone
    ravel(array, order)
}
