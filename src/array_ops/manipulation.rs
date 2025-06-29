use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use ndarray::Order;
use num_traits::Zero;

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
/// let rolled = roll(&a, 2, None).unwrap();
/// assert_eq!(rolled.to_vec(), vec![4, 5, 1, 2, 3]);
///
/// // Roll elements by -1 along axis 0 in 2D array
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let rolled = roll(&b, -1, Some(0)).unwrap();
/// assert_eq!(rolled.to_vec(), vec![4, 5, 6, 1, 2, 3]);
/// ```
pub fn roll<T: Clone>(array: &Array<T>, shift: isize, axis: Option<usize>) -> Result<Array<T>> {
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

            // Roll the array elements along the specified axis
            for i_pre in 0..pre_axis_size {
                for i_axis in 0..axis_size {
                    for i_post in 0..post_axis_size {
                        // Calculate source index
                        let src_axis_idx = i_axis;
                        let src_idx = i_pre * (axis_size * post_axis_size)
                            + src_axis_idx * post_axis_size
                            + i_post;

                        // Calculate destination index with roll
                        let dst_axis_idx = (i_axis + shift_mod as usize) % axis_size;
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
            // Flatten the array, roll, and then reshape back
            let array_vec = array.to_vec();
            let size = array_vec.len();

            // No need to roll if size is 0 or 1
            if size <= 1 {
                return Ok(array.clone());
            }

            // Convert shift to a positive value within range [0, size)
            let shift_mod = ((shift % size as isize) + size as isize) % size as isize;

            // No need to roll if the shift is 0
            if shift_mod == 0 {
                return Ok(array.clone());
            }

            let mut result_vec = vec![array_vec[0].clone(); size];

            // Roll the entire flattened array
            #[allow(clippy::needless_range_loop)]
            for i in 0..size {
                let dst_idx = (i + shift_mod as usize) % size;
                result_vec[dst_idx] = array_vec[i].clone();
            }

            // Reshape the result back to the original shape
            let result_array = Array::from_vec(result_vec);
            Ok(result_array.reshape(&shape))
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
/// let flipped = flip(&a, None).unwrap();
/// assert_eq!(flipped.to_vec(), vec![5, 4, 3, 2, 1]);
///
/// // Flip a 2D array along axis 0
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flipped = flip(&b, Some(0)).unwrap();
/// assert_eq!(flipped.to_vec(), vec![4, 5, 6, 1, 2, 3]);
///
/// // Flip a 2D array along axis 1
/// let c = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let flipped = flip(&c, Some(1)).unwrap();
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
/// let flipped = flipud(&a).unwrap();
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
/// let flipped = fliplr(&a).unwrap();
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
/// let rotated = rot90(&a, 1, None).unwrap();
/// assert_eq!(rotated.shape(), vec![3, 2]);
/// assert_eq!(rotated.to_vec(), vec![5, 6, 3, 4, 1, 2]);
///
/// // Rotate twice by 90 degrees (180 degrees)
/// let rotated = rot90(&a, 2, None).unwrap();
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
/// let expanded = expand_dims(&a, 0).unwrap();
/// assert_eq!(expanded.shape(), vec![1, 3]);
///
/// // Insert axis at position 1
/// let expanded = expand_dims(&a, 1).unwrap();
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
/// let squeezed = squeeze(&a, None).unwrap();
/// assert_eq!(squeezed.shape(), vec![3]);
///
/// // Remove specific axis
/// let b = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
/// let squeezed = squeeze(&b, Some(0)).unwrap();
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
/// let flat = flatten(&a, None).unwrap();
/// assert_eq!(flat.shape(), vec![6]);
/// assert_eq!(flat.to_vec(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn flatten<T: Clone>(array: &Array<T>, order: Option<char>) -> Result<Array<T>> {
    // flatten always returns a copy, so we can just use ravel and clone
    ravel(array, order)
}

/// Delete sub-arrays along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `indices` - Indicate which sub-arrays to remove (can be slice, integer, or array of integers)
/// * `axis` - The axis along which to delete. If None, array is flattened before operation
///
/// # Returns
///
/// A copy of array with the elements specified by indices removed
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Delete element at index 1 from 1D array
/// let a = Array::from_vec(vec![0, 1, 2, 3, 4]);
/// let result = delete(&a, &[1], None).unwrap();
/// assert_eq!(result.to_vec(), vec![0, 2, 3, 4]);
///
/// // Delete multiple elements
/// let result = delete(&a, &[1, 3], None).unwrap();
/// assert_eq!(result.to_vec(), vec![0, 2, 4]);
///
/// // Delete from 2D array along axis 0
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[3, 2]);
/// let result = delete(&b, &[1], Some(0)).unwrap();
/// assert_eq!(result.shape(), vec![2, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 6]);
/// ```
pub fn delete<T: Clone + Zero>(
    array: &Array<T>,
    indices: &[usize],
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        Some(ax) => {
            // Delete along specified axis
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Check if indices are valid
            for &idx in indices {
                if idx >= axis_size {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Index {} out of bounds for axis {} with size {}",
                        idx, ax, axis_size
                    )));
                }
            }

            // Create a sorted, unique list of indices to delete
            let mut delete_indices = indices.to_vec();
            delete_indices.sort_unstable();
            delete_indices.dedup();

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = axis_size - delete_indices.len();

            if new_shape[ax] == 0 {
                // Result would be empty along this axis
                return Ok(Array::zeros(&new_shape));
            }

            // Create result array
            let mut result_data = Vec::with_capacity(new_shape.iter().product());

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through all positions
            let total_size: usize = shape.iter().product();
            for i in 0..total_size {
                // Convert flat index to multi-dimensional indices
                let mut indices_arr = vec![0; shape.len()];
                let mut temp = i;
                for j in 0..shape.len() {
                    indices_arr[j] = temp / strides[j];
                    temp %= strides[j];
                }

                // Check if this position should be deleted
                let axis_pos = indices_arr[ax];
                if !delete_indices.contains(&axis_pos) {
                    result_data.push(array.get(&indices_arr)?);
                }
            }

            Ok(Array::from_vec(result_data).reshape(&new_shape))
        }
        None => {
            // Flatten array and delete from flattened version
            let flat = array.to_vec();
            let flat_size = flat.len();

            // Check if indices are valid
            for &idx in indices {
                if idx >= flat_size {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Index {} out of bounds for flattened array with size {}",
                        idx, flat_size
                    )));
                }
            }

            // Create a sorted, unique list of indices to delete
            let mut delete_indices = indices.to_vec();
            delete_indices.sort_unstable();
            delete_indices.dedup();

            // Create result by including only non-deleted elements
            let mut result_data = Vec::with_capacity(flat_size - delete_indices.len());
            let mut del_idx = 0;

            for (i, val) in flat.iter().enumerate() {
                if del_idx < delete_indices.len() && i == delete_indices[del_idx] {
                    del_idx += 1;
                } else {
                    result_data.push(val.clone());
                }
            }

            Ok(Array::from_vec(result_data))
        }
    }
}

/// Insert values along the given axis before the given indices
///
/// # Parameters
///
/// * `array` - Input array
/// * `indices` - Indices before which values is inserted (single index or slice)
/// * `values` - Values to insert into array
/// * `axis` - Axis along which to insert values. If None, array is flattened first
///
/// # Returns
///
/// A copy of array with values inserted
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Insert single value into 1D array
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = insert(&a, &[1], &[99], None).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 99, 2, 3]);
///
/// // Insert multiple values
/// let result = insert(&a, &[1, 2], &[99, 100], None).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 99, 2, 100, 3]);
///
/// // Insert into 2D array along axis
/// let b = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let values = vec![5, 6];
/// let result = insert(&b, &[1], &values, Some(0)).unwrap();
/// assert_eq!(result.shape(), vec![3, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 6, 3, 4]);
/// ```
pub fn insert<T: Clone + Zero>(
    array: &Array<T>,
    indices: &[usize],
    values: &[T],
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        Some(ax) => {
            // Insert along specified axis
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Sort indices for proper insertion
            let mut sorted_indices: Vec<(usize, usize)> = indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| (idx, i))
                .collect();
            sorted_indices.sort_by_key(|&(idx, _)| idx);

            // Calculate how many elements we're inserting along the axis
            let values_per_insertion = if values.len() == 1 {
                // Single value to be inserted at all positions
                1
            } else if values.len() == indices.len() {
                // One value per index
                1
            } else {
                // Values should be a multiple of indices.len()
                if values.len() % indices.len() != 0 {
                    return Err(NumRs2Error::InvalidOperation(
                        "Values length must be 1, equal to indices length, or a multiple of indices length".into()
                    ));
                }
                values.len() / indices.len()
            };

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = axis_size + indices.len() * values_per_insertion;

            // Calculate total size of sub-array perpendicular to axis
            let mut sub_size = 1;
            for (i, &dim) in shape.iter().enumerate() {
                if i != ax {
                    sub_size *= dim;
                }
            }

            // Build result array
            let mut result_data = Vec::with_capacity(new_shape.iter().product());

            // Process each position along the axis
            let mut src_pos = 0;
            let mut insert_idx = 0;

            for _new_pos in 0..new_shape[ax] {
                // Check if we should insert values at this position
                let mut should_insert = false;
                let mut which_insert = 0;

                if insert_idx < sorted_indices.len() {
                    let (idx, orig_order) = sorted_indices[insert_idx];
                    if src_pos == idx {
                        should_insert = true;
                        which_insert = orig_order;
                    }
                }

                if should_insert {
                    // Insert values
                    for val_idx in 0..values_per_insertion {
                        // Copy the sub-array structure but with inserted values
                        for _sub_idx in 0..sub_size {
                            let value_idx = if values.len() == 1 {
                                0
                            } else if values.len() == indices.len() {
                                which_insert
                            } else {
                                which_insert * values_per_insertion + val_idx
                            };

                            result_data.push(values[value_idx].clone());
                        }
                    }
                    insert_idx += 1;
                } else if src_pos < axis_size {
                    // Copy from original array
                    // Calculate strides for copying
                    let mut indices_arr = vec![0; shape.len()];

                    for sub_idx in 0..sub_size {
                        // Convert sub_idx to multi-dimensional indices
                        let mut temp = sub_idx;
                        for i in (0..shape.len()).rev() {
                            if i == ax {
                                indices_arr[i] = src_pos;
                            } else {
                                let dim = shape[i];
                                if i < shape.len() - 1 {
                                    indices_arr[i] = temp % dim;
                                    temp /= dim;
                                } else {
                                    indices_arr[i] = temp;
                                }
                            }
                        }

                        result_data.push(array.get(&indices_arr)?);
                    }
                    src_pos += 1;
                }
            }

            Ok(Array::from_vec(result_data).reshape(&new_shape))
        }
        None => {
            // Flatten array and insert into flattened version
            let flat = array.to_vec();
            let flat_size = flat.len();

            if indices.len() != values.len() && values.len() != 1 {
                return Err(NumRs2Error::InvalidOperation(
                    "For flat insertion, values must have length 1 or match indices length".into(),
                ));
            }

            // Create pairs of (index, value) and sort by index
            let mut insertions: Vec<(usize, T)> = Vec::new();
            for (i, &idx) in indices.iter().enumerate() {
                let val = if values.len() == 1 {
                    values[0].clone()
                } else {
                    values[i].clone()
                };
                insertions.push((idx, val));
            }
            insertions.sort_by_key(|&(idx, _)| idx);

            // Build result
            let mut result_data = Vec::with_capacity(flat_size + insertions.len());
            let mut orig_idx = 0;
            let mut insert_idx = 0;

            for _pos in 0..flat_size + insertions.len() {
                if insert_idx < insertions.len() && insertions[insert_idx].0 == orig_idx {
                    result_data.push(insertions[insert_idx].1.clone());
                    insert_idx += 1;
                } else if orig_idx < flat_size {
                    result_data.push(flat[orig_idx].clone());
                    orig_idx += 1;
                }
            }

            // Append any remaining insertions at the end
            while insert_idx < insertions.len() {
                result_data.push(insertions[insert_idx].1.clone());
                insert_idx += 1;
            }

            Ok(Array::from_vec(result_data))
        }
    }
}

/// Pad an array
///
/// # Parameters
///
/// * `array` - Array to be padded
/// * `pad_width` - Number of values padded to the edges of each axis.
///   For each axis, provide (before, after) padding sizes.
/// * `mode` - Padding mode:
///   - "constant": Pads with a constant value (default 0)
///   - "edge": Pads with the edge values of array
///   - "reflect": Pads with reflection of array mirrored on the first and last values of the axis
///   - "symmetric": Pads with reflection of array mirrored along the edge of the array
///   - "wrap": Pads with the wrap of the vector along the axis
/// * `constant_values` - Used in 'constant' mode. The values to set the padded values for each axis.
///
/// # Returns
///
/// Padded array of same type as input array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Pad 1D array with constant value
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = pad(&a, &[(2, 3)], "constant", Some(0)).unwrap();
/// assert_eq!(result.to_vec(), vec![0, 0, 1, 2, 3, 0, 0, 0]);
///
/// // Pad 2D array with edge values
/// let b = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let result = pad(&b, &[(1, 1), (2, 2)], "edge", None).unwrap();
/// assert_eq!(result.shape(), vec![4, 6]);
/// ```
pub fn pad<T>(
    array: &Array<T>,
    pad_width: &[(usize, usize)],
    mode: &str,
    constant_values: Option<T>,
) -> Result<Array<T>>
where
    T: Clone + Zero,
{
    let shape = array.shape();

    // Validate pad_width
    if pad_width.len() != shape.len() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "pad_width must have same length as array dimensions. Got {} for {} dimensions",
            pad_width.len(),
            shape.len()
        )));
    }

    // Calculate new shape
    let mut new_shape = Vec::with_capacity(shape.len());
    for (i, &dim) in shape.iter().enumerate() {
        let (before, after) = pad_width[i];
        new_shape.push(before + dim + after);
    }

    // Create result array filled with padding value
    let pad_value = match mode {
        "constant" => constant_values.unwrap_or_else(T::zero),
        _ => T::zero(), // Will be overwritten for other modes
    };

    let total_size: usize = new_shape.iter().product();
    let mut result_data = vec![pad_value.clone(); total_size];

    // Calculate strides for both arrays
    let mut old_strides = vec![1; shape.len()];
    let mut new_strides = vec![1; new_shape.len()];

    for i in (0..shape.len() - 1).rev() {
        old_strides[i] = old_strides[i + 1] * shape[i + 1];
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    // Copy original array into the center of the result
    let original_data = array.to_vec();

    for i in 0..original_data.len() {
        // Convert flat index to multi-dimensional indices in original array
        let mut old_indices = vec![0; shape.len()];
        let mut temp = i;
        for j in 0..shape.len() {
            old_indices[j] = temp / old_strides[j];
            temp %= old_strides[j];
        }

        // Calculate corresponding indices in new array
        let mut new_indices = vec![0; new_shape.len()];
        for j in 0..shape.len() {
            new_indices[j] = old_indices[j] + pad_width[j].0;
        }

        // Calculate flat index in new array
        let mut new_flat_idx = 0;
        for j in 0..new_shape.len() {
            new_flat_idx += new_indices[j] * new_strides[j];
        }

        result_data[new_flat_idx] = original_data[i].clone();
    }

    // Apply padding based on mode
    match mode {
        "constant" => {
            // Already filled with constant value
        }
        "edge" => {
            // Pad with edge values
            for axis in 0..shape.len() {
                let (before, after) = pad_width[axis];

                // Pad before
                if before > 0 {
                    for i in 0..total_size {
                        let indices = index_from_flat(i, &new_shape, &new_strides);

                        if indices[axis] < before {
                            // This is in the padding region
                            let mut source_indices = indices.clone();
                            source_indices[axis] = before; // Edge of original data

                            let source_flat = flat_from_index(&source_indices, &new_strides);
                            result_data[i] = result_data[source_flat].clone();
                        }
                    }
                }

                // Pad after
                if after > 0 {
                    for i in 0..total_size {
                        let indices = index_from_flat(i, &new_shape, &new_strides);

                        if indices[axis] >= before + shape[axis] {
                            // This is in the padding region
                            let mut source_indices = indices.clone();
                            source_indices[axis] = before + shape[axis] - 1; // Edge of original data

                            let source_flat = flat_from_index(&source_indices, &new_strides);
                            result_data[i] = result_data[source_flat].clone();
                        }
                    }
                }
            }
        }
        "reflect" => {
            // Pad with reflection (not including edge)
            for axis in 0..shape.len() {
                #[allow(unused_variables)]
                let (before, after) = pad_width[axis];
                let axis_size = shape[axis];

                // Pad before
                for i in 0..total_size {
                    let indices = index_from_flat(i, &new_shape, &new_strides);

                    if indices[axis] < before {
                        // Calculate reflected position
                        let offset = before - indices[axis];
                        let reflected_pos = if offset < axis_size {
                            before + offset
                        } else {
                            // Handle multiple reflections
                            let period = 2 * (axis_size - 1);
                            let _cycles = offset / period;
                            let remainder = offset % period;

                            if remainder < axis_size {
                                before + remainder
                            } else {
                                before + 2 * (axis_size - 1) - remainder
                            }
                        };

                        let mut source_indices = indices.clone();
                        source_indices[axis] = reflected_pos;

                        let source_flat = flat_from_index(&source_indices, &new_strides);
                        result_data[i] = result_data[source_flat].clone();
                    }
                }

                // Pad after
                for i in 0..total_size {
                    let indices = index_from_flat(i, &new_shape, &new_strides);

                    if indices[axis] >= before + axis_size {
                        // Calculate reflected position
                        let offset = indices[axis] - (before + axis_size - 1);
                        let reflected_pos = if offset < axis_size {
                            before + axis_size - 1 - offset
                        } else {
                            // Handle multiple reflections
                            let period = 2 * (axis_size - 1);
                            let _cycles = offset / period;
                            let remainder = offset % period;

                            if remainder < axis_size {
                                before + axis_size - 1 - remainder
                            } else {
                                before + remainder - (axis_size - 1)
                            }
                        };

                        let mut source_indices = indices.clone();
                        source_indices[axis] = reflected_pos;

                        let source_flat = flat_from_index(&source_indices, &new_strides);
                        result_data[i] = result_data[source_flat].clone();
                    }
                }
            }
        }
        "symmetric" => {
            // Pad with reflection (including edge)
            for axis in 0..shape.len() {
                #[allow(unused_variables)]
                let (before, after) = pad_width[axis];
                let axis_size = shape[axis];

                // Pad before
                for i in 0..total_size {
                    let indices = index_from_flat(i, &new_shape, &new_strides);

                    if indices[axis] < before {
                        // Calculate reflected position
                        let offset = before - indices[axis] - 1;
                        let reflected_pos = if offset < axis_size {
                            before + offset
                        } else {
                            // Handle multiple reflections
                            let period = 2 * axis_size;
                            let _cycles = offset / period;
                            let remainder = offset % period;

                            if remainder < axis_size {
                                before + remainder
                            } else {
                                before + 2 * axis_size - remainder - 1
                            }
                        };

                        let mut source_indices = indices.clone();
                        source_indices[axis] = reflected_pos;

                        let source_flat = flat_from_index(&source_indices, &new_strides);
                        result_data[i] = result_data[source_flat].clone();
                    }
                }

                // Pad after
                for i in 0..total_size {
                    let indices = index_from_flat(i, &new_shape, &new_strides);

                    if indices[axis] >= before + axis_size {
                        // Calculate reflected position
                        let offset = indices[axis] - (before + axis_size);
                        let reflected_pos = if offset < axis_size {
                            before + axis_size - 1 - offset
                        } else {
                            // Handle multiple reflections
                            let period = 2 * axis_size;
                            let _cycles = offset / period;
                            let remainder = offset % period;

                            if remainder < axis_size {
                                before + axis_size - 1 - remainder
                            } else {
                                before + remainder - axis_size
                            }
                        };

                        let mut source_indices = indices.clone();
                        source_indices[axis] = reflected_pos;

                        let source_flat = flat_from_index(&source_indices, &new_strides);
                        result_data[i] = result_data[source_flat].clone();
                    }
                }
            }
        }
        "wrap" => {
            // Pad with wrapping
            for axis in 0..shape.len() {
                #[allow(unused_variables)]
                let (before, after) = pad_width[axis];
                let axis_size = shape[axis];

                // Pad before
                for i in 0..total_size {
                    let indices = index_from_flat(i, &new_shape, &new_strides);

                    if indices[axis] < before {
                        // Calculate wrapped position
                        let offset = before - indices[axis];
                        let wrapped_pos = before + axis_size - (offset % axis_size);

                        let mut source_indices = indices.clone();
                        source_indices[axis] = wrapped_pos;

                        let source_flat = flat_from_index(&source_indices, &new_strides);
                        result_data[i] = result_data[source_flat].clone();
                    }
                }

                // Pad after
                for i in 0..total_size {
                    let indices = index_from_flat(i, &new_shape, &new_strides);

                    if indices[axis] >= before + axis_size {
                        // Calculate wrapped position
                        let offset = indices[axis] - (before + axis_size);
                        let wrapped_pos = before + (offset % axis_size);

                        let mut source_indices = indices.clone();
                        source_indices[axis] = wrapped_pos;

                        let source_flat = flat_from_index(&source_indices, &new_strides);
                        result_data[i] = result_data[source_flat].clone();
                    }
                }
            }
        }
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Unknown pad mode: {}. Must be one of: constant, edge, reflect, symmetric, wrap",
                mode
            )));
        }
    }

    Ok(Array::from_vec(result_data).reshape(&new_shape))
}

// Helper functions for pad
fn index_from_flat(flat_idx: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut temp = flat_idx;

    for i in 0..shape.len() {
        indices[i] = temp / strides[i];
        temp %= strides[i];
    }

    indices
}

fn flat_from_index(indices: &[usize], strides: &[usize]) -> usize {
    let mut flat_idx = 0;
    for i in 0..indices.len() {
        flat_idx += indices[i] * strides[i];
    }
    flat_idx
}

/// Trim the leading and/or trailing zeros from a 1-D array
///
/// # Parameters
///
/// * `array` - Input array
/// * `trim` - A string with 'f' representing trim from front and 'b' to trim from back.
///   Default is 'fb', trim zeros from both front and back of the array.
///
/// # Returns
///
/// 1-D array with trimmed zeros
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Trim zeros from both ends
/// let a = Array::from_vec(vec![0, 0, 0, 1, 2, 3, 0, 0, 0]);
/// let result = trim_zeros(&a, None).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2, 3]);
///
/// // Trim only from front
/// let result = trim_zeros(&a, Some("f")).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 0, 0, 0]);
///
/// // Trim only from back
/// let result = trim_zeros(&a, Some("b")).unwrap();
/// assert_eq!(result.to_vec(), vec![0, 0, 0, 1, 2, 3]);
/// ```
pub fn trim_zeros<T>(array: &Array<T>, trim: Option<&str>) -> Result<Array<T>>
where
    T: Clone + Zero + PartialEq,
{
    // Ensure array is 1-D
    if array.ndim() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "trim_zeros requires a 1-D array".into(),
        ));
    }

    let data = array.to_vec();
    if data.is_empty() {
        return Ok(array.clone());
    }

    let trim_str = trim.unwrap_or("fb");
    let trim_front = trim_str.contains('f');
    let trim_back = trim_str.contains('b');

    let mut start = 0;
    let mut end = data.len();

    // Find the first non-zero element from front
    if trim_front {
        for (i, val) in data.iter().enumerate() {
            if !val.is_zero() {
                start = i;
                break;
            }
        }
        // If all elements are zero
        if start == 0 && data[0].is_zero() {
            let all_zero = data.iter().all(|x| x.is_zero());
            if all_zero {
                return Ok(Array::from_vec(vec![]));
            }
        }
    }

    // Find the first non-zero element from back
    if trim_back {
        for (i, val) in data.iter().enumerate().rev() {
            if !val.is_zero() {
                end = i + 1;
                break;
            }
        }
    }

    // If start >= end, all remaining elements are zeros
    if start >= end {
        return Ok(Array::from_vec(vec![]));
    }

    // Create the trimmed array
    let trimmed_data: Vec<T> = data[start..end].to_vec();
    Ok(Array::from_vec(trimmed_data))
}

/// Extract elements from an array that satisfy a condition
///
/// # Parameters
///
/// * `array` - Input array
/// * `condition` - Boolean array with same shape as `array`
///
/// # Returns
///
/// A 1-D array containing the elements from `array` where `condition` is true
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Extract positive values
/// let a = Array::from_vec(vec![-1, 2, -3, 4, -5]);
/// let condition = Array::from_vec(vec![false, true, false, true, false]);
/// let result = extract(&a, &condition).unwrap();
/// assert_eq!(result.to_vec(), vec![2, 4]);
///
/// // Extract from 2D array
/// let b = Array::from_vec(vec![1, -2, 3, -4, 5, -6]).reshape(&[2, 3]);
/// let cond = Array::from_vec(vec![true, false, true, false, true, false]).reshape(&[2, 3]);
/// let result = extract(&b, &cond).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 3, 5]);
/// ```
pub fn extract<T: Clone>(array: &Array<T>, condition: &Array<bool>) -> Result<Array<T>> {
    // Check that arrays have the same shape
    if array.shape() != condition.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Array and condition must have the same shape, got {:?} and {:?}",
            array.shape(),
            condition.shape()
        )));
    }

    // Flatten both arrays
    let array_flat = array.to_vec();
    let condition_flat = condition.to_vec();

    // Extract elements where condition is true
    let mut result_data = Vec::new();
    for (val, &cond) in array_flat.iter().zip(condition_flat.iter()) {
        if cond {
            result_data.push(val.clone());
        }
    }

    Ok(Array::from_vec(result_data))
}

/// Place values into an array according to a condition
///
/// # Parameters
///
/// * `array` - Array to be modified (modified in-place)
/// * `mask` - Boolean array with same shape as `array`
/// * `values` - Values to place where mask is true
///
/// # Returns
///
/// The modified array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Replace negative values with 0
/// let mut a = Array::from_vec(vec![-1, 2, -3, 4, -5]);
/// let mask = Array::from_vec(vec![true, false, true, false, true]);
/// place(&mut a, &mask, &[0, 0, 0]).unwrap();
/// assert_eq!(a.to_vec(), vec![0, 2, 0, 4, 0]);
///
/// // Place values in 2D array
/// let mut b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let mask = Array::from_vec(vec![false, true, false, true, false, true]).reshape(&[2, 3]);
/// place(&mut b, &mask, &[10, 20, 30]).unwrap();
/// assert_eq!(b.to_vec(), vec![1, 10, 3, 20, 5, 30]);
/// ```
pub fn place<T: Clone>(array: &mut Array<T>, mask: &Array<bool>, values: &[T]) -> Result<()> {
    // Check that arrays have the same shape
    if array.shape() != mask.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Array and mask must have the same shape, got {:?} and {:?}",
            array.shape(),
            mask.shape()
        )));
    }

    // Count true values in mask
    let mask_flat = mask.to_vec();
    let true_count = mask_flat.iter().filter(|&&x| x).count();

    // Check that we have the right number of values
    if values.len() != true_count {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Number of values ({}) must match number of true elements in mask ({})",
            values.len(),
            true_count
        )));
    }

    // Get mutable slice of array data
    let array_slice = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    // Place values where mask is true
    let mut value_idx = 0;
    for (i, &mask_val) in mask_flat.iter().enumerate() {
        if mask_val {
            array_slice[i] = values[value_idx].clone();
            value_idx += 1;
        }
    }

    Ok(())
}

/// Replace specified elements of an array with given values
///
/// # Parameters
///
/// * `array` - Array to be modified (modified in-place)
/// * `indices` - Target indices, can be flattened or multi-dimensional
/// * `values` - Values to place at target indices
///
/// # Returns
///
/// The modified array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Put values at specific indices in 1D array
/// let mut a = Array::from_vec(vec![0, 0, 0, 0, 0]);
/// put(&mut a, &[1, 3], &[10, 30]).unwrap();
/// assert_eq!(a.to_vec(), vec![0, 10, 0, 30, 0]);
///
/// // Put with repeating values
/// let mut b = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// put(&mut b, &[0, 2, 4], &[100]).unwrap();
/// assert_eq!(b.to_vec(), vec![100, 2, 100, 4, 100]);
/// ```
pub fn put<T: Clone>(array: &mut Array<T>, indices: &[usize], values: &[T]) -> Result<()> {
    if values.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Values array cannot be empty".into(),
        ));
    }

    // Get mutable slice of array data
    let array_slice = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;
    let array_size = array_slice.len();

    // Check that all indices are valid
    for &idx in indices {
        if idx >= array_size {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Index {} out of bounds for array with size {}",
                idx, array_size
            )));
        }
    }

    // Place values at indices, cycling through values if necessary
    for (i, &idx) in indices.iter().enumerate() {
        array_slice[idx] = values[i % values.len()].clone();
    }

    Ok(())
}

/// Select slices from an array along a given axis
///
/// # Parameters
///
/// * `array` - Array from which to select slices
/// * `condition` - 1-D boolean array that selects which slices to return
/// * `axis` - The axis along which to take slices. If None, array is flattened
///
/// # Returns
///
/// A new array with slices taken along the specified axis where condition is true
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Compress 1D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let condition = Array::from_vec(vec![true, false, true, false, true]);
/// let result = compress(&a, &condition, None).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 3, 5]);
///
/// // Compress 2D array along axis 0
/// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[3, 2]);
/// let cond = Array::from_vec(vec![true, false, true]);
/// let result = compress(&b, &cond, Some(0)).unwrap();
/// assert_eq!(result.shape(), vec![2, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 6]);
/// ```
pub fn compress<T: Clone + Zero>(
    array: &Array<T>,
    condition: &Array<bool>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    // Ensure condition is 1-D
    if condition.ndim() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "Condition must be a 1-D array".into(),
        ));
    }

    match axis {
        Some(ax) => {
            // Compress along specified axis
            if ax >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Check condition length matches axis size
            if condition.size() != axis_size {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Condition length {} doesn't match axis {} size {}",
                    condition.size(),
                    ax,
                    axis_size
                )));
            }

            // Get indices where condition is true
            let condition_vec = condition.to_vec();
            let selected_indices: Vec<usize> = condition_vec
                .iter()
                .enumerate()
                .filter_map(|(i, &val)| if val { Some(i) } else { None })
                .collect();

            if selected_indices.is_empty() {
                // Return empty array with appropriate shape
                let mut new_shape = shape.clone();
                new_shape[ax] = 0;
                return Ok(Array::from_vec(vec![]).reshape(&new_shape));
            }

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = selected_indices.len();

            // Calculate strides for indexing
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Collect selected slices
            let total_size: usize = new_shape.iter().product();
            let mut result_data = Vec::with_capacity(total_size);

            for i in 0..total_size {
                // Convert flat index to multi-dimensional indices
                let mut indices_arr = vec![0; shape.len()];
                let mut temp = i;
                for j in 0..new_shape.len() {
                    indices_arr[j] = temp / strides[j];
                    temp %= strides[j];
                }

                // Map the index along the compressed axis
                if ax < indices_arr.len() && indices_arr[ax] < selected_indices.len() {
                    indices_arr[ax] = selected_indices[indices_arr[ax]];
                    result_data.push(array.get(&indices_arr)?);
                }
            }

            Ok(Array::from_vec(result_data).reshape(&new_shape))
        }
        None => {
            // Flatten array and compress
            extract(array, condition)
        }
    }
}

/// Pack elements of a binary-valued array into bits in a uint8 array
///
/// # Parameters
///
/// * `array` - Array of binary values (0 or 1) to be packed
/// * `axis` - The dimension along which packing is performed. If None, the array is flattened
/// * `bitorder` - Bit order ('big' or 'little'). 'big' means the most significant bit is at the beginning
///
/// # Returns
///
/// Packed array with type uint8. The dimension along the given axis is divided by 8.
/// If the number of elements is not divisible by 8, the last byte is padded with zeros.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Pack 1D binary array
/// let a = Array::from_vec(vec![1u8, 0, 1, 0, 0, 0, 1, 1]);
/// let packed = packbits(&a, None, Some("big")).unwrap();
/// assert_eq!(packed.to_vec(), vec![163u8]); // 10100011 in binary = 163
///
/// // Pack with padding
/// let b = Array::from_vec(vec![1u8, 1, 1]);
/// let packed = packbits(&b, None, Some("big")).unwrap();
/// assert_eq!(packed.to_vec(), vec![224u8]); // 11100000 in binary = 224
/// ```
pub fn packbits(
    array: &Array<u8>,
    axis: Option<isize>,
    bitorder: Option<&str>,
) -> Result<Array<u8>> {
    let bitorder_str = bitorder.unwrap_or("big");
    if bitorder_str != "big" && bitorder_str != "little" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "bitorder must be 'big' or 'little', got '{}'",
            bitorder_str
        )));
    }

    // Validate input contains only 0s and 1s
    let data = array.to_vec();
    for &val in &data {
        if val != 0 && val != 1 {
            return Err(NumRs2Error::InvalidOperation(
                "packbits requires binary input (0 or 1)".to_string(),
            ));
        }
    }

    match axis {
        Some(ax) => {
            // Pack along specified axis
            let ndim = array.ndim();
            let axis_idx = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis_idx >= ndim {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis_idx];
            let packed_axis_size = axis_size.div_ceil(8); // Ceiling division

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[axis_idx] = packed_axis_size;

            // Calculate total size
            let mut outer_size = 1;
            for i in 0..axis_idx {
                outer_size *= shape[i];
            }
            let mut inner_size = 1;
            for i in (axis_idx + 1)..ndim {
                inner_size *= shape[i];
            }

            let mut packed_data = Vec::with_capacity(outer_size * packed_axis_size * inner_size);

            // Pack bits along the specified axis
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    for packed_idx in 0..packed_axis_size {
                        let mut byte = 0u8;
                        let start_bit = packed_idx * 8;
                        let end_bit = ((packed_idx + 1) * 8).min(axis_size);

                        for bit_idx in start_bit..end_bit {
                            let flat_idx =
                                outer * axis_size * inner_size + bit_idx * inner_size + inner;
                            let bit = data[flat_idx];

                            if bitorder_str == "big" {
                                byte |= bit << (7 - (bit_idx - start_bit));
                            } else {
                                byte |= bit << (bit_idx - start_bit);
                            }
                        }

                        packed_data.push(byte);
                    }
                }
            }

            Ok(Array::from_vec(packed_data).reshape(&new_shape))
        }
        None => {
            // Pack flattened array
            let flat_data = array.to_vec();
            let n = flat_data.len();
            let packed_size = n.div_ceil(8);
            let mut packed = Vec::with_capacity(packed_size);

            for i in 0..packed_size {
                let mut byte = 0u8;
                let start = i * 8;
                let end = ((i + 1) * 8).min(n);

                for j in start..end {
                    let bit = flat_data[j];
                    if bitorder_str == "big" {
                        byte |= bit << (7 - (j - start));
                    } else {
                        byte |= bit << (j - start);
                    }
                }

                packed.push(byte);
            }

            Ok(Array::from_vec(packed))
        }
    }
}

/// Unpack elements of a uint8 array into a binary-valued array
///
/// # Parameters
///
/// * `packed` - Array of type uint8 to be unpacked
/// * `axis` - The dimension along which unpacking is performed. If None, the array is flattened
/// * `count` - The number of elements to unpack along the given axis. If None, unpacks 8 * packed.shape[axis]
/// * `bitorder` - Bit order ('big' or 'little'). 'big' means the most significant bit is at the beginning
///
/// # Returns
///
/// The unpacked array with binary values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Unpack uint8 array
/// let packed = Array::from_vec(vec![163u8]); // 10100011 in binary
/// let unpacked = unpackbits(&packed, None, None, Some("big")).unwrap();
/// assert_eq!(unpacked.to_vec(), vec![1, 0, 1, 0, 0, 0, 1, 1]);
///
/// // Unpack with specific count
/// let packed = Array::from_vec(vec![224u8]); // 11100000 in binary
/// let unpacked = unpackbits(&packed, None, Some(3), Some("big")).unwrap();
/// assert_eq!(unpacked.to_vec(), vec![1, 1, 1]);
/// ```
pub fn unpackbits(
    packed: &Array<u8>,
    axis: Option<isize>,
    count: Option<usize>,
    bitorder: Option<&str>,
) -> Result<Array<u8>> {
    let bitorder_str = bitorder.unwrap_or("big");
    if bitorder_str != "big" && bitorder_str != "little" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "bitorder must be 'big' or 'little', got '{}'",
            bitorder_str
        )));
    }

    match axis {
        Some(ax) => {
            // Unpack along specified axis
            let ndim = packed.ndim();
            let axis_idx = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis_idx >= ndim {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let shape = packed.shape();
            let packed_axis_size = shape[axis_idx];
            let unpacked_axis_size = count.unwrap_or(packed_axis_size * 8);

            // Validate count
            if unpacked_axis_size > packed_axis_size * 8 {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "count ({}) cannot be larger than {} (8 * packed_axis_size)",
                    unpacked_axis_size,
                    packed_axis_size * 8
                )));
            }

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[axis_idx] = unpacked_axis_size;

            // Calculate total size
            let mut outer_size = 1;
            for i in 0..axis_idx {
                outer_size *= shape[i];
            }
            let mut inner_size = 1;
            for i in (axis_idx + 1)..ndim {
                inner_size *= shape[i];
            }

            let packed_data = packed.to_vec();
            let mut unpacked_data =
                Vec::with_capacity(outer_size * unpacked_axis_size * inner_size);

            // Unpack bits along the specified axis
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    for bit_idx in 0..unpacked_axis_size {
                        let packed_idx = bit_idx / 8;
                        let bit_offset = bit_idx % 8;

                        let flat_idx =
                            outer * packed_axis_size * inner_size + packed_idx * inner_size + inner;
                        let byte = packed_data[flat_idx];

                        let bit = if bitorder_str == "big" {
                            (byte >> (7 - bit_offset)) & 1
                        } else {
                            (byte >> bit_offset) & 1
                        };

                        unpacked_data.push(bit);
                    }
                }
            }

            Ok(Array::from_vec(unpacked_data).reshape(&new_shape))
        }
        None => {
            // Unpack flattened array
            let packed_data = packed.to_vec();
            let n_bytes = packed_data.len();
            let n_bits = count.unwrap_or(n_bytes * 8);

            if n_bits > n_bytes * 8 {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "count ({}) cannot be larger than {} (8 * number of bytes)",
                    n_bits,
                    n_bytes * 8
                )));
            }

            let mut unpacked = Vec::with_capacity(n_bits);

            for i in 0..n_bits {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let byte = packed_data[byte_idx];

                let bit = if bitorder_str == "big" {
                    (byte >> (7 - bit_idx)) & 1
                } else {
                    (byte >> bit_idx) & 1
                };

                unpacked.push(bit);
            }

            Ok(Array::from_vec(unpacked))
        }
    }
}

/// Converts a flat index or array of flat indices into a tuple of coordinate arrays
///
/// # Parameters
///
/// * `indices` - An array of flat indices
/// * `shape` - The shape of the array into which the flat indices should be converted
/// * `order` - Order of the indices: 'C' for row-major (default) or 'F' for column-major
///
/// # Returns
///
/// Tuple of arrays, one for each dimension, containing the coordinates
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert single index
/// let indices = Array::from_vec(vec![6]);
/// let coords = unravel_index(&indices, &[3, 4], Some("C")).unwrap();
/// assert_eq!(coords[0].to_vec(), vec![1]); // row 1
/// assert_eq!(coords[1].to_vec(), vec![2]); // col 2
///
/// // Convert multiple indices
/// let indices = Array::from_vec(vec![6, 11, 3, 5]);
/// let coords = unravel_index(&indices, &[3, 4], Some("C")).unwrap();
/// assert_eq!(coords[0].to_vec(), vec![1, 2, 0, 1]); // rows
/// assert_eq!(coords[1].to_vec(), vec![2, 3, 3, 1]); // cols
/// ```
pub fn unravel_index(
    indices: &Array<usize>,
    shape: &[usize],
    order: Option<&str>,
) -> Result<Vec<Array<usize>>> {
    let order_str = order.unwrap_or("C");
    if order_str != "C" && order_str != "F" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "order must be 'C' or 'F', got '{}'",
            order_str
        )));
    }

    if shape.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "shape cannot be empty".to_string(),
        ));
    }

    // Calculate total size of the array
    let total_size: usize = shape.iter().product();

    // Validate indices
    let indices_data = indices.to_vec();
    for &idx in &indices_data {
        if idx >= total_size {
            return Err(NumRs2Error::InvalidOperation(format!(
                "index {} is out of bounds for array with size {}",
                idx, total_size
            )));
        }
    }

    let n_dims = shape.len();
    let n_indices = indices_data.len();

    // Initialize coordinate arrays
    let mut coordinates: Vec<Vec<usize>> = vec![Vec::with_capacity(n_indices); n_dims];

    // Calculate strides based on order
    let mut strides = vec![1; n_dims];
    if order_str == "C" {
        // Row-major order (C-style)
        for i in (0..n_dims - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    } else {
        // Column-major order (Fortran-style)
        for i in 1..n_dims {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
    }

    // Convert each flat index to coordinates
    for &flat_idx in &indices_data {
        let mut remainder = flat_idx;

        if order_str == "C" {
            // Row-major unraveling
            for i in 0..n_dims {
                coordinates[i].push(remainder / strides[i]);
                remainder %= strides[i];
            }
        } else {
            // Column-major unraveling
            for i in 0..n_dims {
                coordinates[i].push(remainder % shape[i]);
                remainder /= shape[i];
            }
        }
    }

    // Convert coordinate vectors to Arrays
    let mut result = Vec::with_capacity(n_dims);
    for coord_vec in coordinates {
        result.push(Array::from_vec(coord_vec).reshape(&indices.shape()));
    }

    Ok(result)
}

/// Converts a tuple of coordinate arrays into an array of flat indices
///
/// # Parameters
///
/// * `multi_index` - Tuple of arrays, one array for each dimension
/// * `dims` - The shape of the array into which the indices will be converted
/// * `mode` - Specifies how out-of-bounds indices are handled:
///   - 'raise' (default): raise error
///   - 'wrap': wrap around
///   - 'clip': clip to the valid range
/// * `order` - Order of the indices: 'C' for row-major (default) or 'F' for column-major
///
/// # Returns
///
/// Array of flat indices
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert single coordinate
/// let row = Array::from_vec(vec![1]);
/// let col = Array::from_vec(vec![2]);
/// let flat = ravel_multi_index(&[&row, &col], &[3, 4], Some("raise"), Some("C")).unwrap();
/// assert_eq!(flat.to_vec(), vec![6]); // 1*4 + 2 = 6
///
/// // Convert multiple coordinates
/// let rows = Array::from_vec(vec![1, 2, 0, 1]);
/// let cols = Array::from_vec(vec![2, 3, 3, 1]);
/// let flat = ravel_multi_index(&[&rows, &cols], &[3, 4], Some("raise"), Some("C")).unwrap();
/// assert_eq!(flat.to_vec(), vec![6, 11, 3, 5]);
/// ```
pub fn ravel_multi_index(
    multi_index: &[&Array<usize>],
    dims: &[usize],
    mode: Option<&str>,
    order: Option<&str>,
) -> Result<Array<usize>> {
    let mode_str = mode.unwrap_or("raise");
    if mode_str != "raise" && mode_str != "wrap" && mode_str != "clip" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "mode must be 'raise', 'wrap', or 'clip', got '{}'",
            mode_str
        )));
    }

    let order_str = order.unwrap_or("C");
    if order_str != "C" && order_str != "F" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "order must be 'C' or 'F', got '{}'",
            order_str
        )));
    }

    if multi_index.len() != dims.len() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "number of index arrays ({}) must match number of dimensions ({})",
            multi_index.len(),
            dims.len()
        )));
    }

    if multi_index.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "multi_index cannot be empty".to_string(),
        ));
    }

    // Check that all index arrays have the same shape
    let result_shape = multi_index[0].shape();
    for idx_array in multi_index.iter().skip(1) {
        if idx_array.shape() != result_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: result_shape.to_vec(),
                actual: idx_array.shape().to_vec(),
            });
        }
    }

    let n_indices = multi_index[0].size();
    let n_dims = dims.len();

    // Calculate strides based on order
    let mut strides = vec![1; n_dims];
    if order_str == "C" {
        // Row-major order (C-style)
        for i in (0..n_dims - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    } else {
        // Column-major order (Fortran-style)
        for i in 1..n_dims {
            strides[i] = strides[i - 1] * dims[i - 1];
        }
    }

    // Convert coordinates to flat indices
    let mut flat_indices = Vec::with_capacity(n_indices);

    // Get data from all coordinate arrays
    let coord_data: Vec<Vec<usize>> = multi_index.iter().map(|arr| arr.to_vec()).collect();

    for idx in 0..n_indices {
        let mut flat_idx = 0;

        for dim in 0..n_dims {
            let coord = coord_data[dim][idx];

            // Handle out-of-bounds coordinates based on mode
            let adjusted_coord = match mode_str {
                "raise" => {
                    if coord >= dims[dim] {
                        return Err(NumRs2Error::InvalidOperation(format!(
                            "index {} is out of bounds for axis {} with size {}",
                            coord, dim, dims[dim]
                        )));
                    }
                    coord
                }
                "wrap" => coord % dims[dim],
                "clip" => coord.min(dims[dim].saturating_sub(1)),
                _ => unreachable!(),
            };

            flat_idx += adjusted_coord * strides[dim];
        }

        flat_indices.push(flat_idx);
    }

    Ok(Array::from_vec(flat_indices).reshape(&result_shape))
}

/// Lower triangle of an array.
///
/// Return a copy of an array with elements above the k-th diagonal zeroed.
///
/// # Arguments
///
/// * `array` - Input array
/// * `k` - Diagonal above which to zero elements. k = 0 (the default) is the main diagonal,
///   k < 0 is below it and k > 0 is above.
///
/// # Returns
///
/// Lower triangle of array, of same shape and data-type as input array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let lower = tril(&a, 0).unwrap();
/// assert_eq!(lower.to_vec(), vec![1, 0, 0, 4, 5, 0, 7, 8, 9]);
///
/// // With k=1 (include first diagonal above main)
/// let lower = tril(&a, 1).unwrap();
/// assert_eq!(lower.to_vec(), vec![1, 2, 0, 4, 5, 6, 7, 8, 9]);
///
/// // With k=-1 (exclude main diagonal)
/// let lower = tril(&a, -1).unwrap();
/// assert_eq!(lower.to_vec(), vec![0, 0, 0, 4, 0, 0, 7, 8, 0]);
/// ```
pub fn tril<T: Clone + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 2 {
        // For 1D arrays, return a copy
        return Ok(array.clone());
    }

    let n_rows = shape[ndim - 2];
    let n_cols = shape[ndim - 1];

    // Create a copy of the array
    let mut result = array.clone();
    let result_data = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    // Calculate the number of 2D matrices in the array
    let n_matrices: usize = shape[..ndim - 2].iter().product();
    let matrix_size = n_rows * n_cols;

    // Zero out elements above the k-th diagonal for each matrix
    for m in 0..n_matrices {
        let matrix_offset = m * matrix_size;

        for i in 0..n_rows {
            for j in 0..n_cols {
                // Check if element is above the k-th diagonal
                if (j as isize) > (i as isize + k) {
                    let idx = matrix_offset + i * n_cols + j;
                    result_data[idx] = T::zero();
                }
            }
        }
    }

    Ok(result)
}

/// Upper triangle of an array.
///
/// Return a copy of an array with elements below the k-th diagonal zeroed.
///
/// # Arguments
///
/// * `array` - Input array
/// * `k` - Diagonal below which to zero elements. k = 0 (the default) is the main diagonal,
///   k < 0 is below it and k > 0 is above.
///
/// # Returns
///
/// Upper triangle of array, of same shape and data-type as input array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
/// let upper = triu(&a, 0).unwrap();
/// assert_eq!(upper.to_vec(), vec![1, 2, 3, 0, 5, 6, 0, 0, 9]);
///
/// // With k=1 (exclude main diagonal)
/// let upper = triu(&a, 1).unwrap();
/// assert_eq!(upper.to_vec(), vec![0, 2, 3, 0, 0, 6, 0, 0, 0]);
///
/// // With k=-1 (include first diagonal below main)
/// let upper = triu(&a, -1).unwrap();
/// assert_eq!(upper.to_vec(), vec![1, 2, 3, 4, 5, 6, 0, 8, 9]);
/// ```
pub fn triu<T: Clone + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if ndim < 2 {
        // For 1D arrays, return a copy
        return Ok(array.clone());
    }

    let n_rows = shape[ndim - 2];
    let n_cols = shape[ndim - 1];

    // Create a copy of the array
    let mut result = array.clone();
    let result_data = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    // Calculate the number of 2D matrices in the array
    let n_matrices: usize = shape[..ndim - 2].iter().product();
    let matrix_size = n_rows * n_cols;

    // Zero out elements below the k-th diagonal for each matrix
    for m in 0..n_matrices {
        let matrix_offset = m * matrix_size;

        for i in 0..n_rows {
            for j in 0..n_cols {
                // Check if element is below the k-th diagonal
                if (j as isize) < (i as isize + k) {
                    let idx = matrix_offset + i * n_cols + j;
                    result_data[idx] = T::zero();
                }
            }
        }
    }

    Ok(result)
}

/// Append values to the end of an array
///
/// # Arguments
/// * `array` - Input array
/// * `values` - Values to append to the array
/// * `axis` - The axis along which values are appended. If None, both arrays are flattened before appending.
///
/// # Returns
/// * `Result<Array<T>>` - A copy of array with values appended to axis
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::manipulation::append;
///
/// // Append to 1D array
/// let arr = Array::from_vec(vec![1, 2, 3]);
/// let result = append(&arr, &[4, 5], None).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
///
/// // Append to 2D array along axis 0
/// let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let values = Array::from_vec(vec![5, 6]).reshape(&[1, 2]);
/// let result = append(&arr, &values.to_vec(), Some(0)).unwrap();
/// assert_eq!(result.shape(), vec![3, 2]);
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // Append to 2D array along axis 1
/// let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let values = vec![5, 6];
/// let result = append(&arr, &values, Some(1)).unwrap();
/// assert_eq!(result.shape(), vec![2, 3]);
/// assert_eq!(result.to_vec(), vec![1, 2, 5, 3, 4, 6]);
/// ```
pub fn append<T: Clone + Zero>(
    array: &Array<T>,
    values: &[T],
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Flatten both arrays and concatenate
            let mut result_data = array.to_vec();
            result_data.extend_from_slice(values);
            Ok(Array::from_vec(result_data))
        }
        Some(ax) => {
            let shape = array.shape();

            if ax >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            // For appending along an axis, values must be reshaped to match
            // the shape of array except along the concatenation axis
            let axis_size = shape[ax];
            let mut values_shape = shape.clone();

            // Calculate the expected size for values
            let expected_values_size: usize = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .product();

            // Check if values can be reshaped appropriately
            if values.len() % expected_values_size != 0 {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![expected_values_size],
                    actual: vec![values.len()],
                });
            }

            // Calculate new size along the axis
            let values_axis_size = values.len() / expected_values_size;
            values_shape[ax] = values_axis_size;

            // Create values array with proper shape
            let values_array = Array::from_vec(values.to_vec()).reshape(&values_shape);

            // Update result shape
            let mut result_shape = shape.clone();
            result_shape[ax] = axis_size + values_axis_size;

            // Calculate sizes for iteration
            let pre_axis_size: usize = shape[..ax].iter().product();
            let post_axis_size: usize = shape[ax + 1..].iter().product();
            let total_size = pre_axis_size * result_shape[ax] * post_axis_size;

            // Allocate result array
            let mut result_data = Vec::with_capacity(total_size);

            // Copy data
            let array_data = array.to_vec();
            let values_data = values_array.to_vec();

            for pre in 0..pre_axis_size {
                // Copy from original array
                for i in 0..axis_size {
                    for post in 0..post_axis_size {
                        let idx = pre * axis_size * post_axis_size + i * post_axis_size + post;
                        result_data.push(array_data[idx].clone());
                    }
                }

                // Copy from values array
                for i in 0..values_axis_size {
                    for post in 0..post_axis_size {
                        let idx =
                            pre * values_axis_size * post_axis_size + i * post_axis_size + post;
                        result_data.push(values_data[idx].clone());
                    }
                }
            }

            Ok(Array::from_vec(result_data).reshape(&result_shape))
        }
    }
}
