use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{IxDyn, SliceInfo, SliceInfoElem};
use std::fmt::Debug;

/// Advanced stride manipulation utilities for NumRS2 arrays.
///
/// This module provides advanced functions for manipulating array strides,
/// enabling sophisticated and memory-efficient array operations similar to
/// NumPy's `numpy.lib.stride_tricks` module.
/// Create a view of the given array with the specified strides without copying.
///
/// This is a lower-level function than `as_strided` as it directly manipulates
/// the strides of the array. The returned array is a view of the original
/// array with modified strides.
///
/// # Arguments
///
/// * `array` - The input array
/// * `strides` - The new strides to use
///
/// # Returns
///
/// * `Ok(Array<T>)` - A view of the input array with the specified strides
/// * `Err(NumRs2Error)` - Error if strides are invalid or dimension mismatch
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
///
/// // Create a view with stride 2 in both dimensions (every other element)
/// let strided = set_strides(&array, &[2, 2]).expect("set_strides should succeed");
/// assert_eq!(strided.shape(), vec![2, 2]);
/// ```
///
/// # Safety
///
/// This function can be unsafe as it allows creating views that might go beyond
/// the bounds of the original array if used incorrectly. The function attempts
/// to validate the strides, but it's the caller's responsibility to ensure they
/// are valid for the given array.
pub fn set_strides<T>(array: &Array<T>, strides: &[isize]) -> Result<Array<T>>
where
    T: Clone + Debug,
{
    if strides.len() != array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Expected {} strides, got {}",
            array.ndim(),
            strides.len()
        )));
    }

    let view = array.array().view();
    let shape = array.shape();

    // Create stride information for each dimension
    let mut slice_info = Vec::with_capacity(array.ndim());

    for (i, &stride) in strides.iter().enumerate() {
        let dim_size = shape[i];

        if stride == 0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Stride for dimension {} cannot be zero",
                i
            )));
        }

        // If stride is positive, create a slice from 0 to dim_size with step stride
        let start = if stride > 0 { 0 } else { dim_size as isize - 1 };
        let end = if stride > 0 { dim_size as isize } else { -1 };

        slice_info.push(SliceInfoElem::Slice {
            start,
            end: Some(end),
            step: stride,
        });
    }

    // Create the slice information
    let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)
        .map_err(|_| NumRs2Error::InvalidOperation("Failed to create slice info".to_string()))?;

    // Slice the array and return the view
    let strided = view.slice(slice_info);
    let result = Array::from_ndarray(strided.to_owned());
    Ok(result)
}

/// Create a new view into the array with the given shape and strides.
///
/// This function is similar to NumPy's `numpy.lib.stride_tricks.as_strided`.
/// It creates a view with a specific shape and strides without copying the data.
///
/// # Arguments
///
/// * `array` - The input array
/// * `shape` - The shape of the new view
/// * `strides` - The strides for the new view, **in elements** (not bytes).
///   The value at output multi-index `idx` is
///   `flat(array)[sum(idx[d] * strides[d] for d in 0..shape.len())]`, where
///   `flat(array)` is `array`'s data read in row-major (C) order. A stride
///   of `0` along an axis is allowed and repeats the same element for every
///   index on that axis -- this is the mechanism `broadcast_to` is built on.
///   Negative strides are allowed as long as every reachable offset stays
///   within `[0, array.size())`; since index `0` always contributes offset
///   `0` on every axis, a negative stride only stays in-bounds when its own
///   axis has size `1` (compare `set_strides`, which rejects stride `0`
///   outright because it slices rather than gathers).
///
/// # Returns
///
/// * `Ok(Array<T>)` - A view of the input array with the specified shape and strides
/// * `Err(NumRs2Error)` - Error if `shape` and `strides` have different lengths,
///   or if any offset reachable by `shape`/`strides` would fall outside the
///   input array's data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
///
/// // Sample every other row and column. [6, 2] is 2x the array's natural
/// // (element) strides [3, 1], so this yields the four corners of the grid.
/// let strided = as_strided(&array, &[2, 2], &[6, 2]).expect("as_strided should succeed");
/// assert_eq!(strided.shape(), vec![2, 2]);
/// assert_eq!(strided.to_vec(), vec![1, 3, 7, 9]);
/// ```
///
/// # Safety
///
/// This function reads `array`'s data via `Array::to_vec()`, which flattens
/// it in row-major (C) order; `strides` are interpreted against that
/// flattened buffer, not against `array`'s own internal memory layout. Every
/// offset reachable by `shape`/`strides` is bounds-checked before any data
/// is read, so out-of-range parameters return `Err` rather than panicking or
/// silently producing garbage -- but nothing stops `shape`/`strides` from
/// describing a view whose elements overlap or repeat, which is inherent to
/// `as_strided` and is the caller's responsibility to use correctly.
pub fn as_strided<T>(array: &Array<T>, shape: &[usize], strides: &[isize]) -> Result<Array<T>>
where
    T: Clone + Debug,
{
    if shape.len() != strides.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape and strides must have the same length, got {} and {}",
            shape.len(),
            strides.len()
        )));
    }

    // Read the array's data in row-major (C) order; `strides` are
    // interpreted against this flattened buffer (see the Safety section
    // above), independently of `array`'s own ndim or internal layout.
    let flat_data = array.to_vec();
    let n = flat_data.len();

    // A shape with any zero-sized dimension has zero total elements: there
    // is nothing to gather, and no meaningful bounds to check.
    let total_size: usize = shape.iter().product();
    if total_size == 0 {
        return Ok(Array::from_vec(Vec::new()).reshape(shape));
    }

    // offset(idx) = sum(idx[d] * strides[d]) is a sum of independent
    // per-axis terms: idx[d] ranges freely over 0..shape[d] regardless of
    // the other axes. So the true minimum/maximum offset reachable across
    // the *entire* output index space is exactly the sum of each axis's own
    // minimum/maximum contribution -- and both extremes are always actually
    // visited, since the index space is a full grid product.
    let mut min_offset: isize = 0;
    let mut max_offset: isize = 0;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let extent = (dim as isize - 1) * stride;
        if extent >= 0 {
            max_offset += extent;
        } else {
            min_offset += extent;
        }
    }

    if min_offset < 0 || max_offset >= n as isize {
        return Err(NumRs2Error::InvalidOperation(format!(
            "as_strided: shape {:?} with strides {:?} would access offsets in [{}, {}], \
             out of bounds for an array of {} elements",
            shape, strides, min_offset, max_offset, n
        )));
    }

    // General N-D strided gather: for each output multi-index (recovered by
    // unraveling the row-major linear index), read the element at the
    // corresponding flat offset.
    let mut result_data = Vec::with_capacity(total_size);
    for linear in 0..total_size {
        let idx = unravel_index(linear, shape);
        let offset: isize = idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();
        // In-bounds by construction: for every possible `idx`, `offset` is
        // bracketed by [min_offset, max_offset] ⊆ [0, n), as validated above.
        result_data.push(flat_data[offset as usize].clone());
    }

    Ok(Array::from_vec(result_data).reshape(shape))
}

/// Convert a linear (row-major) index into a multi-dimensional index for
/// `shape` -- the inverse of C-order flattening. The last axis varies
/// fastest, matching `Array::to_vec()` / `Array::reshape()`'s row-major
/// convention.
fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        if dim > 0 {
            idx[d] = linear % dim;
            linear /= dim;
        }
    }
    idx
}

/// Compute the row-major (C-order) element strides for `shape`: the strides
/// a freshly-allocated, densely-packed array of this shape would have (the
/// last axis has stride `1`, and each preceding axis's stride is the
/// product of all the dimension sizes to its right).
fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as isize;
    }
    strides
}

/// Create a sliding window view of an array.
///
/// This function creates a sliding window view of the input array with the given
/// window shape. The sliding window moves along each dimension of the input array.
///
/// # Arguments
///
/// * `array` - The input array
/// * `window_shape` - The shape of the sliding window
/// * `step` - The step size for each dimension (default is 1)
///
/// # Returns
///
/// * `Ok(Array<T>)` - A view with shape (n1, n2, ..., k1, k2, ...) where (n1, n2, ...)
///   is the number of valid positions of the sliding window, and (k1, k2, ...) is the
///   window shape.
/// * `Err(NumRs2Error)` - Error if parameters are invalid
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
///
/// // Create a 2x2 sliding window view of the array
/// let windows = sliding_window_view(&array, &[2, 2], None).expect("sliding_window_view should succeed");
/// assert_eq!(windows.shape(), vec![2, 2, 2, 2]);
/// assert_eq!(
///     windows.to_vec(),
///     vec![1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9]
/// );
/// ```
pub fn sliding_window_view<T>(
    array: &Array<T>,
    window_shape: &[usize],
    step: Option<&[usize]>,
) -> Result<Array<T>>
where
    T: Clone + Debug,
{
    let step_values = match step {
        Some(s) => {
            if s.len() != array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Step must have the same length as array dimensions, got {} and {}",
                    s.len(),
                    array.ndim()
                )));
            }
            if s.iter().any(|&v| v == 0) {
                return Err(NumRs2Error::InvalidOperation(
                    "Step sizes must be greater than zero".to_string(),
                ));
            }
            s.to_vec()
        }
        None => vec![1; array.ndim()],
    };

    if window_shape.len() != array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Window shape must have the same length as array dimensions, got {} and {}",
            window_shape.len(),
            array.ndim()
        )));
    }

    // Calculate the number of valid window positions along each dimension.
    let array_shape = array.shape();
    let ndim = array.ndim();
    let mut n_windows = Vec::with_capacity(ndim);

    for i in 0..ndim {
        let window_size = window_shape[i];
        let step_size = step_values[i];
        let dim_size = array_shape[i];

        if window_size > dim_size {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Window size {} exceeds array dimension {} of size {}",
                window_size, i, dim_size
            )));
        }

        n_windows.push((dim_size - window_size) / step_size + 1);
    }

    // Output shape is (n_windows..., window_shape...): one axis per input
    // dimension counting window positions, followed by one axis per input
    // dimension spanning the window itself.
    let mut output_shape = n_windows;
    output_shape.extend_from_slice(window_shape);

    // Reuse as_strided (general for any ndim): the "window count" axes step
    // through the array at `step * natural_stride`, and the appended
    // "window shape" axes step one element at a time along the same
    // original axis (natural_stride) -- exactly how NumPy implements
    // sliding_window_view via as_strided.
    let natural_stride = row_major_strides(&array_shape);
    let mut combined_strides = Vec::with_capacity(ndim * 2);
    for i in 0..ndim {
        combined_strides.push(natural_stride[i] * step_values[i] as isize);
    }
    combined_strides.extend_from_slice(&natural_stride);

    as_strided(array, &output_shape, &combined_strides)
}

/// Returns the byte strides of an array.
///
/// Byte strides represent the number of bytes to move along each dimension
/// when navigating the array in memory.
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// A vector containing the byte strides for each dimension of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let strides = byte_strides(&array);
/// ```
pub fn byte_strides<T>(array: &Array<T>) -> Vec<usize>
where
    T: Clone + Debug,
{
    // Get the memory strides in terms of elements
    let elem_strides = array.array().strides();

    // Convert to byte strides by multiplying by the size of T
    let elem_size = std::mem::size_of::<T>();
    elem_strides
        .iter()
        .map(|&s| s as usize * elem_size)
        .collect()
}

/// Create views into arrays in a way that broadcasting might occur.
///
/// This function is similar to NumPy's `broadcast_arrays`, but uses
/// stride manipulation to create the views.
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to broadcast together
///
/// # Returns
///
/// * `Ok(Vec<Array<T>>)` - A vector of arrays that are broadcast to have the same shape
/// * `Err(NumRs2Error)` - Error if arrays cannot be broadcast together
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]).reshape(&[3, 1]);
///
/// let result = broadcast_arrays(&[&a, &b]).expect("broadcast_arrays should succeed");
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].shape(), result[1].shape());
/// assert_eq!(result[0].to_vec(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
/// assert_eq!(result[1].to_vec(), vec![4, 4, 4, 5, 5, 5, 6, 6, 6]);
/// ```
pub fn broadcast_arrays<T>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>>
where
    T: Clone + Debug,
{
    if arrays.is_empty() {
        return Ok(Vec::new());
    }

    // Get the shapes of all arrays
    let shapes: Vec<_> = arrays.iter().map(|a| a.shape()).collect();

    // Determine the output shape (the shape all arrays will be broadcast to)
    let output_shape = broadcast_shape(&shapes)?;

    // Broadcast each array to the output shape
    let mut result = Vec::with_capacity(arrays.len());
    for array in arrays {
        let broadcast = broadcast_to(array, &output_shape)?;
        result.push(broadcast);
    }

    Ok(result)
}

/// Broadcast an array to a new shape using stride tricks.
///
/// This function is similar to NumPy's `broadcast_to`, but uses
/// stride manipulation to create the view.
///
/// # Arguments
///
/// * `array` - The input array to broadcast
/// * `shape` - The target shape to broadcast to
///
/// # Returns
///
/// * `Ok(Array<T>)` - The broadcast array
/// * `Err(NumRs2Error)` - Error if the array cannot be broadcast to the target shape
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let array = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
///
/// // Broadcast to shape [3, 3]
/// let result = broadcast_to(&array, &[3, 3]).expect("broadcast_to should succeed");
/// assert_eq!(result.shape(), vec![3, 3]);
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
/// ```
pub fn broadcast_to<T>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + Debug,
{
    // Check if the array can be broadcast to the target shape
    if !is_broadcastable(&array.shape(), shape) {
        return Err(NumRs2Error::ShapeMismatch {
            expected: shape.to_vec(),
            actual: array.shape(),
        });
    }

    // Get the original shape and its natural (element) strides. `as_strided`
    // interprets strides as element offsets into the row-major flattened
    // buffer (see its docs), so these must be element strides -- not the
    // byte strides `byte_strides()` returns.
    let orig_shape = array.shape();
    let elem_strides = row_major_strides(&orig_shape);

    // Calculate the new strides for the broadcast array
    let mut new_strides = Vec::with_capacity(shape.len());

    // Prepend dimensions to match the length of the target shape
    let prepend_dims = shape.len() - orig_shape.len();
    new_strides.extend(std::iter::repeat_n(0, prepend_dims)); // Stride 0 for broadcast dimensions

    // Set strides for existing dimensions
    for (i, &dim) in orig_shape.iter().enumerate() {
        let target_dim = shape[i + prepend_dims];
        if dim == 1 && target_dim > 1 {
            // Broadcasting from a dimension of size 1 to a larger size
            new_strides.push(0);
        } else {
            // Keep original stride for non-broadcast dimensions
            new_strides.push(elem_strides[i]);
        }
    }

    // Use as_strided to create the broadcast view
    as_strided(array, shape, &new_strides)
}

/// Check if an array shape can be broadcast to a target shape.
///
/// Broadcasting rules:
/// 1. If the two arrays have different numbers of dimensions, prepend the shape
///    of the one with fewer dimensions with 1s until both shapes have the same length.
/// 2. The size in each dimension of the output shape is the maximum of the sizes
///    of the two input arrays in that dimension.
/// 3. An array can be broadcast along a dimension if its size in that dimension is 1
///    or if it doesn't have that dimension.
///
/// # Arguments
///
/// * `source_shape` - The shape of the source array
/// * `target_shape` - The shape to broadcast to
///
/// # Returns
///
/// True if the source shape can be broadcast to the target shape, false otherwise
fn is_broadcastable(source_shape: &[usize], target_shape: &[usize]) -> bool {
    // A scalar can be broadcast to any shape
    if source_shape.is_empty() {
        return true;
    }

    // If the source has more dimensions than target, it cannot be broadcast
    if source_shape.len() > target_shape.len() {
        return false;
    }

    // Check each dimension from the end (right-aligned)
    let offset = target_shape.len() - source_shape.len();
    for (i, &dim) in source_shape.iter().enumerate() {
        let target_dim = target_shape[i + offset];
        if dim != 1 && dim != target_dim {
            return false;
        }
    }

    true
}

/// Determine the output shape when broadcasting arrays together.
///
/// # Arguments
///
/// * `shapes` - A slice of array shapes to broadcast together
///
/// # Returns
///
/// * `Ok(Vec<usize>)` - The broadcast shape
/// * `Err(NumRs2Error)` - Error if shapes cannot be broadcast together
fn broadcast_shape(shapes: &[Vec<usize>]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }

    // Find the maximum number of dimensions
    // Safe: shapes is non-empty (checked above), so max() returns Some
    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);

    // Initialize the output shape with 1s
    let mut output_shape = vec![1; max_ndim];

    // Determine the output shape
    for shape in shapes {
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let out_i = i + offset;
            if output_shape[out_i] == 1 {
                output_shape[out_i] = dim;
            } else if dim != 1 && dim != output_shape[out_i] {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Incompatible shapes for broadcasting: dimension {} has conflicting sizes {} and {}",
                            out_i, output_shape[out_i], dim)
                ));
            }
        }
    }

    Ok(output_shape)
}
