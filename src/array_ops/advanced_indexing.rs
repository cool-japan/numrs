//! Advanced indexing operations for Arrays
//!
//! This module provides advanced indexing functionality similar to NumPy's
//! advanced indexing capabilities, including compress, extract, place, put,
//! and other sophisticated indexing operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

/// Return selected slices of an array along given axis
///
/// This is the canonical `compress` implementation; [`crate::array_ops::manipulation::compress`]
/// (a one-line delegate) and [`crate::Array::compress`] (which adapts a
/// generic, stringify-based condition into `Array<bool>` before delegating)
/// both forward here.
///
/// # Arguments
/// * `array` - Input array
/// * `condition` - 1-D array of booleans corresponding to indices to select. Following NumPy,
///   `condition` need not have the same length as `array` (when `axis` is `None`) or as
///   `array.shape()[axis]` (when `axis` is given): it may be shorter (the tail is treated as
///   absent, not as `false`) and it may be longer as long as no `true` entry falls beyond the
///   selected length -- only an out-of-bounds `true` is an error.
/// * `axis` - Axis along which to take slices. If None, work on flattened array
///
/// # Returns
/// * `Result<Array<T>>` - Array with selected slices
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::compress;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let condition = Array::from_vec(vec![true, false, true, false, true]);
/// let compressed = compress(&arr, &condition, None).expect("operation should succeed");
/// // Returns [1, 3, 5]
///
/// let arr2d = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let cond = Array::from_vec(vec![true, false, true]);
/// let compressed = compress(&arr2d, &cond, Some(1)).expect("operation should succeed");
/// // Returns [[1, 3], [4, 6]] (columns 0 and 2)
///
/// // `condition` shorter than the array is allowed (NumPy: `np.compress([True], [1,2,3])`)
/// let short_cond = Array::from_vec(vec![true]);
/// let compressed = compress(&arr, &short_cond, None).expect("operation should succeed");
/// assert_eq!(compressed.to_vec(), vec![1]);
/// ```
pub fn compress<T: Clone + Zero>(
    array: &Array<T>,
    condition: &Array<bool>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    // NumPy rejects a non-1-D condition outright (`"condition must be a 1-d
    // array"`) regardless of `axis`; neither the flattened nor the
    // axis-given path below assumes anything about condition shape beyond
    // this, so check it once, up front.
    if condition.ndim() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "condition must be a 1-D array".to_string(),
        ));
    }

    // Positions selected by `condition`, in ascending order (so the
    // out-of-bounds check below is a single last-element comparison and the
    // per-position lookup in the axis-given branch can binary-search
    // instead of doing an O(condition-length) scan per output element).
    let selected: Vec<usize> = condition
        .array()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, cond)| if cond { Some(i) } else { None })
        .collect();

    match axis {
        None => {
            // Work on flattened array. `condition` need not match
            // `array`'s length: `zip` truncates to the shorter of the two,
            // which is exactly right when `condition` is shorter (NumPy
            // treats the unreached tail as simply never selected, not as
            // `false`) -- the only remaining way `condition` can be
            // invalid is a `true` beyond `array`'s length, which `zip`
            // would silently drop instead of erroring on, so that's
            // checked explicitly first via `selected`'s largest entry.
            let total = array.size();
            if let Some(&bad) = selected.last().filter(|&&i| i >= total) {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    bad, total
                )));
            }

            // `.array().iter()` walks the operand's logical order directly
            // (zero-copy when contiguous); `zip` with the (already
            // bounds-checked) condition needs no full-size intermediate
            // `Vec` for either side.
            let compressed: Vec<T> = array
                .array()
                .iter()
                .cloned()
                .zip(condition.array().iter().copied())
                .filter_map(|(val, cond)| if cond { Some(val) } else { None })
                .collect();

            Ok(Array::from_vec(compressed))
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

            let axis_size = shape[ax];
            if let Some(&bad) = selected.last().filter(|&&i| i >= axis_size) {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "index {} is out of bounds for axis {} with size {}",
                    bad, ax, axis_size
                )));
            }

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[ax] = selected.len();

            if selected.is_empty() {
                // Return an empty array with the right shape: `new_shape`
                // has a 0 dimension along `ax`, so `vec![]` is already the
                // right (empty) element count for it.
                return Array::from_vec_shape(vec![], &new_shape);
            }

            // Extract selected slices
            let mut result_data = Vec::with_capacity(new_shape.iter().product());

            // Helper to iterate through all indices
            let mut current_indices = vec![0; shape.len()];
            let total_elements: usize = shape.iter().product();

            for _ in 0..total_elements {
                // Check if current index along axis is in our selection
                if selected.binary_search(&current_indices[ax]).is_ok() {
                    let value = array.get(&current_indices)?;
                    result_data.push(value);
                }

                // Increment indices
                let mut carry = true;
                for dim in (0..shape.len()).rev() {
                    if carry {
                        current_indices[dim] += 1;
                        carry = current_indices[dim] >= shape[dim];
                        if carry {
                            current_indices[dim] = 0;
                        }
                    }
                }
            }

            Ok(Array::from_vec_shape(result_data, &new_shape)?)
        }
    }
}

/// Return the elements of an array that satisfy some condition
///
/// This is equivalent to `array[condition]` in NumPy where condition is a boolean array.
///
/// # Arguments
/// * `array` - Input array
/// * `condition` - Boolean array with same shape as `array`
///
/// # Returns
/// * `Result<Array<T>>` - 1-D array with elements where condition is True
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::extract;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let condition = Array::from_vec(vec![true, false, true, false, true, false]).reshape(&[2, 3]);
/// let extracted = extract(&arr, &condition).expect("operation should succeed");
/// // Returns [1, 3, 5]
/// ```
pub fn extract<T: Clone>(array: &Array<T>, condition: &Array<bool>) -> Result<Array<T>> {
    if array.shape() != condition.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: condition.shape(),
        });
    }

    let extracted: Vec<T> = array
        .array()
        .iter()
        .cloned()
        .zip(condition.array().iter().copied())
        .filter_map(|(val, cond)| if cond { Some(val) } else { None })
        .collect();

    Ok(Array::from_vec(extracted))
}

/// Place values into array at specified indices
///
/// # Arguments
/// * `array` - Array to modify (modified in-place)
/// * `mask` - Boolean array indicating where to place values
/// * `values` - Values to place (will be repeated if necessary)
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::place;
///
/// let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let mask = Array::from_vec(vec![false, true, false, true, false]);
/// place(&mut arr, &mask, &[10, 20]).expect("operation should succeed");
/// // arr is now [1, 10, 3, 20, 5]
/// ```
pub fn place<T: Clone>(array: &mut Array<T>, mask: &Array<bool>, values: &[T]) -> Result<()> {
    if array.shape() != mask.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: mask.shape(),
        });
    }

    if values.is_empty() {
        return Err(NumRs2Error::ValueError(
            "values array cannot be empty".to_string(),
        ));
    }

    // Deliberately still `mask.to_vec()`, not `operand(mask)`: this site
    // was converted to a hoisted `operand` borrow earlier in this sweep
    // (removing the copy `mask.array().iter()` re-walked in full for both
    // the count below and the write loop), but measured *slower* than
    // this `to_vec()` original in `probe_place_perf_vs_naive_to_vec`
    // below -- 0.80x at full release, 0.69x at `[profile.test]`'s
    // `opt-level = 2` -- once that probe's "naive" comparison was fixed
    // to be equally generic (`T: Clone`) rather than hand-specialized to
    // `f64`, which is the fair comparison against this actual (generic)
    // function. Root cause not fully pinned down (plausibly the mutable
    // write into `array_data` below losing some alias-analysis/
    // vectorization opportunity against a slice borrowed from a *live*
    // `mask`, vs. one LLVM can prove is a fresh, disjoint allocation), but
    // the measurement itself is reproducible across both optimization
    // levels and is what this revert is based on -- `put`, its structural
    // twin below, shows the same effect (0.56x / 1.18x) and was reverted
    // for the same reason. `take`/`outer`/`real_if_close` in this same
    // sweep are the opposite case: those *did* win with `operand`.
    let mask_data = mask.to_vec();
    let num_true = mask_data.iter().filter(|&&x| x).count();

    if num_true == 0 {
        return Ok(()); // Nothing to place
    }

    // Get mutable slice
    let array_data = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    let mut value_idx = 0;
    for (i, &is_true) in mask_data.iter().enumerate() {
        if is_true {
            array_data[i] = values[value_idx % values.len()].clone();
            value_idx += 1;
        }
    }

    Ok(())
}

/// Replaces specified elements of array with given values
///
/// # Arguments
/// * `array` - Array to modify (modified in-place)
/// * `indices` - 1-D array of indices
/// * `values` - Values to put at those indices
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::put;
///
/// let mut arr = Array::from_vec(vec![0, 0, 0, 0, 0]);
/// let indices = Array::from_vec(vec![0, 2, 4]);
/// put(&mut arr, &indices, &[10, 20, 30]).expect("operation should succeed");
/// // arr is now [10, 0, 20, 0, 30]
/// ```
pub fn put<T: Clone>(array: &mut Array<T>, indices: &Array<usize>, values: &[T]) -> Result<()> {
    if values.is_empty() {
        return Err(NumRs2Error::ValueError(
            "values array cannot be empty".to_string(),
        ));
    }

    // Deliberately still `indices.to_vec()`, not `operand(indices)`: same
    // reasoning and same measurement as `place` just above (this sweep
    // tried hoisting `indices` into a shared `operand` borrow for both the
    // validation pass and the write pass, which is the behavior-preserving
    // fix -- fusing the two into one pass, as `take`'s `None` arm does,
    // would let indices before the first invalid one already be written
    // before an early `Err` return, a caller-visible partial mutation the
    // original never produced -- but even the correctly-two-pass `operand`
    // version measured slower than this `to_vec()` original:
    // `probe_put_perf_vs_naive_to_vec` below is 0.56x at full release,
    // 1.18x only before that probe's "naive" comparison was made equally
    // generic; see `place`'s comment for the fuller explanation and why
    // this is a revert, not an oversight).
    let indices_vec = indices.to_vec();
    let array_len = array.size();

    // Validate indices
    for &idx in &indices_vec {
        if idx >= array_len {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "index {} is out of bounds for array of size {}",
                idx, array_len
            )));
        }
    }

    // Get mutable slice
    let array_data = array
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    for (i, &idx) in indices_vec.iter().enumerate() {
        array_data[idx] = values[i % values.len()].clone();
    }

    Ok(())
}

/// Put values into array using a mask
///
/// # Arguments
/// * `array` - Array to modify (modified in-place)
/// * `mask` - Boolean mask array  
/// * `values` - Array of values to put where mask is True
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::putmask;
///
/// let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// let mask = Array::from_vec(vec![false, true, false, true, false]);
/// let values = Array::from_vec(vec![10, 20]);
/// putmask(&mut arr, &mask, &values).expect("operation should succeed");
/// // arr is now [1, 10, 3, 20, 5]
/// ```
pub fn putmask<T: Clone>(
    array: &mut Array<T>,
    mask: &Array<bool>,
    values: &Array<T>,
) -> Result<()> {
    // `place` only reads `values` (indexed, never mutated/moved), so a
    // zero-copy-when-possible `operand` borrow replaces the old owned
    // `values.to_vec()`; `&values_op` coerces to `&[T]` via `Operand`'s
    // `Deref`, matching `place`'s parameter type.
    let values_op = crate::kernels::borrow::operand(values);
    place(array, mask, &values_op)
}

/// Take values from array along an axis using indices
///
/// This is the canonical `take_along_axis` implementation;
/// [`crate::indexing::take_put::take_along_axis`] (a one-line delegate)
/// forwards here.
///
/// # Arguments
/// * `array` - Input array
/// * `indices` - Array of indices to take
/// * `axis` - Axis along which to take values
///
/// # Returns
/// * `Result<Array<T>>` - Array with values taken along the specified axis
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::take_along_axis;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let indices = Array::from_vec(vec![0, 2, 1, 0]).reshape(&[2, 2]);
/// let result = take_along_axis(&arr, &indices, 1).expect("operation should succeed");
/// // For row 0: takes elements at indices [0, 2] -> [1, 3]
/// // For row 1: takes elements at indices [1, 0] -> [5, 4]
/// // Result is [[1, 3], [5, 4]]
/// assert_eq!(result.to_vec(), vec![1, 3, 5, 4]);
///
/// // `array` and `indices` need not match exactly on non-`axis` dimensions:
/// // NumPy broadcasts them there (equal, or either is 1), same as
/// // `numpy.take_along_axis`.
/// let arr2 = Array::from_vec(vec![10, 20, 30, 40, 50, 60]).reshape(&[2, 3]);
/// let idx2 = Array::from_vec(vec![2, 0]).reshape(&[1, 2]);
/// let result2 = take_along_axis(&arr2, &idx2, 1).expect("operation should succeed");
/// assert_eq!(result2.shape(), vec![2, 2]);
/// assert_eq!(result2.to_vec(), vec![30, 10, 60, 40]);
/// ```
pub fn take_along_axis<T: Clone + Zero>(
    array: &Array<T>,
    indices: &Array<usize>,
    axis: usize,
) -> Result<Array<T>> {
    let arr_shape = array.shape();
    let ind_shape = indices.shape();

    if axis >= arr_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            arr_shape.len()
        )));
    }

    // NumPy requires `array` and `indices` to have the same number of
    // dimensions (no implicit reshape/prepend).
    if arr_shape.len() != ind_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "array and indices must have same number of dimensions, got {} and {}",
            arr_shape.len(),
            ind_shape.len()
        )));
    }

    let ndim = arr_shape.len();

    // NumPy broadcasts `array` and `indices` against each other on every
    // dimension *other than* `axis` (equal, or either is 1) rather than
    // requiring an exact match there -- this previously rejected e.g.
    // `array` shape (2,3)/`indices` shape (1,2) with a `ShapeMismatch`,
    // where real `numpy.take_along_axis` broadcasts dimension 0 (1 -> 2)
    // and returns shape (2,2). `axis` itself is exempt from this: `array`'s
    // extent there only bounds the index *values* (checked below), while
    // `indices`' extent there fixes the output's size along it.
    let mut out_shape = vec![0usize; ndim];
    for d in 0..ndim {
        if d == axis {
            out_shape[d] = ind_shape[d];
            continue;
        }
        out_shape[d] = match (arr_shape[d], ind_shape[d]) {
            (a, i) if a == i => a,
            (1, i) => i,
            (a, 1) => a,
            _ => {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: arr_shape.clone(),
                    actual: ind_shape.clone(),
                })
            }
        };
    }

    let total: usize = out_shape.iter().product();
    let mut result_data = Vec::with_capacity(total);
    let mut out_pos = vec![0usize; ndim];

    for _ in 0..total {
        // A broadcast (size-1) dimension always reads its single element
        // (position 0) regardless of where we are in the output; a
        // non-broadcast dimension reads the output's own coordinate.
        let mut ind_pos = out_pos.clone();
        let mut arr_pos = out_pos.clone();
        for d in 0..ndim {
            if d == axis {
                continue;
            }
            if ind_shape[d] == 1 {
                ind_pos[d] = 0;
            }
            if arr_shape[d] == 1 {
                arr_pos[d] = 0;
            }
        }

        let idx_value = indices.get(&ind_pos)?;
        if idx_value >= arr_shape[axis] {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "index {} is out of bounds for axis {} with size {}",
                idx_value, axis, arr_shape[axis]
            )));
        }
        arr_pos[axis] = idx_value;

        result_data.push(array.get(&arr_pos)?);

        // Advance `out_pos` in row-major (C) order.
        for d in (0..ndim).rev() {
            out_pos[d] += 1;
            if out_pos[d] < out_shape[d] {
                break;
            }
            out_pos[d] = 0;
        }
    }

    Array::from_vec_shape(result_data, &out_shape)
}

/// Apply a function to 1-D slices along the given axis
///
/// # Arguments
/// * `func` - Function to apply to each 1-D slice
/// * `array` - Input array
/// * `axis` - Axis along which array is sliced
///
/// # Returns
/// * `Result<Array<U>>` - Array with function applied along axis
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::apply_along_axis;
///
/// // Sum along axis
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
/// let result = apply_along_axis(
///     |slice: &Array<f64>| -> f64 { slice.sum() },
///     &arr,
///     1
/// ).expect("operation should succeed");
/// // Sums each row: [6.0, 15.0]
/// ```
pub fn apply_along_axis<T, U, F>(func: F, array: &Array<T>, axis: usize) -> Result<Array<U>>
where
    T: Clone + Zero,
    U: Clone + Zero,
    F: Fn(&Array<T>) -> U,
{
    let shape = array.shape();

    if axis >= shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            shape.len()
        )));
    }

    // Calculate output shape (remove the axis dimension)
    let mut out_shape = shape.clone();
    out_shape.remove(axis);

    if out_shape.is_empty() {
        // If only one dimension, return scalar as 1-element array
        let result = func(array);
        return Ok(Array::from_vec(vec![result]));
    }

    let mut result_data = Vec::new();

    // Number of slices to process
    let n_slices: usize = out_shape.iter().product();

    for slice_idx in 0..n_slices {
        // Convert linear index to multi-dimensional position
        let mut slice_pos = vec![0; out_shape.len()];
        let mut temp = slice_idx;
        for i in (0..out_shape.len()).rev() {
            slice_pos[i] = temp % out_shape[i];
            temp /= out_shape[i];
        }

        // Extract 1-D slice along the axis
        let mut slice_data = Vec::with_capacity(shape[axis]);

        for i in 0..shape[axis] {
            // Build full position in original array
            let mut full_pos = Vec::with_capacity(shape.len());
            let mut slice_dim = 0;

            for dim in 0..shape.len() {
                if dim == axis {
                    full_pos.push(i);
                } else {
                    full_pos.push(slice_pos[slice_dim]);
                    slice_dim += 1;
                }
            }

            slice_data.push(array.get(&full_pos)?);
        }

        // Apply function to slice
        let slice_array = Array::from_vec(slice_data);
        let result = func(&slice_array);
        result_data.push(result);
    }

    Array::from_vec_shape(result_data, &out_shape)
}

/// Apply a function over multiple axes
///
/// # Arguments
/// * `func` - Function that takes an array and returns a scalar
/// * `array` - Input array
/// * `axes` - Axes over which to apply the function
///
/// # Returns
/// * `Result<Array<T>>` - Result array with specified axes removed
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::apply_over_axes;
///
/// // Sum over multiple axes
/// let arr = Array::from_vec(vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0
/// ]);
/// let arr = arr.reshape(&[2, 2, 2]);
///
/// let result = apply_over_axes(
///     |a: &Array<f64>| -> Result<Array<f64>> { a.sum_axis(0) },
///     &arr,
///     &[1]
/// ).expect("operation should succeed");
/// // Sums over axis 1, reducing dimension by 1
/// ```
pub fn apply_over_axes<T, F>(func: F, array: &Array<T>, axes: &[usize]) -> Result<Array<T>>
where
    T: Clone + Zero,
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    let mut result = array.clone();

    // Sort axes in descending order to handle removal correctly
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));

    for &axis in &sorted_axes {
        if axis >= result.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "axis {} is out of bounds for array of dimension {}",
                axis,
                result.ndim()
            )));
        }

        // Apply function along this axis
        let temp = func(&result)?;

        // The function should reduce the dimension along the axis
        if temp.ndim() != result.ndim() - 1 {
            return Err(NumRs2Error::InvalidOperation(
                "Function must reduce dimension by 1".to_string(),
            ));
        }

        result = temp;
    }

    Ok(result)
}

/// Take elements from array using integer array indices (fancy indexing)
///
/// This implements NumPy-style fancy indexing where an array of integers
/// is used to select elements from the input array. The result has the
/// same shape as the indices array.
///
/// # Arguments
/// * `array` - Input array
/// * `indices` - Integer array of indices to take
/// * `axis` - Optional axis along which to take. If None, array is flattened
///
/// # Returns
/// * `Result<Array<T>>` - Array with selected elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::take;
///
/// // 1-D fancy indexing
/// let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
/// let indices = Array::from_vec(vec![0, 2, 4, 1]);
/// let result = take(&arr, &indices, None).expect("operation should succeed");
/// // Returns [10, 30, 50, 20]
///
/// // 2-D fancy indexing along axis
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let indices = Array::from_vec(vec![2, 0, 1]);
/// let result = take(&arr, &indices, Some(1)).expect("operation should succeed");
/// // Takes columns [2, 0, 1] from each row
/// ```
pub fn take<T: Clone + Zero>(
    array: &Array<T>,
    indices: &Array<usize>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Flatten array and take elements. `flat` is indexed by
            // arbitrary (repeatable, out-of-order) values from `indices`,
            // so it needs slice-like random access -- `operand` borrows
            // it zero-copy (when contiguous) instead of the old owned
            // `array.to_vec()`.
            //
            // `indices` is walked sequentially, but *twice* in this arm
            // (once to validate, once to gather) -- an earlier version of
            // this fix left that as two `indices.array().iter()` passes
            // (recipe [B]: no buffer), on the reasoning that each pass is
            // individually sequential. Measured against the pre-sweep
            // `indices.to_vec()` baseline (`probe_take_none_perf_vs_naive_to_vec_pair`
            // below), that was **9x slower in release** and ~1.7x slower
            // even at `[profile.test]`'s `opt-level = 2`: `indices.array()`
            // is an `NdArray<usize, IxDyn>` (see `Array::array`'s
            // signature), and `IxDyn`'s generic, rank-erased iterator
            // re-walks dynamic stride bookkeeping on every element every
            // time it's constructed -- LLVM cannot fold that down to a
            // pointer-bump loop the way it does for a `&[usize]`, so
            // paying that cost *twice* over the whole array was actually
            // worse than the one `memcpy`-speed `to_vec()` copy it
            // replaced. Hoisting `indices` into an `operand` borrow once
            // (zero-copy here too, `indices` being contiguous) and fusing
            // the validate-then-gather double pass into one loop over
            // that single fast `&[usize]` iterator fixes both: one flat
            // pass instead of two `IxDyn` passes, and no copy either way.
            // The fused loop preserves the original's exact error
            // behavior -- both scan `indices` in the same left-to-right
            // order, so the first out-of-bounds index reported is
            // identical, and an error return simply drops whatever prefix
            // of `result` had been gathered so far, exactly as the
            // two-pass version discarded its not-yet-started `result` on
            // a validation failure.
            let flat = crate::kernels::borrow::operand(array);
            let idx_op = crate::kernels::borrow::operand(indices);

            let mut result = Vec::with_capacity(idx_op.len());
            for &idx in idx_op.iter() {
                if idx >= flat.len() {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "index {} is out of bounds for flattened array of size {}",
                        idx,
                        flat.len()
                    )));
                }
                result.push(flat[idx].clone());
            }

            // Result has same shape as indices
            Ok(Array::from_vec_shape(result, &indices.shape())?)
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

            // `idx_vec[current_pos[ax]]` below indexes by a position that
            // cycles repeatedly as the multi-dimensional loop advances
            // (not a single sequential pass), so this needs slice-like
            // random access -- `operand` borrows it zero-copy (when
            // contiguous) instead of the old owned `indices.to_vec()`.
            let idx_vec = crate::kernels::borrow::operand(indices);

            // Validate indices
            for &idx in idx_vec.iter() {
                if idx >= shape[ax] {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, ax, shape[ax]
                    )));
                }
            }

            // Calculate result shape
            let mut result_shape = shape.clone();
            result_shape[ax] = idx_vec.len();

            let mut result_data = Vec::with_capacity(result_shape.iter().product());

            // Iterate through all positions
            let mut current_pos = vec![0; shape.len()];
            let total_out: usize = result_shape.iter().product();

            for _ in 0..total_out {
                // Map position in result to position in source
                let mut source_pos = current_pos.clone();
                source_pos[ax] = idx_vec[current_pos[ax]];

                let value = array.get(&source_pos)?;
                result_data.push(value);

                // Increment position
                let mut carry = true;
                for dim in (0..result_shape.len()).rev() {
                    if carry {
                        current_pos[dim] += 1;
                        carry = current_pos[dim] >= result_shape[dim];
                        if carry {
                            current_pos[dim] = 0;
                        }
                    }
                }
            }

            Ok(Array::from_vec_shape(result_data, &result_shape)?)
        }
    }
}

/// Multi-dimensional fancy indexing using coordinate arrays
///
/// Select elements at specific coordinates specified by arrays of indices.
/// This is equivalent to `array[indices[0], indices[1], ...]` in NumPy.
///
/// # Arguments
/// * `array` - Input array
/// * `indices` - Slice of index arrays, one per dimension
///
/// # Returns
/// * `Result<Array<T>>` - 1-D array with selected elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::fancy_index;
///
/// let arr = Array::from_vec(vec![
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9
/// ]).reshape(&[3, 3]);
///
/// // Select elements at (0,0), (1,1), (2,2) - the diagonal
/// let row_idx = Array::from_vec(vec![0, 1, 2]);
/// let col_idx = Array::from_vec(vec![0, 1, 2]);
/// let result = fancy_index(&arr, &[row_idx, col_idx]).expect("operation should succeed");
/// // Returns [1, 5, 9]
///
/// // Select elements at (0,2), (2,0)
/// let row_idx = Array::from_vec(vec![0, 2]);
/// let col_idx = Array::from_vec(vec![2, 0]);
/// let result = fancy_index(&arr, &[row_idx, col_idx]).expect("operation should succeed");
/// // Returns [3, 7]
/// ```
pub fn fancy_index<T: Clone + Zero>(
    array: &Array<T>,
    indices: &[Array<usize>],
) -> Result<Array<T>> {
    let shape = array.shape();

    if indices.is_empty() {
        return Err(NumRs2Error::ValueError(
            "indices array cannot be empty".to_string(),
        ));
    }

    if indices.len() != shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "number of index arrays ({}) must match array dimensions ({})",
            indices.len(),
            shape.len()
        )));
    }

    // All index arrays must have the same shape
    let idx_shape = indices[0].shape();
    for idx_arr in &indices[1..] {
        if idx_arr.shape() != idx_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: idx_shape.clone(),
                actual: idx_arr.shape(),
            });
        }
    }

    let num_elements = indices[0].size();
    let mut result_data = Vec::with_capacity(num_elements);

    // Borrow all index arrays zero-copy (when contiguous). Each is
    // indexed by `i` below across every element position, not walked
    // once sequentially, so `operand`'s slice-like random access replaces
    // the old owned `arr.to_vec()` per index array.
    let idx_vecs: Vec<_> = indices
        .iter()
        .map(crate::kernels::borrow::operand)
        .collect();

    // For each element position
    for i in 0..num_elements {
        // Build coordinate from index arrays
        let mut coord = Vec::with_capacity(shape.len());
        for (dim, idx_vec) in idx_vecs.iter().enumerate() {
            let idx = idx_vec[i];

            // Validate index
            if idx >= shape[dim] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "index {} is out of bounds for dimension {} with size {}",
                    idx, dim, shape[dim]
                )));
            }

            coord.push(idx);
        }

        // Get element at coordinate
        let value = array.get(&coord)?;
        result_data.push(value);
    }

    // Result has same shape as the index arrays
    Array::from_vec_shape(result_data, &idx_shape)
}

/// Boolean indexing convenience method
///
/// Extract elements from array using a boolean mask. This is a convenience
/// wrapper around `extract()` that returns a flattened array.
///
/// # Arguments
/// * `array` - Input array
/// * `mask` - Boolean mask with same shape as array
///
/// # Returns
/// * `Result<Array<T>>` - 1-D array with elements where mask is true
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::boolean_index;
///
/// let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
/// let mask = Array::from_vec(vec![true, false, true, false, true]);
/// let result = boolean_index(&arr, &mask).expect("operation should succeed");
/// // Returns [10, 30, 50]
///
/// // Works with comparisons
/// let arr = Array::from_vec(vec![1, 5, 3, 8, 2]);
/// let mask = arr.map(|x| x > 3);  // [false, true, false, true, false]
/// let result = boolean_index(&arr, &mask).expect("operation should succeed");
/// // Returns [5, 8]
/// ```
pub fn boolean_index<T: Clone>(array: &Array<T>, mask: &Array<bool>) -> Result<Array<T>> {
    extract(array, mask)
}

/// Choose elements from arrays based on conditions
///
/// Given a list of conditions and a list of choices, return an array drawn
/// from elements in choices, based on conditions. This is similar to NumPy's
/// `select()` function.
///
/// # Arguments
/// * `conditions` - List of boolean arrays. Each must have same shape
/// * `choices` - List of arrays to choose from. Each must match conditions shape
/// * `default` - Default value when no condition is met
///
/// # Returns
/// * `Result<Array<T>>` - Array with elements chosen based on conditions
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::advanced_indexing::select;
///
/// let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
///
/// // Condition 1: x < 3 -> return x * 10
/// let cond1 = arr.map(|x| x < 3);
/// let choice1 = arr.map(|x| x * 10);
///
/// // Condition 2: x >= 3 -> return x * 100
/// let cond2 = arr.map(|x| x >= 3);
/// let choice2 = arr.map(|x| x * 100);
///
/// let result = select(&[cond1, cond2], &[choice1, choice2], 0).expect("operation should succeed");
/// // Returns [10, 20, 300, 400, 500]
/// ```
///
/// Delegates to the canonical broadcasting core shared with
/// [`crate::array_ops::conditional::select`] (that function's
/// `Option<T> + Zero`-defaulted `default` versus this one's required
/// `default: T` is the only difference); see its docs for full
/// NumPy-compatible semantics, including its more permissive handling of
/// `condlist`/`choicelist` entries that are broadcastable but not
/// identically shaped (this copy previously required an exact match).
pub fn select<T: Clone>(
    conditions: &[Array<bool>],
    choices: &[Array<T>],
    default: T,
) -> Result<Array<T>> {
    let cond_refs: Vec<&Array<bool>> = conditions.iter().collect();
    let choice_refs: Vec<&Array<T>> = choices.iter().collect();
    crate::array_ops::conditional::select_with_default(&cond_refs, &choice_refs, default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `condition` shorter than `array` (axis=None): NumPy treats the
    /// unreached tail as simply absent, not as `false`. Pinned against
    /// `numpy.compress([True, False], [1,2,3,4])` (numpy 2.4.2) == `[1]`.
    #[test]
    fn test_compress_flat_condition_shorter_than_array() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        let cond = Array::from_vec(vec![true, false]);
        let result = compress(&arr, &cond, None).expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![1]);
    }

    /// `condition` longer than `array` (axis=None) with only trailing
    /// `false`s is not an error -- only a `true` beyond the array's length
    /// is. Pinned against numpy 2.4.2:
    /// `np.compress([True,False,True,False,False], [1,2,3])` == `[1,3]`.
    #[test]
    fn test_compress_flat_condition_longer_with_trailing_false() {
        let arr = Array::from_vec(vec![1, 2, 3]);
        let cond = Array::from_vec(vec![true, false, true, false, false]);
        let result = compress(&arr, &cond, None).expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![1, 3]);
    }

    /// A `true` beyond `array`'s length IS an error, whether reached via
    /// the flattened (`axis=None`) or per-axis path. Pinned against numpy
    /// 2.4.2: `np.compress([True,False,True,False,True], [1,2,3])` raises
    /// `IndexError: index 4 is out of bounds for axis 0 with size 3`.
    #[test]
    fn test_compress_flat_condition_true_out_of_bounds_errors() {
        let arr = Array::from_vec(vec![1, 2, 3]);
        let cond = Array::from_vec(vec![true, false, true, false, true]);
        assert!(compress(&arr, &cond, None).is_err());
    }

    /// This exact axis + shorter-condition combination was the concrete bug
    /// this sweep fixed: the previous implementation required
    /// `condition.size() == shape[axis]` and errored otherwise, and its
    /// axis-given branch (unlike its own `axis=None` branch) additionally
    /// never accounted for a `condition` even one entry longer needing an
    /// out-of-bounds check rather than a silent index-into-nothing. Pinned
    /// against numpy 2.4.2:
    /// `np.compress([True], np.arange(1,7).reshape(2,3), axis=1)` ==
    /// `[[1],[4]]`.
    #[test]
    fn test_compress_axis_condition_shorter_than_axis_size() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let cond = Array::from_vec(vec![true]);
        let result = compress(&arr, &cond, Some(1)).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2, 1]);
        assert_eq!(result.to_vec(), vec![1, 4]);
    }

    /// Pinned against numpy 2.4.2:
    /// `np.compress([True,False,True,True], np.arange(1,7).reshape(2,3),
    /// axis=1)` raises `IndexError: index 3 is out of bounds for axis 1
    /// with size 3` (excess `true` beyond the axis size is an error; excess
    /// `false`, as in the case above generalized, would not be).
    #[test]
    fn test_compress_axis_condition_true_out_of_bounds_errors() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let cond = Array::from_vec(vec![true, false, true, true]);
        assert!(compress(&arr, &cond, Some(1)).is_err());
    }

    /// A non-1-D `condition` is rejected regardless of `axis`. Pinned
    /// against numpy 2.4.2: `np.compress([[True,False]], [1,2])` raises
    /// `ValueError: condition must be a 1-d array`.
    #[test]
    fn test_compress_condition_must_be_1d() {
        let arr = Array::from_vec(vec![1, 2]);
        let cond = Array::from_vec(vec![true, false]).reshape(&[1, 2]);
        assert!(compress(&arr, &cond, None).is_err());
    }

    #[test]
    fn test_take_along_axis_exact_shape() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let indices = Array::from_vec(vec![0, 2, 1, 0]).reshape(&[2, 2]);
        let result = take_along_axis(&arr, &indices, 1).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![1, 3, 5, 4]);
    }

    /// Regression test: `array` and `indices` need only broadcast (not
    /// match exactly) on dimensions other than `axis`. Pinned against
    /// `numpy.take_along_axis` (numpy 2.4.2):
    /// `np.take_along_axis(np.array([[10,20,30],[40,50,60]]),
    /// np.array([[2,0]]), axis=1)` == `[[30, 10], [60, 40]]`, shape `(2,2)`.
    /// This previously raised `ShapeMismatch` because dimension 0 (array:
    /// 2, indices: 1) was required to match exactly rather than broadcast.
    #[test]
    fn test_take_along_axis_broadcasts_non_axis_dims() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50, 60]).reshape(&[2, 3]);
        let indices = Array::from_vec(vec![2, 0]).reshape(&[1, 2]);
        let result = take_along_axis(&arr, &indices, 1).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![30, 10, 60, 40]);
    }

    /// Same broadcasting rule, but with `indices` the larger side (shape
    /// (2,2)) and `array` the size-1 side (shape (1,3)). Pinned against
    /// `numpy.take_along_axis` 2.4.2:
    /// `np.take_along_axis(np.array([[10,20,30]]),
    /// np.array([[2,0],[1,1]]), axis=1)` == `[[30, 10], [20, 20]]`.
    #[test]
    fn test_take_along_axis_broadcasts_array_side() {
        let arr = Array::from_vec(vec![10, 20, 30]).reshape(&[1, 3]);
        let indices = Array::from_vec(vec![2, 0, 1, 1]).reshape(&[2, 2]);
        let result = take_along_axis(&arr, &indices, 1).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![30, 10, 20, 20]);
    }

    #[test]
    fn test_take_along_axis_out_of_bounds_index_errors() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let indices = Array::from_vec(vec![0, 5]).reshape(&[1, 2]);
        assert!(take_along_axis(&arr, &indices, 1).is_err());
    }

    // `place`/`put`/`putmask` had no coverage at all before this sweep
    // touched them (see the `place`/`put` fix comments above for why
    // `mask`/`indices` are now hoisted into one shared `operand` borrow
    // instead of two separate `.array().iter()` passes) -- added here
    // alongside that fix, not just for the new behavior, but because a
    // correctness net was missing for these functions entirely.

    #[test]
    fn test_place_basic() {
        let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![false, true, false, true, false]);
        place(&mut arr, &mask, &[10, 20]).expect("place should succeed");
        assert_eq!(arr.to_vec(), vec![1, 10, 3, 20, 5]);
    }

    #[test]
    fn test_place_values_cycle_when_fewer_than_masked() {
        let mut arr = Array::from_vec(vec![0, 0, 0, 0, 0]);
        let mask = Array::from_vec(vec![true, true, true, true, true]);
        place(&mut arr, &mask, &[1, 2]).expect("place should succeed");
        // Only one value supplied per two `true`s: cycles 1, 2, 1, 2, 1.
        assert_eq!(arr.to_vec(), vec![1, 2, 1, 2, 1]);
    }

    #[test]
    fn test_place_no_true_values_is_noop() {
        let mut arr = Array::from_vec(vec![1, 2, 3]);
        let mask = Array::from_vec(vec![false, false, false]);
        place(&mut arr, &mask, &[99]).expect("place should succeed");
        assert_eq!(arr.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_place_empty_values_errors() {
        let mut arr = Array::from_vec(vec![1, 2, 3]);
        let mask = Array::from_vec(vec![true, false, false]);
        let empty: [i32; 0] = [];
        assert!(place(&mut arr, &mask, &empty).is_err());
    }

    #[test]
    fn test_put_basic() {
        let mut arr = Array::from_vec(vec![0, 0, 0, 0, 0]);
        let indices = Array::from_vec(vec![0, 2, 4]);
        put(&mut arr, &indices, &[10, 20, 30]).expect("put should succeed");
        assert_eq!(arr.to_vec(), vec![10, 0, 20, 0, 30]);
    }

    /// Explicitly requested when `put` gained a hoisted-`operand`
    /// validation pass: confirm the out-of-bounds error path still fires,
    /// and (since `put`'s two passes are deliberately kept separate so
    /// validation never touches `array` -- see `put`'s fix comment) that
    /// `array` is left completely untouched when it does.
    #[test]
    fn test_put_out_of_bounds_leaves_array_untouched() {
        let mut arr = Array::from_vec(vec![1, 2, 3]);
        let original = arr.to_vec();
        let indices = Array::from_vec(vec![0, 5]); // 5 is out of bounds
        let result = put(&mut arr, &indices, &[10, 20]);
        assert!(result.is_err());
        assert_eq!(
            arr.to_vec(),
            original,
            "put must not partially mutate on error"
        );
    }

    #[test]
    fn test_putmask_basic() {
        let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![false, true, false, true, false]);
        let values = Array::from_vec(vec![10, 20]);
        putmask(&mut arr, &mask, &values).expect("putmask should succeed");
        assert_eq!(arr.to_vec(), vec![1, 10, 3, 20, 5]);
    }

    /// Regression guard, not just a perf note: this sweep tried converting
    /// `place`'s `mask.to_vec()` to a hoisted `operand(mask)` borrow (the
    /// same fix that helped `take`/`outer`/`real_if_close` elsewhere in
    /// this lane), and it measured *slower* than the `to_vec()` `place`
    /// keeps today -- 0.80x at full release, 0.69x at `[profile.test]`'s
    /// `opt-level = 2` (both against this exact generic-matched
    /// comparison). `operand_based_place` reproduces that rejected
    /// alternative so the comparison stays runnable: both approaches
    /// still produce identical values (the assertion below passes either
    /// way), so the eprintln! ratio -- not the assertion -- is the signal
    /// to watch if someone re-applies the `operand` conversion here
    /// without re-measuring.
    #[test]
    fn probe_place_perf_vs_naive_to_vec() {
        // Generic over `T: Clone`, matching `place<T: Clone>`'s own bound
        // exactly (instantiated at `f64` below, same as the real call) --
        // comparing a hand-specialized concrete `fn(&mut Array<f64>, ..)`
        // against the actual (generic) `place` would conflate "to_vec()
        // vs operand()" with "concrete vs generic monomorphization",
        // which is a different question this probe isn't meant to answer.
        fn operand_based_place<T: Clone>(array: &mut Array<T>, mask: &Array<bool>, values: &[T]) {
            let mask_op = crate::kernels::borrow::operand(mask);
            let num_true = mask_op.iter().filter(|&&x| x).count();
            if num_true == 0 {
                return;
            }
            let array_data = array
                .array_mut()
                .as_slice_mut()
                .expect("test array is contiguous");
            let mut value_idx = 0;
            for (i, &is_true) in mask_op.iter().enumerate() {
                if is_true {
                    array_data[i] = values[value_idx % values.len()].clone();
                    value_idx += 1;
                }
            }
        }

        let n = 100_000;
        let values = [1.0, 2.0, 3.0];
        let mask_src = Array::from_vec((0..n).map(|i| i % 3 == 0).collect::<Vec<_>>());
        let iters = 100;

        let mut a1 = Array::from_vec(vec![0.0; n]);
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            place(&mut a1, &mask_src, &values).expect("place should succeed");
        }
        let to_vec_current = t0.elapsed();

        let mut a2 = Array::from_vec(vec![0.0; n]);
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            operand_based_place(&mut a2, &mask_src, &values);
        }
        let operand_rejected = t1.elapsed();

        eprintln!(
            "[place, n={n}] to_vec(current)={:.1}us/iter operand(rejected)={:.1}us/iter ({:.2}x)",
            to_vec_current.as_secs_f64() * 1e6 / iters as f64,
            operand_rejected.as_secs_f64() * 1e6 / iters as f64,
            operand_rejected.as_secs_f64() / to_vec_current.as_secs_f64(),
        );

        assert_eq!(
            a1.to_vec(),
            a2.to_vec(),
            "both approaches must produce the same values"
        );
    }

    /// Same shape and same purpose as `probe_place_perf_vs_naive_to_vec`
    /// (see its doc comment): a regression guard reproducing the
    /// `operand`-based alternative this sweep tried and rejected for
    /// `put`'s `indices.to_vec()` (0.56x at full release, 1.18x only
    /// before this probe's comparison was made equally generic -- see
    /// `put`'s comment for the full explanation). Two passes over the
    /// rejected `operand` borrow are kept separate here too (never fused
    /// into one, unlike `take`'s `None` arm) -- fusing would itself be an
    /// observable behavior change (partial mutation on an out-of-bounds
    /// error), a second, independent reason not to prefer this
    /// alternative even where it happened to win.
    #[test]
    fn probe_put_perf_vs_naive_to_vec() {
        // Generic over `T: Clone`, matching `put<T: Clone>` (see
        // `probe_place_perf_vs_naive_to_vec`'s comment above for why).
        fn operand_based_put<T: Clone>(array: &mut Array<T>, indices: &Array<usize>, values: &[T]) {
            let idx_op = crate::kernels::borrow::operand(indices);
            let array_len = array.size();
            for &idx in idx_op.iter() {
                assert!(idx < array_len, "index out of bounds");
            }
            let array_data = array
                .array_mut()
                .as_slice_mut()
                .expect("test array is contiguous");
            for (i, &idx) in idx_op.iter().enumerate() {
                array_data[idx] = values[i % values.len()].clone();
            }
        }

        let n = 100_000;
        let values = [1.0, 2.0, 3.0];
        let indices_src = Array::from_vec((0..n).map(|i| i % 997).collect::<Vec<_>>());
        let iters = 100;

        let mut a1 = Array::from_vec(vec![0.0; n]);
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            put(&mut a1, &indices_src, &values).expect("put should succeed");
        }
        let to_vec_current = t0.elapsed();

        let mut a2 = Array::from_vec(vec![0.0; n]);
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            operand_based_put(&mut a2, &indices_src, &values);
        }
        let operand_rejected = t1.elapsed();

        eprintln!(
            "[put, n={n}] to_vec(current)={:.1}us/iter operand(rejected)={:.1}us/iter ({:.2}x)",
            to_vec_current.as_secs_f64() * 1e6 / iters as f64,
            operand_rejected.as_secs_f64() * 1e6 / iters as f64,
            operand_rejected.as_secs_f64() / to_vec_current.as_secs_f64(),
        );

        assert_eq!(
            a1.to_vec(),
            a2.to_vec(),
            "both approaches must produce the same values"
        );
    }

    #[test]
    fn test_take_1d() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
        let indices = Array::from_vec(vec![0, 2, 4, 1]);

        let result = take(&arr, &indices, None).expect("operation should succeed");

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.to_vec(), vec![10, 30, 50, 20]);
    }

    #[test]
    fn test_take_2d_no_axis() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let indices = Array::from_vec(vec![0, 3, 5]);

        let result = take(&arr, &indices, None).expect("operation should succeed");

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![1, 4, 6]);
    }

    #[test]
    fn test_take_along_axis() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let indices = Array::from_vec(vec![2, 0, 1]);

        let result = take(&arr, &indices, Some(1)).expect("operation should succeed");

        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: [1, 2, 3] -> indices [2, 0, 1] -> [3, 1, 2]
        // Row 1: [4, 5, 6] -> indices [2, 0, 1] -> [6, 4, 5]
        assert_eq!(result.to_vec(), vec![3, 1, 2, 6, 4, 5]);
    }

    #[test]
    fn test_take_out_of_bounds() {
        let arr = Array::from_vec(vec![1, 2, 3]);
        let indices = Array::from_vec(vec![0, 5]); // 5 is out of bounds

        let result = take(&arr, &indices, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fancy_index_diagonal() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

        let row_idx = Array::from_vec(vec![0, 1, 2]);
        let col_idx = Array::from_vec(vec![0, 1, 2]);

        let result = fancy_index(&arr, &[row_idx, col_idx]).expect("operation should succeed");

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![1, 5, 9]);
    }

    #[test]
    fn test_fancy_index_arbitrary_coords() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

        let row_idx = Array::from_vec(vec![0, 2, 1]);
        let col_idx = Array::from_vec(vec![2, 0, 1]);

        let result = fancy_index(&arr, &[row_idx, col_idx]).expect("operation should succeed");

        assert_eq!(result.shape(), &[3]);
        // (0,2) -> 3, (2,0) -> 7, (1,1) -> 5
        assert_eq!(result.to_vec(), vec![3, 7, 5]);
    }

    #[test]
    fn test_fancy_index_2d_indices() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50, 60]).reshape(&[2, 3]);

        let row_idx = Array::from_vec(vec![0, 1, 0, 1]).reshape(&[2, 2]);
        let col_idx = Array::from_vec(vec![0, 1, 2, 2]).reshape(&[2, 2]);

        let result = fancy_index(&arr, &[row_idx, col_idx]).expect("operation should succeed");

        assert_eq!(result.shape(), &[2, 2]);
        // (0,0)->10, (1,1)->50, (0,2)->30, (1,2)->60
        assert_eq!(result.to_vec(), vec![10, 50, 30, 60]);
    }

    #[test]
    fn test_fancy_index_mismatched_shapes() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);

        let row_idx = Array::from_vec(vec![0, 1]);
        let col_idx = Array::from_vec(vec![0, 1, 2]); // Different shape

        let result = fancy_index(&arr, &[row_idx, col_idx]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fancy_index_out_of_bounds() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);

        let row_idx = Array::from_vec(vec![0, 3]); // 3 is out of bounds
        let col_idx = Array::from_vec(vec![0, 0]);

        let result = fancy_index(&arr, &[row_idx, col_idx]);
        assert!(result.is_err());
    }

    #[test]
    fn test_boolean_index_simple() {
        let arr = Array::from_vec(vec![10, 20, 30, 40, 50]);
        let mask = Array::from_vec(vec![true, false, true, false, true]);

        let result = boolean_index(&arr, &mask).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![10, 30, 50]);
    }

    #[test]
    fn test_boolean_index_with_comparison() {
        let arr = Array::from_vec(vec![1, 5, 3, 8, 2]);
        let mask = arr.map(|x| x > 3);

        let result = boolean_index(&arr, &mask).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![5, 8]);
    }

    #[test]
    fn test_boolean_index_2d() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let mask = Array::from_vec(vec![true, false, true, false, true, false]).reshape(&[2, 3]);

        let result = boolean_index(&arr, &mask).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![1, 3, 5]);
    }

    #[test]
    fn test_boolean_index_all_false() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![false; 5]);

        let result = boolean_index(&arr, &mask).expect("operation should succeed");

        assert_eq!(result.size(), 0);
    }

    #[test]
    fn test_boolean_index_all_true() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![true; 5]);

        let result = boolean_index(&arr, &mask).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_select_simple() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);

        let cond1 = arr.map(|x| x < 3);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = arr.map(|x| x >= 3);
        let choice2 = arr.map(|x| x * 100);

        let result =
            select(&[cond1, cond2], &[choice1, choice2], 0).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![10, 20, 300, 400, 500]);
    }

    #[test]
    fn test_select_with_default() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);

        // Only one condition that matches some elements
        let cond = arr.map(|x| x > 3);
        let choice = arr.map(|x| x * 10);

        let result = select(&[cond], &[choice], -1).expect("operation should succeed");

        // Elements > 3 get multiplied by 10, others get -1
        assert_eq!(result.to_vec(), vec![-1, -1, -1, 40, 50]);
    }

    #[test]
    fn test_select_multiple_conditions() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);

        let cond1 = arr.map(|x| x == 1);
        let choice1 = Array::from_vec(vec![100, 100, 100, 100, 100]);

        let cond2 = arr.map(|x| x == 3);
        let choice2 = Array::from_vec(vec![300, 300, 300, 300, 300]);

        let cond3 = arr.map(|x| x == 5);
        let choice3 = Array::from_vec(vec![500, 500, 500, 500, 500]);

        let result = select(&[cond1, cond2, cond3], &[choice1, choice2, choice3], 0)
            .expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![100, 0, 300, 0, 500]);
    }

    #[test]
    fn test_select_2d() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);

        let cond1 = arr.map(|x| x < 3);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = arr.map(|x| x >= 3);
        let choice2 = arr.map(|x| x * 100);

        let result =
            select(&[cond1, cond2], &[choice1, choice2], 0).expect("operation should succeed");

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![10, 20, 300, 400]);
    }

    #[test]
    fn test_select_mismatched_lengths() {
        let arr = Array::from_vec(vec![1, 2, 3]);

        let cond1 = arr.map(|x| x < 2);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = arr.map(|x| x >= 2);
        let choice2 = arr.map(|x| x * 100);

        // 2 conditions but 3 choices
        let choice3 = arr.map(|x| x * 1000);

        let result = select(&[cond1, cond2], &[choice1, choice2, choice3], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_mismatched_shapes() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);

        let cond1 = arr.map(|x| x < 3);
        let choice1 = arr.map(|x| x * 10);

        let cond2 = Array::from_vec(vec![true, false]); // Wrong shape
        let choice2 = arr.map(|x| x * 100);

        let result = select(&[cond1, cond2], &[choice1, choice2], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_combined_indexing_take_and_boolean() {
        // First use take to reorder, then boolean mask
        let arr = Array::from_vec(vec![5, 2, 8, 1, 9, 3]);
        let indices = Array::from_vec(vec![0, 2, 4]); // [5, 8, 9]

        let reordered = take(&arr, &indices, None).expect("operation should succeed");
        let mask = reordered.map(|x| x > 7); // [false, true, true]

        let result = boolean_index(&reordered, &mask).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![8, 9]);
    }

    #[test]
    fn test_take_empty_indices() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let indices: Array<usize> = Array::from_vec(vec![]);

        let result = take(&arr, &indices, None).expect("operation should succeed");

        assert_eq!(result.size(), 0);
    }

    #[test]
    fn test_take_repeated_indices() {
        let arr = Array::from_vec(vec![10, 20, 30]);
        let indices = Array::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let result = take(&arr, &indices, None).expect("operation should succeed");

        assert_eq!(result.to_vec(), vec![10, 10, 20, 20, 30, 30]);
    }

    #[test]
    fn probe_take_none_perf_vs_naive_to_vec_pair() {
        // Old behavior: `array.to_vec()` (owned copy of the whole source
        // buffer) plus `indices.to_vec()` (owned copy of the index
        // buffer) as two separate passes (validate, then gather), each
        // walking a plain `&[_]` slice. Reproduced here verbatim
        // (including the validation pass `take`'s `None` arm has always
        // run before its gather loop) since omitting it would make this
        // "naive" baseline artificially cheaper than the real
        // pre-optimization `take` ever was and understate the actual
        // before/after delta. New behavior: `operand(array)` +
        // `operand(indices)` (both zero-copy when contiguous), fused into
        // *one* validate-and-gather pass -- see `take`'s doc comments
        // above for why the fused single pass matters, not just the
        // to_vec() removal.
        fn naive_take_none(array: &Array<f64>, indices: &Array<usize>) -> Vec<f64> {
            let flat = array.to_vec();
            let idx_vec = indices.to_vec();
            for &idx in &idx_vec {
                assert!(
                    idx < flat.len(),
                    "index {idx} out of bounds for flattened array of size {}",
                    flat.len()
                );
            }
            idx_vec.iter().map(|&idx| flat[idx]).collect()
        }

        let n = 100_000;
        let array = Array::from_vec((0..n).map(|i| i as f64).collect::<Vec<_>>());
        // Repeated-index pattern: cycle through a much smaller range so
        // most lookups repeat, matching the "gathered outputs" shape
        // this hot path is meant for.
        let indices = Array::from_vec((0..n).map(|i| i % 997).collect::<Vec<_>>());
        let iters = 50;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_take_none(&array, &indices));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ =
                std::hint::black_box(take(&array, &indices, None).expect("take should succeed"));
        }
        let operand = t1.elapsed();

        eprintln!(
            "[take(axis=None), n={n}] naive(to_vec_pair)={:.1}us/iter operand={:.1}us/iter ({:.2}x)",
            naive.as_secs_f64() * 1e6 / iters as f64,
            operand.as_secs_f64() * 1e6 / iters as f64,
            naive.as_secs_f64() / operand.as_secs_f64(),
        );

        assert_eq!(
            naive_take_none(&array, &indices),
            take(&array, &indices, None)
                .expect("take should succeed")
                .to_vec()
        );
    }
}
