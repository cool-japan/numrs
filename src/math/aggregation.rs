//! Aggregation functions for array operations.
//!
//! This module provides functions for aggregating array elements along axes:
//!
//! - `amax`, `amin` - Axis-aware maximum/minimum
//! - `max`, `min` - NumPy-style maximum/minimum
//! - `sum` - Sum with axis support
//! - `sort` - Sort along axis
//! - `argpartition` - Indirect partition along axis
//! - `round` - Round to nearest integer
//! - `cumulative_sum`, `cumulative_prod` - Cumulative operations

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};
use std::ops::{Add, Mul};

// Import cumsum and cumprod from parent module
use super::{cumprod, cumsum};

/// Array maximum along a given axis (alias for max)
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find maximum values. If None, the maximum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing maximum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0]).reshape(&[2, 3]);
/// let maxs = amax(&a, Some(1), false).expect("amax should succeed");
/// assert_eq!(maxs.to_vec(), vec![3.0, 5.0]); // max of each row
/// ```
pub fn amax<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    max(array, axis, keepdims)
}

/// Array minimum along a given axis (alias for min)
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find minimum values. If None, the minimum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing minimum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![5.0, 3.0, 2.0, 4.0, 1.0, 6.0]).reshape(&[2, 3]);
/// let mins = amin(&a, Some(1), false).expect("amin should succeed");
/// assert_eq!(mins.to_vec(), vec![2.0, 1.0]); // min of each row
/// ```
pub fn amin<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    min(array, axis, keepdims)
}

/// Find the maximum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find maximum values. If None, the maximum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing maximum values
pub fn max<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot find max of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find max of flattened array
            let data = array.to_vec();
            let max_val =
                data.iter().skip(1).fold(
                    data[0].clone(),
                    |max, x| {
                        if x > &max {
                            x.clone()
                        } else {
                            max
                        }
                    },
                );

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![max_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![max_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find max along the axis
                let mut max_val = None;

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;

                    if max_val.as_ref().is_none_or(|mv| &val > mv) {
                        max_val = Some(val);
                    }
                }

                result_data[out_idx] = max_val.expect("max_val should be set when axis_size > 0");
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Find the minimum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find minimum values. If None, the minimum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing minimum values
pub fn min<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot find min of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find min of flattened array
            let data = array.to_vec();
            let min_val =
                data.iter().skip(1).fold(
                    data[0].clone(),
                    |min, x| {
                        if x < &min {
                            x.clone()
                        } else {
                            min
                        }
                    },
                );

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![min_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![min_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find min along the axis
                let mut min_val = None;

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;

                    if min_val.as_ref().is_none_or(|mv| &val < mv) {
                        min_val = Some(val);
                    }
                }

                result_data[out_idx] = min_val.expect("min_val should be set when axis_size > 0");
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Sum of array elements over a given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to sum. If None, sum over flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing sum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
/// let sums = sum(&a, Some(1), false).expect("sum should succeed");
/// assert_eq!(sums.to_vec(), vec![6.0, 15.0]); // sum of each row
/// ```
pub fn sum<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Zero,
{
    if array.is_empty() {
        return Ok(if keepdims {
            let shape = if axis.is_none() {
                vec![1; array.ndim()]
            } else {
                let mut shape = array.shape();
                let ax = if let Some(a) = axis {
                    if a < 0 {
                        (array.ndim() as isize + a) as usize
                    } else {
                        a as usize
                    }
                } else {
                    0
                };
                if ax < shape.len() {
                    shape[ax] = 1;
                }
                shape
            };
            Array::zeros(&shape)
        } else {
            Array::zeros(&[1])
        });
    }

    match axis {
        None => {
            // Sum of flattened array
            let data = array.to_vec();
            let sum_val = data.iter().fold(T::zero(), |acc, x| acc + *x);

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![sum_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![sum_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Compute sum along the axis
                let mut sum = T::zero();
                for j in 0..axis_size {
                    indices[axis] = j;
                    sum = sum + array.get(&indices)?;
                }

                result_data[out_idx] = sum;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Sort an array along the given axis
///
/// # Parameters
///
/// * `array` - Array to be sorted
/// * `axis` - Axis along which to sort. If None, the array is flattened before sorting
/// * `kind` - Sorting algorithm. `None`, `"quicksort"`, and `"heapsort"` select an unstable
///   sort; `"mergesort"` and `"stable"` select a stable sort (ties keep their original relative
///   order). The exact underlying algorithm is unspecified beyond that stability guarantee.
///   Any other value returns an error.
/// * `order` - Structured-array field names to sort by. Plain `Array<T>` has no named fields,
///   so passing `Some(_)` returns an error instead of silently ignoring it.
///
/// # Returns
///
/// Sorted array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]).reshape(&[2, 3]);
/// let sorted = sort(&a, Some(1), None, None).expect("sort should succeed");
/// assert_eq!(sorted.get(&[0, 0]).expect("valid index"), 1.0);
/// assert_eq!(sorted.get(&[0, 1]).expect("valid index"), 3.0);
/// assert_eq!(sorted.get(&[0, 2]).expect("valid index"), 4.0);
/// ```
pub fn sort<T>(
    array: &Array<T>,
    axis: Option<isize>,
    kind: Option<&str>,
    order: Option<&[&str]>,
) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    if order.is_some() {
        return Err(NumRs2Error::NotImplemented(
            "sort: `order` (structured-array sort keys) is not supported for a plain Array<T>; \
             there are no named fields to sort by"
                .to_string(),
        ));
    }
    let stable = match kind {
        None | Some("quicksort") | Some("heapsort") => false,
        Some("mergesort") | Some("stable") => true,
        Some(other) => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "sort: kind must be one of 'quicksort', 'heapsort', 'mergesort', 'stable' \
                 (got {:?})",
                other
            )));
        }
    };

    if array.is_empty() {
        return Ok(array.clone());
    }

    match axis {
        None => {
            // Sort flattened array
            let mut data = array.to_vec();
            if stable {
                data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            Ok(Array::from_vec(data))
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];
            let total_size: usize = shape.iter().product();
            let mut result_data = vec![T::zero(); total_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Number of sorts to perform
            let n_sorts = total_size / axis_size;

            for sort_idx in 0..n_sorts {
                // Collect values along the axis for this position
                let mut values: Vec<T> = Vec::with_capacity(axis_size);

                // Determine the base indices for this sort
                let mut base_indices = vec![0; array.ndim()];
                let mut temp = sort_idx;

                for i in 0..array.ndim() {
                    if i != axis {
                        let size = shape[i];
                        base_indices[i] = temp % size;
                        temp /= size;
                    }
                }

                // Collect values along the axis
                for j in 0..axis_size {
                    base_indices[axis] = j;
                    values.push(array.get(&base_indices)?);
                }

                // Sort values
                if stable {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    values.sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                // Place sorted values in result
                for (k, val) in values.into_iter().enumerate() {
                    base_indices[axis] = k;
                    let flat_idx = base_indices
                        .iter()
                        .enumerate()
                        .map(|(i, &idx)| idx * strides[i])
                        .sum::<usize>();
                    result_data[flat_idx] = val;
                }
            }

            Ok(Array::from_vec(result_data).reshape(&shape))
        }
    }
}

/// Perform an indirect partition along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `kth` - Element index to partition by. The element at this index will be in its final sorted position
/// * `axis` - Axis along which to sort. If None, the array is flattened
/// * `kind` - Selection algorithm. Only `None` or `"introselect"` are accepted: this path
///   always uses an introselect-style selection (`slice::select_nth_unstable_by`), so there is
///   no separate algorithm to switch to. Any other value returns an error rather than being
///   silently accepted and ignored.
/// * `order` - Structured-array field names to partition by. Plain `Array<T>` has no named
///   fields, so passing `Some(_)` returns an error instead of silently ignoring it.
///
/// # Returns
///
/// Array of indices that partition the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
/// let indices = argpartition(&a, 2, None, None, None).expect("argpartition should succeed");
/// // After partitioning: values at indices[0] and indices[1] are <= value at indices[2]
/// // and value at indices[3] >= value at indices[2]
/// ```
pub fn argpartition<T>(
    array: &Array<T>,
    kth: usize,
    axis: Option<isize>,
    kind: Option<&str>,
    order: Option<&[&str]>,
) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    if order.is_some() {
        return Err(NumRs2Error::NotImplemented(
            "argpartition: `order` (structured-array sort keys) is not supported for a plain \
             Array<T>; there are no named fields to partition by"
                .to_string(),
        ));
    }
    if !matches!(kind, None | Some("introselect")) {
        return Err(NumRs2Error::InvalidOperation(format!(
            "argpartition: kind must be `None` or \"introselect\" (got {:?}); this path always \
             uses an introselect-style selection",
            kind
        )));
    }

    let axis = if let Some(ax) = axis {
        if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        }
    } else {
        // If axis is None, flatten the array
        let data = array.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();

        if kth >= data.len() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "kth ({}) out of bounds for array of size {}",
                kth,
                data.len()
            )));
        }

        // Partition the indices
        indices.select_nth_unstable_by(kth, |&a, &b| {
            data[a]
                .partial_cmp(&data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        return Ok(Array::from_vec(indices));
    };

    if axis >= array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    let shape = array.shape();
    let axis_size = shape[axis];

    if kth >= axis_size {
        return Err(NumRs2Error::InvalidOperation(format!(
            "kth ({}) out of bounds for axis {} of size {}",
            kth, axis, axis_size
        )));
    }

    // The output has the same shape as input
    let total_size: usize = shape.iter().product();
    let mut result_data = vec![0_usize; total_size];

    // Calculate strides
    let mut strides = vec![1; array.ndim()];
    for i in (0..array.ndim() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Number of partitions to perform
    let n_partitions = total_size / axis_size;

    for part_idx in 0..n_partitions {
        // Collect values along the axis for this position
        let mut values_with_indices: Vec<(T, usize)> = Vec::with_capacity(axis_size);

        // Determine the base indices for this partition
        let mut base_indices = vec![0; array.ndim()];
        let mut temp = part_idx;

        for i in 0..array.ndim() {
            if i != axis {
                let size = shape[i];
                base_indices[i] = temp % size;
                temp /= size;
            }
        }

        // Collect values along the axis
        for j in 0..axis_size {
            base_indices[axis] = j;
            let val = array.get(&base_indices)?;
            values_with_indices.push((val, j));
        }

        // Create indices array
        let mut indices: Vec<usize> = (0..axis_size).collect();

        // Partition by kth element
        indices.select_nth_unstable_by(kth, |&a, &b| {
            values_with_indices[a]
                .0
                .partial_cmp(&values_with_indices[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Place partitioned indices in result
        for (k, &idx) in indices.iter().enumerate() {
            base_indices[axis] = k;
            let flat_idx = base_indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| idx * strides[i])
                .sum::<usize>();
            result_data[flat_idx] = values_with_indices[idx].1;
        }
    }

    Ok(Array::from_vec(result_data).reshape(&shape))
}

/// Round array elements to the nearest integer
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with rounded values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::round;
///
/// let a = Array::from_vec(vec![1.5, 2.3, 3.7, 4.5]);
/// let rounded = round(&a).expect("round failed");
/// assert_eq!(rounded.to_vec(), vec![2.0, 2.0, 4.0, 5.0]);
/// ```
pub fn round<T>(array: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    Ok(array.map(|x| x.round()))
}

/// Alias for cumsum - Return the cumulative sum of array elements.
///
/// `out`, when provided, is honored exactly as in [`cumsum`]: its shape must match the
/// result's shape, the result is written into it, and the same array is returned.
pub fn cumulative_sum<T>(
    array: &Array<T>,
    axis: Option<isize>,
    out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Send + Sync + 'static,
{
    cumsum(array, axis, out)
}

/// Alias for cumprod - Return the cumulative product of array elements.
///
/// `out`, when provided, is honored exactly as in [`cumprod`]: its shape must match the
/// result's shape, the result is written into it, and the same array is returned.
pub fn cumulative_prod<T>(
    array: &Array<T>,
    axis: Option<isize>,
    out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T> + Send + Sync,
{
    cumprod(array, axis, out)
}
