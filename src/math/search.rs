//! Search and binning functions for NumRS2 arrays.
//!
//! This module provides functions for searching, binning, and interpolation:
//!
//! - [`bincount`] - Count occurrences of each value in array of non-negative ints
//! - [`digitize`] - Return indices of bins to which each value belongs
//! - [`searchsorted`] - Find indices where elements should be inserted to maintain order
//! - [`partition`] - Partially sort array so that kth element is in its final sorted position
//! - [`interp`] - One-dimensional linear interpolation
//! - [`median`] - Compute the median along the specified axis

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

/// Count occurrences of each value in array of non-negative ints
///
/// This is a thin, unweighted-`usize` specialization of the canonical
/// [`crate::array_ops::sorting::bincount`] (which additionally supports
/// non-`usize` inputs and an optional `weights` array); see that function's
/// docs for the full NumPy-compatible semantics, including the empty-input
/// edge case this delegates rather than reimplements.
///
/// # Parameters
///
/// * `array` - Input array of non-negative integers
/// * `minlength` - Minimum number of bins for output array
///
/// # Returns
///
/// Array where element i is the count of occurrences of i in the input array
pub fn bincount(array: &Array<usize>, minlength: Option<usize>) -> Result<Array<usize>> {
    crate::array_ops::sorting::bincount(array, None::<&Array<usize>>, minlength)
}

/// Return indices of bins to which each value belongs
///
/// Delegates to the canonical [`crate::array_ops::sorting::digitize`], which
/// additionally accepts monotonically *decreasing* bins (this copy
/// previously rejected them outright, and used a linear rather than binary
/// search); see that function's docs for full NumPy-compatible semantics.
///
/// # Parameters
///
/// * `array` - Input array
/// * `bins` - Array of bin edges (must be monotonically increasing or decreasing)
/// * `right` - If true, intervals include right edge; otherwise left edge
///
/// # Returns
///
/// Array of indices indicating which bin each value belongs to
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6]);
/// let bins = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0]);
/// let indices = digitize(&x, &bins, true).expect("digitize should succeed");
/// assert_eq!(indices.to_vec(), vec![1, 4, 3, 2]);
/// ```
pub fn digitize<T>(array: &Array<T>, bins: &Array<T>, right: bool) -> Result<Array<usize>>
where
    T: Float + Clone + PartialOrd,
{
    crate::array_ops::sorting::digitize(array, bins, right)
}

/// Find indices where elements should be inserted to maintain order
///
/// Delegates to the canonical [`crate::array_ops::sorting::searchsorted`],
/// which additionally accepts an optional `sorter` permutation (this copy's
/// `side` is a required `&str` rather than `Option<&str>` and there is no
/// way to reach `sorter` through this signature); see that function's docs
/// for full NumPy-compatible semantics, including the binary- rather than
/// linear-search implementation used underneath.
///
/// # Parameters
///
/// * `sorted_array` - Array that is already sorted
/// * `values` - Values to insert
/// * `side` - If 'left', gives leftmost position; if 'right', gives rightmost
///
/// # Returns
///
/// Array of insertion indices
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let v = Array::from_vec(vec![0.5, 2.0, 3.5, 6.0]);
/// let indices = searchsorted(&a, &v, "left").expect("searchsorted should succeed");
/// assert_eq!(indices.to_vec(), vec![0, 1, 3, 5]);
/// ```
pub fn searchsorted<T>(
    sorted_array: &Array<T>,
    values: &Array<T>,
    side: &str,
) -> Result<Array<usize>>
where
    T: Float + Clone + PartialOrd,
{
    if side != "left" && side != "right" {
        return Err(NumRs2Error::InvalidOperation(
            "side must be 'left' or 'right'".to_string(),
        ));
    }
    crate::array_ops::sorting::searchsorted(sorted_array, values, Some(side), None)
}

/// Partially sort array so that kth element is in its final sorted position
///
/// # Parameters
///
/// * `array` - Input array
/// * `kth` - Index of element to partition by
/// * `axis` - Axis along which to sort
/// * `kind` - Selection algorithm. Only `None` or `"introselect"` are accepted: this path
///   always uses an introselect-style selection (`slice::select_nth_unstable_by`), so there is
///   no separate algorithm to switch to. Any other value returns an error rather than being
///   silently accepted and ignored.
/// * `order` - Structured-array field names to partition by. Plain `Array<T>` has no named
///   fields, so passing `Some(_)` returns an error instead of silently ignoring it.
///
/// # Returns
///
/// Partially sorted array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let mut a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
/// let result = partition(&a, 2, None, None, None).expect("partition should succeed");
/// // Elements at indices 0,1 are <= element at index 2 <= elements at index 3
/// ```
pub fn partition<T>(
    array: &Array<T>,
    kth: usize,
    axis: Option<isize>,
    kind: Option<&str>,
    order: Option<&[&str]>,
) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if order.is_some() {
        return Err(NumRs2Error::NotImplemented(
            "partition: `order` (structured-array sort keys) is not supported for a plain \
             Array<T>; there are no named fields to partition by"
                .to_string(),
        ));
    }
    if !matches!(kind, None | Some("introselect")) {
        return Err(NumRs2Error::InvalidOperation(format!(
            "partition: kind must be `None` or \"introselect\" (got {:?}); this path always \
             uses an introselect-style selection",
            kind
        )));
    }

    let mut result = array.clone();

    if let Some(axis_val) = axis {
        // Partition along axis
        let ax = if axis_val < 0 {
            (array.ndim() as isize + axis_val) as usize
        } else {
            axis_val as usize
        };
        let shape = array.shape();
        let axis_len = shape[ax];

        if kth >= axis_len {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "kth ({}) out of bounds for axis {} of size {}",
                kth, ax, axis_len
            )));
        }

        // Calculate strides for iteration
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let total_size: usize = shape.iter().product();
        let n_slices = total_size / axis_len;

        // Bulk-acquire once for the whole axis-partition pass: `result` was
        // just cloned from `array` above, so this first mutable access is
        // the ONE unshare the COW buffer ever needs here (subsequent access
        // through this same handle is free); every read in the loop below
        // goes through the still-shared, untouched `array` (a distinct
        // `Array` handle from `result`), so nothing here can read stale data
        // through the mutable borrow.
        let result_arr = result.array_mut();

        for slice_idx in 0..n_slices {
            // Get indices for this slice along the axis
            let _indices: Vec<usize> = Vec::with_capacity(axis_len);
            let mut base_indices = vec![0; shape.len()];

            // Convert slice index to multi-dimensional indices
            let mut temp = slice_idx;
            for i in (0..shape.len()).rev() {
                if i != ax {
                    let stride = if i < ax {
                        strides[i] / strides[ax]
                    } else {
                        strides[i]
                    };
                    base_indices[i] = temp / stride;
                    temp %= stride;
                }
            }

            // Collect values along the axis
            let mut values = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                base_indices[ax] = i;
                let _flat_idx = base_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| idx * strides[i])
                    .sum::<usize>();
                let value = array.get(&base_indices)?;
                values.push(value);
            }

            // Partition the values
            values.select_nth_unstable_by(kth, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Write back partitioned values
            for i in 0..axis_len {
                base_indices[ax] = i;
                *result_arr.get_mut(base_indices.as_slice()).ok_or_else(|| {
                    NumRs2Error::IndexOutOfBounds(format!(
                        "Failed to set element at indices {:?}",
                        base_indices
                    ))
                })? = values[i];
            }
        }

        Ok(result)
    } else {
        // Partition flattened array
        let mut data_vec = result.to_vec();
        if kth >= data_vec.len() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "kth ({}) out of bounds for array of size {}",
                kth,
                data_vec.len()
            )));
        }

        data_vec.select_nth_unstable_by(kth, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Array::from_vec_shape(data_vec, &array.shape())?)
    }
}

/// One-dimensional linear interpolation
///
/// # Parameters
///
/// * `x` - The x-coordinates at which to evaluate the interpolated values
/// * `xp` - The x-coordinates of the data points, must be increasing
/// * `fp` - The y-coordinates of the data points
/// * `left` - Value to return for `x < xp[0]`, default is `fp[0]`
/// * `right` - Value to return for `x > xp[-1]`, default is `fp[-1]`
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let xp = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let fp = Array::from_vec(vec![3.0, 2.0, 0.0]);
/// let x = Array::from_vec(vec![0.0, 1.5, 2.72, 3.14]);
/// let y = interp(&x, &xp, &fp, None, None).expect("interp should succeed");
/// ```
pub fn interp<T>(
    x: &Array<T>,
    xp: &Array<T>,
    fp: &Array<T>,
    left: Option<T>,
    right: Option<T>,
) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let xp_vec = xp.to_vec();
    let fp_vec = fp.to_vec();

    if xp_vec.len() != fp_vec.len() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![xp_vec.len()],
            actual: vec![fp_vec.len()],
        });
    }

    if xp_vec.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "xp must have at least 1 point".to_string(),
        ));
    }

    // Verify xp is sorted
    for i in 1..xp_vec.len() {
        if xp_vec[i] <= xp_vec[i - 1] {
            return Err(NumRs2Error::InvalidOperation(
                "xp must be monotonically increasing".to_string(),
            ));
        }
    }

    let left_val = left.unwrap_or_else(|| fp_vec[0]);
    let right_val = right.unwrap_or_else(|| fp_vec[fp_vec.len() - 1]);

    let mut result = Vec::with_capacity(x.len());

    let x_vec = x.to_vec();
    for xi in x_vec.iter() {
        if xi < &xp_vec[0] {
            result.push(left_val);
        } else if xi > &xp_vec[xp_vec.len() - 1] {
            result.push(right_val);
        } else {
            // Binary search for the interval containing xi
            let mut lo = 0;
            let mut hi = xp_vec.len() - 1;

            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if xi < &xp_vec[mid] {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            // Linear interpolation
            let x0 = xp_vec[lo];
            let x1 = xp_vec[hi];
            let y0 = fp_vec[lo];
            let y1 = fp_vec[hi];

            let t = (*xi - x0) / (x1 - x0);
            let yi = y0 + t * (y1 - y0);
            result.push(yi);
        }
    }

    Array::from_vec_shape(result, &x.shape())
}

/// Compute the median along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the median is computed. If None, compute median of flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the median values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let m = median(&a, None, false).expect("median should succeed");
/// assert_eq!(m.to_vec(), vec![3.0]); // median is 3.0
/// ```
pub fn median<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if array.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute median of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Compute median of flattened array
            let mut data = array.to_vec();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = data.len();
            let median_val = if n.is_multiple_of(2) {
                // Even number of elements - average of two middle values
                (data[n / 2 - 1] + data[n / 2]) / T::from(2.0).expect("2.0 should be representable")
            } else {
                // Odd number of elements - middle value
                data[n / 2]
            };

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec_shape(vec![median_val], &shape)?)
            } else {
                Ok(Array::from_vec(vec![median_val]))
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

                // Collect values along the axis
                let mut values = Vec::with_capacity(axis_size);
                for j in 0..axis_size {
                    indices[axis] = j;
                    values.push(array.get(&indices)?);
                }

                // Sort and find median
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_val = if axis_size.is_multiple_of(2) {
                    (values[axis_size / 2 - 1] + values[axis_size / 2])
                        / T::from(2.0).expect("2.0 should be representable")
                } else {
                    values[axis_size / 2]
                };

                result_data[out_idx] = median_val;
            }

            Ok(Array::from_vec_shape(result_data, &out_shape)?)
        }
    }
}
