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
use crate::kernels::{borrow::operand, cast, reduce};
use num_traits::{Float, Zero};
use std::ops::{Add, Mul};

// Import cumsum and cumprod from parent module
use super::{cumprod, cumsum};

/// `true` when `x` is unordered with respect to itself -- the only thing `PartialOrd` alone
/// lets a generic reduction say about a `NaN`-like value.
///
/// For `f64`/`f32` this is exactly `is_nan()` (IEEE-754 makes every comparison involving a
/// `NaN` return `false`, so `partial_cmp` yields `None`). For a totally ordered `T` -- every
/// integer type, and any sensibly implemented user type -- it is always `false`, so the
/// `NaN`-propagation branch it guards costs one comparison per element and never fires.
///
/// Written as `partial_cmp(..).is_none()` rather than `x != x` deliberately: the latter is
/// the same test but trips `clippy::eq_op`.
fn is_unordered<T: PartialOrd>(x: &T) -> bool {
    x.partial_cmp(x).is_none()
}

/// `NaN`-propagating maximum over a non-empty flat slice, for element types that
/// [`crate::kernels::cast`] cannot reinterpret as `f64`/`f32`.
///
/// Returns the offending element itself when the input contains one that is unordered with
/// itself, so the returned `NaN` keeps its original payload; the `f64`/`f32` kernels return a
/// canonical `NAN` instead. Both satisfy `is_nan()`, which is the only property callers (and
/// tests) should depend on -- never a bit pattern.
fn flat_max<T: PartialOrd + Clone>(data: &[T]) -> T {
    let mut acc = data[0].clone();
    for x in data.iter() {
        if is_unordered(x) {
            return x.clone();
        }
        if *x > acc {
            acc = x.clone();
        }
    }
    acc
}

/// `NaN`-propagating minimum over a non-empty flat slice; see [`flat_max`].
fn flat_min<T: PartialOrd + Clone>(data: &[T]) -> T {
    let mut acc = data[0].clone();
    for x in data.iter() {
        if is_unordered(x) {
            return x.clone();
        }
        if *x < acc {
            acc = x.clone();
        }
    }
    acc
}

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
    T: PartialOrd + Clone + Zero + 'static,
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
    T: PartialOrd + Clone + Zero + 'static,
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
///
/// # `NaN` propagates, matching NumPy
///
/// Any `NaN` in the reduced data -- flat, or within a reduced lane for the `Some(axis)` form --
/// makes the corresponding output `NaN`, exactly as `np.max` does. Use
/// [`crate::math::nanmax`] for the `NaN`-ignoring variant.
///
/// This replaced an earlier "poisons only if the very first element scanned is `NaN`" fold,
/// which silently dropped interior `NaN`s: the comparison `x > max` is false whenever either
/// side is `NaN`, so a `NaN` never overwrote an already-finite accumulator. That is not NumPy's
/// rule and it disagreed with `stats::basic::Statistics::max` one call away; there is now one
/// `max`/`min` `NaN` convention across the crate.
///
/// # Dispatch
///
/// For `axis = None` and `T == f64`/`f32`, this routes through
/// `kernels::reduce::max_f64`/`max_f32` (via `kernels::borrow::operand` for a
/// zero-copy-when-possible flat slice and `kernels::cast` to reinterpret it), which is
/// what the `+ 'static` bound is for. Those kernels are built from plain comparisons; they do
/// **not** call `scirs2_core::simd_ops::SimdUnifiedOps::simd_max_element`, which was found to
/// return a wrong, *finite* value (a real maximum silently discarded, neither the extremum nor
/// `NaN`) for some `NaN` placements -- see `kernels::reduce`'s module docs. Every other `T`,
/// and every `Some(axis)` reduction, uses a comparison fold with the same propagate rule.
pub fn max<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero + 'static,
{
    if array.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot find max of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find max of flattened array. Zero-copy via `kernels::borrow::operand` instead
            // of `Array::to_vec`'s unconditional copy, then dtype-dispatched onto
            // `kernels::reduce` for f64/f32 -- see this function's doc comment.
            let op = operand(array);
            let max_val = if let Some(s) = cast::as_f64(&op) {
                cast::f64_to::<T>(reduce::max_f64(s)).expect("T == f64 per cast::as_f64 match")
            } else if let Some(s) = cast::as_f32(&op) {
                cast::f32_to::<T>(reduce::max_f32(s)).expect("T == f32 per cast::as_f32 match")
            } else {
                flat_max(&op)
            };

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec_shape(vec![max_val], &shape)?)
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

                // Find max along the axis. `NaN` propagates (see this function's doc
                // comment): the comparison below cannot carry it -- `val > mv` is false
                // whenever either side is `NaN` -- so an unordered element short-circuits
                // the lane and becomes its result directly.
                let mut max_val = None;

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;

                    if is_unordered(&val) {
                        max_val = Some(val);
                        break;
                    }
                    if max_val.as_ref().is_none_or(|mv| &val > mv) {
                        max_val = Some(val);
                    }
                }

                result_data[out_idx] = max_val.expect("max_val should be set when axis_size > 0");
            }

            Ok(Array::from_vec_shape(result_data, &out_shape)?)
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
///
/// # `NaN` propagates, matching NumPy
///
/// See [`max`]: any `NaN` in the reduced data (or reduced lane) makes the output `NaN`, as
/// `np.min` does, and for `axis = None` with `T == f64`/`f32` this dispatches through
/// `kernels::reduce::min_f64`/`min_f32`. [`crate::math::nanmin`] is the
/// `NaN`-ignoring variant.
pub fn min<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero + 'static,
{
    if array.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot find min of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find min of flattened array; see `max`'s matching branch.
            let op = operand(array);
            let min_val = if let Some(s) = cast::as_f64(&op) {
                cast::f64_to::<T>(reduce::min_f64(s)).expect("T == f64 per cast::as_f64 match")
            } else if let Some(s) = cast::as_f32(&op) {
                cast::f32_to::<T>(reduce::min_f32(s)).expect("T == f32 per cast::as_f32 match")
            } else {
                flat_min(&op)
            };

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec_shape(vec![min_val], &shape)?)
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

                // Find min along the axis; `NaN` propagates -- see `max`'s matching loop.
                let mut min_val = None;

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;

                    if is_unordered(&val) {
                        min_val = Some(val);
                        break;
                    }
                    if min_val.as_ref().is_none_or(|mv| &val < mv) {
                        min_val = Some(val);
                    }
                }

                result_data[out_idx] = min_val.expect("min_val should be set when axis_size > 0");
            }

            Ok(Array::from_vec_shape(result_data, &out_shape)?)
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
    T: Float + Clone + Add<Output = T> + Zero + 'static,
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
            // Sum of flattened array. Dispatches through `kernels::reduce::sum_f64`/`sum_f32`
            // (zero-copy via `kernels::borrow::operand`) when `T` concretely is `f64`/`f32`;
            // any other `T` keeps the original fold, sourced from `operand()` instead of
            // `to_vec()`.
            let op = operand(array);
            let sum_val = if let Some(s) = cast::as_f64(&op) {
                cast::f64_to(reduce::sum_f64(s)).expect("T == f64 per cast::as_f64 match")
            } else if let Some(s) = cast::as_f32(&op) {
                cast::f32_to(reduce::sum_f32(s)).expect("T == f32 per cast::as_f32 match")
            } else {
                op.iter().fold(T::zero(), |acc, x| acc + *x)
            };

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec_shape(vec![sum_val], &shape)?)
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

            // Row-major (C-contiguous) strides for `shape`. `kernels::borrow::operand` below
            // yields data in the array's *logical* order for its current shape regardless of
            // physical layout (a contiguous array is borrowed as-is; a non-contiguous view is
            // materialized by walking `.iter()`, which walks in logical order -- see that
            // function's own docs), so a flat index built from these strides against `shape`
            // always names the same logical element in `op` below that `array.get(&indices)`
            // would have named.
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            // Only `indices[axis]` ever varies across the inner `j` loop below (every other
            // coordinate is fixed once per `out_idx`), so the axis walk can advance the flat
            // index by this constant per step instead of recomputing the full
            // `indices[i] * strides[i]` dot product on every step -- same hoist already used by
            // `cumsum_no_out`/`cumprod_no_out` in `super::statistics` for their axis branches.
            let axis_stride = strides[axis];

            // Zero-copy (when contiguous) borrow of the whole array, taken once, instead of
            // calling `array.get(&indices)?` for every one of `out_size * axis_size` elements.
            // `Array::get` re-derives `self.shape()` -- a fresh `Vec` allocation -- inside its
            // own per-dimension bounds check on *every* call, so the old per-element path cost
            // O(ndim) allocations per element, O(ndim * out_size * axis_size) overall.
            let op = operand(array);

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

                // Flat index of the axis=0 element for this output position (`indices[axis]`
                // is still its initial 0 here -- the loop above never touches it).
                let mut flat_idx: usize = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| idx * strides[i])
                    .sum();

                // Compute sum along the axis: identical left-to-right, `T::zero()`-seeded fold
                // order as before, now reading `op[flat_idx]` and stepping by `axis_stride`
                // instead of calling `array.get(&indices)?` per element.
                let mut sum = T::zero();
                for _ in 0..axis_size {
                    sum = sum + op[flat_idx];
                    flat_idx += axis_stride;
                }

                result_data[out_idx] = sum;
            }

            Ok(Array::from_vec_shape(result_data, &out_shape)?)
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

            Ok(Array::from_vec_shape(result_data, &shape)?)
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

    Array::from_vec_shape(result_data, &shape)
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

#[cfg(test)]
mod sum_max_min_tests {
    use super::*;

    /// Regression test for the `Some(axis)` stride hoist: pins the same small 2-D case the
    /// module docstring's example uses, against a hand-computed expectation sharing no code
    /// with `sum`'s implementation.
    #[test]
    fn sum_axis_matches_naive_2d_small() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let axis0 = sum(&a, Some(0), false).expect("sum should succeed");
        assert_eq!(axis0.to_vec(), vec![5.0, 7.0, 9.0]);
        let axis1 = sum(&a, Some(1), false).expect("sum should succeed");
        assert_eq!(axis1.to_vec(), vec![6.0, 15.0]);
    }

    /// Discriminating check for the `Some(axis)` stride hoist: `strides` there is computed as
    /// C-contiguous row-major from `shape`, while `kernels::borrow::operand` returns an
    /// `Operand::Owned` (materialized in *logical*, not physical, order -- see that function's
    /// own docs) for a non-contiguous view such as `transpose_axis`'s output. This confirms the
    /// two still agree: the flat-index walk over `op` must name the same logical element that
    /// `array.get(&indices)` (logical indexing) named before the hoist.
    #[test]
    fn sum_axis_matches_naive_on_non_contiguous_transposed_view() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let t = a.transpose_axis(0, 1); // shape [3, 2], logically [[1,4],[2,5],[3,6]]
        assert!(!t.is_c_contiguous());

        let axis0 = sum(&t, Some(0), false).expect("sum should succeed");
        // Column sums of [[1,4],[2,5],[3,6]]: [1+2+3, 4+5+6].
        assert_eq!(axis0.to_vec(), vec![6.0, 15.0]);

        let axis1 = sum(&t, Some(1), false).expect("sum should succeed");
        // Row sums: [1+4, 2+5, 3+6].
        assert_eq!(axis1.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    /// Exercises the axis hoist on a shape large enough to be a meaningful timing case (see
    /// this lane's report for the measured before/after numbers), still checked against an
    /// independent naive per-row computation.
    #[test]
    fn sum_axis_matches_naive_larger_2d() {
        let rows = 64usize;
        let cols = 4096usize;
        let data: Vec<f64> = (0..rows * cols).map(|i| (i % 17) as f64 - 8.0).collect();
        let a = Array::from_vec(data.clone()).reshape(&[rows, cols]);

        let got = sum(&a, Some(1), false)
            .expect("sum should succeed")
            .to_vec();
        let expected: Vec<f64> = (0..rows)
            .map(|r| (0..cols).map(|c| data[r * cols + c]).sum())
            .collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }

    #[test]
    fn sum_matches_naive_across_dispatch_tiers() {
        for &n in &[1usize, 10, 100, 20_000] {
            let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 3.0).collect();
            let naive: f64 = data.iter().sum();
            let arr = Array::from_vec(data);
            let got = sum(&arr, None, false).expect("sum should succeed").to_vec()[0];
            assert!(
                (got - naive).abs() / naive.abs().max(1.0) < 1e-9,
                "n={n}: got {got}, naive {naive}"
            );
        }
    }

    #[test]
    fn sum_matches_naive_for_f32() {
        let data: Vec<f32> = (0..500).map(|i| i as f32 * 0.25).collect();
        let naive: f32 = data.iter().sum();
        let arr = Array::from_vec(data);
        let got = sum(&arr, None, false).expect("sum should succeed").to_vec()[0];
        assert!((got - naive).abs() < 1e-1, "got {got}, naive {naive}");
    }

    #[test]
    fn sum_empty_array_is_zero() {
        let arr: Array<f64> = Array::from_vec(vec![]);
        assert_eq!(
            sum(&arr, None, false).expect("sum should succeed").to_vec(),
            vec![0.0]
        );
    }

    #[test]
    fn max_min_no_nan_match_naive_across_dispatch_tiers() {
        for &n in &[1usize, 10, 100, 20_000] {
            let data: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 - 500.0).collect();
            let naive_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let naive_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let arr = Array::from_vec(data);
            let got_min = min(&arr, None, false).expect("min should succeed").to_vec()[0];
            let got_max = max(&arr, None, false).expect("max should succeed").to_vec()[0];
            assert_eq!(got_min, naive_min, "n={n}");
            assert_eq!(got_max, naive_max, "n={n}");
        }
    }

    /// NumPy reference behavior, replacing
    /// `max_min_nan_behavior_is_the_original_fold_not_kernel_dispatch`: `max`/`min` now
    /// dispatch onto `kernels::reduce` for `f64`/`f32` and propagate `NaN` everywhere, so
    /// interior `NaN`s are no longer silently ignored. `np.max([1.0, nan, 5.0])` is `nan`;
    /// `np.nanmax` of the same is `5.0`.
    #[test]
    fn max_min_propagate_nan_like_numpy_and_nanmax_still_ignores_it() {
        let a = Array::from_vec(vec![1.0f64, f64::NAN, 5.0]);
        assert!(
            max(&a, None, false).expect("max should succeed").to_vec()[0].is_nan(),
            "np.max([1.0, nan, 5.0]) is nan"
        );
        assert!(
            min(&a, None, false).expect("min should succeed").to_vec()[0].is_nan(),
            "np.min([1.0, nan, 5.0]) is nan"
        );

        // The NaN-ignoring counterpart is unaffected: np.nanmax([1.0, nan, 5.0]) == 5.0.
        assert_eq!(
            crate::math::nanmax(&a, None)
                .expect("nanmax should succeed")
                .to_vec()[0],
            5.0
        );
        assert_eq!(
            crate::math::nanmin(&a, None)
                .expect("nanmin should succeed")
                .to_vec()[0],
            1.0
        );
    }

    /// Every `NaN` position propagates -- first, last, and interior -- on both the flat
    /// dispatch tier and the `Some(axis)` fold.
    #[test]
    fn max_min_propagate_nan_from_every_position() {
        for data in [
            vec![f64::NAN, 1.0, 2.0],
            vec![1.0, 2.0, f64::NAN],
            vec![1.0, f64::NAN, 2.0],
        ] {
            let arr = Array::from_vec(data.clone());
            assert!(
                min(&arr, None, false).expect("min should succeed").to_vec()[0].is_nan(),
                "{data:?}"
            );
            assert!(
                max(&arr, None, false).expect("max should succeed").to_vec()[0].is_nan(),
                "{data:?}"
            );
        }

        // The vector that exposed the upstream `simd_max_element` wrong-finite-value defect
        // (true maximum 5.0 at index 0, one NaN at index 10, len 64). The old fold returned
        // 5.0 here and the old kernel returned a wrong 1.0; NumPy returns nan, and so does
        // this now.
        let mut c_data = vec![1.0f64; 64];
        c_data[0] = 5.0;
        c_data[10] = f64::NAN;
        let c = Array::from_vec(c_data);
        assert!(max(&c, None, false).expect("max should succeed").to_vec()[0].is_nan());

        // Per-lane propagation on the `Some(axis)` path: [[1, NaN, 3], [4, 5, 6]] along
        // axis 1 -- only the first lane goes NaN.
        let d = Array::from_vec(vec![1.0f64, f64::NAN, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let maxs = max(&d, Some(1), false)
            .expect("max should succeed")
            .to_vec();
        let mins = min(&d, Some(1), false)
            .expect("min should succeed")
            .to_vec();
        assert!(maxs[0].is_nan());
        assert!(mins[0].is_nan());
        assert_eq!(maxs[1], 6.0);
        assert_eq!(mins[1], 4.0);
    }

    /// `f32` takes a different `kernels::cast` branch than `f64`; same rule.
    #[test]
    fn max_min_propagate_nan_for_f32() {
        let a = Array::from_vec(vec![1.0f32, f32::NAN, 5.0]);
        assert!(max(&a, None, false).expect("max should succeed").to_vec()[0].is_nan());
        assert!(min(&a, None, false).expect("min should succeed").to_vec()[0].is_nan());
    }

    #[test]
    fn max_min_fold_matches_naive_for_integers() {
        let arr = Array::from_vec(vec![3_i64, -7, 5, 0, -2]);
        assert_eq!(
            min(&arr, None, false).expect("min should succeed").to_vec()[0],
            -7
        );
        assert_eq!(
            max(&arr, None, false).expect("max should succeed").to_vec()[0],
            5
        );
    }

    #[test]
    fn max_min_empty_array_errors() {
        let arr: Array<f64> = Array::from_vec(vec![]);
        assert!(max(&arr, None, false).is_err());
        assert!(min(&arr, None, false).is_err());
    }
}
