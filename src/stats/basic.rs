//! Basic statistical functions
//!
//! This module provides basic statistical operations including:
//! - Statistics trait with mean, var, std, min, max, percentile methods
//! - Peak-to-peak (ptp) function
//! - Axis-based min/max functions
//! - Weighted average function

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels::{borrow::operand, cast, reduce};
use num_traits::{Float, NumCast, Zero};
use scirs2_core::parallel_ops::*;

use super::quantile::quantile;

/// Threshold for using parallel processing (minimum array size)
pub const PARALLEL_THRESHOLD: usize = 10000;

// Statistical functions
pub trait Statistics<T> {
    fn mean(&self) -> T;
    fn var(&self) -> T;
    fn std(&self) -> T;
    fn min(&self) -> T;
    fn max(&self) -> T;
    fn percentile(&self, q: T) -> T;
}

/// `mean`/`var`/`std`/`min`/`max` dispatch through `crate::kernels::reduce` (via
/// `crate::kernels::borrow::operand` for a zero-copy-when-possible slice, and
/// `crate::kernels::cast` to reinterpret it as `&[f64]`/`&[f32]` when `T` concretely is
/// one of those) for any array length, replacing the old scheme where only `T == f64` and
/// only `len() >= 64` got a SIMD fast path (`len() < 64`, and every `f32` array regardless
/// of length, silently fell back to a plain sequential/parallel fold). Any other `T` keeps
/// exactly that original fold, unchanged.
///
/// # `var`/`std`: population, not sample -- and never `simd_variance`/`simd_std`
///
/// This trait's `var`/`std` have no `ddof` parameter, so they have always computed
/// *population* variance/stddev (divisor `n`, matching NumPy's `ddof=0` default -- see
/// `tests/numpy_compatibility_validation.rs`'s `test_statistical_functions_numpy_equivalence`,
/// which pins `Statistics::var`/`std` against `np.var(x, ddof=0)`/`np.std(x, ddof=0)` at
/// `n == 10`). The previous `len() >= 64 && T == f64` branch called
/// `scirs2_core::simd_ops::SimdUnifiedOps::{simd_variance, simd_std}`, which hardcode
/// *sample* variance/stddev (divisor `n - 1`) -- silently wrong for any f64 array of at
/// least 64 elements, and simply undetected by the `n == 10` regression test above since
/// `10 < 64` never reached that branch. Fixed here by building variance as
/// `sum_sq_dev / n` -- via the fused `reduce::var_f64`/`reduce::var_f32` kernels, which
/// take `ddof` explicitly -- per `reduce`'s own module docs ("never use them"):
/// `simd_variance`/`simd_std` must not be called from here or anywhere else in this crate.
///
/// # `min`/`max`: `NaN` propagates, matching NumPy
///
/// `reduce::min_f64`/`reduce::max_f64` (and the `f32` twins) implement NumPy's
/// `np.min`/`np.max` rule: **any `NaN` anywhere in the array makes the result `NaN`**,
/// whatever the dtype, length or `NaN` position. The generic-`T` tail below implements the
/// same rule, so this trait has exactly one `min`/`max` `NaN` convention -- as does the rest
/// of the crate (`math::aggregation::max`/`min`, [`ptp`], `Array::min_optimized`/
/// `max_optimized`). The `NaN`-*ignoring* counterparts are `math::nanmin`/`math::nanmax`.
///
/// This replaced two earlier conventions, both now gone: the old plain fold's "poisoned only
/// if the very first element is `NaN`", and the `SimdUnifiedOps::simd_min_element`/
/// `simd_max_element` wrapper's placement- and length-dependent behavior. The latter was not
/// merely an unusual convention -- it was found to return a **wrong, finite value** (the true
/// maximum silently dropped, neither the extremum nor `NaN`) for some `NaN` placements, which
/// is why `kernels::reduce` no longer calls it at all. See `reduce`'s module docs for the
/// full finding and `simd_max_element_upstream_wrong_value_is_a_live_bug_not_just_new_nan_convention`
/// below, which calls the upstream function directly and pins the bad value as a tripwire for
/// an upstream fix. That test watches `scirs2-core`, not this crate; nothing in `numrs2`
/// depends on the value it pins any more.
impl<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync + 'static> Statistics<T>
    for Array<T>
{
    fn mean(&self) -> T {
        let op = operand(self);
        if let Some(s) = cast::as_f64(&op) {
            return cast::f64_to(reduce::mean_f64(s)).expect("T == f64 per cast::as_f64 match");
        }
        if let Some(s) = cast::as_f32(&op) {
            return cast::f32_to(reduce::mean_f32(s)).expect("T == f32 per cast::as_f32 match");
        }
        if op.is_empty() {
            return T::zero();
        }

        let sum = if op.len() >= PARALLEL_THRESHOLD {
            // Use parallel processing for large arrays
            op.par_iter()
                .map(|&x| x)
                .reduce(|| T::zero(), |acc, x| acc + x)
        } else {
            // Use sequential processing for small arrays
            op.iter().fold(T::zero(), |acc, &x| acc + x)
        };
        sum / T::from(op.len()).expect("data length should be representable")
    }

    fn var(&self) -> T {
        let op = operand(self);
        // `ddof = 0`: this trait's `var` is population variance (see the impl docs).
        // `reduce::var_*` fuses the mean and sum-of-squared-deviations passes under one
        // length-tier decision -- two separate kernel calls used to take one each, which
        // cost more in `rayon` dispatch than it saved right at the threshold.
        if let Some(s) = cast::as_f64(&op) {
            return cast::f64_to(reduce::var_f64(s, 0)).expect("T == f64 per cast::as_f64 match");
        }
        if let Some(s) = cast::as_f32(&op) {
            return cast::f32_to(reduce::var_f32(s, 0)).expect("T == f32 per cast::as_f32 match");
        }
        if op.is_empty() {
            return T::zero();
        }

        let mean = self.mean();
        let sum_squared_diff = if op.len() >= PARALLEL_THRESHOLD {
            // Use parallel processing for large arrays
            op.par_iter()
                .map(|&x| (x - mean) * (x - mean))
                .reduce(|| T::zero(), |acc, x| acc + x)
        } else {
            // Use sequential processing for small arrays
            op.iter()
                .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
        };

        sum_squared_diff / T::from(op.len()).expect("data length should be representable")
    }

    fn std(&self) -> T {
        // `var()` above already routes through `reduce::var_f64`/`var_f32` (never
        // `simd_variance`/`simd_std` -- see this impl's module docs), so std = sqrt(var)
        // is both correct (population, matching NumPy's `ddof=0`) and, for `f64`/`f32`,
        // just as kernel-accelerated as a dedicated `simd_std` call would have been.
        self.var().sqrt()
    }

    fn min(&self) -> T {
        let op = operand(self);
        if let Some(s) = cast::as_f64(&op) {
            return cast::f64_to(reduce::min_f64(s)).expect("T == f64 per cast::as_f64 match");
        }
        if let Some(s) = cast::as_f32(&op) {
            return cast::f32_to(reduce::min_f32(s)).expect("T == f32 per cast::as_f32 match");
        }
        if op.is_empty() {
            return T::zero();
        }

        // Generic-`T` tail: same NumPy rule as the `f64`/`f32` kernels above -- any `NaN`
        // propagates. Carried alongside the extremum as a `saw_nan` flag rather than through
        // the comparison itself, because `x < acc` is false whenever either side is `NaN`, so
        // a comparison fold silently *drops* interior `NaN`s. The combiner is associative and
        // commutative (min, plus a boolean OR), so the parallel branch cannot depend on how
        // rayon happens to split the work.
        if op.len() >= PARALLEL_THRESHOLD {
            let (acc, saw_nan) = op.par_iter().map(|&x| (x, x.is_nan())).reduce(
                || (op[0], op[0].is_nan()),
                |(a, a_nan), (b, b_nan)| (if b < a { b } else { a }, a_nan | b_nan),
            );
            if saw_nan {
                T::nan()
            } else {
                acc
            }
        } else {
            let (acc, saw_nan) = op.iter().fold((op[0], false), |(acc, saw_nan), &x| {
                (if x < acc { x } else { acc }, saw_nan | x.is_nan())
            });
            if saw_nan {
                T::nan()
            } else {
                acc
            }
        }
    }

    fn max(&self) -> T {
        let op = operand(self);
        if let Some(s) = cast::as_f64(&op) {
            return cast::f64_to(reduce::max_f64(s)).expect("T == f64 per cast::as_f64 match");
        }
        if let Some(s) = cast::as_f32(&op) {
            return cast::f32_to(reduce::max_f32(s)).expect("T == f32 per cast::as_f32 match");
        }
        if op.is_empty() {
            return T::zero();
        }

        // Generic-`T` tail; see `min` above for why the `NaN` flag rides alongside the
        // extremum instead of being carried by the comparison.
        if op.len() >= PARALLEL_THRESHOLD {
            let (acc, saw_nan) = op.par_iter().map(|&x| (x, x.is_nan())).reduce(
                || (op[0], op[0].is_nan()),
                |(a, a_nan), (b, b_nan)| (if b > a { b } else { a }, a_nan | b_nan),
            );
            if saw_nan {
                T::nan()
            } else {
                acc
            }
        } else {
            let (acc, saw_nan) = op.iter().fold((op[0], false), |(acc, saw_nan), &x| {
                (if x > acc { x } else { acc }, saw_nan | x.is_nan())
            });
            if saw_nan {
                T::nan()
            } else {
                acc
            }
        }
    }

    fn percentile(&self, q: T) -> T {
        // Convert to quantile (percentile is in 0-1 range, not 0-100)
        // NumPy percentile uses 0-100 scale, but our internal quantile uses 0-1
        let quantile_val = q; // q is already in 0-1 range

        // Use the more general quantile function directly
        let q_array = Array::from_vec(vec![quantile_val]);
        match quantile(self, &q_array, Some("linear")) {
            Ok(result) => result.to_vec()[0],
            Err(_) => T::zero(),
        }
    }
}

/// Peak-to-peak (maximum minus minimum) range
///
/// # Parameters
///
/// * `a` - Input array
/// * `axis` - Optional axis along which to find peak-to-peak values
///
/// # Returns
///
/// An array with the peak-to-peak values
///
/// # `NaN`
///
/// `NaN` propagates, matching `np.ptp` (which is just `np.max - np.min`, and both of those
/// propagate): any `NaN` in the reduced data -- or in a reduced lane, for the `Some(axis)`
/// form -- makes the corresponding output `NaN`. Errors on an empty array rather than
/// returning the `0` that `kernels::reduce`'s empty convention would give.
pub fn ptp<T: Float + Clone + NumCast + Default + Send + Sync>(
    a: &Array<T>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    if a.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute ptp (peak-to-peak) of an empty array".to_string(),
        ));
    }

    // If no axis specified, calculate the global ptp
    if axis.is_none() {
        // One pass for both extremes and the `NaN` flag. `np.ptp` propagates `NaN` (it is
        // `np.max - np.min`, and both of those do), so a `NaN` anywhere makes the range
        // `NaN` -- the plain comparison folds this replaced silently ignored interior
        // `NaN`s and reported a range computed from the finite elements only.
        let data = a.to_vec();
        let (min_val, max_val, saw_nan) =
            data.iter()
                .fold((data[0], data[0], false), |(min, max, saw_nan), &val| {
                    (
                        if val < min { val } else { min },
                        if val > max { val } else { max },
                        saw_nan | val.is_nan(),
                    )
                });
        let result = vec![if saw_nan { T::nan() } else { max_val - min_val }];
        return Ok(Array::from_vec(result));
    }

    // Calculate min and max along the specified axis
    let axis_val = axis.expect("axis should be Some at this point");

    // This is a simplified implementation - in a real implementation,
    // we would calculate min and max in a single pass for efficiency
    let min_array = min_along_axis(a, axis_val)?;
    let max_array = max_along_axis(a, axis_val)?;

    // Calculate ptp
    let min_data = min_array.to_vec();
    let max_data = max_array.to_vec();

    let mut result = Vec::with_capacity(min_data.len());
    for i in 0..min_data.len() {
        result.push(max_data[i] - min_data[i]);
    }

    Array::from_vec_shape(result, &min_array.shape())
}

/// Calculate minimum values along the specified axis with parallel processing for large arrays
///
/// `NaN` propagates per lane, matching NumPy's `np.min(a, axis=k)`: an output element is
/// `NaN` if *any* element of the lane it reduces is `NaN`. Use `math::nanmin` for the
/// `NaN`-ignoring variant.
pub fn min_along_axis<T: Float + Clone + NumCast + Default + Send + Sync>(
    a: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            a.ndim()
        )));
    }

    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate the shape of the result
    let mut result_shape = shape.clone();
    result_shape.remove(axis);

    // Initialize the result array. The output element count comes straight from
    // `result_shape`; it must NOT be obtained by reshaping an `empty_like(a)` scratch array,
    // which allocates `a.len()` elements and then panics in `reshape` ("Shape mismatch")
    // for every real axis reduction, since removing an axis of size > 1 necessarily changes
    // the element count. That panic made this function -- and `ptp(_, Some(axis))`, its only
    // in-crate caller -- unusable; no test reached the `Some(axis)` path to catch it.
    let data = a.to_vec();

    // For each position in the result array
    let result_size: usize = result_shape.iter().product();
    let mut min_values = vec![T::zero(); result_size];

    // Calculate the initial indices
    let mut indices = vec![0; shape.len()];
    let mut result_indices = vec![0; result_shape.len()];

    // Use parallel processing for large arrays
    if result_size >= PARALLEL_THRESHOLD {
        min_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, min_val)| {
                // Convert flat index to multi-dimensional indices
                let mut remainder = i;
                let mut result_indices = vec![0; result_shape.len()];
                for j in (0..result_shape.len()).rev() {
                    result_indices[j] = remainder % result_shape[j];
                    remainder /= result_shape[j];
                }

                // Copy the result indices to the array indices, accounting for the removed axis
                let mut indices = vec![0; shape.len()];
                let mut result_idx = 0;
                for j in 0..shape.len() {
                    if j == axis {
                        indices[j] = 0; // Start at 0 for the axis we're minimizing
                    } else {
                        indices[j] = result_indices[result_idx];
                        result_idx += 1;
                    }
                }

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                // Initialize min value with the first element. `saw_nan` rides alongside
                // because `<` is false whenever either side is `NaN`, so the comparison
                // alone would drop an interior `NaN` instead of propagating it (NumPy's
                // `np.min` rule, shared with `Statistics::min` and `kernels::reduce`).
                *min_val = data[flat_idx];
                let mut saw_nan = min_val.is_nan();

                // Compare with remaining elements along the axis
                for k in 1..axis_size {
                    indices[axis] = k;

                    // Calculate the new flat index
                    let mut new_idx = 0;
                    let mut new_stride = 1;
                    for j in (0..shape.len()).rev() {
                        new_idx += indices[j] * new_stride;
                        new_stride *= shape[j];
                    }

                    // Update min if needed
                    saw_nan |= data[new_idx].is_nan();
                    if data[new_idx] < *min_val {
                        *min_val = data[new_idx];
                    }
                }

                if saw_nan {
                    *min_val = T::nan();
                }
            });
    } else {
        // Use sequential processing for small arrays
        #[allow(clippy::needless_range_loop)]
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..shape.len() {
                if j == axis {
                    indices[j] = 0; // Start at 0 for the axis we're minimizing
                } else {
                    indices[j] = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Calculate the flat index in the original data
            let mut flat_idx = 0;
            let mut stride = 1;
            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            // Initialize min value with the first element; see the parallel branch above
            // for why `saw_nan` is tracked separately from the comparison.
            min_values[i] = data[flat_idx];
            let mut saw_nan = min_values[i].is_nan();

            // Compare with remaining elements along the axis
            for k in 1..axis_size {
                indices[axis] = k;

                // Calculate the new flat index
                let mut new_idx = 0;
                let mut new_stride = 1;
                for j in (0..shape.len()).rev() {
                    new_idx += indices[j] * new_stride;
                    new_stride *= shape[j];
                }

                // Update min if needed
                saw_nan |= data[new_idx].is_nan();
                if data[new_idx] < min_values[i] {
                    min_values[i] = data[new_idx];
                }
            }

            if saw_nan {
                min_values[i] = T::nan();
            }
        }
    }

    Array::from_vec_shape(min_values, &result_shape)
}

/// Calculate maximum values along the specified axis with parallel processing for large arrays
///
/// `NaN` propagates per lane, matching NumPy's `np.max(a, axis=k)`; see [`min_along_axis`].
pub fn max_along_axis<T: Float + Clone + NumCast + Default + Send + Sync>(
    a: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            a.ndim()
        )));
    }

    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate the shape of the result
    let mut result_shape = shape.clone();
    result_shape.remove(axis);

    // Initialize the result array; see `min_along_axis` for why the element count is taken
    // from `result_shape` rather than from a reshaped `empty_like(a)` scratch array.
    let data = a.to_vec();

    // For each position in the result array
    let result_size: usize = result_shape.iter().product();
    let mut max_values = vec![T::zero(); result_size];

    // Calculate the initial indices
    let mut indices = vec![0; shape.len()];
    let mut result_indices = vec![0; result_shape.len()];

    // Use parallel processing for large arrays
    if result_size >= PARALLEL_THRESHOLD {
        max_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, max_val)| {
                // Convert flat index to multi-dimensional indices
                let mut remainder = i;
                let mut result_indices = vec![0; result_shape.len()];
                for j in (0..result_shape.len()).rev() {
                    result_indices[j] = remainder % result_shape[j];
                    remainder /= result_shape[j];
                }

                // Copy the result indices to the array indices, accounting for the removed axis
                let mut indices = vec![0; shape.len()];
                let mut result_idx = 0;
                for j in 0..shape.len() {
                    if j == axis {
                        indices[j] = 0; // Start at 0 for the axis we're maximizing
                    } else {
                        indices[j] = result_indices[result_idx];
                        result_idx += 1;
                    }
                }

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                // Initialize max value with the first element; `saw_nan` rides alongside
                // for the same reason as in `min_along_axis` (a `>` comparison is false
                // for `NaN` operands and would silently drop an interior `NaN`).
                *max_val = data[flat_idx];
                let mut saw_nan = max_val.is_nan();

                // Compare with remaining elements along the axis
                for k in 1..axis_size {
                    indices[axis] = k;

                    // Calculate the new flat index
                    let mut new_idx = 0;
                    let mut new_stride = 1;
                    for j in (0..shape.len()).rev() {
                        new_idx += indices[j] * new_stride;
                        new_stride *= shape[j];
                    }

                    // Update max if needed
                    saw_nan |= data[new_idx].is_nan();
                    if data[new_idx] > *max_val {
                        *max_val = data[new_idx];
                    }
                }

                if saw_nan {
                    *max_val = T::nan();
                }
            });
    } else {
        // Use sequential processing for small arrays
        #[allow(clippy::needless_range_loop)]
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..shape.len() {
                if j == axis {
                    indices[j] = 0; // Start at 0 for the axis we're maximizing
                } else {
                    indices[j] = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Calculate the flat index in the original data
            let mut flat_idx = 0;
            let mut stride = 1;
            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            // Initialize max value with the first element; see the parallel branch above
            // for why `saw_nan` is tracked separately from the comparison.
            max_values[i] = data[flat_idx];
            let mut saw_nan = max_values[i].is_nan();

            // Compare with remaining elements along the axis
            for k in 1..axis_size {
                indices[axis] = k;

                // Calculate the new flat index
                let mut new_idx = 0;
                let mut new_stride = 1;
                for j in (0..shape.len()).rev() {
                    new_idx += indices[j] * new_stride;
                    new_stride *= shape[j];
                }

                // Update max if needed
                saw_nan |= data[new_idx].is_nan();
                if data[new_idx] > max_values[i] {
                    max_values[i] = data[new_idx];
                }
            }

            if saw_nan {
                max_values[i] = T::nan();
            }
        }
    }

    Array::from_vec_shape(max_values, &result_shape)
}

/// Calculate a weighted average of array elements
///
/// # Parameters
///
/// * `a` - Input array
/// * `weights` - Optional weights for each value
/// * `axis` - Optional axis along which to average
///
/// # Returns
///
/// The weighted average.
///
/// For NumPy's `average(..., returned=True)` semantics — getting the average *and* the sum
/// of weights back — call [`average_with_weights`] instead, which returns
/// `(weighted_average, sum_of_weights)` directly rather than silently dropping the weight sum.
///
/// # Breaking change (pre-1.0)
///
/// This function used to take a fourth `returned: Option<bool>` parameter, but both the
/// `Some(true)` and `Some(false)`/`None` branches returned the identical value — the weight
/// sum was computed and then thrown away, so `returned=True` never actually worked. The dead
/// parameter has been removed rather than fixed in place, since [`average_with_weights`]
/// already provides the correct, honest implementation of that behavior.
pub fn average<T: Float + Clone + Zero + NumCast + Send + Sync>(
    a: &Array<T>,
    weights: Option<&Array<T>>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    // If no weights provided, return mean
    if weights.is_none() {
        if let Some(ax) = axis {
            // Mean along specified axis
            // In a full implementation, this would use a dedicated mean_along_axis function
            return a.sum_axis(ax).map(|sum| {
                sum.scalar_div(T::from(a.shape()[ax]).expect("axis size should be representable"))
            });
        } else {
            // Calculate overall mean manually
            let data = a.to_vec();
            if data.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "Cannot average empty array".to_string(),
                ));
            }

            let sum = data.iter().fold(T::zero(), |acc, &val| acc + val);
            let mean = sum / T::from(data.len()).expect("data length should be representable");
            return Ok(Array::from_vec(vec![mean]));
        }
    }

    let w = weights.expect("weights should be Some at this point");

    // Check if weights are valid
    if a.shape() != w.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: w.shape(),
        });
    }

    // Calculate the weighted average
    let a_data = a.to_vec();
    let w_data = w.to_vec();

    if let Some(ax) = axis {
        // Weighted average along specified axis
        // This is a simplified implementation - in a real implementation,
        // we would calculate both sums in a single pass for efficiency
        let weighted_sum = weighted_sum_along_axis(a, w, ax)?;
        let weight_sum = w.sum_axis(ax)?;

        let w_sum_data = weight_sum.to_vec();
        let weighted_sum_data = weighted_sum.to_vec();

        let mut result = Vec::with_capacity(w_sum_data.len());
        for i in 0..w_sum_data.len() {
            if w_sum_data[i] == T::zero() {
                result.push(T::zero());
            } else {
                result.push(weighted_sum_data[i] / w_sum_data[i]);
            }
        }

        let avg = Array::from_vec_shape(result, &weight_sum.shape())?;

        Ok(avg)
    } else {
        // Overall weighted average
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();

        for i in 0..a_data.len() {
            weighted_sum = weighted_sum + a_data[i] * w_data[i];
            weight_sum = weight_sum + w_data[i];
        }

        let avg = if weight_sum == T::zero() {
            T::zero()
        } else {
            weighted_sum / weight_sum
        };

        Ok(Array::from_vec(vec![avg]))
    }
}

/// Calculate a weighted average and return both the average and the sum of weights
///
/// This is the companion to `average` that fulfils the NumPy `returned=True` semantic.
/// Returns `(weighted_average, sum_of_weights)` as separate `Array<T>` values.
///
/// # Parameters
///
/// * `a`       - Input array
/// * `axis`    - Optional axis along which to average. When `None`, reduces the full array.
/// * `weights` - Optional weights. When `None` every element has implicit weight 1.
///
/// # Returns
///
/// `(weighted_average, sum_of_weights)`
///
/// * For the overall case (no axis): both arrays are scalar (length-1) arrays.
/// * For the axis case: the shapes match the reduced output dimension.
pub fn average_with_weights<T: Float + Clone + Zero + NumCast + Send + Sync>(
    a: &Array<T>,
    axis: Option<usize>,
    weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>)> {
    if let Some(ax) = axis {
        // ---------------------------------------------------------------
        // Axis-reduced case
        // ---------------------------------------------------------------
        let shape = a.shape();
        let axis_size = shape[ax];

        match weights {
            None => {
                // Uniform weights of 1: avg = mean along axis, weight_sum = axis_size
                let weighted_sum = {
                    let mut result_shape = shape.clone();
                    result_shape.remove(ax);
                    let a_data = a.to_vec();
                    let result_size = result_shape.iter().product::<usize>().max(1);
                    let mut sums = vec![T::zero(); result_size];
                    let mut result_indices = vec![0usize; result_shape.len()];
                    let mut indices = vec![0usize; shape.len()];
                    for i in 0..result_size {
                        let mut remainder = i;
                        for j in (0..result_shape.len()).rev() {
                            result_indices[j] = remainder % result_shape[j];
                            remainder /= result_shape[j];
                        }
                        let mut result_idx = 0;
                        #[allow(clippy::needless_range_loop)]
                        for j in 0..shape.len() {
                            if j == ax {
                                indices[j] = 0;
                            } else {
                                indices[j] = result_indices[result_idx];
                                result_idx += 1;
                            }
                        }
                        let mut s = T::zero();
                        for k in 0..axis_size {
                            indices[ax] = k;
                            let mut flat_idx = 0;
                            let mut stride = 1;
                            for j in (0..shape.len()).rev() {
                                flat_idx += indices[j] * stride;
                                stride *= shape[j];
                            }
                            s = s + a_data[flat_idx];
                        }
                        sums[i] = s;
                    }
                    let result_shape_inner = {
                        let mut s = shape.clone();
                        s.remove(ax);
                        s
                    };
                    Array::from_vec_shape(sums, &result_shape_inner)?
                };

                let n = T::from(axis_size).ok_or_else(|| {
                    NumRs2Error::ConversionError("axis size conversion failed".to_string())
                })?;
                let weight_sum_data = weighted_sum.to_vec();
                let avg_data: Vec<T> = weight_sum_data.iter().map(|&s| s / n).collect();
                let out_shape = {
                    let mut s = shape.clone();
                    s.remove(ax);
                    s
                };
                let avg = Array::from_vec_shape(avg_data, &out_shape)?;
                let weight_sum_arr = {
                    let ws: Vec<T> = weight_sum_data.iter().map(|_| n).collect();
                    Array::from_vec_shape(ws, &out_shape)?
                };
                Ok((avg, weight_sum_arr))
            }
            Some(w) => {
                // Explicit weights
                if a.shape() != w.shape() {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: a.shape(),
                        actual: w.shape(),
                    });
                }
                let weighted_sum_arr = weighted_sum_along_axis(a, w, ax)?;
                let weight_sum_arr = w.sum_axis(ax)?;

                let w_sum_data = weight_sum_arr.to_vec();
                let ws_data = weighted_sum_arr.to_vec();
                let avg_data: Vec<T> = ws_data
                    .iter()
                    .zip(w_sum_data.iter())
                    .map(|(&ws, &wsum)| {
                        if wsum == T::zero() {
                            T::zero()
                        } else {
                            ws / wsum
                        }
                    })
                    .collect();
                let out_shape = weight_sum_arr.shape();
                let avg = Array::from_vec_shape(avg_data, &out_shape)?;
                Ok((avg, weight_sum_arr))
            }
        }
    } else {
        // ---------------------------------------------------------------
        // Overall (no-axis) case — returns scalar-valued length-1 arrays
        // ---------------------------------------------------------------
        let a_data = a.to_vec();
        if a_data.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot average empty array".to_string(),
            ));
        }

        match weights {
            None => {
                let n = T::from(a_data.len()).ok_or_else(|| {
                    NumRs2Error::ConversionError("data length conversion failed".to_string())
                })?;
                let sum = a_data.iter().fold(T::zero(), |acc, &v| acc + v);
                let avg = sum / n;
                Ok((Array::from_vec(vec![avg]), Array::from_vec(vec![n])))
            }
            Some(w) => {
                if a.shape() != w.shape() {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: a.shape(),
                        actual: w.shape(),
                    });
                }
                let w_data = w.to_vec();
                let mut weighted_sum = T::zero();
                let mut weight_sum = T::zero();
                for i in 0..a_data.len() {
                    weighted_sum = weighted_sum + a_data[i] * w_data[i];
                    weight_sum = weight_sum + w_data[i];
                }
                let avg = if weight_sum == T::zero() {
                    T::zero()
                } else {
                    weighted_sum / weight_sum
                };
                Ok((
                    Array::from_vec(vec![avg]),
                    Array::from_vec(vec![weight_sum]),
                ))
            }
        }
    }
}

/// Calculate the weighted sum along a specified axis
fn weighted_sum_along_axis<T: Float + Clone + Zero + NumCast + Send + Sync>(
    a: &Array<T>,
    weights: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            a.ndim()
        )));
    }

    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate the shape of the result
    let mut result_shape = shape.clone();
    result_shape.remove(axis);

    // Initialize the result array
    let mut result = Array::zeros(&result_shape);

    // Get the raw data
    let a_data = a.to_vec();
    let w_data = weights.to_vec();

    // Helper function to calculate indices
    let mut indices = vec![0; shape.len()];
    let mut result_indices = vec![0; result_shape.len()];

    // Calculate the total number of elements in the result
    let result_size = result.size();

    // Bulk-acquire once: `result` is write-only across this whole loop
    // (exactly one write per `i`, dynamic-rank index so `get_mut` replaces
    // `Array::set`'s bounds-checked, per-call `Arc::make_mut` path), so one
    // unshare covers all `result_size` writes.
    let result_arr = result.array_mut();

    // For each position in the result array
    for i in 0..result_size {
        // Convert flat index to multi-dimensional indices
        let mut remainder = i;
        for j in (0..result_shape.len()).rev() {
            result_indices[j] = remainder % result_shape[j];
            remainder /= result_shape[j];
        }

        // Copy the result indices to the array indices, accounting for the removed axis
        let mut result_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..shape.len() {
            if j == axis {
                indices[j] = 0; // Start at 0 for the axis we're summing
            } else {
                indices[j] = result_indices[result_idx];
                result_idx += 1;
            }
        }

        // Calculate the weighted sum along the specified axis
        let mut sum = T::zero();
        for k in 0..axis_size {
            indices[axis] = k;

            // Calculate the flat index in the original data
            let mut flat_idx = 0;
            let mut stride = 1;
            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            sum = sum + a_data[flat_idx] * w_data[flat_idx];
        }

        // Set the result value
        *result_arr
            .get_mut(result_indices.as_slice())
            .ok_or_else(|| {
                NumRs2Error::IndexOutOfBounds(format!(
                    "Failed to set element at indices {:?}",
                    result_indices
                ))
            })? = sum;
    }

    Ok(result)
}

#[cfg(test)]
mod mean_var_std_min_max_tests {
    use super::*;

    /// Hand-computed (no kernel code involved) population mean/variance for an
    /// arbitrary `f64` slice, used as independent ground truth below.
    fn hand_population_mean_var(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let sum_sq_dev: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
        (mean, sum_sq_dev / n)
    }

    /// Regression test for the `var`/`std` population-vs-sample bug: at `n == 100`
    /// (well past the old `len() >= 64` threshold that gated the buggy
    /// `simd_variance`/`simd_std` branch), `Statistics::var`/`std` must still match
    /// NumPy's `ddof=0` (population) convention, exactly as the `n == 10` case already
    /// pinned in `tests/numpy_compatibility_validation.rs` -- population variance does
    /// NOT equal sample variance at this size (`n/(n-1)` is a ~1% difference at n=100,
    /// nowhere near float noise), so this test would have failed loudly on the old code.
    #[test]
    fn var_std_are_population_not_sample_at_n_100() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect(); // uniform 0..99
        let (_, expected_var) = hand_population_mean_var(&data);
        // Closed form for a discrete uniform {0, ..., n-1}: (n^2 - 1) / 12.
        assert!(
            (expected_var - 833.25).abs() < 1e-9,
            "sanity: {expected_var}"
        );

        let arr = Array::from_vec(data);
        let got_var = arr.var();
        let got_std = arr.std();

        assert!(
            (got_var - expected_var).abs() < 1e-9,
            "population variance mismatch at n=100: got {got_var}, expected {expected_var} \
             (sample variance would have been {})",
            expected_var * 100.0 / 99.0
        );
        assert!(
            (got_std - expected_var.sqrt()).abs() < 1e-9,
            "population std mismatch at n=100: got {got_std}, expected {}",
            expected_var.sqrt()
        );
    }

    /// Same population-vs-sample check for `f32`, which never had a SIMD fast path in
    /// this trait before (only `T == f64` did) and so was never at risk of the
    /// `simd_variance`/`simd_std` bug itself, but must still land on population semantics
    /// now that it is wired onto `kernels::reduce` too.
    #[test]
    fn var_std_are_population_for_f32_at_n_100() {
        let data_f64: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (_, expected_var) = hand_population_mean_var(&data_f64);

        let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();
        let arr = Array::from_vec(data_f32);
        let got_var = arr.var();
        let got_std = arr.std();

        assert!(
            (got_var as f64 - expected_var).abs() < 1e-2,
            "f32 population variance mismatch at n=100: got {got_var}, expected {expected_var}"
        );
        assert!(
            (got_std as f64 - expected_var.sqrt()).abs() < 1e-2,
            "f32 population std mismatch at n=100: got {got_std}, expected {}",
            expected_var.sqrt()
        );
    }

    /// Below the old 64-element threshold, `var`/`std` were never routed through
    /// `simd_variance`/`simd_std`, so this is a same-answer-everywhere sanity check
    /// rather than a regression pin -- included so the n=100 test above isn't the only
    /// place population semantics are checked.
    #[test]
    fn var_std_are_population_below_old_threshold() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);
        // Population variance of [1,2,3,4,5] is 2.0 (mean 3.0); sample (ddof=1) is 2.5.
        assert!((arr.var() - 2.0).abs() < 1e-12);
        assert!((arr.std() - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn var_std_empty_is_zero() {
        let arr: Array<f64> = Array::from_vec(vec![]);
        assert_eq!(arr.var(), 0.0);
        assert_eq!(arr.std(), 0.0);
    }

    #[test]
    fn mean_matches_naive_across_dispatch_tiers() {
        for &n in &[0usize, 10, 100, 20_000] {
            let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.75 - 1.0).collect();
            let naive = if data.is_empty() {
                0.0
            } else {
                data.iter().sum::<f64>() / data.len() as f64
            };
            let arr = Array::from_vec(data);
            assert!(
                (arr.mean() - naive).abs() / naive.abs().max(1.0) < 1e-9,
                "n={n}: got {}, naive {naive}",
                arr.mean()
            );
        }
    }

    /// `min`/`max` on `f64`/`f32` dispatch through
    /// `kernels::reduce::{min,max}_{f64,f32}` at any length, and those kernels implement
    /// NumPy's rule: `NaN` propagates, wherever it sits. This test replaces an earlier
    /// version that pinned the previous `simd_min_element`/`simd_max_element` wrapper's
    /// placement-dependent values (`[NaN, 1.0, 2.0] -> 1.0`, `[1.0, 2.0, NaN] -> 1.0`);
    /// those values are no longer produced by anything in this crate.
    #[test]
    fn min_max_propagate_nan_like_numpy_below_64() {
        let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, -2.0, f64::NAN]);
        assert!(a.min().is_nan());
        assert!(a.max().is_nan());

        // NaN first.
        let b = Array::from_vec(vec![f64::NAN, 1.0, 2.0]);
        assert!(b.min().is_nan());
        assert!(b.max().is_nan());

        // NaN last -- the case the old comparison fold silently ignored.
        let c = Array::from_vec(vec![1.0, 2.0, f64::NAN]);
        assert!(c.min().is_nan());
        assert!(c.max().is_nan());
    }

    /// Was `#[ignore]`d as an upstream-regression tripwire while `min`/`max` still wrapped
    /// `simd_min_element`/`simd_max_element`. Re-enabled: `kernels::reduce` no longer calls
    /// those, and its own comparison-based kernels return `NaN` here as NumPy does.
    #[test]
    fn min_max_propagate_nan_at_len_64_boundary() {
        // The exact vector that exposed the upstream wrong-finite-value defect: true
        // maximum 5.0 at index 0, one NaN at index 10, len 64. `simd_max_element` returned
        // 1.0 for this; the crate's own kernel returns NaN.
        let mut data = vec![1.0f64; 64];
        data[10] = f64::NAN;
        data[0] = 5.0;
        let arr = Array::from_vec(data);
        assert!(arr.min().is_nan());
        assert!(arr.max().is_nan());
    }

    /// Both tiers of the dispatched kernel, through the public trait: the sequential one
    /// below `kernels::PARALLEL_MIN_LEN` and the chunked parallel one above it, with the
    /// `NaN` in an interior chunk so a kernel that only inspected the first or last chunk
    /// would fail. (The generic-`T` tail below the `cast::as_f64`/`as_f32` dispatch was
    /// given the identical rule, but cannot be exercised from here: `f64` and `f32` are the
    /// only `Float` types this crate has, so every `Statistics` instantiation available to a
    /// test takes the dispatched path.)
    #[test]
    fn min_max_propagate_nan_on_both_dispatch_tiers() {
        for &n in &[5usize, PARALLEL_THRESHOLD + 7] {
            let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            data[n / 2] = f64::NAN;
            let arr = Array::from_vec(data);
            assert!(arr.min().is_nan(), "n={n}");
            assert!(arr.max().is_nan(), "n={n}");
        }
    }

    /// `ptp` is `max - min` and inherits the same rule (`np.ptp` propagates).
    #[test]
    fn ptp_propagates_nan_like_numpy() {
        let arr = Array::from_vec(vec![1.0f64, 5.0, f64::NAN, 2.0]);
        let got = ptp(&arr, None).expect("ptp on non-empty array should succeed");
        assert!(got.to_vec()[0].is_nan());
    }

    /// Per-lane propagation for the `Some(axis)` reductions behind `ptp`: only the lane
    /// containing the `NaN` goes `NaN`, the other lane keeps its real range.
    #[test]
    fn min_max_along_axis_propagate_nan_per_lane() {
        // [[1, NaN, 3], [4, 5, 6]] reduced along axis 1.
        let arr = Array::from_vec(vec![1.0f64, f64::NAN, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let mins = min_along_axis(&arr, 1).expect("min_along_axis should succeed");
        let maxs = max_along_axis(&arr, 1).expect("max_along_axis should succeed");
        assert!(mins.to_vec()[0].is_nan());
        assert!(maxs.to_vec()[0].is_nan());
        assert_eq!(mins.to_vec()[1], 4.0);
        assert_eq!(maxs.to_vec()[1], 6.0);

        let ranges = ptp(&arr, Some(1)).expect("ptp should succeed");
        assert!(ranges.to_vec()[0].is_nan());
        assert_eq!(ranges.to_vec()[1], 2.0);
    }

    /// **Live upstream bug, not a NaN-convention change.** Found while wiring
    /// `math::aggregation::max`/`min` onto `kernels::reduce` (withheld as a result -- see that
    /// module's doc comments): on a 64-element `f64` slice `[5.0, 1.0 (x9), NaN, 1.0 (x53)]`
    /// (true maximum `5.0` at index 0, a single `NaN` at index 10),
    /// `scirs2_core::simd_ops::SimdUnifiedOps::simd_max_element` returns `1.0` -- silently
    /// discarding the real maximum, not `NaN` and not `5.0`. This is called *directly* here,
    /// with no `numrs2` dispatch code (`kernels::borrow::operand`/`kernels::cast`) in between,
    /// specifically to rule out a bug in this crate's own plumbing: the defect reproduces with
    /// zero numrs2 code involved, so it is upstream in `scirs2-core` itself.
    ///
    /// This test intentionally pins the *current, believed-wrong* value (`1.0`) as a tripwire:
    /// it is expected to start FAILING the moment a `scirs2-core` upgrade fixes the underlying
    /// kernel, at which point `min_max_nan_behavior_matches_kernels_reduce_at_len_64_boundary`
    /// above should be un-`#[ignore]`d and `kernels::reduce`'s "NaN handling is pinned, not a
    /// simple rule" module docs (and `math::aggregation::max`/`min`'s withheld dispatch) revisited.
    #[test]
    fn simd_max_element_upstream_wrong_value_is_a_live_bug_not_just_new_nan_convention() {
        use scirs2_core::ndarray::ArrayView1;
        use scirs2_core::simd_ops::SimdUnifiedOps;

        let mut data = vec![1.0f64; 64];
        data[0] = 5.0; // true maximum
        data[10] = f64::NAN;

        let direct = <f64 as SimdUnifiedOps>::simd_max_element(&ArrayView1::from(&data[..]));
        assert_eq!(
            direct, 1.0,
            "if this fails because `direct` is now 5.0 (correct) or NaN (conservative), \
             scirs2-core has changed this kernel's behavior -- see this test's doc comment \
             for what to do next; do NOT just update this assertion to match a new value \
             without re-checking whether other NaN placements are still wrong"
        );
    }

    #[test]
    fn min_max_no_nan_matches_naive_across_dispatch_tiers() {
        for &n in &[1usize, 10, 100, 20_000] {
            let data: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 - 500.0).collect();
            let naive_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let naive_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let arr = Array::from_vec(data);
            assert_eq!(arr.min(), naive_min, "n={n}");
            assert_eq!(arr.max(), naive_max, "n={n}");
        }
    }

    #[test]
    fn min_max_empty_is_zero() {
        let arr: Array<f64> = Array::from_vec(vec![]);
        assert_eq!(arr.min(), 0.0);
        assert_eq!(arr.max(), 0.0);
    }

    #[test]
    fn ptp_empty_array_is_error() {
        let arr: Array<f64> = Array::from_vec(vec![]);
        assert!(ptp(&arr, None).is_err());
    }

    #[test]
    fn ptp_matches_max_minus_min() {
        let arr = Array::from_vec(vec![5.0f64, -3.0, 8.0, 0.5, -10.0, 2.0]);
        let result = ptp(&arr, None).expect("ptp on non-empty array should succeed");
        assert_eq!(result.to_vec(), vec![18.0]); // 8.0 - (-10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_with_weights_overall() -> Result<()> {
        // a=[1,2,3,4], weights=[1,2,3,4]
        // weighted_sum = 1*1+2*2+3*3+4*4 = 1+4+9+16 = 30
        // weight_sum   = 1+2+3+4         = 10
        // avg          = 30/10           = 3.0
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let w = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);

        let (avg, weight_sum) = average_with_weights(&a, None, Some(&w))?;

        let avg_val = avg.to_vec()[0];
        let ws_val = weight_sum.to_vec()[0];

        assert!(
            (avg_val - 3.0).abs() < 1e-12,
            "expected avg=3.0, got {}",
            avg_val
        );
        assert!(
            (ws_val - 10.0).abs() < 1e-12,
            "expected weight_sum=10.0, got {}",
            ws_val
        );
        Ok(())
    }

    #[test]
    fn test_average_with_weights_no_weights() -> Result<()> {
        // a=[1.0,2.0,3.0], no weights → uniform weight 1
        // avg = (1+2+3)/3 = 2.0
        // weight_sum = 3.0
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0]);

        let (avg, weight_sum) = average_with_weights(&a, None, None)?;

        let avg_val = avg.to_vec()[0];
        let ws_val = weight_sum.to_vec()[0];

        assert!(
            (avg_val - 2.0).abs() < 1e-12,
            "expected avg=2.0, got {}",
            avg_val
        );
        assert!(
            (ws_val - 3.0).abs() < 1e-12,
            "expected weight_sum=3.0, got {}",
            ws_val
        );
        Ok(())
    }
}
