//! Array padding operations
//!
//! This module provides the `pad` function for padding arrays with various
//! modes, matching `numpy.pad` (all 11 modes it supports).
//!
//! ## Corner / cascading semantics (N-D arrays)
//!
//! For rank > 1 arrays, NumPy pads axes **in order** (axis 0, then axis 1,
//! ...), and later axes may read values that earlier axes already wrote into
//! the padding region. Concretely, this means corner regions are filled
//! using statistics/reflections *of the already-padded array*, not just the
//! original data -- e.g. for `mode="mean"` on a 2-D array, the corner where
//! both axis 0 and axis 1 are padded holds the mean of the axis-0 padding
//! row (itself a per-column mean), which works out to the grand mean of the
//! original block. This implementation reproduces that behavior exactly by
//! processing axes in the same order and always reading from the (partially
//! built) output buffer.
//!
//! ## Supported modes
//!
//! * `"constant"` - pad with `constant_values` (default 0)
//! * `"edge"` - pad with the edge values of the array
//! * `"linear_ramp"` - linear ramp between `constant_values` (reused as the
//!   ramp's end value, default 0) and the array edge
//! * `"maximum"`, `"mean"`, `"median"`, `"minimum"` - pad with a statistic
//!   computed over the *entire* axis (NumPy's `stat_length=None` default;
//!   this implementation does not support NumPy's partial `stat_length`)
//! * `"reflect"` - reflect, excluding the edge value
//! * `"symmetric"` - reflect, including the edge value
//! * `"wrap"` - wrap values from the opposite edge
//! * `"empty"` - leave the padded region with unspecified content (only the
//!   shape and the copied original data are guaranteed)
//!
//! `"reflect"` and `"symmetric"` default to NumPy's `reflect_type="even"`
//! (an unaltered mirror). The `reflect_type="odd"` variant (the extended
//! part is `2 * edge - mirrored_value`) is available via the `"reflect_odd"`
//! and `"symmetric_odd"` mode strings, since this function's signature has
//! no separate `reflect_type` parameter.
//!
//! `constant_values` is a single scalar applied to both sides of every axis
//! (used as the fill value for `"constant"`, and reused as the ramp end
//! value for `"linear_ramp"`); NumPy's per-axis, per-side tuples are not
//! supported, matching this function's pre-existing `constant_values`
//! simplification.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{NumCast, Zero};
use std::ops::{Add, Div, Mul, Sub};

const VALID_MODES: &[&str] = &[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "reflect_odd",
    "symmetric",
    "symmetric_odd",
    "wrap",
    "empty",
];

/// Pad an array, matching `numpy.pad`.
///
/// # Parameters
///
/// * `array` - Array to be padded
/// * `pad_width` - Number of values padded to the edges of each axis.
///   For each axis, provide (before, after) padding sizes.
/// * `mode` - Padding mode; see the [module documentation](self) for the
///   full list, including the `"reflect_odd"` / `"symmetric_odd"` spelling
///   used to select NumPy's `reflect_type="odd"`.
/// * `constant_values` - Used by `"constant"` (the fill value, default 0)
///   and `"linear_ramp"` (the ramp end value, default 0). Ignored by all
///   other modes.
///
/// # Returns
///
/// Padded array of same type as input array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Pad 1D array with constant value
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = pad(&a, &[(2, 3)], "constant", Some(0)).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![0, 0, 1, 2, 3, 0, 0, 0]);
///
/// // Pad 2D array with edge values
/// let b = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let result = pad(&b, &[(1, 1), (2, 2)], "edge", None).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![4, 6]);
/// ```
pub fn pad<T>(
    array: &Array<T>,
    pad_width: &[(usize, usize)],
    mode: &str,
    constant_values: Option<T>,
) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + PartialOrd
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let shape = array.shape();

    if pad_width.len() != shape.len() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "pad_width must have same length as array dimensions. Got {} for {} dimensions",
            pad_width.len(),
            shape.len()
        )));
    }

    let (base_mode, reflect_odd) = match mode {
        "reflect_odd" => ("reflect", true),
        "symmetric_odd" => ("symmetric", true),
        other => (other, false),
    };
    if !VALID_MODES.contains(&mode) {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Unknown pad mode: {}. Must be one of: {}",
            mode,
            VALID_MODES.join(", ")
        )));
    }

    let mut new_shape = Vec::with_capacity(shape.len());
    for (i, &dim) in shape.iter().enumerate() {
        let (before, after) = pad_width[i];
        new_shape.push(before + dim + after);
    }
    let new_strides = row_major_strides(&new_shape);
    let total_size: usize = new_shape.iter().product();

    let fill_value = if base_mode == "constant" {
        constant_values.clone().unwrap_or_else(T::zero)
    } else {
        T::zero()
    };
    let mut result_data = vec![fill_value; total_size];

    // Copy original data into the center region.
    let old_strides = row_major_strides(&shape);
    let original_data = array.to_vec();
    for (i, value) in original_data.into_iter().enumerate() {
        let old_indices = index_from_flat(i, &shape, &old_strides);
        let mut new_indices = vec![0usize; shape.len()];
        for (j, idx) in old_indices.iter().enumerate() {
            new_indices[j] = idx + pad_width[j].0;
        }
        let new_flat = flat_from_index(&new_indices, &new_strides);
        result_data[new_flat] = value;
    }

    match base_mode {
        "constant" | "empty" => {
            // Already fully handled: `result_data` is either filled with
            // the constant value (with original data copied over it) or,
            // for "empty", left at the arbitrary initial fill -- NumPy also
            // leaves this region's content unspecified.
        }
        "edge" => pad_edge(
            &mut result_data,
            &shape,
            pad_width,
            &new_shape,
            &new_strides,
        ),
        "linear_ramp" => {
            let end_value = constant_values.unwrap_or_else(T::zero);
            pad_linear_ramp(
                &mut result_data,
                &shape,
                pad_width,
                &new_shape,
                &new_strides,
                end_value,
            );
        }
        "maximum" | "mean" | "median" | "minimum" => pad_stat(
            &mut result_data,
            &shape,
            pad_width,
            &new_shape,
            &new_strides,
            base_mode,
        ),
        "reflect" | "symmetric" => pad_reflect_or_symmetric(
            &mut result_data,
            &shape,
            pad_width,
            &new_shape,
            &new_strides,
            base_mode == "symmetric",
            reflect_odd,
        ),
        "wrap" => pad_wrap(
            &mut result_data,
            &shape,
            pad_width,
            &new_shape,
            &new_strides,
        ),
        _ => unreachable!("mode already validated against VALID_MODES"),
    }

    Array::from_vec_shape(result_data, &new_shape)
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

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
    indices.iter().zip(strides).map(|(i, s)| i * s).sum()
}

/// Invoke `f` once for every "lane" along `axis`: every combination of
/// indices for the *other* axes (ranging over the buffer's current, possibly
/// already-partially-padded, `shape`). `f` receives a mutable index buffer
/// whose `axis` slot is left at `0` for the caller to set freely.
fn for_each_lane(shape: &[usize], axis: usize, mut f: impl FnMut(&mut [usize])) {
    let ndim = shape.len();
    let mut lane_shape = shape.to_vec();
    lane_shape[axis] = 1;
    let lane_strides = row_major_strides(&lane_shape);
    let total: usize = lane_shape.iter().product();
    let mut idx = vec![0usize; ndim];
    for flat in 0..total {
        let mut rem = flat;
        for d in 0..ndim {
            idx[d] = rem / lane_strides[d];
            rem %= lane_strides[d];
        }
        idx[axis] = 0;
        f(&mut idx);
    }
}

fn pad_edge<T: Clone>(
    data: &mut [T],
    shape: &[usize],
    pad_width: &[(usize, usize)],
    new_shape: &[usize],
    new_strides: &[usize],
) {
    for axis in 0..shape.len() {
        let (before, after) = pad_width[axis];
        if before == 0 && after == 0 {
            continue;
        }
        let orig_len = shape[axis];
        for_each_lane(new_shape, axis, |idx| {
            idx[axis] = before;
            let edge_before = data[flat_from_index(idx, new_strides)].clone();
            idx[axis] = before + orig_len - 1;
            let edge_after = data[flat_from_index(idx, new_strides)].clone();

            for k in 0..before {
                idx[axis] = k;
                let f = flat_from_index(idx, new_strides);
                data[f] = edge_before.clone();
            }
            for k in 0..after {
                idx[axis] = before + orig_len + k;
                let f = flat_from_index(idx, new_strides);
                data[f] = edge_after.clone();
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn pad_linear_ramp<T>(
    data: &mut [T],
    shape: &[usize],
    pad_width: &[(usize, usize)],
    new_shape: &[usize],
    new_strides: &[usize],
    end_value: T,
) where
    T: Clone
        + Zero
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast,
{
    for axis in 0..shape.len() {
        let (before, after) = pad_width[axis];
        if before == 0 && after == 0 {
            continue;
        }
        let orig_len = shape[axis];
        for_each_lane(new_shape, axis, |idx| {
            idx[axis] = before;
            let edge_before = data[flat_from_index(idx, new_strides)].clone();
            idx[axis] = before + orig_len - 1;
            let edge_after = data[flat_from_index(idx, new_strides)].clone();

            // NumPy: np.linspace(start=end_value, stop=edge, num=width,
            // endpoint=False) -- i.e. step = (edge - end_value) / width,
            // and the point closest to the data (index width-1) is
            // `edge - step`, never `edge` itself.
            if before > 0 {
                let step = (edge_before - end_value.clone())
                    / T::from(before).expect("pad width should be representable");
                for j in 0..before {
                    idx[axis] = j;
                    let val = end_value.clone()
                        + step.clone() * T::from(j).expect("index should be representable");
                    let f = flat_from_index(idx, new_strides);
                    data[f] = val;
                }
            }
            if after > 0 {
                let step = (edge_after - end_value.clone())
                    / T::from(after).expect("pad width should be representable");
                // The raw ramp (end_value -> edge, endpoint excluded) is
                // reversed so the point closest to the data comes first.
                for k in 0..after {
                    let reversed_i = after - 1 - k;
                    idx[axis] = before + orig_len + k;
                    let val = end_value.clone()
                        + step.clone()
                            * T::from(reversed_i).expect("index should be representable");
                    let f = flat_from_index(idx, new_strides);
                    data[f] = val;
                }
            }
        });
    }
}

fn pad_stat<T>(
    data: &mut [T],
    shape: &[usize],
    pad_width: &[(usize, usize)],
    new_shape: &[usize],
    new_strides: &[usize],
    stat_mode: &str,
) where
    T: Clone + Zero + PartialOrd + Add<Output = T> + Div<Output = T> + NumCast,
{
    for axis in 0..shape.len() {
        let (before, after) = pad_width[axis];
        if before == 0 && after == 0 {
            continue;
        }
        let orig_len = shape[axis];
        for_each_lane(new_shape, axis, |idx| {
            let mut lane_vals: Vec<T> = Vec::with_capacity(orig_len);
            for k in 0..orig_len {
                idx[axis] = before + k;
                lane_vals.push(data[flat_from_index(idx, new_strides)].clone());
            }

            let stat_val = compute_stat(&lane_vals, stat_mode);

            for k in 0..before {
                idx[axis] = k;
                let f = flat_from_index(idx, new_strides);
                data[f] = stat_val.clone();
            }
            for k in 0..after {
                idx[axis] = before + orig_len + k;
                let f = flat_from_index(idx, new_strides);
                data[f] = stat_val.clone();
            }
        });
    }
}

fn compute_stat<T>(lane_vals: &[T], stat_mode: &str) -> T
where
    T: Clone + Zero + PartialOrd + Add<Output = T> + Div<Output = T> + NumCast,
{
    match stat_mode {
        "maximum" => lane_vals
            .iter()
            .skip(1)
            .fold(
                lane_vals[0].clone(),
                |acc, v| if *v > acc { v.clone() } else { acc },
            ),
        "minimum" => lane_vals
            .iter()
            .skip(1)
            .fold(
                lane_vals[0].clone(),
                |acc, v| if *v < acc { v.clone() } else { acc },
            ),
        "mean" => {
            let sum = lane_vals.iter().cloned().fold(T::zero(), |acc, v| acc + v);
            sum / T::from(lane_vals.len()).expect("lane length should be representable")
        }
        "median" => {
            let mut sorted_lane = lane_vals.to_vec();
            sorted_lane.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = sorted_lane.len();
            if m % 2 == 1 {
                sorted_lane[m / 2].clone()
            } else {
                (sorted_lane[m / 2 - 1].clone() + sorted_lane[m / 2].clone())
                    / T::from(2).expect("2 should be representable")
            }
        }
        _ => unreachable!("stat_mode already restricted to maximum/mean/median/minimum"),
    }
}

/// One reflect/symmetric round's chunk length, following NumPy's
/// `_set_reflect_both`: the usable chunk is aligned to a whole multiple of
/// the reflection period so a partial period is never used as a source.
fn reflect_old_length(valid_extent: usize, axis_size: usize, include_edge: bool) -> usize {
    if include_edge {
        (valid_extent / axis_size) * axis_size
    } else {
        // axis_size >= 2 is guaranteed by the axis_size == 1 special case
        // handled by the caller.
        ((valid_extent - 1) / (axis_size - 1)) * (axis_size - 1)
    }
}

#[allow(clippy::too_many_arguments)]
fn pad_reflect_or_symmetric<T: Clone>(
    data: &mut [T],
    shape: &[usize],
    pad_width: &[(usize, usize)],
    new_shape: &[usize],
    new_strides: &[usize],
    include_edge: bool,
    odd: bool,
) where
    T: Add<Output = T> + Sub<Output = T>,
{
    for axis in 0..shape.len() {
        let (before, after) = pad_width[axis];
        if before == 0 && after == 0 {
            continue;
        }
        let axis_size = shape[axis];
        let total_len = new_shape[axis];

        if axis_size == 1 {
            // Legacy NumPy special case: reflecting a length-1 axis
            // degenerates to edge-fill (avoids a division by zero below).
            for_each_lane(new_shape, axis, |idx| {
                idx[axis] = before;
                let edge_val = data[flat_from_index(idx, new_strides)].clone();
                for k in 0..before {
                    idx[axis] = k;
                    let f = flat_from_index(idx, new_strides);
                    data[f] = edge_val.clone();
                }
                for k in 0..after {
                    idx[axis] = before + 1 + k;
                    let f = flat_from_index(idx, new_strides);
                    data[f] = edge_val.clone();
                }
            });
            continue;
        }

        for_each_lane(new_shape, axis, |idx| {
            let mut left_remaining = before;
            let mut right_remaining = after;

            while left_remaining > 0 || right_remaining > 0 {
                // Both sides share the same basis for this round, computed
                // once from the round-start remaining widths (NumPy calls
                // `_set_reflect_both` once per round for both sides).
                let valid_extent = total_len - left_remaining - right_remaining;
                let old_length = reflect_old_length(valid_extent, axis_size, include_edge);

                if left_remaining > 0 {
                    let chunk_length = old_length.min(left_remaining);
                    if chunk_length == 0 {
                        // Defensive: avoid an infinite loop; NumPy's own
                        // invariants guarantee old_length > 0 here whenever
                        // axis_size >= 2, but never spin forever.
                        left_remaining = 0;
                    } else {
                        idx[axis] = left_remaining;
                        let pivot = if odd {
                            Some(data[flat_from_index(idx, new_strides)].clone())
                        } else {
                            None
                        };
                        for j in 0..chunk_length {
                            let p = left_remaining - chunk_length + j;
                            let source = if include_edge {
                                2 * left_remaining - 1 - p
                            } else {
                                2 * left_remaining - p
                            };
                            idx[axis] = source;
                            let source_val = data[flat_from_index(idx, new_strides)].clone();
                            let final_val = match &pivot {
                                Some(pv) => pv.clone() + pv.clone() - source_val,
                                None => source_val,
                            };
                            idx[axis] = p;
                            let f = flat_from_index(idx, new_strides);
                            data[f] = final_val;
                        }
                        left_remaining -= chunk_length;
                    }
                }

                if right_remaining > 0 {
                    let chunk_length = old_length.min(right_remaining);
                    if chunk_length == 0 {
                        right_remaining = 0;
                    } else {
                        let right_boundary = total_len - right_remaining - 1;
                        idx[axis] = right_boundary;
                        let pivot = if odd {
                            Some(data[flat_from_index(idx, new_strides)].clone())
                        } else {
                            None
                        };
                        let start_p = total_len - right_remaining;
                        for k in 0..chunk_length {
                            let p = start_p + k;
                            let source = if include_edge {
                                2 * (right_boundary + 1) - 1 - p
                            } else {
                                2 * right_boundary - p
                            };
                            idx[axis] = source;
                            let source_val = data[flat_from_index(idx, new_strides)].clone();
                            let final_val = match &pivot {
                                Some(pv) => pv.clone() + pv.clone() - source_val,
                                None => source_val,
                            };
                            idx[axis] = p;
                            let f = flat_from_index(idx, new_strides);
                            data[f] = final_val;
                        }
                        right_remaining -= chunk_length;
                    }
                }
            }
        });
    }
}

fn pad_wrap<T: Clone>(
    data: &mut [T],
    shape: &[usize],
    pad_width: &[(usize, usize)],
    new_shape: &[usize],
    new_strides: &[usize],
) {
    for axis in 0..shape.len() {
        let (before, after) = pad_width[axis];
        if before == 0 && after == 0 {
            continue;
        }
        let axis_size = shape[axis];
        let total_len = new_shape[axis];

        for_each_lane(new_shape, axis, |idx| {
            let mut left_remaining = before;
            let mut right_remaining = after;

            while left_remaining > 0 || right_remaining > 0 {
                let valid_extent = total_len - left_remaining - right_remaining;
                // Period aligned to a whole multiple of axis_size, so
                // wrapping never uses a partial copy of the original data.
                let period = (valid_extent / axis_size) * axis_size;

                if left_remaining > 0 {
                    let chunk_length = period.min(left_remaining);
                    if chunk_length == 0 {
                        left_remaining = 0;
                    } else {
                        for j in 0..chunk_length {
                            let p = left_remaining - chunk_length + j;
                            idx[axis] = p + period;
                            let source_val = data[flat_from_index(idx, new_strides)].clone();
                            idx[axis] = p;
                            let f = flat_from_index(idx, new_strides);
                            data[f] = source_val;
                        }
                        left_remaining -= chunk_length;
                    }
                }

                if right_remaining > 0 {
                    let chunk_length = period.min(right_remaining);
                    if chunk_length == 0 {
                        right_remaining = 0;
                    } else {
                        let start_p = total_len - right_remaining;
                        for k in 0..chunk_length {
                            let p = start_p + k;
                            idx[axis] = p - period;
                            let source_val = data[flat_from_index(idx, new_strides)].clone();
                            idx[axis] = p;
                            let f = flat_from_index(idx, new_strides);
                            data[f] = source_val;
                        }
                        right_remaining -= chunk_length;
                    }
                }
            }
        });
    }
}
