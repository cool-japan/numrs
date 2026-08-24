use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::cmp;

/// Compute row-major (C-order) strides for the given shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Construct an array by repeating `array` the number of times given by `reps`.
///
/// Follows NumPy's `tile` semantics: letting `d = max(array.ndim(), reps.len())`,
/// both `array`'s shape and `reps` are left-padded with `1`s to length `d`, the
/// output shape is their element-wise product, and each output element is
/// gathered from the input via `out[idx] = a[idx % a_shape]` (per-axis modulo,
/// using N-D index arithmetic).
pub fn tile<T: Clone>(array: &Array<T>, reps: &[usize]) -> Result<Array<T>> {
    let a_shape = array.shape();

    // Force standard (C-contiguous) layout so the flat data below is in the
    // same row-major order that `a_shape` implies, even if `array` happens to
    // be a permuted/strided view (e.g. the result of `moveaxis`/`transpose`).
    let input_vec = array.to_c_layout().to_vec();

    if input_vec.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot tile an empty array".into(),
        ));
    }

    // d = max(a.ndim, len(reps)); left-pad the shorter of `a_shape` / `reps`
    // with 1s to length d.
    let d = cmp::max(a_shape.len(), reps.len());

    let mut a_shape_padded = vec![1usize; d - a_shape.len()];
    a_shape_padded.extend_from_slice(&a_shape);

    let mut reps_padded = vec![1usize; d - reps.len()];
    reps_padded.extend_from_slice(reps);

    let output_shape: Vec<usize> = a_shape_padded
        .iter()
        .zip(reps_padded.iter())
        .map(|(&dim, &rep)| dim * rep)
        .collect();

    let output_size: usize = output_shape.iter().product();

    // Strides (row-major) for the padded input shape and the output shape,
    // used to convert between flat and multi-dimensional indices.
    let in_strides = row_major_strides(&a_shape_padded);
    let out_strides = row_major_strides(&output_shape);

    let mut result_data = Vec::with_capacity(output_size);
    for flat_out in 0..output_size {
        // Unravel the output flat index into per-axis coordinates, take each
        // coordinate modulo the (padded) input shape, then ravel back into a
        // flat index into the input buffer.
        let mut input_flat = 0usize;
        let mut remainder = flat_out;
        for axis in 0..d {
            let out_coord = remainder / out_strides[axis];
            remainder %= out_strides[axis];
            let in_coord = out_coord % a_shape_padded[axis];
            input_flat += in_coord * in_strides[axis];
        }
        result_data.push(input_vec[input_flat].clone());
    }

    Array::from_vec_shape(result_data, &output_shape)
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
            // Flattened repeat - repeat each element individually
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
