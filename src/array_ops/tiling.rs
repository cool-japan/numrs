use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::cmp;

/// Construct an array by repeating array the given number of times
pub fn tile<T: Clone>(array: &Array<T>, reps: &[usize]) -> Result<Array<T>> {
    let a_shape = array.shape();

    // Determine the output shape
    let mut output_shape = Vec::with_capacity(cmp::max(a_shape.len(), reps.len()));

    // Ensure reps has at least as many dimensions as a_shape, filling with 1s if needed
    let mut full_reps = Vec::with_capacity(cmp::max(a_shape.len(), reps.len()));
    let reps_offset = if a_shape.len() > reps.len() {
        a_shape.len() - reps.len()
    } else {
        0
    };

    for i in 0..full_reps.capacity() {
        if i < reps_offset {
            full_reps.push(1);
        } else {
            full_reps.push(reps[i - reps_offset]);
        }
    }

    // Compute the output shape
    for (&a_dim, &rep) in a_shape.iter().zip(full_reps.iter()) {
        output_shape.push(a_dim * rep);
    }

    // Add extra dimensions if reps has more dimensions than a_shape
    if reps.len() > a_shape.len() {
        let a_offset = reps.len() - a_shape.len();
        for &rep in reps.iter().take(a_offset) {
            output_shape.insert(0, rep);
        }
    }

    // Create the output array filled with the first element as a placeholder
    let first_elem = array
        .array()
        .first()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Cannot tile an empty array".into()))?
        .clone();

    let mut result = Array::full(&output_shape, first_elem);

    // Fill the output array by copying the input array in a tiled pattern
    // This is a simplified implementation - for efficiency, we would use
    // more sophisticated slicing and assignment operations

    let result_vec = result
        .array_mut()
        .as_slice_mut()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to get mutable slice".into()))?;

    let input_vec = array.to_vec();
    let input_size = input_vec.len();

    if input_size == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot tile an empty array".into(),
        ));
    }

    // For each position in the output, copy the corresponding element from the input
    for (i, item) in result_vec.iter_mut().enumerate() {
        // Calculate corresponding index in the input array
        // This is a simplification - for a complete implementation, we would need
        // to carefully map N-dimensional indices
        let input_idx = i % input_size;
        *item = input_vec[input_idx].clone();
    }

    Ok(result)
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