//! Array padding operations
//!
//! This module provides the `pad` function for padding arrays with various modes.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

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

    Array::from_vec_shape(result_data, &new_shape)
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
