//! Bit manipulation and indexing operations
//!
//! This module provides functions for:
//! - Packing and unpacking bits
//! - Converting between flat indices and multi-dimensional coordinates

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

/// Pack elements of a binary-valued array into bits in a uint8 array
///
/// # Parameters
///
/// * `array` - Array of binary values (0 or 1) to be packed
/// * `axis` - The dimension along which packing is performed. If None, the array is flattened
/// * `bitorder` - Bit order ('big' or 'little'). 'big' means the most significant bit is at the beginning
///
/// # Returns
///
/// Packed array with type uint8. The dimension along the given axis is divided by 8.
/// If the number of elements is not divisible by 8, the last byte is padded with zeros.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Pack 1D binary array
/// let a = Array::from_vec(vec![1u8, 0, 1, 0, 0, 0, 1, 1]);
/// let packed = packbits(&a, None, Some("big")).expect("operation should succeed");
/// assert_eq!(packed.to_vec(), vec![163u8]); // 10100011 in binary = 163
///
/// // Pack with padding
/// let b = Array::from_vec(vec![1u8, 1, 1]);
/// let packed = packbits(&b, None, Some("big")).expect("operation should succeed");
/// assert_eq!(packed.to_vec(), vec![224u8]); // 11100000 in binary = 224
/// ```
pub fn packbits(
    array: &Array<u8>,
    axis: Option<isize>,
    bitorder: Option<&str>,
) -> Result<Array<u8>> {
    let bitorder_str = bitorder.unwrap_or("big");
    if bitorder_str != "big" && bitorder_str != "little" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "bitorder must be 'big' or 'little', got '{}'",
            bitorder_str
        )));
    }

    // Validate input contains only 0s and 1s
    let data = array.to_vec();
    for &val in &data {
        if val != 0 && val != 1 {
            return Err(NumRs2Error::InvalidOperation(
                "packbits requires binary input (0 or 1)".to_string(),
            ));
        }
    }

    match axis {
        Some(ax) => {
            // Pack along specified axis
            let ndim = array.ndim();
            let axis_idx = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis_idx >= ndim {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis_idx];
            let packed_axis_size = axis_size.div_ceil(8); // Ceiling division

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[axis_idx] = packed_axis_size;

            // Calculate total size
            let mut outer_size = 1;
            for i in 0..axis_idx {
                outer_size *= shape[i];
            }
            let mut inner_size = 1;
            for i in (axis_idx + 1)..ndim {
                inner_size *= shape[i];
            }

            let mut packed_data = Vec::with_capacity(outer_size * packed_axis_size * inner_size);

            // Pack bits along the specified axis
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    for packed_idx in 0..packed_axis_size {
                        let mut byte = 0u8;
                        let start_bit = packed_idx * 8;
                        let end_bit = ((packed_idx + 1) * 8).min(axis_size);

                        for bit_idx in start_bit..end_bit {
                            let flat_idx =
                                outer * axis_size * inner_size + bit_idx * inner_size + inner;
                            let bit = data[flat_idx];

                            if bitorder_str == "big" {
                                byte |= bit << (7 - (bit_idx - start_bit));
                            } else {
                                byte |= bit << (bit_idx - start_bit);
                            }
                        }

                        packed_data.push(byte);
                    }
                }
            }

            Ok(Array::from_vec_shape(packed_data, &new_shape)?)
        }
        None => {
            // Pack flattened array
            let flat_data = array.to_vec();
            let n = flat_data.len();
            let packed_size = n.div_ceil(8);
            let mut packed = Vec::with_capacity(packed_size);

            for i in 0..packed_size {
                let mut byte = 0u8;
                let start = i * 8;
                let end = ((i + 1) * 8).min(n);

                for j in start..end {
                    let bit = flat_data[j];
                    if bitorder_str == "big" {
                        byte |= bit << (7 - (j - start));
                    } else {
                        byte |= bit << (j - start);
                    }
                }

                packed.push(byte);
            }

            Ok(Array::from_vec(packed))
        }
    }
}

/// Unpack elements of a uint8 array into a binary-valued array
///
/// # Parameters
///
/// * `packed` - Array of type uint8 to be unpacked
/// * `axis` - The dimension along which unpacking is performed. If None, the array is flattened
/// * `count` - The number of elements to unpack along the given axis. If None, unpacks `8 * packed.shape[axis]`
/// * `bitorder` - Bit order ('big' or 'little'). 'big' means the most significant bit is at the beginning
///
/// # Returns
///
/// The unpacked array with binary values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Unpack uint8 array
/// let packed = Array::from_vec(vec![163u8]); // 10100011 in binary
/// let unpacked = unpackbits(&packed, None, None, Some("big")).expect("operation should succeed");
/// assert_eq!(unpacked.to_vec(), vec![1, 0, 1, 0, 0, 0, 1, 1]);
///
/// // Unpack with specific count
/// let packed = Array::from_vec(vec![224u8]); // 11100000 in binary
/// let unpacked = unpackbits(&packed, None, Some(3), Some("big")).expect("operation should succeed");
/// assert_eq!(unpacked.to_vec(), vec![1, 1, 1]);
/// ```
pub fn unpackbits(
    packed: &Array<u8>,
    axis: Option<isize>,
    count: Option<usize>,
    bitorder: Option<&str>,
) -> Result<Array<u8>> {
    let bitorder_str = bitorder.unwrap_or("big");
    if bitorder_str != "big" && bitorder_str != "little" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "bitorder must be 'big' or 'little', got '{}'",
            bitorder_str
        )));
    }

    match axis {
        Some(ax) => {
            // Unpack along specified axis
            let ndim = packed.ndim();
            let axis_idx = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis_idx >= ndim {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let shape = packed.shape();
            let packed_axis_size = shape[axis_idx];
            let unpacked_axis_size = count.unwrap_or(packed_axis_size * 8);

            // Validate count
            if unpacked_axis_size > packed_axis_size * 8 {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "count ({}) cannot be larger than {} (8 * packed_axis_size)",
                    unpacked_axis_size,
                    packed_axis_size * 8
                )));
            }

            // Calculate new shape
            let mut new_shape = shape.clone();
            new_shape[axis_idx] = unpacked_axis_size;

            // Calculate total size
            let mut outer_size = 1;
            for i in 0..axis_idx {
                outer_size *= shape[i];
            }
            let mut inner_size = 1;
            for i in (axis_idx + 1)..ndim {
                inner_size *= shape[i];
            }

            let packed_data = packed.to_vec();
            let mut unpacked_data =
                Vec::with_capacity(outer_size * unpacked_axis_size * inner_size);

            // Unpack bits along the specified axis
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    for bit_idx in 0..unpacked_axis_size {
                        let packed_idx = bit_idx / 8;
                        let bit_offset = bit_idx % 8;

                        let flat_idx =
                            outer * packed_axis_size * inner_size + packed_idx * inner_size + inner;
                        let byte = packed_data[flat_idx];

                        let bit = if bitorder_str == "big" {
                            (byte >> (7 - bit_offset)) & 1
                        } else {
                            (byte >> bit_offset) & 1
                        };

                        unpacked_data.push(bit);
                    }
                }
            }

            Ok(Array::from_vec_shape(unpacked_data, &new_shape)?)
        }
        None => {
            // Unpack flattened array
            let packed_data = packed.to_vec();
            let n_bytes = packed_data.len();
            let n_bits = count.unwrap_or(n_bytes * 8);

            if n_bits > n_bytes * 8 {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "count ({}) cannot be larger than {} (8 * number of bytes)",
                    n_bits,
                    n_bytes * 8
                )));
            }

            let mut unpacked = Vec::with_capacity(n_bits);

            for i in 0..n_bits {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let byte = packed_data[byte_idx];

                let bit = if bitorder_str == "big" {
                    (byte >> (7 - bit_idx)) & 1
                } else {
                    (byte >> bit_idx) & 1
                };

                unpacked.push(bit);
            }

            Ok(Array::from_vec(unpacked))
        }
    }
}

/// Converts a flat index or array of flat indices into a tuple of coordinate arrays
///
/// # Parameters
///
/// * `indices` - An array of flat indices
/// * `shape` - The shape of the array into which the flat indices should be converted
/// * `order` - Order of the indices: 'C' for row-major (default) or 'F' for column-major
///
/// # Returns
///
/// Tuple of arrays, one for each dimension, containing the coordinates
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::indexing::unravel_index;
///
/// // Convert single index
/// let indices = Array::from_vec(vec![6usize]);
/// let coords = unravel_index(&indices, &[3, 4]).expect("operation should succeed");
/// assert_eq!(coords[0].to_vec(), vec![1]); // row 1
/// assert_eq!(coords[1].to_vec(), vec![2]); // col 2
///
/// // Convert multiple indices
/// let indices = Array::from_vec(vec![6usize, 11, 3, 5]);
/// let coords = unravel_index(&indices, &[3, 4]).expect("operation should succeed");
/// assert_eq!(coords[0].to_vec(), vec![1, 2, 0, 1]); // rows
/// assert_eq!(coords[1].to_vec(), vec![2, 3, 3, 1]); // cols
/// ```
pub fn unravel_index(
    indices: &Array<usize>,
    shape: &[usize],
    order: Option<&str>,
) -> Result<Vec<Array<usize>>> {
    let order_str = order.unwrap_or("C");
    if order_str != "C" && order_str != "F" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "order must be 'C' or 'F', got '{}'",
            order_str
        )));
    }

    if shape.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "shape cannot be empty".to_string(),
        ));
    }

    // Calculate total size of the array
    let total_size: usize = shape.iter().product();

    // Validate indices
    let indices_data = indices.to_vec();
    for &idx in &indices_data {
        if idx >= total_size {
            return Err(NumRs2Error::InvalidOperation(format!(
                "index {} is out of bounds for array with size {}",
                idx, total_size
            )));
        }
    }

    let n_dims = shape.len();
    let n_indices = indices_data.len();

    // Initialize coordinate arrays
    let mut coordinates: Vec<Vec<usize>> = vec![Vec::with_capacity(n_indices); n_dims];

    // Calculate strides based on order
    let mut strides = vec![1; n_dims];
    if order_str == "C" {
        // Row-major order (C-style)
        for i in (0..n_dims - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    } else {
        // Column-major order (Fortran-style)
        for i in 1..n_dims {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
    }

    // Convert each flat index to coordinates
    for &flat_idx in &indices_data {
        let mut remainder = flat_idx;

        if order_str == "C" {
            // Row-major unraveling
            for i in 0..n_dims {
                coordinates[i].push(remainder / strides[i]);
                remainder %= strides[i];
            }
        } else {
            // Column-major unraveling
            for i in 0..n_dims {
                coordinates[i].push(remainder % shape[i]);
                remainder /= shape[i];
            }
        }
    }

    // Convert coordinate vectors to Arrays
    let mut result = Vec::with_capacity(n_dims);
    for coord_vec in coordinates {
        result.push(Array::from_vec_shape(coord_vec, &indices.shape())?);
    }

    Ok(result)
}

/// Converts a tuple of coordinate arrays into an array of flat indices
///
/// # Parameters
///
/// * `multi_index` - Tuple of arrays, one array for each dimension
/// * `dims` - The shape of the array into which the indices will be converted
/// * `mode` - Specifies how out-of-bounds indices are handled:
///   - 'raise' (default): raise error
///   - 'wrap': wrap around
///   - 'clip': clip to the valid range
/// * `order` - Order of the indices: 'C' for row-major (default) or 'F' for column-major
///
/// # Returns
///
/// Array of flat indices
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::manipulation::ravel_multi_index;
///
/// // Convert single coordinate
/// let row = Array::from_vec(vec![1usize]);
/// let col = Array::from_vec(vec![2usize]);
/// let flat = ravel_multi_index(&[&row, &col], &[3, 4], Some("raise"), Some("C")).expect("operation should succeed");
/// assert_eq!(flat.to_vec(), vec![6]); // 1*4 + 2 = 6
///
/// // Convert multiple coordinates
/// let rows = Array::from_vec(vec![1usize, 2, 0, 1]);
/// let cols = Array::from_vec(vec![2usize, 3, 3, 1]);
/// let flat = ravel_multi_index(&[&rows, &cols], &[3, 4], Some("raise"), Some("C")).expect("operation should succeed");
/// assert_eq!(flat.to_vec(), vec![6, 11, 3, 5]);
/// ```
pub fn ravel_multi_index(
    multi_index: &[&Array<usize>],
    dims: &[usize],
    mode: Option<&str>,
    order: Option<&str>,
) -> Result<Array<usize>> {
    let mode_str = mode.unwrap_or("raise");
    if mode_str != "raise" && mode_str != "wrap" && mode_str != "clip" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "mode must be 'raise', 'wrap', or 'clip', got '{}'",
            mode_str
        )));
    }

    let order_str = order.unwrap_or("C");
    if order_str != "C" && order_str != "F" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "order must be 'C' or 'F', got '{}'",
            order_str
        )));
    }

    if multi_index.len() != dims.len() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "number of index arrays ({}) must match number of dimensions ({})",
            multi_index.len(),
            dims.len()
        )));
    }

    if multi_index.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "multi_index cannot be empty".to_string(),
        ));
    }

    // Check that all index arrays have the same shape
    let result_shape = multi_index[0].shape();
    for idx_array in multi_index.iter().skip(1) {
        if idx_array.shape() != result_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: result_shape.to_vec(),
                actual: idx_array.shape().to_vec(),
            });
        }
    }

    let n_indices = multi_index[0].size();
    let n_dims = dims.len();

    // Calculate strides based on order
    let mut strides = vec![1; n_dims];
    if order_str == "C" {
        // Row-major order (C-style)
        for i in (0..n_dims - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    } else {
        // Column-major order (Fortran-style)
        for i in 1..n_dims {
            strides[i] = strides[i - 1] * dims[i - 1];
        }
    }

    // Convert coordinates to flat indices
    let mut flat_indices = Vec::with_capacity(n_indices);

    // Get data from all coordinate arrays
    let coord_data: Vec<Vec<usize>> = multi_index.iter().map(|arr| arr.to_vec()).collect();

    for idx in 0..n_indices {
        let mut flat_idx = 0;

        for dim in 0..n_dims {
            let coord = coord_data[dim][idx];

            // Handle out-of-bounds coordinates based on mode
            let adjusted_coord = match mode_str {
                "raise" => {
                    if coord >= dims[dim] {
                        return Err(NumRs2Error::InvalidOperation(format!(
                            "index {} is out of bounds for axis {} with size {}",
                            coord, dim, dims[dim]
                        )));
                    }
                    coord
                }
                "wrap" => coord % dims[dim],
                "clip" => coord.min(dims[dim].saturating_sub(1)),
                // INVARIANT: unreachable. `mode_str` is validated at the
                // top of `ravel_multi_index` (`mode_str != "raise" &&
                // mode_str != "wrap" && mode_str != "clip"` returns `Err`
                // before this loop runs) and is never reassigned, so by the
                // time this `match` executes it can only be one of those
                // three strings.
                _ => unreachable!(),
            };

            flat_idx += adjusted_coord * strides[dim];
        }

        flat_indices.push(flat_idx);
    }

    Array::from_vec_shape(flat_indices, &result_shape)
}
