//! Array creation operations
//!
//! This module provides functions for creating arrays from various sources.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

/// Construct an array by executing a function over each coordinate
///
/// # Parameters
///
/// * `function` - Function to call at each coordinate
/// * `shape` - Shape of the output array
/// * `dtype` - Data type of the output array (for type inference)
///
/// # Returns
///
/// A new array where `arr[i,j,k,...] = function(i,j,k,...)`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3x3 array where arr[i,j] = i + j
/// let result = fromfunction(|indices: &[usize]| (indices[0] + indices[1]) as f64, &[3, 3])
///     .expect("fromfunction should succeed for valid shape");
/// assert_eq!(result.get(&[0, 0]).expect("get should succeed for valid index"), 0.0);
/// assert_eq!(result.get(&[0, 1]).expect("get should succeed for valid index"), 1.0);
/// assert_eq!(result.get(&[1, 1]).expect("get should succeed for valid index"), 2.0);
/// assert_eq!(result.get(&[2, 2]).expect("get should succeed for valid index"), 4.0);
///
/// // Create a 2x4 array where arr[i,j] = i * j
/// let result = fromfunction(|indices: &[usize]| (indices[0] * indices[1]) as i32, &[2, 4])
///     .expect("fromfunction should succeed for valid shape");
/// assert_eq!(result.get(&[1, 3]).expect("get should succeed for valid index"), 3);
/// assert_eq!(result.get(&[0, 2]).expect("get should succeed for valid index"), 0);
/// ```
pub fn fromfunction<T, F>(function: F, shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + num_traits::Zero,
    F: Fn(&[usize]) -> T,
{
    if shape.is_empty() {
        return Ok(Array::from_vec(vec![]));
    }

    // Calculate total number of elements
    let total_elements: usize = shape.iter().product();

    // Create result vector
    let mut result_data = Vec::with_capacity(total_elements);

    // Iterate through all indices and compute function values
    let mut indices = vec![0; shape.len()];
    for _ in 0..total_elements {
        // Call the function with current indices
        let value = function(&indices);
        result_data.push(value);

        // Increment indices (like an odometer)
        let mut carry = true;
        for dim in (0..shape.len()).rev() {
            if carry {
                indices[dim] += 1;
                carry = indices[dim] >= shape[dim];
                if carry {
                    indices[dim] = 0;
                }
            }
        }
    }

    // Create and reshape the array
    Ok(Array::from_vec(result_data).reshape(shape))
}

/// Create an array from a raw buffer
///
/// # Parameters
///
/// * `buffer` - The raw buffer as a slice of bytes
/// * `dtype_size` - Size of each element in bytes (e.g., 4 for i32, 8 for f64)
/// * `count` - Number of elements to read from buffer (-1 means read all available)
/// * `offset` - Start reading from this position in the buffer (in bytes)
///
/// # Returns
///
/// A 1D array created from the buffer data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create array from i32 buffer
/// let data: Vec<i32> = vec![1, 2, 3, 4, 5];
/// let buffer = unsafe {
///     std::slice::from_raw_parts(
///         data.as_ptr() as *const u8,
///         data.len() * std::mem::size_of::<i32>()
///     )
/// };
/// let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), -1, 0)
///     .expect("frombuffer should succeed for valid i32 buffer");
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
///
/// // Create array from f64 buffer with count limit
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let buffer = unsafe {
///     std::slice::from_raw_parts(
///         data.as_ptr() as *const u8,
///         data.len() * std::mem::size_of::<f64>()
///     )
/// };
/// let result = frombuffer::<f64>(buffer, std::mem::size_of::<f64>(), 3, 0)
///     .expect("frombuffer should succeed for valid f64 buffer");
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn frombuffer<T: Clone + Default>(
    buffer: &[u8],
    dtype_size: usize,
    count: isize,
    offset: usize,
) -> Result<Array<T>> {
    if dtype_size == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Data type size cannot be zero".to_string(),
        ));
    }

    if offset >= buffer.len() {
        return Err(NumRs2Error::IndexOutOfBounds(format!(
            "Offset {} is beyond buffer size {}",
            offset,
            buffer.len()
        )));
    }

    if dtype_size != std::mem::size_of::<T>() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Data type size mismatch: expected {}, got {}",
            std::mem::size_of::<T>(),
            dtype_size
        )));
    }

    let available_bytes = buffer.len() - offset;
    let max_elements = available_bytes / dtype_size;

    let num_elements = if count < 0 {
        max_elements
    } else {
        let requested = count as usize;
        if requested > max_elements {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Requested {} elements but only {} available in buffer",
                requested, max_elements
            )));
        }
        requested
    };

    if num_elements == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    // Create vector by copying bytes and converting to T
    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let byte_offset = offset + i * dtype_size;
        let element_bytes = &buffer[byte_offset..byte_offset + dtype_size];

        // Safety: We've checked the size matches T and bounds are valid
        let element = unsafe { std::ptr::read(element_bytes.as_ptr() as *const T) };

        result.push(element);
    }

    Ok(Array::from_vec(result))
}

/// Create an array from an iterator
///
/// # Parameters
///
/// * `iter` - Iterator that yields elements
/// * `shape` - Optional shape for the resulting array
///
/// # Returns
///
/// Array created from the iterator elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 1D array from range
/// let result = fromiter((0..5).map(|x| x as f64), None)
///     .expect("fromiter should succeed for valid iterator");
/// assert_eq!(result.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// // Create 2D array from range with specified shape
/// let result = fromiter((0..6).map(|x| x as i32), Some(&[2, 3]))
///     .expect("fromiter should succeed for valid iterator with shape");
/// assert_eq!(result.shape(), vec![2, 3]);
/// assert_eq!(result.to_vec(), vec![0, 1, 2, 3, 4, 5]);
/// ```
pub fn fromiter<T: Clone, I: Iterator<Item = T>>(
    iter: I,
    shape: Option<&[usize]>,
) -> Result<Array<T>> {
    let data: Vec<T> = iter.collect();

    match shape {
        Some(s) => {
            let expected_size: usize = s.iter().product();
            if data.len() != expected_size {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![expected_size],
                    actual: vec![data.len()],
                });
            }
            Ok(Array::from_vec(data).reshape(s))
        }
        None => Ok(Array::from_vec(data)),
    }
}
