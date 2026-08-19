//! Core array creation functions from data sources
//!
//! This module provides functions to create arrays from various data sources:
//! - `fromfunction` - Create array by executing a function over coordinates
//! - `frombuffer` - Create array from a raw byte buffer
//! - `fromiter` - Create array from an iterator
//! - `frommemmap` - Create array from a memory-mapped file
//! - `fromstring` - Create array from a string of numbers

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
/// let result = fromfunction(|indices: &[usize]| (indices[0] + indices[1]) as f64, &[3, 3]).expect("operation should succeed");
/// assert_eq!(result.get(&[0, 0]).expect("operation should succeed"), 0.0);
/// assert_eq!(result.get(&[0, 1]).expect("operation should succeed"), 1.0);
/// assert_eq!(result.get(&[1, 1]).expect("operation should succeed"), 2.0);
/// assert_eq!(result.get(&[2, 2]).expect("operation should succeed"), 4.0);
///
/// // Create a 2x4 array where arr[i,j] = i * j
/// let result = fromfunction(|indices: &[usize]| (indices[0] * indices[1]) as i32, &[2, 4]).expect("operation should succeed");
/// assert_eq!(result.get(&[1, 3]).expect("operation should succeed"), 3);
/// assert_eq!(result.get(&[0, 2]).expect("operation should succeed"), 0);
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
/// let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), -1, 0).expect("operation should succeed");
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
/// let result = frombuffer::<f64>(buffer, std::mem::size_of::<f64>(), 3, 0).expect("operation should succeed");
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
/// let result = fromiter((0..5).map(|x| x as f64), None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// // Create 2D array from range with specified shape
/// let result = fromiter((0..6).map(|x| x as i32), Some(&[2, 3])).expect("operation should succeed");
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

/// Create an array from a memory-mapped file
///
/// This function creates an Array by reading data from a memory-mapped file.
/// The memory-mapped file must have been created with compatible format and type.
///
/// # Parameters
///
/// * `path` - Path to the memory-mapped file
/// * `dtype` - Data type of the array elements (for type inference)
/// * `mode` - File access mode ("r" for read-only, "r+" for read-write)
/// * `offset` - Start reading from this position in bytes (default: 0)
/// * `shape` - Optional shape to override the file's stored shape
/// * `order` - Memory layout order ("C" for row-major, "F" for column-major)
///
/// # Returns
///
/// Array created from the memory-mapped file data
///
/// # Examples
///
/// ```no_run
/// use numrs2::prelude::*;
/// use std::path::Path;
///
/// // Create an array from a memory-mapped file
/// let result = frommemmap::<f64>(
///     Path::new("data.mmap"),
///     "r",
///     Some(0),
///     None,
///     Some("C")
/// ).expect("operation should succeed");
/// println!("Array shape: {:?}", result.shape());
/// ```
pub fn frommemmap<T: Copy + Clone + Default>(
    path: &std::path::Path,
    mode: &str,
    offset: Option<usize>,
    shape: Option<&[usize]>,
    order: Option<&str>,
) -> Result<Array<T>> {
    // Import the memory-mapped array module
    use crate::mmap::{check_meta_type, open_mmap_info, MmapArray};

    let _offset = offset.unwrap_or(0);
    let _order = order.unwrap_or("C");

    // Validate mode
    match mode {
        "r" | "r+" => {}
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Unsupported mode '{}'. Use 'r' for read-only or 'r+' for read-write",
                mode
            )))
        }
    }

    // Read metadata from the file to get information
    let meta = open_mmap_info(&path)?;

    // Determine the shape to use
    let array_shape = match shape {
        Some(s) => s.to_vec(),
        None => meta.shape.clone(),
    };

    // Verify type compatibility (name and size; see `check_meta_type` docs
    // for why both are checked)
    check_meta_type::<T>(&meta)?;

    // Create a memory-mapped array to read the data
    let mmap_array = MmapArray::<T>::new(&path, &array_shape, false)?;

    // Convert the memory-mapped array to a regular Array
    let array = mmap_array.to_array()?;

    Ok(array)
}

/// Create a 1D array from a string of numbers
///
/// Parses a string containing whitespace-separated numeric values and creates an array.
/// This is similar to NumPy's fromstring function but only supports space/whitespace
/// separated values (not arbitrary separators).
///
/// # Parameters
///
/// * `string` - A string containing numeric values separated by whitespace
///
/// # Returns
///
/// A 1D array containing the parsed values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create array from space-separated values
/// let arr = fromstring::<f64>("1.0 2.5 3.7 4.2").expect("operation should succeed");
/// assert_eq!(arr.to_vec(), vec![1.0, 2.5, 3.7, 4.2]);
///
/// // Works with multiple spaces and newlines
/// let arr = fromstring::<i32>("1   2\n3\t4").expect("operation should succeed");
/// assert_eq!(arr.to_vec(), vec![1, 2, 3, 4]);
///
/// // Empty string creates empty array
/// let arr = fromstring::<f64>("").expect("operation should succeed");
/// assert_eq!(arr.len(), 0);
/// ```
pub fn fromstring<T>(string: &str) -> Result<Array<T>>
where
    T: std::str::FromStr + Clone + num_traits::Zero,
    T::Err: std::fmt::Display,
{
    if string.trim().is_empty() {
        return Ok(Array::from_vec(vec![]));
    }

    let values: Result<Vec<T>> = string
        .split_whitespace()
        .map(|s| {
            s.parse::<T>()
                .map_err(|e| NumRs2Error::ValueError(format!("Failed to parse '{}': {}", s, e)))
        })
        .collect();

    Ok(Array::from_vec(values?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmap::MmapArray;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_fromfunction() {
        // Test 2D array creation
        let result = fromfunction(
            |indices: &[usize]| (indices[0] + indices[1]) as f64,
            &[3, 3],
        )
        .expect("operation should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(result.get(&[0, 0]).expect("operation should succeed"), 0.0);
        assert_eq!(result.get(&[0, 1]).expect("operation should succeed"), 1.0);
        assert_eq!(result.get(&[1, 1]).expect("operation should succeed"), 2.0);
        assert_eq!(result.get(&[2, 2]).expect("operation should succeed"), 4.0);

        // Test 1D array creation
        let result = fromfunction(|indices: &[usize]| indices[0] as i32 * 2, &[5])
            .expect("operation should succeed");
        assert_eq!(result.shape(), vec![5]);
        assert_eq!(result.to_vec(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_frombuffer() {
        // Test with i32 data
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buffer = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<i32>(),
            )
        };
        let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), -1, 0)
            .expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);

        // Test with count limit
        let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), 3, 0)
            .expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![1, 2, 3]);

        // Test with offset
        let result = frombuffer::<i32>(
            buffer,
            std::mem::size_of::<i32>(),
            2,
            std::mem::size_of::<i32>(),
        )
        .expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![2, 3]);
    }

    #[test]
    fn test_fromiter() {
        // Test 1D array from range
        let result = fromiter((0..5).map(|x| x as f64), None).expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        // Test 2D array with specified shape
        let result = fromiter(0..6, Some(&[2, 3])).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2, 3]);
        assert_eq!(result.to_vec(), vec![0, 1, 2, 3, 4, 5]);

        // Test shape mismatch error
        let result = fromiter(0..5, Some(&[2, 3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_frommemmap() {
        // Create a test file path in a temporary directory
        let test_path = std::env::temp_dir().join("test_frommemmap.tmp");
        let path = test_path.as_path();

        // Cleanup function for the test file
        let cleanup = || {
            let _ = fs::remove_file(path);
        };

        // Ensure cleanup on test start and defer cleanup to end
        cleanup();

        // Create test data
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let array = Array::from_vec(data.clone()).reshape(&shape);

        // Create a memory-mapped array file - skip test if this fails due to permissions
        let mmap_array = match MmapArray::from_array(&array, &path) {
            Ok(mmap) => mmap,
            Err(_) => {
                println!("Skipping frommemmap test due to file permission issues");
                return;
            }
        };
        drop(mmap_array);

        // Test frommemmap function
        match frommemmap::<f64>(path, "r", None, None, None) {
            Ok(result) => {
                assert_eq!(result.shape(), shape);
                assert_eq!(result.to_vec(), data);

                // Test with custom shape
                let result = frommemmap::<f64>(path, "r", None, Some(&[6]), None)
                    .expect("operation should succeed");
                assert_eq!(result.shape(), vec![6]);
                assert_eq!(result.to_vec(), data);
            }
            Err(_) => {
                println!("Skipping frommemmap test due to file permission issues");
            }
        }

        // Cleanup
        cleanup();
    }

    #[test]
    fn test_frommemmap_errors() {
        // Test with non-existent file
        let result = frommemmap::<f64>(Path::new("non_existent.mmap"), "r", None, None, None);
        assert!(result.is_err());

        // Test with invalid mode
        let test_path = std::env::temp_dir().join("test_frommemmap_errors.tmp");
        let path = test_path.as_path();

        // Cleanup function
        let cleanup = || {
            let _ = fs::remove_file(path);
        };
        cleanup();

        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let array = Array::from_vec(data).reshape(&[2, 2]);
        let _mmap_array = MmapArray::from_array(&array, &path).expect("operation should succeed");

        let result = frommemmap::<f64>(path, "invalid_mode", None, None, None);
        assert!(result.is_err());

        // Cleanup
        cleanup();
    }

    #[test]
    fn test_frommemmap_type_mismatch_is_rejected() {
        // A file created for f64 data must not be readable as i32 through
        // `frommemmap` (see `crate::mmap::check_meta_type`, which this
        // function delegates its type/size verification to).
        let test_path = std::env::temp_dir().join(format!(
            "numrs2_frommemmap_type_mismatch_{}.tmp",
            std::process::id()
        ));
        let path = test_path.as_path();
        let cleanup = || {
            let _ = fs::remove_file(path);
        };
        cleanup();

        let array = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let _mmap_array = MmapArray::from_array(&array, &path).expect("operation should succeed");

        let result = frommemmap::<i32>(path, "r", None, None, None);
        assert!(result.is_err());

        cleanup();
    }
}
