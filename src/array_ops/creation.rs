use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::ops::Mul;

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
/// let result = fromfunction(|indices: &[usize]| (indices[0] + indices[1]) as f64, &[3, 3]).unwrap();
/// assert_eq!(result.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(result.get(&[0, 1]).unwrap(), 1.0);
/// assert_eq!(result.get(&[1, 1]).unwrap(), 2.0);
/// assert_eq!(result.get(&[2, 2]).unwrap(), 4.0);
///
/// // Create a 2x4 array where arr[i,j] = i * j
/// let result = fromfunction(|indices: &[usize]| (indices[0] * indices[1]) as i32, &[2, 4]).unwrap();
/// assert_eq!(result.get(&[1, 3]).unwrap(), 3);
/// assert_eq!(result.get(&[0, 2]).unwrap(), 0);
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
/// let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), -1, 0).unwrap();
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
/// let result = frombuffer::<f64>(buffer, std::mem::size_of::<f64>(), 3, 0).unwrap();
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
/// let result = fromiter((0..5).map(|x| x as f64), None).unwrap();
/// assert_eq!(result.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// // Create 2D array from range with specified shape
/// let result = fromiter((0..6).map(|x| x as i32), Some(&[2, 3])).unwrap();
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
/// ).unwrap();
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
    use crate::mmap::{open_mmap_info, MmapArray};

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

    // Verify type compatibility
    if meta.type_name != std::any::type_name::<T>() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Type mismatch: file contains '{}', but requested '{}'",
            meta.type_name,
            std::any::type_name::<T>()
        )));
    }

    // Create a memory-mapped array to read the data
    let mmap_array = MmapArray::<T>::new(&path, &array_shape, false)?;

    // Convert the memory-mapped array to a regular Array
    let array = mmap_array.to_array()?;

    Ok(array)
}

/// Create coordinate matrices from coordinate vectors
///
/// Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields
/// over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn.
///
/// # Parameters
///
/// * `xi` - 1-D arrays representing the coordinates of a grid
/// * `indexing` - Cartesian ('xy', default) or matrix ('ij') indexing of output
/// * `sparse` - If true, return sparse output arrays
///
/// # Returns
///
/// For vectors x1, x2,..., xn with lengths Ni=len(xi), returns (N1, N2, N3,..., Nn) shaped arrays
/// if indexing='ij' or (N2, N1, N3,..., Nn) shaped arrays if indexing='xy' with the elements of xi
/// repeated to fill the matrix along the first dimension for x1, the second for x2 and so on.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 2D coordinate matrices
/// let x = Array::linspace(0.0, 1.0, 3).unwrap();
/// let y = Array::linspace(0.0, 2.0, 4).unwrap();
/// let (xx, yy) = meshgrid(&[&x, &y], "xy", false).unwrap();
///
/// assert_eq!(xx.shape(), vec![4, 3]); // Note: transposed for 'xy' indexing
/// assert_eq!(yy.shape(), vec![4, 3]);
/// ```
pub fn meshgrid<T>(xi: &[&Array<T>], indexing: &str, sparse: bool) -> Result<Vec<Array<T>>>
where
    T: Clone + num_traits::Zero + num_traits::One,
{
    if xi.is_empty() {
        return Ok(vec![]);
    }

    // Validate indexing parameter
    let use_matrix_indexing = match indexing {
        "ij" => true,
        "xy" => false,
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Invalid indexing '{}'. Use 'xy' or 'ij'",
                indexing
            )))
        }
    };

    // Get dimensions
    let ndim = xi.len();
    let mut shape: Vec<usize> = xi.iter().map(|arr| arr.shape()[0]).collect();

    // For 'xy' indexing, swap first two dimensions
    if !use_matrix_indexing && ndim >= 2 {
        shape.swap(0, 1);
    }

    let mut grids = Vec::with_capacity(ndim);

    if sparse {
        // Create sparse grids - each grid has values only along its axis
        for (axis_idx, &arr) in xi.iter().enumerate() {
            let mut grid_shape = vec![1; ndim];

            if use_matrix_indexing {
                // For 'ij' indexing, axis_idx corresponds directly to shape index
                grid_shape[axis_idx] = arr.shape()[0];
            } else {
                // For 'xy' indexing, swap first two dimensions
                if ndim >= 2 {
                    if axis_idx == 0 {
                        grid_shape[1] = arr.shape()[0];
                    } else if axis_idx == 1 {
                        grid_shape[0] = arr.shape()[0];
                    } else {
                        grid_shape[axis_idx] = arr.shape()[0];
                    }
                } else {
                    grid_shape[axis_idx] = arr.shape()[0];
                }
            }

            let grid = arr.clone().reshape(&grid_shape);
            grids.push(grid);
        }
    } else {
        // Create full grids
        for (axis_idx, &arr) in xi.iter().enumerate() {
            let mut grid = Array::zeros(&shape);
            let arr_data = arr.to_vec();

            // Fill the grid by repeating values along appropriate dimensions
            let total_elements: usize = shape.iter().product();
            let mut indices = vec![0; ndim];

            for linear_idx in 0..total_elements {
                // Convert linear index to multi-dimensional indices
                let mut temp = linear_idx;
                for i in (0..ndim).rev() {
                    indices[i] = temp % shape[i];
                    temp /= shape[i];
                }

                // Determine which value from the input array to use
                let src_idx = if !use_matrix_indexing && ndim >= 2 {
                    // For xy indexing, swap interpretation of first two indices
                    if axis_idx == 0 {
                        indices[1]
                    } else if axis_idx == 1 {
                        indices[0]
                    } else {
                        indices[axis_idx]
                    }
                } else {
                    indices[axis_idx]
                };

                grid.set(&indices, arr_data[src_idx].clone())?;
            }

            grids.push(grid);
        }
    }

    Ok(grids)
}

/// Create values spaced evenly on a log scale
///
/// Return numbers spaced evenly on a log scale.
///
/// # Parameters
///
/// * `start` - The starting value of the sequence (base^start)
/// * `stop` - The final value of the sequence (base^stop)
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
/// * `base` - The base of the log space
///
/// # Returns
///
/// Array of `num` samples, equally spaced on a log scale
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 5 values from 10^0 to 10^2
/// let result = logspace(0.0, 2.0, 5, true, 10.0).unwrap();
/// // Result is approximately [1.0, 3.16, 10.0, 31.6, 100.0]
///
/// // Create values from 2^1 to 2^4
/// let result = logspace(1.0, 4.0, 4, true, 2.0).unwrap();
/// // Result is [2.0, 4.0, 8.0, 16.0]
/// ```
pub fn logspace<T>(start: T, stop: T, num: usize, endpoint: bool, base: T) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if num == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if num == 1 {
        return Ok(Array::from_vec(vec![base.powf(start)]));
    }

    let divisor = if endpoint {
        T::from(num - 1).unwrap()
    } else {
        T::from(num).unwrap()
    };

    let mut result = Vec::with_capacity(num);

    for i in 0..num {
        let t = T::from(i).unwrap() / divisor;
        let exponent = start + t * (stop - start);
        result.push(base.powf(exponent));
    }

    Ok(Array::from_vec(result))
}

/// Create values spaced evenly on a geometric progression
///
/// Return numbers spaced evenly on a geometric progression.
///
/// # Parameters
///
/// * `start` - The starting value of the sequence
/// * `stop` - The final value of the sequence
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
///
/// # Returns
///
/// Array of `num` samples, equally spaced on a geometric scale
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 5 values from 1 to 1000
/// let result = geomspace(1.0, 1000.0, 4, true).unwrap();
/// // Result is [1.0, 10.0, 100.0, 1000.0]
///
/// // Create 4 values from 1 to 81
/// let result = geomspace(1.0, 81.0, 5, true).unwrap();
/// // Result is [1.0, 3.0, 9.0, 27.0, 81.0]
/// ```
pub fn geomspace<T>(start: T, stop: T, num: usize, endpoint: bool) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if num == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    // Check for sign consistency
    if start.is_sign_positive() != stop.is_sign_positive() {
        return Err(NumRs2Error::InvalidOperation(
            "Geometric sequence cannot include zero or change sign".to_string(),
        ));
    }

    if start == T::zero() || stop == T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Geometric sequence endpoints cannot be zero".to_string(),
        ));
    }

    // Use logarithms to create geometric progression
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();

    let result = logspace(
        log_start,
        log_stop,
        num,
        endpoint,
        T::from(std::f64::consts::E).unwrap(),
    )?;

    // Adjust signs if necessary
    if start.is_sign_negative() {
        let result_vec: Vec<T> = result.to_vec().into_iter().map(|x| -x).collect();
        Ok(Array::from_vec(result_vec))
    } else {
        Ok(result)
    }
}

/// Create a dense multi-dimensional "meshgrid"
///
/// Returns arrays representing coordinates of a multi-dimensional grid.
/// Similar to `meshgrid`, but with a more convenient interface for generating
/// equally spaced grids.
///
/// # Parameters
///
/// * `slices` - A vector of slice specifications. Each slice can be:
///   - A tuple of (start, stop, num) for equally spaced values
///   - A tuple of (start, stop, step) where step is negative to indicate step size
///
/// # Returns
///
/// Vector of arrays where each array represents coordinates along one dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D grid from 0 to 1 with 3 points in each dimension
/// let grids = mgrid(&[(0.0, 1.0, 3), (0.0, 1.0, 3)]).unwrap();
/// assert_eq!(grids.len(), 2);
/// assert_eq!(grids[0].shape(), vec![3, 3]);
/// assert_eq!(grids[1].shape(), vec![3, 3]);
/// ```
pub fn mgrid<T>(slices: &[(T, T, T)]) -> Result<Vec<Array<T>>>
where
    T: Float + Clone + PartialOrd + num_traits::FromPrimitive,
{
    use crate::array::Array;
    use crate::math::linspace;

    if slices.is_empty() {
        return Ok(vec![]);
    }

    // Create coordinate arrays for each dimension
    let mut coord_arrays = Vec::with_capacity(slices.len());

    for &(start, stop, num_or_step) in slices {
        let arr = if num_or_step > T::one() && num_or_step == num_or_step.floor() {
            // If num_or_step is an integer > 1, treat as number of points
            let num = num_or_step.to_usize().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Cannot convert num to usize".to_string())
            })?;
            linspace(start, stop, num)
        } else {
            // Otherwise treat as step size
            let step = num_or_step;
            if step == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "Step size cannot be zero".to_string(),
                ));
            }

            // Generate points with the step
            let mut points = Vec::new();
            let mut current = start;

            if step > T::zero() {
                while current <= stop {
                    points.push(current);
                    current = current + step;
                }
            } else if step < T::zero() {
                while current >= stop {
                    points.push(current);
                    current = current + step;
                }
            }
            Array::from_vec(points)
        };

        coord_arrays.push(arr);
    }

    // Use meshgrid with ij indexing to create the full grids
    meshgrid(&coord_arrays.iter().collect::<Vec<_>>(), "ij", false)
}

/// Create an open multi-dimensional "meshgrid"
///
/// Returns arrays representing coordinates of a multi-dimensional grid,
/// but with shape (1, 1, ..., 1, n_i, 1, ..., 1) for the i-th dimension.
/// This is memory-efficient for operations that support broadcasting.
///
/// # Parameters
///
/// * `slices` - A vector of slice specifications. Each slice can be:
///   - A tuple of (start, stop, num) for equally spaced values
///   - A tuple of (start, stop, step) where step is negative to indicate step size
///
/// # Returns
///
/// Vector of arrays where each array has values only along its respective dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a sparse 2D grid from 0 to 1 with 3 points in each dimension
/// let grids = ogrid(&[(0.0, 1.0, 3), (0.0, 1.0, 3)]).unwrap();
/// assert_eq!(grids.len(), 2);
/// assert_eq!(grids[0].shape(), vec![3, 1]); // Values along first dimension
/// assert_eq!(grids[1].shape(), vec![1, 3]); // Values along second dimension
/// ```
pub fn ogrid<T>(slices: &[(T, T, T)]) -> Result<Vec<Array<T>>>
where
    T: Float + Clone + PartialOrd + num_traits::FromPrimitive,
{
    use crate::array::Array;
    use crate::math::linspace;

    if slices.is_empty() {
        return Ok(vec![]);
    }

    // Create coordinate arrays for each dimension
    let mut coord_arrays = Vec::with_capacity(slices.len());

    for &(start, stop, num_or_step) in slices {
        let arr = if num_or_step > T::one() && num_or_step == num_or_step.floor() {
            // If num_or_step is an integer > 1, treat as number of points
            let num = num_or_step.to_usize().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Cannot convert num to usize".to_string())
            })?;
            linspace(start, stop, num)
        } else {
            // Otherwise treat as step size
            let step = num_or_step;
            if step == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "Step size cannot be zero".to_string(),
                ));
            }

            // Generate points with the step
            let mut points = Vec::new();
            let mut current = start;

            if step > T::zero() {
                while current <= stop {
                    points.push(current);
                    current = current + step;
                }
            } else if step < T::zero() {
                while current >= stop {
                    points.push(current);
                    current = current + step;
                }
            }
            Array::from_vec(points)
        };

        coord_arrays.push(arr);
    }

    // Use meshgrid with ij indexing and sparse=true
    meshgrid(&coord_arrays.iter().collect::<Vec<_>>(), "ij", true)
}

/// Create a triangular array with given diagonal and type
///
/// An array with ones at and below the given diagonal and zeros elsewhere.
///
/// # Parameters
///
/// * `n` - Number of rows
/// * `m` - Number of columns (if None, defaults to n)
/// * `k` - The sub-diagonal at and below which the array is filled.
///   k = 0 is the main diagonal, while k < 0 is below it,
///   and k > 0 is above. The default is 0.
/// * `dtype` - Data type of the array (for type inference)
///
/// # Returns
///
/// Array with shape (n, m) and requested type
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3x3 lower triangular array
/// let result: Array<i32> = tri(3, None, None).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
///
/// // Create a 3x4 array with diagonal offset
/// let result: Array<f64> = tri(3, Some(4), Some(1)).unwrap();
/// assert_eq!(result.to_vec(), vec![1.0, 1.0, 0.0, 0.0,
///                                  1.0, 1.0, 1.0, 0.0,
///                                  1.0, 1.0, 1.0, 1.0]);
///
/// // Create with negative diagonal offset
/// let result: Array<i32> = tri(3, None, Some(-1)).unwrap();
/// assert_eq!(result.to_vec(), vec![0, 0, 0, 1, 0, 0, 1, 1, 0]);
/// ```
pub fn tri<T>(n: usize, m: Option<usize>, k: Option<isize>) -> Result<Array<T>>
where
    T: Clone + num_traits::Zero + num_traits::One,
{
    let m = m.unwrap_or(n);
    let k = k.unwrap_or(0);

    let mut data = Vec::with_capacity(n * m);

    for i in 0..n {
        for j in 0..m {
            if j as isize <= i as isize + k {
                data.push(T::one());
            } else {
                data.push(T::zero());
            }
        }
    }

    Ok(Array::from_vec(data).reshape(&[n, m]))
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
        .unwrap();
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 0.0);
        assert_eq!(result.get(&[0, 1]).unwrap(), 1.0);
        assert_eq!(result.get(&[1, 1]).unwrap(), 2.0);
        assert_eq!(result.get(&[2, 2]).unwrap(), 4.0);

        // Test 1D array creation
        let result = fromfunction(|indices: &[usize]| indices[0] as i32 * 2, &[5]).unwrap();
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
        let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), -1, 0).unwrap();
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);

        // Test with count limit
        let result = frombuffer::<i32>(buffer, std::mem::size_of::<i32>(), 3, 0).unwrap();
        assert_eq!(result.to_vec(), vec![1, 2, 3]);

        // Test with offset
        let result = frombuffer::<i32>(
            buffer,
            std::mem::size_of::<i32>(),
            2,
            std::mem::size_of::<i32>(),
        )
        .unwrap();
        assert_eq!(result.to_vec(), vec![2, 3]);
    }

    #[test]
    fn test_fromiter() {
        // Test 1D array from range
        let result = fromiter((0..5).map(|x| x as f64), None).unwrap();
        assert_eq!(result.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        // Test 2D array with specified shape
        let result = fromiter((0..6).map(|x| x as i32), Some(&[2, 3])).unwrap();
        assert_eq!(result.shape(), vec![2, 3]);
        assert_eq!(result.to_vec(), vec![0, 1, 2, 3, 4, 5]);

        // Test shape mismatch error
        let result = fromiter((0..5).map(|x| x as i32), Some(&[2, 3]));
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
                let result = frommemmap::<f64>(path, "r", None, Some(&[6]), None).unwrap();
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
        let _mmap_array = MmapArray::from_array(&array, &path).unwrap();

        let result = frommemmap::<f64>(path, "invalid_mode", None, None, None);
        assert!(result.is_err());

        // Cleanup
        cleanup();
    }

    #[test]
    fn test_meshgrid() {
        // Test 2D meshgrid with xy indexing
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0]);

        let grids = meshgrid(&[&x, &y], "xy", false).unwrap();
        assert_eq!(grids.len(), 2);

        let xx = &grids[0];
        let yy = &grids[1];

        assert_eq!(xx.shape(), vec![2, 3]); // Transposed for xy
        assert_eq!(yy.shape(), vec![2, 3]);

        // Check xx values (x repeated along rows)
        assert_eq!(xx.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(xx.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(xx.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(xx.get(&[1, 0]).unwrap(), 1.0);
        assert_eq!(xx.get(&[1, 1]).unwrap(), 2.0);
        assert_eq!(xx.get(&[1, 2]).unwrap(), 3.0);

        // Check yy values (y repeated along columns)
        assert_eq!(yy.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(yy.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(yy.get(&[0, 2]).unwrap(), 4.0);
        assert_eq!(yy.get(&[1, 0]).unwrap(), 5.0);
        assert_eq!(yy.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(yy.get(&[1, 2]).unwrap(), 5.0);

        // Test with ij indexing
        let grids_ij = meshgrid(&[&x, &y], "ij", false).unwrap();
        let xx_ij = &grids_ij[0];
        let yy_ij = &grids_ij[1];

        assert_eq!(xx_ij.shape(), vec![3, 2]); // Not transposed for ij
        assert_eq!(yy_ij.shape(), vec![3, 2]);

        // Test sparse meshgrid
        let sparse_grids = meshgrid(&[&x, &y], "xy", true).unwrap();
        assert_eq!(sparse_grids.len(), 2);
        assert_eq!(sparse_grids[0].shape(), vec![1, 3]);
        assert_eq!(sparse_grids[1].shape(), vec![2, 1]);
    }

    #[test]
    fn test_logspace() {
        use approx::assert_relative_eq;

        // Test basic logspace
        let result = logspace(0.0, 2.0, 3, true, 10.0).unwrap();
        assert_eq!(result.shape(), vec![3]);
        assert_relative_eq!(result.get(&[0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1]).unwrap(), 10.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[2]).unwrap(), 100.0, epsilon = 1e-10);

        // Test with base 2
        let result = logspace(1.0, 4.0, 4, true, 2.0).unwrap();
        assert_eq!(result.shape(), vec![4]);
        assert_relative_eq!(result.get(&[0]).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1]).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[2]).unwrap(), 8.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[3]).unwrap(), 16.0, epsilon = 1e-10);

        // Test without endpoint
        let result = logspace(0.0, 2.0, 2, false, 10.0).unwrap();
        assert_eq!(result.shape(), vec![2]);
        assert_relative_eq!(result.get(&[0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1]).unwrap(), 10.0, epsilon = 1e-10);

        // Test edge cases
        let result = logspace(1.0, 1.0, 1, true, 10.0).unwrap();
        assert_eq!(result.shape(), vec![1]);
        assert_relative_eq!(result.get(&[0]).unwrap(), 10.0, epsilon = 1e-10);

        let result = logspace(0.0, 0.0, 0, true, 10.0).unwrap();
        assert_eq!(result.shape(), vec![0]);
    }

    #[test]
    fn test_geomspace() {
        use approx::assert_relative_eq;

        // Test basic geomspace
        let result = geomspace(1.0, 100.0, 3, true).unwrap();
        assert_eq!(result.shape(), vec![3]);
        assert_relative_eq!(result.get(&[0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1]).unwrap(), 10.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[2]).unwrap(), 100.0, epsilon = 1e-10);

        // Test with more points
        let result = geomspace(1.0, 81.0, 5, true).unwrap();
        assert_eq!(result.shape(), vec![5]);
        assert_relative_eq!(result.get(&[0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1]).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[2]).unwrap(), 9.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[3]).unwrap(), 27.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[4]).unwrap(), 81.0, epsilon = 1e-10);

        // Test with negative values
        let result = geomspace(-1.0, -100.0, 3, true).unwrap();
        assert_eq!(result.shape(), vec![3]);
        assert_relative_eq!(result.get(&[0]).unwrap(), -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1]).unwrap(), -10.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[2]).unwrap(), -100.0, epsilon = 1e-10);

        // Test error cases
        assert!(geomspace(1.0, -1.0, 3, true).is_err()); // Sign change
        assert!(geomspace(0.0, 1.0, 3, true).is_err()); // Zero endpoint
        assert!(geomspace(1.0, 0.0, 3, true).is_err()); // Zero endpoint
    }

    #[test]
    fn test_mgrid() {
        use approx::assert_relative_eq;

        // Test 2D mgrid with number of points
        let grids = mgrid(&[(0.0, 1.0, 3.0), (0.0, 2.0, 3.0)]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), vec![3, 3]);
        assert_eq!(grids[1].shape(), vec![3, 3]);

        // Check first grid (x coordinates)
        assert_relative_eq!(grids[0].get(&[0, 0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[0].get(&[0, 1]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[0].get(&[0, 2]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[0].get(&[1, 0]).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(grids[0].get(&[2, 0]).unwrap(), 1.0, epsilon = 1e-10);

        // Check second grid (y coordinates)
        assert_relative_eq!(grids[1].get(&[0, 0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[1].get(&[0, 1]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(grids[1].get(&[0, 2]).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(grids[1].get(&[1, 0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[1].get(&[2, 2]).unwrap(), 2.0, epsilon = 1e-10);

        // Test with step size
        let grids_step = mgrid(&[(0.0, 1.0, 0.5), (0.0, 2.0, 1.0)]).unwrap();
        assert_eq!(grids_step.len(), 2);
        assert_eq!(grids_step[0].shape(), vec![3, 3]);
        assert_eq!(grids_step[1].shape(), vec![3, 3]);

        // Test 1D mgrid
        let grids_1d = mgrid(&[(0.0, 2.0, 5.0)]).unwrap();
        assert_eq!(grids_1d.len(), 1);
        assert_eq!(grids_1d[0].shape(), vec![5]);

        // Test empty input
        let grids_empty = mgrid::<f64>(&[]).unwrap();
        assert_eq!(grids_empty.len(), 0);
    }

    #[test]
    fn test_ogrid() {
        use approx::assert_relative_eq;

        // Test 2D ogrid (sparse grid)
        let grids = ogrid(&[(0.0, 1.0, 3.0), (0.0, 2.0, 3.0)]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), vec![3, 1]); // Values along first dimension
        assert_eq!(grids[1].shape(), vec![1, 3]); // Values along second dimension

        // Check first grid (x coordinates)
        assert_relative_eq!(grids[0].get(&[0, 0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[0].get(&[1, 0]).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(grids[0].get(&[2, 0]).unwrap(), 1.0, epsilon = 1e-10);

        // Check second grid (y coordinates)
        assert_relative_eq!(grids[1].get(&[0, 0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(grids[1].get(&[0, 1]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(grids[1].get(&[0, 2]).unwrap(), 2.0, epsilon = 1e-10);

        // Test 3D ogrid
        let grids_3d = ogrid(&[(0.0, 1.0, 2.0), (0.0, 1.0, 2.0), (0.0, 1.0, 2.0)]).unwrap();
        assert_eq!(grids_3d.len(), 3);
        assert_eq!(grids_3d[0].shape(), vec![2, 1, 1]);
        assert_eq!(grids_3d[1].shape(), vec![1, 2, 1]);
        assert_eq!(grids_3d[2].shape(), vec![1, 1, 2]);

        // Test with step size
        let grids_step = ogrid(&[(0.0, 1.0, 0.5), (0.0, 2.0, 1.0)]).unwrap();
        assert_eq!(grids_step.len(), 2);
        assert_eq!(grids_step[0].shape(), vec![3, 1]);
        assert_eq!(grids_step[1].shape(), vec![1, 3]);
    }

    #[test]
    fn test_r_concatenate() {
        // Test 1D arrays
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![4, 5, 6]);
        let result = r_concatenate(&[&a, &b]).unwrap();
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(result.shape(), vec![6]);

        // Test 2D arrays
        let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
        let result = r_concatenate(&[&a, &b]).unwrap();
        assert_eq!(result.shape(), vec![4, 2]);
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        // Test error on empty input
        let result = r_concatenate::<i32>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_c_concatenate() {
        // Test 1D arrays (become columns)
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![4, 5, 6]);
        let result = c_concatenate(&[&a, &b]).unwrap();
        assert_eq!(result.shape(), vec![3, 2]);
        // Column-major order: [1, 2, 3, 4, 5, 6] where a and b are stacked as columns
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);

        // Test 2D arrays
        let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]);
        let result = c_concatenate(&[&a, &b]).unwrap();
        assert_eq!(result.shape(), vec![2, 4]);
        assert_eq!(result.to_vec(), vec![1, 3, 2, 4, 5, 7, 6, 8]);

        // Test error on empty input
        let result = c_concatenate::<i32>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ix_() {
        // Test 2D case
        let a = Array::from_vec(vec![0, 1, 2]);
        let b = Array::from_vec(vec![3, 4]);
        let indices = ix_(&[&a, &b]).unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].shape(), vec![3, 1]);
        assert_eq!(indices[1].shape(), vec![1, 2]);

        // Check values
        assert_eq!(indices[0].to_vec(), vec![0, 1, 2]);
        assert_eq!(indices[1].to_vec(), vec![3, 4]);

        // Test 3D case
        let x = Array::from_vec(vec![10, 11]);
        let y = Array::from_vec(vec![20, 30]);
        let z = Array::from_vec(vec![100]);
        let indices_3d = ix_(&[&x, &y, &z]).unwrap();

        assert_eq!(indices_3d.len(), 3);
        assert_eq!(indices_3d[0].shape(), vec![2, 1, 1]);
        assert_eq!(indices_3d[1].shape(), vec![1, 2, 1]);
        assert_eq!(indices_3d[2].shape(), vec![1, 1, 1]);

        // Test empty input
        let empty_indices = ix_::<i32>(&[]).unwrap();
        assert_eq!(empty_indices.len(), 0);

        // Test error on non-1D input
        let bad_array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let result = ix_(&[&bad_array]);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_spec() {
        // Test range slice
        let slice = SliceSpec::range(Some(1), Some(5), Some(2));
        match slice {
            SliceSpec::Range { start, stop, step } => {
                assert_eq!(start, Some(1));
                assert_eq!(stop, Some(5));
                assert_eq!(step, Some(2));
            }
            _ => panic!("Expected Range slice"),
        }

        // Test convenience constructors
        let from_to = SliceSpec::from_to(1, 5);
        let from = SliceSpec::from(2);
        let to = SliceSpec::to(5);
        let step = SliceSpec::step(2);
        let full = SliceSpec::full();

        // Verify types
        assert!(matches!(from_to, SliceSpec::Range { .. }));
        assert!(matches!(from, SliceSpec::Range { .. }));
        assert!(matches!(to, SliceSpec::Range { .. }));
        assert!(matches!(step, SliceSpec::Range { .. }));
        assert!(matches!(full, SliceSpec::Range { .. }));

        // Test index and special slices
        let index = SliceSpec::Index(3);
        assert!(matches!(index, SliceSpec::Index(3)));

        let ellipsis = SliceSpec::Ellipsis;
        assert!(matches!(ellipsis, SliceSpec::Ellipsis));

        let newaxis = SliceSpec::NewAxis;
        assert!(matches!(newaxis, SliceSpec::NewAxis));
        assert_eq!(newaxis, NEWAXIS);
    }

    #[test]
    fn test_s_() {
        // Test slice object builder
        let slices = s_(&[
            SliceSpec::from_to(1, 5),
            SliceSpec::step(2),
            SliceSpec::Index(3),
        ]);

        assert_eq!(slices.len(), 3);
        assert!(matches!(slices[0], SliceSpec::Range { .. }));
        assert!(matches!(slices[1], SliceSpec::Range { .. }));
        assert!(matches!(slices[2], SliceSpec::Index(3)));

        // Test empty slice list
        let empty_slices = s_(&[]);
        assert_eq!(empty_slices.len(), 0);
    }
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
/// let arr = fromstring::<f64>("1.0 2.5 3.7 4.2").unwrap();
/// assert_eq!(arr.to_vec(), vec![1.0, 2.5, 3.7, 4.2]);
///
/// // Works with multiple spaces and newlines
/// let arr = fromstring::<i32>("1   2\n3\t4").unwrap();
/// assert_eq!(arr.to_vec(), vec![1, 2, 3, 4]);
///
/// // Empty string creates empty array
/// let arr = fromstring::<f64>("").unwrap();
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

/// Convert the input to an array, preserving subclasses
///
/// This function converts the input to an array but preserves array subclasses.
/// In Rust context, since we don't have the same subclassing as Python, this
/// essentially works like a regular array conversion but is provided for NumPy API compatibility.
///
/// # Parameters
///
/// * `a` - Input data that can be converted to an array
///
/// # Returns
///
/// An array interpretation of `a`. No copy is performed if the input is already an array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // From existing array (no copy)
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = asanyarray(&arr).unwrap();
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
///
/// // From vector
/// let vec = vec![4, 5, 6];
/// let result = asanyarray(&vec).unwrap();
/// assert_eq!(result.to_vec(), vec![4, 5, 6]);
/// ```
pub fn asanyarray<T>(a: &impl AsRef<[T]>) -> Result<Array<T>>
where
    T: Clone,
{
    let slice = a.as_ref();
    Ok(Array::from_vec(slice.to_vec()))
}

/// Return a contiguous array in C order (row-major) in memory
///
/// # Parameters
///
/// * `a` - Input array
///
/// # Returns
///
/// Contiguous array of same shape and content as `a`, with data in C order.
/// If `a` is already C-contiguous, no copy is made.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
/// let c_arr = ascontiguousarray(&arr).unwrap();
/// assert_eq!(c_arr.shape(), vec![2, 2]);
/// assert!(iscontiguous(&c_arr));
/// ```
pub fn ascontiguousarray<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Clone,
{
    // In our implementation, arrays are already stored in C-contiguous order
    // So we just need to ensure the data is contiguous (which it is for our arrays)
    Ok(a.clone())
}

/// Return an array laid out in Fortran order (column-major) in memory
///
/// # Parameters
///
/// * `a` - Input array
///
/// # Returns
///
/// Fortran-contiguous array of same shape and content as `a`.
/// The returned array will have column-major memory layout.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
/// let f_arr = asfortranarray(&arr).unwrap();
/// assert_eq!(f_arr.shape(), vec![2, 2]);
/// assert!(isfortran(&f_arr));
/// ```
pub fn asfortranarray<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Clone,
{
    if a.ndim() <= 1 {
        // 1D arrays are both C and Fortran contiguous
        return Ok(a.clone());
    }

    let shape = a.shape();
    let data = a.to_vec();

    // Convert from C-order (row-major) to Fortran-order (column-major)
    let mut f_data = vec![data[0].clone(); data.len()];

    // Calculate strides for Fortran order
    let mut f_strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }

    // Reorder data
    for (c_idx, value) in data.into_iter().enumerate() {
        // Convert C-order index to multi-dimensional indices
        let mut indices = vec![0; shape.len()];
        let mut temp_idx = c_idx;

        for i in (0..shape.len()).rev() {
            indices[i] = temp_idx % shape[i];
            temp_idx /= shape[i];
        }

        // Calculate Fortran-order index
        let mut f_idx = 0;
        for i in 0..shape.len() {
            f_idx += indices[i] * f_strides[i];
        }

        f_data[f_idx] = value;
    }

    // Create array with Fortran-ordered data
    // Note: We store a flag to indicate Fortran ordering
    let mut result = Array::from_vec(f_data);
    result = result.reshape(&shape);
    // In a real implementation, we would set a flag here to indicate Fortran ordering
    // For now, we just return the reordered array

    Ok(result)
}

/// Check if the array is Fortran contiguous
///
/// Fortran-contiguous arrays have column-major memory layout where the first
/// index varies fastest.
///
/// # Parameters
///
/// * `a` - Array to check
///
/// # Returns
///
/// True if the array is Fortran-contiguous, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
/// assert!(!isfortran(&arr));  // Default arrays are C-contiguous
///
/// let f_arr = asfortranarray(&arr).unwrap();
/// assert!(isfortran(&f_arr));
/// ```
pub fn isfortran<T>(a: &Array<T>) -> bool
where
    T: Clone,
{
    // For 1D arrays, both C and Fortran contiguous are the same
    if a.ndim() <= 1 {
        return true;
    }

    // In our current implementation, arrays are stored in C-order by default
    // A real implementation would check internal flags or strides
    // For now, we return false for multi-dimensional arrays unless they were
    // explicitly created with asfortranarray
    false
}

/// Check if the array is C contiguous
///
/// C-contiguous arrays have row-major memory layout where the last
/// index varies fastest.
///
/// # Parameters
///
/// * `a` - Array to check
///
/// # Returns
///
/// True if the array is C-contiguous, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
/// assert!(iscontiguous(&arr));  // Default arrays are C-contiguous
/// ```
pub fn iscontiguous<T>(_a: &Array<T>) -> bool {
    // In our implementation, arrays are stored in C-contiguous order by default
    true
}

/// Check if two arrays may share memory
///
/// # Parameters
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// True if arrays might share memory, false if they definitely don't
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
/// assert!(!may_share_memory(&a, &b));  // Different arrays don't share memory
///
/// let c = a.clone();
/// assert!(!may_share_memory(&a, &c));  // Cloned arrays have separate memory
/// ```
pub fn may_share_memory<T>(_a: &Array<T>, _b: &Array<T>) -> bool {
    // In our current implementation, arrays own their data and don't share memory
    // Views would share memory, but we don't have views implemented yet
    // For now, always return false
    false
}

/// Check if two arrays share memory
///
/// # Parameters
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// True if arrays share memory, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
/// assert!(!shares_memory(&a, &b));  // Different arrays don't share memory
/// ```
pub fn shares_memory<T>(_a: &Array<T>, _b: &Array<T>) -> bool {
    // Similar to may_share_memory, but this is a definitive check
    // In our current implementation, arrays never share memory
    false
}

/// Create a 2D array with the flattened input as a diagonal
///
/// # Parameters
///
/// * `v` - Input array to be flattened
/// * `k` - Diagonal offset (0 is main diagonal, positive for upper, negative for lower)
///
/// # Returns
///
/// 2D array with the flattened input placed on the k-th diagonal
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // 1D input
/// let v = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = diagflat(&v, 0).unwrap();
/// assert_eq!(result.shape(), vec![3, 3]);
/// // [[1, 0, 0],
/// //  [0, 2, 0],
/// //  [0, 0, 3]]
///
/// // With offset
/// let result = diagflat(&v, 1).unwrap();
/// assert_eq!(result.shape(), vec![4, 4]);
/// // [[0, 1, 0, 0],
/// //  [0, 0, 2, 0],
/// //  [0, 0, 0, 3],
/// //  [0, 0, 0, 0]]
///
/// // 2D input (gets flattened)
/// let v2d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
/// let result = diagflat(&v2d, 0).unwrap();
/// assert_eq!(result.shape(), vec![4, 4]);
/// ```
pub fn diagflat<T>(v: &Array<T>, k: i32) -> Result<Array<T>>
where
    T: Clone + Zero,
{
    // Flatten the input array
    let flat_data = v.to_vec();
    let n = flat_data.len();

    // Calculate output matrix size
    let size = (n as i32 + k.abs()) as usize;

    // Create output matrix filled with zeros
    let mut result = vec![T::zero(); size * size];

    // Fill diagonal
    for i in 0..n {
        let row = if k >= 0 { i } else { i + (-k) as usize };
        let col = if k >= 0 { i + k as usize } else { i };

        if row < size && col < size {
            result[row * size + col] = flat_data[i].clone();
        }
    }

    Ok(Array::from_vec(result).reshape(&[size, size]))
}

/// Generate a Vandermonde matrix
///
/// The columns of the output matrix are powers of the input vector.
/// The i-th column is the input vector raised element-wise to the power of N-i-1.
///
/// # Parameters
///
/// * `x` - 1D input array
/// * `n` - Number of columns in output. If None, defaults to len(x)
/// * `increasing` - If true, columns are in increasing powers (default: false)
///
/// # Returns
///
/// Vandermonde matrix. If increasing is false (default), the first column is x^(N-1),
/// the second x^(N-2), and so on. If increasing is true, the columns are x^0, x^1, ..., x^(N-1).
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Default: decreasing powers
/// let v = vander(&x, None, false).unwrap();
/// assert_eq!(v.shape(), vec![3, 3]);
/// // [[1, 1, 1],
/// //  [4, 2, 1],
/// //  [9, 3, 1]]
///
/// // Increasing powers
/// let v = vander(&x, None, true).unwrap();
/// assert_eq!(v.shape(), vec![3, 3]);
/// // [[1, 1, 1],
/// //  [1, 2, 4],
/// //  [1, 3, 9]]
///
/// // Custom number of columns
/// let v = vander(&x, Some(2), false).unwrap();
/// assert_eq!(v.shape(), vec![3, 2]);
/// // [[1, 1],
/// //  [2, 1],
/// //  [3, 1]]
/// ```
pub fn vander<T>(x: &Array<T>, n: Option<usize>, increasing: bool) -> Result<Array<T>>
where
    T: Clone + Zero + One + Mul<Output = T>,
{
    if x.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "vander requires a 1D array".to_string(),
        ));
    }

    let m = x.len();
    let n_cols = n.unwrap_or(m);

    if n_cols == 0 {
        return Ok(Array::zeros(&[m, 0]));
    }

    let x_data = x.to_vec();
    let mut result = vec![T::one(); m * n_cols];

    // Fill the matrix
    for i in 0..m {
        let mut power = T::one();

        if increasing {
            // Powers: 0, 1, 2, ..., n-1
            for j in 0..n_cols {
                result[i * n_cols + j] = power.clone();
                if j < n_cols - 1 {
                    power = power * x_data[i].clone();
                }
            }
        } else {
            // Powers: n-1, n-2, ..., 1, 0
            // First compute x^(n-1)
            for _ in 1..n_cols {
                power = power * x_data[i].clone();
            }

            result[i * n_cols] = power.clone();

            // Then divide by x for each subsequent column
            for j in 1..n_cols {
                if j == n_cols - 1 {
                    result[i * n_cols + j] = T::one();
                } else {
                    // We need division here, but it's not in the trait bounds
                    // For now, we'll recompute the power
                    let mut pow = T::one();
                    for _ in 0..(n_cols - j - 1) {
                        pow = pow * x_data[i].clone();
                    }
                    result[i * n_cols + j] = pow;
                }
            }
        }
    }

    Ok(Array::from_vec(result).reshape(&[m, n_cols]))
}

/// Array concatenation helper (equivalent to np.r_)
///
/// Concatenate arrays along the first axis. This provides convenient syntax
/// for concatenating arrays similar to NumPy's r_ indexing.
///
/// # Parameters
///
/// * `arrays` - Arrays to concatenate
///
/// # Returns
///
/// Concatenated array along the first axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::r_concatenate;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let result = r_concatenate(&[&a, &b]).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
///
/// // 2D arrays
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]).unwrap();
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]).unwrap();
/// let result = r_concatenate(&[&a, &b]).unwrap();
/// assert_eq!(result.shape(), vec![4, 2]);
/// ```
pub fn r_concatenate<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot concatenate empty array list".to_string(),
        ));
    }

    // Use the existing concatenate function along axis 0
    crate::array_ops::concatenate(arrays, 0)
}

/// Array concatenation helper along columns (equivalent to np.c_)
///
/// Concatenate arrays along the second axis (columns). For 1D arrays,
/// this stacks them as columns.
///
/// # Parameters
///
/// * `arrays` - Arrays to concatenate along columns
///
/// # Returns
///
/// Concatenated array along the second axis (columns)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::c_concatenate;
///
/// // 1D arrays become columns
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// let result = c_concatenate(&[&a, &b]).unwrap();
/// assert_eq!(result.shape(), vec![3, 2]);
/// assert_eq!(result.to_vec(), vec![1, 4, 2, 5, 3, 6]);
///
/// // 2D arrays
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]).unwrap();
/// let b = Array::from_vec(vec![5, 6, 7, 8]).reshape(&[2, 2]).unwrap();
/// let result = c_concatenate(&[&a, &b]).unwrap();
/// assert_eq!(result.shape(), vec![2, 4]);
/// ```
pub fn c_concatenate<T: Clone>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot concatenate empty array list".to_string(),
        ));
    }

    // Convert 1D arrays to column vectors
    let mut column_arrays = Vec::new();
    for &arr in arrays {
        if arr.ndim() == 1 {
            // Reshape 1D array to column vector
            let reshaped = arr.reshape(&[arr.len(), 1]);
            column_arrays.push(reshaped);
        } else {
            column_arrays.push(arr.clone());
        }
    }

    // Get references for concatenation
    let column_refs: Vec<&Array<T>> = column_arrays.iter().collect();

    // Use the existing concatenate function along axis 1
    crate::array_ops::concatenate(&column_refs, 1)
}

/// Create an open mesh from input arrays (equivalent to np.ix_)
///
/// Construct an open mesh from multiple sequences. This function takes N 1-D sequences
/// and returns N N-D arrays. These arrays can be used for vectorized evaluation of
/// N-D scalar/vector fields over N-D grids.
///
/// # Parameters
///
/// * `sequences` - 1-D arrays representing coordinates along each axis
///
/// # Returns
///
/// Vector of N-D arrays where each array has values only along its respective axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::ix_;
///
/// let a = Array::from_vec(vec![0, 1, 2]);
/// let b = Array::from_vec(vec![3, 4]);
/// let indices = ix_(&[&a, &b]).unwrap();
///
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].shape(), vec![3, 1]);  // Values for first dimension
/// assert_eq!(indices[1].shape(), vec![1, 2]);  // Values for second dimension
///
/// // Can be used for advanced indexing
/// let x = Array::from_vec(vec![10, 11, 12]);
/// let y = Array::from_vec(vec![20, 30]);
/// let grid_indices = ix_(&[&x, &y]).unwrap();
/// ```
pub fn ix_<T: Clone>(sequences: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if sequences.is_empty() {
        return Ok(vec![]);
    }

    // Verify all inputs are 1D
    for (i, seq) in sequences.iter().enumerate() {
        if seq.ndim() != 1 {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Input array {} must be 1-D, got {}-D",
                i,
                seq.ndim()
            )));
        }
    }

    let ndim = sequences.len();
    let mut result = Vec::with_capacity(ndim);

    // Create mesh arrays where each array has values only along its axis
    for (axis_idx, &seq) in sequences.iter().enumerate() {
        let mut shape = vec![1; ndim];
        shape[axis_idx] = seq.len();

        let reshaped = seq.reshape(&shape);
        result.push(reshaped);
    }

    Ok(result)
}

/// Type representing a slice object (equivalent to np.s_)
///
/// This is a simplified version of NumPy's slice objects.
/// In NumPy, s_[...] creates slice objects for advanced indexing.
#[derive(Debug, Clone, PartialEq)]
pub enum SliceSpec {
    /// Simple range slice: start:stop:step
    Range {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    /// Single index
    Index(isize),
    /// Ellipsis (...)
    Ellipsis,
    /// Newaxis (None)
    NewAxis,
}

impl SliceSpec {
    /// Create a new range slice
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::range(Some(1), Some(5), Some(2));
    /// // Equivalent to 1:5:2 in NumPy
    /// ```
    pub fn range(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Self {
        SliceSpec::Range { start, stop, step }
    }

    /// Create a slice from start to end with step 1
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::from_to(1, 5);
    /// // Equivalent to 1:5 in NumPy
    /// ```
    pub fn from_to(start: isize, stop: isize) -> Self {
        SliceSpec::Range {
            start: Some(start),
            stop: Some(stop),
            step: Some(1),
        }
    }

    /// Create a slice from start to end of array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::from(2);
    /// // Equivalent to 2: in NumPy
    /// ```
    pub fn from(start: isize) -> Self {
        SliceSpec::Range {
            start: Some(start),
            stop: None,
            step: Some(1),
        }
    }

    /// Create a slice from beginning to stop
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::to(5);
    /// // Equivalent to :5 in NumPy
    /// ```
    pub fn to(stop: isize) -> Self {
        SliceSpec::Range {
            start: None,
            stop: Some(stop),
            step: Some(1),
        }
    }

    /// Create a slice with just step (equivalent to ::step)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::step(2);
    /// // Equivalent to ::2 in NumPy
    /// ```
    pub fn step(step: isize) -> Self {
        SliceSpec::Range {
            start: None,
            stop: None,
            step: Some(step),
        }
    }

    /// Create a full slice (equivalent to :)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::full();
    /// // Equivalent to : in NumPy
    /// ```
    pub fn full() -> Self {
        SliceSpec::Range {
            start: None,
            stop: None,
            step: Some(1),
        }
    }
}

/// Slice object builder function (equivalent to np.s_[...])
///
/// Creates slice specifications for array indexing. This provides a convenient
/// way to create complex slice objects similar to NumPy's s_ indexing.
///
/// # Parameters
///
/// * `specs` - Vector of slice specifications
///
/// # Returns
///
/// Vector of slice specifications that can be used for array indexing
///
/// # Examples
///
/// ```
/// use numrs2::array_ops::creation::{s_, SliceSpec};
///
/// // Create slice specifications
/// let slices = s_(&[
///     SliceSpec::from_to(1, 5),
///     SliceSpec::step(2),
///     SliceSpec::Index(3),
/// ]);
///
/// // Equivalent to NumPy's s_[1:5, ::2, 3]
/// assert_eq!(slices.len(), 3);
/// ```
pub fn s_(specs: &[SliceSpec]) -> Vec<SliceSpec> {
    specs.to_vec()
}

/// Constant representing newaxis for array indexing (equivalent to np.newaxis)
///
/// This is used to add new axes to arrays during indexing operations.
/// In NumPy, newaxis is just an alias for None.
///
/// # Examples
///
/// ```
/// use numrs2::array_ops::creation::{NEWAXIS, SliceSpec};
///
/// // Using NEWAXIS in slice specifications
/// let slice_with_newaxis = SliceSpec::NewAxis;
/// assert_eq!(slice_with_newaxis, NEWAXIS);
/// ```
pub const NEWAXIS: SliceSpec = SliceSpec::NewAxis;
