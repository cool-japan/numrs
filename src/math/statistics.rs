//! Statistical and aggregation functions for NumRS2 arrays.
//!
//! This module provides functions for computing statistics on arrays, including:
//! - Index operations: `argmax`, `argmin`, `argsort`
//! - Rounding: `around`
//! - Cumulative operations: `cumsum`, `cumprod`
//! - Aggregation: `mean`, `prod`
//! - Array transformation: `resize`
//! - Variability: `std`, `var`
//! - Value limiting: `clip`
//!
//! Most functions support axis-specific operations and dimension preservation
//! through the `keepdims` parameter.
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::math::{mean, var, std, cumsum};
//!
//! let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//!
//! // Compute mean
//! let m = mean(&a, None, false).expect("mean should succeed");
//! assert_eq!(m.to_vec(), vec![3.0]);
//!
//! // Compute variance
//! let v = var(&a, None, 0, false).expect("var should succeed");
//! assert_eq!(v.to_vec(), vec![2.0]);
//!
//! // Compute cumulative sum
//! let cs = cumsum(&a, None, None).expect("cumsum should succeed");
//! assert_eq!(cs.to_vec(), vec![1.0, 3.0, 6.0, 10.0, 15.0]);
//! ```

use crate::array::Array;
use crate::error::Result;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::{Add, Div, Mul, Sub};

// Helper functions for SIMD operations
use crate::math::{from_nd_array, to_nd_array_f32, to_nd_array_f64};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Find the indices of the maximum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - The axis along which to find the maximum indices. If None, the array is flattened
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array of indices of the maximum values along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0]).reshape(&[2, 3]);
///
/// // Find argmax along axis 1
/// let indices = argmax(&a, Some(1), false).expect("argmax should succeed");
/// assert_eq!(indices.to_vec(), vec![1, 1]); // max indices in each row
///
/// // Find argmax of flattened array
/// let index = argmax(&a, None, false).expect("argmax should succeed");
/// assert_eq!(index.to_vec(), vec![4]); // index 4 has value 5.0
/// ```
pub fn argmax<T>(array: &Array<T>, axis: Option<usize>, keepdims: bool) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find argmax of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find argmax of flattened array
            let data = array.to_vec();
            let mut max_idx = 0;
            let mut max_val = &data[0];

            for (i, val) in data.iter().enumerate().skip(1) {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            Ok(Array::from_vec(vec![max_idx]))
        }
        Some(ax) => {
            if ax >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[ax] = 1;
            } else {
                out_shape.remove(ax);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![0_usize; out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > ax || (i == ax && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find max along the axis
                let mut max_idx = 0;
                let mut max_val = None;

                for j in 0..axis_size {
                    indices[ax] = j;
                    let val = array.get(&indices)?;

                    if max_val.as_ref().is_none_or(|mv| &val > mv) {
                        max_val = Some(val);
                        max_idx = j;
                    }
                }

                result_data[out_idx] = max_idx;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Find the indices of the minimum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - The axis along which to find the minimum indices. If None, the array is flattened
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array of indices of the minimum values along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![5.0, 3.0, 2.0, 4.0, 1.0, 6.0]).reshape(&[2, 3]);
///
/// // Find argmin along axis 1
/// let indices = argmin(&a, Some(1), false).expect("argmin should succeed");
/// assert_eq!(indices.to_vec(), vec![2, 1]); // min indices in each row
///
/// // Find argmin of flattened array
/// let index = argmin(&a, None, false).expect("argmin should succeed");
/// assert_eq!(index.to_vec(), vec![4]); // index 4 has value 1.0
/// ```
pub fn argmin<T>(array: &Array<T>, axis: Option<usize>, keepdims: bool) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find argmin of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find argmin of flattened array
            let data = array.to_vec();
            let mut min_idx = 0;
            let mut min_val = &data[0];

            for (i, val) in data.iter().enumerate().skip(1) {
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
            }

            Ok(Array::from_vec(vec![min_idx]))
        }
        Some(ax) => {
            if ax >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[ax] = 1;
            } else {
                out_shape.remove(ax);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![0_usize; out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > ax || (i == ax && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find min along the axis
                let mut min_idx = 0;
                let mut min_val = None;

                for j in 0..axis_size {
                    indices[ax] = j;
                    let val = array.get(&indices)?;

                    if min_val.as_ref().is_none_or(|mv| &val < mv) {
                        min_val = Some(val);
                        min_idx = j;
                    }
                }

                result_data[out_idx] = min_idx;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Returns the indices that would sort an array
///
/// # Parameters
///
/// * `array` - Input array to sort
/// * `axis` - Axis along which to sort. If None, the array is flattened before sorting
/// * `kind` - Sorting algorithm (currently only supports default stable sort)
/// * `order` - Not used (for NumPy compatibility)
///
/// # Returns
///
/// Array of indices that sort the array along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![3.0, 1.0, 2.0]);
/// let indices = argsort(&a, None, None, None).expect("argsort should succeed");
/// assert_eq!(indices.to_vec(), vec![1, 2, 0]); // sorted order: [1.0, 2.0, 3.0]
/// ```
pub fn argsort<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _kind: Option<&str>,
    _order: Option<&[&str]>,
) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
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
        indices.sort_by(|&a, &b| {
            data[a]
                .partial_cmp(&data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return Ok(Array::from_vec(indices));
    };

    if axis >= array.ndim() {
        return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    let shape = array.shape();
    let axis_size = shape[axis];
    let result_shape = shape.clone();

    // The output has the same shape as input
    let total_size: usize = shape.iter().product();
    let mut result_data = vec![0_usize; total_size];

    // Calculate strides
    let mut strides = vec![1; array.ndim()];
    for i in (0..array.ndim() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Number of sorts to perform
    let n_sorts = total_size / axis_size;

    for sort_idx in 0..n_sorts {
        // Collect values along the axis for this position
        let mut values_with_indices: Vec<(T, usize)> = Vec::with_capacity(axis_size);

        // Determine the base indices for this sort
        let mut base_indices = vec![0; array.ndim()];
        let mut temp = sort_idx;
        let _dim_idx = 0;

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

        // Sort by value
        values_with_indices
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Place sorted indices in result
        for (k, (_, idx)) in values_with_indices.into_iter().enumerate() {
            base_indices[axis] = k;
            let flat_idx = base_indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| idx * strides[i])
                .sum::<usize>();
            result_data[flat_idx] = idx;
        }
    }

    Ok(Array::from_vec(result_data).reshape(&result_shape))
}

/// Round array elements to the given number of decimals
///
/// # Parameters
///
/// * `array` - Input array
/// * `decimals` - Number of decimal places to round to (default: 0)
/// * `out` - Optional output array
///
/// # Returns
///
/// Array with rounded values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.234, 2.567, 3.891]);
/// let rounded = around(&a, Some(2), None).expect("around should succeed");
/// assert_eq!(rounded.to_vec(), vec![1.23, 2.57, 3.89]);
/// ```
pub fn around<T>(
    array: &Array<T>,
    decimals: Option<i32>,
    out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let decimals = decimals.unwrap_or(0);
    let multiplier = T::from(10.0_f64.powi(decimals)).expect("power of 10 should be representable");

    let result = array.map(|x| {
        if decimals >= 0 {
            (x * multiplier).round() / multiplier
        } else {
            let divisor =
                T::from(10.0_f64.powi(-decimals)).expect("power of 10 should be representable");
            (x / divisor).round() * divisor
        }
    });

    if let Some(out_array) = out {
        if out_array.shape() != result.shape() {
            return Err(crate::error::NumRs2Error::ShapeMismatch {
                expected: result.shape(),
                actual: out_array.shape(),
            });
        }
        *out_array = result;
        Ok(out_array.clone())
    } else {
        Ok(result)
    }
}

/// Return the cumulative sum of array elements along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the cumulative sum is computed. If None, the array is flattened
/// * `dtype` - Data type of the returned array (uses input type if None)
/// * `out` - Alternative output array to place the result
///
/// # Returns
///
/// Array with cumulative sum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let cs = cumsum(&a, None, None).expect("cumsum should succeed");
/// assert_eq!(cs.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
/// ```
pub fn cumsum<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    /// Threshold for using parallel processing
    const PARALLEL_THRESHOLD: usize = 1000;

    if array.is_empty() {
        return Ok(array.clone());
    }

    match axis {
        None => {
            // Use SIMD for f64/f32 arrays with sufficient size
            if array.len() >= 64 {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let nd = to_nd_array_f64(unsafe {
                        std::mem::transmute::<&Array<T>, &Array<f64>>(array)
                    });
                    let result = f64::simd_cumsum(&nd.view());
                    return Ok(unsafe {
                        std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                            result,
                            &array.shape(),
                        ))
                    });
                }
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let nd = to_nd_array_f32(unsafe {
                        std::mem::transmute::<&Array<T>, &Array<f32>>(array)
                    });
                    let result = f32::simd_cumsum(&nd.view());
                    return Ok(unsafe {
                        std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                            result,
                            &array.shape(),
                        ))
                    });
                }
            }

            // Flatten and compute cumsum (inherently sequential)
            let data = array.to_vec();
            let mut result = Vec::with_capacity(data.len());
            let mut sum = T::zero();

            for val in data {
                sum = sum + val;
                result.push(sum);
            }

            Ok(Array::from_vec(result))
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let data = array.to_vec();

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through all positions perpendicular to the axis
            let axis_size = shape[axis];
            let n_iterations = array.size() / axis_size;

            // Use parallel processing for large arrays
            if is_parallel_enabled() && n_iterations >= PARALLEL_THRESHOLD {
                let result_data: Vec<T> = (0..n_iterations)
                    .into_par_iter()
                    .flat_map(|iter_idx| {
                        // Determine the base indices for this iteration
                        let mut indices = vec![0; shape.len()];
                        let mut temp = iter_idx;

                        for i in (0..shape.len()).rev() {
                            if i != axis {
                                indices[i] = temp % shape[i];
                                temp /= shape[i];
                            }
                        }

                        // Compute cumulative sum along the axis
                        let mut local_results = Vec::with_capacity(axis_size);
                        let mut sum = T::zero();
                        for j in 0..axis_size {
                            indices[axis] = j;
                            let flat_idx = indices
                                .iter()
                                .enumerate()
                                .map(|(i, &idx)| idx * strides[i])
                                .sum::<usize>();
                            sum = sum + data[flat_idx];
                            local_results.push((flat_idx, sum));
                        }
                        local_results
                    })
                    .collect::<Vec<(usize, T)>>()
                    .into_iter()
                    .fold(vec![T::zero(); array.size()], |mut acc, (idx, val)| {
                        acc[idx] = val;
                        acc
                    });

                Ok(Array::from_vec(result_data).reshape(&shape))
            } else {
                // Sequential processing for small arrays
                let mut result_data = vec![T::zero(); array.size()];

                for iter_idx in 0..n_iterations {
                    // Determine the base indices for this iteration
                    let mut indices = vec![0; array.ndim()];
                    let mut temp = iter_idx;

                    for i in (0..array.ndim()).rev() {
                        if i != axis {
                            indices[i] = temp % shape[i];
                            temp /= shape[i];
                        }
                    }

                    // Compute cumulative sum along the axis
                    let mut sum = T::zero();
                    for j in 0..axis_size {
                        indices[axis] = j;
                        let flat_idx = indices
                            .iter()
                            .enumerate()
                            .map(|(i, &idx)| idx * strides[i])
                            .sum::<usize>();
                        sum = sum + data[flat_idx];
                        result_data[flat_idx] = sum;
                    }
                }

                Ok(Array::from_vec(result_data).reshape(&shape))
            }
        }
    }
}

/// Return the cumulative product of array elements along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the cumulative product is computed. If None, the array is flattened
/// * `dtype` - Data type of the returned array (uses input type if None)
/// * `out` - Alternative output array to place the result
///
/// # Returns
///
/// Array with cumulative product values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let cp = cumprod(&a, None, None).expect("cumprod should succeed");
/// assert_eq!(cp.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
/// ```
pub fn cumprod<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T> + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    /// Threshold for using parallel processing
    const PARALLEL_THRESHOLD: usize = 1000;

    if array.is_empty() {
        return Ok(array.clone());
    }

    match axis {
        None => {
            // Flatten and compute cumprod (inherently sequential)
            let data = array.to_vec();
            let mut result = Vec::with_capacity(data.len());
            let mut prod = T::one();

            for val in data {
                prod = prod * val;
                result.push(prod);
            }

            Ok(Array::from_vec(result))
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let data = array.to_vec();

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through all positions perpendicular to the axis
            let axis_size = shape[axis];
            let n_iterations = array.size() / axis_size;

            // Use parallel processing for large arrays
            if is_parallel_enabled() && n_iterations >= PARALLEL_THRESHOLD {
                let result_data: Vec<T> = (0..n_iterations)
                    .into_par_iter()
                    .flat_map(|iter_idx| {
                        // Determine the base indices for this iteration
                        let mut indices = vec![0; shape.len()];
                        let mut temp = iter_idx;

                        for i in (0..shape.len()).rev() {
                            if i != axis {
                                indices[i] = temp % shape[i];
                                temp /= shape[i];
                            }
                        }

                        // Compute cumulative product along the axis
                        let mut local_results = Vec::with_capacity(axis_size);
                        let mut prod = T::one();
                        for j in 0..axis_size {
                            indices[axis] = j;
                            let flat_idx = indices
                                .iter()
                                .enumerate()
                                .map(|(i, &idx)| idx * strides[i])
                                .sum::<usize>();
                            prod = prod * data[flat_idx];
                            local_results.push((flat_idx, prod));
                        }
                        local_results
                    })
                    .collect::<Vec<(usize, T)>>()
                    .into_iter()
                    .fold(vec![T::zero(); array.size()], |mut acc, (idx, val)| {
                        acc[idx] = val;
                        acc
                    });

                Ok(Array::from_vec(result_data).reshape(&shape))
            } else {
                // Sequential processing for small arrays
                let mut result_data = vec![T::zero(); array.size()];

                for iter_idx in 0..n_iterations {
                    // Determine the base indices for this iteration
                    let mut indices = vec![0; array.ndim()];
                    let mut temp = iter_idx;

                    for i in (0..array.ndim()).rev() {
                        if i != axis {
                            indices[i] = temp % shape[i];
                            temp /= shape[i];
                        }
                    }

                    // Compute cumulative product along the axis
                    let mut prod = T::one();
                    for j in 0..axis_size {
                        indices[axis] = j;
                        let flat_idx = indices
                            .iter()
                            .enumerate()
                            .map(|(i, &idx)| idx * strides[i])
                            .sum::<usize>();
                        prod = prod * data[flat_idx];
                        result_data[flat_idx] = prod;
                    }
                }

                Ok(Array::from_vec(result_data).reshape(&shape))
            }
        }
    }
}

/// Compute the arithmetic mean along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the mean is computed. If None, compute mean of flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the mean values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
/// let m = mean(&a, Some(1), false).expect("mean should succeed");
/// assert_eq!(m.to_vec(), vec![2.0, 5.0]); // mean of each row
/// ```
pub fn mean<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Div<Output = T> + NumCast,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute mean of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Compute mean of flattened array
            let data = array.to_vec();
            let sum = data.iter().fold(T::zero(), |acc, x| acc + *x);
            let count = T::from(data.len()).expect("data length should be representable");
            let mean_val = sum / count;

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![mean_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![mean_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];
            let axis_size_t = T::from(axis_size).expect("axis_size should be representable");

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

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

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

                // Compute sum along the axis
                let mut sum = T::zero();
                for j in 0..axis_size {
                    indices[axis] = j;
                    sum = sum + array.get(&indices)?;
                }

                result_data[out_idx] = sum / axis_size_t;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Return the product of array elements over a given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the product is computed. If None, compute product of flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
/// * `initial` - Starting value for the product
///
/// # Returns
///
/// Array containing the product values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let p = prod(&a, None, false, None).expect("prod should succeed");
/// assert_eq!(p.to_vec(), vec![24.0]); // 1 * 2 * 3 * 4
/// ```
pub fn prod<T>(
    array: &Array<T>,
    axis: Option<isize>,
    keepdims: bool,
    initial: Option<T>,
) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T>,
{
    if array.is_empty() && initial.is_none() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute product of empty array without initial value".to_string(),
        ));
    }

    let init_val = initial.unwrap_or_else(|| T::one());

    match axis {
        None => {
            // Compute product of flattened array
            let data = array.to_vec();
            let product = data.iter().fold(init_val, |acc, x| acc * *x);

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![product]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![product]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
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

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

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

                // Compute product along the axis
                let mut product = init_val;
                for j in 0..axis_size {
                    indices[axis] = j;
                    product = product * array.get(&indices)?;
                }

                result_data[out_idx] = product;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Return a new array with the specified shape
///
/// # Parameters
///
/// * `array` - Input array
/// * `new_shape` - Shape of resized array
///
/// # Returns
///
/// New array with the specified shape. If the new array is larger, it is filled
/// with repeated copies of the input array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let resized = resize(&a, &[5]).expect("resize should succeed");
/// assert_eq!(resized.to_vec(), vec![1, 2, 3, 1, 2]); // repeated to fill
/// ```
pub fn resize<T>(array: &Array<T>, new_shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + Zero,
{
    let new_size: usize = new_shape.iter().product();

    if new_size == 0 || array.is_empty() {
        return Ok(Array::zeros(new_shape));
    }

    let old_data = array.to_vec();
    let old_size = old_data.len();

    let mut new_data = Vec::with_capacity(new_size);

    // Fill new array by cycling through old data
    for i in 0..new_size {
        new_data.push(old_data[i % old_size].clone());
    }

    Ok(Array::from_vec(new_data).reshape(new_shape))
}

/// Compute the standard deviation along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the standard deviation is computed. If None, compute over flattened array
/// * `ddof` - Delta degrees of freedom. The divisor is N - ddof (default: 0)
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the standard deviation values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::std;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let s = std(&a, None, 0, false).expect("std failed");
/// assert!((s.to_vec()[0] - 1.414f64).abs() < 0.01); // approximately sqrt(2)
/// ```
pub fn std<T>(
    array: &Array<T>,
    axis: Option<isize>,
    ddof: usize,
    keepdims: bool,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast,
{
    // First compute variance, then take square root
    let variance = var(array, axis, ddof, keepdims)?;
    Ok(variance.map(|x| x.sqrt()))
}

/// Compute the variance along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the variance is computed. If None, compute over flattened array
/// * `ddof` - Delta degrees of freedom. The divisor is N - ddof (default: 0)
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the variance values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let v = var(&a, None, 0, false).expect("var should succeed");
/// assert_eq!(v.to_vec(), vec![2.0]); // variance is 2.0
/// ```
pub fn var<T>(
    array: &Array<T>,
    axis: Option<isize>,
    ddof: usize,
    keepdims: bool,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute variance of empty array".to_string(),
        ));
    }

    // First compute the mean
    let mean_arr = mean(array, axis, keepdims)?;

    match axis {
        None => {
            // Compute variance of flattened array
            let data = array.to_vec();
            let mean_val = mean_arr.to_vec()[0];

            let sum_sq = data
                .iter()
                .map(|x| {
                    let diff = *x - mean_val;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);

            let n = data.len();
            if n <= ddof {
                return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                    "Degrees of freedom {} >= number of elements {}",
                    ddof, n
                )));
            }

            let divisor = T::from(n - ddof).expect("n-ddof should be representable");
            let var_val = sum_sq / divisor;

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![var_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![var_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            if axis_size <= ddof {
                return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                    "Degrees of freedom {} >= axis size {}",
                    ddof, axis_size
                )));
            }

            let divisor =
                T::from(axis_size - ddof).expect("axis_size-ddof should be representable");

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

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Get mean values as vector
            let mean_vec = mean_arr.to_vec();

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

                // Compute sum of squared differences along the axis
                let mean_val = mean_vec[out_idx];
                let mut sum_sq = T::zero();

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;
                    let diff = val - mean_val;
                    sum_sq = sum_sq + (diff * diff);
                }

                result_data[out_idx] = sum_sq / divisor;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Clip (limit) the values in an array
///
/// # Parameters
///
/// * `array` - Input array
/// * `min` - Minimum value. If None, clipping is not performed on lower interval edge
/// * `max` - Maximum value. If None, clipping is not performed on upper interval edge
///
/// # Returns
///
/// Array with values clipped to [min, max]
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let clipped = clip(&a, Some(2.0), Some(4.0)).expect("clip should succeed");
/// assert_eq!(clipped.to_vec(), vec![2.0, 2.0, 3.0, 4.0, 4.0]);
/// ```
pub fn clip<T>(array: &Array<T>, min: Option<T>, max: Option<T>) -> Result<Array<T>>
where
    T: PartialOrd + Clone,
{
    Ok(array.map(|x| {
        let mut val = x;
        if let Some(ref min_val) = min {
            if &val < min_val {
                val = min_val.clone();
            }
        }
        if let Some(ref max_val) = max {
            if &val > max_val {
                val = max_val.clone();
            }
        }
        val
    }))
}
