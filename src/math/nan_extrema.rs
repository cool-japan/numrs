//! NaN-aware extrema and cumulative functions
//!
//! This module provides functions for finding extrema (maximum/minimum) indices
//! and computing cumulative products while ignoring NaN values.
//!
//! # Functions
//!
//! - [`nanargmax`] - Find indices of maximum values, ignoring NaN
//! - [`nanargmin`] - Find indices of minimum values, ignoring NaN
//! - [`nancumprod`] - Cumulative product treating NaN as 1
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::math::{nanargmax, nanargmin, nancumprod};
//!
//! let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 2.0]);
//!
//! // Find index of maximum value, ignoring NaN
//! let max_idx = nanargmax(&a, None, false).expect("nanargmax should succeed");
//! assert_eq!(max_idx.to_vec(), vec![2]); // index 2 has value 3.0
//!
//! // Find index of minimum value, ignoring NaN
//! let min_idx = nanargmin(&a, None, false).expect("nanargmin should succeed");
//! assert_eq!(min_idx.to_vec(), vec![0]); // index 0 has value 1.0
//!
//! // Cumulative product treating NaN as 1
//! let cumprod = nancumprod(&a, None).expect("nancumprod should succeed");
//! assert_eq!(cumprod.to_vec(), vec![1.0, 1.0, 3.0, 6.0]);
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::ops::Mul;

/// Find the indices of the maximum values along an axis, ignoring NaN values
///
/// This function is similar to `argmax` but ignores NaN values in the comparison.
/// If a slice contains only NaN values, the function returns 0 as the index.
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - The axis along which to find the maximum indices. If None, the array is flattened
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array of indices of the maximum values along the specified axis, ignoring NaN
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanargmax;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 2.0]);
///
/// // Find nanargmax of flattened array
/// let index = nanargmax(&a, None, false).expect("nanargmax should succeed");
/// assert_eq!(index.to_vec(), vec![2]); // index 2 has value 3.0
///
/// // With axis
/// let b = Array::from_vec(vec![1.0, f64::NAN, 3.0, 2.0, 4.0, 1.0]).reshape(&[2, 3]);
/// let indices = nanargmax(&b, Some(1), false).expect("nanargmax should succeed");
/// assert_eq!(indices.to_vec(), vec![2, 1]); // max indices in each row, ignoring NaN
/// ```
pub fn nanargmax<T>(array: &Array<T>, axis: Option<usize>, keepdims: bool) -> Result<Array<usize>>
where
    T: Float + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find nanargmax of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find nanargmax of flattened array
            let data = array.to_vec();
            let mut max_idx = 0;
            let mut max_val: Option<T> = None;

            for (i, val) in data.iter().enumerate() {
                if !val.is_nan() && max_val.is_none_or(|mv| *val > mv) {
                    max_val = Some(*val);
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

                // Find max along the axis, ignoring NaN
                let mut max_idx = 0;
                let mut max_val: Option<T> = None;

                for j in 0..axis_size {
                    indices[ax] = j;
                    let val = array.get(&indices)?;

                    if !val.is_nan() && max_val.is_none_or(|mv| val > mv) {
                        max_val = Some(val);
                        max_idx = j;
                    }
                }

                result_data[out_idx] = max_idx;
            }

            Ok(Array::from_vec_shape(result_data, &out_shape)?)
        }
    }
}

/// Find the indices of the minimum values along an axis, ignoring NaN values
///
/// This function is similar to `argmin` but ignores NaN values in the comparison.
/// If a slice contains only NaN values, the function returns 0 as the index.
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - The axis along which to find the minimum indices. If None, the array is flattened
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array of indices of the minimum values along the specified axis, ignoring NaN
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanargmin;
///
/// let a = Array::from_vec(vec![3.0, f64::NAN, 1.0, 2.0]);
///
/// // Find nanargmin of flattened array
/// let index = nanargmin(&a, None, false).expect("nanargmin should succeed");
/// assert_eq!(index.to_vec(), vec![2]); // index 2 has value 1.0
///
/// // With axis
/// let b = Array::from_vec(vec![3.0, f64::NAN, 1.0, 2.0, 4.0, 0.5]).reshape(&[2, 3]);
/// let indices = nanargmin(&b, Some(1), false).expect("nanargmin should succeed");
/// assert_eq!(indices.to_vec(), vec![2, 2]); // min indices in each row, ignoring NaN
/// ```
pub fn nanargmin<T>(array: &Array<T>, axis: Option<usize>, keepdims: bool) -> Result<Array<usize>>
where
    T: Float + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find nanargmin of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find nanargmin of flattened array
            let data = array.to_vec();
            let mut min_idx = 0;
            let mut min_val: Option<T> = None;

            for (i, val) in data.iter().enumerate() {
                if !val.is_nan() && min_val.is_none_or(|mv| *val < mv) {
                    min_val = Some(*val);
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

                // Find min along the axis, ignoring NaN
                let mut min_idx = 0;
                let mut min_val: Option<T> = None;

                for j in 0..axis_size {
                    indices[ax] = j;
                    let val = array.get(&indices)?;

                    if !val.is_nan() && min_val.is_none_or(|mv| val < mv) {
                        min_val = Some(val);
                        min_idx = j;
                    }
                }

                result_data[out_idx] = min_idx;
            }

            Ok(Array::from_vec_shape(result_data, &out_shape)?)
        }
    }
}

/// Return cumulative product of elements along axis, treating NaN as 1
///
/// Similar to `cumprod` but treats NaN values as 1 in the cumulative product.
/// The output array has the same shape as the input.
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute cumulative product. If None, the array is flattened
///
/// # Returns
///
/// Array of cumulative products with same shape as input
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nancumprod;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 2.0, 3.0]);
/// let cumprod = nancumprod(&a, None).expect("nancumprod failed");
/// assert_eq!(cumprod.to_vec(), vec![1.0, 1.0, 2.0, 6.0]);
/// ```
pub fn nancumprod<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T> + Zero + One,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        if ax >= array.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax,
                array.ndim()
            )));
        }

        // Compute cumulative product along specified axis
        let shape = array.shape();
        let mut result = Array::ones(&shape);
        let axis_len = shape[ax];

        // Calculate strides for iteration
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let total_elems: usize = shape.iter().product();
        let axis_stride = strides[ax];
        let group_size = axis_stride * axis_len;

        // Process each group independently
        for group_start in (0..total_elems).step_by(group_size) {
            for offset in 0..axis_stride {
                let mut cumprod = T::one();

                for i in 0..axis_len {
                    let idx = group_start + i * axis_stride + offset;
                    let flat_idx = idx;

                    // Convert flat index to multi-dimensional
                    let mut indices = vec![0; shape.len()];
                    let mut temp = flat_idx;
                    for j in 0..shape.len() {
                        indices[j] = temp / strides[j];
                        temp %= strides[j];
                    }

                    let value = array.get(&indices)?;
                    if !value.is_nan() {
                        cumprod = cumprod * value;
                    }
                    result.set(&indices, cumprod)?;
                }
            }
        }

        Ok(result)
    } else {
        // Flatten array and compute cumulative product
        let flat = array.to_vec();
        let mut result = Vec::with_capacity(flat.len());
        let mut cumprod = T::one();

        for value in flat {
            if !value.is_nan() {
                cumprod = cumprod * value;
            }
            result.push(cumprod);
        }

        Ok(Array::from_vec_shape(result, &array.shape())?)
    }
}
