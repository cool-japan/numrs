//! Value checking functions for arrays.
//!
//! This module provides functions for testing array elements for various conditions:
//!
//! - **Special value detection**: `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`
//! - **Number type detection**: `isnormal`, `isreal`, `iscomplex`
//! - **Value replacement**: `nan_to_num`
//! - **Non-zero element operations**: `count_nonzero`, `nonzero`, `flatnonzero`
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::math::{isnan, isinf, isfinite, nan_to_num};
//!
//! // Check for special values
//! let a = Array::from_vec(vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
//! let nan_mask = isnan(&a);
//! let inf_mask = isinf(&a);
//! let finite_mask = isfinite(&a);
//!
//! // Replace special values
//! let clean = nan_to_num(&a, None, None, None).expect("nan_to_num should succeed");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

/// Test element-wise for NaN
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of boolean values where True indicates NaN
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, f64::INFINITY]);
/// let nan_mask = isnan(&a);
/// assert_eq!(nan_mask.to_vec(), vec![false, true, false, false]);
/// ```
pub fn isnan<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_nan())
}

/// Test element-wise for positive or negative infinity
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of boolean values where True indicates infinity
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
/// let inf_mask = isinf(&a);
/// assert_eq!(inf_mask.to_vec(), vec![false, true, true, false]);
/// ```
pub fn isinf<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_infinite())
}

/// Test element-wise for finiteness (not infinity and not NaN)
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of boolean values where True indicates finite values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NAN, 2.0]);
/// let finite_mask = isfinite(&a);
/// assert_eq!(finite_mask.to_vec(), vec![true, false, false, true]);
/// ```
pub fn isfinite<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_finite())
}

/// Test element-wise for positive infinity
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates positive infinity
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isposinf;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
/// let posinf_mask = isposinf(&a);
/// assert_eq!(posinf_mask.to_vec(), vec![false, true, false, false]);
/// ```
pub fn isposinf<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_infinite() && x.is_sign_positive())
}

/// Test element-wise for negative infinity
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates negative infinity
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isneginf;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
/// let neginf_mask = isneginf(&a);
/// assert_eq!(neginf_mask.to_vec(), vec![false, false, true, false]);
/// ```
pub fn isneginf<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_infinite() && x.is_sign_negative())
}

/// Test element-wise for normal numbers (not zero, subnormal, infinite or NaN)
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates normal numbers
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isnormal;
///
/// let a = Array::from_vec(vec![1.0, 0.0, f64::INFINITY, f64::NAN]);
/// let normal_mask = isnormal(&a);
/// assert_eq!(normal_mask.to_vec(), vec![true, false, false, false]);
/// ```
pub fn isnormal<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_normal())
}

/// Test element-wise for real numbers (opposite of complex)
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates real numbers (always True for real arrays)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isreal;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let real_mask = isreal(&a);
/// assert_eq!(real_mask.to_vec(), vec![true, true, true]);
/// ```
pub fn isreal<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|_x| true) // Real arrays are always real
}

/// Test element-wise for complex numbers
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates complex numbers (always False for real arrays)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::iscomplex;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let complex_mask = iscomplex(&a);
/// assert_eq!(complex_mask.to_vec(), vec![false, false, false]);
/// ```
pub fn iscomplex<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|_x| false) // Real arrays are never complex
}

/// Replace NaN with zero and infinity with large finite numbers
///
/// # Parameters
///
/// * `array` - Input array
/// * `nan` - Value to replace NaN (default: 0.0)
/// * `posinf` - Value to replace positive infinity (default: very large positive number)
/// * `neginf` - Value to replace negative infinity (default: very large negative number)
///
/// # Returns
///
/// Array with replaced values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
/// let clean = nan_to_num(&a, None, None, None).expect("nan_to_num should succeed");
/// // NaN -> 0.0, inf -> f64::MAX, -inf -> f64::MIN
/// ```
pub fn nan_to_num<T>(
    array: &Array<T>,
    nan: Option<T>,
    posinf: Option<T>,
    neginf: Option<T>,
) -> Result<Array<T>>
where
    T: Float + std::fmt::Debug,
{
    let nan_val = nan.unwrap_or_else(T::zero);
    let posinf_val = posinf.unwrap_or_else(T::max_value);
    let neginf_val = neginf.unwrap_or_else(T::min_value);

    Ok(array.map(|x| {
        if x.is_nan() {
            nan_val
        } else if x.is_infinite() {
            if x.is_sign_positive() {
                posinf_val
            } else {
                neginf_val
            }
        } else {
            x
        }
    }))
}

/// Count the number of non-zero values in the array
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to count non-zeros. If None, count over flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Number of non-zero values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::count_nonzero;
///
/// let a = Array::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
/// let count = count_nonzero(&a, None, false).expect("count_nonzero failed");
/// assert_eq!(count.to_vec(), vec![3]); // 3 non-zero elements
/// ```
pub fn count_nonzero<T>(
    array: &Array<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<Array<usize>>
where
    T: Clone + Zero + PartialEq,
{
    match axis {
        None => {
            // Count non-zeros in flattened array
            let data = array.to_vec();
            let count = data.iter().filter(|&x| x != &T::zero()).count();

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![count]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![count]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
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
            let mut result_data = vec![0usize; out_size];

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

                // Count non-zeros along the axis
                let mut count = 0;
                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;
                    if val != T::zero() {
                        count += 1;
                    }
                }

                result_data[out_idx] = count;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Return the indices of the elements that are non-zero
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Tuple of arrays, one for each dimension, containing indices of non-zero elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nonzero;
///
/// let a = Array::from_vec(vec![0, 1, 0, 3, 0, 5]).reshape(&[2, 3]);
/// let indices = nonzero(&a).expect("nonzero failed");
/// assert_eq!(indices.len(), 2); // 2D array has 2 index arrays
/// assert_eq!(indices[0].to_vec(), vec![0, 1, 1]);
/// assert_eq!(indices[1].to_vec(), vec![1, 0, 2]);
/// ```
pub fn nonzero<T>(array: &Array<T>) -> Result<Vec<Array<usize>>>
where
    T: Clone + Zero + PartialEq,
{
    let shape = array.shape();
    let ndim = array.ndim();

    // Find all non-zero positions
    let mut nonzero_positions = Vec::new();
    let data = array.to_vec();

    for (idx, value) in data.iter().enumerate() {
        if value != &T::zero() {
            // Convert flat index to multi-dimensional indices
            let mut indices = vec![0; ndim];
            let mut temp = idx;

            for i in (0..ndim).rev() {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            nonzero_positions.push(indices);
        }
    }

    // Transpose the positions to get separate arrays for each dimension
    let mut result = Vec::with_capacity(ndim);
    for dim in 0..ndim {
        let dim_indices: Vec<usize> = nonzero_positions.iter().map(|pos| pos[dim]).collect();
        result.push(Array::from_vec(dim_indices));
    }

    Ok(result)
}

/// Return indices that are non-zero in the flattened version of the input array
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of indices in the flattened array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![0, 1, 0, 3, 0, 5]);
/// let indices = flatnonzero(&a).expect("flatnonzero should succeed");
/// assert_eq!(indices.to_vec(), vec![1, 3, 5]);
/// ```
pub fn flatnonzero<T>(array: &Array<T>) -> Result<Array<usize>>
where
    T: Clone + Zero + PartialEq,
{
    let data = array.to_vec();
    let nonzero_indices: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, value)| *value != &T::zero())
        .map(|(idx, _)| idx)
        .collect();

    Ok(Array::from_vec(nonzero_indices))
}
