//! NaN-aware statistical functions
//!
//! This module provides statistical functions that ignore NaN values:
//! - nanmean: Mean ignoring NaN values
//! - nanstd: Standard deviation ignoring NaN values
//! - nanvar: Variance ignoring NaN values
//! - nanmin: Minimum ignoring NaN values
//! - nanmax: Maximum ignoring NaN values
//! - nansum: Sum ignoring NaN values
//! - nanprod: Product ignoring NaN values

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, Zero};
use scirs2_core::parallel_ops::*;

use super::basic::PARALLEL_THRESHOLD;

/// Helper function to compute flat index from multi-dimensional indices
#[inline]
fn indices_to_flat_idx(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| idx * strides[i])
        .sum()
}

/// Helper function to compute strides for flat indexing
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the arithmetic mean along the specified axis, ignoring NaNs with parallel processing
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the mean is computed (None for all elements)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with NaN values ignored in the mean calculation
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nanmean;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
/// let result = nanmean(&a, None, false).expect("nanmean should succeed");
/// assert_eq!(result.to_vec()[0], 8.0 / 3.0); // (1 + 3 + 4) / 3
/// ```
pub fn nanmean<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Compute mean of all elements with parallel processing for large arrays
            let data = array.to_vec();

            if data.len() >= PARALLEL_THRESHOLD {
                // Use parallel processing
                let (sum, count) = data
                    .par_iter()
                    .filter(|x| !x.is_nan())
                    .fold(
                        || (T::zero(), 0usize),
                        |(sum, count), &x| (sum + x, count + 1),
                    )
                    .reduce(
                        || (T::zero(), 0usize),
                        |(sum1, count1), (sum2, count2)| (sum1 + sum2, count1 + count2),
                    );

                if count == 0 {
                    Ok(Array::from_vec(vec![T::nan()]))
                } else {
                    let mean = sum / T::from(count).expect("count should be representable");
                    Ok(Array::from_vec(vec![mean]))
                }
            } else {
                // Use sequential processing for small arrays
                let filtered: Vec<T> = data.into_iter().filter(|x| !x.is_nan()).collect();

                if filtered.is_empty() {
                    Ok(Array::from_vec(vec![T::nan()]))
                } else {
                    let sum = filtered.iter().fold(T::zero(), |acc, &x| acc + x);
                    let mean = sum
                        / T::from(filtered.len()).expect("filtered length should be representable");
                    Ok(Array::from_vec(vec![mean]))
                }
            }
        }
        Some(ax) => {
            // Full axis support for multi-dimensional arrays
            let shape = array.shape();
            let ndim = array.ndim();

            if ax >= ndim {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let axis_size = shape[ax];
            let data = array.to_vec();
            let strides = compute_strides(&shape);

            // Compute output shape (remove axis dimension)
            let mut out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();

            // Use parallel processing for large arrays
            let result_data: Vec<T> = if out_size >= PARALLEL_THRESHOLD {
                (0..out_size)
                    .into_par_iter()
                    .map(|out_idx| {
                        let mut indices = vec![0usize; ndim];
                        let mut temp = out_idx;
                        for i in 0..ndim {
                            if i != ax {
                                let dim_size = shape[i];
                                indices[i] = temp % dim_size;
                                temp /= dim_size;
                            }
                        }

                        // Compute mean along the axis, ignoring NaN
                        let mut sum = T::zero();
                        let mut count = 0usize;
                        for j in 0..axis_size {
                            indices[ax] = j;
                            let flat_idx = indices_to_flat_idx(&indices, &strides);
                            let val = data[flat_idx];
                            if !val.is_nan() {
                                sum = sum + val;
                                count += 1;
                            }
                        }

                        if count == 0 {
                            T::nan()
                        } else {
                            sum / T::from(count).expect("count should be representable")
                        }
                    })
                    .collect()
            } else {
                // Sequential processing for small arrays
                (0..out_size)
                    .map(|out_idx| {
                        let mut indices = vec![0usize; ndim];
                        let mut temp = out_idx;
                        for i in 0..ndim {
                            if i != ax {
                                let dim_size = shape[i];
                                indices[i] = temp % dim_size;
                                temp /= dim_size;
                            }
                        }

                        // Compute mean along the axis, ignoring NaN
                        let mut sum = T::zero();
                        let mut count = 0usize;
                        for j in 0..axis_size {
                            indices[ax] = j;
                            let flat_idx = indices_to_flat_idx(&indices, &strides);
                            let val = data[flat_idx];
                            if !val.is_nan() {
                                sum = sum + val;
                                count += 1;
                            }
                        }

                        if count == 0 {
                            T::nan()
                        } else {
                            sum / T::from(count).expect("count should be representable")
                        }
                    })
                    .collect()
            };

            let result = Array::from_vec_shape(result_data, &out_shape)?;

            if keepdims {
                // Insert dimension of size 1 at the axis position
                let mut keepdim_shape = out_shape;
                keepdim_shape.insert(ax, 1);
                Ok(result.reshape(&keepdim_shape))
            } else {
                Ok(result)
            }
        }
    }
}

/// Compute the standard deviation along the specified axis, ignoring NaNs
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the std is computed (None for all elements)
/// * `ddof` - Delta degrees of freedom (default 0)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with NaN values ignored in the std calculation
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nanstd;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
/// let result = nanstd(&a, None, Some(0), false).expect("nanstd should succeed");
/// // Standard deviation of [1, 3, 4]
/// ```
pub fn nanstd<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    ddof: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    let variance = nanvar(array, axis, ddof, keepdims)?;
    Ok(variance.map(|x| x.sqrt()))
}

/// Compute the variance along the specified axis, ignoring NaNs with parallel processing
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the var is computed (None for all elements)
/// * `ddof` - Delta degrees of freedom (default 0)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with NaN values ignored in the variance calculation
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nanvar;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
/// let result = nanvar(&a, None, Some(0), false).expect("nanvar should succeed");
/// // Variance of [1, 3, 4]
/// ```
pub fn nanvar<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    ddof: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    let ddof_val = ddof.unwrap_or(0);

    match axis {
        None => {
            // Compute variance of all elements with parallel processing for large arrays
            let data = array.to_vec();

            if data.len() >= PARALLEL_THRESHOLD {
                // Use parallel processing
                let (sum, count) = data
                    .par_iter()
                    .filter(|x| !x.is_nan())
                    .fold(
                        || (T::zero(), 0usize),
                        |(sum, count), &x| (sum + x, count + 1),
                    )
                    .reduce(
                        || (T::zero(), 0usize),
                        |(sum1, count1), (sum2, count2)| (sum1 + sum2, count1 + count2),
                    );

                if count <= ddof_val {
                    Ok(Array::from_vec(vec![T::nan()]))
                } else {
                    let mean = sum / T::from(count).expect("count should be representable");

                    let sum_squared_diff = data
                        .par_iter()
                        .filter(|x| !x.is_nan())
                        .map(|&x| (x - mean) * (x - mean))
                        .reduce(|| T::zero(), |acc, x| acc + x);

                    let variance = sum_squared_diff
                        / T::from(count - ddof_val).expect("count-ddof should be representable");
                    Ok(Array::from_vec(vec![variance]))
                }
            } else {
                // Use sequential processing for small arrays
                let filtered: Vec<T> = data.into_iter().filter(|x| !x.is_nan()).collect();

                if filtered.len() <= ddof_val {
                    Ok(Array::from_vec(vec![T::nan()]))
                } else {
                    let mean = filtered.iter().fold(T::zero(), |acc, &x| acc + x)
                        / T::from(filtered.len()).expect("filtered length should be representable");

                    let sum_squared_diff = filtered
                        .iter()
                        .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean));

                    let variance = sum_squared_diff
                        / T::from(filtered.len() - ddof_val)
                            .expect("filtered len-ddof should be representable");
                    Ok(Array::from_vec(vec![variance]))
                }
            }
        }
        Some(ax) => {
            // Full axis support for multi-dimensional arrays
            let shape = array.shape();
            let ndim = array.ndim();

            if ax >= ndim {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let axis_size = shape[ax];
            let data = array.to_vec();
            let strides = compute_strides(&shape);

            // Compute output shape (remove axis dimension)
            let mut out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = Vec::with_capacity(out_size);

            // Iterate over all output positions
            for out_idx in 0..out_size {
                // Reconstruct indices for all dimensions except the axis
                let mut indices = vec![0usize; ndim];
                let mut temp = out_idx;
                for i in 0..ndim {
                    if i != ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    }
                }

                // First pass: compute mean along the axis, ignoring NaN
                let mut sum = T::zero();
                let mut count = 0usize;
                for j in 0..axis_size {
                    indices[ax] = j;
                    let flat_idx = indices_to_flat_idx(&indices, &strides);
                    let val = data[flat_idx];
                    if !val.is_nan() {
                        sum = sum + val;
                        count += 1;
                    }
                }

                if count <= ddof_val {
                    result_data.push(T::nan());
                } else {
                    let mean = sum / T::from(count).expect("count should be representable");

                    // Second pass: compute sum of squared differences
                    let mut sum_sq_diff = T::zero();
                    for j in 0..axis_size {
                        indices[ax] = j;
                        let flat_idx = indices_to_flat_idx(&indices, &strides);
                        let val = data[flat_idx];
                        if !val.is_nan() {
                            let diff = val - mean;
                            sum_sq_diff = sum_sq_diff + diff * diff;
                        }
                    }

                    let variance = sum_sq_diff
                        / T::from(count - ddof_val).expect("count-ddof should be representable");
                    result_data.push(variance);
                }
            }

            let result = Array::from_vec_shape(result_data, &out_shape)?;

            if keepdims {
                let mut keepdim_shape = out_shape;
                keepdim_shape.insert(ax, 1);
                Ok(result.reshape(&keepdim_shape))
            } else {
                Ok(result)
            }
        }
    }
}

/// Compute the minimum of an array along the specified axis, ignoring NaNs
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the minimum is computed (None for all elements)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with minimum values ignoring NaNs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nanmin;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 0.5]);
/// let result = nanmin(&a, None, false).expect("nanmin should succeed");
/// assert_eq!(result.to_vec()[0], 0.5);
/// ```
pub fn nanmin<T: Float + Clone + Zero + NumCast + std::fmt::Display>(
    array: &Array<T>,
    axis: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    match axis {
        None => {
            let data = array.to_vec();
            let filtered: Vec<T> = data.into_iter().filter(|x| !x.is_nan()).collect();

            if filtered.is_empty() {
                Ok(Array::from_vec(vec![T::nan()]))
            } else {
                let min_val = filtered.iter().fold(filtered[0], |acc, &x| acc.min(x));
                Ok(Array::from_vec(vec![min_val]))
            }
        }
        Some(ax) => {
            // Full axis support for multi-dimensional arrays
            let shape = array.shape();
            let ndim = array.ndim();

            if ax >= ndim {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let axis_size = shape[ax];
            let data = array.to_vec();
            let strides = compute_strides(&shape);

            // Compute output shape (remove axis dimension)
            let mut out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = Vec::with_capacity(out_size);

            // Iterate over all output positions
            for out_idx in 0..out_size {
                let mut indices = vec![0usize; ndim];
                let mut temp = out_idx;
                for i in 0..ndim {
                    if i != ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    }
                }

                // Find minimum along the axis, ignoring NaN
                let mut min_val: Option<T> = None;
                for j in 0..axis_size {
                    indices[ax] = j;
                    let flat_idx = indices_to_flat_idx(&indices, &strides);
                    let val = data[flat_idx];
                    if !val.is_nan() {
                        min_val = Some(match min_val {
                            Some(current) => current.min(val),
                            None => val,
                        });
                    }
                }

                result_data.push(min_val.unwrap_or(T::nan()));
            }

            let result = Array::from_vec_shape(result_data, &out_shape)?;

            if keepdims {
                let mut keepdim_shape = out_shape;
                keepdim_shape.insert(ax, 1);
                Ok(result.reshape(&keepdim_shape))
            } else {
                Ok(result)
            }
        }
    }
}

/// Compute the maximum of an array along the specified axis, ignoring NaNs
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the maximum is computed (None for all elements)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with maximum values ignoring NaNs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nanmax;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 0.5]);
/// let result = nanmax(&a, None, false).expect("nanmax should succeed");
/// assert_eq!(result.to_vec()[0], 3.0);
/// ```
pub fn nanmax<T: Float + Clone + Zero + NumCast + std::fmt::Display>(
    array: &Array<T>,
    axis: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    match axis {
        None => {
            let data = array.to_vec();
            let filtered: Vec<T> = data.into_iter().filter(|x| !x.is_nan()).collect();

            if filtered.is_empty() {
                Ok(Array::from_vec(vec![T::nan()]))
            } else {
                let max_val = filtered.iter().fold(filtered[0], |acc, &x| acc.max(x));
                Ok(Array::from_vec(vec![max_val]))
            }
        }
        Some(ax) => {
            // Full axis support for multi-dimensional arrays
            let shape = array.shape();
            let ndim = array.ndim();

            if ax >= ndim {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let axis_size = shape[ax];
            let data = array.to_vec();
            let strides = compute_strides(&shape);

            // Compute output shape (remove axis dimension)
            let mut out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = Vec::with_capacity(out_size);

            // Iterate over all output positions
            for out_idx in 0..out_size {
                let mut indices = vec![0usize; ndim];
                let mut temp = out_idx;
                for i in 0..ndim {
                    if i != ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    }
                }

                // Find maximum along the axis, ignoring NaN
                let mut max_val: Option<T> = None;
                for j in 0..axis_size {
                    indices[ax] = j;
                    let flat_idx = indices_to_flat_idx(&indices, &strides);
                    let val = data[flat_idx];
                    if !val.is_nan() {
                        max_val = Some(match max_val {
                            Some(current) => current.max(val),
                            None => val,
                        });
                    }
                }

                result_data.push(max_val.unwrap_or(T::nan()));
            }

            let result = Array::from_vec_shape(result_data, &out_shape)?;

            if keepdims {
                let mut keepdim_shape = out_shape;
                keepdim_shape.insert(ax, 1);
                Ok(result.reshape(&keepdim_shape))
            } else {
                Ok(result)
            }
        }
    }
}

/// Compute the sum of an array along the specified axis, ignoring NaNs with parallel processing
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the sum is computed (None for all elements)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with sum values ignoring NaNs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nansum;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 2.0]);
/// let result = nansum(&a, None, false).expect("nansum should succeed");
/// assert_eq!(result.to_vec()[0], 6.0); // 1 + 3 + 2
/// ```
pub fn nansum<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    match axis {
        None => {
            let data = array.to_vec();

            let sum = if data.len() >= PARALLEL_THRESHOLD {
                // Use parallel processing for large arrays
                data.par_iter()
                    .filter(|x| !x.is_nan())
                    .cloned()
                    .reduce(|| T::zero(), |acc, x| acc + x)
            } else {
                // Use sequential processing for small arrays
                data.iter()
                    .fold(T::zero(), |acc, &x| if x.is_nan() { acc } else { acc + x })
            };
            Ok(Array::from_vec(vec![sum]))
        }
        Some(ax) => {
            // Full axis support for multi-dimensional arrays
            let shape = array.shape();
            let ndim = array.ndim();

            if ax >= ndim {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let axis_size = shape[ax];
            let data = array.to_vec();
            let strides = compute_strides(&shape);

            // Compute output shape (remove axis dimension)
            let mut out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();

            // Use parallel processing for large arrays
            let result_data: Vec<T> = if out_size >= PARALLEL_THRESHOLD {
                (0..out_size)
                    .into_par_iter()
                    .map(|out_idx| {
                        let mut indices = vec![0usize; ndim];
                        let mut temp = out_idx;
                        for i in 0..ndim {
                            if i != ax {
                                let dim_size = shape[i];
                                indices[i] = temp % dim_size;
                                temp /= dim_size;
                            }
                        }

                        // Compute sum along the axis, ignoring NaN
                        let mut sum = T::zero();
                        for j in 0..axis_size {
                            indices[ax] = j;
                            let flat_idx = indices_to_flat_idx(&indices, &strides);
                            let val = data[flat_idx];
                            if !val.is_nan() {
                                sum = sum + val;
                            }
                        }
                        sum
                    })
                    .collect()
            } else {
                // Sequential processing for small arrays
                (0..out_size)
                    .map(|out_idx| {
                        let mut indices = vec![0usize; ndim];
                        let mut temp = out_idx;
                        for i in 0..ndim {
                            if i != ax {
                                let dim_size = shape[i];
                                indices[i] = temp % dim_size;
                                temp /= dim_size;
                            }
                        }

                        // Compute sum along the axis, ignoring NaN
                        let mut sum = T::zero();
                        for j in 0..axis_size {
                            indices[ax] = j;
                            let flat_idx = indices_to_flat_idx(&indices, &strides);
                            let val = data[flat_idx];
                            if !val.is_nan() {
                                sum = sum + val;
                            }
                        }
                        sum
                    })
                    .collect()
            };

            let result = Array::from_vec_shape(result_data, &out_shape)?;

            if keepdims {
                let mut keepdim_shape = out_shape;
                keepdim_shape.insert(ax, 1);
                Ok(result.reshape(&keepdim_shape))
            } else {
                Ok(result)
            }
        }
    }
}

/// Compute the product of an array along the specified axis, ignoring NaNs
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which the product is computed (None for all elements)
/// * `keepdims` - Whether to keep the dimensions of the result
///
/// # Returns
///
/// Array with product values ignoring NaNs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::nanprod;
///
/// let a = Array::from_vec(vec![2.0, f64::NAN, 3.0, 4.0]);
/// let result = nanprod(&a, None, false).expect("nanprod should succeed");
/// assert_eq!(result.to_vec()[0], 24.0); // 2 * 3 * 4
/// ```
pub fn nanprod<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    keepdims: bool,
) -> Result<Array<T>> {
    match axis {
        None => {
            let data = array.to_vec();

            let product = if data.len() >= PARALLEL_THRESHOLD {
                // Use parallel processing for large arrays
                data.par_iter()
                    .filter(|x| !x.is_nan())
                    .cloned()
                    .reduce(|| T::one(), |acc, x| acc * x)
            } else {
                data.iter()
                    .fold(T::one(), |acc, &x| if x.is_nan() { acc } else { acc * x })
            };
            Ok(Array::from_vec(vec![product]))
        }
        Some(ax) => {
            // Full axis support for multi-dimensional arrays
            let shape = array.shape();
            let ndim = array.ndim();

            if ax >= ndim {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }

            let axis_size = shape[ax];
            let data = array.to_vec();
            let strides = compute_strides(&shape);

            // Compute output shape (remove axis dimension)
            let mut out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();

            // Use parallel processing for large arrays
            let result_data: Vec<T> = if out_size >= PARALLEL_THRESHOLD {
                (0..out_size)
                    .into_par_iter()
                    .map(|out_idx| {
                        let mut indices = vec![0usize; ndim];
                        let mut temp = out_idx;
                        for i in 0..ndim {
                            if i != ax {
                                let dim_size = shape[i];
                                indices[i] = temp % dim_size;
                                temp /= dim_size;
                            }
                        }

                        // Compute product along the axis, ignoring NaN
                        let mut product = T::one();
                        for j in 0..axis_size {
                            indices[ax] = j;
                            let flat_idx = indices_to_flat_idx(&indices, &strides);
                            let val = data[flat_idx];
                            if !val.is_nan() {
                                product = product * val;
                            }
                        }
                        product
                    })
                    .collect()
            } else {
                // Sequential processing for small arrays
                (0..out_size)
                    .map(|out_idx| {
                        let mut indices = vec![0usize; ndim];
                        let mut temp = out_idx;
                        for i in 0..ndim {
                            if i != ax {
                                let dim_size = shape[i];
                                indices[i] = temp % dim_size;
                                temp /= dim_size;
                            }
                        }

                        // Compute product along the axis, ignoring NaN
                        let mut product = T::one();
                        for j in 0..axis_size {
                            indices[ax] = j;
                            let flat_idx = indices_to_flat_idx(&indices, &strides);
                            let val = data[flat_idx];
                            if !val.is_nan() {
                                product = product * val;
                            }
                        }
                        product
                    })
                    .collect()
            };

            let result = Array::from_vec_shape(result_data, &out_shape)?;

            if keepdims {
                let mut keepdim_shape = out_shape;
                keepdim_shape.insert(ax, 1);
                Ok(result.reshape(&keepdim_shape))
            } else {
                Ok(result)
            }
        }
    }
}
