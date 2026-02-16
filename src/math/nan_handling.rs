//! NaN-aware mathematical functions for NumRS2
//!
//! This module provides functions that handle NaN (Not a Number) values
//! by ignoring them during computations. These are essential for working
//! with real-world data that may contain missing or invalid values.
//!
//! # Functions
//!
//! - [`nansum`] - Sum ignoring NaN values
//! - [`nanmean`] - Mean ignoring NaN values
//! - [`nanstd`] - Standard deviation ignoring NaN values
//! - [`nanvar`] - Variance ignoring NaN values
//! - [`nanmin`] - Minimum ignoring NaN values
//! - [`nanmax`] - Maximum ignoring NaN values
//! - [`nancumsum`] - Cumulative sum ignoring NaN values
//! - [`nanprod`] - Product ignoring NaN values
//! - [`nanpercentile`] - Percentile ignoring NaN values
//! - [`nanquantile`] - Quantile ignoring NaN values
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::math::{nansum, nanmean, nanmin, nanmax};
//!
//! let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
//!
//! // Sum ignoring NaN
//! let sum = nansum(&a, None).expect("nansum failed");
//! assert_eq!(sum.to_vec(), vec![7.0]);
//!
//! // Mean ignoring NaN
//! let mean = nanmean(&a, None).expect("nanmean failed");
//! assert!((mean.to_vec()[0] - 2.333333333333333).abs() < 1e-10);
//!
//! // Min/Max ignoring NaN
//! let min = nanmin(&a, None).expect("nanmin failed");
//! let max = nanmax(&a, None).expect("nanmax failed");
//! assert_eq!(min.to_vec(), vec![1.0]);
//! assert_eq!(max.to_vec(), vec![4.0]);
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, One, Zero};
use std::ops::{Add, Div, Mul, Sub};

/// Compute sum ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute sum (None for flattened array)
///
/// # Returns
///
/// Sum of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nansum;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let sum = nansum(&a, None).expect("nansum failed");
/// assert_eq!(sum.to_vec(), vec![7.0]);
/// ```
pub fn nansum<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Zero,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Sum along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::zeros(&new_shape);

        // Calculate strides for iteration
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut sum = T::zero();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Sum along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    sum = sum + value;
                }
            }

            result.set(&res_indices, sum)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let sum = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::zero(), |acc, x| acc + *x);
        Ok(Array::from_vec(vec![sum]))
    }
}

/// Compute mean ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute mean (None for flattened array)
///
/// # Returns
///
/// Mean of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanmean;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let mean = nanmean(&a, None).expect("nanmean failed");
/// assert!((mean.to_vec()[0] - 2.333333333333333).abs() < 1e-10);
/// ```
pub fn nanmean<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Div<Output = T> + Zero,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        let sums = nansum(array, Some(ax as isize))?;

        // Count non-NaN values along axis
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut counts = Array::zeros(&new_shape);

        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut count = T::zero();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Count along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    count = count + T::one();
                }
            }

            counts.set(&res_indices, count)?;
        }

        // Divide sums by counts
        Ok(sums.zip_with(
            &counts,
            |s, c| {
                if c == T::zero() {
                    T::nan()
                } else {
                    s / c
                }
            },
        )?)
    } else {
        let mut sum = T::zero();
        let mut count = 0;

        let array_vec = array.to_vec();
        for value in array_vec.iter() {
            if !value.is_nan() {
                sum = sum + *value;
                count += 1;
            }
        }

        if count == 0 {
            Ok(Array::from_vec(vec![T::nan()]))
        } else {
            Ok(Array::from_vec(vec![
                sum / T::from(count).expect("count should be representable"),
            ]))
        }
    }
}

/// Compute standard deviation ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute std (None for flattened array)
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Standard deviation of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanstd;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let std = nanstd(&a, None, 0).expect("nanstd failed");
/// ```
pub fn nanstd<T>(array: &Array<T>, axis: Option<isize>, ddof: usize) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero,
{
    let variance = nanvar(array, axis, ddof)?;
    Ok(variance.map(|x| x.sqrt()))
}

/// Compute variance ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute variance (None for flattened array)
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Variance of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanvar;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let var = nanvar(&a, None, 0).expect("nanvar failed");
/// ```
pub fn nanvar<T>(array: &Array<T>, axis: Option<isize>, ddof: usize) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero,
{
    let mean = nanmean(array, axis)?;

    if let Some(axis_val) = axis {
        // Compute variance along axis
        let ax = if axis_val < 0 {
            (array.ndim() as isize + axis_val) as usize
        } else {
            axis_val as usize
        };

        // Expand mean for broadcasting
        let mut mean_shape = vec![1; array.ndim()];
        mean_shape[ax] = array.shape()[ax];
        let mean_expanded = mean.reshape(&mean_shape);

        // Compute squared differences
        let diff = array.zip_with(&mean_expanded, |a, m| a - m)?;
        let diff_sq = diff.map(|x| if x.is_nan() { T::zero() } else { x * x });

        // Count non-NaN values along axis
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut counts = Array::zeros(&new_shape);

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut count = T::zero();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    count = count + T::one();
                }
            }

            counts.set(&res_indices, count)?;
        }

        // Sum squared differences along axis
        let sum_sq = nansum(&diff_sq, Some(ax as isize))?;

        // Compute variance
        Ok(sum_sq.zip_with(&counts, |s, c| {
            let adjusted_count = c - T::from(ddof).expect("ddof should be representable");
            if adjusted_count <= T::zero() {
                T::nan()
            } else {
                s / adjusted_count
            }
        })?)
    } else {
        // Compute variance for flattened array
        let mean_val = mean.to_vec()[0];
        let mut sum_sq = T::zero();
        let mut count = 0;

        let array_vec = array.to_vec();
        for value in array_vec.iter() {
            if !value.is_nan() {
                let diff = *value - mean_val;
                sum_sq = sum_sq + diff * diff;
                count += 1;
            }
        }

        if count <= ddof {
            Ok(Array::from_vec(vec![T::nan()]))
        } else {
            Ok(Array::from_vec(vec![
                sum_sq / T::from(count - ddof).expect("count-ddof should be representable"),
            ]))
        }
    }
}

/// Compute minimum ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute min (None for flattened array)
///
/// # Returns
///
/// Minimum of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanmin;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let min = nanmin(&a, None).expect("nanmin failed");
/// assert_eq!(min.to_vec(), vec![1.0]);
/// ```
pub fn nanmin<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Find min along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::full(&new_shape, T::nan());

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut min_val = T::nan();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Find min along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() && (min_val.is_nan() || value < min_val) {
                    min_val = value;
                }
            }

            result.set(&res_indices, min_val)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let min = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(T::nan());
        Ok(Array::from_vec(vec![min]))
    }
}

/// Compute maximum ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute max (None for flattened array)
///
/// # Returns
///
/// Maximum of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanmax;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let max = nanmax(&a, None).expect("nanmax failed");
/// assert_eq!(max.to_vec(), vec![4.0]);
/// ```
pub fn nanmax<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Find max along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::full(&new_shape, T::nan());

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut max_val = T::nan();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Find max along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() && (max_val.is_nan() || value > max_val) {
                    max_val = value;
                }
            }

            result.set(&res_indices, max_val)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let max = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(T::nan());
        Ok(Array::from_vec(vec![max]))
    }
}

/// Return cumulative sum of elements along axis, ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute cumulative sum. If None, flattened array
///
/// # Returns
///
/// Cumulative sum array with same shape as input
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nancumsum;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
/// let cumsum = nancumsum(&a, None).expect("nancumsum failed");
/// assert_eq!(cumsum.to_vec(), vec![1.0, 1.0, 4.0, 8.0]);
/// ```
pub fn nancumsum<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Zero,
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

        // Compute cumulative sum along specified axis
        let shape = array.shape();
        let mut result = Array::zeros(&shape);
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
                let mut cumsum = T::zero();

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
                        cumsum = cumsum + value;
                    }
                    result.set(&indices, cumsum)?;
                }
            }
        }

        Ok(result)
    } else {
        // Flatten array and compute cumulative sum
        let flat = array.to_vec();
        let mut result = Vec::with_capacity(flat.len());
        let mut cumsum = T::zero();

        for value in flat {
            if !value.is_nan() {
                cumsum = cumsum + value;
            }
            result.push(cumsum);
        }

        Ok(Array::from_vec(result).reshape(&array.shape()))
    }
}

/// Return product of elements along axis, ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute product. If None, compute over flattened array
///
/// # Returns
///
/// Product of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanprod;
///
/// let a = Array::from_vec(vec![2.0, f64::NAN, 3.0, 4.0]);
/// let prod = nanprod(&a, None).expect("nanprod failed");
/// assert_eq!(prod.to_vec(), vec![24.0]);
/// ```
pub fn nanprod<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T> + One,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Find product along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::full(&new_shape, T::one());

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut prod = T::one();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Find product along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    prod = prod * value;
                }
            }

            result.set(&res_indices, prod)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let prod = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::one(), |acc, &x| acc * x);
        Ok(Array::from_vec(vec![prod]))
    }
}

/// Compute percentile of array ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `q` - Percentile to compute (0-100)
/// * `axis` - Axis along which to compute percentile. If None, compute over flattened array
/// * `method` - Method to use for percentile computation (same as percentile)
///
/// # Returns
///
/// Percentile value(s)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanpercentile;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
/// let p50 = nanpercentile(&a, 50.0, None, None).expect("nanpercentile failed");
/// assert_eq!(p50.to_vec(), vec![3.0]);
/// ```
pub fn nanpercentile<T>(
    array: &Array<T>,
    q: T,
    axis: Option<isize>,
    method: Option<&str>,
) -> Result<Array<T>>
where
    T: Float + Clone + NumCast + std::fmt::Display,
{
    // Convert percentile to array and call nanquantile
    let q_arr = Array::from_vec(vec![
        q / T::from(100.0).expect("100.0 should be representable"),
    ]);
    nanquantile(array, &q_arr, axis, method)
}

/// Compute quantile of array ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `q` - Quantile(s) to compute (0-1)
/// * `axis` - Axis along which to compute quantile. If None, compute over flattened array
/// * `method` - Method to use for quantile computation
///
/// # Returns
///
/// Quantile value(s)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanquantile;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
/// let q = Array::from_vec(vec![0.5]);
/// let median = nanquantile(&a, &q, None, None).expect("nanquantile failed");
/// assert_eq!(median.to_vec(), vec![3.0]);
/// ```
pub fn nanquantile<T>(
    array: &Array<T>,
    q: &Array<T>,
    axis: Option<isize>,
    method: Option<&str>,
) -> Result<Array<T>>
where
    T: Float + Clone + NumCast + std::fmt::Display,
{
    let method_str = method.unwrap_or("linear");

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

        // Compute quantile along specified axis
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let q_vec = q.to_vec();
        let n_quantiles = q_vec.len();

        // Result shape is new_shape + [n_quantiles]
        let mut result_shape = new_shape.clone();
        result_shape.push(n_quantiles);
        let mut result = Array::zeros(&result_shape);

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Collect non-NaN values along the axis
            let mut values = Vec::new();
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    values.push(value);
                }
            }

            // Sort values and compute quantiles
            if values.is_empty() {
                // All values were NaN
                for (q_idx, _) in q_vec.iter().enumerate() {
                    let mut result_indices = res_indices.clone();
                    result_indices.push(q_idx);
                    result.set(&result_indices, T::nan())?;
                }
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = values.len();

                for (q_idx, &q_val) in q_vec.iter().enumerate() {
                    if q_val < T::zero() || q_val > T::one() {
                        return Err(NumRs2Error::InvalidOperation(format!(
                            "Quantile value {} out of bounds [0, 1]",
                            q_val
                        )));
                    }

                    let idx_float = q_val * T::from(n - 1).expect("n-1 should be representable");
                    let idx_lower = idx_float.floor();
                    let idx_upper = idx_float.ceil();
                    let idx_lower_usize = idx_lower
                        .to_usize()
                        .expect("lower index should be convertible");
                    let idx_upper_usize = idx_upper
                        .to_usize()
                        .expect("upper index should be convertible");

                    let quantile = match method_str {
                        "linear" => {
                            if idx_lower == idx_upper {
                                values[idx_lower_usize]
                            } else {
                                let fraction = idx_float - idx_lower;
                                values[idx_lower_usize]
                                    + fraction * (values[idx_upper_usize] - values[idx_lower_usize])
                            }
                        }
                        "lower" => values[idx_lower_usize],
                        "higher" => values[idx_upper_usize],
                        "nearest" => {
                            if idx_float - idx_lower < idx_upper - idx_float {
                                values[idx_lower_usize]
                            } else {
                                values[idx_upper_usize]
                            }
                        }
                        "midpoint" => {
                            if idx_lower == idx_upper {
                                values[idx_lower_usize]
                            } else {
                                (values[idx_lower_usize] + values[idx_upper_usize])
                                    / T::from(2.0).expect("2.0 should be representable")
                            }
                        }
                        _ => {
                            return Err(NumRs2Error::InvalidOperation(format!(
                                "Invalid method '{}'. Must be one of 'linear', 'lower', 'higher', 'nearest', 'midpoint'",
                                method_str
                            )))
                        }
                    };

                    let mut result_indices = res_indices.clone();
                    result_indices.push(q_idx);
                    result.set(&result_indices, quantile)?;
                }
            }
        }

        Ok(result)
    } else {
        // Flatten array and compute quantiles
        let array_vec = array.to_vec();
        let mut values: Vec<T> = array_vec.into_iter().filter(|x| !x.is_nan()).collect();

        if values.is_empty() {
            // All values were NaN
            let q_vec = q.to_vec();
            let result = vec![T::nan(); q_vec.len()];
            return Ok(Array::from_vec(result));
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len();
        let q_vec = q.to_vec();
        let mut result = Vec::with_capacity(q_vec.len());

        for &q_val in &q_vec {
            if q_val < T::zero() || q_val > T::one() {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Quantile value {} out of bounds [0, 1]",
                    q_val
                )));
            }

            let idx_float = q_val * T::from(n - 1).expect("n-1 should be representable");
            let idx_lower = idx_float.floor();
            let idx_upper = idx_float.ceil();
            let idx_lower_usize = idx_lower
                .to_usize()
                .expect("lower index should be convertible");
            let idx_upper_usize = idx_upper
                .to_usize()
                .expect("upper index should be convertible");

            let quantile = match method_str {
                "linear" => {
                    if idx_lower == idx_upper {
                        values[idx_lower_usize]
                    } else {
                        let fraction = idx_float - idx_lower;
                        values[idx_lower_usize]
                            + fraction * (values[idx_upper_usize] - values[idx_lower_usize])
                    }
                }
                "lower" => values[idx_lower_usize],
                "higher" => values[idx_upper_usize],
                "nearest" => {
                    if idx_float - idx_lower < idx_upper - idx_float {
                        values[idx_lower_usize]
                    } else {
                        values[idx_upper_usize]
                    }
                }
                "midpoint" => {
                    if idx_lower == idx_upper {
                        values[idx_lower_usize]
                    } else {
                        (values[idx_lower_usize] + values[idx_upper_usize])
                            / T::from(2.0).expect("2.0 should be representable")
                    }
                }
                _ => {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Invalid method '{}'. Must be one of 'linear', 'lower', 'higher', 'nearest', 'midpoint'",
                        method_str
                    )))
                }
            };

            result.push(quantile);
        }

        Ok(Array::from_vec(result))
    }
}
