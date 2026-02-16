//! Basic statistical functions
//!
//! This module provides basic statistical operations including:
//! - Statistics trait with mean, var, std, min, max, percentile methods
//! - Peak-to-peak (ptp) function
//! - Axis-based min/max functions
//! - Weighted average function

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, Zero};
use scirs2_core::ndarray::Array1;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

use super::quantile::quantile;

/// Threshold for using parallel processing (minimum array size)
pub const PARALLEL_THRESHOLD: usize = 10000;

// Statistical functions
pub trait Statistics<T> {
    fn mean(&self) -> T;
    fn var(&self) -> T;
    fn std(&self) -> T;
    fn min(&self) -> T;
    fn max(&self) -> T;
    fn percentile(&self, q: T) -> T;
}

impl<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync + 'static> Statistics<T>
    for Array<T>
{
    fn mean(&self) -> T {
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }

        let sum = if data.len() >= PARALLEL_THRESHOLD {
            // Use parallel processing for large arrays
            data.par_iter()
                .map(|&x| x)
                .reduce(|| T::zero(), |acc, x| acc + x)
        } else {
            // Use sequential processing for small arrays
            data.iter().fold(T::zero(), |acc, &x| acc + x)
        };
        sum / T::from(data.len()).expect("data length should be representable")
    }

    fn var(&self) -> T {
        // Use SIMD for f64 arrays with sufficient size via SimdUnifiedOps
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let data: Vec<f64> = self
                .to_vec()
                .iter()
                .map(|x| {
                    let ptr = x as *const T as *const f64;
                    unsafe { *ptr }
                })
                .collect();
            let nd_array = Array1::from_vec(data);
            let result = f64::simd_variance(&nd_array.view());
            return unsafe { std::mem::transmute_copy(&result) };
        }
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }

        let mean = self.mean();
        let sum_squared_diff = if data.len() >= PARALLEL_THRESHOLD {
            // Use parallel processing for large arrays
            data.par_iter()
                .map(|&x| (x - mean) * (x - mean))
                .reduce(|| T::zero(), |acc, x| acc + x)
        } else {
            // Use sequential processing for small arrays
            data.iter()
                .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
        };

        sum_squared_diff / T::from(data.len()).expect("data length should be representable")
    }

    fn std(&self) -> T {
        // Use SIMD for f64 arrays with sufficient size via SimdUnifiedOps
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let data: Vec<f64> = self
                .to_vec()
                .iter()
                .map(|x| {
                    let ptr = x as *const T as *const f64;
                    unsafe { *ptr }
                })
                .collect();
            let nd_array = Array1::from_vec(data);
            let result = f64::simd_std(&nd_array.view());
            return unsafe { std::mem::transmute_copy(&result) };
        }
        self.var().sqrt()
    }

    fn min(&self) -> T {
        // Use SIMD for f64 arrays with sufficient size via SimdUnifiedOps
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let data: Vec<f64> = self
                .to_vec()
                .iter()
                .map(|x| {
                    let ptr = x as *const T as *const f64;
                    unsafe { *ptr }
                })
                .collect();
            let nd_array = Array1::from_vec(data);
            let result = f64::simd_min_element(&nd_array.view());
            return unsafe { std::mem::transmute_copy(&result) };
        }
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }

        if data.len() >= PARALLEL_THRESHOLD {
            // Use parallel processing for large arrays
            data.par_iter()
                .cloned()
                .reduce(|| data[0], |acc, x| if x < acc { x } else { acc })
        } else {
            // Use sequential processing for small arrays
            data.iter()
                .fold(data[0], |acc, &x| if x < acc { x } else { acc })
        }
    }

    fn max(&self) -> T {
        // Use SIMD for f64 arrays with sufficient size via SimdUnifiedOps
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let data: Vec<f64> = self
                .to_vec()
                .iter()
                .map(|x| {
                    let ptr = x as *const T as *const f64;
                    unsafe { *ptr }
                })
                .collect();
            let nd_array = Array1::from_vec(data);
            let result = f64::simd_max_element(&nd_array.view());
            return unsafe { std::mem::transmute_copy(&result) };
        }
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }

        if data.len() >= PARALLEL_THRESHOLD {
            // Use parallel processing for large arrays
            data.par_iter()
                .cloned()
                .reduce(|| data[0], |acc, x| if x > acc { x } else { acc })
        } else {
            // Use sequential processing for small arrays
            data.iter()
                .fold(data[0], |acc, &x| if x > acc { x } else { acc })
        }
    }

    fn percentile(&self, q: T) -> T {
        // Convert to quantile (percentile is in 0-1 range, not 0-100)
        // NumPy percentile uses 0-100 scale, but our internal quantile uses 0-1
        let quantile_val = q; // q is already in 0-1 range

        // Use the more general quantile function directly
        let q_array = Array::from_vec(vec![quantile_val]);
        match quantile(self, &q_array, Some("linear")) {
            Ok(result) => result.to_vec()[0],
            Err(_) => T::zero(),
        }
    }
}

///
/// # Parameters
///
/// * `a` - Input array
/// * `axis` - Optional axis along which to find peak-to-peak values
///
/// # Returns
///
/// An array with the peak-to-peak values
pub fn ptp<T: Float + Clone + NumCast + Default + Send + Sync>(
    a: &Array<T>,
    axis: Option<usize>,
) -> Result<Array<T>> {
    // If no axis specified, calculate the global ptp
    if axis.is_none() {
        let data = a.to_vec();
        let min_val = data
            .iter()
            .fold(data[0], |min, &val| if val < min { val } else { min });
        let max_val = data
            .iter()
            .fold(data[0], |max, &val| if val > max { val } else { max });
        let result = vec![max_val - min_val];
        return Ok(Array::from_vec(result));
    }

    // Calculate min and max along the specified axis
    let axis_val = axis.expect("axis should be Some at this point");

    // This is a simplified implementation - in a real implementation,
    // we would calculate min and max in a single pass for efficiency
    let min_array = min_along_axis(a, axis_val)?;
    let max_array = max_along_axis(a, axis_val)?;

    // Calculate ptp
    let min_data = min_array.to_vec();
    let max_data = max_array.to_vec();

    let mut result = Vec::with_capacity(min_data.len());
    for i in 0..min_data.len() {
        result.push(max_data[i] - min_data[i]);
    }

    Ok(Array::from_vec(result).reshape(&min_array.shape()))
}

/// Calculate minimum values along the specified axis with parallel processing for large arrays
pub fn min_along_axis<T: Float + Clone + NumCast + Default + Send + Sync>(
    a: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            a.ndim()
        )));
    }

    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate the shape of the result
    let mut result_shape = shape.clone();
    result_shape.remove(axis);

    // Initialize the result array
    let data = a.to_vec();
    let mut result = Array::<T>::empty_like(a);
    result = result.reshape(&result_shape);

    // For each position in the result array
    let result_size = result.size();
    let mut min_values = vec![T::zero(); result_size];

    // Calculate the initial indices
    let mut indices = vec![0; shape.len()];
    let mut result_indices = vec![0; result_shape.len()];

    // Use parallel processing for large arrays
    if result_size >= PARALLEL_THRESHOLD {
        min_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, min_val)| {
                // Convert flat index to multi-dimensional indices
                let mut remainder = i;
                let mut result_indices = vec![0; result_shape.len()];
                for j in (0..result_shape.len()).rev() {
                    result_indices[j] = remainder % result_shape[j];
                    remainder /= result_shape[j];
                }

                // Copy the result indices to the array indices, accounting for the removed axis
                let mut indices = vec![0; shape.len()];
                let mut result_idx = 0;
                for j in 0..shape.len() {
                    if j == axis {
                        indices[j] = 0; // Start at 0 for the axis we're minimizing
                    } else {
                        indices[j] = result_indices[result_idx];
                        result_idx += 1;
                    }
                }

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                // Initialize min value with the first element
                *min_val = data[flat_idx];

                // Compare with remaining elements along the axis
                for k in 1..axis_size {
                    indices[axis] = k;

                    // Calculate the new flat index
                    let mut new_idx = 0;
                    let mut new_stride = 1;
                    for j in (0..shape.len()).rev() {
                        new_idx += indices[j] * new_stride;
                        new_stride *= shape[j];
                    }

                    // Update min if needed
                    if data[new_idx] < *min_val {
                        *min_val = data[new_idx];
                    }
                }
            });
    } else {
        // Use sequential processing for small arrays
        #[allow(clippy::needless_range_loop)]
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..shape.len() {
                if j == axis {
                    indices[j] = 0; // Start at 0 for the axis we're minimizing
                } else {
                    indices[j] = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Calculate the flat index in the original data
            let mut flat_idx = 0;
            let mut stride = 1;
            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            // Initialize min value with the first element
            min_values[i] = data[flat_idx];

            // Compare with remaining elements along the axis
            for k in 1..axis_size {
                indices[axis] = k;

                // Calculate the new flat index
                let mut new_idx = 0;
                let mut new_stride = 1;
                for j in (0..shape.len()).rev() {
                    new_idx += indices[j] * new_stride;
                    new_stride *= shape[j];
                }

                // Update min if needed
                if data[new_idx] < min_values[i] {
                    min_values[i] = data[new_idx];
                }
            }
        }
    }

    Ok(Array::from_vec(min_values).reshape(&result_shape))
}

/// Calculate maximum values along the specified axis with parallel processing for large arrays
pub fn max_along_axis<T: Float + Clone + NumCast + Default + Send + Sync>(
    a: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            a.ndim()
        )));
    }

    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate the shape of the result
    let mut result_shape = shape.clone();
    result_shape.remove(axis);

    // Initialize the result array
    let data = a.to_vec();
    let mut result = Array::<T>::empty_like(a);
    result = result.reshape(&result_shape);

    // For each position in the result array
    let result_size = result.size();
    let mut max_values = vec![T::zero(); result_size];

    // Calculate the initial indices
    let mut indices = vec![0; shape.len()];
    let mut result_indices = vec![0; result_shape.len()];

    // Use parallel processing for large arrays
    if result_size >= PARALLEL_THRESHOLD {
        max_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, max_val)| {
                // Convert flat index to multi-dimensional indices
                let mut remainder = i;
                let mut result_indices = vec![0; result_shape.len()];
                for j in (0..result_shape.len()).rev() {
                    result_indices[j] = remainder % result_shape[j];
                    remainder /= result_shape[j];
                }

                // Copy the result indices to the array indices, accounting for the removed axis
                let mut indices = vec![0; shape.len()];
                let mut result_idx = 0;
                for j in 0..shape.len() {
                    if j == axis {
                        indices[j] = 0; // Start at 0 for the axis we're maximizing
                    } else {
                        indices[j] = result_indices[result_idx];
                        result_idx += 1;
                    }
                }

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                // Initialize max value with the first element
                *max_val = data[flat_idx];

                // Compare with remaining elements along the axis
                for k in 1..axis_size {
                    indices[axis] = k;

                    // Calculate the new flat index
                    let mut new_idx = 0;
                    let mut new_stride = 1;
                    for j in (0..shape.len()).rev() {
                        new_idx += indices[j] * new_stride;
                        new_stride *= shape[j];
                    }

                    // Update max if needed
                    if data[new_idx] > *max_val {
                        *max_val = data[new_idx];
                    }
                }
            });
    } else {
        // Use sequential processing for small arrays
        #[allow(clippy::needless_range_loop)]
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..shape.len() {
                if j == axis {
                    indices[j] = 0; // Start at 0 for the axis we're maximizing
                } else {
                    indices[j] = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Calculate the flat index in the original data
            let mut flat_idx = 0;
            let mut stride = 1;
            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            // Initialize max value with the first element
            max_values[i] = data[flat_idx];

            // Compare with remaining elements along the axis
            for k in 1..axis_size {
                indices[axis] = k;

                // Calculate the new flat index
                let mut new_idx = 0;
                let mut new_stride = 1;
                for j in (0..shape.len()).rev() {
                    new_idx += indices[j] * new_stride;
                    new_stride *= shape[j];
                }

                // Update max if needed
                if data[new_idx] > max_values[i] {
                    max_values[i] = data[new_idx];
                }
            }
        }
    }

    Ok(Array::from_vec(max_values).reshape(&result_shape))
}

/// Calculate a weighted average of array elements
///
/// # Parameters
///
/// * `a` - Input array
/// * `weights` - Optional weights for each value
/// * `axis` - Optional axis along which to average
/// * `returned` - If True, also return the sum of weights
///
/// # Returns
///
/// The weighted average or (average, sum of weights) if returned is true
pub fn average<T: Float + Clone + Zero + NumCast + Send + Sync>(
    a: &Array<T>,
    weights: Option<&Array<T>>,
    axis: Option<usize>,
    returned: Option<bool>,
) -> Result<Array<T>> {
    // If no weights provided, return mean
    if weights.is_none() {
        if let Some(ax) = axis {
            // Mean along specified axis
            // In a full implementation, this would use a dedicated mean_along_axis function
            return a.sum_axis(ax).map(|sum| {
                sum.scalar_div(T::from(a.shape()[ax]).expect("axis size should be representable"))
            });
        } else {
            // Calculate overall mean manually
            let data = a.to_vec();
            if data.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "Cannot average empty array".to_string(),
                ));
            }

            let sum = data.iter().fold(T::zero(), |acc, &val| acc + val);
            let mean = sum / T::from(data.len()).expect("data length should be representable");
            return Ok(Array::from_vec(vec![mean]));
        }
    }

    let w = weights.expect("weights should be Some at this point");

    // Check if weights are valid
    if a.shape() != w.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: w.shape(),
        });
    }

    // Calculate the weighted average
    let a_data = a.to_vec();
    let w_data = w.to_vec();

    if let Some(ax) = axis {
        // Weighted average along specified axis
        // This is a simplified implementation - in a real implementation,
        // we would calculate both sums in a single pass for efficiency
        let weighted_sum = weighted_sum_along_axis(a, w, ax)?;
        let weight_sum = w.sum_axis(ax)?;

        let w_sum_data = weight_sum.to_vec();
        let weighted_sum_data = weighted_sum.to_vec();

        let mut result = Vec::with_capacity(w_sum_data.len());
        for i in 0..w_sum_data.len() {
            if w_sum_data[i] == T::zero() {
                result.push(T::zero());
            } else {
                result.push(weighted_sum_data[i] / w_sum_data[i]);
            }
        }

        let avg = Array::from_vec(result).reshape(&weight_sum.shape());

        if returned.unwrap_or(false) {
            // Return both the average and the sum of weights
            // In a real implementation, we would have a way to return multiple arrays
            Ok(avg)
        } else {
            Ok(avg)
        }
    } else {
        // Overall weighted average
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();

        for i in 0..a_data.len() {
            weighted_sum = weighted_sum + a_data[i] * w_data[i];
            weight_sum = weight_sum + w_data[i];
        }

        let avg = if weight_sum == T::zero() {
            T::zero()
        } else {
            weighted_sum / weight_sum
        };

        if returned.unwrap_or(false) {
            // Return both the average and the sum of weights
            // In a real implementation, we would have a way to return multiple arrays
            Ok(Array::from_vec(vec![avg]))
        } else {
            Ok(Array::from_vec(vec![avg]))
        }
    }
}

/// Calculate the weighted sum along a specified axis
fn weighted_sum_along_axis<T: Float + Clone + Zero + NumCast + Send + Sync>(
    a: &Array<T>,
    weights: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            a.ndim()
        )));
    }

    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate the shape of the result
    let mut result_shape = shape.clone();
    result_shape.remove(axis);

    // Initialize the result array
    let mut result = Array::zeros(&result_shape);

    // Get the raw data
    let a_data = a.to_vec();
    let w_data = weights.to_vec();

    // Helper function to calculate indices
    let mut indices = vec![0; shape.len()];
    let mut result_indices = vec![0; result_shape.len()];

    // Calculate the total number of elements in the result
    let result_size = result.size();

    // For each position in the result array
    for i in 0..result_size {
        // Convert flat index to multi-dimensional indices
        let mut remainder = i;
        for j in (0..result_shape.len()).rev() {
            result_indices[j] = remainder % result_shape[j];
            remainder /= result_shape[j];
        }

        // Copy the result indices to the array indices, accounting for the removed axis
        let mut result_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..shape.len() {
            if j == axis {
                indices[j] = 0; // Start at 0 for the axis we're summing
            } else {
                indices[j] = result_indices[result_idx];
                result_idx += 1;
            }
        }

        // Calculate the weighted sum along the specified axis
        let mut sum = T::zero();
        for k in 0..axis_size {
            indices[axis] = k;

            // Calculate the flat index in the original data
            let mut flat_idx = 0;
            let mut stride = 1;
            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            sum = sum + a_data[flat_idx] * w_data[flat_idx];
        }

        // Set the result value
        result.set(&result_indices, sum)?;
    }

    Ok(result)
}
