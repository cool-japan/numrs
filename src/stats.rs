use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero, NumCast};

// Statistical functions
pub trait Statistics<T> {
    fn mean(&self) -> T;
    fn var(&self) -> T;
    fn std(&self) -> T;
    fn min(&self) -> T;
    fn max(&self) -> T;
    fn percentile(&self, q: T) -> T;
}

impl<T: Float + Clone + Zero + NumCast + std::fmt::Display> Statistics<T> for Array<T> {
    fn mean(&self) -> T {
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }
        
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        sum / T::from(data.len()).unwrap()
    }
    
    fn var(&self) -> T {
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }
        
        let mean = self.mean();
        let sum_squared_diff = data.iter()
            .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean));
            
        sum_squared_diff / T::from(data.len()).unwrap()
    }
    
    fn std(&self) -> T {
        self.var().sqrt()
    }
    
    fn min(&self) -> T {
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }
        
        data.iter()
            .fold(data[0], |acc, &x| if x < acc { x } else { acc })
    }
    
    fn max(&self) -> T {
        let data = self.to_vec();
        if data.is_empty() {
            return T::zero();
        }
        
        data.iter()
            .fold(data[0], |acc, &x| if x > acc { x } else { acc })
    }
    
    fn percentile(&self, q: T) -> T {
        // Convert to quantile (percentile is in 0-1 range, not 0-100)
        // NumPy percentile uses 0-100 scale, but our internal quantile uses 0-1
        let quantile_val = q; // q is already in 0-1 range
        
        // Use the more general quantile function directly
        let q_array = Array::from_vec(vec![quantile_val]);
        match quantile(self, &q_array, Some("linear")) {
            Ok(result) => result.to_vec()[0],
            Err(_) => T::zero()
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
pub fn ptp<T: Float + Clone + NumCast + Default>(a: &Array<T>, axis: Option<usize>) -> Result<Array<T>> {
    // If no axis specified, calculate the global ptp
    if axis.is_none() {
        let data = a.to_vec();
        let min_val = data.iter().fold(data[0], |min, &val| if val < min { val } else { min });
        let max_val = data.iter().fold(data[0], |max, &val| if val > max { val } else { max });
        let result = vec![max_val - min_val];
        return Ok(Array::from_vec(result));
    }
    
    // Calculate min and max along the specified axis
    let axis_val = axis.unwrap();
    
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

/// Calculate minimum values along the specified axis
pub fn min_along_axis<T: Float + Clone + NumCast + Default>(a: &Array<T>, axis: usize) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, a.ndim())
        ));
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
    
    // Initialize the min values with the first elements
    for i in 0..result_size {
        // Convert flat index to multi-dimensional indices
        let mut remainder = i;
        for j in (0..result_shape.len()).rev() {
            result_indices[j] = remainder % result_shape[j];
            remainder /= result_shape[j];
        }
        
        // Copy the result indices to the array indices, accounting for the removed axis
        let mut result_idx = 0;
        for j in 0..shape.len() {
            if j == axis {
                indices[j] = 0;  // Start at 0 for the axis we're minimizing
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
    
    Ok(Array::from_vec(min_values).reshape(&result_shape))
}

/// Calculate maximum values along the specified axis
pub fn max_along_axis<T: Float + Clone + NumCast + Default>(a: &Array<T>, axis: usize) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, a.ndim())
        ));
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
    
    // Initialize the max values with the first elements
    for i in 0..result_size {
        // Convert flat index to multi-dimensional indices
        let mut remainder = i;
        for j in (0..result_shape.len()).rev() {
            result_indices[j] = remainder % result_shape[j];
            remainder /= result_shape[j];
        }
        
        // Copy the result indices to the array indices, accounting for the removed axis
        let mut result_idx = 0;
        for j in 0..shape.len() {
            if j == axis {
                indices[j] = 0;  // Start at 0 for the axis we're maximizing
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
pub fn average<T: Float + Clone + Zero + NumCast>(
    a: &Array<T>, 
    weights: Option<&Array<T>>,
    axis: Option<usize>,
    returned: Option<bool>
) -> Result<Array<T>> {
    // If no weights provided, return mean
    if weights.is_none() {
        if let Some(ax) = axis {
            // Mean along specified axis
            // In a full implementation, this would use a dedicated mean_along_axis function
            return a.sum_axis(ax).map(|sum| sum.scalar_div(T::from(a.shape()[ax]).unwrap()));
        } else {
            // Calculate overall mean manually
            let data = a.to_vec();
            if data.is_empty() {
                return Err(NumRs2Error::InvalidOperation("Cannot average empty array".to_string()));
            }
            
            let sum = data.iter().fold(T::zero(), |acc, &val| acc + val);
            let mean = sum / T::from(data.len()).unwrap();
            return Ok(Array::from_vec(vec![mean]));
        }
    }
    
    let w = weights.unwrap();
    
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
            return Ok(avg);
        } else {
            return Ok(avg);
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
            return Ok(Array::from_vec(vec![avg]));
        } else {
            return Ok(Array::from_vec(vec![avg]));
        }
    }
}

/// Calculate the weighted sum along a specified axis
fn weighted_sum_along_axis<T: Float + Clone + Zero + NumCast>(
    a: &Array<T>, 
    weights: &Array<T>,
    axis: usize
) -> Result<Array<T>> {
    if axis >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Axis {} out of bounds for array of dimension {}", axis, a.ndim())
        ));
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
        for j in 0..shape.len() {
            if j == axis {
                indices[j] = 0;  // Start at 0 for the axis we're summing
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

// Additional statistics functions
pub fn cov<T: Float + Clone + Zero + NumCast + std::fmt::Display>(a: &Array<T>, b: &Array<T>) -> Result<T> {
    if a.size() != b.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![a.size()],
            actual: vec![b.size()],
        });
    }
    
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let n = T::from(a_data.len()).unwrap();
    
    // Calculate means manually
    let a_sum = a_data.iter().fold(T::zero(), |acc, &val| acc + val);
    let a_mean = a_sum / n;
    
    let b_sum = b_data.iter().fold(T::zero(), |acc, &val| acc + val);
    let b_mean = b_sum / n;
    
    let mut sum_ab = T::zero();
    for i in 0..a_data.len() {
        sum_ab = sum_ab + (a_data[i] - a_mean) * (b_data[i] - b_mean);
    }
    
    Ok(sum_ab / n)
}

pub fn corrcoef<T: Float + Clone + Zero + NumCast + std::fmt::Display>(a: &Array<T>, b: &Array<T>) -> Result<T> {
    let cov_ab = cov(a, b)?;
    
    // Calculate std manually
    let a_data = a.to_vec();
    let a_sum = a_data.iter().fold(T::zero(), |acc, &val| acc + val);
    let a_mean = a_sum / T::from(a_data.len()).unwrap();
    let a_var_sum = a_data.iter().fold(T::zero(), |acc, &val| {
        let diff = val - a_mean;
        acc + diff * diff
    });
    let std_a = (a_var_sum / T::from(a_data.len()).unwrap()).sqrt();
    
    let b_data = b.to_vec();
    let b_sum = b_data.iter().fold(T::zero(), |acc, &val| acc + val);
    let b_mean = b_sum / T::from(b_data.len()).unwrap();
    let b_var_sum = b_data.iter().fold(T::zero(), |acc, &val| {
        let diff = val - b_mean;
        acc + diff * diff
    });
    let std_b = (b_var_sum / T::from(b_data.len()).unwrap()).sqrt();
    
    if std_a == T::zero() || std_b == T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot calculate correlation coefficient with zero standard deviation".to_string()
        ));
    }
    
    Ok(cov_ab / (std_a * std_b))
}

/// Compute the quantiles of a dataset
/// 
/// # Parameters
/// 
/// * `a` - Input array
/// * `q` - Quantile or sequence of quantiles to compute, in range [0, 1]
/// * `method` - Method to use for quantile calculation:
///   * 'linear': Linear interpolation between points
///   * 'lower': Use the lower data point
///   * 'higher': Use the higher data point
///   * 'nearest': Use the nearest data point
///   * 'midpoint': Use the midpoint between adjacent data points
/// 
/// # Returns
/// 
/// Array of quantile values
pub fn quantile<T: Float + Clone + NumCast + std::fmt::Display>(
    a: &Array<T>, 
    q: &Array<T>, 
    method: Option<&str>
) -> Result<Array<T>> {
    let method_str = method.unwrap_or("linear");
    let data = a.to_vec();
    
    if data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute quantiles of an empty array".to_string()
        ));
    }
    
    let q_data = q.to_vec();
    let mut result = Vec::with_capacity(q_data.len());
    
    // Sort the data
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted_data.len();
    
    for &q_val in &q_data {
        // Check if q is in the valid range [0, 1]
        if q_val < T::zero() || q_val > T::one() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Quantile value {} out of bounds [0, 1]", q_val)
            ));
        }
        
        // Calculate the index
        let idx_float = q_val * T::from(n - 1).unwrap();
        let idx_lower = idx_float.floor();
        let idx_upper = idx_float.ceil();
        let idx_lower_usize = idx_lower.to_usize().unwrap();
        let idx_upper_usize = idx_upper.to_usize().unwrap();
        
        // Get quantile based on the method
        let quantile = match method_str {
            "linear" => {
                if idx_lower == idx_upper {
                    sorted_data[idx_lower_usize]
                } else {
                    let fraction = idx_float - idx_lower;
                    let lower_val = sorted_data[idx_lower_usize];
                    let upper_val = sorted_data[idx_upper_usize];
                    lower_val + fraction * (upper_val - lower_val)
                }
            },
            "lower" => sorted_data[idx_lower_usize],
            "higher" => sorted_data[idx_upper_usize],
            "nearest" => {
                if idx_float - idx_lower < idx_upper - idx_float {
                    sorted_data[idx_lower_usize]
                } else {
                    sorted_data[idx_upper_usize]
                }
            },
            "midpoint" => {
                if idx_lower == idx_upper {
                    sorted_data[idx_lower_usize]
                } else {
                    let lower_val = sorted_data[idx_lower_usize];
                    let upper_val = sorted_data[idx_upper_usize];
                    (lower_val + upper_val) / T::from(2.0).unwrap()
                }
            },
            _ => return Err(NumRs2Error::InvalidOperation(
                format!("Invalid method '{}'. Must be one of 'linear', 'lower', 'higher', 'nearest', 'midpoint'", method_str)
            ))
        };
        
        result.push(quantile);
    }
    
    Ok(Array::from_vec(result))
}

/// Compute the percentiles of a dataset
/// 
/// # Parameters
/// 
/// * `a` - Input array
/// * `q` - Percentile or sequence of percentiles to compute, in range [0, 100]
/// * `method` - Method to use for percentile calculation (same as quantile)
/// 
/// # Returns
/// 
/// Array of percentile values
pub fn percentile<T: Float + Clone + NumCast + std::fmt::Display>(
    a: &Array<T>, 
    q: &Array<T>, 
    method: Option<&str>
) -> Result<Array<T>> {
    // Convert percentiles to quantiles (0-100 to 0-1)
    let quantiles = q.map(|x| x / T::from(100.0).unwrap());
    
    // Call quantile with the converted values
    quantile(a, &quantiles, method)
}

/// Calculate a histogram of a dataset
/// 
/// # Parameters
/// 
/// * `a` - Input array
/// * `bins` - Number of bins
/// * `range` - Optional tuple of (min, max) to use for bin edges
/// * `weights` - Optional array of weights for each value
/// 
/// # Returns
/// 
/// A tuple of (histogram counts, bin edges)
pub fn histogram<T: Float + Clone + NumCast + std::fmt::Display>(
    a: &Array<T>, 
    bins: usize,
    range: Option<(T, T)>,
    weights: Option<&Array<T>>
) -> Result<(Array<T>, Array<T>)> {
    let data = a.to_vec();
    if data.is_empty() || bins == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram of an empty array or with zero bins".to_string()
        ));
    }
    
    // Get min and max values - either from range parameter or from data
    let (min_val, max_val) = match range {
        Some((min, max)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Range ({}, {}) is invalid: min must be less than max", min, max)
                ));
            }
            (min, max)
        },
        None => (a.min(), a.max())
    };
    
    // Create bin edges
    let step = (max_val - min_val) / T::from(bins).unwrap();
    let mut bin_edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        bin_edges.push(min_val + step * T::from(i).unwrap());
    }
    
    // Count values in each bin with optional weights
    let mut counts = vec![T::zero(); bins];
    
    if let Some(w) = weights {
        let weights_data = w.to_vec();
        
        if weights_data.len() != data.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![data.len()],
                actual: vec![weights_data.len()],
            });
        }
        
        for (i, &val) in data.iter().enumerate() {
            if val < min_val || val > max_val {
                continue;  // Skip values outside the range
            }
            
            if val == max_val {
                // Handle edge case for the maximum value
                counts[bins - 1] = counts[bins - 1] + weights_data[i];
            } else {
                let bin_idx = ((val - min_val) / step).to_usize().unwrap();
                if bin_idx < bins {
                    counts[bin_idx] = counts[bin_idx] + weights_data[i];
                }
            }
        }
    } else {
        // No weights - just count occurrences
        for &val in &data {
            if val < min_val || val > max_val {
                continue;  // Skip values outside the range
            }
            
            if val == max_val {
                // Handle edge case for the maximum value
                counts[bins - 1] = counts[bins - 1] + T::one();
            } else {
                let bin_idx = ((val - min_val) / step).to_usize().unwrap();
                if bin_idx < bins {
                    counts[bin_idx] = counts[bin_idx] + T::one();
                }
            }
        }
    }
    
    Ok((
        Array::from_vec(counts),
        Array::from_vec(bin_edges)
    ))
}

/// Calculate a 2D histogram of a dataset
/// 
/// # Parameters
/// 
/// * `x` - Input array for x coordinates
/// * `y` - Input array for y coordinates
/// * `bins` - Either a tuple (nx, ny) to specify bins in each dimension,
///           or a single value to use the same number of bins in both dimensions
/// * `range` - Optional tuple ((xmin, xmax), (ymin, ymax)) to use for bin edges
/// * `weights` - Optional array of weights for each value
/// 
/// # Returns
/// 
/// A tuple of (histogram counts, x_edges, y_edges)
pub fn histogram2d<T: Float + Clone + NumCast + std::fmt::Display>(
    x: &Array<T>, 
    y: &Array<T>, 
    bins: impl Into<HistBins>,
    range: Option<((T, T), (T, T))>,
    weights: Option<&Array<T>>
) -> Result<(Array<T>, Array<T>, Array<T>)> {
    let bins_val = bins.into();
    let (x_bins, y_bins) = match bins_val {
        HistBins::Single(n) => (n, n),
        HistBins::Tuple(nx, ny) => (nx, ny),
    };
    
    // Check inputs
    let x_data = x.to_vec();
    let y_data = y.to_vec();
    
    if x_data.len() != y_data.len() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![x_data.len()],
            actual: vec![y_data.len()],
        });
    }
    
    if x_data.is_empty() || x_bins == 0 || y_bins == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram2d with empty arrays or zero bins".to_string()
        ));
    }
    
    // Get min and max values - either from range parameter or from data
    let (x_min, x_max) = match range {
        Some(((min, max), _)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(
                    format!("X range ({}, {}) is invalid: min must be less than max", min, max)
                ));
            }
            (min, max)
        },
        None => (x.min(), x.max())
    };
    
    let (y_min, y_max) = match range {
        Some((_, (min, max))) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Y range ({}, {}) is invalid: min must be less than max", min, max)
                ));
            }
            (min, max)
        },
        None => (y.min(), y.max())
    };
    
    // Create bin edges
    let x_step = (x_max - x_min) / T::from(x_bins).unwrap();
    let mut x_edges = Vec::with_capacity(x_bins + 1);
    for i in 0..=x_bins {
        x_edges.push(x_min + x_step * T::from(i).unwrap());
    }
    
    let y_step = (y_max - y_min) / T::from(y_bins).unwrap();
    let mut y_edges = Vec::with_capacity(y_bins + 1);
    for i in 0..=y_bins {
        y_edges.push(y_min + y_step * T::from(i).unwrap());
    }
    
    // Initialize 2D histogram
    let mut hist = vec![vec![T::zero(); y_bins]; x_bins];
    
    // Fill the histogram with data points
    if let Some(w) = weights {
        let weights_data = w.to_vec();
        
        if weights_data.len() != x_data.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![x_data.len()],
                actual: vec![weights_data.len()],
            });
        }
        
        for i in 0..x_data.len() {
            let x_val = x_data[i];
            let y_val = y_data[i];
            let weight = weights_data[i];
            
            if x_val < x_min || x_val > x_max || y_val < y_min || y_val > y_max {
                continue;  // Skip values outside the range
            }
            
            // Calculate bin indices
            let x_idx = if x_val == x_max {
                x_bins - 1
            } else {
                ((x_val - x_min) / x_step).to_usize().unwrap()
            };
            
            let y_idx = if y_val == y_max {
                y_bins - 1
            } else {
                ((y_val - y_min) / y_step).to_usize().unwrap()
            };
            
            // Add weight to the histogram
            if x_idx < x_bins && y_idx < y_bins {
                hist[x_idx][y_idx] = hist[x_idx][y_idx] + weight;
            }
        }
    } else {
        // No weights - just count occurrences
        for i in 0..x_data.len() {
            let x_val = x_data[i];
            let y_val = y_data[i];
            
            if x_val < x_min || x_val > x_max || y_val < y_min || y_val > y_max {
                continue;  // Skip values outside the range
            }
            
            // Calculate bin indices
            let x_idx = if x_val == x_max {
                x_bins - 1
            } else {
                ((x_val - x_min) / x_step).to_usize().unwrap()
            };
            
            let y_idx = if y_val == y_max {
                y_bins - 1
            } else {
                ((y_val - y_min) / y_step).to_usize().unwrap()
            };
            
            // Increment the histogram count
            if x_idx < x_bins && y_idx < y_bins {
                hist[x_idx][y_idx] = hist[x_idx][y_idx] + T::one();
            }
        }
    }
    
    // Convert 2D vector to 1D and create the Array
    let mut flat_hist = Vec::with_capacity(x_bins * y_bins);
    for row in hist {
        flat_hist.extend(row);
    }
    
    Ok((
        Array::from_vec(flat_hist).reshape(&[x_bins, y_bins]),
        Array::from_vec(x_edges),
        Array::from_vec(y_edges)
    ))
}

/// Calculate counts of each unique value in an array
/// 
/// # Parameters
/// 
/// * `a` - Input array
/// * `weights` - Optional weights for each value
/// * `minlength` - Minimum length of the output array
/// 
/// # Returns
/// 
/// An array of counts for each value (assuming values are integers from 0 to n-1)
pub fn bincount<T: Float + Clone + NumCast>(
    a: &Array<T>, 
    weights: Option<&Array<T>>,
    minlength: Option<usize>
) -> Result<Array<T>> {
    let data = a.to_vec();
    
    if data.is_empty() {
        let min_len = minlength.unwrap_or(0);
        let counts = vec![T::zero(); min_len];
        return Ok(Array::from_vec(counts));
    }
    
    // Find the maximum value to determine the output array size
    let data_cloned = data.clone();
    let max_val = data_cloned.iter().fold(data_cloned[0], |max, &val| if val > max { val } else { max });
    if max_val < T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "All values in bincount input array must be non-negative".to_string()
        ));
    }
    
    let max_idx = max_val.to_usize().unwrap();
    let min_length = minlength.unwrap_or(0);
    let bin_count = if max_idx + 1 > min_length { max_idx + 1 } else { min_length };
    
    // Initialize output array
    let mut counts = vec![T::zero(); bin_count];
    
    // Add each value to the bin
    if let Some(w) = weights {
        let weights_data = w.to_vec();
        
        if weights_data.len() != data.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![data.len()],
                actual: vec![weights_data.len()],
            });
        }
        
        for (i, &val) in data.iter().enumerate() {
            if val < T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "All values in bincount input array must be non-negative".to_string()
                ));
            }
            
            let idx = val.to_usize().unwrap();
            if idx < bin_count {
                counts[idx] = counts[idx] + weights_data[i];
            }
        }
    } else {
        for &val in &data {
            if val < T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "All values in bincount input array must be non-negative".to_string()
                ));
            }
            
            let idx = val.to_usize().unwrap();
            if idx < bin_count {
                counts[idx] = counts[idx] + T::one();
            }
        }
    }
    
    Ok(Array::from_vec(counts))
}

/// Return the indices of the bins to which each value in input array belongs.
/// 
/// # Parameters
/// 
/// * `x` - Input array
/// * `bins` - Array of bin edges
/// * `right` - Whether the intervals include the right or the left bin edge
/// 
/// # Returns
/// 
/// Array of indices the same shape as x
pub fn digitize<T: Float + Clone + NumCast>(
    x: &Array<T>, 
    bins: &Array<T>,
    right: Option<bool>
) -> Result<Array<usize>> {
    let x_data = x.to_vec();
    let bins_data = bins.to_vec();
    
    if bins_data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Bins array cannot be empty".to_string()
        ));
    }
    
    // Check if bins are monotonic
    let mut increasing = true;
    let mut decreasing = true;
    
    for i in 1..bins_data.len() {
        if bins_data[i] > bins_data[i-1] {
            decreasing = false;
        }
        if bins_data[i] < bins_data[i-1] {
            increasing = false;
        }
    }
    
    if !increasing && !decreasing {
        return Err(NumRs2Error::InvalidOperation(
            "Bins must be monotonically increasing or decreasing".to_string()
        ));
    }
    
    // Determine bin membership
    let use_right = right.unwrap_or(false);
    let mut result = Vec::with_capacity(x_data.len());
    
    if increasing {
        for &val in &x_data {
            let mut idx = 0;
            for (i, &edge) in bins_data.iter().enumerate() {
                if (use_right && val <= edge) || (!use_right && val < edge) {
                    idx = i;
                    break;
                }
                // If we reach the last bin, index is equal to the number of bins
                idx = bins_data.len();
            }
            result.push(idx);
        }
    } else {
        // Bins are decreasing
        for &val in &x_data {
            let mut idx = 0;
            for (i, &edge) in bins_data.iter().enumerate() {
                if (use_right && val >= edge) || (!use_right && val > edge) {
                    idx = i;
                    break;
                }
                // If we reach the last bin, index is equal to the number of bins
                idx = bins_data.len();
            }
            result.push(idx);
        }
    }
    
    Ok(Array::from_vec(result))
}

/// Helper enum to specify bins for histogram2d
pub enum HistBins {
    Single(usize),
    Tuple(usize, usize),
}

impl From<usize> for HistBins {
    fn from(val: usize) -> Self {
        HistBins::Single(val)
    }
}

impl From<(usize, usize)> for HistBins {
    fn from(val: (usize, usize)) -> Self {
        HistBins::Tuple(val.0, val.1)
    }
}