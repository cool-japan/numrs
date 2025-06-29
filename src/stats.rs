use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, Zero};
use rayon::prelude::*;

/// Threshold for using parallel processing (minimum array size)
const PARALLEL_THRESHOLD: usize = 10000;

// Statistical functions
pub trait Statistics<T> {
    fn mean(&self) -> T;
    fn var(&self) -> T;
    fn std(&self) -> T;
    fn min(&self) -> T;
    fn max(&self) -> T;
    fn percentile(&self, q: T) -> T;
}

impl<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync> Statistics<T>
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
        sum / T::from(data.len()).unwrap()
    }

    fn var(&self) -> T {
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
            return a
                .sum_axis(ax)
                .map(|sum| sum.scalar_div(T::from(a.shape()[ax]).unwrap()));
        } else {
            // Calculate overall mean manually
            let data = a.to_vec();
            if data.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "Cannot average empty array".to_string(),
                ));
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

// Additional statistics functions

/// Estimate covariance matrix of variables with parallel processing for large datasets
///
/// # Parameters
///
/// * `x` - A 2-D array containing multiple variables and observations.
///   Each row represents a variable, and each column a single observation
///   of all those variables.
/// * `y` - Optional additional data. If provided, it is appended to x
/// * `rowvar` - If true (default), each row represents a variable, with observations in columns.
///   Otherwise, the relationship is transposed
/// * `bias` - If false (default), use Bessel's correction with ddof=1
/// * `ddof` - Delta degrees of freedom. By default ddof=None -> ddof=1.
///
/// # Returns
///
/// The covariance matrix of the variables.
pub fn cov<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    x: &Array<T>,
    y: Option<&Array<T>>,
    rowvar: Option<bool>,
    bias: Option<bool>,
    ddof: Option<usize>,
) -> Result<Array<T>> {
    let rowvar_val = rowvar.unwrap_or(true);
    let bias_val = bias.unwrap_or(false);
    let ddof_val = if bias_val { 0 } else { ddof.unwrap_or(1) };

    // Prepare data matrix - ensure 2D
    let mut data = if x.ndim() == 1 {
        // Convert 1D array to 2D: (n,) -> (1, n) for rowvar=true or (n, 1) for rowvar=false
        if rowvar_val {
            x.reshape(&[1, x.len()])
        } else {
            x.reshape(&[x.len(), 1])
        }
    } else if rowvar_val {
        x.clone()
    } else {
        // Transpose if columns represent variables
        x.transpose()
    };

    // Append y if provided
    if let Some(y_arr) = y {
        let y_data = if y_arr.ndim() == 1 {
            // Convert 1D array to 2D: (n,) -> (1, n) for rowvar=true or (n, 1) for rowvar=false
            if rowvar_val {
                y_arr.reshape(&[1, y_arr.len()])
            } else {
                y_arr.reshape(&[y_arr.len(), 1])
            }
        } else if rowvar_val {
            y_arr.clone()
        } else {
            y_arr.transpose()
        };

        // Check dimensions match
        let x_obs = data.shape()[1];
        let y_obs = y_data.shape()[1];
        if x_obs != y_obs {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![x_obs],
                actual: vec![y_obs],
            });
        }

        // Concatenate along first dimension (variables)
        data = concatenate(&[&data, &y_data], 0)?;
    }

    let shape = data.shape();
    let n_vars = shape[0];
    let n_obs = shape[1];

    if n_obs <= ddof_val {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Not enough observations ({}) for ddof ({})",
            n_obs, ddof_val
        )));
    }

    // Calculate means for each variable
    let mut means = Vec::with_capacity(n_vars);
    let data_vec = data.to_vec();

    for i in 0..n_vars {
        let mut sum = T::zero();
        for j in 0..n_obs {
            sum = sum + data_vec[i * n_obs + j];
        }
        means.push(sum / T::from(n_obs).unwrap());
    }

    // Calculate covariance matrix with parallel processing for large datasets
    let mut cov_matrix = vec![T::zero(); n_vars * n_vars];
    let factor = T::from(n_obs - ddof_val).unwrap();

    if n_vars * n_obs >= PARALLEL_THRESHOLD {
        // Use parallel processing for large covariance calculations
        let covariances: Vec<(usize, usize, T)> = (0..n_vars)
            .into_par_iter()
            .flat_map(|i| {
                let data_vec_clone = data_vec.clone();
                let means_clone = means.clone();
                (0..=i).into_par_iter().map(move |j| {
                    let sum = (0..n_obs)
                        .into_par_iter()
                        .map(|k| {
                            let xi = data_vec_clone[i * n_obs + k] - means_clone[i];
                            let xj = data_vec_clone[j * n_obs + k] - means_clone[j];
                            xi * xj
                        })
                        .reduce(|| T::zero(), |acc, x| acc + x);
                    let cov_val = sum / factor;
                    (i, j, cov_val)
                })
            })
            .collect();

        // Fill the covariance matrix
        for (i, j, cov_val) in covariances {
            cov_matrix[i * n_vars + j] = cov_val;
            if i != j {
                cov_matrix[j * n_vars + i] = cov_val; // Symmetric
            }
        }
    } else {
        // Use sequential processing for small datasets
        for i in 0..n_vars {
            for j in 0..=i {
                // Only compute lower triangular part
                let mut sum = T::zero();
                for k in 0..n_obs {
                    let xi = data_vec[i * n_obs + k] - means[i];
                    let xj = data_vec[j * n_obs + k] - means[j];
                    sum = sum + xi * xj;
                }
                let cov_val = sum / factor;
                cov_matrix[i * n_vars + j] = cov_val;
                if i != j {
                    cov_matrix[j * n_vars + i] = cov_val; // Symmetric
                }
            }
        }
    }

    Ok(Array::from_vec(cov_matrix).reshape(&[n_vars, n_vars]))
}

/// Return Pearson product-moment correlation coefficients with parallel processing.
///
/// # Parameters
///
/// * `x` - A 2-D array containing multiple variables and observations.
///   Each row represents a variable, and each column a single observation
///   of all those variables.
/// * `y` - Optional additional data. If provided, it is appended to x
/// * `rowvar` - If true (default), each row represents a variable
///
/// # Returns
///
/// The correlation coefficient matrix of the variables.
pub fn corrcoef<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    x: &Array<T>,
    y: Option<&Array<T>>,
    rowvar: Option<bool>,
) -> Result<Array<T>> {
    // Get covariance matrix
    let c = cov(x, y, rowvar, Some(false), None)?;

    // Get standard deviations (diagonal of covariance matrix)
    let shape = c.shape();
    let n = shape[0];
    let c_vec = c.to_vec();

    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        let var = c_vec[i * n + i];
        if var < T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "Negative variance encountered".to_string(),
            ));
        }
        d.push(var.sqrt());
    }

    // Normalize to get correlation coefficients with parallel processing for large matrices
    let mut corr_matrix = vec![T::zero(); n * n];

    if n * n >= PARALLEL_THRESHOLD {
        // Use parallel processing for large correlation matrices
        let correlations: Vec<(usize, T)> = (0..n * n)
            .into_par_iter()
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                let corr_val = if d[i] == T::zero() || d[j] == T::zero() {
                    // Handle zero variance
                    if i == j {
                        T::one()
                    } else {
                        T::zero()
                    }
                } else {
                    c_vec[i * n + j] / (d[i] * d[j])
                };
                (idx, corr_val)
            })
            .collect();

        for (idx, corr_val) in correlations {
            corr_matrix[idx] = corr_val;
        }
    } else {
        // Use sequential processing for small matrices
        for i in 0..n {
            for j in 0..n {
                if d[i] == T::zero() || d[j] == T::zero() {
                    // Handle zero variance
                    if i == j {
                        corr_matrix[i * n + j] = T::one();
                    } else {
                        corr_matrix[i * n + j] = T::zero();
                    }
                } else {
                    corr_matrix[i * n + j] = c_vec[i * n + j] / (d[i] * d[j]);
                }
            }
        }
    }

    Ok(Array::from_vec(corr_matrix).reshape(&[n, n]))
}

use crate::array_ops::joining::concatenate;

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
pub fn quantile<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    a: &Array<T>,
    q: &Array<T>,
    method: Option<&str>,
) -> Result<Array<T>> {
    let method_str = method.unwrap_or("linear");
    let data = a.to_vec();

    if data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute quantiles of an empty array".to_string(),
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
            return Err(NumRs2Error::InvalidOperation(format!(
                "Quantile value {} out of bounds [0, 1]",
                q_val
            )));
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
pub fn percentile<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    a: &Array<T>,
    q: &Array<T>,
    method: Option<&str>,
) -> Result<Array<T>> {
    // Convert percentiles to quantiles (0-100 to 0-1)
    let quantiles = q.map(|x| x / T::from(100.0).unwrap());

    // Call quantile with the converted values
    quantile(a, &quantiles, method)
}

/// Calculate a histogram of a dataset with parallel processing for large arrays
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
pub fn histogram<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    a: &Array<T>,
    bins: usize,
    range: Option<(T, T)>,
    weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>)> {
    let data = a.to_vec();
    if data.is_empty() || bins == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram of an empty array or with zero bins".to_string(),
        ));
    }

    // Get min and max values - either from range parameter or from data
    let (min_val, max_val) = match range {
        Some((min, max)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => (a.min(), a.max()),
    };

    // Create bin edges
    let step = (max_val - min_val) / T::from(bins).unwrap();
    let mut bin_edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        bin_edges.push(min_val + step * T::from(i).unwrap());
    }

    // Count values in each bin with optional weights using parallel processing for large datasets
    let mut counts = vec![T::zero(); bins];

    if data.len() >= PARALLEL_THRESHOLD {
        // Use parallel processing for large datasets
        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            // Parallel reduction to compute bin counts with weights
            let bin_contributions: Vec<Vec<T>> = data
                .par_iter()
                .zip(weights_data.par_iter())
                .enumerate()
                .map(|(_i, (&val, &weight))| {
                    let mut local_counts = vec![T::zero(); bins];

                    if val >= min_val && val <= max_val {
                        if val == max_val {
                            // Handle edge case for the maximum value
                            local_counts[bins - 1] = weight;
                        } else {
                            let bin_idx = ((val - min_val) / step).to_usize().unwrap();
                            if bin_idx < bins {
                                local_counts[bin_idx] = weight;
                            }
                        }
                    }
                    local_counts
                })
                .collect();

            // Sum up all contributions
            for contribution in bin_contributions {
                for (i, &contrib) in contribution.iter().enumerate() {
                    counts[i] = counts[i] + contrib;
                }
            }
        } else {
            // No weights - parallel counting
            let bin_contributions: Vec<Vec<T>> = data
                .par_iter()
                .map(|&val| {
                    let mut local_counts = vec![T::zero(); bins];

                    if val >= min_val && val <= max_val {
                        if val == max_val {
                            // Handle edge case for the maximum value
                            local_counts[bins - 1] = T::one();
                        } else {
                            let bin_idx = ((val - min_val) / step).to_usize().unwrap();
                            if bin_idx < bins {
                                local_counts[bin_idx] = T::one();
                            }
                        }
                    }
                    local_counts
                })
                .collect();

            // Sum up all contributions
            for contribution in bin_contributions {
                for (i, &contrib) in contribution.iter().enumerate() {
                    counts[i] = counts[i] + contrib;
                }
            }
        }
    } else {
        // Use sequential processing for small datasets
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
                    continue; // Skip values outside the range
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
                    continue; // Skip values outside the range
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
    }

    Ok((Array::from_vec(counts), Array::from_vec(bin_edges)))
}

/// Calculate a 2D histogram of a dataset
///
/// # Parameters
///
/// * `x` - Input array for x coordinates
/// * `y` - Input array for y coordinates
/// * `bins` - Either a tuple (nx, ny) to specify bins in each dimension,
///   or a single value to use the same number of bins in both dimensions
/// * `range` - Optional tuple ((xmin, xmax), (ymin, ymax)) to use for bin edges
/// * `weights` - Optional array of weights for each value
///
/// # Returns
///
/// A tuple of (histogram counts, x_edges, y_edges)
pub fn histogram2d<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    x: &Array<T>,
    y: &Array<T>,
    bins: impl Into<HistBins>,
    range: Option<((T, T), (T, T))>,
    weights: Option<&Array<T>>,
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
            "Cannot compute histogram2d with empty arrays or zero bins".to_string(),
        ));
    }

    // Get min and max values - either from range parameter or from data
    let (x_min, x_max) = match range {
        Some(((min, max), _)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "X range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => {
            let x_data = x.to_vec();
            let x_min = x_data
                .iter()
                .fold(x_data[0], |acc, &val| if val < acc { val } else { acc });
            let x_max = x_data
                .iter()
                .fold(x_data[0], |acc, &val| if val > acc { val } else { acc });
            (x_min, x_max)
        }
    };

    let (y_min, y_max) = match range {
        Some((_, (min, max))) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Y range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => {
            let y_data = y.to_vec();
            let y_min = y_data
                .iter()
                .fold(y_data[0], |acc, &val| if val < acc { val } else { acc });
            let y_max = y_data
                .iter()
                .fold(y_data[0], |acc, &val| if val > acc { val } else { acc });
            (y_min, y_max)
        }
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
                continue; // Skip values outside the range
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
                continue; // Skip values outside the range
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
        Array::from_vec(y_edges),
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
pub fn bincount<T: Float + Clone + NumCast + Send + Sync>(
    a: &Array<T>,
    weights: Option<&Array<T>>,
    minlength: Option<usize>,
) -> Result<Array<T>> {
    let data = a.to_vec();

    if data.is_empty() {
        let min_len = minlength.unwrap_or(0);
        let counts = vec![T::zero(); min_len];
        return Ok(Array::from_vec(counts));
    }

    // Find the maximum value to determine the output array size
    let data_cloned = data.clone();
    let max_val = data_cloned.iter().fold(
        data_cloned[0],
        |max, &val| if val > max { val } else { max },
    );
    if max_val < T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "All values in bincount input array must be non-negative".to_string(),
        ));
    }

    let max_idx = max_val.to_usize().unwrap();
    let min_length = minlength.unwrap_or(0);
    let bin_count = if max_idx + 1 > min_length {
        max_idx + 1
    } else {
        min_length
    };

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
                    "All values in bincount input array must be non-negative".to_string(),
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
                    "All values in bincount input array must be non-negative".to_string(),
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
pub fn digitize<T: Float + Clone + NumCast + Send + Sync>(
    x: &Array<T>,
    bins: &Array<T>,
    right: Option<bool>,
) -> Result<Array<usize>> {
    let x_data = x.to_vec();
    let bins_data = bins.to_vec();

    if bins_data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Bins array cannot be empty".to_string(),
        ));
    }

    // Check if bins are monotonic
    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..bins_data.len() {
        if bins_data[i] > bins_data[i - 1] {
            decreasing = false;
        }
        if bins_data[i] < bins_data[i - 1] {
            increasing = false;
        }
    }

    if !increasing && !decreasing {
        return Err(NumRs2Error::InvalidOperation(
            "Bins must be monotonically increasing or decreasing".to_string(),
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

/// Calculate a multi-dimensional histogram of a dataset
///
/// # Parameters
///
/// * `sample` - Array of shape (N, D) containing N samples in D dimensions
/// * `bins` - Number of bins for each dimension. Can be:
///   - A single usize: Same number of bins for all dimensions
///   - A vector of usize: Different number of bins for each dimension
/// * `range` - Optional vector of (min, max) tuples for each dimension
/// * `weights` - Optional array of weights for each sample
///
/// # Returns
///
/// A tuple of (histogram counts, vector of bin edges for each dimension)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 2D data points
/// let data = Array::from_vec(vec![
///     0.0, 0.0,
///     0.5, 0.5,
///     1.0, 1.0,
///     0.3, 0.7,
/// ]).reshape(&[4, 2]);
///
/// // Compute 2D histogram with 2 bins in each dimension
/// let (hist, edges) = histogram_dd(&data, &[2, 2], None, None).unwrap();
/// assert_eq!(hist.shape(), vec![2, 2]);
/// ```
pub fn histogram_dd<T: Float + Clone + NumCast + std::fmt::Display>(
    sample: &Array<T>,
    bins: &[usize],
    range: Option<Vec<(T, T)>>,
    weights: Option<&Array<T>>,
) -> Result<(Array<T>, Vec<Array<T>>)> {
    let shape = sample.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "histogram_dd requires 2D input array of shape (N, D)".to_string(),
        ));
    }

    let n_samples = shape[0];
    let n_dims = shape[1];

    if n_samples == 0 || n_dims == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram of empty data".to_string(),
        ));
    }

    // Validate bins
    if bins.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "bins array cannot be empty".to_string(),
        ));
    }

    let bin_counts = if bins.len() == 1 {
        // Use same number of bins for all dimensions
        vec![bins[0]; n_dims]
    } else if bins.len() == n_dims {
        bins.to_vec()
    } else {
        return Err(NumRs2Error::InvalidOperation(format!(
            "bins length {} does not match number of dimensions {}",
            bins.len(),
            n_dims
        )));
    };

    // Check for zero bins
    for &b in &bin_counts {
        if b == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Number of bins must be greater than 0".to_string(),
            ));
        }
    }

    // Validate weights if provided
    if let Some(w) = weights {
        if w.shape()[0] != n_samples {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n_samples],
                actual: w.shape().to_vec(),
            });
        }
    }

    // Determine ranges for each dimension
    let mut ranges = Vec::with_capacity(n_dims);
    let sample_data = sample.to_vec();

    if let Some(r) = range {
        if r.len() != n_dims {
            return Err(NumRs2Error::InvalidOperation(format!(
                "range length {} does not match number of dimensions {}",
                r.len(),
                n_dims
            )));
        }
        ranges = r;
    } else {
        // Compute min and max for each dimension
        for d in 0..n_dims {
            let mut min_val = sample_data[d];
            let mut max_val = sample_data[d];

            for i in 0..n_samples {
                let val = sample_data[i * n_dims + d];
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            // Add small epsilon to max to ensure last value is included
            let epsilon = T::from(1e-10).unwrap();
            max_val = max_val + epsilon;

            ranges.push((min_val, max_val));
        }
    }

    // Create bin edges for each dimension
    let mut edges = Vec::with_capacity(n_dims);
    let mut bin_steps = Vec::with_capacity(n_dims);

    for (d, &n_bins) in bin_counts.iter().enumerate() {
        let (min_val, max_val) = ranges[d];
        let step = (max_val - min_val) / T::from(n_bins).unwrap();
        bin_steps.push(step);

        let mut dim_edges = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            dim_edges.push(min_val + step * T::from(i).unwrap());
        }
        edges.push(Array::from_vec(dim_edges));
    }

    // Initialize multi-dimensional histogram
    let hist_shape: Vec<usize> = bin_counts.clone();
    let total_bins: usize = hist_shape.iter().product();
    let mut hist_data = vec![T::zero(); total_bins];

    // Helper function to convert multi-dimensional indices to linear index
    let indices_to_linear = |indices: &[usize]| -> usize {
        let mut linear = 0;
        let mut stride = 1;
        for i in (0..n_dims).rev() {
            linear += indices[i] * stride;
            stride *= hist_shape[i];
        }
        linear
    };

    // Fill the histogram
    if let Some(w) = weights {
        let weights_data = w.to_vec();

        for i in 0..n_samples {
            let mut indices = Vec::with_capacity(n_dims);
            let mut in_bounds = true;

            for d in 0..n_dims {
                let val = sample_data[i * n_dims + d];
                let (min_val, max_val) = ranges[d];

                if val < min_val || val > max_val {
                    in_bounds = false;
                    break;
                }

                let mut idx = ((val - min_val) / bin_steps[d]).to_usize().unwrap();
                // Handle edge case where value equals max
                if idx >= bin_counts[d] {
                    idx = bin_counts[d] - 1;
                }
                indices.push(idx);
            }

            if in_bounds {
                let linear_idx = indices_to_linear(&indices);
                hist_data[linear_idx] = hist_data[linear_idx] + weights_data[i];
            }
        }
    } else {
        // No weights, just count
        for i in 0..n_samples {
            let mut indices = Vec::with_capacity(n_dims);
            let mut in_bounds = true;

            for d in 0..n_dims {
                let val = sample_data[i * n_dims + d];
                let (min_val, max_val) = ranges[d];

                if val < min_val || val > max_val {
                    in_bounds = false;
                    break;
                }

                let mut idx = ((val - min_val) / bin_steps[d]).to_usize().unwrap();
                // Handle edge case where value equals max
                if idx >= bin_counts[d] {
                    idx = bin_counts[d] - 1;
                }
                indices.push(idx);
            }

            if in_bounds {
                let linear_idx = indices_to_linear(&indices);
                hist_data[linear_idx] = hist_data[linear_idx] + T::one();
            }
        }
    }

    // Create the histogram array with proper shape
    let hist = Array::from_vec(hist_data).reshape(&hist_shape);

    Ok((hist, edges))
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
/// let result = nanmean(&a, None, false).unwrap();
/// assert_eq!(result.to_vec()[0], 8.0 / 3.0); // (1 + 3 + 4) / 3
/// ```
pub fn nanmean<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    _keepdims: bool,
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
                    let mean = sum / T::from(count).unwrap();
                    Ok(Array::from_vec(vec![mean]))
                }
            } else {
                // Use sequential processing for small arrays
                let filtered: Vec<T> = data.into_iter().filter(|x| !x.is_nan()).collect();

                if filtered.is_empty() {
                    Ok(Array::from_vec(vec![T::nan()]))
                } else {
                    let sum = filtered.iter().fold(T::zero(), |acc, &x| acc + x);
                    let mean = sum / T::from(filtered.len()).unwrap();
                    Ok(Array::from_vec(vec![mean]))
                }
            }
        }
        Some(_) => {
            // For now, implement simple version - could be extended for axis support
            nanmean(array, None, _keepdims)
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
/// let result = nanstd(&a, None, Some(0), false).unwrap();
/// // Standard deviation of [1, 3, 4]
/// ```
pub fn nanstd<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    ddof: Option<usize>,
    _keepdims: bool,
) -> Result<Array<T>> {
    let variance = nanvar(array, axis, ddof, _keepdims)?;
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
/// let result = nanvar(&a, None, Some(0), false).unwrap();
/// // Variance of [1, 3, 4]
/// ```
pub fn nanvar<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    ddof: Option<usize>,
    _keepdims: bool,
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
                    let mean = sum / T::from(count).unwrap();

                    let sum_squared_diff = data
                        .par_iter()
                        .filter(|x| !x.is_nan())
                        .map(|&x| (x - mean) * (x - mean))
                        .reduce(|| T::zero(), |acc, x| acc + x);

                    let variance = sum_squared_diff / T::from(count - ddof_val).unwrap();
                    Ok(Array::from_vec(vec![variance]))
                }
            } else {
                // Use sequential processing for small arrays
                let filtered: Vec<T> = data.into_iter().filter(|x| !x.is_nan()).collect();

                if filtered.len() <= ddof_val {
                    Ok(Array::from_vec(vec![T::nan()]))
                } else {
                    let mean = filtered.iter().fold(T::zero(), |acc, &x| acc + x)
                        / T::from(filtered.len()).unwrap();

                    let sum_squared_diff = filtered
                        .iter()
                        .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean));

                    let variance = sum_squared_diff / T::from(filtered.len() - ddof_val).unwrap();
                    Ok(Array::from_vec(vec![variance]))
                }
            }
        }
        Some(_) => {
            // For now, implement simple version - could be extended for axis support
            nanvar(array, None, ddof, _keepdims)
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
/// let result = nanmin(&a, None, false).unwrap();
/// assert_eq!(result.to_vec()[0], 0.5);
/// ```
pub fn nanmin<T: Float + Clone + Zero + NumCast + std::fmt::Display>(
    array: &Array<T>,
    axis: Option<usize>,
    _keepdims: bool,
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
        Some(_) => {
            // For now, implement simple version
            nanmin(array, None, _keepdims)
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
/// let result = nanmax(&a, None, false).unwrap();
/// assert_eq!(result.to_vec()[0], 3.0);
/// ```
pub fn nanmax<T: Float + Clone + Zero + NumCast + std::fmt::Display>(
    array: &Array<T>,
    axis: Option<usize>,
    _keepdims: bool,
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
        Some(_) => {
            // For now, implement simple version
            nanmax(array, None, _keepdims)
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
/// let result = nansum(&a, None, false).unwrap();
/// assert_eq!(result.to_vec()[0], 6.0); // 1 + 3 + 2
/// ```
pub fn nansum<T: Float + Clone + Zero + NumCast + std::fmt::Display + Send + Sync>(
    array: &Array<T>,
    axis: Option<usize>,
    _keepdims: bool,
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
        Some(_) => {
            // For now, implement simple version
            nansum(array, None, _keepdims)
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
/// let result = nanprod(&a, None, false).unwrap();
/// assert_eq!(result.to_vec()[0], 24.0); // 2 * 3 * 4
/// ```
pub fn nanprod<T: Float + Clone + Zero + NumCast + std::fmt::Display>(
    array: &Array<T>,
    axis: Option<usize>,
    _keepdims: bool,
) -> Result<Array<T>> {
    match axis {
        None => {
            let data = array.to_vec();
            let product = data
                .iter()
                .fold(T::one(), |acc, &x| if x.is_nan() { acc } else { acc * x });
            Ok(Array::from_vec(vec![product]))
        }
        Some(_) => {
            // For now, implement simple version
            nanprod(array, None, _keepdims)
        }
    }
}

/// Compute the mode (most frequent value) of an array
///
/// The mode is the value that appears most often in a dataset. This implementation
/// returns the most frequent value along with its count. For arrays with multiple
/// modes (equally frequent values), the smallest value is returned.
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute the mode (None for flattened array)
/// * `nan_policy` - How to handle NaN values:
///   - "propagate": Return NaN if any NaN values are present (default)
///   - "omit": Ignore NaN values in computation
///   - "raise": Raise an error if any NaN values are present
///
/// # Returns
///
/// A tuple of (mode, count) where:
/// - mode: Array containing the most frequent values
/// - count: Array containing the counts of the mode values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::mode;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0, 1.0]);
/// let (mode_val, count) = mode(&a, None, None).unwrap();
/// assert_eq!(mode_val.to_vec()[0], 1.0);  // 1.0 appears 3 times
/// assert_eq!(count.to_vec()[0], 3.0);     // Count is 3
///
/// // Example with multiple values having same frequency
/// let b = Array::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
/// let (mode_val, count) = mode(&b, None, None).unwrap();
/// assert_eq!(mode_val.to_vec()[0], 1.0);  // Smallest value with max frequency
/// assert_eq!(count.to_vec()[0], 2.0);     // Each appears 2 times
/// ```
pub fn mode<T>(
    array: &Array<T>,
    axis: Option<usize>,
    nan_policy: Option<&str>,
) -> Result<(Array<T>, Array<T>)>
where
    T: Float + Clone + PartialOrd + std::fmt::Display + NumCast,
{
    let policy = nan_policy.unwrap_or("propagate");

    match axis {
        None => {
            // Flatten the array and compute mode
            let data = array.to_vec();

            if data.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "Cannot compute mode of empty array".to_string(),
                ));
            }

            // Handle NaN policy
            let filtered_data: Vec<T> = match policy {
                "propagate" => {
                    // Check if any NaN values exist
                    if data.iter().any(|x| x.is_nan()) {
                        return Ok((
                            Array::from_vec(vec![T::nan()]),
                            Array::from_vec(vec![T::zero()]),
                        ));
                    }
                    data
                }
                "omit" => {
                    // Filter out NaN values
                    data.into_iter().filter(|x| !x.is_nan()).collect()
                }
                "raise" => {
                    // Check for NaN values and raise error if found
                    if data.iter().any(|x| x.is_nan()) {
                        return Err(NumRs2Error::InvalidOperation(
                            "NaN values found in array with nan_policy='raise'".to_string(),
                        ));
                    }
                    data
                }
                _ => {
                    return Err(NumRs2Error::InvalidOperation(format!(
                        "Invalid nan_policy '{}'. Use 'propagate', 'omit', or 'raise'",
                        policy
                    )));
                }
            };

            if filtered_data.is_empty() {
                return Err(NumRs2Error::InvalidOperation(
                    "No valid (non-NaN) values found".to_string(),
                ));
            }

            // Count frequency of each value
            use std::collections::HashMap;
            let mut counts: HashMap<String, (T, usize)> = HashMap::new();

            for &value in &filtered_data {
                let key = format!("{:.15}", value); // Use string key for floating point comparison
                let entry = counts.entry(key).or_insert((value, 0));
                entry.1 += 1;
            }

            // Find the value(s) with maximum frequency
            let max_count = counts.values().map(|(_, count)| *count).max().unwrap();

            // Among values with max frequency, find the smallest one
            let mut mode_candidates: Vec<T> = counts
                .values()
                .filter(|(_, count)| *count == max_count)
                .map(|(value, _)| *value)
                .collect();

            mode_candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mode_value = mode_candidates[0];
            let mode_count = T::from(max_count).unwrap();

            Ok((
                Array::from_vec(vec![mode_value]),
                Array::from_vec(vec![mode_count]),
            ))
        }
        Some(axis_val) => {
            // For axis-specific mode computation
            let shape = array.shape();
            if axis_val >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis_val,
                    shape.len()
                )));
            }

            // This is a simplified implementation - for a full implementation,
            // we would need to iterate along the specified axis
            // For now, fall back to the flattened version
            mode(array, None, nan_policy)
        }
    }
}
