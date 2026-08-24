//! Correlation and covariance functions
//!
//! This module provides correlation and covariance calculation:
//! - cov: Covariance matrix estimation
//! - corrcoef: Pearson correlation coefficients

use crate::array::Array;
use crate::array_ops::joining::concatenate;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, Zero};
use scirs2_core::parallel_ops::*;

use super::basic::PARALLEL_THRESHOLD;

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
        means.push(sum / T::from(n_obs).expect("n_obs should be representable"));
    }

    // Calculate covariance matrix with parallel processing for large datasets
    let mut cov_matrix = vec![T::zero(); n_vars * n_vars];
    let factor = T::from(n_obs - ddof_val).expect("n_obs-ddof should be representable");

    if n_vars * n_obs >= PARALLEL_THRESHOLD {
        // Generate all unique pairs (i, j) where j <= i
        let pairs: Vec<(usize, usize)> = (0..n_vars)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .collect();

        // Use parallel processing for large covariance calculations
        // Process pairs in parallel, but compute inner sums sequentially for better cache locality
        let covariances: Vec<(usize, usize, T)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                // Sequential sum for better cache access pattern
                let mut sum = T::zero();
                let mean_i = means[i];
                let mean_j = means[j];
                let base_i = i * n_obs;
                let base_j = j * n_obs;
                for k in 0..n_obs {
                    let xi = data_vec[base_i + k] - mean_i;
                    let xj = data_vec[base_j + k] - mean_j;
                    sum = sum + xi * xj;
                }
                let cov_val = sum / factor;
                (i, j, cov_val)
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

    Array::from_vec_shape(cov_matrix, &[n_vars, n_vars])
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

    Array::from_vec_shape(corr_matrix, &[n, n])
}
