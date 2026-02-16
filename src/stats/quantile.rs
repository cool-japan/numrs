//! Quantile and percentile functions
//!
//! This module provides quantile and percentile calculation:
//! - quantile: Compute quantiles of a dataset
//! - percentile: Compute percentiles of a dataset

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};

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
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        let idx_float = q_val * T::from(n - 1).expect("n-1 should be representable");
        let idx_lower = idx_float.floor();
        let idx_upper = idx_float.ceil();
        let idx_lower_usize = idx_lower
            .to_usize()
            .expect("lower index should be convertible");
        let idx_upper_usize = idx_upper
            .to_usize()
            .expect("upper index should be convertible");

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
                    (lower_val + upper_val) / T::from(2.0).expect("2.0 should be representable")
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
    let quantiles = q.map(|x| x / T::from(100.0).expect("100.0 should be representable"));

    // Call quantile with the converted values
    quantile(a, &quantiles, method)
}
