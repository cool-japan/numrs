//! Net Present Value Calculations
//!
//! This module implements the net present value (NPV) calculation function,
//! compatible with NumPy's financial functions.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, Zero, One};
use std::fmt::Debug;

/// Calculate the net present value of a cash flow series.
///
/// The net present value is calculated as:
/// ```
/// NPV = CF₀ + CF₁/(1+rate)¹ + CF₂/(1+rate)² + ... + CFₙ/(1+rate)ⁿ
/// ```
///
/// where CF₀ is the initial cash flow (typically negative for an investment),
/// and CF₁, CF₂, ..., CFₙ are the subsequent cash flows.
///
/// # Arguments
///
/// * `rate` - Discount rate per period
/// * `values` - Array of cash flows, where the first value is the initial investment
///
/// # Returns
///
/// Net present value of the cash flows
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::npv;
///
/// // Calculate NPV for an investment
/// let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0, 600.0]);
/// let result = npv(0.1, &cash_flows).unwrap();
/// assert!((result - 447.07).abs() < 0.01);
/// ```
pub fn npv<T>(rate: T, values: &Array<T>) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if rate.is_nan() {
        return Err(NumRs2Error::ComputationError(
            "Discount rate cannot be NaN".to_string(),
        ));
    }

    if rate.is_infinite() {
        return Err(NumRs2Error::ComputationError(
            "Discount rate cannot be infinite".to_string(),
        ));
    }

    let cash_flows = values.to_vec();
    
    if cash_flows.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "Cash flow array cannot be empty".to_string(),
        ));
    }

    let mut npv_value = T::zero();
    let one_plus_rate = T::one() + rate;
    
    for (i, &cash_flow) in cash_flows.iter().enumerate() {
        if cash_flow.is_nan() {
            return Err(NumRs2Error::ComputationError(
                format!("Cash flow at index {} is NaN", i),
            ));
        }
        
        if cash_flow.is_infinite() {
            return Err(NumRs2Error::ComputationError(
                format!("Cash flow at index {} is infinite", i),
            ));
        }

        if i == 0 {
            // Initial cash flow is not discounted
            npv_value = npv_value + cash_flow;
        } else {
            // Discount future cash flows
            let discount_factor = one_plus_rate.powf(T::from(i).unwrap());
            if discount_factor.is_zero() {
                return Err(NumRs2Error::ComputationError(
                    "Discount factor became zero (rate too negative)".to_string(),
                ));
            }
            npv_value = npv_value + cash_flow / discount_factor;
        }
    }

    Ok(npv_value)
}

/// Calculate NPV for multiple scenarios with different rates
///
/// # Arguments
///
/// * `rates` - Array of discount rates per period
/// * `values` - Array of cash flows
///
/// # Returns
///
/// Array of net present values, one for each rate
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::npv_rates;
///
/// let rates = Array::from_vec(vec![0.05, 0.10, 0.15]);
/// let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
/// let result = npv_rates(&rates, &cash_flows).unwrap();
/// assert_eq!(result.shape(), vec![3]);
/// ```
pub fn npv_rates<T>(rates: &Array<T>, values: &Array<T>) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    let rate_vec = rates.to_vec();
    let mut result_vec = Vec::with_capacity(rate_vec.len());

    for &rate in &rate_vec {
        let npv_result = npv(rate, values)?;
        result_vec.push(npv_result);
    }

    Ok(Array::from_vec(result_vec).reshape(&rates.shape()))
}

/// Calculate NPV for multiple cash flow series with the same rate
///
/// # Arguments
///
/// * `rate` - Discount rate per period
/// * `values_matrix` - 2D array where each row represents a different cash flow series
///
/// # Returns
///
/// Array of net present values, one for each cash flow series
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::npv_multiple_series;
///
/// // Two investment projects
/// let cash_flows = Array::from_vec(vec![
///     -1000.0, 300.0, 400.0, 500.0,  // Project 1
///     -1200.0, 400.0, 500.0, 600.0   // Project 2
/// ]).reshape(&[2, 4]);
/// let result = npv_multiple_series(0.1, &cash_flows).unwrap();
/// assert_eq!(result.shape(), vec![2]);
/// ```
pub fn npv_multiple_series<T>(rate: T, values_matrix: &Array<T>) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    if values_matrix.ndim() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Values matrix must be 2-dimensional".to_string(),
        ));
    }

    let shape = values_matrix.shape();
    let rows = shape[0];
    let cols = shape[1];
    let data = values_matrix.to_vec();

    let mut result_vec = Vec::with_capacity(rows);

    for i in 0..rows {
        let row_start = i * cols;
        let row_end = row_start + cols;
        let row_data = data[row_start..row_end].to_vec();
        let row_array = Array::from_vec(row_data);
        
        let npv_result = npv(rate, &row_array)?;
        result_vec.push(npv_result);
    }

    Ok(Array::from_vec(result_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_npv_basic() {
        // Test basic NPV calculation
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0, 600.0]);
        let result = npv(0.1, &cash_flows).unwrap();
        assert_relative_eq!(result, 388.771259, epsilon = 1e-5);
    }

    #[test]
    fn test_npv_zero_rate() {
        // Test NPV with zero discount rate
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
        let result = npv(0.0, &cash_flows).unwrap();
        assert_relative_eq!(result, 200.0, epsilon = 1e-9);
    }

    #[test]
    fn test_npv_negative_rate() {
        // Test NPV with negative discount rate
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0]);
        let result = npv(-0.05, &cash_flows).unwrap();
        // With negative rate, future cash flows are discounted at negative rate
        // Expected: -1000 + 300/0.95 + 400/0.95^2 = -240.997
        assert_relative_eq!(result, -240.997230, epsilon = 1e-5);
    }

    #[test]
    fn test_npv_single_cash_flow() {
        // Test NPV with only initial investment
        let cash_flows = Array::from_vec(vec![-1000.0]);
        let result = npv(0.1, &cash_flows).unwrap();
        assert_relative_eq!(result, -1000.0, epsilon = 1e-9);
    }

    #[test]
    fn test_npv_positive_initial() {
        // Test NPV with positive initial cash flow
        let cash_flows = Array::from_vec(vec![1000.0, -300.0, -400.0, -500.0]);
        let result = npv(0.1, &cash_flows).unwrap();
        // Expected: 1000 - 300/1.1 - 400/1.1^2 - 500/1.1^3 = 21.037
        assert_relative_eq!(result, 21.036814, epsilon = 1e-5);
    }

    #[test]
    fn test_npv_rates() {
        let rates = Array::from_vec(vec![0.05, 0.10, 0.15]);
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
        let result = npv_rates(&rates, &cash_flows).unwrap();
        assert_eq!(result.shape(), vec![3]);
        
        let values = result.to_vec();
        // Higher discount rates should give lower NPVs
        assert!(values[0] > values[1]);
        assert!(values[1] > values[2]);
    }

    #[test]
    fn test_npv_multiple_series() {
        let cash_flows = Array::from_vec(vec![
            -1000.0, 300.0, 400.0, 500.0,  // Project 1
            -1200.0, 400.0, 500.0, 600.0   // Project 2
        ]).reshape(&[2, 4]);
        let result = npv_multiple_series(0.1, &cash_flows).unwrap();
        assert_eq!(result.shape(), vec![2]);
        
        let values = result.to_vec();
        assert_relative_eq!(values[0], -21.036814, epsilon = 1e-5); // Project 1 NPV
        assert_relative_eq!(values[1], 27.648385, epsilon = 1e-5); // Project 2 NPV
    }

    #[test]
    fn test_npv_error_cases() {
        // Test with empty array
        let empty_flows = Array::from_vec(Vec::<f64>::new());
        assert!(npv(0.1, &empty_flows).is_err());

        // Test with NaN rate
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0]);
        assert!(npv(f64::NAN, &cash_flows).is_err());

        // Test with infinite rate
        assert!(npv(f64::INFINITY, &cash_flows).is_err());
    }
}