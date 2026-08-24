//! Number of Periods Calculations
//!
//! This module implements the number of periods (NPER) calculation function,
//! compatible with NumPy's financial functions.

use super::validate_financial_params;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Calculate the number of periodic payments.
///
/// The number of periods is computed by solving the following equation:
/// ```text
/// pv + pmt * [(1 + rate)^nper - 1] / rate * (1 + rate)^(-nper) + fv * (1 + rate)^(-nper) = 0
/// ```
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `pmt` - Payment made each period; this value is assumed to be the same each period
/// * `pv` - Present value (the principal amount)
/// * `fv` - Future value, the desired balance after the last payment (default is 0)
/// * `when` - When payments are due ('begin' (1) or 'end' (0) of each period)
///
/// # Returns
///
/// Number of periods
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Calculate number of periods to pay off a loan
/// let result = nper(0.05/12.0, -188.71, 10000.0, 0.0, 0).expect("nper calculation failed");
/// assert!((result - 60.0_f64).abs() < 0.01);
/// ```
pub fn nper<T>(rate: T, pmt: T, pv: T, fv: T, when: i32) -> Result<T>
where
    T: Float + Debug + Clone,
{
    validate_financial_params(rate, T::zero(), pmt, pv, fv)?;

    let when_factor = if when == 1 { T::one() + rate } else { T::one() };

    if rate.is_zero() {
        // Special case when rate is 0
        if pmt.is_zero() {
            return Err(NumRs2Error::ComputationError(
                "Cannot calculate number of periods when both rate and payment are zero"
                    .to_string(),
            ));
        }
        return Ok(-(pv + fv) / pmt);
    }

    // Adjust payment for when factor
    let adjusted_pmt = pmt * when_factor;

    if adjusted_pmt.is_zero() {
        // No payment case - simple compound interest
        if pv.is_zero() || (pv < T::zero()) == (fv < T::zero()) {
            return Err(NumRs2Error::ComputationError(
                "Present value and future value must have opposite signs when payment is zero"
                    .to_string(),
            ));
        }
        let compound_ratio = -fv / pv;
        if compound_ratio <= T::zero() {
            return Err(NumRs2Error::ComputationError(
                "Invalid compound ratio for zero payment case".to_string(),
            ));
        }
        return Ok(compound_ratio.ln() / (T::one() + rate).ln());
    }

    // General case with payments
    // Using the standard NPER formula: log((pmt - fv*rate) / (pmt + pv*rate)) / log(1 + rate)
    // But we need to handle the signs correctly for the loan case
    let pmt_fv_term = adjusted_pmt - fv * rate;
    let pmt_pv_term = adjusted_pmt + pv * rate;

    // Check for division by zero
    let epsilon = T::from(1e-15).expect("Failed to convert epsilon value");
    if pmt_pv_term.abs() < epsilon {
        return Err(NumRs2Error::ComputationError(
            "Division by zero in number of periods calculation".to_string(),
        ));
    }

    let ratio = pmt_fv_term / pmt_pv_term;

    // Check if ratio is positive (required for logarithm)
    if ratio <= T::zero() {
        return Err(NumRs2Error::ComputationError(
            "Invalid ratio in number of periods calculation - check cash flow signs".to_string(),
        ));
    }

    let log_ratio = ratio.ln();
    let log_one_plus_rate = (T::one() + rate).ln();

    Ok(log_ratio / log_one_plus_rate)
}

/// Calculate number of periods for arrays of inputs
///
/// # Arguments
///
/// * `rate` - Array of interest rates per period
/// * `pmt` - Array of payments made each period
/// * `pv` - Array of present values
/// * `fv` - Array of future values
/// * `when` - When payments are due (0 for end, 1 for beginning)
///
/// # Returns
///
/// Array of number of periods
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::nper_array;
///
/// let rates = Array::from_vec(vec![0.05/12.0, 0.06/12.0, 0.07/12.0]);
/// let pmts = Array::from_vec(vec![-188.71, -250.0, -300.0]);
/// let pvs = Array::from_vec(vec![10000.0, 12000.0, 15000.0]);
/// let fvs = Array::from_vec(vec![0.0, 0.0, 0.0]);
///
/// let result = nper_array(&rates, &pmts, &pvs, &fvs, 0).expect("nper_array calculation failed");
/// assert_eq!(result.shape(), vec![3]);
/// ```
pub fn nper_array<T>(
    rate: &Array<T>,
    pmt: &Array<T>,
    pv: &Array<T>,
    fv: &Array<T>,
    when: i32,
) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    // Check that all arrays have the same shape
    if rate.shape() != pmt.shape() || rate.shape() != pv.shape() || rate.shape() != fv.shape() {
        return Err(NumRs2Error::DimensionMismatch(
            "All input arrays must have the same shape".to_string(),
        ));
    }

    let rate_vec = rate.to_vec();
    let pmt_vec = pmt.to_vec();
    let pv_vec = pv.to_vec();
    let fv_vec = fv.to_vec();

    let mut result_vec = Vec::with_capacity(rate_vec.len());

    for i in 0..rate_vec.len() {
        let nper_result = nper(rate_vec[i], pmt_vec[i], pv_vec[i], fv_vec[i], when)?;
        result_vec.push(nper_result);
    }

    Array::from_vec_shape(result_vec, &rate.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nper_basic_loan() {
        // Test number of periods for a loan payment
        let monthly_rate = 0.05 / 12.0;
        let result =
            nper(monthly_rate, -188.71, 10000.0, 0.0, 0).expect("nper calculation should succeed");
        assert_relative_eq!(result, 60.0, epsilon = 1e-2);
    }

    #[test]
    fn test_nper_savings_goal() {
        // Test number of periods to reach a savings goal
        let result = nper(0.05, -100.0, 0.0, 1257.79, 0).expect("nper calculation should succeed");
        assert_relative_eq!(result, 10.0, epsilon = 1e-2);
    }

    #[test]
    fn test_nper_zero_payment() {
        // Test number of periods with no payment (simple compound interest)
        let result = nper(0.05, 0.0, -1000.0, 1628.89, 0).expect("nper calculation should succeed");
        assert_relative_eq!(result, 10.0, epsilon = 1e-2);
    }

    #[test]
    fn test_nper_zero_rate() {
        // Test number of periods with zero interest rate
        let result = nper(0.0, -100.0, 1000.0, 0.0, 0).expect("nper calculation should succeed");
        assert_relative_eq!(result, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn test_nper_beginning_of_period() {
        // Test number of periods with payments at beginning
        let monthly_rate = 0.05 / 12.0;
        let result =
            nper(monthly_rate, -188.71, 10000.0, 0.0, 1).expect("nper calculation should succeed");
        // Should be slightly less than 60 months
        assert!(result < 60.0 && result > 58.0);
    }

    #[test]
    fn test_nper_with_future_value() {
        // Test number of periods with both present and future values
        let result =
            nper(0.05, -200.0, 1000.0, 5000.0, 0).expect("nper calculation should succeed");
        assert!(result > 0.0); // Should be positive
    }

    #[test]
    fn test_nper_array() {
        let rates = Array::from_vec(vec![0.05 / 12.0, 0.06 / 12.0]);
        let pmts = Array::from_vec(vec![-188.71, -250.0]);
        let pvs = Array::from_vec(vec![10000.0, 12000.0]);
        let fvs = Array::from_vec(vec![0.0, 0.0]);

        let result = nper_array(&rates, &pmts, &pvs, &fvs, 0)
            .expect("nper_array calculation should succeed");
        assert_eq!(result.shape(), vec![2]);

        let values = result.to_vec();
        assert_relative_eq!(values[0], 60.0, epsilon = 1e-2);
        assert!(values[1] > 0.0); // Should be positive
    }

    #[test]
    fn test_nper_invalid_parameters() {
        // Test error cases
        let result = nper(0.0, 0.0, 1000.0, 1000.0, 0);
        assert!(result.is_err());
    }
}
