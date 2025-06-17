//! Payment Calculations
//!
//! This module implements the payment (PMT) calculation function,
//! compatible with NumPy's financial functions.

use super::{annuity_factor, compound_factor, validate_financial_params};
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Calculate the payment against loan principal plus interest.
///
/// The payment is computed by solving the following equation:
/// ```text
/// pv + pmt * [(1 + rate)^nper - 1] / rate * (1 + rate)^(-nper) + fv * (1 + rate)^(-nper) = 0
/// ```
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Number of compounding periods
/// * `pv` - Present value (the principal amount)
/// * `fv` - Future value, the desired balance after the last payment (default is 0)
/// * `when` - When payments are due ('begin' (1) or 'end' (0) of each period)
///
/// # Returns
///
/// Payment amount per period
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Calculate monthly payment for a $10,000 loan at 5% annual interest for 5 years
/// let monthly_rate = 0.05 / 12.0;
/// let months = 5.0 * 12.0;
/// let result = pmt(monthly_rate, months, 10000.0, 0.0, 0).unwrap();
/// assert!((result - (-188.71_f64)).abs() < 0.01);
/// ```
pub fn pmt<T>(rate: T, nper: T, pv: T, fv: T, when: i32) -> Result<T>
where
    T: Float + Debug + Clone,
{
    validate_financial_params(rate, nper, T::zero(), pv, fv)?;

    let when_factor = if when == 1 { T::one() + rate } else { T::one() };

    if rate.is_zero() {
        // Special case when rate is 0
        return Ok(-(pv + fv) / nper);
    }

    let compound = compound_factor(rate, nper);
    let annuity = annuity_factor(rate, nper);

    let numerator = pv * compound + fv;
    let denominator = annuity * when_factor;

    Ok(-numerator / denominator)
}

/// Calculate payment for arrays of inputs
///
/// # Arguments
///
/// * `rate` - Array of interest rates per period
/// * `nper` - Array of number of compounding periods
/// * `pv` - Array of present values
/// * `fv` - Array of future values
/// * `when` - When payments are due (0 for end, 1 for beginning)
///
/// # Returns
///
/// Array of payment amounts
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::pmt_array;
///
/// let rates = Array::from_vec(vec![0.05/12.0, 0.06/12.0, 0.07/12.0]);
/// let npers = Array::from_vec(vec![60.0, 72.0, 84.0]);
/// let pvs = Array::from_vec(vec![10000.0, 15000.0, 20000.0]);
/// let fvs = Array::from_vec(vec![0.0, 0.0, 0.0]);
///
/// let result = pmt_array(&rates, &npers, &pvs, &fvs, 0).unwrap();
/// assert_eq!(result.shape(), vec![3]);
/// ```
pub fn pmt_array<T>(
    rate: &Array<T>,
    nper: &Array<T>,
    pv: &Array<T>,
    fv: &Array<T>,
    when: i32,
) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    // Check that all arrays have the same shape
    if rate.shape() != nper.shape() || rate.shape() != pv.shape() || rate.shape() != fv.shape() {
        return Err(NumRs2Error::DimensionMismatch(
            "All input arrays must have the same shape".to_string(),
        ));
    }

    let rate_vec = rate.to_vec();
    let nper_vec = nper.to_vec();
    let pv_vec = pv.to_vec();
    let fv_vec = fv.to_vec();

    let mut result_vec = Vec::with_capacity(rate_vec.len());

    for i in 0..rate_vec.len() {
        let pmt_result = pmt(rate_vec[i], nper_vec[i], pv_vec[i], fv_vec[i], when)?;
        result_vec.push(pmt_result);
    }

    Ok(Array::from_vec(result_vec).reshape(&rate.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pmt_basic_loan() {
        // Test basic loan payment calculation
        let monthly_rate = 0.05 / 12.0;
        let months = 5.0 * 12.0;
        let result = pmt(monthly_rate, months, 10000.0, 0.0, 0).unwrap();
        assert_relative_eq!(result, -188.7107, epsilon = 1e-2);
    }

    #[test]
    fn test_pmt_with_future_value() {
        // Test payment calculation with target future value
        let result = pmt(0.05, 10.0, 0.0, 10000.0, 0).unwrap();
        assert_relative_eq!(result, -795.04, epsilon = 1e-2);
    }

    #[test]
    fn test_pmt_beginning_of_period() {
        // Test payment with payments at beginning of period
        let monthly_rate = 0.05 / 12.0;
        let months = 5.0 * 12.0;
        let result = pmt(monthly_rate, months, 10000.0, 0.0, 1).unwrap();
        assert_relative_eq!(result, -187.93, epsilon = 1e-2);
    }

    #[test]
    fn test_pmt_zero_rate() {
        // Test payment with zero interest rate
        let result = pmt(0.0, 10.0, 1000.0, 0.0, 0).unwrap();
        assert_relative_eq!(result, -100.0, epsilon = 1e-9);
    }

    #[test]
    fn test_pmt_savings() {
        // Test payment for savings goal (negative PV, positive FV)
        let result = pmt(0.05, 10.0, 0.0, 10000.0, 0).unwrap();
        assert_relative_eq!(result, -795.04, epsilon = 1e-2);
    }

    #[test]
    fn test_pmt_array() {
        let rates = Array::from_vec(vec![0.05 / 12.0, 0.06 / 12.0]);
        let npers = Array::from_vec(vec![60.0, 72.0]);
        let pvs = Array::from_vec(vec![10000.0, 15000.0]);
        let fvs = Array::from_vec(vec![0.0, 0.0]);

        let result = pmt_array(&rates, &npers, &pvs, &fvs, 0).unwrap();
        assert_eq!(result.shape(), vec![2]);

        let values = result.to_vec();
        assert_relative_eq!(values[0], -188.7107, epsilon = 1e-2);
        assert_relative_eq!(values[1], -248.59, epsilon = 1e-2);
    }
}
