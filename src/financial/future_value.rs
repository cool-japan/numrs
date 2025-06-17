//! Future Value Calculations
//!
//! This module implements the future value (FV) calculation function,
//! compatible with NumPy's financial functions.

use super::{annuity_factor, compound_factor, validate_financial_params};
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Calculate the future value of a present value sum, or a series of payments.
///
/// The future value is computed by solving the following equation:
/// ```text
/// fv + pv * (1 + rate)^nper + pmt * [(1 + rate)^nper - 1] / rate = 0
/// ```
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Number of compounding periods
/// * `pmt` - Payment made each period; this value is assumed to be the same each period
/// * `pv` - Present value (the principal amount)
/// * `when` - When payments are due ('begin' (1) or 'end' (0) of each period)
///
/// # Returns
///
/// Future value of the cash flows
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Calculate future value of $1000 invested for 10 periods at 5% interest
/// let result = fv(0.05, 10.0, 0.0, -1000.0, 0).unwrap();
/// assert!((result - 1628.895_f64).abs() < 0.001);
/// ```
pub fn fv<T>(rate: T, nper: T, pmt: T, pv: T, when: i32) -> Result<T>
where
    T: Float + Debug + Clone,
{
    validate_financial_params(rate, nper, pmt, pv, T::zero())?;

    let when_factor = if when == 1 { T::one() + rate } else { T::one() };

    if rate.is_zero() {
        // Special case when rate is 0
        return Ok(-(pv + pmt * nper));
    }

    let compound = compound_factor(rate, nper);
    let annuity = annuity_factor(rate, nper);

    let fv_lump_sum = pv * compound;
    let fv_annuity = pmt * annuity * when_factor;

    Ok(-(fv_lump_sum + fv_annuity))
}

/// Calculate future value for arrays of inputs
///
/// # Arguments
///
/// * `rate` - Array of interest rates per period
/// * `nper` - Array of number of compounding periods
/// * `pmt` - Array of payments made each period
/// * `pv` - Array of present values
/// * `when` - When payments are due (0 for end, 1 for beginning)
///
/// # Returns
///
/// Array of future values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let rates = Array::from_vec(vec![0.05, 0.06, 0.07]);
/// let npers = Array::from_vec(vec![10.0, 15.0, 20.0]);
/// let pmts = Array::from_vec(vec![0.0, 0.0, 0.0]);
/// let pvs = Array::from_vec(vec![-1000.0, -2000.0, -3000.0]);
///
/// let result = fv_array(&rates, &npers, &pmts, &pvs, 0).unwrap();
/// assert_eq!(result.shape(), vec![3]);
/// ```
pub fn fv_array<T>(
    rate: &Array<T>,
    nper: &Array<T>,
    pmt: &Array<T>,
    pv: &Array<T>,
    when: i32,
) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    // Check that all arrays have the same shape
    if rate.shape() != nper.shape() || rate.shape() != pmt.shape() || rate.shape() != pv.shape() {
        return Err(NumRs2Error::DimensionMismatch(
            "All input arrays must have the same shape".to_string(),
        ));
    }

    let rate_vec = rate.to_vec();
    let nper_vec = nper.to_vec();
    let pmt_vec = pmt.to_vec();
    let pv_vec = pv.to_vec();

    let mut result_vec = Vec::with_capacity(rate_vec.len());

    for i in 0..rate_vec.len() {
        let fv_result = fv(rate_vec[i], nper_vec[i], pmt_vec[i], pv_vec[i], when)?;
        result_vec.push(fv_result);
    }

    Ok(Array::from_vec(result_vec).reshape(&rate.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fv_basic_lump_sum() {
        // Test basic future value calculation for lump sum
        let result = fv(0.05, 10.0, 0.0, -1000.0, 0).unwrap();
        assert_relative_eq!(result, 1628.8946, epsilon = 1e-4);
    }

    #[test]
    fn test_fv_annuity() {
        // Test future value of annuity
        let result = fv(0.05, 10.0, -100.0, 0.0, 0).unwrap();
        assert_relative_eq!(result, 1257.7893, epsilon = 1e-4);
    }

    #[test]
    fn test_fv_combined() {
        // Test future value with both present value and payments
        let result = fv(0.05, 10.0, -100.0, -1000.0, 0).unwrap();
        assert_relative_eq!(result, 2886.6839, epsilon = 1e-4);
    }

    #[test]
    fn test_fv_beginning_of_period() {
        // Test future value with payments at beginning of period
        let result = fv(0.05, 10.0, -100.0, 0.0, 1).unwrap();
        assert_relative_eq!(result, 1320.6787, epsilon = 1e-4);
    }

    #[test]
    fn test_fv_zero_rate() {
        // Test future value with zero interest rate
        let result = fv(0.0, 10.0, -100.0, -1000.0, 0).unwrap();
        assert_relative_eq!(result, 2000.0, epsilon = 1e-9);
    }

    #[test]
    fn test_fv_array() {
        let rates = Array::from_vec(vec![0.05, 0.06]);
        let npers = Array::from_vec(vec![10.0, 15.0]);
        let pmts = Array::from_vec(vec![0.0, 0.0]);
        let pvs = Array::from_vec(vec![-1000.0, -2000.0]);

        let result = fv_array(&rates, &npers, &pmts, &pvs, 0).unwrap();
        assert_eq!(result.shape(), vec![2]);

        let values = result.to_vec();
        assert_relative_eq!(values[0], 1628.8946, epsilon = 1e-4);
        assert_relative_eq!(values[1], 4793.116386, epsilon = 1e-4);
    }
}
