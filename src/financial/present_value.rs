//! Present Value Calculations
//!
//! This module implements the present value (PV) calculation function,
//! compatible with NumPy's financial functions.

use super::{annuity_factor, compound_factor, validate_financial_params};
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Calculate the present value of a payment or series of payments.
///
/// The present value is computed by solving the following equation:
/// ```text
/// pv + pmt * [(1 + rate)^nper - 1] / rate * (1 + rate)^(-nper) + fv * (1 + rate)^(-nper) = 0
/// ```
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Number of compounding periods
/// * `pmt` - Payment made each period; this value is assumed to be the same each period
/// * `fv` - Future value, the desired balance after the last payment (default is 0)
/// * `when` - When payments are due ('begin' (1) or 'end' (0) of each period)
///
/// # Returns
///
/// Present value of the cash flows
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Calculate present value of $100 payments for 10 periods at 5% interest
/// let result = pv(0.05, 10.0, 100.0, 0.0, 0).unwrap();
/// assert!((result - (-772.173_f64)).abs() < 0.001);
/// ```
pub fn pv<T>(rate: T, nper: T, pmt: T, fv: T, when: i32) -> Result<T>
where
    T: Float + Debug + Clone,
{
    validate_financial_params(rate, nper, pmt, T::zero(), fv)?;

    let when_factor = if when == 1 { T::one() + rate } else { T::one() };

    if rate.is_zero() {
        // Special case when rate is 0
        return Ok(-(fv + pmt * nper));
    }

    let compound = compound_factor(rate, nper);
    let annuity = annuity_factor(rate, nper);

    let pv_annuity = pmt * annuity * when_factor / compound;
    let pv_lump_sum = fv / compound;

    Ok(-(pv_annuity + pv_lump_sum))
}

/// Calculate present value for arrays of inputs
///
/// # Arguments
///
/// * `rate` - Array of interest rates per period
/// * `nper` - Array of number of compounding periods
/// * `pmt` - Array of payments made each period
/// * `fv` - Array of future values
/// * `when` - When payments are due (0 for end, 1 for beginning)
///
/// # Returns
///
/// Array of present values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::pv_array;
///
/// let rates = Array::from_vec(vec![0.05, 0.06, 0.07]);
/// let npers = Array::from_vec(vec![10.0, 15.0, 20.0]);
/// let pmts = Array::from_vec(vec![100.0, 200.0, 300.0]);
/// let fvs = Array::from_vec(vec![0.0, 0.0, 0.0]);
///
/// let result = pv_array(&rates, &npers, &pmts, &fvs, 0).unwrap();
/// assert_eq!(result.shape(), vec![3]);
/// ```
pub fn pv_array<T>(
    rate: &Array<T>,
    nper: &Array<T>,
    pmt: &Array<T>,
    fv: &Array<T>,
    when: i32,
) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    // Check that all arrays have the same shape
    if rate.shape() != nper.shape() || rate.shape() != pmt.shape() || rate.shape() != fv.shape() {
        return Err(NumRs2Error::DimensionMismatch(
            "All input arrays must have the same shape".to_string(),
        ));
    }

    let rate_vec = rate.to_vec();
    let nper_vec = nper.to_vec();
    let pmt_vec = pmt.to_vec();
    let fv_vec = fv.to_vec();

    let mut result_vec = Vec::with_capacity(rate_vec.len());

    for i in 0..rate_vec.len() {
        let pv_result = pv(rate_vec[i], nper_vec[i], pmt_vec[i], fv_vec[i], when)?;
        result_vec.push(pv_result);
    }

    Ok(Array::from_vec(result_vec).reshape(&rate.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pv_basic() {
        // Test basic present value calculation
        let result = pv(0.05, 10.0, 100.0, 0.0, 0).unwrap();
        assert_relative_eq!(result, -772.1734, epsilon = 1e-4);
    }

    #[test]
    fn test_pv_with_future_value() {
        // Test present value with future value
        let result = pv(0.05, 10.0, 100.0, 1000.0, 0).unwrap();
        assert_relative_eq!(result, -1386.087, epsilon = 1e-3);
    }

    #[test]
    fn test_pv_beginning_of_period() {
        // Test present value with payments at beginning of period
        let result = pv(0.05, 10.0, 100.0, 0.0, 1).unwrap();
        assert_relative_eq!(result, -810.782, epsilon = 1e-3);
    }

    #[test]
    fn test_pv_zero_rate() {
        // Test present value with zero interest rate
        let result = pv(0.0, 10.0, 100.0, 0.0, 0).unwrap();
        assert_relative_eq!(result, -1000.0, epsilon = 1e-9);
    }

    #[test]
    fn test_pv_array() {
        let rates = Array::from_vec(vec![0.05, 0.06]);
        let npers = Array::from_vec(vec![10.0, 15.0]);
        let pmts = Array::from_vec(vec![100.0, 200.0]);
        let fvs = Array::from_vec(vec![0.0, 0.0]);

        let result = pv_array(&rates, &npers, &pmts, &fvs, 0).unwrap();
        assert_eq!(result.shape(), vec![2]);

        let values = result.to_vec();
        assert_relative_eq!(values[0], -772.1734, epsilon = 1e-4);
        assert_relative_eq!(values[1], -1942.449798, epsilon = 1e-4);
    }
}
