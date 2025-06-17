//! Interest Rate Calculations
//!
//! This module implements the interest rate (RATE) calculation function,
//! compatible with NumPy's financial functions.

use super::validate_financial_params;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Calculate the interest rate per period.
///
/// The rate is computed by solving the following equation iteratively:
/// ```text
/// pv + pmt * [(1 + rate)^nper - 1] / rate * (1 + rate)^(-nper) + fv * (1 + rate)^(-nper) = 0
/// ```
///
/// This function uses Newton-Raphson method for iterative solution.
///
/// # Arguments
///
/// * `nper` - Number of compounding periods
/// * `pmt` - Payment made each period; this value is assumed to be the same each period
/// * `pv` - Present value (the principal amount)
/// * `fv` - Future value, the desired balance after the last payment (default is 0)
/// * `when` - When payments are due ('begin' (1) or 'end' (0) of each period)
/// * `guess` - Initial guess for the interest rate (default is 0.1 or 10%)
/// * `tol` - Tolerance for convergence (default is 1e-6)
/// * `maxiter` - Maximum number of iterations (default is 100)
///
/// # Returns
///
/// Interest rate per period
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Calculate interest rate for a loan
/// let result = rate(60.0, -188.71, 10000.0, 0.0, 0, Some(0.1), Some(1e-6), Some(100)).unwrap();
/// assert!((result - (0.05_f64/12.0)).abs() < 0.001);
/// ```
pub fn rate<T>(
    nper: T,
    pmt: T,
    pv: T,
    fv: T,
    when: i32,
    guess: Option<T>,
    tol: Option<T>,
    maxiter: Option<usize>,
) -> Result<T>
where
    T: Float + Debug + Clone,
{
    validate_financial_params(T::zero(), nper, pmt, pv, fv)?;

    let guess = guess.unwrap_or_else(|| T::from(0.1).unwrap());
    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let maxiter = maxiter.unwrap_or(100);

    let when_factor = if when == 1 { T::one() } else { T::zero() };

    // Newton-Raphson method
    let mut rate = guess;

    for _ in 0..maxiter {
        let (f_val, f_prime) = rate_function_and_derivative(rate, nper, pmt, pv, fv, when_factor);

        if f_prime.abs() < T::from(1e-15).unwrap() {
            return Err(NumRs2Error::ComputationError(
                "Rate calculation failed: derivative too small".to_string(),
            ));
        }

        let new_rate = rate - f_val / f_prime;

        if (new_rate - rate).abs() < tol {
            return Ok(new_rate);
        }

        rate = new_rate;

        // Prevent negative rates in some contexts
        if rate < T::from(-0.99).unwrap() {
            rate = T::from(-0.99).unwrap();
        }
    }

    Err(NumRs2Error::ComputationError(
        "Rate calculation failed to converge".to_string(),
    ))
}

/// Calculate the function value and its derivative for Newton-Raphson method
fn rate_function_and_derivative<T>(rate: T, nper: T, pmt: T, pv: T, fv: T, when_factor: T) -> (T, T)
where
    T: Float + Debug + Clone,
{
    if rate.abs() < T::from(1e-12).unwrap() {
        // Linear approximation when rate is very close to zero
        let f_val = pv + pmt * nper * (T::one() + when_factor * rate) + fv;
        let f_prime = pmt * nper * when_factor;
        return (f_val, f_prime);
    }

    let one_plus_rate = T::one() + rate;
    let compound = one_plus_rate.powf(nper);
    let compound_inv = T::one() / compound;

    // Function value: pv + pmt * [(1+r)^n - 1] / r * (1+r)^(-n) * (1 + r*when) + fv * (1+r)^(-n)
    let annuity_term = (compound - T::one()) / rate;
    let pmt_term = pmt * annuity_term * compound_inv * (T::one() + rate * when_factor);
    let fv_term = fv * compound_inv;
    let f_val = pv + pmt_term + fv_term;

    // Derivative calculation
    let d_compound_d_rate = nper * one_plus_rate.powf(nper - T::one());
    let d_compound_inv_d_rate = -compound_inv * compound_inv * d_compound_d_rate;

    let d_annuity_d_rate = (d_compound_d_rate * rate - compound + T::one()) / (rate * rate);

    let d_pmt_term_d_rate = pmt
        * (d_annuity_d_rate * compound_inv * (T::one() + rate * when_factor)
            + annuity_term * d_compound_inv_d_rate * (T::one() + rate * when_factor)
            + annuity_term * compound_inv * when_factor);

    let d_fv_term_d_rate = fv * d_compound_inv_d_rate;
    let f_prime = d_pmt_term_d_rate + d_fv_term_d_rate;

    (f_val, f_prime)
}

/// Calculate interest rate for arrays of inputs
///
/// # Arguments
///
/// * `nper` - Array of number of compounding periods
/// * `pmt` - Array of payments made each period
/// * `pv` - Array of present values
/// * `fv` - Array of future values
/// * `when` - When payments are due (0 for end, 1 for beginning)
/// * `guess` - Initial guess for the interest rate
/// * `tol` - Tolerance for convergence
/// * `maxiter` - Maximum number of iterations
///
/// # Returns
///
/// Array of interest rates
pub fn rate_array<T>(
    nper: &Array<T>,
    pmt: &Array<T>,
    pv: &Array<T>,
    fv: &Array<T>,
    when: i32,
    guess: Option<T>,
    tol: Option<T>,
    maxiter: Option<usize>,
) -> Result<Array<T>>
where
    T: Float + Debug + Clone,
{
    // Check that all arrays have the same shape
    if nper.shape() != pmt.shape() || nper.shape() != pv.shape() || nper.shape() != fv.shape() {
        return Err(NumRs2Error::DimensionMismatch(
            "All input arrays must have the same shape".to_string(),
        ));
    }

    let nper_vec = nper.to_vec();
    let pmt_vec = pmt.to_vec();
    let pv_vec = pv.to_vec();
    let fv_vec = fv.to_vec();

    let mut result_vec = Vec::with_capacity(nper_vec.len());

    for i in 0..nper_vec.len() {
        let rate_result = rate(
            nper_vec[i],
            pmt_vec[i],
            pv_vec[i],
            fv_vec[i],
            when,
            guess,
            tol,
            maxiter,
        )?;
        result_vec.push(rate_result);
    }

    Ok(Array::from_vec(result_vec).reshape(&nper.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rate_basic_loan() {
        // Test interest rate calculation for a known loan
        let monthly_rate = 0.05 / 12.0;
        let result = rate(
            60.0,
            -188.71,
            10000.0,
            0.0,
            0,
            Some(0.1),
            Some(1e-6),
            Some(100),
        )
        .unwrap();
        assert_relative_eq!(result, monthly_rate, epsilon = 1e-4);
    }

    #[test]
    fn test_rate_simple_case() {
        // Simple case: find rate for doubling money in 10 periods with no payments
        let result = rate(
            10.0,
            0.0,
            -1000.0,
            2000.0,
            0,
            Some(0.1),
            Some(1e-6),
            Some(100),
        )
        .unwrap();
        let expected = 2.0_f64.powf(1.0 / 10.0) - 1.0; // ~7.18%
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_rate_annuity() {
        // Test rate calculation for an annuity
        let result = rate(
            10.0,
            -100.0,
            772.17,
            0.0,
            0,
            Some(0.1),
            Some(1e-6),
            Some(100),
        )
        .unwrap();
        assert_relative_eq!(result, 0.05, epsilon = 1e-3);
    }

    #[test]
    fn test_rate_zero_payment() {
        // Test rate with zero payment (simple compound interest)
        let result = rate(
            5.0,
            0.0,
            -1000.0,
            1276.28,
            0,
            Some(0.1),
            Some(1e-6),
            Some(100),
        )
        .unwrap();
        assert_relative_eq!(result, 0.05, epsilon = 1e-4);
    }

    #[test]
    fn test_rate_array() {
        let npers = Array::from_vec(vec![10.0, 20.0]);
        let pmts = Array::from_vec(vec![0.0, 0.0]);
        let pvs = Array::from_vec(vec![-1000.0, -2000.0]);
        let fvs = Array::from_vec(vec![1628.89, 6536.00]);

        let result = rate_array(
            &npers,
            &pmts,
            &pvs,
            &fvs,
            0,
            Some(0.1),
            Some(1e-6),
            Some(100),
        )
        .unwrap();
        assert_eq!(result.shape(), vec![2]);

        let values = result.to_vec();
        assert_relative_eq!(values[0], 0.05, epsilon = 1e-4);
        assert_relative_eq!(values[1], 0.06, epsilon = 1e-3);
    }
}
