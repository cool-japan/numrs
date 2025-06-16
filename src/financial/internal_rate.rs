//! Internal Rate of Return Calculations
//!
//! This module implements the internal rate of return (IRR) calculation function,
//! compatible with NumPy's financial functions.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[allow(unused_imports)] // Used via T::zero(), T::one(), .is_zero() methods
use num_traits::{Float, Zero, One};
use std::fmt::Debug;

/// Calculate the internal rate of return for a series of cash flows.
///
/// The internal rate of return (IRR) is the discount rate that makes the net present value
/// of a series of cash flows equal to zero. It is calculated by finding the root of:
/// ```
/// NPV = CF₀ + CF₁/(1+IRR)¹ + CF₂/(1+IRR)² + ... + CFₙ/(1+IRR)ⁿ = 0
/// ```
///
/// This function uses the Newton-Raphson method for iterative solution.
///
/// # Arguments
///
/// * `values` - Array of cash flows, where the first value is typically the initial investment (negative)
/// * `guess` - Initial guess for the IRR (default is 0.1 or 10%)
/// * `tol` - Tolerance for convergence (default is 1e-6)
/// * `maxiter` - Maximum number of iterations (default is 100)
///
/// # Returns
///
/// Internal rate of return as a decimal (e.g., 0.1 for 10%)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::irr;
///
/// // Calculate IRR for an investment
/// let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0, 600.0]);
/// let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
/// assert!((result - 0.28).abs() < 0.01); // Approximately 28%
/// ```
pub fn irr<T>(
    values: &Array<T>,
    guess: Option<T>,
    tol: Option<T>,
    maxiter: Option<usize>,
) -> Result<T>
where
    T: Float + Debug + Clone,
{
    let cash_flows = values.to_vec();
    
    if cash_flows.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "Cash flow array cannot be empty".to_string(),
        ));
    }

    if cash_flows.len() < 2 {
        return Err(NumRs2Error::ComputationError(
            "IRR calculation requires at least 2 cash flows".to_string(),
        ));
    }

    // Check for alternating signs (necessary for IRR to exist)
    let has_positive = cash_flows.iter().any(|&x| x > T::zero());
    let has_negative = cash_flows.iter().any(|&x| x < T::zero());
    
    if !has_positive || !has_negative {
        return Err(NumRs2Error::ComputationError(
            "IRR calculation requires cash flows with both positive and negative values".to_string(),
        ));
    }

    let guess = guess.unwrap_or_else(|| T::from(0.1).unwrap());
    let tol = tol.unwrap_or_else(|| T::from(1e-6).unwrap());
    let maxiter = maxiter.unwrap_or(100);

    // Newton-Raphson method
    let mut rate = guess;
    
    for iteration in 0..maxiter {
        let (npv_val, npv_derivative) = irr_function_and_derivative(&cash_flows, rate);
        
        if npv_derivative.abs() < T::from(1e-15).unwrap() {
            return Err(NumRs2Error::ComputationError(
                "IRR calculation failed: derivative too small".to_string(),
            ));
        }

        let new_rate = rate - npv_val / npv_derivative;
        
        if (new_rate - rate).abs() < tol {
            return Ok(new_rate);
        }
        
        rate = new_rate;

        // Prevent extremely negative rates that could cause numerical issues
        if rate < T::from(-0.999).unwrap() {
            rate = T::from(-0.99).unwrap();
        }

        // If rate becomes too large, restart with a different guess
        if rate > T::from(10.0).unwrap() && iteration < maxiter / 2 {
            rate = T::from(0.5).unwrap();
        }
    }

    Err(NumRs2Error::ComputationError(
        "IRR calculation failed to converge".to_string(),
    ))
}

/// Calculate the NPV function value and its derivative for Newton-Raphson method
fn irr_function_and_derivative<T>(cash_flows: &[T], rate: T) -> (T, T)
where
    T: Float + Debug + Clone,
{
    let mut npv_val = T::zero();
    let mut npv_derivative = T::zero();
    let one_plus_rate = T::one() + rate;
    
    for (i, &cash_flow) in cash_flows.iter().enumerate() {
        if i == 0 {
            // Initial cash flow is not discounted
            npv_val = npv_val + cash_flow;
            // Derivative contribution is 0 for the initial cash flow
        } else {
            let period = T::from(i).unwrap();
            let discount_factor = one_plus_rate.powf(period);
            
            // NPV contribution
            npv_val = npv_val + cash_flow / discount_factor;
            
            // Derivative contribution: -i * CF_i / (1 + rate)^(i+1)
            let derivative_term = -period * cash_flow / (discount_factor * one_plus_rate);
            npv_derivative = npv_derivative + derivative_term;
        }
    }
    
    (npv_val, npv_derivative)
}

/// Calculate IRR for multiple cash flow series
///
/// # Arguments
///
/// * `values_matrix` - 2D array where each row represents a different cash flow series
/// * `guess` - Initial guess for the IRR
/// * `tol` - Tolerance for convergence
/// * `maxiter` - Maximum number of iterations
///
/// # Returns
///
/// Array of internal rates of return, one for each cash flow series
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::irr_multiple_series;
///
/// // Two investment projects
/// let cash_flows = Array::from_vec(vec![
///     -1000.0, 300.0, 400.0, 500.0,  // Project 1
///     -1200.0, 400.0, 500.0, 600.0   // Project 2
/// ]).reshape(&[2, 4]);
/// let result = irr_multiple_series(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
/// assert_eq!(result.shape(), vec![2]);
/// ```
pub fn irr_multiple_series<T>(
    values_matrix: &Array<T>,
    guess: Option<T>,
    tol: Option<T>,
    maxiter: Option<usize>,
) -> Result<Array<T>>
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
        
        let irr_result = irr(&row_array, guess, tol, maxiter)?;
        result_vec.push(irr_result);
    }

    Ok(Array::from_vec(result_vec))
}

/// Calculate Modified Internal Rate of Return (MIRR)
///
/// The MIRR assumes that positive cash flows are reinvested at the finance rate
/// and negative cash flows are financed at the finance rate.
///
/// # Arguments
///
/// * `values` - Array of cash flows
/// * `finance_rate` - Interest rate paid on the cash flows
/// * `reinvest_rate` - Interest rate received on reinvestment of positive cash flows
///
/// # Returns
///
/// Modified internal rate of return
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::financial::mirr;
///
/// let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
/// let result = mirr(&cash_flows, 0.10, 0.12).unwrap();
/// assert!(result > 0.0 && result < 1.0); // Should be a reasonable rate
/// ```
pub fn mirr<T>(values: &Array<T>, finance_rate: T, reinvest_rate: T) -> Result<T>
where
    T: Float + Debug + Clone,
{
    let cash_flows = values.to_vec();
    
    if cash_flows.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "Cash flow array cannot be empty".to_string(),
        ));
    }

    if cash_flows.len() < 2 {
        return Err(NumRs2Error::ComputationError(
            "MIRR calculation requires at least 2 cash flows".to_string(),
        ));
    }

    let n = cash_flows.len();
    let n_float = T::from(n - 1).unwrap();
    
    // Calculate present value of negative cash flows (financing)
    let mut pv_negative = T::zero();
    // Calculate future value of positive cash flows (reinvestment)
    let mut fv_positive = T::zero();
    
    let one_plus_finance = T::one() + finance_rate;
    let one_plus_reinvest = T::one() + reinvest_rate;
    
    for (i, &cash_flow) in cash_flows.iter().enumerate() {
        let period = T::from(i).unwrap();
        
        if cash_flow < T::zero() {
            // Negative cash flow - discount to present value
            let discount_factor = one_plus_finance.powf(period);
            pv_negative = pv_negative + cash_flow / discount_factor;
        } else if cash_flow > T::zero() {
            // Positive cash flow - compound to future value
            let periods_to_end = n_float - period;
            let compound_factor = one_plus_reinvest.powf(periods_to_end);
            fv_positive = fv_positive + cash_flow * compound_factor;
        }
    }
    
    if pv_negative.is_zero() || fv_positive.is_zero() {
        return Err(NumRs2Error::ComputationError(
            "MIRR calculation requires both positive and negative cash flows".to_string(),
        ));
    }
    
    // MIRR = (FV_positive / -PV_negative)^(1/n) - 1
    let ratio = fv_positive / (-pv_negative);
    let exponent = T::one() / n_float;
    let mirr_value = ratio.powf(exponent) - T::one();
    
    Ok(mirr_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_irr_basic() {
        // Test basic IRR calculation
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0, 600.0]);
        let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
        assert_relative_eq!(result, 0.248883, epsilon = 1e-5);
    }

    #[test]
    fn test_irr_simple_case() {
        // Simple case: invest $100, get back $110 next period
        let cash_flows = Array::from_vec(vec![-100.0, 110.0]);
        let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
        assert_relative_eq!(result, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_irr_break_even() {
        // Break-even case: IRR should be 0
        let cash_flows = Array::from_vec(vec![-100.0, 50.0, 50.0]);
        let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_irr_high_return() {
        // High return case
        let cash_flows = Array::from_vec(vec![-100.0, 200.0]);
        let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-6); // 100% return
    }

    #[test]
    fn test_irr_multiple_series() {
        let cash_flows = Array::from_vec(vec![
            -1000.0, 300.0, 400.0, 500.0,  // Project 1
            -100.0, 110.0, 0.0, 0.0        // Project 2 (simple case)
        ]).reshape(&[2, 4]);
        let result = irr_multiple_series(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
        assert_eq!(result.shape(), vec![2]);
        
        let values = result.to_vec();
        assert!(values[0] > 0.0); // Project 1 should have positive IRR
        assert_relative_eq!(values[1], 0.1, epsilon = 1e-2); // Project 2 should be ~10%
    }

    #[test]
    fn test_mirr_basic() {
        let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
        let result = mirr(&cash_flows, 0.10, 0.12).unwrap();
        assert!(result > 0.0 && result < 1.0); // Should be a reasonable rate
    }

    #[test]
    fn test_irr_error_cases() {
        // Test with empty array
        let empty_flows = Array::from_vec(Vec::<f64>::new());
        assert!(irr(&empty_flows, Some(0.1), Some(1e-6), Some(100)).is_err());

        // Test with single cash flow
        let single_flow = Array::from_vec(vec![-1000.0]);
        assert!(irr(&single_flow, Some(0.1), Some(1e-6), Some(100)).is_err());

        // Test with all positive cash flows
        let all_positive = Array::from_vec(vec![100.0, 200.0, 300.0]);
        assert!(irr(&all_positive, Some(0.1), Some(1e-6), Some(100)).is_err());

        // Test with all negative cash flows
        let all_negative = Array::from_vec(vec![-100.0, -200.0, -300.0]);
        assert!(irr(&all_negative, Some(0.1), Some(1e-6), Some(100)).is_err());
    }
}