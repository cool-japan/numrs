//! Financial Functions Module
//!
//! This module provides NumPy-compatible financial functions for time value of money calculations.
//! All functions support both scalar values and array inputs, maintaining full compatibility with
//! NumPy's financial functions.
//!
//! # Available Functions
//!
//! ## Basic Time Value of Money
//! - `pv()` - Present value calculation
//! - `fv()` - Future value calculation  
//! - `pmt()` - Payment calculation
//! - `rate()` - Interest rate calculation
//! - `nper()` - Number of periods calculation
//! - `npv()` - Net present value calculation
//! - `irr()` - Internal rate of return calculation
//!
//! ## Bond Pricing and Analysis
//! - `bond_price()` - Bond valuation from cash flows
//! - `bond_duration()` - Macaulay duration calculation
//! - `modified_duration()` - Modified duration for sensitivity
//! - `bond_convexity()` - Convexity measurement
//! - `bond_yield()` - Yield to maturity calculation
//! - `accrued_interest()` - Accrued interest calculation
//! - `bond_equivalent_yield()` - Bond equivalent yield conversion
//!
//! ## Options Pricing
//! - `black_scholes()` - Black-Scholes option pricing
//! - `black_scholes_greeks()` - Option Greeks calculation
//! - `implied_volatility()` - Implied volatility from market prices
//! - `binomial_option_price()` - Binomial tree option pricing

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

mod advanced;
mod bond_pricing;
mod future_value;
mod internal_rate;
mod net_present_value;
mod options;
mod payment;
mod periods;
mod present_value;
mod rate_calculation;

pub use advanced::*;
pub use bond_pricing::*;
pub use future_value::*;
pub use internal_rate::*;
pub use net_present_value::*;
pub use options::*;
pub use payment::*;
pub use periods::*;
pub use present_value::*;
pub use rate_calculation::*;

/// Trait for financial calculations supporting both scalar and array operations
pub trait FinancialCalculation<T>
where
    T: Float + Debug + Clone,
{
    /// Apply the financial calculation
    fn calculate(&self) -> Result<T>;
}

/// Helper function to validate financial parameters
pub fn validate_financial_params<T: Float + Debug>(
    rate: T,
    nper: T,
    pmt: T,
    pv: T,
    fv: T,
) -> Result<()> {
    // Check for NaN values
    if rate.is_nan() || nper.is_nan() || pmt.is_nan() || pv.is_nan() || fv.is_nan() {
        return Err(NumRs2Error::ComputationError(
            "Financial parameters cannot be NaN".to_string(),
        ));
    }

    // Check for infinite values
    if rate.is_infinite()
        || nper.is_infinite()
        || pmt.is_infinite()
        || pv.is_infinite()
        || fv.is_infinite()
    {
        return Err(NumRs2Error::ComputationError(
            "Financial parameters cannot be infinite".to_string(),
        ));
    }

    // Additional validations can be added here for specific constraints
    Ok(())
}

/// Helper function for compound interest calculation
#[inline]
pub fn compound_factor<T: Float>(rate: T, nper: T) -> T {
    if rate.is_zero() {
        T::one()
    } else {
        (T::one() + rate).powf(nper)
    }
}

/// Helper function for annuity factor calculation
#[inline]
pub fn annuity_factor<T: Float>(rate: T, nper: T) -> T {
    if rate.is_zero() {
        nper
    } else {
        let compound = compound_factor(rate, nper);
        (compound - T::one()) / rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_compound_factor() {
        let result = compound_factor(0.05, 10.0);
        assert_relative_eq!(result, 1.6288946267, epsilon = 1e-9);
    }

    #[test]
    fn test_annuity_factor() {
        let result = annuity_factor(0.05, 10.0);
        assert_relative_eq!(result, 12.5778925, epsilon = 1e-6);
    }

    #[test]
    fn test_validate_financial_params() {
        // Valid parameters should pass
        assert!(validate_financial_params(0.05, 10.0, -100.0, 0.0, 1000.0).is_ok());

        // NaN parameters should fail
        assert!(validate_financial_params(f64::NAN, 10.0, -100.0, 0.0, 1000.0).is_err());

        // Infinite parameters should fail
        assert!(validate_financial_params(f64::INFINITY, 10.0, -100.0, 0.0, 1000.0).is_err());
    }
}
