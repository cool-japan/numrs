//! Bond Pricing Functions
//!
//! This module provides functions for bond valuation and analysis.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Calculate the price of a bond given its cash flows and yield
///
/// # Arguments
///
/// * `cash_flows` - Array of cash flows (including coupon payments and principal)
/// * `periods` - Array of time periods for each cash flow
/// * `yield_rate` - Yield to maturity (as decimal, e.g., 0.05 for 5%)
///
/// # Returns
///
/// Bond price as present value of cash flows
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let cash_flows = Array::from_vec(vec![50.0, 50.0, 50.0, 1050.0]); // 5% annual coupon, $1000 face
/// let periods = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let price = bond_price(&cash_flows, &periods, 0.06).unwrap();
/// ```
pub fn bond_price<T>(cash_flows: &Array<T>, periods: &Array<T>, yield_rate: T) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if cash_flows.size() != periods.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![cash_flows.size()],
            actual: vec![periods.size()],
        });
    }

    let cf_vec = cash_flows.to_vec();
    let per_vec = periods.to_vec();

    let mut price = T::zero();
    for (cf, period) in cf_vec.iter().zip(per_vec.iter()) {
        let discount_factor = (T::one() + yield_rate).powf(-*period);
        price = price + *cf * discount_factor;
    }

    Ok(price)
}

/// Calculate bond duration (Macaulay duration)
///
/// Duration measures the weighted average time to receive cash flows.
///
/// # Arguments
///
/// * `cash_flows` - Array of cash flows
/// * `periods` - Array of time periods
/// * `yield_rate` - Yield to maturity
///
/// # Returns
///
/// Macaulay duration in years
pub fn bond_duration<T>(cash_flows: &Array<T>, periods: &Array<T>, yield_rate: T) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if cash_flows.size() != periods.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![cash_flows.size()],
            actual: vec![periods.size()],
        });
    }

    let price = bond_price(cash_flows, periods, yield_rate)?;
    let cf_vec = cash_flows.to_vec();
    let per_vec = periods.to_vec();

    let mut weighted_time = T::zero();
    for (cf, period) in cf_vec.iter().zip(per_vec.iter()) {
        let pv_cf = *cf / (T::one() + yield_rate).powf(*period);
        weighted_time = weighted_time + *period * pv_cf;
    }

    Ok(weighted_time / price)
}

/// Calculate modified duration
///
/// Modified duration measures price sensitivity to yield changes.
///
/// # Arguments
///
/// * `macaulay_duration` - Macaulay duration
/// * `yield_rate` - Yield to maturity
///
/// # Returns
///
/// Modified duration
pub fn modified_duration<T>(macaulay_duration: T, yield_rate: T) -> T
where
    T: Float,
{
    macaulay_duration / (T::one() + yield_rate)
}

/// Calculate bond convexity
///
/// Convexity measures the curvature of price-yield relationship.
///
/// # Arguments
///
/// * `cash_flows` - Array of cash flows
/// * `periods` - Array of time periods
/// * `yield_rate` - Yield to maturity
///
/// # Returns
///
/// Bond convexity
pub fn bond_convexity<T>(cash_flows: &Array<T>, periods: &Array<T>, yield_rate: T) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if cash_flows.size() != periods.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![cash_flows.size()],
            actual: vec![periods.size()],
        });
    }

    let price = bond_price(cash_flows, periods, yield_rate)?;
    let cf_vec = cash_flows.to_vec();
    let per_vec = periods.to_vec();

    let mut convexity_sum = T::zero();
    let discount_factor_base = T::one() + yield_rate;

    for (cf, period) in cf_vec.iter().zip(per_vec.iter()) {
        let pv_cf = *cf / discount_factor_base.powf(*period);
        let time_squared_plus_time = *period * (*period + T::one());
        convexity_sum = convexity_sum + time_squared_plus_time * pv_cf;
    }

    let yield_squared = (T::one() + yield_rate).powi(2);
    Ok(convexity_sum / (price * yield_squared))
}

/// Calculate yield to maturity using Newton-Raphson method
///
/// # Arguments
///
/// * `price` - Current bond price
/// * `cash_flows` - Array of cash flows
/// * `periods` - Array of time periods
/// * `initial_guess` - Initial guess for yield (optional)
///
/// # Returns
///
/// Yield to maturity
pub fn bond_yield<T>(
    price: T,
    cash_flows: &Array<T>,
    periods: &Array<T>,
    initial_guess: Option<T>,
) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if cash_flows.size() != periods.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![cash_flows.size()],
            actual: vec![periods.size()],
        });
    }

    let mut yield_guess = initial_guess.unwrap_or_else(|| T::from(0.05).unwrap());
    let tolerance = T::from(1e-8).unwrap();
    let max_iterations = 100;

    for _ in 0..max_iterations {
        let calculated_price = bond_price(cash_flows, periods, yield_guess)?;
        let price_diff = calculated_price - price;

        if price_diff.abs() < tolerance {
            return Ok(yield_guess);
        }

        // Calculate derivative (negative duration times price)
        let duration = bond_duration(cash_flows, periods, yield_guess)?;
        let price_derivative = -duration * calculated_price;

        // Newton-Raphson update
        yield_guess = yield_guess - price_diff / price_derivative;

        // Ensure yield stays positive
        if yield_guess < T::zero() {
            yield_guess = T::from(0.001).unwrap();
        }
    }

    Err(NumRs2Error::ComputationError(
        "Yield calculation did not converge".to_string(),
    ))
}

/// Calculate accrued interest for a bond
///
/// # Arguments
///
/// * `coupon_rate` - Annual coupon rate (as decimal)
/// * `face_value` - Face value of the bond
/// * `days_since_last_coupon` - Days since last coupon payment
/// * `days_in_coupon_period` - Total days in coupon period
///
/// # Returns
///
/// Accrued interest amount
pub fn accrued_interest<T>(
    coupon_rate: T,
    face_value: T,
    days_since_last_coupon: T,
    days_in_coupon_period: T,
) -> T
where
    T: Float,
{
    let annual_coupon = coupon_rate * face_value;
    let fraction_of_period = days_since_last_coupon / days_in_coupon_period;
    annual_coupon * fraction_of_period
}

/// Calculate bond equivalent yield
///
/// Converts a discount rate to bond equivalent yield for comparison.
///
/// # Arguments
///
/// * `discount_rate` - Discount rate (as decimal)
/// * `days_to_maturity` - Days to maturity
///
/// # Returns
///
/// Bond equivalent yield
pub fn bond_equivalent_yield<T>(discount_rate: T, days_to_maturity: T) -> T
where
    T: Float,
{
    let days_per_year = T::from(365.0).unwrap();
    let price = T::one() - discount_rate * (days_to_maturity / days_per_year);
    (discount_rate * days_per_year) / (days_to_maturity * price)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bond_price() {
        // 4-year bond with 5% annual coupon, $1000 face value, 6% yield
        let cash_flows = Array::from_vec(vec![50.0, 50.0, 50.0, 1050.0]);
        let periods = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let price = bond_price(&cash_flows, &periods, 0.06).unwrap();

        // Expected price should be less than face value since yield > coupon rate
        assert!(price < 1000.0);
        assert_relative_eq!(price, 965.35, epsilon = 1.0);
    }

    #[test]
    fn test_bond_duration() {
        let cash_flows = Array::from_vec(vec![50.0, 50.0, 50.0, 1050.0]);
        let periods = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let duration = bond_duration(&cash_flows, &periods, 0.06).unwrap();

        // Duration should be less than maturity for coupon bonds
        assert!(duration < 4.0);
        assert!(duration > 3.0);
    }

    #[test]
    fn test_modified_duration() {
        let macaulay_dur = 3.5;
        let yield_rate = 0.06;
        let mod_dur = modified_duration(macaulay_dur, yield_rate);

        assert_relative_eq!(mod_dur, 3.5 / 1.06, epsilon = 1e-6);
    }

    #[test]
    fn test_accrued_interest() {
        let accrued = accrued_interest(0.05, 1000.0, 30.0, 180.0);
        assert_relative_eq!(accrued, 8.33, epsilon = 0.1);
    }
}
