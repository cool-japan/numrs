//! Advanced Financial Functions
//!
//! This module provides additional financial functions for comprehensive
//! financial analysis including:
//! - Payment breakdown (PPMT, IPMT)
//! - Cumulative payments (CUMIPMT, CUMPRINC)
//! - Rate conversions (EFFECT, NOMINAL)
//! - Depreciation methods (SLN, SYD, DB, DDB)
//!
//! Note: MIRR (Modified Internal Rate of Return) is available in internal_rate.rs

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

// =============================================================================
// PAYMENT BREAKDOWN
// =============================================================================

/// Calculate the interest portion of a payment
///
/// Returns the interest payment for a given period for an investment
/// based on periodic, constant payments and a constant interest rate.
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `per` - Period for which to find the interest (1-based)
/// * `nper` - Total number of payment periods
/// * `pv` - Present value (loan amount)
/// * `fv` - Future value (default 0)
/// * `when` - When payments are due: 0 = end of period (default), 1 = beginning
///
/// # Returns
///
/// * `Result<T>` - Interest portion of the payment
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // $2,500 loan at 8.5% for 4 years
/// let interest = ipmt(0.085/12.0, 1, 48, 2500.0, 0.0, 0).expect("ipmt calculation failed");
/// // First month interest is approximately -17.71
/// ```
pub fn ipmt<T>(rate: T, per: usize, nper: usize, pv: T, fv: T, when: u8) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if per < 1 || per > nper {
        return Err(NumRs2Error::ValueError(format!(
            "Period {} is out of range [1, {}]",
            per, nper
        )));
    }

    let nper_t = T::from(nper).expect("Failed to convert nper to type T");
    let when_t = T::from(when).expect("Failed to convert when to type T");

    // Calculate payment amount
    let pmt = calculate_pmt(rate, nper_t, pv, fv, when_t)?;

    // Calculate balance at beginning of period
    let per_minus_1 = T::from(per - 1).expect("Failed to convert per-1 to type T");

    let balance = if rate.is_zero() {
        pv + pmt * per_minus_1
    } else {
        let factor = (T::one() + rate).powf(per_minus_1);
        pv * factor + pmt * (factor - T::one()) / rate * (T::one() + rate * when_t)
    };

    // Interest for this period (negative for outflow convention)
    let interest = if when == 1 && per == 1 {
        T::zero() // First payment at beginning has no interest yet
    } else {
        -balance * rate // Negate to match payment sign convention
    };

    Ok(interest)
}

/// Calculate the principal portion of a payment
///
/// Returns the principal payment for a given period for an investment
/// based on periodic, constant payments and a constant interest rate.
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `per` - Period for which to find the principal (1-based)
/// * `nper` - Total number of payment periods
/// * `pv` - Present value (loan amount)
/// * `fv` - Future value (default 0)
/// * `when` - When payments are due: 0 = end of period (default), 1 = beginning
///
/// # Returns
///
/// * `Result<T>` - Principal portion of the payment
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // $2,500 loan at 8.5% for 4 years
/// let principal = ppmt(0.085/12.0, 1, 48, 2500.0, 0.0, 0).expect("ppmt calculation failed");
/// // First month principal is approximately -43.00
/// ```
pub fn ppmt<T>(rate: T, per: usize, nper: usize, pv: T, fv: T, when: u8) -> Result<T>
where
    T: Float + Debug + Clone,
{
    let nper_t = T::from(nper).expect("Failed to convert nper to type T");
    let when_t = T::from(when).expect("Failed to convert when to type T");

    let pmt = calculate_pmt(rate, nper_t, pv, fv, when_t)?;
    let interest = ipmt(rate, per, nper, pv, fv, when)?;

    Ok(pmt - interest)
}

/// Calculate cumulative interest paid between two periods
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value (loan amount)
/// * `start_period` - Starting period (1-based, inclusive)
/// * `end_period` - Ending period (1-based, inclusive)
/// * `when` - When payments are due: 0 = end of period (default), 1 = beginning
///
/// # Returns
///
/// * `Result<T>` - Cumulative interest paid
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Total interest paid in first year of $125,000 mortgage at 9% for 30 years
/// let cum_interest = cumipmt(0.09/12.0, 360, 125000.0, 1, 12, 0).expect("cumipmt calculation failed");
/// // First year interest is approximately -11135.23
/// ```
pub fn cumipmt<T>(
    rate: T,
    nper: usize,
    pv: T,
    start_period: usize,
    end_period: usize,
    when: u8,
) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if start_period < 1 || start_period > end_period || end_period > nper {
        return Err(NumRs2Error::ValueError(format!(
            "Invalid period range [{}, {}] for nper={}",
            start_period, end_period, nper
        )));
    }

    let mut total_interest = T::zero();
    for per in start_period..=end_period {
        total_interest = total_interest + ipmt(rate, per, nper, pv, T::zero(), when)?;
    }

    Ok(total_interest)
}

/// Calculate cumulative principal paid between two periods
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value (loan amount)
/// * `start_period` - Starting period (1-based, inclusive)
/// * `end_period` - Ending period (1-based, inclusive)
/// * `when` - When payments are due: 0 = end of period (default), 1 = beginning
///
/// # Returns
///
/// * `Result<T>` - Cumulative principal paid
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Total principal paid in first year of $125,000 mortgage at 9% for 30 years
/// let cum_principal = cumprinc(0.09/12.0, 360, 125000.0, 1, 12, 0).expect("cumprinc calculation failed");
/// // First year principal is approximately -927.43
/// ```
pub fn cumprinc<T>(
    rate: T,
    nper: usize,
    pv: T,
    start_period: usize,
    end_period: usize,
    when: u8,
) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if start_period < 1 || start_period > end_period || end_period > nper {
        return Err(NumRs2Error::ValueError(format!(
            "Invalid period range [{}, {}] for nper={}",
            start_period, end_period, nper
        )));
    }

    let mut total_principal = T::zero();
    for per in start_period..=end_period {
        total_principal = total_principal + ppmt(rate, per, nper, pv, T::zero(), when)?;
    }

    Ok(total_principal)
}

// Helper function to calculate payment
fn calculate_pmt<T>(rate: T, nper: T, pv: T, fv: T, when: T) -> Result<T>
where
    T: Float + Debug,
{
    if rate.is_zero() {
        Ok(-(pv + fv) / nper)
    } else {
        let factor = (T::one() + rate).powf(nper);
        let denominator = (factor - T::one()) / rate * (T::one() + rate * when);
        Ok(-(pv * factor + fv) / denominator)
    }
}

// =============================================================================
// RATE CONVERSIONS
// =============================================================================

/// Convert nominal annual interest rate to effective annual rate
///
/// The effective rate takes into account the effect of compounding.
///
/// # Arguments
///
/// * `nominal_rate` - Nominal annual interest rate
/// * `nper` - Number of compounding periods per year
///
/// # Returns
///
/// * `Result<T>` - Effective annual interest rate
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert 10% nominal rate compounded monthly to effective rate
/// let effective = effect(0.10, 12).expect("effect calculation failed");
/// // Effective rate is approximately 0.1047 or 10.47%
/// ```
pub fn effect<T>(nominal_rate: T, nper: usize) -> Result<T>
where
    T: Float + Debug,
{
    if nper == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of compounding periods must be positive".to_string(),
        ));
    }

    let nper_t = T::from(nper).expect("Failed to convert nper to type T");
    let periodic_rate = nominal_rate / nper_t;

    Ok((T::one() + periodic_rate).powf(nper_t) - T::one())
}

/// Convert effective annual interest rate to nominal annual rate
///
/// # Arguments
///
/// * `effective_rate` - Effective annual interest rate
/// * `nper` - Number of compounding periods per year
///
/// # Returns
///
/// * `Result<T>` - Nominal annual interest rate
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert 10.47% effective rate to nominal rate compounded monthly
/// let nominal = nominal(0.1047, 12).expect("nominal calculation failed");
/// // Nominal rate is approximately 0.10 or 10%
/// ```
pub fn nominal<T>(effective_rate: T, nper: usize) -> Result<T>
where
    T: Float + Debug,
{
    if nper == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of compounding periods must be positive".to_string(),
        ));
    }

    let nper_t = T::from(nper).expect("Failed to convert nper to type T");

    // (1 + effective_rate)^(1/nper) - 1 = periodic_rate
    // nominal_rate = periodic_rate * nper
    let periodic_rate = (T::one() + effective_rate).powf(T::one() / nper_t) - T::one();

    Ok(periodic_rate * nper_t)
}

// =============================================================================
// DEPRECIATION METHODS
// =============================================================================

/// Straight-Line Depreciation
///
/// Returns the depreciation of an asset for one period using the
/// straight-line method.
///
/// # Arguments
///
/// * `cost` - Initial cost of the asset
/// * `salvage` - Value at the end of the depreciation (salvage value)
/// * `life` - Number of periods over which the asset is depreciated
///
/// # Returns
///
/// * `Result<T>` - Depreciation for one period
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Asset costing $30,000 with $7,500 salvage over 10 years
/// let depreciation = sln(30000.0, 7500.0, 10.0).expect("sln calculation failed");
/// // Annual depreciation is 2,250
/// ```
pub fn sln<T>(cost: T, salvage: T, life: T) -> Result<T>
where
    T: Float + Debug,
{
    if life <= T::zero() {
        return Err(NumRs2Error::ValueError("Life must be positive".to_string()));
    }

    Ok((cost - salvage) / life)
}

/// Sum-of-Years-Digits Depreciation
///
/// Returns the depreciation of an asset for a specified period using the
/// sum-of-years-digits method.
///
/// # Arguments
///
/// * `cost` - Initial cost of the asset
/// * `salvage` - Value at the end of the depreciation (salvage value)
/// * `life` - Number of periods over which the asset is depreciated
/// * `per` - Period for which to calculate depreciation (1-based)
///
/// # Returns
///
/// * `Result<T>` - Depreciation for the specified period
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Asset costing $30,000 with $7,500 salvage over 10 years, year 1
/// let depreciation = syd(30000.0, 7500.0, 10, 1).expect("syd calculation failed");
/// // First year depreciation is approximately 4,090.91
/// ```
pub fn syd<T>(cost: T, salvage: T, life: usize, per: usize) -> Result<T>
where
    T: Float + Debug,
{
    if life == 0 {
        return Err(NumRs2Error::ValueError("Life must be positive".to_string()));
    }

    if per < 1 || per > life {
        return Err(NumRs2Error::ValueError(format!(
            "Period {} is out of range [1, {}]",
            per, life
        )));
    }

    let life_t = T::from(life).expect("Failed to convert life to type T");
    let per_t = T::from(per).expect("Failed to convert per to type T");

    // Sum of years digits = n(n+1)/2
    let sum_of_years =
        life_t * (life_t + T::one()) / T::from(2.0).expect("Failed to convert 2.0 to type T");

    // Remaining life at start of period
    let remaining = life_t - per_t + T::one();

    Ok((cost - salvage) * remaining / sum_of_years)
}

/// Declining Balance Depreciation
///
/// Returns the depreciation of an asset for a specified period using the
/// fixed-declining balance method.
///
/// # Arguments
///
/// * `cost` - Initial cost of the asset
/// * `salvage` - Value at the end of the depreciation (salvage value)
/// * `life` - Number of periods over which the asset is depreciated
/// * `period` - Period for which to calculate depreciation (1-based)
/// * `month` - Number of months in the first year (default 12)
///
/// # Returns
///
/// * `Result<T>` - Depreciation for the specified period
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Asset costing $1,000,000 with $100,000 salvage over 6 years, year 1
/// let depreciation = db(1000000.0, 100000.0, 6, 1, 7).expect("db calculation failed");
/// // First year depreciation with 7 months is approximately 186,083.33
/// ```
pub fn db<T>(cost: T, salvage: T, life: usize, period: usize, month: usize) -> Result<T>
where
    T: Float + Debug,
{
    if life == 0 {
        return Err(NumRs2Error::ValueError("Life must be positive".to_string()));
    }

    if period < 1 || period > life + 1 {
        return Err(NumRs2Error::ValueError(format!(
            "Period {} is out of range [1, {}]",
            period,
            life + 1
        )));
    }

    let life_t = T::from(life).expect("Failed to convert life to type T");
    let month_t = T::from(month).expect("Failed to convert month to type T");

    // Calculate depreciation rate
    // rate = 1 - (salvage/cost)^(1/life), rounded to 3 decimal places
    let rate = T::one() - (salvage / cost).powf(T::one() / life_t);
    let thousand = T::from(1000.0).expect("Failed to convert 1000.0 to type T");
    let rate = (rate * thousand).round() / thousand;

    let mut book_value = cost;
    let twelve = T::from(12.0).expect("Failed to convert 12.0 to type T");

    // Calculate depreciation for each period up to the target
    for p in 1..=period {
        let depreciation = if p == 1 {
            // First year: prorated by months
            book_value * rate * month_t / twelve
        } else if p == life + 1 {
            // Last period: remaining book value minus salvage
            let remaining = book_value - salvage;
            if remaining > T::zero() {
                remaining
            } else {
                T::zero()
            }
        } else {
            // Normal year
            book_value * rate
        };

        if p == period {
            return Ok(depreciation);
        }

        book_value = book_value - depreciation;
    }

    Ok(T::zero())
}

/// Double Declining Balance Depreciation
///
/// Returns the depreciation of an asset for a specified period using the
/// double-declining balance method or other specified factor.
///
/// # Arguments
///
/// * `cost` - Initial cost of the asset
/// * `salvage` - Value at the end of the depreciation (salvage value)
/// * `life` - Number of periods over which the asset is depreciated
/// * `period` - Period for which to calculate depreciation (1-based)
/// * `factor` - Rate at which the balance declines (default 2 for double declining)
///
/// # Returns
///
/// * `Result<T>` - Depreciation for the specified period
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Asset costing $2,400 with $300 salvage over 10 years, year 1
/// let depreciation = ddb(2400.0, 300.0, 10, 1, 2.0).expect("ddb calculation failed");
/// // First year DDB depreciation is 480
/// ```
pub fn ddb<T>(cost: T, salvage: T, life: usize, period: usize, factor: T) -> Result<T>
where
    T: Float + Debug,
{
    if life == 0 {
        return Err(NumRs2Error::ValueError("Life must be positive".to_string()));
    }

    if period < 1 || period > life {
        return Err(NumRs2Error::ValueError(format!(
            "Period {} is out of range [1, {}]",
            period, life
        )));
    }

    let life_t = T::from(life).expect("Failed to convert life to type T");

    // Depreciation rate = factor / life
    let rate = factor / life_t;

    let mut book_value = cost;
    let mut depreciation = T::zero();

    for p in 1..=period {
        // Calculate depreciation for this period
        let max_depreciation = book_value * rate;

        // Cannot depreciate below salvage value
        let allowed = book_value - salvage;
        depreciation = if allowed > T::zero() {
            max_depreciation.min(allowed)
        } else {
            T::zero()
        };

        if p == period {
            return Ok(depreciation);
        }

        book_value = book_value - depreciation;
    }

    Ok(depreciation)
}

/// Amortization schedule
///
/// Generate a full amortization schedule for a loan.
///
/// # Arguments
///
/// * `principal` - Loan principal amount
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
///
/// # Returns
///
/// * `Result<AmortizationSchedule>` - Schedule with period, payment, principal, interest, balance
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let schedule = amortization_schedule(10000.0, 0.05/12.0, 12).expect("amortization_schedule calculation failed");
/// // Returns full 12-month schedule
/// ```
pub fn amortization_schedule<T>(
    principal: T,
    rate: T,
    nper: usize,
) -> Result<AmortizationSchedule<T>>
where
    T: Float + Debug + Clone,
{
    if nper == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of periods must be positive".to_string(),
        ));
    }

    let nper_t = T::from(nper).expect("Failed to convert nper to type T");
    let pmt = calculate_pmt(rate, nper_t, principal, T::zero(), T::zero())?;

    let mut periods = Vec::with_capacity(nper);
    let mut payments = Vec::with_capacity(nper);
    let mut principals = Vec::with_capacity(nper);
    let mut interests = Vec::with_capacity(nper);
    let mut balances = Vec::with_capacity(nper);

    let mut balance = principal;

    for per in 1..=nper {
        let interest = balance * rate;
        let principal_payment = -pmt - interest;
        balance = balance - principal_payment;

        // Ensure balance doesn't go negative due to rounding
        if balance < T::zero() {
            balance = T::zero();
        }

        periods.push(per);
        payments.push(pmt);
        principals.push(-principal_payment);
        interests.push(-interest);
        balances.push(balance);
    }

    Ok(AmortizationSchedule {
        periods,
        payments,
        principals,
        interests,
        balances,
    })
}

/// Amortization schedule result
#[derive(Debug, Clone)]
pub struct AmortizationSchedule<T> {
    /// Period numbers (1-based)
    pub periods: Vec<usize>,
    /// Payment amounts (negative = outflow)
    pub payments: Vec<T>,
    /// Principal portions of payments
    pub principals: Vec<T>,
    /// Interest portions of payments
    pub interests: Vec<T>,
    /// Remaining balances after each payment
    pub balances: Vec<T>,
}

impl<T: Float + Clone> AmortizationSchedule<T> {
    /// Get total interest paid
    pub fn total_interest(&self) -> T {
        self.interests.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    /// Get total payments made
    pub fn total_payments(&self) -> T {
        self.payments.iter().fold(T::zero(), |acc, &x| acc + x)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ipmt() {
        // $2,500 loan at 8.5% annual for 4 years (monthly payments)
        let rate = 0.085 / 12.0;
        let nper = 48;
        let pv = 2500.0;

        let interest = ipmt(rate, 1, nper, pv, 0.0, 0).expect("ipmt calculation should succeed");
        // First month interest should be about -17.71
        assert_relative_eq!(interest, -17.708333, epsilon = 0.01);
    }

    #[test]
    fn test_ppmt() {
        let rate = 0.085 / 12.0;
        let nper = 48;
        let pv = 2500.0;

        let principal = ppmt(rate, 1, nper, pv, 0.0, 0).expect("ppmt calculation should succeed");
        // First month principal portion
        assert!(principal < 0.0); // Should be negative (outflow)
    }

    #[test]
    fn test_payment_breakdown() {
        // Verify IPMT + PPMT = PMT for any period
        let rate = 0.08 / 12.0;
        let nper = 60;
        let pv = 20000.0;

        let pmt =
            calculate_pmt(rate, nper as f64, pv, 0.0, 0.0).expect("pmt calculation should succeed");

        for per in 1..=nper {
            let interest =
                ipmt(rate, per, nper, pv, 0.0, 0).expect("ipmt calculation should succeed");
            let principal =
                ppmt(rate, per, nper, pv, 0.0, 0).expect("ppmt calculation should succeed");
            assert_relative_eq!(interest + principal, pmt, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cumipmt() {
        // First year interest on a mortgage
        let rate = 0.09 / 12.0;
        let nper = 360;
        let pv = 125000.0;

        let cum_interest =
            cumipmt(rate, nper, pv, 1, 12, 0).expect("cumipmt calculation should succeed");
        // Verify it's negative and in reasonable range
        assert!(cum_interest < 0.0, "Cumulative interest should be negative");
        assert!(
            cum_interest > -12000.0 && cum_interest < -10000.0,
            "First year interest should be between -12000 and -10000, got {}",
            cum_interest
        );
    }

    #[test]
    fn test_cumprinc() {
        let rate = 0.09 / 12.0;
        let nper = 360;
        let pv = 125000.0;

        let cum_principal =
            cumprinc(rate, nper, pv, 1, 12, 0).expect("cumprinc calculation should succeed");
        // Verify it's negative (principal is being paid down)
        assert!(
            cum_principal < 0.0,
            "Cumulative principal should be negative"
        );

        // Verify relationship: cumipmt + cumprinc ≈ total payments for first year
        let cum_interest =
            cumipmt(rate, nper, pv, 1, 12, 0).expect("cumipmt calculation should succeed");
        let pmt =
            calculate_pmt(rate, nper as f64, pv, 0.0, 0.0).expect("pmt calculation should succeed");
        let total_payments = pmt * 12.0;

        assert_relative_eq!(
            cum_interest + cum_principal,
            total_payments,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_effect() {
        // 10% nominal compounded monthly
        let eff = effect(0.10, 12).expect("effect calculation should succeed");
        assert_relative_eq!(eff, 0.10471307, epsilon = 1e-6);
    }

    #[test]
    fn test_nominal() {
        // Convert back from effective
        let nom = nominal(0.10471307, 12).expect("nominal calculation should succeed");
        assert_relative_eq!(nom, 0.10, epsilon = 1e-6);
    }

    #[test]
    fn test_sln() {
        let dep = sln(30000.0, 7500.0, 10.0).expect("sln calculation should succeed");
        assert_relative_eq!(dep, 2250.0, epsilon = 0.01);
    }

    #[test]
    fn test_syd() {
        // First year SYD depreciation
        let dep = syd(30000.0, 7500.0, 10, 1).expect("syd calculation should succeed");
        assert_relative_eq!(dep, 4090.909, epsilon = 0.01);

        // Sum of all years should equal depreciable amount
        let mut total = 0.0;
        for per in 1..=10 {
            total += syd(30000.0, 7500.0, 10, per).expect("syd calculation should succeed");
        }
        assert_relative_eq!(total, 22500.0, epsilon = 0.01);
    }

    #[test]
    fn test_ddb() {
        // Double declining balance
        let dep = ddb(2400.0, 300.0, 10, 1, 2.0).expect("ddb calculation should succeed");
        assert_relative_eq!(dep, 480.0, epsilon = 0.01);

        // Year 2
        let dep2 = ddb(2400.0, 300.0, 10, 2, 2.0).expect("ddb calculation should succeed");
        assert_relative_eq!(dep2, 384.0, epsilon = 0.01);
    }

    #[test]
    fn test_amortization_schedule() {
        let schedule = amortization_schedule(10000.0, 0.05 / 12.0, 12)
            .expect("amortization_schedule calculation should succeed");

        // Should have 12 periods
        assert_eq!(schedule.periods.len(), 12);

        // Final balance should be close to zero
        assert!(
            schedule
                .balances
                .last()
                .expect("balances should not be empty")
                .abs()
                < 0.01
        );

        // Total principal paid should equal original loan
        let total_principal: f64 = schedule.principals.iter().sum();
        assert_relative_eq!(total_principal, -10000.0, epsilon = 0.01);
    }
}
