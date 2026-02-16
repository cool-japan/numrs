//! Options Pricing Functions
//!
//! This module provides functions for option valuation using various models.

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Calculate the cumulative standard normal distribution
fn norm_cdf<T>(x: T) -> T
where
    T: Float,
{
    // Approximation using error function
    let sqrt_2 = T::from(2.0)
        .expect("Failed to convert 2.0 to type T")
        .sqrt();
    (T::one() + erf_approx(x / sqrt_2)) / T::from(2.0).expect("Failed to convert 2.0 to type T")
}

/// Approximate error function implementation
fn erf_approx<T>(x: T) -> T
where
    T: Float,
{
    // Abramowitz and Stegun approximation
    let a1 = T::from(0.254829592).expect("Failed to convert erf coefficient a1");
    let a2 = T::from(-0.284496736).expect("Failed to convert erf coefficient a2");
    let a3 = T::from(1.421413741).expect("Failed to convert erf coefficient a3");
    let a4 = T::from(-1.453152027).expect("Failed to convert erf coefficient a4");
    let a5 = T::from(1.061405429).expect("Failed to convert erf coefficient a5");
    let p = T::from(0.3275911).expect("Failed to convert erf coefficient p");

    let sign = if x >= T::zero() { T::one() } else { -T::one() };
    let x = x.abs();

    let t = T::one() / (T::one() + p * x);
    let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Black-Scholes option pricing model
///
/// Calculate European option prices using the Black-Scholes formula.
///
/// # Arguments
///
/// * `spot_price` - Current stock price
/// * `strike_price` - Strike price of the option
/// * `time_to_expiry` - Time to expiration in years
/// * `risk_free_rate` - Risk-free interest rate (as decimal)
/// * `volatility` - Volatility of the underlying asset (as decimal)
/// * `option_type` - "call" for call option, "put" for put option
///
/// # Returns
///
/// Option price
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let call_price = black_scholes(100.0, 105.0, 0.25, 0.05, 0.2, "call").expect("black_scholes call failed");
/// let put_price = black_scholes(100.0, 105.0, 0.25, 0.05, 0.2, "put").expect("black_scholes put failed");
/// ```
pub fn black_scholes<T>(
    spot_price: T,
    strike_price: T,
    time_to_expiry: T,
    risk_free_rate: T,
    volatility: T,
    option_type: &str,
) -> Result<T>
where
    T: Float + Debug,
{
    if time_to_expiry <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Time to expiry must be positive".to_string(),
        ));
    }

    if volatility <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Volatility must be positive".to_string(),
        ));
    }

    let sqrt_t = time_to_expiry.sqrt();
    let vol_sqrt_t = volatility * sqrt_t;

    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate
            + volatility * volatility / T::from(2.0).expect("Failed to convert 2.0 to type T"))
            * time_to_expiry)
        / vol_sqrt_t;

    let d2 = d1 - vol_sqrt_t;

    let discount_factor = (-risk_free_rate * time_to_expiry).exp();

    match option_type.to_lowercase().as_str() {
        "call" => {
            let call_price =
                spot_price * norm_cdf(d1) - strike_price * discount_factor * norm_cdf(d2);
            Ok(call_price)
        }
        "put" => {
            let put_price =
                strike_price * discount_factor * norm_cdf(-d2) - spot_price * norm_cdf(-d1);
            Ok(put_price)
        }
        _ => Err(NumRs2Error::InvalidOperation(
            "Option type must be 'call' or 'put'".to_string(),
        )),
    }
}

/// Calculate the Greeks for Black-Scholes options
///
/// # Arguments
///
/// * `spot_price` - Current stock price
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiration in years
/// * `risk_free_rate` - Risk-free rate
/// * `volatility` - Volatility
/// * `option_type` - "call" or "put"
///
/// # Returns
///
/// Tuple of (delta, gamma, theta, vega, rho)
pub fn black_scholes_greeks<T>(
    spot_price: T,
    strike_price: T,
    time_to_expiry: T,
    risk_free_rate: T,
    volatility: T,
    option_type: &str,
) -> Result<(T, T, T, T, T)>
where
    T: Float + Debug,
{
    if time_to_expiry <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Time to expiry must be positive".to_string(),
        ));
    }

    let sqrt_t = time_to_expiry.sqrt();
    let vol_sqrt_t = volatility * sqrt_t;

    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate
            + volatility * volatility / T::from(2.0).expect("Failed to convert 2.0 to type T"))
            * time_to_expiry)
        / vol_sqrt_t;

    let d2 = d1 - vol_sqrt_t;

    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let phi_d1 = norm_pdf(d1);

    let discount_factor = (-risk_free_rate * time_to_expiry).exp();

    match option_type.to_lowercase().as_str() {
        "call" => {
            let delta = nd1;
            let gamma = phi_d1 / (spot_price * vol_sqrt_t);
            let theta = -(spot_price * phi_d1 * volatility)
                / (T::from(2.0).expect("Failed to convert 2.0 to type T") * sqrt_t)
                - risk_free_rate * strike_price * discount_factor * nd2;
            let vega = spot_price * phi_d1 * sqrt_t;
            let rho = strike_price * time_to_expiry * discount_factor * nd2;

            Ok((delta, gamma, theta, vega, rho))
        }
        "put" => {
            let delta = nd1 - T::one();
            let gamma = phi_d1 / (spot_price * vol_sqrt_t);
            let theta = -(spot_price * phi_d1 * volatility)
                / (T::from(2.0).expect("Failed to convert 2.0 to type T") * sqrt_t)
                + risk_free_rate * strike_price * discount_factor * norm_cdf(-d2);
            let vega = spot_price * phi_d1 * sqrt_t;
            let rho = -strike_price * time_to_expiry * discount_factor * norm_cdf(-d2);

            Ok((delta, gamma, theta, vega, rho))
        }
        _ => Err(NumRs2Error::InvalidOperation(
            "Option type must be 'call' or 'put'".to_string(),
        )),
    }
}

/// Standard normal probability density function
fn norm_pdf<T>(x: T) -> T
where
    T: Float,
{
    let sqrt_2pi = T::from(2.0 * PI)
        .expect("Failed to convert 2*PI to type T")
        .sqrt();
    (-x * x / T::from(2.0).expect("Failed to convert 2.0 to type T")).exp() / sqrt_2pi
}

/// Calculate implied volatility using Newton-Raphson method
///
/// # Arguments
///
/// * `market_price` - Observed market price of the option
/// * `spot_price` - Current stock price
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiration
/// * `risk_free_rate` - Risk-free rate
/// * `option_type` - "call" or "put"
/// * `initial_vol` - Initial guess for volatility (optional)
///
/// # Returns
///
/// Implied volatility
pub fn implied_volatility<T>(
    market_price: T,
    spot_price: T,
    strike_price: T,
    time_to_expiry: T,
    risk_free_rate: T,
    option_type: &str,
    initial_vol: Option<T>,
) -> Result<T>
where
    T: Float + Debug,
{
    let mut vol =
        initial_vol.unwrap_or_else(|| T::from(0.2).expect("Failed to convert 0.2 to type T"));
    let tolerance = T::from(1e-6).expect("Failed to convert 1e-6 to type T");
    let max_iterations = 100;

    for _ in 0..max_iterations {
        let bs_price = black_scholes(
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            vol,
            option_type,
        )?;
        let price_diff = bs_price - market_price;

        if price_diff.abs() < tolerance {
            return Ok(vol);
        }

        // Calculate vega (sensitivity to volatility)
        let (_delta, _gamma, _theta, vega, _rho) = black_scholes_greeks(
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            vol,
            option_type,
        )?;

        // Newton-Raphson update
        vol = vol - price_diff / vega;

        // Ensure volatility stays positive
        if vol <= T::zero() {
            vol = T::from(0.001).expect("Failed to convert 0.001 to type T");
        }
    }

    Err(NumRs2Error::ComputationError(
        "Implied volatility calculation did not converge".to_string(),
    ))
}

/// Binomial option pricing model
///
/// Calculate option prices using the binomial tree method.
///
/// # Arguments
///
/// * `spot_price` - Current stock price
/// * `strike_price` - Strike price
/// * `time_to_expiry` - Time to expiration
/// * `risk_free_rate` - Risk-free rate
/// * `volatility` - Volatility
/// * `steps` - Number of time steps in the binomial tree
/// * `option_type` - "call" or "put"
/// * `exercise_style` - "european" or "american"
///
/// # Returns
///
/// Option price
pub fn binomial_option_price<T>(
    spot_price: T,
    strike_price: T,
    time_to_expiry: T,
    risk_free_rate: T,
    volatility: T,
    steps: usize,
    option_type: &str,
    exercise_style: &str,
) -> Result<T>
where
    T: Float + Debug + Clone,
{
    if steps == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Number of steps must be positive".to_string(),
        ));
    }

    let dt = time_to_expiry / T::from(steps).expect("Failed to convert steps to type T");
    let up = (volatility * dt.sqrt()).exp();
    let down = T::one() / up;
    let r_factor = (risk_free_rate * dt).exp();
    let p = (r_factor - down) / (up - down);

    // Initialize option values at expiration
    let mut option_values = vec![T::zero(); steps + 1];

    for i in 0..=steps {
        let stock_price = spot_price
            * up.powf(
                T::from(2 * i).expect("Failed to convert 2*i to type T")
                    - T::from(steps).expect("Failed to convert steps to type T"),
            );

        let intrinsic = match option_type.to_lowercase().as_str() {
            "call" => (stock_price - strike_price).max(T::zero()),
            "put" => (strike_price - stock_price).max(T::zero()),
            _ => {
                return Err(NumRs2Error::InvalidOperation(
                    "Option type must be 'call' or 'put'".to_string(),
                ))
            }
        };

        option_values[i] = intrinsic;
    }

    // Work backwards through the tree
    for step in (0..steps).rev() {
        for i in 0..=step {
            // Expected option value
            let expected_value =
                (p * option_values[i + 1] + (T::one() - p) * option_values[i]) / r_factor;

            if exercise_style.to_lowercase() == "american" {
                // For American options, check early exercise
                let stock_price = spot_price
                    * up.powf(
                        T::from(2 * i).expect("Failed to convert 2*i to type T")
                            - T::from(step).expect("Failed to convert step to type T"),
                    );
                let intrinsic = match option_type.to_lowercase().as_str() {
                    "call" => (stock_price - strike_price).max(T::zero()),
                    "put" => (strike_price - stock_price).max(T::zero()),
                    _ => T::zero(),
                };

                option_values[i] = expected_value.max(intrinsic);
            } else {
                option_values[i] = expected_value;
            }
        }
    }

    Ok(option_values[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_black_scholes_call() {
        let call_price = black_scholes(100.0, 105.0, 0.25, 0.05, 0.2, "call")
            .expect("black_scholes call calculation should succeed");
        assert!(call_price > 0.0);
        assert!(call_price < 100.0); // Call price should be less than stock price
    }

    #[test]
    fn test_black_scholes_put() {
        let put_price = black_scholes(100.0, 105.0, 0.25, 0.05, 0.2, "put")
            .expect("black_scholes put calculation should succeed");
        assert!(put_price > 0.0);
        assert!(put_price < 105.0); // Put price should be less than strike price
    }

    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let vol = 0.2;

        let call = black_scholes(s, k, t, r, vol, "call")
            .expect("black_scholes call calculation should succeed");
        let put = black_scholes(s, k, t, r, vol, "put")
            .expect("black_scholes put calculation should succeed");

        // Put-Call Parity: C - P = S - K*e^(-r*T)
        let parity_left = call - put;
        let parity_right = s - k * (-r * t).exp();

        assert_relative_eq!(parity_left, parity_right, epsilon = 1e-10);
    }

    #[test]
    fn test_binomial_vs_black_scholes() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let vol = 0.2;

        let bs_call = black_scholes(s, k, t, r, vol, "call")
            .expect("black_scholes call calculation should succeed");
        let bin_call = binomial_option_price(s, k, t, r, vol, 1000, "call", "european")
            .expect("binomial_option_price calculation should succeed");

        // Binomial should converge to Black-Scholes with many steps
        assert_relative_eq!(bs_call, bin_call, epsilon = 0.1);
    }

    #[test]
    fn test_greeks() {
        let (delta, gamma, _theta, vega, _rho) =
            black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
                .expect("black_scholes_greeks calculation should succeed");

        // Basic sanity checks
        assert!(delta > 0.0 && delta < 1.0); // Call delta should be between 0 and 1
        assert!(gamma > 0.0); // Gamma should be positive
        assert!(vega > 0.0); // Vega should be positive
    }
}
