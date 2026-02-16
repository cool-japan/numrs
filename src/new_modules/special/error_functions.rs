//! Error functions module
//!
//! This module provides implementations of error functions (erf, erfc)
//! and their inverses (erfinv, erfcinv).

use crate::array::Array;
use num_traits::Float;
use std::fmt::Debug;

/// Compute the error function (erf) of an array of values
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array containing error function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 0.5, 1.0]);
/// let result = erf(&x);
/// ```
pub fn erf<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| erf_scalar(v))
}

/// Compute the complementary error function (erfc) of an array of values
/// erfc(x) = 1 - erf(x)
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array containing complementary error function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 0.5, 1.0]);
/// let result = erfc(&x);
/// ```
pub fn erfc<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| erfc_scalar(v))
}

/// Compute the inverse error function (erf^-1) of an array of values
///
/// # Arguments
///
/// * `x` - Input array (values should be in the range [-1, 1])
///
/// # Returns
///
/// Array containing inverse error function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 0.5, 0.8]);
/// let result = erfinv(&x);
/// ```
pub fn erfinv<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| erfinv_scalar(v))
}

/// Compute the inverse complementary error function (erfc^-1) of an array of values
///
/// # Arguments
///
/// * `x` - Input array (values should be in the range [0, 2])
///
/// # Returns
///
/// Array containing inverse complementary error function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.1, 0.5, 1.0]);
/// let result = erfcinv(&x);
/// ```
pub fn erfcinv<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| erfcinv_scalar(v))
}

/// Error function for a scalar value
pub(crate) fn erf_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // Improved error function using rational approximation
    // Based on W. J. Cody's algorithm with high precision coefficients

    let zero = T::zero();
    let one = T::one();
    let abs_x = x.abs();

    // Handle special cases
    if x.is_nan() {
        return x;
    }
    if x == zero {
        return zero;
    }
    if x.is_infinite() {
        return if x > zero { one } else { -one };
    }

    let sign = if x < zero { -one } else { one };

    // For small x, use series expansion
    if abs_x < T::from(0.5).expect("0.5 should convert to float type") {
        let x2 = abs_x * abs_x;
        let sqrt_pi = T::from(1.7724538509055160272981674833411)
            .expect("sqrt(PI) should convert to float type"); // sqrt(pi)

        // erf(x) = (2/sqrt(pi)) * x * sum((-1)^n * x^(2n) / (n! * (2n+1)))
        let mut sum = one;
        let mut term = one;

        for n in 1..=50 {
            term = term * (-x2) / T::from(n as f64).expect("n should convert to float type");
            let add_term =
                term / T::from((2 * n + 1) as f64).expect("2n+1 should convert to float type");
            sum = sum + add_term;

            if add_term.abs() < T::from(1e-15).expect("1e-15 should convert to float type") {
                break;
            }
        }

        return sign
            * (T::from(2.0).expect("2.0 should convert to float type") / sqrt_pi)
            * abs_x
            * sum;
    }

    // For larger x, use Chebyshev rational approximation
    // Based on Hart et al. approximations
    if abs_x < T::from(4.0).expect("4.0 should convert to float type") {
        let t = one
            / (one + T::from(0.3275911).expect("coefficient should convert to float type") * abs_x);

        // Coefficients for improved approximation
        let a1 = T::from(0.254829592).expect("a1 coefficient should convert to float type");
        let a2 = T::from(-0.284496736).expect("a2 coefficient should convert to float type");
        let a3 = T::from(1.421413741).expect("a3 coefficient should convert to float type");
        let a4 = T::from(-1.453152027).expect("a4 coefficient should convert to float type");
        let a5 = T::from(1.061405429).expect("a5 coefficient should convert to float type");

        let poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1;
        let result = one - poly * t * (-abs_x * abs_x).exp();

        return sign * result;
    }

    // For very large x, erf(x) approaches +/-1
    sign * one
}

/// Complementary error function for a scalar value
fn erfc_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // erfc(x) = 1 - erf(x)
    T::one() - erf_scalar(x)
}

/// Robust inverse error function using Newton-Raphson iteration
/// Based on a simple but reliable algorithm
fn erfinv_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // Check input range
    if x < T::from(-1.0).expect("-1.0 should convert to float type") {
        return T::neg_infinity();
    }
    if x > T::one() {
        return T::infinity();
    }
    if x == T::zero() {
        return T::zero();
    }

    // Use symmetry: erfinv(-x) = -erfinv(x)
    let sign = if x < T::zero() {
        T::from(-1.0).expect("-1.0 should convert to float type")
    } else {
        T::one()
    };
    let abs_x = x.abs();

    // Initial guess using a simple rational approximation
    let mut y = if abs_x <= T::from(0.7).expect("0.7 should convert to float type") {
        // For central region, use simple polynomial approximation
        let t = abs_x * abs_x;
        abs_x
            * (T::from(0.8862269254527579).expect("coefficient should convert to float type")
                + t * T::from(0.23201607781175).expect("coefficient should convert to float type"))
    } else {
        // For tail region, use logarithmic approximation
        let w = (-((T::one() - abs_x) * (T::one() + abs_x)).ln()).sqrt();
        if abs_x < T::from(0.97).expect("0.97 should convert to float type") {
            w * (T::from(1.641345311).expect("coefficient should convert to float type")
                - T::from(0.329912874).expect("coefficient should convert to float type") * w)
        } else {
            w * (T::from(1.641345311).expect("coefficient should convert to float type")
                - T::from(0.329912874).expect("coefficient should convert to float type") * w
                + T::from(0.012229801).expect("coefficient should convert to float type") * w * w)
        }
    };

    // Newton-Raphson iteration to refine the result
    let sqrt_pi = T::from(std::f64::consts::PI)
        .expect("PI should convert to float type")
        .sqrt();
    let two_over_sqrt_pi = T::from(2.0).expect("2.0 should convert to float type") / sqrt_pi;

    for _ in 0..3 {
        // Calculate erf(y) and the error
        let erf_y = erf_scalar(y);
        let error = erf_y - abs_x;

        // Check convergence
        if error.abs() < T::epsilon() * T::from(100.0).expect("100.0 should convert to float type")
        {
            break;
        }

        // Newton-Raphson step: y_{n+1} = y_n - f(y_n)/f'(y_n)
        // where f(y) = erf(y) - target and f'(y) = (2/sqrt(pi)) * exp(-y^2)
        let derivative = two_over_sqrt_pi * (-y * y).exp();
        y = y - error / derivative;
    }

    sign * y
}

/// Inverse complementary error function for a scalar value
fn erfcinv_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // erfcinv(x) = erfinv(1 - x)
    erfinv_scalar(T::one() - x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_erf() {
        let values = Array::from_vec(vec![0.0, 0.5, 1.0, -0.5]);
        let result = erf(&values);

        // Known values of erf
        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 0.5204998778130465, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.8427007929497149, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], -0.5204998778130465, epsilon = 1e-4);
    }

    #[test]
    fn test_erfc() {
        let values = Array::from_vec(vec![0.0, 0.5, 1.0, 2.0]);
        let result = erfc(&values);

        // Known values of erfc
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 0.4795001221869535, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.15729920705028513, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], 0.0046777349810472645, epsilon = 1e-4);
    }
}
