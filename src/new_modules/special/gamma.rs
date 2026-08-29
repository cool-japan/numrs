//! Gamma and Beta functions module
//!
//! This module provides implementations of the gamma function, beta function,
//! and related functions (digamma, incomplete gamma, incomplete beta).

use crate::array::Array;
use crate::error::Result;
use num_traits::Float;
use std::fmt::Debug;

// Gamma functions

/// Compute the gamma function (Gamma(x)) of an array of values
///
/// # Arguments
///
/// * `x` - Input array (values should be positive and not integers <= 0)
///
/// # Returns
///
/// Array containing gamma function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.5]);
/// let result = gamma(&x);
/// ```
pub fn gamma<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| gamma_scalar(v))
}

/// Compute the natural logarithm of the gamma function (ln(Gamma(x))) of an array of values
///
/// # Arguments
///
/// * `x` - Input array (values should be positive)
///
/// # Returns
///
/// Array containing logarithm of gamma function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 10.0]);
/// let result = gammaln(&x);
/// ```
pub fn gammaln<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| gammaln_scalar(v))
}

/// Compute the digamma function (Psi(x) = d/dx ln(Gamma(x))) of an array of values
///
/// # Arguments
///
/// * `x` - Input array (values should not be integers <= 0)
///
/// # Returns
///
/// Array containing digamma function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = digamma(&x);
/// ```
pub fn digamma<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| digamma_scalar(v))
}

/// Compute the incomplete gamma function gamma(a, x) of array values
///
/// # Arguments
///
/// * `a` - Shape parameter array
/// * `x` - Upper limit array
///
/// # Returns
///
/// Array containing incomplete gamma function values
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let x = Array::from_vec(vec![1.0, 1.0, 1.0]);
/// let result = gammainc(&a, &x).expect("gammainc should succeed");
/// ```
pub fn gammainc<T>(a: &Array<T>, x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Float + Debug,
{
    a.zip_with(x, |a_val, x_val| gammainc_scalar(a_val, x_val))
}

// Beta functions

/// Compute the beta function B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b) for arrays of values
///
/// # Arguments
///
/// * `a` - First parameter array
/// * `b` - Second parameter array
///
/// # Returns
///
/// Array containing beta function values
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = beta(&a, &b).expect("beta should succeed");
/// ```
pub fn beta<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Float + Debug,
{
    a.zip_with(b, |a_val, b_val| beta_scalar(a_val, b_val))
}

/// Compute the incomplete beta function I_x(a, b) for arrays of values
///
/// # Arguments
///
/// * `a` - First parameter array
/// * `b` - Second parameter array
/// * `x` - Upper limit array (should be in `[0,1]`)
///
/// # Returns
///
/// Array containing incomplete beta function values
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let x = Array::from_vec(vec![0.5, 0.5, 0.5]);
/// let result = betainc(&a, &b, &x).expect("betainc should succeed");
/// ```
pub fn betainc<T>(a: &Array<T>, b: &Array<T>, x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Float + Debug,
{
    // For simplicity, assume all arrays have the same shape and use element-wise operations
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let x_vec = x.to_vec();

    let mut result = Vec::with_capacity(a_vec.len());
    for i in 0..a_vec.len() {
        result.push(betainc_scalar(a_vec[i], b_vec[i], x_vec[i]));
    }

    Ok(Array::from_vec(result))
}

// Scalar implementations

/// Lanczos approximation coefficients for gamma function
fn lanczos_coefficients<T>() -> Vec<T>
where
    T: Float + Debug,
{
    // Lanczos coefficients for g=7
    let coeffs = [
        0.999_999_999_999_81,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653,
        -176.615_029_162_141,
        12.507_343_278_687,
        -0.138_571_095_265_72,
        9.984_369_578_02e-6,
        1.505_632_735_15e-7,
    ];

    coeffs
        .iter()
        .map(|&x| T::from(x).expect("x should convert to float type"))
        .collect()
}

/// Gamma function for a scalar value
pub(crate) fn gamma_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // Using Lanczos approximation for the gamma function

    // Handle special cases
    if x == T::zero() {
        return T::infinity();
    }
    if x < T::zero() {
        // Reflection formula
        let pi = T::from(std::f64::consts::PI).expect("PI should convert to float type");
        return pi / ((pi * x).sin() * gamma_scalar(T::one() - x));
    }

    let g = T::from(7.0).expect("7.0 should convert to float type"); // Lanczos parameter
    let coeffs = lanczos_coefficients::<T>();

    // Shift x by 1 to accommodate the algorithm
    let x = x - T::one();

    // Calculate the approximation
    let mut sum = coeffs[0];
    for (i, &coeff) in coeffs.iter().enumerate().skip(1) {
        sum = sum + coeff / (x + T::from(i).expect("i should convert to float type"));
    }

    let t = x + g + T::from(0.5).expect("0.5 should convert to float type");
    let sqrt_2pi = T::from(2.506_628_274_631).expect("sqrt(2*pi) should convert to float type"); // sqrt(2*pi)

    sqrt_2pi
        * sum
        * t.powf(x + T::from(0.5).expect("0.5 should convert to float type"))
        * (-t).exp()
}

/// Natural logarithm of gamma function for a scalar value
fn gammaln_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // Handle special cases
    if x <= T::zero() {
        return T::infinity();
    }

    // For larger values, use Stirling's approximation
    if x > T::from(10.0).expect("10.0 should convert to float type") {
        let pi = T::from(std::f64::consts::PI).expect("PI should convert to float type");
        let e = T::from(std::f64::consts::E).expect("E should convert to float type");

        return (T::from(0.5).expect("0.5 should convert to float type")
            * (T::from(2.0).expect("2.0 should convert to float type") * pi / x).ln())
            + (x * (x / e).ln());
    }

    gamma_scalar(x).ln()
}

/// Digamma function for a scalar value
fn digamma_scalar<T>(x: T) -> T
where
    T: Float + Debug,
{
    // For x <= 0, use the reflection formula
    if x <= T::zero() {
        let pi = T::from(std::f64::consts::PI).expect("PI should convert to float type");
        return digamma_scalar(T::one() - x) - pi / (pi * x).tan();
    }

    // For small x, use recurrence relation to increase argument
    if x < T::from(10.0).expect("10.0 should convert to float type") {
        return digamma_scalar(x + T::one()) - T::one() / x;
    }

    // For large x, use asymptotic expansion
    let inv_x = T::one() / x;
    let inv_x2 = inv_x * inv_x;

    // Coefficients from the asymptotic expansion
    x.ln()
        - inv_x / T::from(2.0).expect("2.0 should convert to float type")
        - inv_x2 / T::from(12.0).expect("12.0 should convert to float type")
        + inv_x2 * inv_x2 / T::from(120.0).expect("120.0 should convert to float type")
        - inv_x2 * inv_x2 * inv_x2 / T::from(252.0).expect("252.0 should convert to float type")
}

/// Incomplete gamma function for scalar values
fn gammainc_scalar<T>(a: T, x: T) -> T
where
    T: Float + Debug,
{
    // Handle special cases
    if x <= T::zero() {
        return T::zero();
    }
    if x > T::from(1000.0).expect("1000.0 should convert to float type")
        && a < T::from(1000.0).expect("1000.0 should convert to float type")
    {
        // For very large x, the result is approximately 1
        return T::one();
    }

    // Series expansion for small x
    if x < a + T::one() {
        let mut result = T::one() / a;
        let mut term = result;

        for n in 1..100 {
            term = term * x / (a + T::from(n).expect("n should convert to float type"));
            result = result + term;

            // Check for convergence
            if term.abs()
                < result.abs() * T::from(1e-14).expect("1e-14 should convert to float type")
            {
                break;
            }
        }

        return result * x.powf(a) * (-x).exp() / gamma_scalar(a);
    }

    // Continued fraction for larger x
    // Lentz's algorithm from Numerical Recipes §6.2 (gcf routine)
    // Evaluates: Q(a,x) = e^{-x} x^a / Gamma(a) * CF
    // CF numerators: a_n = -n*(n-a); denominators: b_n = x + 2n + 1 - a
    let fpmin = T::from(1.0e-30).expect("1.0e-30 should convert to float type");
    let eps = T::from(1.0e-14).expect("1.0e-14 should convert to float type");
    let two = T::from(2.0).expect("2.0 should convert to float type");

    let mut b = x + T::one() - a;
    let mut c = T::one() / fpmin;
    let mut d = if b.abs() < fpmin { fpmin } else { T::one() / b };
    let mut h = d;

    for i in 1..200 {
        let i_t = T::from(i).expect("i should convert to float type");
        let an = -i_t * (i_t - a);

        b = b + two;

        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        d = T::one() / d;

        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }

        let del = d * c;
        h = h * del;

        if (del - T::one()).abs() < eps {
            break;
        }
    }

    T::one() - h * x.powf(a) * (-x).exp() / gamma_scalar(a)
}

/// Beta function for scalar values
fn beta_scalar<T>(a: T, b: T) -> T
where
    T: Float + Debug,
{
    // B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)
    gamma_scalar(a) * gamma_scalar(b) / gamma_scalar(a + b)
}

/// Incomplete beta function for scalar values using continued fraction
fn betainc_scalar<T>(a: T, b: T, x: T) -> T
where
    T: Float + Debug,
{
    // Handle boundary cases
    if x <= T::zero() {
        return T::zero();
    }
    if x >= T::one() {
        return T::one();
    }

    // For small a or b, use series expansion
    // Otherwise use continued fraction
    let bt = x.powf(a) * (T::one() - x).powf(b) / beta_scalar(a, b);

    if x < (a + T::one()) / (a + b + T::from(2.0).expect("2.0 should convert to float type")) {
        // Use continued fraction from the lower tail
        bt * betcf_scalar(a, b, x) / a
    } else {
        // Use continued fraction from the upper tail
        T::one() - bt * betcf_scalar(b, a, T::one() - x) / b
    }
}

/// Continued fraction for incomplete beta function
fn betcf_scalar<T>(a: T, b: T, x: T) -> T
where
    T: Float + Debug,
{
    let eps = T::from(1.0e-15).expect("1.0e-15 should convert to float type");
    let fpmin = T::from(1.0e-30).expect("1.0e-30 should convert to float type");

    let qab = a + b;
    let qap = a + T::one();
    let qam = a - T::one();
    let mut c = T::one();
    let mut d = T::one() - qab * x / qap;

    if d.abs() < fpmin {
        d = fpmin;
    }
    d = T::one() / d;
    let mut h = d;

    for m in 1..100 {
        let m_t = T::from(m).expect("m should convert to float type");
        let m2 = T::from(2 * m).expect("2*m should convert to float type");

        // Even step
        let aa = m_t * (b - m_t) * x / ((qam + m2) * (a + m2));
        d = T::one() + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = T::one() + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = T::one() / d;
        h = h * d * c;

        // Odd step
        let aa = -(a + m_t) * (qab + m_t) * x / ((a + m2) * (qap + m2));
        d = T::one() + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = T::one() + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = T::one() / d;
        let del = d * c;
        h = h * del;

        if (del - T::one()).abs() < eps {
            break;
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma() {
        let values = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma(&values);

        // Known values of gamma
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[4], 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_beta() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let result = beta(&a, &b).expect("beta should succeed");

        // B(1,1) = 1, B(2,2) = 1/6, B(3,3) = 1/30
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[1], 1.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[2], 1.0 / 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_betainc() {
        let a = Array::from_vec(vec![1.0, 2.0]);
        let b = Array::from_vec(vec![1.0, 2.0]);
        let x = Array::from_vec(vec![0.5, 0.5]);
        let result = betainc(&a, &b, &x).expect("betainc should succeed");

        // I_0.5(1,1) = 0.5, I_0.5(2,2) = 0.5
        assert_relative_eq!(result.to_vec()[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[1], 0.5, epsilon = 1e-4);
    }
}
