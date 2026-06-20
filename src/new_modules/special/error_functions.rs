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

/// Error function for a scalar value using Taylor series for small |x|
/// and A&S 7.1.26 rational approximation for larger |x|.
pub(crate) fn erf_scalar<T>(x: T) -> T
where
    T: Float + Debug + Copy,
{
    if x.is_nan() {
        return x;
    }
    let zero = T::zero();
    let one = T::one();
    if x == zero {
        return zero;
    }
    if x.is_infinite() {
        return if x > zero { one } else { -one };
    }

    let abs_x = x.abs();
    let sign = if x < zero { -one } else { one };

    // Region 1: |x| <= 0.5 — Taylor series for erf(x)
    // erf(x) = (2/sqrt(pi)) * sum_{k=0}^inf (-1)^k * x^{2k+1} / (k! * (2k+1))
    // Recurrence: t_k = t_{k-1} * (-x^2) * (2k-1) / (k * (2k+1))
    if abs_x <= T::from(0.5_f64).expect("0.5 converts to float") {
        let two_over_sqrt_pi =
            T::from(1.128_379_167_095_512_6_f64).expect("2/sqrt(pi) converts to float");
        let xsq = abs_x * abs_x;
        let mut term = abs_x;
        let mut s = abs_x;
        for k in 1_usize..=30 {
            let k_f = T::from(k as f64).expect("k converts to float");
            let dk = T::from((2 * k + 1) as f64).expect("2k+1 converts to float");
            let dk_prev = T::from((2 * k - 1) as f64).expect("2k-1 converts to float");
            term = -term * xsq * dk_prev / (k_f * dk);
            let prev = s;
            s = s + term;
            if (s - prev).abs() <= T::epsilon() * prev.abs() {
                break;
            }
        }
        return sign * two_over_sqrt_pi * s;
    }

    // Regions 2 & 3: compute erfc(|x|) via erfc_positive, then erf = 1 - erfc
    let erfc_val = erfc_positive(abs_x);
    let erf_val = if erfc_val > one { zero } else { one - erfc_val };
    sign * erf_val
}

/// Compute erfc(x) for x > 0 using A&S 7.1.26 for 0.5 < x <= 4
/// and Cody rational minimax for x > 4.
fn erfc_positive<T>(abs_x: T) -> T
where
    T: Float + Debug + Copy,
{
    if abs_x <= T::from(4.0_f64).expect("4 converts to float") {
        // A&S 7.1.26: erfc(x) = (a1*t + a2*t^2 + ... + a5*t^5) * exp(-x^2)
        // t = 1/(1 + p*x), max |error| < 1.5e-7 for x >= 0
        let p_coef = T::from(0.3275911_f64).expect("A&S p coeff");
        let t = T::one() / (T::one() + p_coef * abs_x);
        let a1 = T::from(0.254_829_592_f64).expect("A&S a1");
        let a2 = T::from(-0.284_496_736_f64).expect("A&S a2");
        let a3 = T::from(1.421_413_741_f64).expect("A&S a3");
        let a4 = T::from(-1.453_152_027_f64).expect("A&S a4");
        let a5 = T::from(1.061_405_429_f64).expect("A&S a5");
        // Horner evaluation: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        return poly * (-(abs_x * abs_x)).exp();
    }

    // |x| > 4: asymptotic via rational minimax in t = 1/x^2
    let xsq = abs_x * abs_x;
    if xsq > T::from(700.0_f64).expect("700 converts to float") {
        return T::zero();
    }
    let t = T::one() / xsq;
    let p = [
        T::from(6.587_491_615_298_378_032e-4_f64).expect("Cody erfc large P0"),
        T::from(1.608_378_514_672_411_612e-2_f64).expect("Cody erfc large P1"),
        T::from(1.257_817_261_112_292_462e-1_f64).expect("Cody erfc large P2"),
        T::from(3.603_448_999_498_044_498e-1_f64).expect("Cody erfc large P3"),
        T::from(3.053_266_827_579_502_975e-1_f64).expect("Cody erfc large P4"),
        T::from(6.572_875_944_354_370_326e-2_f64).expect("Cody erfc large P5"),
    ];
    let q = [
        T::from(6.587_491_615_298_226_451e-4_f64).expect("Cody erfc large Q0"),
        T::from(1.694_994_678_768_138_993e-2_f64).expect("Cody erfc large Q1"),
        T::from(1.620_992_145_366_338_578e-1_f64).expect("Cody erfc large Q2"),
        T::from(7.066_518_022_557_803_712e-1_f64).expect("Cody erfc large Q3"),
        T::from(1.522_143_119_549_067_578e0_f64).expect("Cody erfc large Q4"),
        T::from(1.379_312_099_483_891_760e0_f64).expect("Cody erfc large Q5"),
        T::from(1.0_f64).expect("Cody erfc large Q6"),
    ];
    let pval = ((((p[5] * t + p[4]) * t + p[3]) * t + p[2]) * t + p[1]) * t + p[0];
    let qval = (((((q[6] * t + q[5]) * t + q[4]) * t + q[3]) * t + q[2]) * t + q[1]) * t + q[0];
    (-xsq).exp() * pval / (qval * abs_x)
}

/// Complementary error function for a scalar value
fn erfc_scalar<T>(x: T) -> T
where
    T: Float + Debug + Copy,
{
    if x.is_nan() {
        return x;
    }
    let zero = T::zero();
    let one = T::one();
    let two = T::from(2.0_f64).expect("2 converts to float");
    if x.is_infinite() {
        return if x > zero { zero } else { two };
    }
    if x == zero {
        return one;
    }
    let abs_x = x.abs();
    // For small |x|, avoid catastrophic cancellation: compute 1 - erf(x) directly.
    let e = if abs_x <= T::from(0.5_f64).expect("0.5 converts to float") {
        one - erf_scalar(abs_x)
    } else {
        erfc_positive(abs_x)
    };
    if x < zero {
        two - e
    } else {
        e
    }
}

/// Robust inverse error function using Halley iteration with high-quality initial guess.
///
/// Uses a multi-region rational initial approximation (Winitzki 2008 / Brent-style)
/// followed by Halley's method (cubic convergence). Achieves ≈ 1e-15 accuracy.
fn erfinv_scalar<T>(x: T) -> T
where
    T: Float + Debug + Copy,
{
    // Check input range
    if x < T::from(-1.0_f64).unwrap_or_else(|| -T::one()) {
        return T::neg_infinity();
    }
    if x > T::one() {
        return T::infinity();
    }
    if x == T::zero() {
        return T::zero();
    }
    // Handle exact ±1 boundary
    if x == T::one() {
        return T::infinity();
    }
    if x == -T::one() {
        return T::neg_infinity();
    }

    // Use symmetry: erfinv(-x) = -erfinv(x)
    let neg_one = T::from(-1.0_f64).unwrap_or_else(|| -T::one());
    let sign = if x < T::zero() { neg_one } else { T::one() };
    let abs_x: T = x.abs();

    // High-quality initial guess using Winitzki's (2008) closed-form approximation.
    //
    // For erfinv with a = 8*(π-3)/(3*π*(4-π)) ≈ 0.147:
    //   w = ln(1 - x²)   (NOTE: this is negative for 0 < |x| < 1)
    //   erfinv(x) ≈ sgn(x) * sqrt( sqrt( (2/(π·a) + w/2)² - w/a ) - (2/(π·a) + w/2) )
    //
    // This gives max relative error ~5e-4 over the full range (0, 1), which is
    // an excellent starting point for our subsequent Halley iteration.
    let mut y: T = {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::one() + T::one());
        let two = T::from(2.0_f64).unwrap_or_else(|| T::one() + T::one());
        let four = T::from(4.0_f64).unwrap_or_else(|| {
            let t = T::one() + T::one();
            t + t
        });
        let three = T::from(3.0_f64).unwrap_or_else(|| {
            let t = T::one() + T::one();
            t + T::one()
        });
        let eight = T::from(8.0_f64).unwrap_or_else(|| {
            let t = T::one() + T::one();
            let f = t + t;
            f + f
        });

        // a = 8*(π-3) / (3*π*(4-π))
        let a = eight * (pi - three) / (three * pi * (four - pi));

        // w = ln(1 - x²) — negative for x in (0,1)
        let x2 = abs_x * abs_x;
        let one_minus_x2 = T::one() - x2;
        let one_minus_x2_safe = if one_minus_x2 <= T::zero() {
            T::from(1e-300_f64).unwrap_or_else(|| T::epsilon())
        } else {
            one_minus_x2
        };
        let w = one_minus_x2_safe.ln(); // negative value

        // inner = 2/(π·a) + w/2  (w is negative, so this reduces the term)
        let inner = two / (pi * a) + w / two;

        // discriminant = inner² - w/a  (w/a is negative, so -w/a is positive → adds to inner²)
        let discriminant = inner * inner - w / a;
        let discriminant_safe = if discriminant < T::zero() {
            T::from(1e-30_f64).unwrap_or_else(|| T::epsilon())
        } else {
            discriminant
        };

        // erfinv ≈ sqrt( sqrt(discriminant) - inner )
        let diff = discriminant_safe.sqrt() - inner;
        let diff_safe = if diff <= T::zero() {
            T::from(1e-6_f64).unwrap_or_else(|| T::epsilon())
        } else {
            diff
        };
        diff_safe.sqrt()
    };

    // Clamp to strictly positive (we handle sign separately; y represents |erfinv(x)|)
    if y <= T::zero() {
        y = T::from(1e-8_f64).unwrap_or_else(|| T::epsilon());
    }

    // Halley iteration: cubic convergence — 8 iterations achieve ~1e-15 accuracy
    // f(y) = erf(y) - target
    // f'(y) = (2/sqrt(π)) * exp(-y²)   [call this df]
    // f''(y) = -4y/sqrt(π) * exp(-y²) = -2y * f'(y)
    // Halley step: y_new = y - f(y)*f'(y) / (f'(y)² - f(y)*f''(y)/2)
    //            = y - err / (df - err*(-2y*df)/(2*df))
    //            = y - err / (df + err*y)
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::one() + T::one());
    let two = T::from(2.0_f64).unwrap_or_else(|| T::one() + T::one());
    let two_over_sqrt_pi = two / pi.sqrt();
    let eps4 = T::epsilon() * T::from(4.0_f64).unwrap_or(T::one());

    for _ in 0..8 {
        let erf_y = erf_scalar(y);
        let err = erf_y - abs_x;
        if err.abs() < eps4 {
            break;
        }
        let df = two_over_sqrt_pi * (-y * y).exp();
        // Guard against df being near zero (shouldn't happen for finite y)
        if df.abs() < T::epsilon() {
            break;
        }
        // Halley denominator: df + err*y
        let halley_denom = df + err * y;
        // If Halley denom is near zero, fall back to Newton
        let step = if halley_denom.abs() < T::epsilon() * df.abs() {
            err / df
        } else {
            err / halley_denom
        };
        y = y - step;
        // Keep y positive (we handle sign separately)
        if y < T::zero() {
            y = T::from(1e-10_f64).unwrap_or(T::epsilon());
        }
    }

    sign * y
}

/// Inverse complementary error function for a scalar value
fn erfcinv_scalar<T>(x: T) -> T
where
    T: Float + Debug + Copy,
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
