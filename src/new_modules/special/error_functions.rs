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

    // Region 1: |x| <= 0.5 — Cody rational P/Q approximation for erf(x)/x
    if abs_x <= T::from(0.5_f64).expect("0.5 converts to float") {
        let xsq = abs_x * abs_x;
        let p = [
            T::from(3.209_377_589_138_469_476e3_f64).expect("Cody erf P0"),
            T::from(3.774_852_376_853_020_208e2_f64).expect("Cody erf P1"),
            T::from(1.138_641_541_510_501_556e2_f64).expect("Cody erf P2"),
            T::from(3.161_529_361_120_769_797_f64).expect("Cody erf P3"),
            T::from(1.857_777_061_846_031_527e-1_f64).expect("Cody erf P4"),
        ];
        let q = [
            T::from(2.844_236_833_439_170_622e3_f64).expect("Cody erf Q0"),
            T::from(1.282_168_360_946_988_021e3_f64).expect("Cody erf Q1"),
            T::from(2.440_246_417_242_701_700e2_f64).expect("Cody erf Q2"),
            T::from(2.360_129_095_234_412_093e1_f64).expect("Cody erf Q3"),
            T::from(1.0_f64).expect("Cody erf Q4"),
        ];
        let pval = (((p[4] * xsq + p[3]) * xsq + p[2]) * xsq + p[1]) * xsq + p[0];
        let qval = (((q[4] * xsq + q[3]) * xsq + q[2]) * xsq + q[1]) * xsq + q[0];
        return sign * abs_x * pval / qval;
    }

    // Regions 2 & 3: compute erfc(x) via rational, then erf = 1 - erfc
    let erfc_val = erfc_inner(abs_x);
    let erf_val = if erfc_val > one { zero } else { one - erfc_val };
    sign * erf_val
}

/// Compute erfc(x) for x >= 0 using Cody rational approximation.
/// Called from both erf_scalar and erfc_scalar.
fn erfc_inner<T>(abs_x: T) -> T
where
    T: Float + Debug,
{
    let zero = T::zero();
    let one = T::one();

    if abs_x <= T::from(0.5_f64).expect("0.5 converts to float") {
        // For small x, compute 1 - erf(x) via the same Cody rational
        let xsq = abs_x * abs_x;
        let p = [
            T::from(3.209_377_589_138_469_476e3_f64).expect("Cody P0"),
            T::from(3.774_852_376_853_020_208e2_f64).expect("Cody P1"),
            T::from(1.138_641_541_510_501_556e2_f64).expect("Cody P2"),
            T::from(3.161_529_361_120_769_797_f64).expect("Cody P3"),
            T::from(1.857_777_061_846_031_527e-1_f64).expect("Cody P4"),
        ];
        let q = [
            T::from(2.844_236_833_439_170_622e3_f64).expect("Cody Q0"),
            T::from(1.282_168_360_946_988_021e3_f64).expect("Cody Q1"),
            T::from(2.440_246_417_242_701_700e2_f64).expect("Cody Q2"),
            T::from(2.360_129_095_234_412_093e1_f64).expect("Cody Q3"),
            T::from(1.0_f64).expect("Cody Q4"),
        ];
        let pval = (((p[4] * xsq + p[3]) * xsq + p[2]) * xsq + p[1]) * xsq + p[0];
        let qval = (((q[4] * xsq + q[3]) * xsq + q[2]) * xsq + q[1]) * xsq + q[0];
        return one - abs_x * pval / qval;
    }

    if abs_x <= T::from(4.0_f64).expect("4 converts to float") {
        // 0.5 < |x| <= 4: Cody rational for erfc * exp(x^2), Table II
        let p = [
            T::from(1.230_339_354_797_997_253e4_f64).expect("Cody erfc mid P0"),
            T::from(2.051_078_377_826_071_984e3_f64).expect("Cody erfc mid P1"),
            T::from(-2.128_533_369_396_987_752e2_f64).expect("Cody erfc mid P2"),
            T::from(-2.743_530_793_251_120_636e1_f64).expect("Cody erfc mid P3"),
            T::from(-3.223_090_474_350_511_077e1_f64).expect("Cody erfc mid P4"),
            T::from(-2.700_655_787_090_503_063e0_f64).expect("Cody erfc mid P5"),
            T::from(-1.660_375_222_507_191_527e-2_f64).expect("Cody erfc mid P6"),
        ];
        let q = [
            T::from(1.230_242_360_400_291_894e4_f64).expect("Cody erfc mid Q0"),
            T::from(6.021_058_801_583_350_006e3_f64).expect("Cody erfc mid Q1"),
            T::from(1.478_620_748_818_053_498e3_f64).expect("Cody erfc mid Q2"),
            T::from(2.295_501_944_242_069_539e2_f64).expect("Cody erfc mid Q3"),
            T::from(2.291_647_688_436_398_677e1_f64).expect("Cody erfc mid Q4"),
            T::from(1.370_978_506_694_551_876e0_f64).expect("Cody erfc mid Q5"),
            T::from(1.0_f64).expect("Cody erfc mid Q6"),
        ];
        let pval = (((((p[6] * abs_x + p[5]) * abs_x + p[4]) * abs_x + p[3]) * abs_x + p[2]) * abs_x + p[1]) * abs_x + p[0];
        let qval = (((((q[6] * abs_x + q[5]) * abs_x + q[4]) * abs_x + q[3]) * abs_x + q[2]) * abs_x + q[1]) * abs_x + q[0];
        return (-(abs_x * abs_x)).exp() * pval / qval;
    }

    // |x| > 4: asymptotic via rational minimax in t = 1/x^2
    let xsq = abs_x * abs_x;
    if xsq > T::from(700.0_f64).expect("700 converts to float") {
        return zero;
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
    T: Float + Debug,
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
    let e = erfc_inner(abs_x);
    if x < zero { two - e } else { e }
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

    // Halley iteration: converges cubically — much faster than Newton for erfinv.
    // f(y) = erf(y) - target
    // f'(y) = (2/sqrt(π)) * exp(-y²)   [call this df]
    // f''(y) = -2y * f'(y)
    // Halley denom = f' - f*f''/(2*f') = df - err*(-2y*df)/(2*df) = df + err*y
    let sqrt_pi = T::from(std::f64::consts::PI)
        .expect("PI should convert to float type")
        .sqrt();
    let two_over_sqrt_pi = T::from(2.0_f64).expect("2 converts to float") / sqrt_pi;

    for _ in 0..5 {
        let erf_y = erf_scalar(y);
        let err = erf_y - abs_x;
        if err.abs() < T::epsilon() * T::from(4.0_f64).expect("4 converts to float") {
            break;
        }
        let df = two_over_sqrt_pi * (-y * y).exp();
        // Halley step: y -= err / (df + err*y)
        let halley_denom = df + err * y;
        y = y - err / halley_denom;
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
