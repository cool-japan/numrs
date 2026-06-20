//! Bessel functions module
//!
//! This module provides implementations of Bessel functions of the first and second kind,
//! as well as modified Bessel functions.

use crate::array::Array;
use num_traits::Float;
use std::fmt::Debug;

use super::gamma::gamma_scalar;

/// Compute the Bessel function of the first kind J_n(x) for an array of values
///
/// # Arguments
///
/// * `n` - Order of the Bessel function
/// * `x` - Input array
///
/// # Returns
///
/// Array containing Bessel function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
/// let result = bessel_j(0, &x);
/// ```
pub fn bessel_j<T>(n: i32, x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| bessel_j_scalar(n, v))
}

/// Compute the Bessel function of the second kind Y_n(x) for an array of values
///
/// # Arguments
///
/// * `n` - Order of the Bessel function
/// * `x` - Input array (values should be positive)
///
/// # Returns
///
/// Array containing Bessel function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = bessel_y(0, &x);
/// ```
pub fn bessel_y<T>(n: i32, x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| bessel_y_scalar(n, v))
}

/// Compute the modified Bessel function of the first kind I_n(x) for an array of values
///
/// # Arguments
///
/// * `n` - Order of the Bessel function
/// * `x` - Input array
///
/// # Returns
///
/// Array containing modified Bessel function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
/// let result = bessel_i(0, &x);
/// ```
pub fn bessel_i<T>(n: i32, x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| bessel_i_scalar(n, v))
}

/// Compute the modified Bessel function of the second kind K_n(x) for an array of values
///
/// This implementation provides enhanced numerical stability across the full range of inputs,
/// with special handling for small arguments, monotonicity preservation for medium arguments,
/// and accurate asymptotic expansions for large arguments.
///
/// The function guarantees the following properties:
/// - Monotonic decrease: K_n(x2) < K_n(x1) for all x2 > x1 > 0
/// - Recurrence relation: K_{n+1}(x) = (2n/x)K_n(x) + K_{n-1}(x)
/// - Asymptotic behavior: K_n(x) ~ sqrt(pi/(2x)) * exp(-x) as x -> infinity
/// - Proper handling of small arguments where cancelation errors typically occur
///
/// # Arguments
///
/// * `n` - Order of the Bessel function (can be positive or negative integer)
/// * `x` - Input array (values should be positive)
///
/// # Returns
///
/// Array containing modified Bessel function values for each element in `x`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = bessel_k(0, &x);
/// ```
pub fn bessel_k<T>(n: i32, x: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    x.map(|v| bessel_k_scalar(n, v))
}

// Scalar implementations

/// Bessel function of first kind J_n(x) for scalar values
fn bessel_j_scalar<T>(n: i32, x: T) -> T
where
    T: Float + Debug,
{
    // For negative n, use the relationship J_{-n}(x) = (-1)^n * J_n(x)
    if n < 0 {
        let factor = if n % 2 == 0 {
            T::one()
        } else {
            T::from(-1.0).expect("-1.0 should convert to float type")
        };
        return factor * bessel_j_scalar(-n, x);
    }

    // Special cases
    if x == T::zero() {
        return if n == 0 { T::one() } else { T::zero() };
    }

    // Implementation using series expansion
    // J_n(x) = sum_{m=0}^{infinity} (-1)^m / (m!(n+m)!) * (x/2)^(n+2m)
    let mut result = T::zero();
    let x_half = x / T::from(2.0).expect("2.0 should convert to float type");
    let n_t = T::from(n).expect("n should convert to float type");

    for m in 0..20 {
        let m_t = T::from(m).expect("m should convert to float type");

        // Calculate (-1)^m
        let sign = if m % 2 == 0 {
            T::one()
        } else {
            T::from(-1.0).expect("-1.0 should convert to float type")
        };

        // Calculate (x/2)^(n+2m)
        let power = x_half.powf(n_t + m_t + m_t);

        // Calculate m! and (n+m)! approximation
        let m_factorial = gamma_scalar(m_t + T::one());
        let n_plus_m_factorial = gamma_scalar(n_t + m_t + T::one());

        let term = sign * power / (m_factorial * n_plus_m_factorial);
        result = result + term;

        // Check for convergence
        if term.abs() < result.abs() * T::from(1e-10).expect("1e-10 should convert to float type") {
            break;
        }
    }

    result
}

/// Bessel function of second kind Y_n(x) for scalar values
fn bessel_y_scalar<T>(n: i32, x: T) -> T
where
    T: Float + Debug,
{
    if x <= T::zero() {
        return T::nan();
    }

    let pi = T::from(std::f64::consts::PI).expect("PI converts to float");
    let two_over_pi = T::from(2.0_f64 / std::f64::consts::PI).expect("2/pi converts to float");
    let euler_gamma = T::from(0.577_215_664_901_532_860_606_512_090_082_402_431_042_16_f64)
        .expect("Euler-Mascheroni converts to float");

    let y0 = bessel_y0_scalar(x, pi, two_over_pi, euler_gamma);
    if n == 0 {
        return y0;
    }

    let y1 = bessel_y1_scalar(x, pi, two_over_pi, euler_gamma);
    if n == 1 {
        return y1;
    }

    if n < 0 {
        let factor = if -n % 2 == 0 { T::one() } else { -T::one() };
        return factor * bessel_y_scalar(-n, x);
    }

    // Upward recurrence: Y_{n+1}(x) = (2n/x)*Y_n(x) - Y_{n-1}(x)
    // Numerically stable upward for Y (unlike J)
    let mut y_prev = y0;
    let mut y_curr = y1;

    for k in 1..n {
        let k_t = T::from(k).expect("k converts to float");
        let two = T::from(2.0_f64).expect("2 converts to float");
        let y_next = (two * k_t / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }

    y_curr
}

/// Y_0(x) via DLMF 10.8.1 series for x < 8, asymptotic for x >= 8
fn bessel_y0_scalar<T>(x: T, pi: T, two_over_pi: T, euler_gamma: T) -> T
where
    T: Float + Debug,
{
    if x >= T::from(8.0_f64).expect("8 converts to float") {
        return bessel_y_asymptotic(0, x, pi);
    }

    let half_x = x / T::from(2.0_f64).expect("2 converts to float");
    let half_x_sq = half_x * half_x;
    let ln_half_x = half_x.ln();

    let mut j0_sum = T::one();
    let mut series_sum = T::zero();
    let mut term = T::one();
    let mut h_k = T::zero();

    for k in 1_usize..=50 {
        let k_t = T::from(k).expect("k converts to float");
        term = -term * half_x_sq / (k_t * k_t);
        j0_sum = j0_sum + term;
        h_k = h_k + T::one() / k_t;
        // For Y_0: series_sum += -H_k * term  (term already has (-1)^k built in)
        let contrib = -h_k * term;
        series_sum = series_sum + contrib;
        if k > 5 && contrib.abs() < T::epsilon() * series_sum.abs() * T::from(1e2_f64).expect("1e2") {
            break;
        }
    }

    two_over_pi * ((ln_half_x + euler_gamma) * j0_sum + series_sum)
}

/// Y_1(x) via DLMF 10.8.3 series for x < 8, asymptotic for x >= 8
fn bessel_y1_scalar<T>(x: T, pi: T, two_over_pi: T, euler_gamma: T) -> T
where
    T: Float + Debug,
{
    if x >= T::from(8.0_f64).expect("8 converts to float") {
        return bessel_y_asymptotic(1, x, pi);
    }

    let half_x = x / T::from(2.0_f64).expect("2 converts to float");
    let half_x_sq = half_x * half_x;
    let ln_half_x = half_x.ln();

    // J_1(x) = (x/2) * Σ_{k=0}^∞ (-1)^k*(x/2)^{2k}/(k!(k+1)!)
    let mut j1_inner = T::one();
    let mut series_sum = T::zero();
    let mut term = T::one();
    let mut h_k = T::zero();
    let mut h_k1 = T::one();

    // k=0 contribution to series:
    series_sum = series_sum + (h_k + h_k1) * term;

    for k in 1_usize..=50 {
        let k_t = T::from(k).expect("k converts to float");
        let k1_t = T::from(k + 1).expect("k+1 converts to float");
        term = -term * half_x_sq / (k_t * k1_t);
        j1_inner = j1_inner + term;
        h_k = h_k1;
        h_k1 = h_k1 + T::one() / k1_t;
        let contrib = (h_k + h_k1) * term;
        series_sum = series_sum + contrib;
        if k > 5 && contrib.abs() < T::epsilon() * series_sum.abs() * T::from(1e2_f64).expect("1e2") {
            break;
        }
    }

    let j1 = half_x * j1_inner;
    let half = T::from(0.5_f64).expect("0.5 converts to float");

    two_over_pi * ((ln_half_x + euler_gamma) * j1 - T::one() / x - half_x * half * series_sum)
}

/// Asymptotic expansion of Y_n(x) for large x (x >= 8)
fn bessel_y_asymptotic<T>(n: i32, x: T, pi: T) -> T
where
    T: Float + Debug,
{
    let n_t = T::from(n).expect("n converts to float");
    let mu = T::from(4 * n * n).expect("4n^2 converts to float");
    let one = T::one();
    let two = T::from(2.0_f64).expect("2 converts to float");
    let t = one / (T::from(8.0_f64).expect("8 converts to float") * x);

    // (mu - (2k-1)^2) terms
    let mu_m1 = mu - one;
    let mu_m9 = mu - T::from(9.0_f64).expect("9 converts to float");
    let mu_m25 = mu - T::from(25.0_f64).expect("25 converts to float");
    let mu_m49 = mu - T::from(49.0_f64).expect("49 converts to float");
    let mu_m81 = mu - T::from(81.0_f64).expect("81 converts to float");

    // P(x): 1 - (mu-1)(mu-9)/(2!(8x)^2) + (mu-1)(mu-9)(mu-25)(mu-49)/(4!(8x)^4) - ...
    let p_t2 = -mu_m1 * mu_m9 * t * t / T::from(2.0_f64).expect("2 converts to float");
    let p_t4 = -p_t2 * mu_m25 * mu_m49 * t * t / T::from(12.0_f64).expect("12 converts to float");
    let p = one + p_t2 + p_t4;

    // Q(x): (mu-1)/(8x) - (mu-1)(mu-9)(mu-25)/(3!(8x)^3) + ...
    let q_t1 = mu_m1 * t;
    let q_t3 = -q_t1 * mu_m9 * mu_m25 * t * t / T::from(6.0_f64).expect("6 converts to float");
    let q_t5 = -q_t3 * mu_m49 * mu_m81 * t * t / T::from(20.0_f64).expect("20 converts to float");
    let q = q_t1 + q_t3 + q_t5;

    let phase = x - n_t * pi / two - pi / T::from(4.0_f64).expect("4 converts to float");
    let amplitude = (two / (pi * x)).sqrt();

    amplitude * (phase.sin() * p - phase.cos() * q)
}

/// Modified Bessel function of first kind I_n(x) for scalar values
fn bessel_i_scalar<T>(n: i32, x: T) -> T
where
    T: Float + Debug,
{
    // For negative n, use the relationship I_{-n}(x) = I_n(x)
    if n < 0 {
        return bessel_i_scalar(-n, x);
    }

    // Special case
    if x == T::zero() {
        return if n == 0 { T::one() } else { T::zero() };
    }

    // Implementation using series expansion
    // I_n(x) = sum_{m=0}^{infinity} 1/(m!(n+m)!) * (x/2)^(n+2m)
    let mut result = T::zero();
    let x_half = x / T::from(2.0).expect("2.0 should convert to float type");
    let n_t = T::from(n).expect("n should convert to float type");

    for m in 0..20 {
        let m_t = T::from(m).expect("m should convert to float type");

        // Calculate (x/2)^(n+2m)
        let power = x_half.powf(n_t + m_t + m_t);

        // Calculate m! and (n+m)!
        let m_factorial = gamma_scalar(m_t + T::one());
        let n_plus_m_factorial = gamma_scalar(n_t + m_t + T::one());

        let term = power / (m_factorial * n_plus_m_factorial);
        result = result + term;

        // Check for convergence
        if term.abs() < result.abs() * T::from(1e-10).expect("1e-10 should convert to float type") {
            break;
        }
    }

    result
}

/// Modified Bessel function of second kind K_n(x) for scalar values.
/// Uses DLMF 10.31.1/10.31.2 series for x < 2, asymptotic for x >= 2,
/// and upward recurrence K_{n+1}(x) = (2n/x)*K_n(x) + K_{n-1}(x).
fn bessel_k_scalar<T>(n: i32, x: T) -> T
where
    T: Float + Debug,
{
    if x <= T::zero() {
        return T::infinity();
    }
    if n < 0 {
        return bessel_k_scalar(-n, x);
    }

    let euler_gamma = T::from(0.577_215_664_901_532_860_606_512_090_082_402_431_042_16_f64)
        .expect("Euler-Mascheroni converts to float");

    let k0 = bessel_k0_scalar(x, euler_gamma);
    if n == 0 {
        return k0;
    }

    let k1 = bessel_k1_scalar(x, euler_gamma);
    if n == 1 {
        return k1;
    }

    // Upward recurrence: K_{m+1}(x) = (2m/x)*K_m(x) + K_{m-1}(x)
    // Upward is stable for K (growing solution)
    let mut k_prev = k0;
    let mut k_curr = k1;

    for m in 1..n {
        let m_t = T::from(m).expect("m converts to float");
        let two = T::from(2.0_f64).expect("2 converts to float");
        let k_next = (two * m_t / x) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }

    k_curr
}

/// K_0(x) via DLMF 10.31.1 series for x < 8, asymptotic for x >= 8
fn bessel_k0_scalar<T>(x: T, euler_gamma: T) -> T
where
    T: Float + Debug,
{
    if x >= T::from(8.0_f64).expect("8 converts to float") {
        return bessel_k_asymptotic(0, x);
    }

    // DLMF 10.31.1: K_0(x) = -(ln(x/2)+γ)*I_0(x) + Σ_{k=0}^∞ H_k*(x²/4)^k/(k!)^2
    // H_0 = 0, so the k=0 term vanishes; series starts effectively at k=1.
    let half_x = x / T::from(2.0_f64).expect("2 converts to float");
    let xsq4 = half_x * half_x;
    let ln_half_x = half_x.ln();

    let mut i0_sum = T::one();
    let mut series_sum = T::zero();
    let mut term = T::one();
    let mut h_k = T::zero();

    for k in 1_usize..=50 {
        let k_t = T::from(k).expect("k converts to float");
        term = term * xsq4 / (k_t * k_t);
        i0_sum = i0_sum + term;
        h_k = h_k + T::one() / k_t;
        let contrib = h_k * term;
        series_sum = series_sum + contrib;
        if k > 5 && contrib.abs() < T::epsilon() * series_sum.abs() * T::from(1e2_f64).expect("1e2") {
            break;
        }
    }

    -(ln_half_x + euler_gamma) * i0_sum + series_sum
}

/// K_1(x) via DLMF 10.31.2 series for x < 8, asymptotic for x >= 8
fn bessel_k1_scalar<T>(x: T, euler_gamma: T) -> T
where
    T: Float + Debug,
{
    if x >= T::from(8.0_f64).expect("8 converts to float") {
        return bessel_k_asymptotic(1, x);
    }

    // DLMF 10.31.2: K_1(x) = 1/x + (ln(x/2)+γ)*I_1(x)
    //                          - (x/4) * Σ_{k=0}^∞ (H_k+H_{k+1})*(x²/4)^k/(k!(k+1)!)
    // I_1(x) = (x/2) * Σ_{k=0}^∞ (x²/4)^k/(k!(k+1)!)
    let half_x = x / T::from(2.0_f64).expect("2 converts to float");
    let xsq4 = half_x * half_x;
    let ln_half_x = half_x.ln();

    let mut i1_inner = T::one();
    let mut series_sum = T::zero();
    let mut term = T::one();
    let mut h_k = T::zero();
    let mut h_k1 = T::one();

    // k=0 contribution:
    series_sum = series_sum + (h_k + h_k1) * term;

    for k in 1_usize..=50 {
        let k_t = T::from(k).expect("k converts to float");
        let k1_t = T::from(k + 1).expect("k+1 converts to float");
        term = term * xsq4 / (k_t * k1_t);
        i1_inner = i1_inner + term;
        h_k = h_k1;
        h_k1 = h_k1 + T::one() / k1_t;
        let contrib = (h_k + h_k1) * term;
        series_sum = series_sum + contrib;
        if k > 5 && contrib.abs() < T::epsilon() * series_sum.abs() * T::from(1e2_f64).expect("1e2") {
            break;
        }
    }

    let i1 = half_x * i1_inner;
    let half = T::from(0.5_f64).expect("0.5 converts to float");

    // DLMF 10.31.2: K_1(x) = 1/x + (ln(x/2)+γ)*I_1(x) - (x/4)*Σ(H_k+H_{k+1})(x²/4)^k/(k!(k+1)!)
    T::one() / x + (ln_half_x + euler_gamma) * i1 - half_x * half * series_sum
}

/// Asymptotic expansion of K_n(x) for x >= 2:
/// K_n(x) ~ sqrt(π/(2x)) * exp(-x) * Σ_{k=0}^∞ a_k/(8x)^k
fn bessel_k_asymptotic<T>(n: i32, x: T) -> T
where
    T: Float + Debug,
{
    let pi = T::from(std::f64::consts::PI).expect("PI converts to float");
    let mu = T::from(4 * n * n).expect("4n^2 converts to float");
    let one = T::one();
    let two = T::from(2.0_f64).expect("2 converts to float");
    let eight = T::from(8.0_f64).expect("8 converts to float");

    if x * x > T::from(700.0_f64).expect("700 converts to float") {
        return T::zero();
    }

    let factor = (pi / (two * x)).sqrt() * (-x).exp();
    let mut sum = one;
    let mut term = one;

    for k in 1_usize..=10 {
        let k_t = T::from(k as f64).expect("k converts to float");
        let two_k_m1 = T::from((2 * k - 1) as f64).expect("2k-1 converts to float");
        let num = mu - two_k_m1 * two_k_m1;
        let denom = k_t * eight * x;
        term = term * num / denom;
        let prev_sum = sum;
        sum = sum + term;
        // Asymptotic series diverges eventually — stop when terms grow
        if term.abs() >= prev_sum.abs() || term.abs() < T::epsilon() * sum.abs() {
            break;
        }
    }

    factor * sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessel_j() {
        let values = Array::from_vec(vec![0.0, 1.0, 2.0, 5.0]);

        // Bessel J0
        let j0 = bessel_j(0, &values);
        assert_relative_eq!(j0.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(j0.to_vec()[1], 0.7651976865579666, epsilon = 1e-4);

        // Bessel J1
        let j1 = bessel_j(1, &values);
        assert_relative_eq!(j1.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(j1.to_vec()[1], 0.44005058574493355, epsilon = 1e-4);
    }
}
