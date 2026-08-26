//! Elliptic functions and integrals module
//!
//! This module provides implementations of elliptic integrals (complete and incomplete)
//! and Jacobi elliptic functions.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

// Complete Elliptic Integrals

/// Compute the complete elliptic integral of the first kind K(m) for an array of values
///
/// # Arguments
///
/// * `m` - Input array (parameter values)
///
/// # Returns
///
/// Array containing elliptic integral values for each element in `m`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let m = Array::from_vec(vec![0.0, 0.5, 0.9]);
/// let result = ellipk(&m);
/// ```
pub fn ellipk<T>(m: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    m.map(|v| ellipk_scalar(v))
}

/// Compute the complete elliptic integral of the second kind E(m) for an array of values
///
/// # Arguments
///
/// * `m` - Input array (parameter values)
///
/// # Returns
///
/// Array containing elliptic integral values for each element in `m`
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let m = Array::from_vec(vec![0.0, 0.5, 0.9]);
/// let result = ellipe(&m);
/// ```
pub fn ellipe<T>(m: &Array<T>) -> Array<T>
where
    T: Clone + Float + Debug,
{
    m.map(|v| ellipe_scalar(v))
}

// Incomplete Elliptic Integrals

/// Incomplete elliptic integral of the first kind F(phi, m)
///
/// F(phi, m) = integral from 0 to phi of dt/sqrt(1-m*sin^2(t))
///
/// # Arguments
///
/// * `phi` - Amplitude array
/// * `m` - Parameter array
///
/// # Returns
///
/// Array containing F(phi, m) values
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let phi = Array::from_vec(vec![0.0, std::f64::consts::PI/4.0, std::f64::consts::PI/2.0]);
/// let m = Array::from_vec(vec![0.5, 0.5, 0.5]);
/// let result = ellipf(&phi, &m).expect("ellipf should succeed");
/// ```
pub fn ellipf<T>(phi: &Array<T>, m: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Float + Debug,
{
    phi.zip_with(m, |p, k| ellipf_scalar(p, k))
}

/// Incomplete elliptic integral of the second kind E(phi, m)
///
/// E(phi, m) = integral from 0 to phi of sqrt(1-m*sin^2(t)) dt
///
/// # Arguments
///
/// * `phi` - Amplitude array
/// * `m` - Parameter array
///
/// # Returns
///
/// Array containing E(phi, m) values
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let phi = Array::from_vec(vec![0.0, std::f64::consts::PI/4.0, std::f64::consts::PI/2.0]);
/// let m = Array::from_vec(vec![0.5, 0.5, 0.5]);
/// let result = ellipeinc(&phi, &m).expect("ellipeinc should succeed");
/// ```
pub fn ellipeinc<T>(phi: &Array<T>, m: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Float + Debug,
{
    phi.zip_with(m, |p, k| ellipeinc_scalar(p, k))
}

// Jacobi Elliptic Functions

/// Compute Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m)
///
/// # Arguments
///
/// * `u` - Argument array
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
///
/// Tuple of (sn, cn, dn) arrays
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let u = Array::from_vec(vec![0.0, 0.5, 1.0]);
/// let (sn, cn, dn) = jacobi_elliptic(&u, 0.5).expect("jacobi_elliptic should succeed");
/// ```
pub fn jacobi_elliptic<T>(u: &Array<T>, m: T) -> Result<(Array<T>, Array<T>, Array<T>)>
where
    T: Clone + Float + Debug,
{
    if m < T::zero() || m > T::one() {
        return Err(NumRs2Error::ValueError(
            "Parameter m must be in [0, 1]".to_string(),
        ));
    }

    let u_vec = u.to_vec();
    let mut sn_vec = Vec::with_capacity(u_vec.len());
    let mut cn_vec = Vec::with_capacity(u_vec.len());
    let mut dn_vec = Vec::with_capacity(u_vec.len());

    for &u_val in &u_vec {
        let (sn, cn, dn) = jacobi_elliptic_scalar(u_val, m);
        sn_vec.push(sn);
        cn_vec.push(cn);
        dn_vec.push(dn);
    }

    Ok((
        Array::from_vec(sn_vec),
        Array::from_vec(cn_vec),
        Array::from_vec(dn_vec),
    ))
}

// Scalar implementations

/// Complete elliptic integral of the first kind K(m) for scalar values
pub(crate) fn ellipk_scalar<T>(m: T) -> T
where
    T: Float + Debug,
{
    // Check input range
    if m > T::one() {
        return T::nan();
    }
    if m == T::one() {
        return T::infinity();
    }
    if m == T::zero() {
        return T::from(std::f64::consts::PI / 2.0).expect("PI/2 should convert to float type");
    }

    // Implementation using arithmetic-geometric mean method
    let pi = T::from(std::f64::consts::PI).expect("PI should convert to float type");
    let one_minus_m = T::one() - m;

    // Initialize arithmetic and geometric means
    let mut a = T::one();
    let mut g = one_minus_m.sqrt();

    // Convergence criterion: converge to near machine epsilon
    let eps = T::epsilon() * T::from(4.0_f64).expect("4.0 converts to float")
        + T::from(1e-14_f64).expect("1e-14 converts to float");

    // Iterate until convergence
    while (a - g).abs() > a * eps {
        let a_next = (a + g) / T::from(2.0).expect("2.0 should convert to float type");
        let g_next = (a * g).sqrt();

        a = a_next;
        g = g_next;
    }

    pi / (T::from(2.0).expect("2.0 should convert to float type") * a)
}

/// Complete elliptic integral of the second kind E(m) for scalar values
pub(crate) fn ellipe_scalar<T>(m: T) -> T
where
    T: Float + Debug,
{
    if m > T::one() {
        return T::nan();
    }
    if m == T::one() {
        return T::one();
    }
    if m == T::zero() {
        return T::from(std::f64::consts::PI / 2.0).expect("PI/2 converts to float");
    }

    let pi = T::from(std::f64::consts::PI).expect("PI converts to float");
    let two = T::from(2.0_f64).expect("2 converts to float");
    let half = T::from(0.5_f64).expect("0.5 converts to float");

    // AGM method for E(m):
    // E(m) = K(m) * (1 - Σ_{n=0}^∞ 2^{n-1} * c_n^2)
    // where a_0=1, b_0=sqrt(1-m), c_0=sqrt(m)
    // iteration: a_{n+1}=(a_n+b_n)/2, b_{n+1}=sqrt(a_n*b_n), c_{n+1}=(a_n-b_n)/2
    let mut a = T::one();
    let mut b = (T::one() - m).sqrt();
    let mut c = m.sqrt();
    let eps = T::from(1e-15_f64).expect("1e-15 converts to float");

    // First term: n=0, power = 2^{-1} = 0.5
    let mut power = half;
    let mut e_sum = power * c * c;

    for _ in 0..50 {
        let a_new = (a + b) * half;
        let b_new = (a * b).sqrt();
        let c_new = (a - b) * half;
        a = a_new;
        b = b_new;
        c = c_new;
        power = power * two; // next power is 2^n for n=1,2,...
        e_sum = e_sum + power * c * c;
        if c.abs() <= eps * a.abs() {
            break;
        }
    }

    // K = pi/(2*a) at convergence
    pi / (two * a) * (T::one() - e_sum)
}

/// Scalar incomplete elliptic integral of the first kind
fn ellipf_scalar<T>(phi: T, m: T) -> T
where
    T: Float + Debug,
{
    let zero = T::zero();
    let one = T::one();
    let eps = T::from(1e-15).expect("1e-15 should convert to float type");

    if phi.abs() < eps {
        return zero;
    }

    // For phi = pi/2, return complete elliptic integral
    let pi_half = T::from(std::f64::consts::PI / 2.0).expect("PI/2 should convert to float type");
    if (phi.abs() - pi_half).abs() < eps {
        return ellipk_scalar(m).copysign(phi);
    }

    // Use Carlson's RF function for numerical stability
    // F(phi, m) = sin(phi) * RF(cos^2(phi), 1 - m*sin^2(phi), 1)
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let sin2 = sin_phi * sin_phi;
    let cos2 = cos_phi * cos_phi;

    sin_phi.abs() * carlson_rf(cos2, one - m * sin2, one).copysign(phi)
}

/// Carlson's RF symmetric elliptic integral
fn carlson_rf<T>(x: T, y: T, z: T) -> T
where
    T: Float + Debug,
{
    let one = T::one();
    let three = T::from(3.0).expect("3.0 should convert to float type");
    let eps = T::from(1e-10).expect("1e-10 should convert to float type");

    let mut xn = x;
    let mut yn = y;
    let mut zn = z;

    for _ in 0..30 {
        let lambda = (xn * yn).sqrt() + (yn * zn).sqrt() + (zn * xn).sqrt();
        xn = (xn + lambda) / T::from(4.0).expect("4.0 should convert to float type");
        yn = (yn + lambda) / T::from(4.0).expect("4.0 should convert to float type");
        zn = (zn + lambda) / T::from(4.0).expect("4.0 should convert to float type");

        let mean = (xn + yn + zn) / three;
        let dx = (mean - xn) / mean;
        let dy = (mean - yn) / mean;
        let dz = (mean - zn) / mean;

        if dx.abs() < eps && dy.abs() < eps && dz.abs() < eps {
            break;
        }
    }

    one / (xn + yn + zn).sqrt() / three.sqrt()
}

/// Scalar incomplete elliptic integral of the second kind
fn ellipeinc_scalar<T>(phi: T, m: T) -> T
where
    T: Float + Debug,
{
    let zero = T::zero();
    let eps = T::from(1e-15).expect("1e-15 should convert to float type");

    if phi.abs() < eps {
        return zero;
    }

    // For phi = pi/2, return complete elliptic integral
    let pi_half = T::from(std::f64::consts::PI / 2.0).expect("PI/2 should convert to float type");
    if (phi.abs() - pi_half).abs() < eps {
        return ellipe_scalar(m).copysign(phi);
    }

    // Use numerical integration (Simpson's rule)
    let n = 100;
    let h = phi / T::from(n).expect("n should convert to float type");

    let mut sum = T::zero();
    for i in 0..=n {
        let t = T::from(i).expect("i should convert to float type") * h;
        let sin_t = t.sin();
        let integrand = (T::one() - m * sin_t * sin_t).sqrt();

        let weight = if i == 0 || i == n {
            T::one()
        } else if i % 2 == 1 {
            T::from(4.0).expect("4.0 should convert to float type")
        } else {
            T::from(2.0).expect("2.0 should convert to float type")
        };

        sum = sum + weight * integrand;
    }

    sum * h / T::from(3.0).expect("3.0 should convert to float type")
}

/// Scalar Jacobi elliptic functions using arithmetic-geometric mean
fn jacobi_elliptic_scalar<T>(u: T, m: T) -> (T, T, T)
where
    T: Float + Debug,
{
    let one = T::one();
    let eps = T::from(1e-15).expect("1e-15 should convert to float type");

    // Special cases
    if m.abs() < eps {
        // m = 0: sn = sin, cn = cos, dn = 1
        return (u.sin(), u.cos(), one);
    }
    if (m - one).abs() < eps {
        // m = 1: sn = tanh, cn = dn = sech
        let sech = one / u.cosh();
        return (u.tanh(), sech, sech);
    }

    // Use Landen transformation
    let sqrt_m = m.sqrt();
    let mut a = vec![one];
    let mut b = vec![(one - m).sqrt()];
    let mut c = vec![sqrt_m];

    // AGM iteration
    while c.last().expect("c should not be empty").abs() > eps {
        let an = *a.last().expect("a should not be empty");
        let bn = *b.last().expect("b should not be empty");
        a.push((an + bn) / T::from(2.0).expect("2.0 should convert to float type"));
        b.push((an * bn).sqrt());
        c.push((an - bn) / T::from(2.0).expect("2.0 should convert to float type"));
    }

    // Back substitution
    let n = a.len() - 1;
    let mut phi = T::from(2.0)
        .expect("2.0 should convert to float type")
        .powi(n as i32)
        * a[n]
        * u;

    for i in (0..n).rev() {
        phi = (phi + phi.sin().asin().copysign(phi))
            / T::from(2.0).expect("2.0 should convert to float type")
            - (c[i + 1] / a[i + 1] * phi.sin()).asin();
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (one - m * sn * sn).sqrt();

    (sn, cn, dn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ellipk() {
        let values = Array::from_vec(vec![0.0, 0.5, 0.9]);
        let result = ellipk(&values);

        // Known values
        assert_relative_eq!(
            result.to_vec()[0],
            std::f64::consts::PI / 2.0,
            epsilon = 1e-10
        );
        assert!(result.to_vec()[1] > std::f64::consts::PI / 2.0);
        assert!(result.to_vec()[2] > result.to_vec()[1]);
    }

    #[test]
    fn test_jacobi_elliptic_special_cases() {
        let u = Array::from_vec(vec![0.0, 1.0, 2.0]);

        // m = 0: sn(u, 0) = sin(u), cn(u, 0) = cos(u), dn(u, 0) = 1
        let (sn, cn, dn) = jacobi_elliptic(&u, 0.0).expect("jacobi_elliptic should succeed");
        assert_relative_eq!(sn.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cn.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dn.to_vec()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_incomplete_elliptic_integrals() {
        // F(0, m) = 0 for any m
        let phi = Array::from_vec(vec![0.0]);
        let m = Array::from_vec(vec![0.5]);
        let f = ellipf(&phi, &m).expect("ellipf should succeed");
        assert_relative_eq!(f.to_vec()[0], 0.0, epsilon = 1e-10);

        // E(0, m) = 0 for any m
        let e = ellipeinc(&phi, &m).expect("ellipeinc should succeed");
        assert_relative_eq!(e.to_vec()[0], 0.0, epsilon = 1e-10);
    }
}
