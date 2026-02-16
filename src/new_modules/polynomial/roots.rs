//! Polynomial root finding and construction from roots
//!
//! This module provides functions for finding polynomial roots
//! and constructing polynomials from given roots.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use scirs2_core::Complex;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use super::core::Polynomial;

/// Find the roots of a polynomial
pub fn roots<T>(p: &Polynomial<T>) -> Result<Array<Complex<T>>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + Debug
        + std::ops::Neg<Output = T>
        + Float,
{
    let mut coeffs = p.coefficients().to_vec();

    // Remove leading zeros
    while coeffs.len() > 1 && coeffs[0] == T::zero() {
        coeffs.remove(0);
    }

    let degree = coeffs.len() - 1;

    // Constant polynomial, no roots
    if degree == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    // Linear polynomial: ax + b = 0 => x = -b/a
    if degree == 1 {
        let root = -coeffs[1] / coeffs[0];
        return Ok(Array::from_vec(vec![Complex::new(root, T::zero())]));
    }

    // Quadratic polynomial: ax^2 + bx + c = 0
    if degree == 2 {
        let a = coeffs[0];
        let b = coeffs[1];
        let c = coeffs[2];
        let discriminant = b * b - T::from(4.0).expect("4.0 should convert to float type") * a * c;

        if discriminant >= T::zero() {
            // Real roots
            let sqrt_d = discriminant.sqrt();
            let root1 =
                (-b + sqrt_d) / (T::from(2.0).expect("2.0 should convert to float type") * a);
            let root2 =
                (-b - sqrt_d) / (T::from(2.0).expect("2.0 should convert to float type") * a);
            return Ok(Array::from_vec(vec![
                Complex::new(root1, T::zero()),
                Complex::new(root2, T::zero()),
            ]));
        } else {
            // Complex roots
            let real_part = -b / (T::from(2.0).expect("2.0 should convert to float type") * a);
            let imag_part = (-discriminant).sqrt()
                / (T::from(2.0).expect("2.0 should convert to float type") * a);
            return Ok(Array::from_vec(vec![
                Complex::new(real_part, imag_part),
                Complex::new(real_part, -imag_part),
            ]));
        }
    }

    // For higher degree polynomials, use companion matrix eigenvalues
    // Normalize the polynomial
    let leading = coeffs[0];
    for coeff in &mut coeffs {
        *coeff = *coeff / leading;
    }

    // In practice, users should use the eigenvalues module directly:
    Err(NumRs2Error::InvalidOperation(
        "For polynomial root-finding for degree > 2, please use the eigenvalues module to compute the eigenvalues of the companion matrix".to_string()
    ))
}

/// Find polynomial with given roots
///
/// Given an array of roots, returns the polynomial whose roots are the given values.
/// For example, if roots = [r1, r2, r3], returns the polynomial:
/// (x - r1) * (x - r2) * (x - r3)
///
/// # Parameters
///
/// * `roots` - Array of roots
///
/// # Returns
///
/// A polynomial whose roots are the given values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let roots = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let p = poly(&roots)?;
/// // Returns polynomial (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
/// # Ok(())
/// # }
/// ```
pub fn poly<T>(roots: &Array<T>) -> Result<Polynomial<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    if roots.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "poly requires 1D array of roots".to_string(),
        ));
    }

    let roots_vec = roots.to_vec();
    if roots_vec.is_empty() {
        return Ok(Polynomial::new(vec![T::one()]));
    }

    // Start with polynomial p(x) = x - roots[0]
    let mut result = Polynomial::new(vec![T::one(), -roots_vec[0].clone()]);

    // Multiply by (x - roots[i]) for each subsequent root
    for i in 1..roots_vec.len() {
        let factor = Polynomial::new(vec![T::one(), -roots_vec[i].clone()]);
        result = result * factor;
    }

    Ok(result)
}

/// Return a polynomial whose roots are the given values
///
/// This is the inverse operation to finding polynomial roots.
/// Given values r1, r2, ..., rn, returns the polynomial
/// (x - r1) * (x - r2) * ... * (x - rn)
///
/// # Parameters
///
/// * `roots` - Array of polynomial roots
///
/// # Returns
///
/// Array of polynomial coefficients (highest degree first)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let roots = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let coeffs = polyfromroots(&roots)?;
/// // Returns coefficients of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
/// # Ok(())
/// # }
/// ```
pub fn polyfromroots<T>(roots: &Array<T>) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    let result = poly(roots)?;
    Ok(Array::from_vec(result.coefficients().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_roots_linear() {
        // p(x) = 2x - 4 has root x = 2
        let p = Polynomial::new(vec![2.0, -4.0]);
        let r = roots(&p).expect("Linear polynomial root finding should succeed");
        assert_eq!(r.len(), 1);
        assert_relative_eq!(r.to_vec()[0].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(r.to_vec()[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roots_quadratic_real() {
        // p(x) = x^2 - 3x + 2 = (x-1)(x-2) has roots x = 1, 2
        let p = Polynomial::new(vec![1.0, -3.0, 2.0]);
        let r = roots(&p).expect("Quadratic real roots finding should succeed");
        assert_eq!(r.len(), 2);

        let roots_vec = r.to_vec();
        // Sort roots by real part for consistent testing
        let (r1, r2) = if roots_vec[0].re < roots_vec[1].re {
            (roots_vec[0], roots_vec[1])
        } else {
            (roots_vec[1], roots_vec[0])
        };

        assert_relative_eq!(r1.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(r2.re, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roots_quadratic_complex() {
        // p(x) = x^2 + 1 has roots x = +/- i
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let r = roots(&p).expect("Quadratic complex roots finding should succeed");
        assert_eq!(r.len(), 2);

        let roots_vec = r.to_vec();
        assert_relative_eq!(roots_vec[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(roots_vec[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(roots_vec[0].im.abs(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(roots_vec[1].im.abs(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poly_from_roots() {
        let roots = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let p = poly(&roots).expect("Polynomial from roots construction should succeed");

        // Should be x^3 - 6x^2 + 11x - 6
        let coeffs = p.coefficients();
        assert_eq!(coeffs.len(), 4);
        assert_relative_eq!(coeffs[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[1], -6.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[2], 11.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[3], -6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyfromroots() {
        let roots = Array::from_vec(vec![1.0, -1.0]);
        let coeffs =
            polyfromroots(&roots).expect("Polynomial coefficients from roots should succeed");

        // (x-1)(x+1) = x^2 - 1
        let coeffs_vec = coeffs.to_vec();
        assert_eq!(coeffs_vec.len(), 3);
        assert_relative_eq!(coeffs_vec[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs_vec[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs_vec[2], -1.0, epsilon = 1e-10);
    }
}
