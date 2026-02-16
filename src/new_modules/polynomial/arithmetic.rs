//! Polynomial arithmetic operations
//!
//! This module provides arithmetic operations for polynomials including
//! addition, subtraction, multiplication, and division.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Sub};

use super::core::Polynomial;

// Arithmetic operations for polynomials
impl<T> Add for Polynomial<T>
where
    T: Clone + Zero + Add<Output = T> + PartialEq,
{
    type Output = Polynomial<T>;

    fn add(self, other: Polynomial<T>) -> Polynomial<T> {
        let self_degree = self.degree();
        let other_degree = other.degree();
        let max_degree = std::cmp::max(self_degree, other_degree);

        let mut result = vec![T::zero(); max_degree + 1];

        for i in 0..=self_degree {
            result[max_degree - self_degree + i] = self.coefficients[i].clone();
        }

        for i in 0..=other_degree {
            let idx = max_degree - other_degree + i;
            result[idx] = result[idx].clone() + other.coefficients[i].clone();
        }

        Polynomial::new(result)
    }
}

impl<T> Sub for Polynomial<T>
where
    T: Clone + Zero + Sub<Output = T> + PartialEq + std::ops::Neg<Output = T>,
{
    type Output = Polynomial<T>;

    fn sub(self, other: Polynomial<T>) -> Polynomial<T> {
        let self_degree = self.degree();
        let other_degree = other.degree();
        let max_degree = std::cmp::max(self_degree, other_degree);

        let mut result = vec![T::zero(); max_degree + 1];

        for i in 0..=self_degree {
            result[max_degree - self_degree + i] = self.coefficients[i].clone();
        }

        for i in 0..=other_degree {
            let idx = max_degree - other_degree + i;
            result[idx] = result[idx].clone() - other.coefficients[i].clone();
        }

        Polynomial::new(result)
    }
}

impl<T> Mul for Polynomial<T>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    type Output = Polynomial<T>;

    fn mul(self, other: Polynomial<T>) -> Polynomial<T> {
        let self_degree = self.degree();
        let other_degree = other.degree();
        let result_degree = self_degree + other_degree;

        let mut result = vec![T::zero(); result_degree + 1];

        for i in 0..=self_degree {
            for j in 0..=other_degree {
                let idx = i + j;
                let term = self.coefficients[i].clone() * other.coefficients[j].clone();
                result[idx] = result[idx].clone() + term;
            }
        }

        Polynomial::new(result)
    }
}

impl<T> Polynomial<T>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq,
{
    /// Divide this polynomial by another polynomial
    /// Returns (quotient, remainder) such that self = divisor * quotient + remainder
    pub fn divide(&self, divisor: &Self) -> Result<(Self, Self)> {
        if divisor.degree() == 0 && divisor.coefficients[0] == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "Division by zero polynomial".to_string(),
            ));
        }

        let mut dividend = self.clone();
        let mut quotient_coeffs = Vec::new();

        while dividend.degree() >= divisor.degree() && !dividend.coefficients.is_empty() {
            // Get the leading coefficient ratio
            let coeff = dividend.coefficients[0].clone() / divisor.coefficients[0].clone();
            quotient_coeffs.push(coeff.clone());

            // Subtract divisor * coeff from dividend
            let _deg_diff = dividend.degree() - divisor.degree();
            for i in 0..divisor.coefficients.len() {
                dividend.coefficients[i] = dividend.coefficients[i].clone()
                    - divisor.coefficients[i].clone() * coeff.clone();
            }

            // Remove the leading zero coefficient
            if !dividend.coefficients.is_empty() {
                dividend.coefficients.remove(0);
            }
            dividend = Polynomial::new(dividend.coefficients);
        }

        let quotient = if quotient_coeffs.is_empty() {
            Polynomial::zero()
        } else {
            Polynomial::new(quotient_coeffs)
        };

        Ok((quotient, dividend))
    }
}

/// Add two polynomials element-wise
///
/// Given polynomial coefficient arrays (highest degree first),
/// returns the coefficients of their sum.
///
/// # Parameters
///
/// * `p1` - First polynomial coefficients
/// * `p2` - Second polynomial coefficients
///
/// # Returns
///
/// Array of sum polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let p1 = Array::from_vec(vec![1.0, 2.0, 3.0]); // x^2 + 2x + 3
/// let p2 = Array::from_vec(vec![1.0, 1.0]);      // x + 1
/// let sum = polyadd(&p1, &p2)?;
/// // Result: x^2 + 3x + 4
/// # Ok(())
/// # }
/// ```
pub fn polyadd<T>(p1: &Array<T>, p2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + Add<Output = T> + PartialEq,
{
    if p1.ndim() != 1 || p2.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyadd requires 1D arrays".to_string(),
        ));
    }

    let poly1 = Polynomial::new(p1.to_vec());
    let poly2 = Polynomial::new(p2.to_vec());
    let result = poly1 + poly2;

    Ok(Array::from_vec(result.coefficients().to_vec()))
}

/// Subtract two polynomials element-wise
///
/// Given polynomial coefficient arrays (highest degree first),
/// returns the coefficients of their difference.
///
/// # Parameters
///
/// * `p1` - First polynomial coefficients
/// * `p2` - Second polynomial coefficients
///
/// # Returns
///
/// Array of difference polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let p1 = Array::from_vec(vec![1.0, 2.0, 3.0]); // x^2 + 2x + 3
/// let p2 = Array::from_vec(vec![1.0, 1.0]);      // x + 1
/// let diff = polysub(&p1, &p2)?;
/// // Result: x^2 + x + 2
/// # Ok(())
/// # }
/// ```
pub fn polysub<T>(p1: &Array<T>, p2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + Sub<Output = T> + PartialEq + std::ops::Neg<Output = T>,
{
    if p1.ndim() != 1 || p2.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polysub requires 1D arrays".to_string(),
        ));
    }

    let poly1 = Polynomial::new(p1.to_vec());
    let poly2 = Polynomial::new(p2.to_vec());
    let result = poly1 - poly2;

    Ok(Array::from_vec(result.coefficients().to_vec()))
}

/// Multiply two polynomials
///
/// Given polynomial coefficient arrays (highest degree first),
/// returns the coefficients of their product.
///
/// # Parameters
///
/// * `p1` - First polynomial coefficients
/// * `p2` - Second polynomial coefficients
///
/// # Returns
///
/// Array of product polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let p1 = Array::from_vec(vec![1.0, 1.0]);      // x + 1
/// let p2 = Array::from_vec(vec![1.0, -1.0]);     // x - 1
/// let prod = polymul(&p1, &p2)?;
/// // Result: x^2 - 1
/// # Ok(())
/// # }
/// ```
pub fn polymul<T>(p1: &Array<T>, p2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    if p1.ndim() != 1 || p2.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polymul requires 1D arrays".to_string(),
        ));
    }

    let poly1 = Polynomial::new(p1.to_vec());
    let poly2 = Polynomial::new(p2.to_vec());
    let result = poly1 * poly2;

    Ok(Array::from_vec(result.coefficients().to_vec()))
}

/// Polynomial division
///
/// Divides polynomial u by polynomial v, returning quotient q and remainder r
/// such that u = q*v + r and degree(r) < degree(v)
///
/// # Parameters
///
/// * `u` - Dividend polynomial
/// * `v` - Divisor polynomial
///
/// # Returns
///
/// Tuple of (quotient, remainder) polynomials
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let u = Array::from_vec(vec![1.0, -6.0, 11.0, -6.0]); // x^3 - 6x^2 + 11x - 6
/// let v = Array::from_vec(vec![1.0, -2.0]); // x - 2
/// let (q, r) = polydiv(&u, &v)?;
/// // q = x^2 - 4x + 3, r = 0 (since x-2 is a factor)
/// # Ok(())
/// # }
/// ```
pub fn polydiv<T>(u: &Array<T>, v: &Array<T>) -> Result<(Polynomial<T>, Polynomial<T>)>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq,
{
    if u.ndim() != 1 || v.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polydiv requires 1D arrays".to_string(),
        ));
    }

    let dividend = Polynomial::new(u.to_vec());
    let divisor = Polynomial::new(v.to_vec());

    if divisor.coefficients().is_empty() || divisor.coefficients()[0] == T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Division by zero polynomial".to_string(),
        ));
    }

    dividend.divide(&divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_addition() {
        // p(x) = 3x^2 + 2x + 1
        let p1 = Polynomial::new(vec![3.0, 2.0, 1.0]);
        // q(x) = 2x^3 + x^2 + 4
        let p2 = Polynomial::new(vec![2.0, 1.0, 0.0, 4.0]);

        // r(x) = 2x^3 + 4x^2 + 2x + 5
        let r = p1 + p2;
        assert_eq!(r.coefficients(), &[2.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_polynomial_multiplication() {
        // p(x) = x + 1
        let p1 = Polynomial::new(vec![1.0, 1.0]);
        // q(x) = x + 2
        let p2 = Polynomial::new(vec![1.0, 2.0]);

        // r(x) = x^2 + 3x + 2
        let r = p1 * p2;
        assert_eq!(r.coefficients(), &[1.0, 3.0, 2.0]);
    }
}
