//! Core polynomial structure and basic operations
//!
//! This module provides the fundamental `Polynomial` type and its basic operations.

use crate::array::Array;
use crate::error::Result;
use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Sub};

/// Polynomial functionality and interpolation for NumRS2
/// Represents a polynomial with coefficients in descending order of degree
/// e.g., p(x) = c\[0\] * x^n + c\[1\] * x^(n-1) + ... + c\[n-1\] * x + c\[n\]
#[derive(Clone, Debug)]
pub struct Polynomial<T> {
    /// Coefficients of the polynomial in descending order
    pub(crate) coefficients: Vec<T>,
}

impl<T> Polynomial<T>
where
    T: Clone + Zero + PartialEq,
{
    /// Create a new polynomial from coefficients in descending order of degree
    pub fn new(coefficients: Vec<T>) -> Self {
        // Remove leading zeros
        let mut coefs = coefficients;
        while coefs.len() > 1 && coefs[0] == T::zero() {
            coefs.remove(0);
        }

        Polynomial {
            coefficients: coefs,
        }
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty()
            || (self.coefficients.len() == 1 && self.coefficients[0] == T::zero())
        {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Get the coefficients of the polynomial
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }

    /// Convert to an Array for compatibility with other NumRS functions
    pub fn to_array(&self) -> Array<T> {
        Array::from_vec(self.coefficients.clone())
    }
}

impl<T> Polynomial<T>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    /// Create a polynomial representing x^n
    pub fn monomial(degree: usize) -> Self {
        let mut coefficients = vec![T::zero(); degree + 1];
        coefficients[0] = T::one();
        Polynomial { coefficients }
    }

    /// Create the zero polynomial (p(x) = 0)
    pub fn zero() -> Self {
        Polynomial {
            coefficients: vec![T::zero()],
        }
    }

    /// Create the constant polynomial (p(x) = 1)
    pub fn one() -> Self {
        Polynomial {
            coefficients: vec![T::one()],
        }
    }

    /// Evaluate the polynomial at a point x
    pub fn evaluate(&self, x: T) -> T {
        if self.coefficients.is_empty() {
            return T::zero();
        }

        // Using Horner's method for efficient polynomial evaluation
        // For a polynomial a_0 * x^n + a_1 * x^(n-1) + ... + a_{n-1} * x + a_n
        // We compute ((a_0 * x + a_1) * x + a_2) * ... * x + a_n
        let mut result = self.coefficients[0].clone();

        for i in 1..self.coefficients.len() {
            result = result * x.clone() + self.coefficients[i].clone();
        }

        result
    }

    /// Evaluate the polynomial for an array of x values
    pub fn evaluate_array(&self, x: &Array<T>) -> Result<Array<T>> {
        let x_vec = x.to_vec();
        let mut result = Vec::with_capacity(x_vec.len());

        for x_val in &x_vec {
            result.push(self.evaluate(x_val.clone()));
        }

        Ok(Array::from_vec(result))
    }
}

impl<T> Polynomial<T>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + From<i32>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    /// Compute the derivative of the polynomial
    pub fn derivative(&self) -> Self {
        if self.degree() == 0 {
            return Polynomial::zero();
        }

        let n = self.coefficients.len();
        let mut derivative_coeffs = Vec::with_capacity(n - 1);

        for i in 0..(n - 1) {
            let degree = n - 1 - i;
            let coef = self.coefficients[i].clone();
            let term = coef * T::from(degree as i32);
            derivative_coeffs.push(term);
        }

        Polynomial::new(derivative_coeffs)
    }

    /// Compute the integral of the polynomial (with constant of integration = 0)
    pub fn integral(&self) -> Self {
        let n = self.coefficients.len();
        let mut integral_coeffs = Vec::with_capacity(n + 1);

        for i in 0..n {
            let degree = n - i;
            let coef = self.coefficients[i].clone();
            let term = coef / T::from(degree as i32);
            integral_coeffs.push(term);
        }

        // Add constant of integration (0)
        integral_coeffs.push(T::zero());

        Polynomial::new(integral_coeffs)
    }

    /// Compute a definite integral of the polynomial over an interval [a, b]
    pub fn definite_integral(&self, a: T, b: T) -> T {
        let integral = self.integral();
        integral.evaluate(b) - integral.evaluate(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_creation() {
        let p = Polynomial::new(vec![3.0, 2.0, 1.0]);
        assert_eq!(p.degree(), 2);
        assert_eq!(p.coefficients(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_polynomial_evaluation() {
        // p(x) = 3x^2 + 2x + 1
        let p = Polynomial::new(vec![3.0, 2.0, 1.0]);

        assert_relative_eq!(p.evaluate(0.0), 1.0);
        assert_relative_eq!(p.evaluate(1.0), 6.0);
        assert_relative_eq!(p.evaluate(2.0), 17.0);
    }

    #[test]
    fn test_polynomial_derivative() {
        // p(x) = 3x^3 + 2x^2 + x + 1
        let p = Polynomial::new(vec![3.0, 2.0, 1.0, 1.0]);

        // p'(x) = 9x^2 + 4x + 1
        let dp = p.derivative();

        assert_eq!(dp.degree(), 2);
        assert_eq!(dp.coefficients(), &[9.0, 4.0, 1.0]);

        // Check evaluation of derivative
        assert_relative_eq!(dp.evaluate(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(dp.evaluate(1.0), 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_integral() {
        // p(x) = 2x + 1
        let p = Polynomial::new(vec![2.0, 1.0]);

        // integral p(x)dx = x^2 + x + C (C = 0)
        let int_p = p.integral();

        assert_eq!(int_p.degree(), 2);
        assert_eq!(int_p.coefficients(), &[1.0, 1.0, 0.0]);

        // Check evaluation of integral
        assert_relative_eq!(int_p.evaluate(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(int_p.evaluate(1.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(int_p.evaluate(2.0), 6.0, epsilon = 1e-10);

        // Check definite integral
        assert_relative_eq!(p.definite_integral(0.0, 1.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(p.definite_integral(1.0, 2.0), 4.0, epsilon = 1e-10);
    }
}
