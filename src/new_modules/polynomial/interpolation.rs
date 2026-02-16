//! Polynomial interpolation methods
//!
//! This module provides various polynomial interpolation techniques including
//! Lagrange interpolation, Newton's divided differences, and cubic splines.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use super::core::Polynomial;

/// Polynomial interpolation methods
pub struct PolynomialInterpolation;

impl PolynomialInterpolation {
    /// Interpolate a polynomial through points (x, y) using Lagrange interpolation
    pub fn lagrange<T>(x: &Array<T>, y: &Array<T>) -> Result<Polynomial<T>>
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
            + std::ops::Neg<Output = T>,
    {
        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Lagrange interpolation requires 1D arrays of points".to_string(),
            ));
        }

        if x_shape[0] != y_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x_shape,
                actual: y_shape,
            });
        }

        let n = x_shape[0];
        let x_data = x.to_vec();
        let y_data = y.to_vec();

        // Check for duplicate x values
        for i in 0..n {
            for j in (i + 1)..n {
                if x_data[i] == x_data[j] {
                    return Err(NumRs2Error::InvalidOperation(
                        "Lagrange interpolation requires unique x values".to_string(),
                    ));
                }
            }
        }

        // Start with zero polynomial
        let mut result = Polynomial::zero();

        for i in 0..n {
            // Compute the Lagrange basis polynomial for x_i
            let mut numerator = Polynomial::one();
            let mut denominator = T::one();

            for j in 0..n {
                if i != j {
                    // (x - x_j)
                    let neg_xj = T::zero() - x_data[j].clone();
                    let linear_term = Polynomial::new(vec![T::one(), neg_xj]);
                    numerator = numerator * linear_term;

                    // (x_i - x_j)
                    denominator = denominator * (x_data[i].clone() - x_data[j].clone());
                }
            }

            // Scale by y_i / denominator
            let scale = y_data[i].clone() / denominator;
            let mut term = numerator;
            term.coefficients = term
                .coefficients
                .iter()
                .map(|c| c.clone() * scale.clone())
                .collect();

            // Add to the result
            result = result + term;
        }

        Ok(result)
    }

    /// Interpolate a polynomial through points (x, y) using Newton's divided differences
    pub fn newton<T>(x: &Array<T>, y: &Array<T>) -> Result<Polynomial<T>>
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
            + std::ops::Neg<Output = T>,
    {
        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Newton interpolation requires 1D arrays of points".to_string(),
            ));
        }

        if x_shape[0] != y_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x_shape,
                actual: y_shape,
            });
        }

        let n = x_shape[0];
        let x_data = x.to_vec();
        let y_data = y.to_vec();

        // Compute divided differences table
        let mut divided_diff = vec![vec![T::zero(); n]; n];

        // First column is just y values
        for i in 0..n {
            divided_diff[i][0] = y_data[i].clone();
        }

        // Compute the table of divided differences
        for j in 1..n {
            for i in 0..(n - j) {
                divided_diff[i][j] = (divided_diff[i + 1][j - 1].clone()
                    - divided_diff[i][j - 1].clone())
                    / (x_data[i + j].clone() - x_data[i].clone());
            }
        }

        // Build the Newton form of the interpolating polynomial
        let mut result = Polynomial::new(vec![divided_diff[0][0].clone()]);
        let mut term: Polynomial<T> = Polynomial::one();

        for j in 1..n {
            // Multiply by (x - x_j-1)
            let neg_xj = T::zero() - x_data[j - 1].clone();
            let linear_term = Polynomial::new(vec![T::one(), neg_xj]);
            term = term * linear_term;

            // Add a_j * term
            let mut scaled_term = term.clone();
            scaled_term.coefficients = scaled_term
                .coefficients
                .iter()
                .map(|c| c.clone() * divided_diff[0][j].clone())
                .collect();

            result = result + scaled_term;
        }

        Ok(result)
    }
}

/// Spline interpolation for smoother curves
pub struct CubicSpline<T> {
    /// x coordinates of the knots
    knots: Vec<T>,
    /// Coefficients for each spline segment (for each segment: a, b, c, d where a*x^3 + b*x^2 + c*x + d)
    coefficients: Vec<[T; 4]>,
}

impl<T> CubicSpline<T>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Debug
        + Float,
{
    /// Create a new natural cubic spline from x and y data points
    pub fn new(x: &Array<T>, y: &Array<T>) -> Result<Self> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Cubic spline requires 1D arrays of points".to_string(),
            ));
        }

        if x_shape[0] != y_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x_shape,
                actual: y_shape,
            });
        }

        let n = x_shape[0];
        if n < 3 {
            return Err(NumRs2Error::InvalidOperation(
                "Cubic spline requires at least 3 points".to_string(),
            ));
        }

        let x_data = x.to_vec();
        let y_data = y.to_vec();

        // Check that x values are in ascending order
        for i in 1..n {
            if x_data[i] <= x_data[i - 1] {
                return Err(NumRs2Error::InvalidOperation(
                    "x values must be in strictly ascending order for cubic spline".to_string(),
                ));
            }
        }

        // Compute second derivatives using the tridiagonal algorithm
        let mut a = vec![T::zero(); n - 1];
        let mut b = vec![T::zero(); n];
        let mut c = vec![T::zero(); n - 1];
        let mut d = vec![T::zero(); n];

        // Set up the tridiagonal system
        for i in 1..n - 1 {
            let h_prev = x_data[i] - x_data[i - 1];
            let h_next = x_data[i + 1] - x_data[i];

            a[i - 1] = h_prev;
            b[i] = T::from(2.0).expect("2.0 should convert to float type") * (h_prev + h_next);
            c[i] = h_next;

            let dy_prev = y_data[i] - y_data[i - 1];
            let dy_next = y_data[i + 1] - y_data[i];

            d[i] = T::from(6.0).expect("6.0 should convert to float type")
                * (dy_next / h_next - dy_prev / h_prev);
        }

        // Natural boundary conditions: second derivatives at endpoints are zero
        b[0] = T::one();
        b[n - 1] = T::one();
        c[0] = T::zero();
        a[n - 2] = T::zero();
        d[0] = T::zero();
        d[n - 1] = T::zero();

        // Solve the tridiagonal system using Thomas algorithm
        // Forward elimination
        for i in 1..n {
            let m = a[i - 1] / b[i - 1];
            b[i] = b[i] - m * c[i - 1];
            d[i] = d[i] - m * d[i - 1];
        }

        // Back substitution
        let mut second_derivs = vec![T::zero(); n];
        second_derivs[n - 1] = d[n - 1] / b[n - 1];

        for i in (0..n - 1).rev() {
            second_derivs[i] = (d[i] - c[i] * second_derivs[i + 1]) / b[i];
        }

        // Compute the coefficients for each segment
        let mut coefficients = Vec::with_capacity(n - 1);

        for i in 0..n - 1 {
            let h = x_data[i + 1] - x_data[i];
            let a = (second_derivs[i + 1] - second_derivs[i])
                / (T::from(6.0).expect("6.0 should convert to float type") * h);
            let b = second_derivs[i] / T::from(2.0).expect("2.0 should convert to float type");
            let c = (y_data[i + 1] - y_data[i]) / h
                - (second_derivs[i + 1]
                    + T::from(2.0).expect("2.0 should convert to float type") * second_derivs[i])
                    * h
                    / T::from(6.0).expect("6.0 should convert to float type");
            let d = y_data[i];

            coefficients.push([a, b, c, d]);
        }

        Ok(CubicSpline {
            knots: x_data,
            coefficients,
        })
    }

    /// Evaluate the spline at a point x
    pub fn evaluate(&self, x: T) -> Result<T> {
        // Find the appropriate segment
        if x < self.knots[0] || x > self.knots[self.knots.len() - 1] {
            return Err(NumRs2Error::InvalidOperation(
                "Evaluation point outside the domain of the spline".to_string(),
            ));
        }

        // Binary search to find the segment
        let mut left = 0;
        let mut right = self.coefficients.len() - 1;

        while left <= right {
            let mid = (left + right) / 2;

            if x >= self.knots[mid] && x <= self.knots[mid + 1] {
                // Found the segment
                let t = x - self.knots[mid];
                let coeffs = &self.coefficients[mid];

                // Use Horner's method for polynomial evaluation
                let c0 = coeffs[0];
                let c1 = coeffs[1];
                let c2 = coeffs[2];
                let c3 = coeffs[3];

                return Ok(((c0 * t + c1) * t + c2) * t + c3);
            }

            if x < self.knots[mid] {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        // If we get here, we're in the last segment
        let last_idx = self.coefficients.len() - 1;
        let t = x - self.knots[last_idx];
        let coeffs = &self.coefficients[last_idx];

        // Uses Horner's method for polynomial evaluation without unnecessary clones
        let c0 = coeffs[0];
        let c1 = coeffs[1];
        let c2 = coeffs[2];
        let c3 = coeffs[3];

        Ok(((c0 * t + c1) * t + c2) * t + c3)
    }

    /// Evaluate the spline at multiple points
    pub fn evaluate_array(&self, x: &Array<T>) -> Result<Array<T>> {
        let x_data = x.to_vec();
        let mut result = Vec::with_capacity(x_data.len());

        for &x_val in &x_data {
            result.push(self.evaluate(x_val)?);
        }

        Ok(Array::from_vec(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lagrange_interpolation() {
        // Create points (0,1), (1,2), (2,4) - should fit y = x^2 + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 2.0, 5.0]);

        let p = PolynomialInterpolation::lagrange(&x, &y)
            .expect("Lagrange interpolation should succeed");

        // Check that p(x) = x^2 + 1
        assert_relative_eq!(p.coefficients()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(p.coefficients()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(p.coefficients()[2], 1.0, epsilon = 1e-10);

        // Check evaluations
        assert_relative_eq!(p.evaluate(0.0), 1.0);
        assert_relative_eq!(p.evaluate(1.0), 2.0);
        assert_relative_eq!(p.evaluate(2.0), 5.0);
        assert_relative_eq!(p.evaluate(3.0), 10.0);
    }

    #[test]
    fn test_newton_interpolation() {
        // Create points (0,1), (1,3), (2,7) - should fit y = x^2 + 2x + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 4.0, 9.0]);

        let p =
            PolynomialInterpolation::newton(&x, &y).expect("Newton interpolation should succeed");

        // Check evaluations
        assert_relative_eq!(p.evaluate(0.0), 1.0);
        assert_relative_eq!(p.evaluate(1.0), 4.0);
        assert_relative_eq!(p.evaluate(2.0), 9.0);
        assert_relative_eq!(p.evaluate(3.0), 16.0);
    }

    #[test]
    fn test_cubic_spline() {
        // Create points for y = x^2
        let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![0.0, 1.0, 4.0, 9.0]);

        let spline = CubicSpline::new(&x, &y).expect("Cubic spline creation should succeed");

        // Check interpolation at knots
        assert_relative_eq!(
            spline.evaluate(0.0).expect("eval at 0.0 should succeed"),
            0.0
        );
        assert_relative_eq!(
            spline.evaluate(1.0).expect("eval at 1.0 should succeed"),
            1.0
        );
        assert_relative_eq!(
            spline.evaluate(2.0).expect("eval at 2.0 should succeed"),
            4.0
        );
        assert_relative_eq!(
            spline.evaluate(3.0).expect("eval at 3.0 should succeed"),
            9.0
        );

        // Check that the spline approximates the function at intermediate points
        assert!(spline.evaluate(0.5).expect("eval at 0.5 should succeed") > 0.0);
        assert!(spline.evaluate(0.5).expect("eval at 0.5 should succeed") < 1.0);
        assert!(spline.evaluate(1.5).expect("eval at 1.5 should succeed") > 1.0);
        assert!(spline.evaluate(1.5).expect("eval at 1.5 should succeed") < 4.0);
    }
}
