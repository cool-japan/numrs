//! Polynomial fitting and evaluation functions
//!
//! This module provides functions for fitting polynomials to data points
//! and evaluating polynomials at given points.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use super::core::Polynomial;

/// Fit a polynomial of specified degree to the data points
#[allow(clippy::needless_range_loop)]
pub fn polyfit<T>(x: &Array<T>, y: &Array<T>, degree: usize) -> Result<Polynomial<T>>
where
    T: Clone
        + Zero
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + Debug
        + std::ops::Neg<Output = T>
        + Float,
{
    let x_shape = x.shape();
    let y_shape = y.shape();

    if x_shape.len() != 1 || y_shape.len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyfit requires 1D arrays of points".to_string(),
        ));
    }

    if x_shape[0] != y_shape[0] {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x_shape,
            actual: y_shape,
        });
    }

    let n = x_shape[0];
    if n <= degree {
        return Err(NumRs2Error::InvalidOperation(format!(
            "polyfit: number of data points must be greater than degree (got {} points for degree {})",
            n, degree
        )));
    }

    let x_data = x.to_vec();
    let y_data = y.to_vec();

    // Create Vandermonde matrix
    let mut vandermonde = vec![vec![T::zero(); degree + 1]; n];

    for i in 0..n {
        let mut x_pow = T::one();
        for j in 0..=degree {
            vandermonde[i][degree - j] = x_pow;
            x_pow = x_pow * x_data[i];
        }
    }

    // Solve the linear system using normal equations: (V^T V)p = V^T y
    // Compute V^T * V (coefficient matrix)
    let mut coeff_matrix = vec![vec![T::zero(); degree + 1]; degree + 1];

    for i in 0..=degree {
        for j in 0..=degree {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + vandermonde[k][i] * vandermonde[k][j];
            }
            coeff_matrix[i][j] = sum;
        }
    }

    // Compute V^T * y (right-hand side)
    let mut rhs = vec![T::zero(); degree + 1];

    for i in 0..=degree {
        let mut sum = T::zero();
        for k in 0..n {
            sum = sum + vandermonde[k][i] * y_data[k];
        }
        rhs[i] = sum;
    }

    // Solve the system using Gaussian elimination
    // Forward elimination
    for i in 0..=degree {
        // Find pivot
        let mut max_row = i;
        let mut max_val = coeff_matrix[i][i].abs();

        for j in (i + 1)..=degree {
            let val = coeff_matrix[j][i].abs();
            if val > max_val {
                max_val = val;
                max_row = j;
            }
        }

        // Swap rows if necessary
        if max_row != i {
            coeff_matrix.swap(i, max_row);
            rhs.swap(i, max_row);
        }

        // Eliminate
        for j in (i + 1)..=degree {
            let factor = coeff_matrix[j][i] / coeff_matrix[i][i];
            rhs[j] = rhs[j] - factor * rhs[i];

            for k in i..=degree {
                coeff_matrix[j][k] = coeff_matrix[j][k] - factor * coeff_matrix[i][k];
            }
        }
    }

    // Back substitution
    let mut coefficients = vec![T::zero(); degree + 1];

    for i in (0..=degree).rev() {
        let mut sum = T::zero();
        for j in (i + 1)..=degree {
            sum = sum + coeff_matrix[i][j] * coefficients[j];
        }
        coefficients[i] = (rhs[i] - sum) / coeff_matrix[i][i];
    }

    Ok(Polynomial::new(coefficients))
}

/// Evaluate a polynomial at points
pub fn polyval<T>(p: &Polynomial<T>, x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    p.evaluate_array(x)
}

/// Return the derivative of a polynomial
///
/// Given polynomial coefficients in descending order of degree,
/// returns the coefficients of the polynomial derivative.
///
/// # Arguments
/// * `c` - Array of polynomial coefficients (highest degree first)
/// * `m` - Order of derivative (default is 1)
///
/// # Returns
/// * `Result<Array<T>>` - Array of derivative coefficients
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polyder;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// // p(x) = x^3 + 2x^2 + 3x + 4
/// let p = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// // p'(x) = 3x^2 + 4x + 3
/// let dp = polyder(&p, 1)?;
/// assert_eq!(dp.to_vec(), vec![3.0, 4.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub fn polyder<T>(c: &Array<T>, m: usize) -> Result<Array<T>>
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
    if c.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyder requires a 1D array of coefficients".to_string(),
        ));
    }

    let coeffs = c.to_vec();

    // Handle special cases
    if m == 0 {
        return Ok(c.clone());
    }

    if coeffs.is_empty() || (coeffs.len() == 1 && coeffs[0] == T::zero()) {
        return Ok(Array::from_vec(vec![T::zero()]));
    }

    // Create polynomial and compute derivative m times
    let mut poly = Polynomial::new(coeffs);

    for _ in 0..m {
        poly = poly.derivative();
        if poly.degree() == 0 && poly.coefficients()[0] == T::zero() {
            break;
        }
    }

    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

/// Return the integral of a polynomial
///
/// Given polynomial coefficients in descending order of degree,
/// returns the coefficients of the polynomial integral.
///
/// # Arguments
/// * `c` - Array of polynomial coefficients (highest degree first)
/// * `m` - Order of integration (default is 1)
/// * `k` - Integration constants. If None, all constants are 0.
///   If Some, must have length m (one for each integration)
///
/// # Returns
/// * `Result<Array<T>>` - Array of integral coefficients
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polyint;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// // p(x) = 3x^2 + 4x + 3
/// let p = Array::from_vec(vec![3.0, 4.0, 3.0]);
/// // integral p(x)dx = x^3 + 2x^2 + 3x + C (with C=0)
/// let ip = polyint(&p, 1, None)?;
/// assert_eq!(ip.to_vec(), vec![1.0, 2.0, 3.0, 0.0]);
/// # Ok(())
/// # }
/// ```
pub fn polyint<T>(c: &Array<T>, m: usize, k: Option<&[T]>) -> Result<Array<T>>
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
    if c.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyint requires a 1D array of coefficients".to_string(),
        ));
    }

    let coeffs = c.to_vec();

    // Handle special cases
    if m == 0 {
        return Ok(c.clone());
    }

    // Check integration constants
    let constants = if let Some(k_vals) = k {
        if k_vals.len() != m {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Number of integration constants ({}) must match order of integration ({})",
                k_vals.len(),
                m
            )));
        }
        k_vals.to_vec()
    } else {
        vec![T::zero(); m]
    };

    // Create polynomial and compute integral m times
    let mut poly = Polynomial::new(coeffs);

    for i in 0..m {
        poly = poly.integral();

        // Replace the constant of integration with the provided value
        if i < constants.len() {
            let mut new_coeffs = poly.coefficients().to_vec();
            if !new_coeffs.is_empty() {
                *new_coeffs.last_mut().expect("new_coeffs is not empty") = constants[i].clone();
            }
            poly = Polynomial::new(new_coeffs);
        }
    }

    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

/// Fit a polynomial using weighted least squares
///
/// Fits a polynomial of the specified degree using weighted least squares.
///
/// # Parameters
///
/// * `x` - Array of x coordinates
/// * `y` - Array of y coordinates
/// * `degree` - Degree of the polynomial
/// * `weights` - Optional weights for each point
///
/// # Returns
///
/// Polynomial fitted to the data
pub fn polyfit_weighted<T>(
    x: &Array<T>,
    y: &Array<T>,
    degree: usize,
    weights: Option<&Array<T>>,
) -> Result<Polynomial<T>>
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
    if weights.is_none() {
        return polyfit(x, y, degree);
    }

    let w = weights.expect("weights checked to be Some above");
    if x.shape() != w.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: w.shape(),
        });
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let w_vec = w.to_vec();
    let n = x_vec.len();

    if n <= degree {
        return Err(NumRs2Error::InvalidOperation(format!(
            "polyfit: number of data points must be greater than degree (got {} points for degree {})",
            n, degree
        )));
    }

    // Create weighted Vandermonde matrix
    let mut vandermonde = vec![vec![T::zero(); degree + 1]; n];

    for i in 0..n {
        let wi = w_vec[i].sqrt(); // Weight by sqrt for least squares
        let mut x_pow = T::one();
        for j in 0..=degree {
            vandermonde[i][degree - j] = x_pow * wi;
            x_pow = x_pow * x_vec[i];
        }
    }

    // Weight y values
    let mut weighted_y = Vec::with_capacity(n);
    for i in 0..n {
        weighted_y.push(y_vec[i] * w_vec[i].sqrt());
    }

    // Solve using normal equations (V^T V)p = V^T y
    let mut coeff_matrix = vec![vec![T::zero(); degree + 1]; degree + 1];

    for i in 0..=degree {
        for j in 0..=degree {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + vandermonde[k][i] * vandermonde[k][j];
            }
            coeff_matrix[i][j] = sum;
        }
    }

    let mut rhs = vec![T::zero(); degree + 1];
    for i in 0..=degree {
        let mut sum = T::zero();
        for k in 0..n {
            sum = sum + vandermonde[k][i] * weighted_y[k];
        }
        rhs[i] = sum;
    }

    // Gaussian elimination
    for i in 0..=degree {
        let mut max_row = i;
        let mut max_val = coeff_matrix[i][i].abs();

        for j in (i + 1)..=degree {
            let val = coeff_matrix[j][i].abs();
            if val > max_val {
                max_val = val;
                max_row = j;
            }
        }

        if max_row != i {
            coeff_matrix.swap(i, max_row);
            rhs.swap(i, max_row);
        }

        for j in (i + 1)..=degree {
            let factor = coeff_matrix[j][i] / coeff_matrix[i][i];
            rhs[j] = rhs[j] - factor * rhs[i];

            for k in i..=degree {
                coeff_matrix[j][k] = coeff_matrix[j][k] - factor * coeff_matrix[i][k];
            }
        }
    }

    // Back substitution
    let mut coefficients = vec![T::zero(); degree + 1];

    for i in (0..=degree).rev() {
        let mut sum = T::zero();
        for j in (i + 1)..=degree {
            sum = sum + coeff_matrix[i][j] * coefficients[j];
        }
        coefficients[i] = (rhs[i] - sum) / coeff_matrix[i][i];
    }

    Ok(Polynomial::new(coefficients))
}

/// Extrapolate polynomial to new points
///
/// Uses the polynomial fitted to the given data points to extrapolate
/// or interpolate at new points.
///
/// # Parameters
///
/// * `x` - Known x coordinates
/// * `y` - Known y coordinates
/// * `new_x` - New x coordinates for extrapolation
/// * `degree` - Degree of polynomial to fit
///
/// # Returns
///
/// Array of extrapolated y values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
/// let y = Array::from_vec(vec![1.0, 4.0, 9.0]); // y = x^2 + 1
/// let new_x = Array::from_vec(vec![3.0, 4.0]);
/// let result = polyextrap(&x, &y, &new_x, 2)?;
/// // Returns approximately [10.0, 17.0]
/// # Ok(())
/// # }
/// ```
pub fn polyextrap<T>(
    x: &Array<T>,
    y: &Array<T>,
    new_x: &Array<T>,
    degree: usize,
) -> Result<Array<T>>
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
    // Fit polynomial to the data
    let poly = polyfit(x, y, degree)?;

    // Evaluate at new points
    polyval(&poly, new_x)
}

/// Compute polynomial residual
///
/// Returns the residual (fitting error) from polynomial fitting.
///
/// # Parameters
///
/// * `c` - Polynomial coefficients
/// * `x` - Array of x coordinates
/// * `y` - Array of expected y values
///
/// # Returns
///
/// Sum of squared residuals
pub fn polyresidual<T>(c: &Array<T>, x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialEq,
{
    if c.ndim() != 1 || x.ndim() != 1 || y.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyresidual requires 1D arrays".to_string(),
        ));
    }

    if x.size() != y.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: y.shape(),
        });
    }

    let poly = Polynomial::new(c.to_vec());
    let y_fitted = poly.evaluate_array(x)?;

    let y_vec = y.to_vec();
    let fitted_vec = y_fitted.to_vec();

    let mut ssr = T::zero();
    for i in 0..y_vec.len() {
        let residual = y_vec[i].clone() - fitted_vec[i].clone();
        ssr = ssr + residual.clone() * residual;
    }

    Ok(ssr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_polyfit() {
        // Create points for y = 2x^2 + 3x + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array::from_vec(vec![1.0, 6.0, 15.0, 28.0, 45.0]);

        // Fit a quadratic polynomial
        let p = polyfit(&x, &y, 2).expect("Polynomial fitting should succeed");

        // Check coefficients
        assert_relative_eq!(p.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(p.coefficients()[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(p.coefficients()[2], 1.0, epsilon = 1e-10);

        // Check evaluations
        assert_relative_eq!(p.evaluate(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(p.evaluate(1.0), 6.0, epsilon = 1e-10);
        assert_relative_eq!(p.evaluate(2.0), 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyval() {
        // Create a polynomial p(x) = 2x^2 + 3x + 1
        let p = Polynomial::new(vec![2.0, 3.0, 1.0]);

        // Evaluate at multiple points
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = polyval(&p, &x).expect("Polynomial evaluation should succeed");

        // Check results
        assert_eq!(y.shape(), vec![3]);
        assert_relative_eq!(y.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(y.to_vec()[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(y.to_vec()[2], 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyresidual() {
        // Create points for y = 2x + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 3.0, 5.0]);

        // Perfect fit polynomial
        let c = Array::from_vec(vec![2.0, 1.0]); // 2x + 1

        let residual =
            polyresidual(&c, &x, &y).expect("Polynomial residual calculation should succeed");
        assert_relative_eq!(residual, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyresidual_nonzero() {
        // Create points that don't exactly fit a line
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 3.0, 4.0]); // Not exactly 2x + 1

        let c = Array::from_vec(vec![2.0, 1.0]); // 2x + 1

        let residual =
            polyresidual(&c, &x, &y).expect("Polynomial residual calculation should succeed");
        // Expected: (1-1)^2 + (3-3)^2 + (5-4)^2 = 1
        assert_relative_eq!(residual, 1.0, epsilon = 1e-10);
    }
}
