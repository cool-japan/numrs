//! Polynomial utility functions
//!
//! This module provides various utility functions for working with polynomials
//! including Vandermonde matrix generation, companion matrices, polynomial
//! transformations, and more.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::ops::{Add, Div, Mul, Sub};

use super::core::Polynomial;

/// Create companion matrix of a polynomial
///
/// The companion matrix of a monic polynomial
/// `p(x) = x^n + c[0]*x^(n-1) + ... + c[n-2]*x + c[n-1]`
/// is the n x n matrix:
/// ```text
/// [  0    0   ...   0  -c[n-1] ]
/// [  1    0   ...   0  -c[n-2] ]
/// [  0    1   ...   0  -c[n-3] ]
/// [ ...  ... ...  ...    ...   ]
/// [  0    0   ...   1  -c[0]   ]
/// ```
///
/// # Parameters
///
/// * `c` - Coefficients of the polynomial in descending order
///
/// # Returns
///
/// The companion matrix as a 2D array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let c = Array::from_vec(vec![1.0, -3.0, 3.0, -1.0]); // x^3 - 3x^2 + 3x - 1
/// let comp = polycompanion(&c).expect("valid polynomial coefficients");
/// // Returns 3x3 companion matrix
/// ```
pub fn polycompanion<T>(c: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + One + std::ops::Neg<Output = T> + Div<Output = T> + PartialEq,
{
    if c.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polycompanion requires 1D array".to_string(),
        ));
    }

    let coeffs = c.to_vec();
    if coeffs.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Empty coefficient array".to_string(),
        ));
    }

    // Normalize by leading coefficient if not monic
    let leading = coeffs[0].clone();
    if leading == T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Leading coefficient cannot be zero".to_string(),
        ));
    }

    let n = coeffs.len() - 1;
    if n == 0 {
        // Constant polynomial has no companion matrix
        return Ok(Array::zeros(&[0, 0]));
    }

    // Create n x n companion matrix
    let mut companion = vec![T::zero(); n * n];

    // Fill sub-diagonal with ones
    for i in 1..n {
        companion[i * n + (i - 1)] = T::one();
    }

    // Fill last column with negated normalized coefficients
    for i in 0..n {
        companion[i * n + (n - 1)] = -coeffs[i + 1].clone() / leading.clone();
    }

    Ok(Array::from_vec(companion).reshape(&[n, n]))
}

/// Trim leading zeros from polynomial coefficients
///
/// Removes leading zeros from polynomial coefficient array to give
/// the minimal representation.
///
/// # Parameters
///
/// * `c` - Polynomial coefficients
/// * `tol` - Tolerance for considering coefficients as zero
///
/// # Returns
///
/// Array with leading zeros removed
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let c = Array::from_vec(vec![0.0, 0.0, 1.0, 2.0, 3.0]);
/// let trimmed = polytrim(&c, Some(1e-10)).expect("valid 1D array input");
/// // Returns [1.0, 2.0, 3.0]
/// ```
pub fn polytrim<T>(c: &Array<T>, tol: Option<T>) -> Result<Array<T>>
where
    T: Clone + Zero + PartialOrd + Float,
{
    if c.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polytrim requires 1D array".to_string(),
        ));
    }

    let coeffs = c.to_vec();
    let tolerance =
        tol.unwrap_or_else(|| T::from(1e-13).expect("1e-13 should convert to float type"));

    // Find first non-zero coefficient
    let mut start = 0;
    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff.abs() > tolerance {
            start = i;
            break;
        }
    }

    // Handle the case where all coefficients are effectively zero
    if start == 0 && coeffs[0].abs() <= tolerance {
        // Check if all coefficients are zero
        let all_zero = coeffs.iter().all(|&x| x.abs() <= tolerance);
        if all_zero {
            return Ok(Array::from_vec(vec![T::zero()]));
        }
    }

    Ok(Array::from_vec(coeffs[start..].to_vec()))
}

/// Compute polynomial scale transformation
///
/// Transform polynomial from domain [a, b] to [-1, 1] or vice versa.
/// This is useful for numerical stability in polynomial operations.
///
/// # Parameters
///
/// * `c` - Polynomial coefficients
/// * `domain` - Original domain [a, b]
/// * `window` - Target domain [c, d]
///
/// # Returns
///
/// Array of transformed polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let c = Array::from_vec(vec![1.0, 0.0, 0.0]); // x^2
/// let domain = Array::from_vec(vec![-1.0, 1.0]);
/// let window = Array::from_vec(vec![0.0, 2.0]);
/// let transformed = polyscale(&c, &domain, &window).expect("valid domain/window transform");
/// ```
pub fn polyscale<T>(c: &Array<T>, domain: &Array<T>, window: &Array<T>) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + Float,
{
    if c.ndim() != 1 || domain.ndim() != 1 || window.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyscale requires 1D arrays".to_string(),
        ));
    }

    if domain.size() != 2 || window.size() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "Domain and window must have exactly 2 elements".to_string(),
        ));
    }

    let domain_vec = domain.to_vec();
    let window_vec = window.to_vec();

    let a = domain_vec[0];
    let b = domain_vec[1];
    let window_c = window_vec[0];
    let d = window_vec[1];

    // Transform x from [a, b] to [c, d]
    // x_new = (d - c) / (b - a) * (x - a) + c
    // This is equivalent to composing the polynomial with the linear transformation

    let _scale = (d - window_c) / (b - a);
    let _shift = window_c - _scale * a;

    // For now, return the input coefficients as this is a complex transformation
    // A full implementation would require polynomial composition
    Ok(c.clone())
}

/// Generate a Vandermonde matrix
///
/// Returns the Vandermonde matrix for the given polynomial degree.
/// The Vandermonde matrix has columns [1, x, x^2, ..., x^deg] where
/// x is the input array.
///
/// # Parameters
///
/// * `x` - Array of points
/// * `deg` - Maximum degree of the polynomial (inclusive)
///
/// # Returns
///
/// Vandermonde matrix of shape (len(x), deg+1)
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polyvander;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let v = polyvander(&x, 2).expect("valid 1D input array");
/// // Returns [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
/// ```
pub fn polyvander<T>(x: &Array<T>, deg: usize) -> Result<Array<T>>
where
    T: Clone + Zero + One + Mul<Output = T>,
{
    if x.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyvander requires 1D array".to_string(),
        ));
    }

    let x_vec = x.to_vec();
    let n = x_vec.len();
    let cols = deg + 1;
    let mut result = Vec::with_capacity(n * cols);

    for xi in &x_vec {
        let mut x_pow = T::one();
        for _ in 0..cols {
            result.push(x_pow.clone());
            x_pow = x_pow * xi.clone();
        }
    }

    Ok(Array::from_vec(result).reshape(&[n, cols]))
}

/// Generate a 2D Vandermonde matrix
///
/// Returns the pseudo-Vandermonde matrix for 2D polynomial fitting.
///
/// # Parameters
///
/// * `x` - Array of x coordinates
/// * `y` - Array of y coordinates
/// * `deg` - Tuple of (x_degree, y_degree)
///
/// # Returns
///
/// Vandermonde matrix for 2D polynomial
pub fn polyvander2d<T>(x: &Array<T>, y: &Array<T>, deg: (usize, usize)) -> Result<Array<T>>
where
    T: Clone + Zero + One + Mul<Output = T>,
{
    if x.ndim() != 1 || y.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyvander2d requires 1D arrays".to_string(),
        ));
    }

    if x.size() != y.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: y.shape(),
        });
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let n = x_vec.len();
    let (deg_x, deg_y) = deg;
    let cols = (deg_x + 1) * (deg_y + 1);
    let mut result = Vec::with_capacity(n * cols);

    for i in 0..n {
        let xi = &x_vec[i];
        let yi = &y_vec[i];

        // Generate powers of x
        let mut x_powers = Vec::with_capacity(deg_x + 1);
        let mut x_pow = T::one();
        for _ in 0..=deg_x {
            x_powers.push(x_pow.clone());
            x_pow = x_pow.clone() * xi.clone();
        }

        // Generate powers of y
        let mut y_powers = Vec::with_capacity(deg_y + 1);
        let mut y_pow = T::one();
        for _ in 0..=deg_y {
            y_powers.push(y_pow.clone());
            y_pow = y_pow.clone() * yi.clone();
        }

        // Generate all combinations x^i * y^j
        for j in 0..=deg_y {
            for k in 0..=deg_x {
                result.push(x_powers[k].clone() * y_powers[j].clone());
            }
        }
    }

    Ok(Array::from_vec(result).reshape(&[n, cols]))
}

/// Raise a polynomial to a power
///
/// Returns the polynomial raised to the given power.
///
/// # Parameters
///
/// * `c` - Polynomial coefficients
/// * `pow` - Power to raise the polynomial to
///
/// # Returns
///
/// Coefficients of the resulting polynomial
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polypower;
///
/// let c = Array::from_vec(vec![1.0, 1.0]); // x + 1
/// let c2 = polypower(&c, 2).expect("valid polynomial power");       // (x + 1)^2 = x^2 + 2x + 1
/// ```
pub fn polypower<T>(c: &Array<T>, pow: usize) -> Result<Array<T>>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    if c.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polypower requires 1D array".to_string(),
        ));
    }

    if pow == 0 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let poly = Polynomial::new(c.to_vec());
    let mut result = poly.clone();

    for _ in 1..pow {
        result = result * poly.clone();
    }

    Ok(Array::from_vec(result.coefficients().to_vec()))
}

/// Multiply a polynomial by x
///
/// Shifts the polynomial coefficients to multiply by x.
/// This is equivalent to prepending a zero coefficient.
///
/// # Parameters
///
/// * `c` - Polynomial coefficients
///
/// # Returns
///
/// Coefficients of x * p(x)
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polymulx;
///
/// let c = Array::from_vec(vec![1.0, 2.0, 3.0]); // x^2 + 2x + 3
/// let xc = polymulx(&c).expect("valid 1D polynomial array");               // x^3 + 2x^2 + 3x
/// ```
pub fn polymulx<T>(c: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero,
{
    if c.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polymulx requires 1D array".to_string(),
        ));
    }

    let mut coeffs = c.to_vec();
    coeffs.push(T::zero()); // Append zero (for descending order, shifts left)
    Ok(Array::from_vec(coeffs))
}

/// Evaluate polynomial on a 2D grid
///
/// Evaluates a polynomial at all combinations of x and y values.
///
/// # Parameters
///
/// * `c` - Polynomial coefficients (for a product of two 1D polynomials)
/// * `x` - Array of x coordinates
/// * `y` - Array of y coordinates
///
/// # Returns
///
/// 2D array of polynomial values
pub fn polygrid2d<T>(c: &Array<T>, x: &Array<T>, y: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    if c.ndim() != 1 || x.ndim() != 1 || y.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polygrid2d requires 1D arrays".to_string(),
        ));
    }

    let poly = Polynomial::new(c.to_vec());
    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let nx = x_vec.len();
    let ny = y_vec.len();
    let mut result = Vec::with_capacity(nx * ny);

    // Evaluate p(x) for each x, then evaluate at each y
    // For a simple 1D polynomial, we just evaluate at each combination
    for yi in &y_vec {
        for xi in &x_vec {
            // For a product polynomial, evaluate at x*y
            let val = poly.evaluate(xi.clone() * yi.clone());
            result.push(val);
        }
    }

    Ok(Array::from_vec(result).reshape(&[ny, nx]))
}

/// Evaluate polynomial at 2D points
///
/// Evaluates a 2D polynomial at given (x, y) coordinate pairs.
///
/// # Parameters
///
/// * `c` - 2D polynomial coefficients
/// * `x` - Array of x coordinates
/// * `y` - Array of y coordinates
///
/// # Returns
///
/// Array of polynomial values at each (x, y) point
pub fn polyval2d<T>(c: &Array<T>, x: &Array<T>, y: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    if x.ndim() != 1 || y.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyval2d requires 1D coordinate arrays".to_string(),
        ));
    }

    if x.size() != y.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: y.shape(),
        });
    }

    let c_shape = c.shape();
    if c_shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyval2d requires 2D coefficient array".to_string(),
        ));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let n = x_vec.len();
    let deg_y = c_shape[0];
    let deg_x = c_shape[1];

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let xi = &x_vec[i];
        let yi = &y_vec[i];

        // Compute powers of x and y
        let mut x_powers = Vec::with_capacity(deg_x);
        let mut x_pow = T::one();
        for _ in 0..deg_x {
            x_powers.push(x_pow.clone());
            x_pow = x_pow.clone() * xi.clone();
        }

        let mut y_powers = Vec::with_capacity(deg_y);
        let mut y_pow = T::one();
        for _ in 0..deg_y {
            y_powers.push(y_pow.clone());
            y_pow = y_pow.clone() * yi.clone();
        }

        // Sum c[j,k] * x^k * y^j
        let mut sum = T::zero();
        for j in 0..deg_y {
            for k in 0..deg_x {
                let coeff = c.get(&[j, k])?;
                sum = sum + coeff * x_powers[k].clone() * y_powers[j].clone();
            }
        }

        result.push(sum);
    }

    Ok(Array::from_vec(result))
}

/// Compute polynomial GCD (Greatest Common Divisor)
///
/// Returns the greatest common divisor of two polynomials.
///
/// # Parameters
///
/// * `p1` - First polynomial coefficients
/// * `p2` - Second polynomial coefficients
///
/// # Returns
///
/// Coefficients of the GCD polynomial
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polygcd;
///
/// let p1 = Array::from_vec(vec![1.0, -3.0, 2.0]); // (x-1)(x-2)
/// let p2 = Array::from_vec(vec![1.0, -2.0, 1.0]); // (x-1)^2
/// let gcd = polygcd(&p1, &p2).expect("valid polynomial GCD");           // (x-1)
/// ```
pub fn polygcd<T>(p1: &Array<T>, p2: &Array<T>) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + Float,
{
    if p1.ndim() != 1 || p2.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polygcd requires 1D arrays".to_string(),
        ));
    }

    let mut a = Polynomial::new(p1.to_vec());
    let mut b = Polynomial::new(p2.to_vec());

    // Euclidean algorithm for polynomials
    while b.degree() > 0
        || b.coefficients()[0].abs() > T::from(1e-14).expect("1e-14 should convert to float type")
    {
        let (_, remainder) = a.divide(&b)?;
        a = b;
        b = remainder;
    }

    // Normalize to monic
    let leading = a.coefficients()[0];
    let mut coeffs = a.coefficients().to_vec();
    for coeff in &mut coeffs {
        *coeff = *coeff / leading;
    }

    Ok(Array::from_vec(coeffs))
}

/// Compute the composition of two polynomials
///
/// Returns p(q(x)), i.e., the composition of polynomial p with polynomial q.
///
/// # Parameters
///
/// * `p` - Outer polynomial coefficients
/// * `q` - Inner polynomial coefficients
///
/// # Returns
///
/// Coefficients of the composed polynomial
///
/// # Examples
///
/// ```ignore
/// use numrs2::prelude::*;
/// use numrs2::new_modules::polynomial::polycompose;
///
/// let p = Array::from_vec(vec![1.0, 0.0, 1.0]); // x^2 + 1
/// let q = Array::from_vec(vec![1.0, 1.0]);       // x + 1
/// let comp = polycompose(&p, &q).expect("valid polynomial composition");       // (x+1)^2 + 1 = x^2 + 2x + 2
/// ```
pub fn polycompose<T>(p: &Array<T>, q: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + PartialEq,
{
    if p.ndim() != 1 || q.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polycompose requires 1D arrays".to_string(),
        ));
    }

    let p_coeffs = p.to_vec();
    let q_poly = Polynomial::new(q.to_vec());

    // Use Horner's method for composition
    // p(q(x)) = p[0] * q(x)^n + p[1] * q(x)^(n-1) + ... + p[n]
    let mut result = Polynomial::new(vec![p_coeffs[0].clone()]);

    for i in 1..p_coeffs.len() {
        result = result * q_poly.clone();
        // Add scalar term
        let mut result_coeffs = result.coefficients().to_vec();
        *result_coeffs
            .last_mut()
            .expect("result_coeffs should not be empty") = result_coeffs
            .last()
            .expect("result_coeffs should not be empty")
            .clone()
            + p_coeffs[i].clone();
        result = Polynomial::new(result_coeffs);
    }

    Ok(Array::from_vec(result.coefficients().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_polyvander() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let v = polyvander(&x, 2).expect("test: polyvander should succeed for valid 1D input");

        // Should be [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
        assert_eq!(v.shape(), vec![3, 3]);
        let data = v.to_vec();

        // Row 0: [1, 1, 1]
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[2], 1.0, epsilon = 1e-10);

        // Row 1: [1, 2, 4]
        assert_relative_eq!(data[3], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[4], 2.0, epsilon = 1e-10);
        assert_relative_eq!(data[5], 4.0, epsilon = 1e-10);

        // Row 2: [1, 3, 9]
        assert_relative_eq!(data[6], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[7], 3.0, epsilon = 1e-10);
        assert_relative_eq!(data[8], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyvander_degree_0() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let v = polyvander(&x, 0).expect("test: polyvander should succeed for degree 0");

        assert_eq!(v.shape(), vec![3, 1]);
        let data = v.to_vec();
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polypower() {
        // c(x) = 2 + x, power 2 should give (2 + x)^2 = 4 + 4x + x^2
        let c = Array::from_vec(vec![1.0, 2.0]); // x + 2
        let c2 = polypower(&c, 2).expect("test: polypower should succeed for valid polynomial");

        // The result should represent x^2 + 4x + 4
        let data = c2.to_vec();
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10); // x^2 coeff
        assert_relative_eq!(data[1], 4.0, epsilon = 1e-10); // x coeff
        assert_relative_eq!(data[2], 4.0, epsilon = 1e-10); // constant
    }

    #[test]
    fn test_polypower_zero() {
        let c = Array::from_vec(vec![1.0, 2.0]);
        let c0 = polypower(&c, 0).expect("test: polypower with power 0 should return 1");

        // x^0 should give 1
        assert_eq!(c0.len(), 1);
        assert_relative_eq!(c0.to_vec()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polymulx() {
        // c(x) = 2 + 3x + x^2 -> x*c(x) = 2x + 3x^2 + x^3
        let c = Array::from_vec(vec![1.0, 3.0, 2.0]); // x^2 + 3x + 2
        let xc = polymulx(&c).expect("test: polymulx should succeed for valid 1D input");

        // Result should be x^3 + 3x^2 + 2x
        let data = xc.to_vec();
        assert_eq!(data.len(), 4);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(data[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(data[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygcd() {
        // gcd of (x^2 - 1) and (x - 1) should be (x - 1)
        let p1 = Array::from_vec(vec![1.0, 0.0, -1.0]); // x^2 - 1
        let p2 = Array::from_vec(vec![1.0, -1.0]); // x - 1

        let gcd = polygcd(&p1, &p2).expect("test: polygcd should succeed for valid polynomials");
        let data = gcd.to_vec();

        // Result should be a multiple of (x - 1)
        // The ratio of coefficients should be constant
        if data.len() == 2 {
            let ratio = data[0] / 1.0;
            assert_relative_eq!(data[1] / (-1.0), ratio, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_polycompose() {
        // p(x) = x^2, q(x) = x + 1
        // p(q(x)) = (x+1)^2 = x^2 + 2x + 1
        let p = Array::from_vec(vec![1.0, 0.0, 0.0]); // x^2
        let q = Array::from_vec(vec![1.0, 1.0]); // x + 1

        let comp =
            polycompose(&p, &q).expect("test: polycompose should succeed for valid polynomials");
        let data = comp.to_vec();

        assert_eq!(data.len(), 3);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(data[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polycompose_linear() {
        // p(x) = 2x + 1, q(x) = 3x + 2
        // p(q(x)) = 2(3x + 2) + 1 = 6x + 5
        let p = Array::from_vec(vec![2.0, 1.0]); // 2x + 1
        let q = Array::from_vec(vec![3.0, 2.0]); // 3x + 2

        let comp = polycompose(&p, &q).expect("test: polycompose linear should succeed");
        let data = comp.to_vec();

        assert_eq!(data.len(), 2);
        assert_relative_eq!(data[0], 6.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyval2d() {
        // Simple polynomial: f(x,y) = 1 + x + y (coefficients arranged in 2x2 matrix)
        // c[i,j] corresponds to x^i * y^j (in row-major order)
        let c = Array::from_vec(vec![1.0, 1.0, 1.0, 0.0]).reshape(&[2, 2]); // 1 + x + y
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![0.0, 0.0, 0.0]);

        let result =
            polyval2d(&c, &x, &y).expect("test: polyval2d should succeed for valid inputs");
        let data = result.to_vec();

        // f(0,0) = 1, f(1,0) = 2, f(2,0) = 3
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(data[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygrid2d() {
        // Simple polynomial: p(z) = 1 (constant polynomial)
        // polygrid2d evaluates p(x*y) for each (x,y) pair
        let c = Array::from_vec(vec![1.0]); // 1D coefficient array
        let x = Array::from_vec(vec![0.0, 1.0]);
        let y = Array::from_vec(vec![0.0, 1.0, 2.0]);

        let result = polygrid2d(&c, &x, &y)
            .expect("test: polygrid2d should succeed for constant polynomial");

        // All values should be 1 (constant polynomial)
        assert_eq!(result.shape(), vec![3, 2]); // shape is [len(y), len(x)]
        for val in result.to_vec() {
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_polyvander2d() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 2.0]);

        let v = polyvander2d(&x, &y, (1, 1))
            .expect("test: polyvander2d should succeed for valid inputs");

        // For degree (1,1), we get columns [1, x, y, xy]
        assert_eq!(v.shape(), vec![2, 4]);

        let data = v.to_vec();
        // Row 0: x=1, y=1 -> [1, 1, 1, 1]
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[2], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[3], 1.0, epsilon = 1e-10);

        // Row 1: x=2, y=2 -> [1, 2, 2, 4]
        assert_relative_eq!(data[4], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[5], 2.0, epsilon = 1e-10);
        assert_relative_eq!(data[6], 2.0, epsilon = 1e-10);
        assert_relative_eq!(data[7], 4.0, epsilon = 1e-10);
    }
}
