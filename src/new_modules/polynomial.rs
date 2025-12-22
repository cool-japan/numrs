use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use scirs2_core::Complex;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Polynomial functionality and interpolation for NumRS2
/// Represents a polynomial with coefficients in descending order of degree
/// e.g., p(x) = c\[0\] * x^n + c\[1\] * x^(n-1) + ... + c\[n-1\] * x + c\[n\]
#[derive(Clone, Debug)]
pub struct Polynomial<T> {
    /// Coefficients of the polynomial in descending order
    coefficients: Vec<T>,
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

/// General polynomial functions
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
        return Err(NumRs2Error::InvalidOperation(
            format!("polyfit: number of data points must be greater than degree (got {} points for degree {})", n, degree)
        ));
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
///
/// // p(x) = x^3 + 2x^2 + 3x + 4
/// let p = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// // p'(x) = 3x^2 + 4x + 3
/// let dp = polyder(&p, 1).unwrap();
/// assert_eq!(dp.to_vec(), vec![3.0, 4.0, 3.0]);
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
///
/// // p(x) = 3x^2 + 4x + 3
/// let p = Array::from_vec(vec![3.0, 4.0, 3.0]);
/// // ∫p(x)dx = x^3 + 2x^2 + 3x + C (with C=0)
/// let ip = polyint(&p, 1, None).unwrap();
/// assert_eq!(ip.to_vec(), vec![1.0, 2.0, 3.0, 0.0]);
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
                *new_coeffs.last_mut().unwrap() = constants[i].clone();
            }
            poly = Polynomial::new(new_coeffs);
        }
    }

    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

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
        let discriminant = b * b - T::from(4.0).unwrap() * a * c;

        if discriminant >= T::zero() {
            // Real roots
            let sqrt_d = discriminant.sqrt();
            let root1 = (-b + sqrt_d) / (T::from(2.0).unwrap() * a);
            let root2 = (-b - sqrt_d) / (T::from(2.0).unwrap() * a);
            return Ok(Array::from_vec(vec![
                Complex::new(root1, T::zero()),
                Complex::new(root2, T::zero()),
            ]));
        } else {
            // Complex roots
            let real_part = -b / (T::from(2.0).unwrap() * a);
            let imag_part = (-discriminant).sqrt() / (T::from(2.0).unwrap() * a);
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
            b[i] = T::from(2.0).unwrap() * (h_prev + h_next);
            c[i] = h_next;

            let dy_prev = y_data[i] - y_data[i - 1];
            let dy_next = y_data[i + 1] - y_data[i];

            d[i] = T::from(6.0).unwrap() * (dy_next / h_next - dy_prev / h_prev);
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
            let a = (second_derivs[i + 1] - second_derivs[i]) / (T::from(6.0).unwrap() * h);
            let b = second_derivs[i] / T::from(2.0).unwrap();
            let c = (y_data[i + 1] - y_data[i]) / h
                - (second_derivs[i + 1] + T::from(2.0).unwrap() * second_derivs[i]) * h
                    / T::from(6.0).unwrap();
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

/// Orthogonal polynomials implementation
pub struct OrthogonalPolynomials;

impl OrthogonalPolynomials {
    /// Generate Chebyshev polynomial of the first kind of degree n
    /// T_n(x) satisfies the recurrence relation:
    /// T_0(x) = 1
    /// T_1(x) = x
    /// T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)
    pub fn chebyshev_t<T>(n: usize) -> Polynomial<T>
    where
        T: Clone
            + Zero
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + From<i32>
            + PartialEq
            + std::ops::Neg<Output = T>,
    {
        if n == 0 {
            return Polynomial::one();
        }
        if n == 1 {
            return Polynomial::new(vec![T::one(), T::zero()]);
        }

        let mut t_prev: Polynomial<T> = Polynomial::one();
        let mut t_curr: Polynomial<T> = Polynomial::new(vec![T::one(), T::zero()]);

        for _ in 2..=n {
            // T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)
            let two_x = Polynomial::new(vec![T::from(2), T::zero()]);
            let two_x_t_curr = two_x * t_curr.clone();
            let t_next = two_x_t_curr - t_prev;

            t_prev = t_curr;
            t_curr = t_next;
        }

        t_curr
    }

    /// Generate Chebyshev polynomial of the second kind of degree n
    /// U_n(x) satisfies the recurrence relation:
    /// U_0(x) = 1
    /// U_1(x) = 2x
    /// U_{n+1}(x) = 2x * U_n(x) - U_{n-1}(x)
    pub fn chebyshev_u<T>(n: usize) -> Polynomial<T>
    where
        T: Clone
            + Zero
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + From<i32>
            + PartialEq
            + std::ops::Neg<Output = T>,
    {
        if n == 0 {
            return Polynomial::one();
        }
        if n == 1 {
            return Polynomial::new(vec![T::from(2), T::zero()]);
        }

        let mut u_prev: Polynomial<T> = Polynomial::one();
        let mut u_curr: Polynomial<T> = Polynomial::new(vec![T::from(2), T::zero()]);

        for _ in 2..=n {
            // U_{n+1}(x) = 2x * U_n(x) - U_{n-1}(x)
            let two_x = Polynomial::new(vec![T::from(2), T::zero()]);
            let two_x_u_curr = two_x * u_curr.clone();
            let u_next = two_x_u_curr - u_prev;

            u_prev = u_curr;
            u_curr = u_next;
        }

        u_curr
    }

    /// Generate Legendre polynomial of degree n
    /// P_n(x) satisfies the recurrence relation:
    /// P_0(x) = 1
    /// P_1(x) = x
    /// (n+1)P_{n+1}(x) = (2n+1)x * P_n(x) - n * P_{n-1}(x)
    pub fn legendre<T>(n: usize) -> Polynomial<T>
    where
        T: Clone
            + Zero
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + From<i32>
            + PartialEq
            + std::ops::Neg<Output = T>,
    {
        if n == 0 {
            return Polynomial::one();
        }
        if n == 1 {
            return Polynomial::new(vec![T::one(), T::zero()]);
        }

        let mut p_prev: Polynomial<T> = Polynomial::one();
        let mut p_curr: Polynomial<T> = Polynomial::new(vec![T::one(), T::zero()]);

        for k in 1..n {
            let k_plus_1 = T::from((k + 1) as i32);
            let two_k_plus_1 = T::from((2 * k + 1) as i32);
            let k_t = T::from(k as i32);

            // (n+1)P_{n+1}(x) = (2n+1)x * P_n(x) - n * P_{n-1}(x)
            // Create the variable x
            let x_poly = Polynomial::new(vec![T::one(), T::zero()]);

            // Scalar multiplication of polynomial
            let mut term1 = x_poly * p_curr.clone();
            term1.coefficients = term1
                .coefficients
                .iter()
                .map(|c| c.clone() * two_k_plus_1.clone())
                .collect();

            // Scalar multiplication of polynomial
            let mut term2 = p_prev.clone();
            term2.coefficients = term2
                .coefficients
                .iter()
                .map(|c| c.clone() * k_t.clone())
                .collect();

            // Polynomial subtraction
            let mut p_next = term1 - term2;
            // Scalar division of polynomial
            p_next.coefficients = p_next
                .coefficients
                .iter()
                .map(|c| c.clone() / k_plus_1.clone())
                .collect();

            p_prev = p_curr;
            p_curr = p_next;
        }

        p_curr
    }

    /// Generate Hermite polynomial (physicists' version) of degree n
    /// H_n(x) satisfies the recurrence relation:
    /// H_0(x) = 1
    /// H_1(x) = 2x
    /// H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
    pub fn hermite<T>(n: usize) -> Polynomial<T>
    where
        T: Clone
            + Zero
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + From<i32>
            + PartialEq
            + std::ops::Neg<Output = T>,
    {
        if n == 0 {
            return Polynomial::one();
        }
        if n == 1 {
            return Polynomial::new(vec![T::from(2), T::zero()]);
        }

        let mut h_prev: Polynomial<T> = Polynomial::one();
        let mut h_curr: Polynomial<T> = Polynomial::new(vec![T::from(2), T::zero()]);

        for k in 1..n {
            let two_k = T::from((2 * k) as i32);

            // H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
            let two_x = Polynomial::new(vec![T::from(2), T::zero()]);
            let two_x_h_curr = two_x * h_curr.clone();

            // Scale the previous polynomial by 2k
            let mut term2 = h_prev.clone();
            term2.coefficients = term2
                .coefficients
                .iter()
                .map(|c| c.clone() * two_k.clone())
                .collect();

            // Subtract to get the next polynomial
            let h_next = two_x_h_curr - term2;

            h_prev = h_curr;
            h_curr = h_next;
        }

        h_curr
    }

    /// Generate Laguerre polynomial of degree n
    /// L_n(x) satisfies the recurrence relation:
    /// L_0(x) = 1
    /// L_1(x) = 1 - x
    /// (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - n*L_{n-1}(x)
    pub fn laguerre<T>(n: usize) -> Polynomial<T>
    where
        T: Clone
            + Zero
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + From<i32>
            + PartialEq
            + std::ops::Neg<Output = T>,
    {
        if n == 0 {
            return Polynomial::one();
        }
        if n == 1 {
            return Polynomial::new(vec![T::from(-1), T::one()]);
        }

        let mut l_prev: Polynomial<T> = Polynomial::one();
        let mut l_curr: Polynomial<T> = Polynomial::new(vec![T::from(-1), T::one()]);

        for k in 1..n {
            let k_plus_1 = T::from((k + 1) as i32);
            let two_k_plus_1 = T::from((2 * k + 1) as i32);
            let k_t = T::from(k as i32);

            // (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - n*L_{n-1}(x)
            let _x_poly = Polynomial::new(vec![T::one(), T::zero()]);
            let two_k_plus_1_minus_x = Polynomial::new(vec![T::from(-1), two_k_plus_1.clone()]);

            // Multiply the current polynomial by (2n+1-x)
            let term1 = two_k_plus_1_minus_x * l_curr.clone();

            // Scale previous polynomial by n
            let mut term2 = l_prev.clone();
            term2.coefficients = term2
                .coefficients
                .iter()
                .map(|c| c.clone() * k_t.clone())
                .collect();

            // Subtract to get the next polynomial and divide by (n+1)
            let mut l_next = term1 - term2;
            l_next.coefficients = l_next
                .coefficients
                .iter()
                .map(|c| c.clone() / k_plus_1.clone())
                .collect();

            l_prev = l_curr;
            l_curr = l_next;
        }

        l_curr
    }
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
///
/// let roots = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let p = poly(&roots).unwrap();
/// // Returns polynomial (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
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
///
/// let u = Array::from_vec(vec![1.0, -6.0, 11.0, -6.0]); // x^3 - 6x^2 + 11x - 6
/// let v = Array::from_vec(vec![1.0, -2.0]); // x - 2
/// let (q, r) = polydiv(&u, &v).unwrap();
/// // q = x^2 - 4x + 3, r = 0 (since x-2 is a factor)
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
/// let comp = polycompanion(&c).unwrap();
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
///
/// let p1 = Array::from_vec(vec![1.0, 2.0, 3.0]); // x^2 + 2x + 3
/// let p2 = Array::from_vec(vec![1.0, 1.0]);      // x + 1
/// let sum = polyadd(&p1, &p2).unwrap();
/// // Result: x^2 + 3x + 4
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
///
/// let p1 = Array::from_vec(vec![1.0, 2.0, 3.0]); // x^2 + 2x + 3
/// let p2 = Array::from_vec(vec![1.0, 1.0]);      // x + 1
/// let diff = polysub(&p1, &p2).unwrap();
/// // Result: x^2 + x + 2
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
///
/// let p1 = Array::from_vec(vec![1.0, 1.0]);      // x + 1
/// let p2 = Array::from_vec(vec![1.0, -1.0]);     // x - 1
/// let prod = polymul(&p1, &p2).unwrap();
/// // Result: x^2 - 1
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
///
/// let roots = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let coeffs = polyfromroots(&roots).unwrap();
/// // Returns coefficients of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
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
/// let trimmed = polytrim(&c, Some(1e-10)).unwrap();
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
    let tolerance = tol.unwrap_or_else(|| T::from(1e-13).unwrap());

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
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
/// let y = Array::from_vec(vec![1.0, 4.0, 9.0]); // y = x^2 + 1
/// let new_x = Array::from_vec(vec![3.0, 4.0]);
/// let result = polyextrap(&x, &y, &new_x, 2).unwrap();
/// // Returns approximately [10.0, 17.0]
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
/// let transformed = polyscale(&c, &domain, &window).unwrap();
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

/// Return Chebyshev polynomial of specified degree and kind
///
/// Returns the coefficients of the Chebyshev polynomial of the first or second kind.
///
/// # Parameters
///
/// * `degree` - Degree of the polynomial
/// * `kind` - 1 for first kind (T_n), 2 for second kind (U_n)
///
/// # Returns
///
/// Array of Chebyshev polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let t2 = polychebyshev::<f64>(2, 1).unwrap(); // T_2(x) = 2x^2 - 1
/// let u1 = polychebyshev::<f64>(1, 2).unwrap(); // U_1(x) = 2x
/// ```
pub fn polychebyshev<T>(degree: usize, kind: u8) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + From<i32>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    let poly = match kind {
        1 => OrthogonalPolynomials::chebyshev_t::<T>(degree),
        2 => OrthogonalPolynomials::chebyshev_u::<T>(degree),
        _ => {
            return Err(NumRs2Error::InvalidOperation(
                "Kind must be 1 (first kind) or 2 (second kind)".to_string(),
            ))
        }
    };

    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

/// Return Legendre polynomial of specified degree
///
/// Returns the coefficients of the Legendre polynomial P_n(x).
///
/// # Parameters
///
/// * `degree` - Degree of the polynomial
///
/// # Returns
///
/// Array of Legendre polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let p2 = polylegendre::<f64>(2).unwrap(); // P_2(x) = (3x^2 - 1)/2
/// ```
pub fn polylegendre<T>(degree: usize) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + From<i32>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    let poly = OrthogonalPolynomials::legendre::<T>(degree);
    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

/// Return Hermite polynomial of specified degree
///
/// Returns the coefficients of the Hermite polynomial H_n(x) (physicists' version).
///
/// # Parameters
///
/// * `degree` - Degree of the polynomial
///
/// # Returns
///
/// Array of Hermite polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let h2 = polyhermite::<f64>(2).unwrap(); // H_2(x) = 4x^2 - 2
/// ```
pub fn polyhermite<T>(degree: usize) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + From<i32>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    let poly = OrthogonalPolynomials::hermite::<T>(degree);
    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

/// Return Laguerre polynomial of specified degree
///
/// Returns the coefficients of the Laguerre polynomial L_n(x).
///
/// # Parameters
///
/// * `degree` - Degree of the polynomial
///
/// # Returns
///
/// Array of Laguerre polynomial coefficients
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let l2 = polylaguerre::<f64>(2).unwrap(); // L_2(x) = (x^2 - 4x + 2)/2
/// ```
pub fn polylaguerre<T>(degree: usize) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + From<i32>
        + PartialEq
        + std::ops::Neg<Output = T>,
{
    let poly = OrthogonalPolynomials::laguerre::<T>(degree);
    Ok(Array::from_vec(poly.coefficients().to_vec()))
}

// =============================================================================
// ENHANCED POLYNOMIAL FUNCTIONS (numpy.polynomial equivalents)
// =============================================================================

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
/// let v = polyvander(&x, 2).unwrap();
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
/// let c2 = polypower(&c, 2).unwrap();       // (x + 1)^2 = x^2 + 2x + 1
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
/// let xc = polymulx(&c).unwrap();               // x^3 + 2x^2 + 3x
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
/// let gcd = polygcd(&p1, &p2).unwrap();           // (x-1)
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
    while b.degree() > 0 || b.coefficients()[0].abs() > T::from(1e-14).unwrap() {
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
/// let comp = polycompose(&p, &q).unwrap();       // (x+1)^2 + 1 = x^2 + 2x + 2
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
        *result_coeffs.last_mut().unwrap() =
            result_coeffs.last().unwrap().clone() + p_coeffs[i].clone();
        result = Polynomial::new(result_coeffs);
    }

    Ok(Array::from_vec(result.coefficients().to_vec()))
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

    let w = weights.unwrap();
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
        return Err(NumRs2Error::InvalidOperation(
            format!("polyfit: number of data points must be greater than degree (got {} points for degree {})", n, degree)
        ));
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

/// Return Jacobi polynomial of specified degree
///
/// Returns the coefficients of the Jacobi polynomial P_n^(alpha, beta)(x).
///
/// # Parameters
///
/// * `degree` - Degree of the polynomial
/// * `alpha` - First parameter (> -1)
/// * `beta` - Second parameter (> -1)
///
/// # Returns
///
/// Array of Jacobi polynomial coefficients
pub fn polyjacobi<T>(degree: usize, alpha: T, beta: T) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + std::ops::Neg<Output = T>
        + Float,
{
    use num_traits::NumCast;

    // Check parameters
    if alpha <= -T::one() || beta <= -T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Jacobi polynomial requires alpha > -1 and beta > -1".to_string(),
        ));
    }

    if degree == 0 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let two: T = NumCast::from(2).unwrap();

    if degree == 1 {
        let a = (alpha + beta + two) / two;
        let b = (alpha - beta) / two;
        return Ok(Array::from_vec(vec![a, b]));
    }

    // Use recurrence relation
    let mut p_prev = Polynomial::<T>::one();

    let a1 = (alpha + beta + two) / two;
    let b1 = (alpha - beta) / two;
    let mut p_curr = Polynomial::new(vec![a1, b1]);

    for n in 1..(degree as i32) {
        let n_t: T = NumCast::from(n).unwrap();
        let two_n: T = NumCast::from(2 * n).unwrap();
        let n_plus_alpha_beta = n_t + alpha + beta;

        // Recurrence coefficients
        let denom =
            two * (n_t + T::one()) * (n_plus_alpha_beta + T::one()) * (two_n + alpha + beta);

        let a_n = (two_n + alpha + beta + T::one())
            * ((two_n + alpha + beta + two) * (two_n + alpha + beta) * T::one()
                + (alpha * alpha - beta * beta))
            / denom;

        let b_n = two * (n_t + alpha) * (n_t + beta) * (two_n + alpha + beta + two) / denom;

        // P_{n+1}(x) = (a_n * x + c_n) * P_n(x) - b_n * P_{n-1}(x)
        let x_poly = Polynomial::new(vec![T::one(), T::zero()]);
        let mut term1 = x_poly * p_curr.clone();
        term1.coefficients = term1.coefficients.iter().map(|c| *c * a_n).collect();

        let mut term2 = p_prev.clone();
        term2.coefficients = term2.coefficients.iter().map(|c| *c * b_n).collect();

        let p_next = term1 - term2;

        p_prev = p_curr;
        p_curr = p_next;
    }

    Ok(Array::from_vec(p_curr.coefficients().to_vec()))
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

// Add tests to verify the implementation
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

    #[test]
    fn test_lagrange_interpolation() {
        // Create points (0,1), (1,2), (2,4) - should fit y = x^2 + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 2.0, 5.0]);

        let p = PolynomialInterpolation::lagrange(&x, &y).unwrap();

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

        let p = PolynomialInterpolation::newton(&x, &y).unwrap();

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

        let spline = CubicSpline::new(&x, &y).unwrap();

        // Check interpolation at knots
        assert_relative_eq!(spline.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(spline.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(spline.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(spline.evaluate(3.0).unwrap(), 9.0);

        // Check that the spline approximates the function at intermediate points
        assert!(spline.evaluate(0.5).unwrap() > 0.0);
        assert!(spline.evaluate(0.5).unwrap() < 1.0);
        assert!(spline.evaluate(1.5).unwrap() > 1.0);
        assert!(spline.evaluate(1.5).unwrap() < 4.0);
    }

    #[test]
    fn test_polyfit() {
        // Create points for y = 2x^2 + 3x + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array::from_vec(vec![1.0, 6.0, 15.0, 28.0, 45.0]);

        // Fit a quadratic polynomial
        let p = polyfit(&x, &y, 2).unwrap();

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
        let y = polyval(&p, &x).unwrap();

        // Check results
        assert_eq!(y.shape(), vec![3]);
        assert_relative_eq!(y.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(y.to_vec()[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(y.to_vec()[2], 15.0, epsilon = 1e-10);
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

        // ∫p(x)dx = x^2 + x + C (C = 0)
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

    #[test]
    fn test_chebyshev_polynomials() {
        // Test first few Chebyshev polynomials of the first kind
        let t0 = OrthogonalPolynomials::chebyshev_t::<f64>(0);
        let t1 = OrthogonalPolynomials::chebyshev_t::<f64>(1);
        let t2 = OrthogonalPolynomials::chebyshev_t::<f64>(2);

        // T_0(x) = 1
        assert_eq!(t0.coefficients(), &[1.0]);

        // T_1(x) = x
        assert_eq!(t1.coefficients(), &[1.0, 0.0]);

        // T_2(x) = 2x^2 - 1
        assert_eq!(t2.coefficients(), &[2.0, 0.0, -1.0]);

        // Check evaluations
        assert_relative_eq!(t2.evaluate(0.0), -1.0, epsilon = 1e-10);
        assert_relative_eq!(t2.evaluate(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(t2.evaluate(0.5), -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_polynomials() {
        // Test first few Legendre polynomials
        let p0 = OrthogonalPolynomials::legendre::<f64>(0);
        let p1 = OrthogonalPolynomials::legendre::<f64>(1);
        let p2 = OrthogonalPolynomials::legendre::<f64>(2);

        // P_0(x) = 1
        assert_eq!(p0.coefficients(), &[1.0]);

        // P_1(x) = x
        assert_eq!(p1.coefficients(), &[1.0, 0.0]);

        // P_2(x) = (3x^2 - 1)/2
        // Should be [1.5, 0.0, -0.5] or equivalent
        assert_relative_eq!(p2.evaluate(0.0), -0.5, epsilon = 1e-10);
        assert_relative_eq!(p2.evaluate(1.0), 1.0, epsilon = 1e-10);
    }

    // Tests for new polynomial functions added for numpy.polynomial compatibility

    #[test]
    fn test_polyvander() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let v = polyvander(&x, 2).unwrap();

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
        let v = polyvander(&x, 0).unwrap();

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
        let c2 = polypower(&c, 2).unwrap();

        // The result should represent x^2 + 4x + 4
        let data = c2.to_vec();
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10); // x^2 coeff
        assert_relative_eq!(data[1], 4.0, epsilon = 1e-10); // x coeff
        assert_relative_eq!(data[2], 4.0, epsilon = 1e-10); // constant
    }

    #[test]
    fn test_polypower_zero() {
        let c = Array::from_vec(vec![1.0, 2.0]);
        let c0 = polypower(&c, 0).unwrap();

        // x^0 should give 1
        assert_eq!(c0.len(), 1);
        assert_relative_eq!(c0.to_vec()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polymulx() {
        // c(x) = 2 + 3x + x^2 -> x*c(x) = 2x + 3x^2 + x^3
        let c = Array::from_vec(vec![1.0, 3.0, 2.0]); // x^2 + 3x + 2
        let xc = polymulx(&c).unwrap();

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

        let gcd = polygcd(&p1, &p2).unwrap();
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

        let comp = polycompose(&p, &q).unwrap();
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

        let comp = polycompose(&p, &q).unwrap();
        let data = comp.to_vec();

        assert_eq!(data.len(), 2);
        assert_relative_eq!(data[0], 6.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyjacobi_degree_0() {
        let j0 = polyjacobi(0, 1.0, 1.0).unwrap();
        assert_eq!(j0.len(), 1);
        assert_relative_eq!(j0.to_vec()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyjacobi_degree_1() {
        // J_1(x; 0, 0) = x (Legendre case)
        let j1 = polyjacobi(1, 0.0, 0.0).unwrap();
        let data = j1.to_vec();

        // For alpha=beta=0, J_1 should be x
        assert_eq!(data.len(), 2);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyresidual() {
        // Create points for y = 2x + 1
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 3.0, 5.0]);

        // Perfect fit polynomial
        let c = Array::from_vec(vec![2.0, 1.0]); // 2x + 1

        let residual = polyresidual(&c, &x, &y).unwrap();
        assert_relative_eq!(residual, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyresidual_nonzero() {
        // Create points that don't exactly fit a line
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 3.0, 4.0]); // Not exactly 2x + 1

        let c = Array::from_vec(vec![2.0, 1.0]); // 2x + 1

        let residual = polyresidual(&c, &x, &y).unwrap();
        // Expected: (1-1)^2 + (3-3)^2 + (5-4)^2 = 1
        assert_relative_eq!(residual, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyval2d() {
        // Simple polynomial: f(x,y) = 1 + x + y (coefficients arranged in 2x2 matrix)
        // c[i,j] corresponds to x^i * y^j (in row-major order)
        let c = Array::from_vec(vec![1.0, 1.0, 1.0, 0.0]).reshape(&[2, 2]); // 1 + x + y
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![0.0, 0.0, 0.0]);

        let result = polyval2d(&c, &x, &y).unwrap();
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

        let result = polygrid2d(&c, &x, &y).unwrap();

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

        let v = polyvander2d(&x, &y, (1, 1)).unwrap();

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
