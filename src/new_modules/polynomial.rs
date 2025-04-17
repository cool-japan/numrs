use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero, One};
use num_complex::Complex;
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div};

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
        if self.coefficients.is_empty() || (self.coefficients.len() == 1 && self.coefficients[0] == T::zero()) {
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
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + 
       From<i32> + PartialEq + std::ops::Neg<Output = T>,
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

/// General polynomial functions

/// Fit a polynomial of specified degree to the data points
pub fn polyfit<T>(x: &Array<T>, y: &Array<T>, degree: usize) -> Result<Polynomial<T>>
where
    T: Clone + Zero + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + 
       PartialEq + Debug + std::ops::Neg<Output = T> + Float
{
    let x_shape = x.shape();
    let y_shape = y.shape();
    
    if x_shape.len() != 1 || y_shape.len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "polyfit requires 1D arrays of points".to_string()
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
        
        for j in (i+1)..=degree {
            let val = coeff_matrix[j][i].abs();
            if val > max_val {
                max_val = val;
                max_row = j;
            }
        }
        
        // Swap rows if necessary
        if max_row != i {
            coeff_matrix.swap(i, max_row);
            let temp = rhs[i];
            rhs[i] = rhs[max_row];
            rhs[max_row] = temp;
        }
        
        // Eliminate
        for j in (i+1)..=degree {
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
        for j in (i+1)..=degree {
            sum = sum + coeff_matrix[i][j] * coefficients[j];
        }
        coefficients[i] = (rhs[i] - sum) / coeff_matrix[i][i];
    }
    
    Ok(Polynomial::new(coefficients))
}

/// Evaluate a polynomial at points
pub fn polyval<T>(p: &Polynomial<T>, x: &Array<T>) -> Result<Array<T>> 
where
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T> + PartialEq
{
    p.evaluate_array(x)
}

/// Find the roots of a polynomial
pub fn roots<T>(p: &Polynomial<T>) -> Result<Array<Complex<T>>> 
where
    T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + 
       Div<Output = T> + PartialEq + Debug + std::ops::Neg<Output = T> + Float
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
                Complex::new(root2, T::zero())
            ]));
        } else {
            // Complex roots
            let real_part = -b / (T::from(2.0).unwrap() * a);
            let imag_part = (-discriminant).sqrt() / (T::from(2.0).unwrap() * a);
            return Ok(Array::from_vec(vec![
                Complex::new(real_part, imag_part),
                Complex::new(real_part, -imag_part)
            ]));
        }
    }
    
    // For higher degree polynomials, use companion matrix eigenvalues
    // Normalize the polynomial
    let leading = coeffs[0];
    for i in 0..coeffs.len() {
        coeffs[i] = coeffs[i] / leading;
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
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialEq + Debug + std::ops::Neg<Output = T>,
    {
        let x_shape = x.shape();
        let y_shape = y.shape();
        
        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Lagrange interpolation requires 1D arrays of points".to_string()
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
            for j in (i+1)..n {
                if x_data[i] == x_data[j] {
                    return Err(NumRs2Error::InvalidOperation(
                        "Lagrange interpolation requires unique x values".to_string()
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
            term.coefficients = term.coefficients.iter().map(|c| c.clone() * scale.clone()).collect();
            
            // Add to the result
            result = result + term;
        }
        
        Ok(result)
    }
    
    /// Interpolate a polynomial through points (x, y) using Newton's divided differences
    pub fn newton<T>(x: &Array<T>, y: &Array<T>) -> Result<Polynomial<T>>
    where
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialEq + Debug + std::ops::Neg<Output = T>,
    {
        let x_shape = x.shape();
        let y_shape = y.shape();
        
        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Newton interpolation requires 1D arrays of points".to_string()
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
            for i in 0..(n-j) {
                divided_diff[i][j] = (divided_diff[i+1][j-1].clone() - divided_diff[i][j-1].clone()) /
                                      (x_data[i+j].clone() - x_data[i].clone());
            }
        }
        
        // Build the Newton form of the interpolating polynomial
        let mut result = Polynomial::new(vec![divided_diff[0][0].clone()]);
        let mut term: Polynomial<T> = Polynomial::one();
        
        for j in 1..n {
            // Multiply by (x - x_j-1)
            let neg_xj = T::zero() - x_data[j-1].clone();
            let linear_term = Polynomial::new(vec![T::one(), neg_xj]);
            term = term * linear_term;
            
            // Add a_j * term
            let mut scaled_term = term.clone();
            scaled_term.coefficients = scaled_term.coefficients.iter()
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
    T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + Debug + Float,
{
    /// Create a new natural cubic spline from x and y data points
    pub fn new(x: &Array<T>, y: &Array<T>) -> Result<Self> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        
        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Cubic spline requires 1D arrays of points".to_string()
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
                "Cubic spline requires at least 3 points".to_string()
            ));
        }
        
        let x_data = x.to_vec();
        let y_data = y.to_vec();
        
        // Check that x values are in ascending order
        for i in 1..n {
            if x_data[i] <= x_data[i-1] {
                return Err(NumRs2Error::InvalidOperation(
                    "x values must be in strictly ascending order for cubic spline".to_string()
                ));
            }
        }
        
        // Compute second derivatives using the tridiagonal algorithm
        let mut a = vec![T::zero(); n-1];
        let mut b = vec![T::zero(); n];
        let mut c = vec![T::zero(); n-1];
        let mut d = vec![T::zero(); n];
        
        // Set up the tridiagonal system
        for i in 1..n-1 {
            let h_prev = x_data[i].clone() - x_data[i-1].clone();
            let h_next = x_data[i+1].clone() - x_data[i].clone();
            
            a[i-1] = h_prev;
            b[i] = T::from(2.0).unwrap() * (h_prev.clone() + h_next.clone());
            c[i] = h_next;
            
            let dy_prev = y_data[i].clone() - y_data[i-1].clone();
            let dy_next = y_data[i+1].clone() - y_data[i].clone();
            
            d[i] = T::from(6.0).unwrap() * (dy_next / h_next - dy_prev / h_prev);
        }
        
        // Natural boundary conditions: second derivatives at endpoints are zero
        b[0] = T::one();
        b[n-1] = T::one();
        c[0] = T::zero();
        a[n-2] = T::zero();
        d[0] = T::zero();
        d[n-1] = T::zero();
        
        // Solve the tridiagonal system using Thomas algorithm
        // Forward elimination
        for i in 1..n {
            let m = a[i-1].clone() / b[i-1].clone();
            b[i] = b[i].clone() - m.clone() * c[i-1].clone();
            d[i] = d[i].clone() - m.clone() * d[i-1].clone();
        }
        
        // Back substitution
        let mut second_derivs = vec![T::zero(); n];
        second_derivs[n-1] = d[n-1].clone() / b[n-1].clone();
        
        for i in (0..n-1).rev() {
            second_derivs[i] = (d[i].clone() - c[i].clone() * second_derivs[i+1].clone()) / b[i].clone();
        }
        
        // Compute the coefficients for each segment
        let mut coefficients = Vec::with_capacity(n-1);
        
        for i in 0..n-1 {
            let h = x_data[i+1].clone() - x_data[i].clone();
            let a = (second_derivs[i+1].clone() - second_derivs[i].clone()) / (T::from(6.0).unwrap() * h.clone());
            let b = second_derivs[i].clone() / T::from(2.0).unwrap();
            let c = (y_data[i+1].clone() - y_data[i].clone()) / h.clone() -
                   (second_derivs[i+1].clone() + T::from(2.0).unwrap() * second_derivs[i].clone()) * h.clone() / T::from(6.0).unwrap();
            let d = y_data[i].clone();
            
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
                "Evaluation point outside the domain of the spline".to_string()
            ));
        }
        
        // Binary search to find the segment
        let mut left = 0;
        let mut right = self.coefficients.len() - 1;
        
        while left <= right {
            let mid = (left + right) / 2;
            
            if x >= self.knots[mid] && x <= self.knots[mid + 1] {
                // Found the segment
                let t = x - self.knots[mid].clone();
                let coeffs = &self.coefficients[mid];
                
                return Ok(((coeffs[0].clone() * t.clone() + coeffs[1].clone()) * t.clone() + coeffs[2].clone()) * t.clone() + coeffs[3].clone());
            }
            
            if x < self.knots[mid] {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        // If we get here, we're in the last segment
        let last_idx = self.coefficients.len() - 1;
        let t = x - self.knots[last_idx].clone();
        let coeffs = &self.coefficients[last_idx];
        
        Ok(((coeffs[0].clone() * t.clone() + coeffs[1].clone()) * t.clone() + coeffs[2].clone()) * t.clone() + coeffs[3].clone())
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
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<i32> + PartialEq + std::ops::Neg<Output = T>
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
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<i32> + PartialEq + std::ops::Neg<Output = T>
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
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32> + PartialEq + std::ops::Neg<Output = T>
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
            term1.coefficients = term1.coefficients.iter().map(|c| c.clone() * two_k_plus_1.clone()).collect();
            
            // Scalar multiplication of polynomial
            let mut term2 = p_prev.clone();
            term2.coefficients = term2.coefficients.iter().map(|c| c.clone() * k_t.clone()).collect();
            
            // Polynomial subtraction
            let mut p_next = term1 - term2;
            // Scalar division of polynomial
            p_next.coefficients = p_next.coefficients.iter().map(|c| c.clone() / k_plus_1.clone()).collect();
            
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
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<i32> + PartialEq + std::ops::Neg<Output = T>
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
            term2.coefficients = term2.coefficients.iter().map(|c| c.clone() * two_k.clone()).collect();
            
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
        T: Clone + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32> + PartialEq + std::ops::Neg<Output = T>
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
            term2.coefficients = term2.coefficients.iter().map(|c| c.clone() * k_t.clone()).collect();
            
            // Subtract to get the next polynomial and divide by (n+1)
            let mut l_next = term1 - term2;
            l_next.coefficients = l_next.coefficients.iter().map(|c| c.clone() / k_plus_1.clone()).collect();
            
            l_prev = l_curr;
            l_curr = l_next;
        }
        
        l_curr
    }
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
}
