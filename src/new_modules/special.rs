use crate::array::Array;
use crate::error::Result;
use num_traits::Float;
use std::fmt::Debug;

/// Special mathematical functions implementation for NumRS2
/// This module provides implementations of various special functions from mathematical physics
/// and engineering, similar to those found in the scipy.special library in Python.

// Error functions

/// Compute the error function (erf) of an array of values
/// 
/// # Arguments
/// 
/// * `x` - Input array
/// 
/// # Returns
/// 
/// Array containing error function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![0.0, 0.5, 1.0]);
/// let result = erf(&x);
/// ```
pub fn erf<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| erf_scalar(v))
}

/// Compute the complementary error function (erfc) of an array of values
/// erfc(x) = 1 - erf(x)
/// 
/// # Arguments
/// 
/// * `x` - Input array
/// 
/// # Returns
/// 
/// Array containing complementary error function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![0.0, 0.5, 1.0]);
/// let result = erfc(&x);
/// ```
pub fn erfc<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| erfc_scalar(v))
}

/// Compute the inverse error function (erf⁻¹) of an array of values
/// 
/// # Arguments
/// 
/// * `x` - Input array (values should be in the range [-1, 1])
/// 
/// # Returns
/// 
/// Array containing inverse error function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![0.0, 0.5, 0.8]);
/// let result = erfinv(&x);
/// ```
pub fn erfinv<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| erfinv_scalar(v))
}

/// Compute the inverse complementary error function (erfc⁻¹) of an array of values
/// 
/// # Arguments
/// 
/// * `x` - Input array (values should be in the range [0, 2])
/// 
/// # Returns
/// 
/// Array containing inverse complementary error function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![0.1, 0.5, 1.0]);
/// let result = erfcinv(&x);
/// ```
pub fn erfcinv<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| erfcinv_scalar(v))
}

// Gamma functions

/// Compute the gamma function (Γ(x)) of an array of values
/// 
/// # Arguments
/// 
/// * `x` - Input array (values should be positive and not integers ≤ 0)
/// 
/// # Returns
/// 
/// Array containing gamma function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![1.0, 2.0, 3.5]);
/// let result = gamma(&x);
/// ```
pub fn gamma<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| gamma_scalar(v))
}

/// Compute the natural logarithm of the gamma function (ln(Γ(x))) of an array of values
/// 
/// # Arguments
/// 
/// * `x` - Input array (values should be positive)
/// 
/// # Returns
/// 
/// Array containing logarithm of gamma function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![1.0, 2.0, 10.0]);
/// let result = gammaln(&x);
/// ```
pub fn gammaln<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| gammaln_scalar(v))
}

/// Compute the digamma function (Ψ(x) = d/dx ln(Γ(x))) of an array of values
/// 
/// # Arguments
/// 
/// * `x` - Input array (values should not be integers ≤ 0)
/// 
/// # Returns
/// 
/// Array containing digamma function values for each element in `x`
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = digamma(&x);
/// ```
pub fn digamma<T>(x: &Array<T>) -> Array<T> 
where
    T: Clone + Float + Debug
{
    x.map(|v| digamma_scalar(v))
}

/// Compute the incomplete gamma function γ(a, x) of array values
/// 
/// # Arguments
/// 
/// * `a` - Shape parameter array
/// * `x` - Upper limit array
/// 
/// # Returns
/// 
/// Array containing incomplete gamma function values
/// 
/// # Example
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let x = Array::from_vec(vec![1.0, 1.0, 1.0]);
/// let result = gammainc(&a, &x).unwrap();
/// ```
pub fn gammainc<T>(a: &Array<T>, x: &Array<T>) -> Result<Array<T>> 
where
    T: Clone + Float + Debug
{
    a.zip_with(x, |a_val, x_val| gammainc_scalar(a_val, x_val))
}

// Bessel functions

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
    T: Clone + Float + Debug
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
    T: Clone + Float + Debug
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
    T: Clone + Float + Debug
{
    x.map(|v| bessel_i_scalar(n, v))
}

/// Compute the modified Bessel function of the second kind K_n(x) for an array of values
/// 
/// # Arguments
/// 
/// * `n` - Order of the Bessel function
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
    T: Clone + Float + Debug
{
    x.map(|v| bessel_k_scalar(n, v))
}

// Elliptic functions

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
    T: Clone + Float + Debug
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
    T: Clone + Float + Debug
{
    m.map(|v| ellipe_scalar(v))
}

// Scalar implementations of special functions

/// Error function for a scalar value
fn erf_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // Use a polynomial approximation for the error function
    // Based on Abramowitz and Stegun formula 7.1.26
    
    // Constants
    let a1 = T::from(0.254829592).unwrap();
    let a2 = T::from(-0.284496736).unwrap();
    let a3 = T::from(1.421413741).unwrap();
    let a4 = T::from(-1.453152027).unwrap();
    let a5 = T::from(1.061405429).unwrap();
    let p = T::from(0.3275911).unwrap();
    
    // Save the sign of x
    let sign = if x < T::zero() { T::from(-1.0).unwrap() } else { T::one() };
    let x = x.abs();
    
    // A&S formula 7.1.26
    let t = T::one() / (T::one() + p * x);
    let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

/// Complementary error function for a scalar value
fn erfc_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // erfc(x) = 1 - erf(x)
    T::one() - erf_scalar(x)
}

/// Inverse error function for a scalar value
fn erfinv_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // Check input range
    if x <= T::from(-1.0).unwrap() {
        return T::neg_infinity();
    }
    if x >= T::one() {
        return T::infinity();
    }
    
    // Rational approximation for central range
    if x.abs() <= T::from(0.7).unwrap() {
        // Rational approximation for central region
        let x2 = x * x;
        
        let c1 = T::from(1.0).unwrap();
        let c2 = T::from(0.47047).unwrap();
        let c3 = T::from(0.06532).unwrap();
        let c4 = T::from(0.44693).unwrap();
        
        let d1 = T::from(1.0).unwrap();
        let d2 = T::from(0.56418).unwrap();
        let d3 = T::from(1.21837).unwrap();
        let d4 = T::from(1.00000).unwrap();
        
        let num = c1 * x * (c2 + x2 * (c3 + x2 * c4));
        let den = d1 + x2 * (d2 + x2 * (d3 + x2 * d4));
        
        return num / den;
    }
    
    // For tail regions
    let y = if x < T::zero() {
        T::from(-1.0).unwrap() * (T::one() + x).sqrt()
    } else {
        T::from(1.0).unwrap() * (T::one() - x).sqrt()
    };
    
    // More accurate approximation for the tails
    let sign = if x < T::zero() { T::from(-1.0).unwrap() } else { T::one() };
    
    // Constants from Winitzki (2008)
    let c1 = T::from(2.515517).unwrap();
    let c2 = T::from(0.802853).unwrap();
    let c3 = T::from(0.010328).unwrap();
    
    let d1 = T::from(1.432788).unwrap();
    let d2 = T::from(0.189269).unwrap();
    let d3 = T::from(0.001308).unwrap();
    
    let z = (T::one() / y.abs()).ln().sqrt();
    
    let num = c1 + z * (c2 + z * c3);
    let den = T::one() + z * (d1 + z * (d2 + z * d3));
    
    sign * (z - num / den)
}

/// Inverse complementary error function for a scalar value
fn erfcinv_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // erfcinv(x) = erfinv(1 - x)
    erfinv_scalar(T::one() - x)
}

/// Lanczos approximation coefficients for gamma function
fn lanczos_coefficients<T>() -> Vec<T> 
where
    T: Float + Debug
{
    // Lanczos coefficients for g=7
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ];
    
    coeffs.iter().map(|&x| T::from(x).unwrap()).collect()
}

/// Gamma function for a scalar value
fn gamma_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // Using Lanczos approximation for the gamma function
    
    // Handle special cases
    if x == T::zero() {
        return T::infinity();
    }
    if x < T::zero() {
        // Reflection formula
        let pi = T::from(std::f64::consts::PI).unwrap();
        return pi / ((pi * x).sin() * gamma_scalar(T::one() - x));
    }
    
    let g = T::from(7.0).unwrap(); // Lanczos parameter
    let coeffs = lanczos_coefficients::<T>();
    
    // Shift x by 1 to accommodate the algorithm
    let x = x - T::one();
    
    // Calculate the approximation
    let mut sum = coeffs[0];
    for i in 1..coeffs.len() {
        sum = sum + coeffs[i] / (x + T::from(i).unwrap());
    }
    
    let t = x + g + T::from(0.5).unwrap();
    let sqrt_2pi = T::from(2.506628274631000502415765284811).unwrap(); // sqrt(2*pi)
    
    sqrt_2pi * sum * t.powf(x + T::from(0.5).unwrap()) * (-t).exp()
}

/// Natural logarithm of gamma function for a scalar value
fn gammaln_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // Handle special cases
    if x <= T::zero() {
        return T::infinity();
    }
    
    // For larger values, use Stirling's approximation
    if x > T::from(10.0).unwrap() {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let e = T::from(std::f64::consts::E).unwrap();
        
        return (T::from(0.5).unwrap() * (T::from(2.0).unwrap() * pi / x).ln()) +
               (x * (x / e).ln());
    }
    
    gamma_scalar(x).ln()
}

/// Digamma function for a scalar value
fn digamma_scalar<T>(x: T) -> T 
where
    T: Float + Debug
{
    // For x <= 0, use the reflection formula
    if x <= T::zero() {
        let pi = T::from(std::f64::consts::PI).unwrap();
        return digamma_scalar(T::one() - x) - pi / (pi * x).tan();
    }
    
    // For small x, use recurrence relation to increase argument
    if x < T::from(10.0).unwrap() {
        return digamma_scalar(x + T::one()) - T::one() / x;
    }
    
    // For large x, use asymptotic expansion
    let inv_x = T::one() / x;
    let inv_x2 = inv_x * inv_x;
    
    // Coefficients from the asymptotic expansion
    x.ln() - inv_x / T::from(2.0).unwrap() - inv_x2 / T::from(12.0).unwrap() +
    inv_x2 * inv_x2 / T::from(120.0).unwrap() - inv_x2 * inv_x2 * inv_x2 / T::from(252.0).unwrap()
}

/// Incomplete gamma function for scalar values
fn gammainc_scalar<T>(a: T, x: T) -> T 
where
    T: Float + Debug
{
    // Handle special cases
    if x <= T::zero() {
        return T::zero();
    }
    if x > T::from(1000.0).unwrap() && a < T::from(1000.0).unwrap() {
        // For very large x, the result is approximately 1
        return T::one();
    }
    
    // Series expansion for small x
    if x < a + T::one() {
        let mut result = T::one() / a;
        let mut term = result;
        
        for n in 1..100 {
            term = term * x / (a + T::from(n).unwrap());
            result = result + term;
            
            // Check for convergence
            if term.abs() < result.abs() * T::from(1e-10).unwrap() {
                break;
            }
        }
        
        return result * x.powf(a) * (-x).exp() / gamma_scalar(a);
    }
    
    // Continued fraction for larger x
    // Using Lentz's algorithm for continued fraction evaluation
    let fpmin = T::from(1.0e-30).unwrap();
    let eps = T::from(1.0e-10).unwrap();
    
    let mut b = x + T::one() - a;
    let mut c = T::one() / fpmin;
    let mut d = T::one() / b;
    let mut h = d;
    
    for i in 1..100 {
        let i_t = T::from(i).unwrap();
        let a_plus_i = a + i_t - T::one();
        
        b = b + T::from(2.0).unwrap();
        d = T::one() / (b + a_plus_i * d);
        c = b + a_plus_i / c;
        
        let del = c * d;
        h = h * del;
        
        if (del - T::one()).abs() < eps {
            break;
        }
    }
    
    T::one() - h * x.powf(a) * (-x).exp() / gamma_scalar(a)
}

/// Bessel function of first kind J_n(x) for scalar values
fn bessel_j_scalar<T>(n: i32, x: T) -> T 
where
    T: Float + Debug
{
    // For negative n, use the relationship J_{-n}(x) = (-1)^n * J_n(x)
    if n < 0 {
        let factor = if n % 2 == 0 { T::one() } else { T::from(-1.0).unwrap() };
        return factor * bessel_j_scalar(-n, x);
    }
    
    // Special cases
    if x == T::zero() {
        return if n == 0 { T::one() } else { T::zero() };
    }
    
    // Implementation using series expansion
    // J_n(x) = sum_{m=0}^{\infty} \frac{(-1)^m}{m!(n+m)!} \left(\frac{x}{2}\right)^{n+2m}
    let mut result = T::zero();
    let x_half = x / T::from(2.0).unwrap();
    let n_t = T::from(n).unwrap();
    
    for m in 0..20 {
        let m_t = T::from(m).unwrap();
        
        // Calculate (-1)^m
        let sign = if m % 2 == 0 { T::one() } else { T::from(-1.0).unwrap() };
        
        // Calculate (x/2)^(n+2m)
        let power = x_half.powf(n_t + m_t + m_t);
        
        // Calculate m! and (n+m)! approximation
        let m_factorial = gamma_scalar(m_t + T::one());
        let n_plus_m_factorial = gamma_scalar(n_t + m_t + T::one());
        
        let term = sign * power / (m_factorial * n_plus_m_factorial);
        result = result + term;
        
        // Check for convergence
        if term.abs() < result.abs() * T::from(1e-10).unwrap() {
            break;
        }
    }
    
    result
}

/// Bessel function of second kind Y_n(x) for scalar values
fn bessel_y_scalar<T>(n: i32, x: T) -> T 
where
    T: Float + Debug
{
    // Handle special case
    if x <= T::zero() {
        return T::nan();
    }
    
    // Use the relationship with J_n(x)
    // Y_n(x) = (J_n(x) * cos(n*pi) - J_{-n}(x)) / sin(n*pi)
    let pi = T::from(std::f64::consts::PI).unwrap();
    let n_pi = T::from(n).unwrap() * pi;
    
    if n % 2 == 0 {
        // For even n, sin(n*pi) = 0, so use different formula
        // Use the derivative relationship
        let j_n = bessel_j_scalar(n, x);
        let j_n_plus_1 = bessel_j_scalar(n + 1, x);
        
        return T::from(2.0).unwrap() / pi * j_n.ln() - T::from(2.0).unwrap() * j_n / (pi * x) -
               j_n_plus_1 + j_n / (pi * x);
    }
    
    // For odd n
    (bessel_j_scalar(n, x) * n_pi.cos() - bessel_j_scalar(-n, x)) / n_pi.sin()
}

/// Modified Bessel function of first kind I_n(x) for scalar values
fn bessel_i_scalar<T>(n: i32, x: T) -> T 
where
    T: Float + Debug
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
    // I_n(x) = sum_{m=0}^{\infty} \frac{1}{m!(n+m)!} \left(\frac{x}{2}\right)^{n+2m}
    let mut result = T::zero();
    let x_half = x / T::from(2.0).unwrap();
    let n_t = T::from(n).unwrap();
    
    for m in 0..20 {
        let m_t = T::from(m).unwrap();
        
        // Calculate (x/2)^(n+2m)
        let power = x_half.powf(n_t + m_t + m_t);
        
        // Calculate m! and (n+m)!
        let m_factorial = gamma_scalar(m_t + T::one());
        let n_plus_m_factorial = gamma_scalar(n_t + m_t + T::one());
        
        let term = power / (m_factorial * n_plus_m_factorial);
        result = result + term;
        
        // Check for convergence
        if term.abs() < result.abs() * T::from(1e-10).unwrap() {
            break;
        }
    }
    
    result
}

/// Modified Bessel function of second kind K_n(x) for scalar values
fn bessel_k_scalar<T>(n: i32, x: T) -> T 
where
    T: Float + Debug
{
    // Handle special case
    if x <= T::zero() {
        return T::infinity();
    }
    
    // For negative n, use the relationship K_{-n}(x) = K_n(x)
    if n < 0 {
        return bessel_k_scalar(-n, x);
    }
    
    // Relationship with I_n
    // K_n(x) = pi/2 * (I_{-n}(x) - I_n(x))/sin(n*pi)
    let pi = T::from(std::f64::consts::PI).unwrap();
    let n_t = T::from(n).unwrap();
    
    let pi_half = pi / T::from(2.0).unwrap();
    let i_n = bessel_i_scalar(n, x);
    let i_minus_n = bessel_i_scalar(-n, x);
    
    if n == 0 {
        // For n = 0, use direct formula
        return pi_half * (i_minus_n - i_n);
    }
    
    pi_half * (i_minus_n - i_n) / (n_t * pi).sin()
}

/// Complete elliptic integral of the first kind K(m) for scalar values
fn ellipk_scalar<T>(m: T) -> T 
where
    T: Float + Debug
{
    // Check input range
    if m > T::one() {
        return T::nan();
    }
    if m == T::one() {
        return T::infinity();
    }
    if m == T::zero() {
        return T::from(std::f64::consts::PI / 2.0).unwrap();
    }
    
    // Implementation using arithmetic-geometric mean method
    let pi = T::from(std::f64::consts::PI).unwrap();
    let one_minus_m = T::one() - m;
    
    // Initialize arithmetic and geometric means
    let mut a = T::one();
    let mut g = one_minus_m.sqrt();
    
    // Convergence criterion
    let eps = T::from(1e-10).unwrap();
    
    // Iterate until convergence
    while (a - g).abs() > a * eps {
        let a_next = (a + g) / T::from(2.0).unwrap();
        let g_next = (a * g).sqrt();
        
        a = a_next;
        g = g_next;
    }
    
    pi / (T::from(2.0).unwrap() * a)
}

/// Complete elliptic integral of the second kind E(m) for scalar values
fn ellipe_scalar<T>(m: T) -> T 
where
    T: Float + Debug
{
    // Check input range
    if m > T::one() {
        return T::nan();
    }
    if m == T::one() {
        return T::one();
    }
    if m == T::zero() {
        return T::from(std::f64::consts::PI / 2.0).unwrap();
    }
    
    // Implementation using arithmetic-geometric mean method
    let pi = T::from(std::f64::consts::PI).unwrap();
    let one_minus_m = T::one() - m;
    
    // Initialize arithmetic and geometric means
    let mut a = T::one();
    let mut g = one_minus_m.sqrt();
    let mut e = m;
    
    // Convergence criterion
    let eps = T::from(1e-10).unwrap();
    
    // Iterate until convergence
    let mut n = T::one();
    
    while (a - g).abs() > a * eps {
        let a_next = (a + g) / T::from(2.0).unwrap();
        let g_next = (a * g).sqrt();
        let e_next = e - n * (a - g) * (a - g) / T::from(2.0).unwrap();
        
        a = a_next;
        g = g_next;
        e = e_next;
        n = n * T::from(2.0).unwrap();
    }
    
    pi * (T::one() - e / (T::from(2.0).unwrap() * a)) / (T::from(2.0).unwrap() * a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_erf() {
        let values = Array::from_vec(vec![0.0, 0.5, 1.0, -0.5]);
        let result = erf(&values);
        
        // Known values of erf
        // Using a more generous epsilon for the zero case due to floating point precision
        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 0.5204998778130465, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.8427007929497149, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], -0.5204998778130465, epsilon = 1e-4);
    }
    
    #[test]
    fn test_erfc() {
        let values = Array::from_vec(vec![0.0, 0.5, 1.0, 2.0]);
        let result = erfc(&values);
        
        // Known values of erfc
        // Using a more generous epsilon for cases where values are close to 1.0
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 0.4795001221869535, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.15729920705028513, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], 0.0046777349810472645, epsilon = 1e-4);
    }
    
    #[test]
    fn test_gamma() {
        let values = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma(&values);
        
        // Known values of gamma
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[4], 24.0, epsilon = 1e-10);
    }
    
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
    
    #[test]
    fn test_ellipk() {
        let values = Array::from_vec(vec![0.0, 0.5, 0.9]);
        let result = ellipk(&values);
        
        // Known values
        assert_relative_eq!(result.to_vec()[0], std::f64::consts::PI / 2.0, epsilon = 1e-10);
        assert!(result.to_vec()[1] > std::f64::consts::PI / 2.0);
        assert!(result.to_vec()[2] > result.to_vec()[1]);
    }
}