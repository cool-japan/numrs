//! Special and orthogonal polynomials
//!
//! This module provides implementations of various special polynomials including
//! Chebyshev, Legendre, Hermite, Laguerre, and Jacobi polynomials.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::ops::{Add, Div, Mul, Sub};

use super::core::Polynomial;

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
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let t2 = polychebyshev::<f64>(2, 1)?; // T_2(x) = 2x^2 - 1
/// let u1 = polychebyshev::<f64>(1, 2)?; // U_1(x) = 2x
/// # Ok(())
/// # }
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
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let p2 = polylegendre::<f64>(2)?; // P_2(x) = (3x^2 - 1)/2
/// # Ok(())
/// # }
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
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let h2 = polyhermite::<f64>(2)?; // H_2(x) = 4x^2 - 2
/// # Ok(())
/// # }
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
/// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
///
/// let l2 = polylaguerre::<f64>(2)?; // L_2(x) = (x^2 - 4x + 2)/2
/// # Ok(())
/// # }
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

    let two: T = NumCast::from(2).expect("2 should convert to numeric type");

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
        let n_t: T = NumCast::from(n).expect("n should convert to numeric type");
        let two_n: T = NumCast::from(2 * n).expect("2*n should convert to numeric type");
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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

    #[test]
    fn test_polyjacobi_degree_0() {
        let j0 = polyjacobi(0, 1.0, 1.0).expect("Jacobi polynomial degree 0 should succeed");
        assert_eq!(j0.len(), 1);
        assert_relative_eq!(j0.to_vec()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polyjacobi_degree_1() {
        // J_1(x; 0, 0) = x (Legendre case)
        let j1 = polyjacobi(1, 0.0, 0.0).expect("Jacobi polynomial degree 1 should succeed");
        let data = j1.to_vec();

        // For alpha=beta=0, J_1 should be x
        assert_eq!(data.len(), 2);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(data[1], 0.0, epsilon = 1e-10);
    }
}
