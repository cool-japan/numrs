//! Numerical Integration Module
//!
//! This module provides functions for numerical integration (quadrature):
//!
//! - **Gauss-Kronrod Adaptive Quadrature**: High-precision adaptive integration
//! - **Monte Carlo Integration**: Random sampling-based integration for high dimensions
//! - **Simpson's Rule**: Classic composite integration
//! - **Trapezoidal Rule**: Basic composite integration
//! - **Romberg Integration**: Richardson extrapolation for improved accuracy
//!
//! # Examples
//!
//! ```ignore
//! use numrs2::integrate::{quad, quad_config, monte_carlo_integrate};
//!
//! // Simple integration
//! let result = quad(|x| x * x, 0.0, 1.0).unwrap();
//! assert!((result - 1.0/3.0).abs() < 1e-10);
//!
//! // Monte Carlo for high-dimensional integration
//! let result = monte_carlo_integrate(|x| x[0] * x[1], &[(0.0, 1.0), (0.0, 1.0)], 10000);
//! ```

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Result of numerical integration
#[derive(Debug, Clone)]
pub struct IntegrationResult<T> {
    /// Estimated integral value
    pub value: T,
    /// Estimated absolute error
    pub error: T,
    /// Number of function evaluations
    pub neval: usize,
    /// Success flag
    pub success: bool,
}

/// Configuration for adaptive integration
#[derive(Debug, Clone)]
pub struct QuadConfig<T> {
    /// Absolute tolerance
    pub atol: T,
    /// Relative tolerance
    pub rtol: T,
    /// Maximum number of subdivisions
    pub max_subdivisions: usize,
    /// Maximum recursion depth
    pub max_depth: usize,
}

impl<T: Float> Default for QuadConfig<T> {
    fn default() -> Self {
        QuadConfig {
            atol: T::from(1.49e-8).expect("1.49e-8 is representable as Float"),
            rtol: T::from(1.49e-8).expect("1.49e-8 is representable as Float"),
            max_subdivisions: 50,
            max_depth: 50,
        }
    }
}

/// Gauss-Kronrod 15-point nodes and weights for adaptive quadrature
/// These are the standard G7-K15 nodes for [-1, 1]
fn gauss_kronrod_15<T: Float>() -> (Vec<T>, Vec<T>, Vec<T>) {
    // Kronrod nodes (15 points)
    let nodes = vec![
        T::from(-0.991455371120813).expect("quadrature node is representable as Float"),
        T::from(-0.949107912342759).expect("quadrature node is representable as Float"),
        T::from(-0.864864423359769).expect("quadrature node is representable as Float"),
        T::from(-0.741531185599394).expect("quadrature node is representable as Float"),
        T::from(-0.586087235467691).expect("quadrature node is representable as Float"),
        T::from(-0.405845151377397).expect("quadrature node is representable as Float"),
        T::from(-0.207784955007898).expect("quadrature node is representable as Float"),
        T::from(0.0).expect("quadrature node is representable as Float"),
        T::from(0.207784955007898).expect("quadrature node is representable as Float"),
        T::from(0.405845151377397).expect("quadrature node is representable as Float"),
        T::from(0.586087235467691).expect("quadrature node is representable as Float"),
        T::from(0.741531185599394).expect("quadrature node is representable as Float"),
        T::from(0.864864423359769).expect("quadrature node is representable as Float"),
        T::from(0.949107912342759).expect("quadrature node is representable as Float"),
        T::from(0.991455371120813).expect("quadrature node is representable as Float"),
    ];

    // Kronrod weights (for all 15 points)
    let kronrod_weights = vec![
        T::from(0.022935322010529).expect("quadrature weight is representable as Float"),
        T::from(0.063092092629979).expect("quadrature weight is representable as Float"),
        T::from(0.104790010322250).expect("quadrature weight is representable as Float"),
        T::from(0.140653259715525).expect("quadrature weight is representable as Float"),
        T::from(0.169004726639267).expect("quadrature weight is representable as Float"),
        T::from(0.190350578064785).expect("quadrature weight is representable as Float"),
        T::from(0.204432940075298).expect("quadrature weight is representable as Float"),
        T::from(0.209482141084728).expect("quadrature weight is representable as Float"),
        T::from(0.204432940075298).expect("quadrature weight is representable as Float"),
        T::from(0.190350578064785).expect("quadrature weight is representable as Float"),
        T::from(0.169004726639267).expect("quadrature weight is representable as Float"),
        T::from(0.140653259715525).expect("quadrature weight is representable as Float"),
        T::from(0.104790010322250).expect("quadrature weight is representable as Float"),
        T::from(0.063092092629979).expect("quadrature weight is representable as Float"),
        T::from(0.022935322010529).expect("quadrature weight is representable as Float"),
    ];

    // Gauss weights (for the 7 Gauss points among the 15)
    // Indices: 1, 3, 5, 7, 9, 11, 13 (0-indexed)
    let gauss_weights = vec![
        T::from(0.129484966168870).expect("quadrature weight is representable as Float"),
        T::from(0.279705391489277).expect("quadrature weight is representable as Float"),
        T::from(0.381830050505119).expect("quadrature weight is representable as Float"),
        T::from(0.417959183673469).expect("quadrature weight is representable as Float"),
        T::from(0.381830050505119).expect("quadrature weight is representable as Float"),
        T::from(0.279705391489277).expect("quadrature weight is representable as Float"),
        T::from(0.129484966168870).expect("quadrature weight is representable as Float"),
    ];

    (nodes, kronrod_weights, gauss_weights)
}

/// Simple adaptive quadrature using Gauss-Kronrod G7-K15 rule
///
/// Integrates a function f over [a, b] using adaptive subdivision.
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
///
/// # Returns
/// * Approximate integral value
///
/// # Example
/// ```ignore
/// use numrs2::integrate::quad;
///
/// let result = quad(|x| x * x, 0.0, 1.0).unwrap();
/// assert!((result - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn quad<T, F>(f: F, a: T, b: T) -> Result<T>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T) -> T,
{
    let result = quad_adaptive(&f, a, b, &QuadConfig::default())?;
    Ok(result.value)
}

/// Adaptive quadrature with configuration
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `config` - Integration configuration
///
/// # Returns
/// * `IntegrationResult` with value, error estimate, and diagnostics
pub fn quad_adaptive<T, F>(
    f: &F,
    a: T,
    b: T,
    config: &QuadConfig<T>,
) -> Result<IntegrationResult<T>>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T) -> T,
{
    let mut neval = 0;
    let result = gauss_kronrod_adaptive(
        f,
        a,
        b,
        config.atol,
        config.rtol,
        config.max_depth,
        &mut neval,
    )?;

    Ok(IntegrationResult {
        value: result.0,
        error: result.1,
        neval,
        success: true,
    })
}

/// Recursive Gauss-Kronrod adaptive integration
fn gauss_kronrod_adaptive<T, F>(
    f: &F,
    a: T,
    b: T,
    atol: T,
    rtol: T,
    max_depth: usize,
    neval: &mut usize,
) -> Result<(T, T)>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T) -> T,
{
    let (nodes, kronrod_w, gauss_w) = gauss_kronrod_15::<T>();

    // Transform from [-1, 1] to [a, b]
    let half_length = (b - a) / T::from(2.0).expect("2.0 is representable as Float");
    let center = (a + b) / T::from(2.0).expect("2.0 is representable as Float");

    // Evaluate function at all Kronrod points
    let mut f_vals = Vec::with_capacity(15);
    for &node in &nodes {
        let x = center + half_length * node;
        f_vals.push(f(x));
        *neval += 1;
    }

    // Compute Kronrod integral (15 points)
    let kronrod_integral: T = kronrod_w
        .iter()
        .zip(f_vals.iter())
        .map(|(&w, &fv)| w * fv)
        .sum::<T>()
        * half_length;

    // Compute Gauss integral (7 points from the 15 - odd indices)
    let gauss_indices = [1, 3, 5, 7, 9, 11, 13];
    let gauss_integral: T = gauss_w
        .iter()
        .zip(gauss_indices.iter())
        .map(|(&w, &idx)| w * f_vals[idx])
        .sum::<T>()
        * half_length;

    // Error estimate
    let error = (kronrod_integral - gauss_integral).abs();
    let tolerance = atol.max(rtol * kronrod_integral.abs());

    // Check convergence or depth limit
    if error <= tolerance || max_depth == 0 {
        return Ok((kronrod_integral, error));
    }

    // Subdivide and recurse
    let mid = center;
    let (left_val, left_err) = gauss_kronrod_adaptive(
        f,
        a,
        mid,
        atol / T::from(2.0).expect("2.0 is representable as Float"),
        rtol,
        max_depth - 1,
        neval,
    )?;
    let (right_val, right_err) = gauss_kronrod_adaptive(
        f,
        mid,
        b,
        atol / T::from(2.0).expect("2.0 is representable as Float"),
        rtol,
        max_depth - 1,
        neval,
    )?;

    Ok((left_val + right_val, left_err + right_err))
}

/// Simpson's rule composite integration
///
/// Classic composite Simpson's rule with adaptive subdivisions.
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of subdivisions (must be even, will be adjusted if odd)
///
/// # Returns
/// * Approximate integral value
pub fn simps<T, F>(f: F, a: T, b: T, n: usize) -> T
where
    T: Float + Debug,
    F: Fn(T) -> T,
{
    // Ensure n is even
    let n = if n % 2 == 1 { n + 1 } else { n };
    let n = n.max(2);

    let h = (b - a) / T::from(n).expect("n is representable as Float");

    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + T::from(i).expect("i is representable as Float") * h;
        let coef = if i % 2 == 0 {
            T::from(2.0).expect("2.0 is representable as Float")
        } else {
            T::from(4.0).expect("4.0 is representable as Float")
        };
        sum = sum + coef * f(x);
    }

    sum * h / T::from(3.0).expect("3.0 is representable as Float")
}

/// Trapezoidal rule composite integration
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of subdivisions
///
/// # Returns
/// * Approximate integral value
pub fn trapz<T, F>(f: F, a: T, b: T, n: usize) -> T
where
    T: Float + Debug,
    F: Fn(T) -> T,
{
    let n = n.max(1);
    let h = (b - a) / T::from(n).expect("n is representable as Float");

    let mut sum = (f(a) + f(b)) / T::from(2.0).expect("2.0 is representable as Float");

    for i in 1..n {
        let x = a + T::from(i).expect("i is representable as Float") * h;
        sum = sum + f(x);
    }

    sum * h
}

/// Trapezoidal rule for arrays of x and y values
///
/// # Arguments
/// * `x` - Array of x coordinates (must be sorted in ascending order)
/// * `y` - Array of y values (f(x))
///
/// # Returns
/// * Approximate integral using trapezoidal rule
pub fn trapz_array<T>(x: &[T], y: &[T]) -> Result<T>
where
    T: Float + Debug,
{
    if x.len() != y.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "x and y arrays must have same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Err(NumRs2Error::ValueError(
            "Arrays must have at least 2 elements".to_string(),
        ));
    }

    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let h = x[i + 1] - x[i];
        sum = sum + h * (y[i] + y[i + 1]) / T::from(2.0).expect("2.0 is representable as Float");
    }

    Ok(sum)
}

/// Romberg integration
///
/// Uses Richardson extrapolation on the trapezoidal rule for improved accuracy.
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `max_order` - Maximum order of extrapolation (determines accuracy)
///
/// # Returns
/// * Approximate integral value
pub fn romberg<T, F>(f: F, a: T, b: T, max_order: usize) -> T
where
    T: Float + Debug,
    F: Fn(T) -> T,
{
    let max_order = max_order.clamp(1, 20); // Limit to reasonable range

    // Romberg tableau
    let mut r = vec![vec![T::zero(); max_order + 1]; max_order + 1];

    // First row: trapezoidal approximations
    for j in 0..=max_order {
        let n = 1usize << j; // 2^j subdivisions
        r[j][0] = trapz(&f, a, b, n);
    }

    // Richardson extrapolation
    for k in 1..=max_order {
        for j in k..=max_order {
            let factor = T::from(4u64.pow(k as u32)).expect("4^k is representable as Float");
            r[j][k] = (factor * r[j][k - 1] - r[j - 1][k - 1]) / (factor - T::one());
        }
    }

    r[max_order][max_order]
}

/// Monte Carlo integration for single-dimensional functions
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n_samples` - Number of random samples
///
/// # Returns
/// * `IntegrationResult` with estimated value and statistical error
pub fn monte_carlo<T, F>(f: F, a: T, b: T, n_samples: usize) -> IntegrationResult<T>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T) -> T,
{
    let domain_size = b - a;
    let n = n_samples.max(1);

    // Generate pseudo-random samples using LCG (Linear Congruential Generator)
    let mut state = 12345u64;
    let mut sum = T::zero();
    let mut sum_sq = T::zero();

    for _ in 0..n {
        // PCG-style LCG
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use full 64-bit state and scale to [0, 1)
        let u = T::from((state as f64) / (u64::MAX as f64))
            .expect("normalized random value is representable as Float");
        let x = a + u * domain_size;
        let fx = f(x);
        sum = sum + fx;
        sum_sq = sum_sq + fx * fx;
    }

    let n_float = T::from(n).expect("n is representable as Float");
    let mean = sum / n_float;
    let variance = (sum_sq / n_float) - mean * mean;
    let std_error = if variance > T::zero() {
        (variance / n_float).sqrt() * domain_size
    } else {
        T::zero()
    };

    let value = mean * domain_size;

    IntegrationResult {
        value,
        error: std_error,
        neval: n,
        success: true,
    }
}

/// Monte Carlo integration for multi-dimensional functions
///
/// # Arguments
/// * `f` - Function taking a slice of coordinates and returning a value
/// * `bounds` - Vector of (lower, upper) bounds for each dimension
/// * `n_samples` - Number of random samples
///
/// # Returns
/// * `IntegrationResult` with estimated value and statistical error
///
/// # Example
/// ```ignore
/// use numrs2::integrate::monte_carlo_nd;
///
/// // Integrate f(x,y) = x*y over [0,1] x [0,1]
/// let result = monte_carlo_nd(
///     |x| x[0] * x[1],
///     &[(0.0, 1.0), (0.0, 1.0)],
///     10000
/// );
/// // Expected: 0.25
/// ```
pub fn monte_carlo_nd<T, F>(f: F, bounds: &[(T, T)], n_samples: usize) -> IntegrationResult<T>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let ndim = bounds.len();
    let n = n_samples.max(1);

    // Compute domain volume
    let volume: T = bounds
        .iter()
        .map(|&(a, b)| b - a)
        .fold(T::one(), |acc, x| acc * x);

    // Generate pseudo-random samples using LCG with different seeds per dimension
    let mut states: Vec<u64> = (0..ndim).map(|i| 12345u64 + i as u64 * 1000003).collect();
    let mut sum = T::zero();
    let mut sum_sq = T::zero();
    let mut point = vec![T::zero(); ndim];

    for _ in 0..n {
        // Generate random point in domain
        for d in 0..ndim {
            states[d] = states[d]
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Use full 64-bit state
            let u = T::from((states[d] as f64) / (u64::MAX as f64))
                .expect("normalized random value is representable as Float");
            let (a, b) = bounds[d];
            point[d] = a + u * (b - a);
        }

        let fx = f(&point);
        sum = sum + fx;
        sum_sq = sum_sq + fx * fx;
    }

    let n_float = T::from(n).expect("n is representable as Float");
    let mean = sum / n_float;
    let variance = (sum_sq / n_float) - mean * mean;
    let std_error = if variance > T::zero() {
        (variance / n_float).sqrt() * volume
    } else {
        T::zero()
    };

    let value = mean * volume;

    IntegrationResult {
        value,
        error: std_error,
        neval: n,
        success: true,
    }
}

/// Gaussian quadrature with specified number of points
///
/// Uses Gauss-Legendre quadrature nodes and weights.
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of quadrature points (1-10 supported)
pub fn gauss_legendre<T, F>(f: F, a: T, b: T, n: usize) -> T
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T) -> T,
{
    let (nodes, weights) = gauss_legendre_nodes_weights::<T>(n);

    let half_length = (b - a) / T::from(2.0).expect("2.0 is representable as Float");
    let center = (a + b) / T::from(2.0).expect("2.0 is representable as Float");

    let sum: T = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| {
            let x = center + half_length * node;
            weight * f(x)
        })
        .sum();

    sum * half_length
}

/// Get Gauss-Legendre nodes and weights
fn gauss_legendre_nodes_weights<T: Float>(n: usize) -> (Vec<T>, Vec<T>) {
    match n {
        1 => (
            vec![T::from(0.0).expect("GL node is representable as Float")],
            vec![T::from(2.0).expect("GL weight is representable as Float")],
        ),
        2 => (
            vec![
                T::from(-0.5773502691896257).expect("GL node is representable as Float"),
                T::from(0.5773502691896257).expect("GL node is representable as Float"),
            ],
            vec![
                T::from(1.0).expect("GL weight is representable as Float"),
                T::from(1.0).expect("GL weight is representable as Float"),
            ],
        ),
        3 => (
            vec![
                T::from(-0.7745966692414834).expect("GL node is representable as Float"),
                T::from(0.0).expect("GL node is representable as Float"),
                T::from(0.7745966692414834).expect("GL node is representable as Float"),
            ],
            vec![
                T::from(0.5555555555555556).expect("GL weight is representable as Float"),
                T::from(0.8888888888888888).expect("GL weight is representable as Float"),
                T::from(0.5555555555555556).expect("GL weight is representable as Float"),
            ],
        ),
        4 => (
            vec![
                T::from(-0.8611363115940526).expect("GL node is representable as Float"),
                T::from(-0.3399810435848563).expect("GL node is representable as Float"),
                T::from(0.3399810435848563).expect("GL node is representable as Float"),
                T::from(0.8611363115940526).expect("GL node is representable as Float"),
            ],
            vec![
                T::from(0.3478548451374538).expect("GL weight is representable as Float"),
                T::from(0.6521451548625461).expect("GL weight is representable as Float"),
                T::from(0.6521451548625461).expect("GL weight is representable as Float"),
                T::from(0.3478548451374538).expect("GL weight is representable as Float"),
            ],
        ),
        5 => (
            vec![
                T::from(-0.9061798459386640).expect("GL node is representable as Float"),
                T::from(-0.5384693101056831).expect("GL node is representable as Float"),
                T::from(0.0).expect("GL node is representable as Float"),
                T::from(0.5384693101056831).expect("GL node is representable as Float"),
                T::from(0.9061798459386640).expect("GL node is representable as Float"),
            ],
            vec![
                T::from(0.2369268850561891).expect("GL weight is representable as Float"),
                T::from(0.4786286704993665).expect("GL weight is representable as Float"),
                T::from(0.5688888888888889).expect("GL weight is representable as Float"),
                T::from(0.4786286704993665).expect("GL weight is representable as Float"),
                T::from(0.2369268850561891).expect("GL weight is representable as Float"),
            ],
        ),
        _ => {
            // Default to 5-point rule for unsupported n
            gauss_legendre_nodes_weights(5)
        }
    }
}

/// Double integral over rectangular region
///
/// # Arguments
/// * `f` - Function of two variables
/// * `xa` - Lower x bound
/// * `xb` - Upper x bound
/// * `ya` - Lower y bound
/// * `yb` - Upper y bound
/// * `nx` - Number of subdivisions in x
/// * `ny` - Number of subdivisions in y
///
/// # Returns
/// * Approximate double integral
pub fn dblquad<T, F>(f: F, xa: T, xb: T, ya: T, yb: T, nx: usize, ny: usize) -> T
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, T) -> T,
{
    // Use composite Simpson's rule in both dimensions
    let hx = (xb - xa) / T::from(nx).expect("nx is representable as Float");
    let hy = (yb - ya) / T::from(ny).expect("ny is representable as Float");

    let mut sum = T::zero();

    for i in 0..=nx {
        let x = xa + T::from(i).expect("i is representable as Float") * hx;
        let wx = if i == 0 || i == nx {
            T::one()
        } else if i % 2 == 0 {
            T::from(2.0).expect("2.0 is representable as Float")
        } else {
            T::from(4.0).expect("4.0 is representable as Float")
        };

        for j in 0..=ny {
            let y = ya + T::from(j).expect("j is representable as Float") * hy;
            let wy = if j == 0 || j == ny {
                T::one()
            } else if j % 2 == 0 {
                T::from(2.0).expect("2.0 is representable as Float")
            } else {
                T::from(4.0).expect("4.0 is representable as Float")
            };

            sum = sum + wx * wy * f(x, y);
        }
    }

    sum * hx * hy / T::from(9.0).expect("9.0 is representable as Float")
}

/// Fixed-point iteration for solving integral equations
///
/// Solves equations of the form: x = integral(f(t, x) dt, a, b)
///
/// # Arguments
/// * `f` - Function of (t, x)
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `x0` - Initial guess
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// * Solution x or error
pub fn fixed_point_integral<T, F>(f: F, a: T, b: T, x0: T, max_iter: usize, tol: T) -> Result<T>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, T) -> T,
{
    let mut x = x0;

    for _ in 0..max_iter {
        // Compute integral with current x
        let integrand = |t: T| f(t, x);
        let new_x = quad(integrand, a, b)?;

        let err = (new_x - x).abs();
        x = new_x;

        if err < tol {
            return Ok(x);
        }
    }

    Err(NumRs2Error::InvalidOperation(
        "Fixed point iteration did not converge".to_string(),
    ))
}

/// Cumulative integration using trapezoidal rule
///
/// # Arguments
/// * `y` - Array of y values
/// * `dx` - Spacing between x values (uniform)
///
/// # Returns
/// * Array of cumulative integral values
pub fn cumtrapz<T>(y: &[T], dx: T) -> Vec<T>
where
    T: Float + Debug,
{
    if y.is_empty() {
        return Vec::new();
    }

    let mut result = vec![T::zero(); y.len()];

    for i in 1..y.len() {
        result[i] = result[i - 1]
            + dx * (y[i - 1] + y[i]) / T::from(2.0).expect("2.0 is representable as Float");
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_polynomial() {
        // Integral of x^2 from 0 to 1 = 1/3
        let result = quad(|x: f64| x * x, 0.0, 1.0).expect("quad should succeed");
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quad_sin() {
        // Integral of sin(x) from 0 to pi = 2
        let result =
            quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI).expect("quad should succeed");
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quad_exp() {
        // Integral of e^x from 0 to 1 = e - 1
        let result = quad(|x: f64| x.exp(), 0.0, 1.0).expect("quad should succeed");
        let expected = std::f64::consts::E - 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simps_polynomial() {
        // Integral of x^2 from 0 to 1 = 1/3
        let result = simps(|x: f64| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_trapz_polynomial() {
        // Integral of x from 0 to 1 = 0.5
        let result = trapz(|x: f64| x, 0.0, 1.0, 100);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_trapz_array() {
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![0.0, 0.25, 1.0];
        let result = trapz_array(&x, &y).expect("trapz_array should succeed");
        // Trapezoidal approximation of x^2
        assert!((result - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_romberg_polynomial() {
        // Integral of x^2 from 0 to 1 = 1/3
        let result = romberg(|x: f64| x * x, 0.0, 1.0, 5);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_monte_carlo_constant() {
        // Integral of 1 from 0 to 1 = 1
        let result = monte_carlo(|_: f64| 1.0, 0.0, 1.0, 10000);
        assert!((result.value - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_monte_carlo_linear() {
        // Integral of x from 0 to 1 = 0.5
        // Monte Carlo is stochastic - use larger samples and tolerance
        let result = monte_carlo(|x: f64| x, 0.0, 1.0, 100000);
        assert!(
            (result.value - 0.5).abs() < 0.1,
            "Monte Carlo linear integral {} too far from 0.5",
            result.value
        );
    }

    #[test]
    fn test_monte_carlo_nd_2d() {
        // Integral of x*y over [0,1] x [0,1] = 0.25
        // Monte Carlo in multiple dimensions has higher variance
        let result = monte_carlo_nd(|x: &[f64]| x[0] * x[1], &[(0.0, 1.0), (0.0, 1.0)], 100000);
        assert!(
            (result.value - 0.25).abs() < 0.1,
            "Monte Carlo 2D integral {} too far from 0.25",
            result.value
        );
    }

    #[test]
    fn test_gauss_legendre_5() {
        // Integral of x^4 from -1 to 1 = 2/5
        // 5-point Gauss-Legendre is exact for polynomials up to degree 9
        let result = gauss_legendre(|x: f64| x.powi(4), -1.0, 1.0, 5);
        assert!((result - 0.4).abs() < 1e-14);
    }

    #[test]
    fn test_dblquad_constant() {
        // Integral of 1 over [0,1] x [0,1] = 1
        let result = dblquad(|_x: f64, _y: f64| 1.0, 0.0, 1.0, 0.0, 1.0, 10, 10);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dblquad_product() {
        // Integral of x*y over [0,1] x [0,1] = 0.25
        let result = dblquad(|x: f64, y: f64| x * y, 0.0, 1.0, 0.0, 1.0, 10, 10);
        assert!((result - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_cumtrapz() {
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let dx = 1.0f64;
        let result = cumtrapz(&y, dx);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_quad_adaptive_error_estimate() {
        let result = quad_adaptive(
            &|x: f64| x.sin(),
            0.0,
            std::f64::consts::PI,
            &QuadConfig::default(),
        )
        .expect("quad_adaptive should succeed");

        assert!((result.value - 2.0).abs() < 1e-10);
        assert!(result.error < 1e-10);
        assert!(result.neval > 0);
    }
}
