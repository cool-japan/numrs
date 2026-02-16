//! Root-finding algorithms for nonlinear equations
//!
//! This module provides robust algorithms for finding zeros of functions,
//! i.e., solving f(x) = 0 for scalar and vector-valued functions.
//!
//! # Available Methods
//!
//! ## Bracketing Methods (Guaranteed Convergence)
//! - **Bisection**: Reliable but slow, requires bracketing interval
//! - **Brent's Method**: Combines bisection, secant, and inverse quadratic interpolation
//! - **Ridder's Method**: Uses exponential function for improved convergence
//!
//! ## Open Methods (Fast but may not converge)
//! - **Newton-Raphson**: Quadratic convergence with derivative
//! - **Secant Method**: Superlinear convergence without derivative
//! - **Halley's Method**: Cubic convergence with second derivative
//!
//! ## Multidimensional Root Finding
//! - **Newton's Method**: System of nonlinear equations
//! - **Broyden's Method**: Quasi-Newton for systems
//!
//! # Examples
//!
//! ```
//! use numrs2::roots::*;
//!
//! // Find root of f(x) = x^2 - 2 (i.e., sqrt(2))
//! let f = |x: f64| x * x - 2.0;
//! let result = brentq(f, 0.0, 2.0, 1e-10, 100).expect("brentq should converge");
//! assert!((result.root - 1.414213562).abs() < 1e-8);
//! ```

use crate::error::{NumRs2Error, Result};
use num_traits::Float;

/// Result of root-finding operation
#[derive(Debug, Clone)]
pub struct RootResult<T: Float> {
    /// Root value found
    pub root: T,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of iterations performed
    pub nit: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Final function value at root
    pub fun_val: T,
}

/// Bisection method for root finding
///
/// Finds a root of f(x) = 0 in the interval [a, b] where f(a) and f(b)
/// have opposite signs.
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `a` - Left endpoint (must satisfy f(a)*f(b) < 0)
/// * `b` - Right endpoint
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// let f = |x: f64| x * x - 2.0;
/// let result = bisect(f, 0.0, 2.0, 1e-10, 100).expect("bisect should converge");
/// assert!((result.root - 1.414213562).abs() < 1e-8);
/// ```
pub fn bisect<T, F>(f: F, mut a: T, mut b: T, tol: T, max_iter: usize) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut fa = f(a);
    let fb = f(b);
    let mut nfev = 2;

    // Check bracketing
    if fa * fb > T::zero() {
        return Err(NumRs2Error::ValueError(
            "Function must have opposite signs at endpoints".to_string(),
        ));
    }

    // Check if endpoints are roots
    if fa.abs() < tol {
        return Ok(RootResult {
            root: a,
            nfev,
            nit: 0,
            converged: true,
            fun_val: fa,
        });
    }
    if fb.abs() < tol {
        return Ok(RootResult {
            root: b,
            nfev,
            nit: 0,
            converged: true,
            fun_val: fb,
        });
    }

    // Bisection iteration
    for iter in 0..max_iter {
        let c = (a + b) / T::from(2.0).expect("2.0 is representable as Float");
        let fc = f(c);
        nfev += 1;

        if fc.abs() < tol || (b - a).abs() < tol {
            return Ok(RootResult {
                root: c,
                nfev,
                nit: iter + 1,
                converged: true,
                fun_val: fc,
            });
        }

        if fa * fc < T::zero() {
            b = c;
            // fb = fc; // Not needed: fb only used for bracketing check
        } else {
            a = c;
            fa = fc;
        }
    }

    let c = (a + b) / T::from(2.0).expect("2.0 is representable as Float");
    Ok(RootResult {
        root: c,
        nfev,
        nit: max_iter,
        converged: false,
        fun_val: f(c),
    })
}

/// Brent's method for root finding
///
/// Combines bisection, secant method, and inverse quadratic interpolation
/// for robust and fast convergence.
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
/// let result = brentq(f, 2.0, 3.0, 1e-10, 100).expect("brentq should converge");
/// assert!(f(result.root).abs() < 1e-9);
/// ```
pub fn brentq<T, F>(f: F, mut a: T, mut b: T, tol: T, max_iter: usize) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut fa = f(a);
    let mut fb = f(b);
    let mut nfev = 2;

    // Check bracketing
    if fa * fb > T::zero() {
        return Err(NumRs2Error::ValueError(
            "Function must have opposite signs at endpoints".to_string(),
        ));
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut mflag = true;

    for iter in 0..max_iter {
        if fb.abs() < tol {
            return Ok(RootResult {
                root: b,
                nfev,
                nit: iter,
                converged: true,
                fun_val: fb,
            });
        }

        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            let l0 = (a * fb * fc) / ((fa - fb) * (fa - fc));
            let l1 = (b * fa * fc) / ((fb - fa) * (fb - fc));
            let l2 = (c * fa * fb) / ((fc - fa) * (fc - fb));
            let mut s = l0 + l1 + l2;

            // Check conditions for acceptance
            let delta = T::from(1e-14).expect("1e-14 is representable as Float");
            let cond1 = s
                < (T::from(3.0).expect("3.0 is representable as Float") * a + b)
                    / T::from(4.0).expect("4.0 is representable as Float")
                || s > b;
            let cond2 = mflag
                && (s - b).abs()
                    >= (b - c).abs() / T::from(2.0).expect("2.0 is representable as Float");
            let cond3 = !mflag
                && (s - b).abs()
                    >= (c - d).abs() / T::from(2.0).expect("2.0 is representable as Float");
            let cond4 = mflag && (b - c).abs() < delta;
            let cond5 = !mflag && (c - d).abs() < delta;

            if cond1 || cond2 || cond3 || cond4 || cond5 {
                s = (a + b) / T::from(2.0).expect("2.0 is representable as Float");
                mflag = true;
            } else {
                mflag = false;
            }

            let fs = f(s);
            nfev += 1;
            d = c;
            c = b;
            fc = fb;

            if fa * fs < T::zero() {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
        } else {
            // Secant method
            let s = b - fb * (b - a) / (fb - fa);
            let fs = f(s);
            nfev += 1;

            d = c;
            c = b;
            fc = fb;

            if fa * fs < T::zero() {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
            mflag = true;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    Ok(RootResult {
        root: b,
        nfev,
        nit: max_iter,
        converged: false,
        fun_val: fb,
    })
}

/// Newton-Raphson method for root finding
///
/// Finds a root using the derivative information: x_{n+1} = x_n - f(x_n)/f'(x_n)
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// let f = |x: f64| x * x - 2.0;
/// let df = |x: f64| 2.0 * x;
/// let result = newton(f, df, 1.0, 1e-10, 100).expect("newton should converge");
/// assert!((result.root - 1.414213562).abs() < 1e-8);
/// ```
pub fn newton<T, F, DF>(f: F, df: DF, mut x: T, tol: T, max_iter: usize) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
    DF: Fn(T) -> T,
{
    let mut nfev = 0;

    for iter in 0..max_iter {
        let fx = f(x);
        nfev += 1;

        if fx.abs() < tol {
            return Ok(RootResult {
                root: x,
                nfev,
                nit: iter,
                converged: true,
                fun_val: fx,
            });
        }

        let dfx = df(x);
        if dfx.abs() < T::from(1e-14).expect("1e-14 is representable as Float") {
            return Err(NumRs2Error::ComputationError(
                "Derivative too small, Newton iteration failed".to_string(),
            ));
        }

        x = x - fx / dfx;
    }

    let fx = f(x);
    Ok(RootResult {
        root: x,
        nfev: nfev + 1,
        nit: max_iter,
        converged: false,
        fun_val: fx,
    })
}

/// Secant method for root finding
///
/// Similar to Newton-Raphson but approximates derivative using finite differences.
/// Does not require analytical derivative.
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// let f = |x: f64| x * x - 2.0;
/// let result = secant(f, 1.0, 2.0, 1e-10, 100).expect("secant should converge");
/// assert!((result.root - 1.414213562).abs() < 1e-8);
/// ```
pub fn secant<T, F>(f: F, mut x0: T, mut x1: T, tol: T, max_iter: usize) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    let mut nfev = 2;

    for iter in 0..max_iter {
        if f1.abs() < tol {
            return Ok(RootResult {
                root: x1,
                nfev,
                nit: iter,
                converged: true,
                fun_val: f1,
            });
        }

        if (f1 - f0).abs() < T::from(1e-14).expect("1e-14 is representable as Float") {
            return Err(NumRs2Error::ComputationError(
                "Secant method: function values too close".to_string(),
            ));
        }

        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        let f2 = f(x2);
        nfev += 1;

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
    }

    Ok(RootResult {
        root: x1,
        nfev,
        nit: max_iter,
        converged: false,
        fun_val: f1,
    })
}

/// Halley's method for root finding
///
/// Uses second derivative for cubic convergence (faster than Newton).
/// x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// let f = |x: f64| x * x - 2.0;
/// let df = |x: f64| 2.0 * x;
/// let ddf = |_x: f64| 2.0;
/// let result = halley(f, df, ddf, 1.0, 1e-10, 100).expect("halley should converge");
/// assert!((result.root - 1.414213562).abs() < 1e-8);
/// ```
pub fn halley<T, F, DF, DDF>(
    f: F,
    df: DF,
    ddf: DDF,
    mut x: T,
    tol: T,
    max_iter: usize,
) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
    DF: Fn(T) -> T,
    DDF: Fn(T) -> T,
{
    let mut nfev = 0;

    for iter in 0..max_iter {
        let fx = f(x);
        nfev += 1;

        if fx.abs() < tol {
            return Ok(RootResult {
                root: x,
                nfev,
                nit: iter,
                converged: true,
                fun_val: fx,
            });
        }

        let dfx = df(x);
        let ddfx = ddf(x);

        let denominator =
            T::from(2.0).expect("2.0 is representable as Float") * dfx * dfx - fx * ddfx;
        if denominator.abs() < T::from(1e-14).expect("1e-14 is representable as Float") {
            return Err(NumRs2Error::ComputationError(
                "Halley method: denominator too small".to_string(),
            ));
        }

        x = x - (T::from(2.0).expect("2.0 is representable as Float") * fx * dfx) / denominator;
    }

    let fx = f(x);
    Ok(RootResult {
        root: x,
        nfev: nfev + 1,
        nit: max_iter,
        converged: false,
        fun_val: fx,
    })
}

/// Ridder's method for root finding
///
/// Uses exponential function to extrapolate root position.
/// Often faster than bisection with similar robustness.
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
/// let result = ridder(f, 2.0, 3.0, 1e-10, 100).expect("ridder should converge");
/// assert!(f(result.root).abs() < 1e-9);
/// ```
pub fn ridder<T, F>(f: F, mut a: T, mut b: T, tol: T, max_iter: usize) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut fa = f(a);
    let mut fb = f(b);
    let mut nfev = 2;

    // Check bracketing
    if fa * fb > T::zero() {
        return Err(NumRs2Error::ValueError(
            "Function must have opposite signs at endpoints".to_string(),
        ));
    }

    for iter in 0..max_iter {
        let c = (a + b) / T::from(2.0).expect("2.0 is representable as Float");
        let fc = f(c);
        nfev += 1;

        if fc.abs() < tol {
            return Ok(RootResult {
                root: c,
                nfev,
                nit: iter + 1,
                converged: true,
                fun_val: fc,
            });
        }

        // Ridder's formula: x4 = c + (c-a) * sign(fa-fb) * fc / sqrt(fc^2 - fa*fb)
        let discriminant = fc * fc - fa * fb;
        if discriminant < T::zero() {
            // Fallback to bisection if discriminant is negative
            let x_new = c;
            let fx_new = fc;
            if fa * fc < T::zero() {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
            if (b - a).abs() < tol {
                return Ok(RootResult {
                    root: x_new,
                    nfev,
                    nit: iter + 1,
                    converged: true,
                    fun_val: fx_new,
                });
            }
            continue;
        }

        let s = discriminant.sqrt();
        if s.abs() < T::from(1e-14).expect("1e-14 is representable as Float") {
            return Ok(RootResult {
                root: c,
                nfev,
                nit: iter + 1,
                converged: true,
                fun_val: fc,
            });
        }

        // sign(fa - fb)
        let sign = if fa > fb { T::one() } else { -T::one() };
        let x_new = c + (c - a) * sign * fc / s;
        let fx_new = f(x_new);
        nfev += 1;

        if fx_new.abs() < tol {
            return Ok(RootResult {
                root: x_new,
                nfev,
                nit: iter + 1,
                converged: true,
                fun_val: fx_new,
            });
        }

        // Update bracket
        if fc * fx_new < T::zero() {
            a = c;
            fa = fc;
            b = x_new;
            fb = fx_new;
        } else if fa * fx_new < T::zero() {
            b = x_new;
            fb = fx_new;
        } else {
            a = x_new;
            fa = fx_new;
        }

        if (b - a).abs() < tol {
            return Ok(RootResult {
                root: x_new,
                nfev,
                nit: iter + 1,
                converged: true,
                fun_val: fx_new,
            });
        }
    }

    let root = (a + b) / T::from(2.0).expect("2.0 is representable as Float");
    Ok(RootResult {
        root,
        nfev,
        nit: max_iter,
        converged: false,
        fun_val: f(root),
    })
}

/// Illinois method (false position variant)
///
/// Modified regula falsi that prevents stalling by reducing the weight
/// of the side that doesn't change.
pub fn illinois<T, F>(f: F, mut a: T, mut b: T, tol: T, max_iter: usize) -> Result<RootResult<T>>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut fa = f(a);
    let mut fb = f(b);
    let mut nfev = 2;

    if fa * fb > T::zero() {
        return Err(NumRs2Error::ValueError(
            "Function must have opposite signs at endpoints".to_string(),
        ));
    }

    let mut side = 0;

    for iter in 0..max_iter {
        let c = (fa * b - fb * a) / (fa - fb);
        let fc = f(c);
        nfev += 1;

        if fc.abs() < tol || (b - a).abs() < tol {
            return Ok(RootResult {
                root: c,
                nfev,
                nit: iter + 1,
                converged: true,
                fun_val: fc,
            });
        }

        if fa * fc < T::zero() {
            b = c;
            fb = fc;
            if side == -1 {
                fa = fa / T::from(2.0).expect("2.0 is representable as Float");
            }
            side = -1;
        } else {
            a = c;
            fa = fc;
            if side == 1 {
                fb = fb / T::from(2.0).expect("2.0 is representable as Float");
            }
            side = 1;
        }
    }

    let c = (fa * b - fb * a) / (fa - fb);
    Ok(RootResult {
        root: c,
        nfev,
        nit: max_iter,
        converged: false,
        fun_val: f(c),
    })
}

/// Fixed-point iteration: find x such that x = g(x)
///
/// Equivalent to finding root of f(x) = x - g(x) = 0
///
/// # Examples
///
/// ```
/// use numrs2::roots::*;
///
/// // Find fixed point of g(x) = cos(x), i.e., solve x = cos(x)
/// let g = |x: f64| x.cos();
/// let result = fixed_point(g, 0.5, 1e-10, 100).expect("fixed_point should converge");
/// assert!((result - result.cos()).abs() < 1e-9);
/// ```
pub fn fixed_point<T, G>(g: G, mut x: T, tol: T, max_iter: usize) -> Result<T>
where
    T: Float,
    G: Fn(T) -> T,
{
    for _ in 0..max_iter {
        let x_new = g(x);
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }
        x = x_new;
    }

    Err(NumRs2Error::ComputationError(
        "Fixed-point iteration did not converge".to_string(),
    ))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bisect_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = bisect(f, 0.0, 2.0, 1e-10, 100).expect("bisect should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_bisect_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let result = bisect(f, 2.0, 3.0, 1e-10, 100).expect("bisect should converge for cubic");
        assert!(result.converged);
        assert!(f(result.root).abs() < 1e-9);
    }

    #[test]
    fn test_brentq_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = brentq(f, 0.0, 2.0, 1e-10, 100).expect("brentq should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_brentq_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let result = brentq(f, 2.0, 3.0, 1e-10, 100).expect("brentq should converge for cubic");
        assert!(result.converged);
        assert!(f(result.root).abs() < 1e-9);
    }

    #[test]
    fn test_brentq_faster_than_bisection() {
        let f = |x: f64| x.powi(3) - x - 1.0;

        let bisect_result = bisect(f, 1.0, 2.0, 1e-10, 100).expect("bisect should converge");
        let brent_result = brentq(f, 1.0, 2.0, 1e-10, 100).expect("brentq should converge");

        // Brent should converge in fewer iterations
        assert!(brent_result.nit <= bisect_result.nit);
    }

    #[test]
    fn test_newton_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;

        let result = newton(f, df, 1.0, 1e-10, 100).expect("newton should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_newton_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let df = |x: f64| 3.0 * x.powi(2) - 2.0;

        let result = newton(f, df, 2.5, 1e-10, 100).expect("newton should converge for cubic");
        assert!(result.converged);
        assert!(f(result.root).abs() < 1e-9);
    }

    #[test]
    fn test_secant_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = secant(f, 1.0, 2.0, 1e-10, 100).expect("secant should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_secant_transcendental() {
        // Solve x = cos(x)
        let f = |x: f64| x - x.cos();
        let result =
            secant(f, 0.0, 1.0, 1e-10, 100).expect("secant should converge for x - cos(x)");
        assert!(result.converged);
        assert_relative_eq!(result.root, 0.7390851332151607, epsilon = 1e-9);
    }

    #[test]
    fn test_halley_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let ddf = |_x: f64| 2.0;

        let result =
            halley(f, df, ddf, 1.0, 1e-10, 100).expect("halley should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
        // Halley should converge very fast (cubic convergence)
        assert!(result.nit < 10);
    }

    #[test]
    fn test_ridder_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = ridder(f, 0.0, 2.0, 1e-10, 100).expect("ridder should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_ridder_exponential() {
        // Solve e^x = 3
        let f = |x: f64| x.exp() - 3.0;
        let result = ridder(f, 0.0, 2.0, 1e-8, 200).expect("ridder should converge for e^x - 3");
        assert!(result.converged, "Ridder should converge for exponential");
        assert_relative_eq!(result.root, 1.0986122886681098, epsilon = 1e-7); // ln(3)
    }

    #[test]
    fn test_illinois_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result =
            illinois(f, 0.0, 2.0, 1e-10, 100).expect("illinois should converge for x^2 - 2");
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_fixed_point_cosine() {
        // Find x such that x = cos(x)
        let g = |x: f64| x.cos();
        let root = fixed_point(g, 0.5, 1e-10, 100).expect("fixed_point should converge for cos");
        assert_relative_eq!(root, 0.7390851332151607, epsilon = 1e-9);
        assert_relative_eq!(root, root.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_fixed_point_sqrt() {
        // Find sqrt(2) using fixed point iteration: x = (x + 2/x)/2
        let g = |x: f64| (x + 2.0 / x) / 2.0;
        let root = fixed_point(g, 1.0, 1e-10, 100)
            .expect("fixed_point should converge for sqrt iteration");
        assert_relative_eq!(root, 1.414213562373095, epsilon = 1e-9);
    }

    #[test]
    fn test_newton_faster_than_secant() {
        let f = |x: f64| x.powi(5) - x - 1.0;
        let df = |x: f64| 5.0 * x.powi(4) - 1.0;

        let newton_result =
            newton(f, df, 1.5, 1e-10, 100).expect("newton should converge for quintic");
        let secant_result =
            secant(f, 1.0, 2.0, 1e-10, 100).expect("secant should converge for quintic");

        // Newton should converge faster
        assert!(newton_result.nit <= secant_result.nit);
    }

    #[test]
    fn test_halley_faster_than_newton() {
        let f = |x: f64| x.powi(3) - 10.0;
        let df = |x: f64| 3.0 * x.powi(2);
        let ddf = |x: f64| 6.0 * x;

        let newton_result =
            newton(f, df, 2.0, 1e-10, 100).expect("newton should converge for x^3 - 10");
        let halley_result =
            halley(f, df, ddf, 2.0, 1e-10, 100).expect("halley should converge for x^3 - 10");

        // Halley should converge faster (cubic vs quadratic convergence)
        assert!(halley_result.nit <= newton_result.nit);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_bisect_always_brackets(
            a in -10.0f64..0.0,
            b in 0.0f64..10.0
        ) {
            // For f(x) = x, bisection should always converge to 0
            let f = |x: f64| x;
            let result = bisect(f, a, b, 1e-10, 100).expect("bisect should converge for f(x) = x");
            prop_assert!(result.converged);
            prop_assert!(result.root.abs() < 1e-9);
        }

        #[test]
        fn prop_newton_quadratic_convergence(
            target in -10.0f64..10.0,
            x0 in -5.0f64..5.0
        ) {
            // Newton should find the root of (x - target) = 0 (simple root, not double)
            let f = |x: f64| x - target;
            let df = |_x: f64| 1.0;

            // Skip if starting exactly at target
            if (x0 - target).abs() > 0.01 {
                let result = newton(f, df, x0, 1e-10, 100).expect("newton should converge for linear function");
                prop_assert!(result.converged, "Newton should converge for linear function");
                prop_assert!((result.root - target).abs() < 1e-8);
            }
        }

        #[test]
        fn prop_brentq_robustness(
            coeff in 1.0f64..10.0,
            a in -5.0f64..0.0,
            b in 0.0f64..5.0
        ) {
            // For f(x) = coeff * x, root is always at 0
            let f = |x: f64| coeff * x;
            let result = brentq(f, a, b, 1e-10, 100).expect("brentq should converge for linear function");
            prop_assert!(result.converged);
            prop_assert!(result.root.abs() < 1e-9);
        }

        #[test]
        fn prop_secant_consistency(
            target in -10.0f64..10.0
        ) {
            // Secant should find roots consistently
            let f = |x: f64| x - target;
            let x0 = target - 1.0;
            let x1 = target + 1.0;

            let result = secant(f, x0, x1, 1e-10, 100).expect("secant should converge for linear function");
            prop_assert!(result.converged);
            prop_assert!((result.root - target).abs() < 1e-8);
        }
    }
}
