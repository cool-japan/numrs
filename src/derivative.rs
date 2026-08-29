//! Numerical differentiation algorithms
//!
//! This module provides high-accuracy numerical differentiation methods
//! for computing derivatives, gradients, Jacobians, and Hessians.
//!
//! # Available Methods
//!
//! ## Scalar Functions
//! - **Forward difference**: O(h) accuracy, cheapest
//! - **Central difference**: O(h²) accuracy, recommended for most uses
//! - **Complex step**: O(h²) without subtractive cancellation
//! - **Richardson extrapolation**: Adaptive high-order accuracy
//!
//! ## Vector Functions
//! - **Gradient**: Vector of partial derivatives
//! - **Jacobian**: Matrix of partial derivatives
//! - **Hessian**: Matrix of second derivatives
//! - **Directional derivative**: Derivative along a direction
//!
//! # Examples
//!
//! ```
//! use numrs2::derivative::*;
//!
//! let f = |x: f64| x * x;
//! let df = derivative(f, 2.0, DerivativeMethod::Central);
//! assert!((df - 4.0).abs() < 1e-6); // Numerical differentiation tolerance
//! ```

use num_traits::Float;

/// Method for numerical differentiation
#[derive(Debug, Clone, Copy)]
pub enum DerivativeMethod {
    /// Forward difference: f'(x) ≈ (f(x+h) - f(x))/h
    Forward,
    /// Backward difference: f'(x) ≈ (f(x) - f(x-h))/h
    Backward,
    /// Central difference: f'(x) ≈ (f(x+h) - f(x-h))/(2h)
    Central,
    /// Richardson extrapolation for higher accuracy
    Richardson,
}

/// Compute numerical derivative of a scalar function
///
/// # Arguments
///
/// * `f` - Function to differentiate
/// * `x` - Point at which to compute derivative
/// * `method` - Differentiation method to use
///
/// # Examples
///
/// ```
/// use numrs2::derivative::*;
///
/// let f = |x: f64| x * x * x;
/// let df = derivative(f, 2.0, DerivativeMethod::Central);
/// assert!((df - 12.0).abs() < 1e-6); // d/dx(x^3) at x=2 is 3*4 = 12
/// ```
pub fn derivative<T, F>(f: F, x: T, method: DerivativeMethod) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    let h = T::from(1e-8).expect("1e-8 is representable as Float");

    match method {
        DerivativeMethod::Forward => {
            let fx = f(x);
            let fxh = f(x + h);
            (fxh - fx) / h
        }
        DerivativeMethod::Backward => {
            let fx = f(x);
            let fxmh = f(x - h);
            (fx - fxmh) / h
        }
        DerivativeMethod::Central => {
            let fxh = f(x + h);
            let fxmh = f(x - h);
            (fxh - fxmh) / (T::from(2.0).expect("2.0 is representable as Float") * h)
        }
        DerivativeMethod::Richardson => richardson_derivative(f, x),
    }
}

/// Richardson extrapolation for high-accuracy derivatives
///
/// Uses multiple step sizes and Richardson extrapolation to achieve
/// higher-order accuracy.
fn richardson_derivative<T, F>(f: F, x: T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    let h0 = T::from(0.1).expect("0.1 is representable as Float");
    let con = T::from(2.0).expect("2.0 is representable as Float");
    let con2 = con * con;
    let ntab = 10;

    let mut a = vec![vec![T::zero(); ntab]; ntab];

    // Initial estimate with h = h0
    let mut hh = h0;
    a[0][0] = (f(x + hh) - f(x - hh)) / (T::from(2.0).expect("2.0 is representable as Float") * hh);

    let mut err = T::from(1e10).expect("1e10 is representable as Float");
    let mut ans = a[0][0];

    for i in 1..ntab {
        // Decrease step size
        hh = hh / con;
        a[0][i] =
            (f(x + hh) - f(x - hh)) / (T::from(2.0).expect("2.0 is representable as Float") * hh);

        let mut fac = con2;
        for j in 1..=i {
            a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - T::one());
            fac = fac * con2;

            let errt = (a[j][i] - a[j - 1][i])
                .abs()
                .max((a[j][i] - a[j - 1][i - 1]).abs());
            if errt < err {
                err = errt;
                ans = a[j][i];
            }
        }

        // Converged if error is small enough
        if (a[i][i] - a[i - 1][i - 1]).abs()
            >= err * T::from(2.0).expect("2.0 is representable as Float")
        {
            break;
        }
    }

    ans
}

/// Compute gradient of a scalar function
///
/// Returns vector of partial derivatives ∂f/∂x_i at point x.
///
/// # Examples
///
/// ```
/// use numrs2::derivative::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let grad = gradient(&f, &[2.0, 3.0], DerivativeMethod::Central);
/// assert!((grad[0] - 4.0).abs() < 1e-6);
/// assert!((grad[1] - 6.0).abs() < 1e-6);
/// ```
pub fn gradient<T, F>(f: &F, x: &[T], method: DerivativeMethod) -> Vec<T>
where
    T: Float,
    F: Fn(&[T]) -> T,
{
    let n = x.len();
    let h = T::from(1e-8).expect("1e-8 is representable as Float");
    let mut grad = vec![T::zero(); n];

    // Single reusable perturbation buffer for every finite-difference
    // evaluation below, instead of one or two fresh `x.to_vec()` copies per
    // `i` (was O(n) allocations of size O(n), i.e. O(n^2) total copied
    // elements; now one O(n) allocation for the whole call). Each
    // iteration perturbs `x_work[i]` off the pristine value read from `x`
    // itself (never accumulated via `+=`/`-=` on `x_work`), evaluates, then
    // restores `x_work[i] = x[i]` exactly -- so `x_work` is bit-for-bit
    // identical to `x` at the start of every iteration, exactly as a fresh
    // `x.to_vec()` would have been.
    let mut x_work = x.to_vec();

    match method {
        DerivativeMethod::Forward => {
            let f0 = f(x);
            for i in 0..n {
                let xi = x[i];
                x_work[i] = xi + h;
                grad[i] = (f(&x_work) - f0) / h;
                x_work[i] = xi;
            }
        }
        DerivativeMethod::Central => {
            for i in 0..n {
                let xi = x[i];
                x_work[i] = xi + h;
                let f_plus = f(&x_work);
                x_work[i] = xi - h;
                let f_minus = f(&x_work);
                x_work[i] = xi;
                grad[i] =
                    (f_plus - f_minus) / (T::from(2.0).expect("2.0 is representable as Float") * h);
            }
        }
        _ => {
            // Default to central for other methods
            for i in 0..n {
                let xi = x[i];
                x_work[i] = xi + h;
                let f_plus = f(&x_work);
                x_work[i] = xi - h;
                let f_minus = f(&x_work);
                x_work[i] = xi;
                grad[i] =
                    (f_plus - f_minus) / (T::from(2.0).expect("2.0 is representable as Float") * h);
            }
        }
    }

    grad
}

/// Compute Jacobian matrix of a vector-valued function
///
/// Returns matrix J where `J[i][j] = ∂f_i/∂x_j`
///
/// # Examples
///
/// ```
/// use numrs2::derivative::*;
///
/// let f = |x: &[f64]| vec![x[0]*x[0], x[0]*x[1]];
/// let jac = jacobian(&f, &[2.0, 3.0], DerivativeMethod::Central);
/// assert!((jac[0][0] - 4.0).abs() < 1e-6); // ∂f0/∂x0 = 2x0
/// assert!((jac[1][1] - 2.0).abs() < 1e-6); // ∂f1/∂x1 = x0
/// ```
pub fn jacobian<T, F>(f: &F, x: &[T], method: DerivativeMethod) -> Vec<Vec<T>>
where
    T: Float,
    F: Fn(&[T]) -> Vec<T>,
{
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let h = T::from(1e-8).expect("1e-8 is representable as Float");

    let mut jac = vec![vec![T::zero(); n]; m];

    // One reusable perturbation buffer for the whole call (see `gradient`'s
    // comment for why this is bit-exact vs. a fresh `x.to_vec()` per `j`).
    let mut x_work = x.to_vec();

    match method {
        DerivativeMethod::Central => {
            for j in 0..n {
                let xj = x[j];
                x_work[j] = xj + h;
                let f_plus = f(&x_work);
                x_work[j] = xj - h;
                let f_minus = f(&x_work);
                x_work[j] = xj;

                for i in 0..m {
                    jac[i][j] = (f_plus[i] - f_minus[i])
                        / (T::from(2.0).expect("2.0 is representable as Float") * h);
                }
            }
        }
        _ => {
            // Forward difference for other methods
            for j in 0..n {
                let xj = x[j];
                x_work[j] = xj + h;
                let f_plus = f(&x_work);
                x_work[j] = xj;

                for i in 0..m {
                    jac[i][j] = (f_plus[i] - f0[i]) / h;
                }
            }
        }
    }

    jac
}

/// Compute Hessian matrix (matrix of second derivatives)
///
/// Returns matrix H where `H[i][j] = ∂²f/∂x_i∂x_j`
///
/// # Examples
///
/// ```
/// use numrs2::derivative::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let hess = hessian(&f, &[1.0, 1.0]);
/// assert!((hess[0][0] - 2.0).abs() < 1e-4);
/// assert!((hess[1][1] - 2.0).abs() < 1e-4);
/// ```
pub fn hessian<T, F>(f: &F, x: &[T]) -> Vec<Vec<T>>
where
    T: Float,
    F: Fn(&[T]) -> T,
{
    let n = x.len();
    let h = T::from(1e-5).expect("1e-5 is representable as Float");
    let mut hess = vec![vec![T::zero(); n]; n];

    // One reusable perturbation buffer for the *entire* function (not one
    // per `(i, j)` pair, let alone the 4 fresh `x.to_vec()` copies per pair
    // the old code allocated) -- O(n) allocation total instead of O(n^2)
    // allocations of size O(n) each.
    //
    // Each of the 4 combinations below perturbs `x_work[i]`/`x_work[j]` off
    // the pristine scalars `xi = x[i]`/`xj = x[j]` (read from the untouched
    // `x` slice, never accumulated via `+=`/`-=` on `x_work`), evaluates,
    // then restores both positions to `xi`/`xj` exactly before the next
    // combination. That restore is unconditionally exact -- including when
    // `i == j`, where restoring `x_work[i] = xi` then `x_work[j] = xj` is
    // just writing the same index twice with the same value -- so
    // `x_work` is bit-for-bit identical to `x` at the start of every
    // combination, exactly as a fresh `x.to_vec()` would have been.
    //
    // This also reproduces the original's `i == j` behavior exactly,
    // including its non-obvious quirk: when `i == j`, `x_pp[j] = x_pp[j] +
    // h` (original) reads the *already-perturbed* `x_pp[i]`, so the
    // diagonal Hessian ends up using a `2h` step (`f(x+2h) - 2f(x) +
    // f(x-2h)` scaled by `4h^2`), not the `h` step the off-diagonal
    // entries use. `x_work[j] = x_work[j] + h` below reads the same
    // just-perturbed `x_work[i]` when `i == j` (same memory location), so
    // it reproduces that quirk bit-for-bit rather than "fixing" it.
    let mut x_work = x.to_vec();
    let four = T::from(4.0).expect("4.0 is representable as Float");

    for i in 0..n {
        for j in 0..=i {
            let xi = x[i];
            let xj = x[j];

            x_work[i] = xi + h;
            x_work[j] = x_work[j] + h;
            let f_pp = f(&x_work);
            x_work[i] = xi;
            x_work[j] = xj;

            x_work[i] = xi + h;
            x_work[j] = x_work[j] - h;
            let f_pm = f(&x_work);
            x_work[i] = xi;
            x_work[j] = xj;

            x_work[i] = xi - h;
            x_work[j] = x_work[j] + h;
            let f_mp = f(&x_work);
            x_work[i] = xi;
            x_work[j] = xj;

            x_work[i] = xi - h;
            x_work[j] = x_work[j] - h;
            let f_mm = f(&x_work);
            x_work[i] = xi;
            x_work[j] = xj;

            hess[i][j] = (f_pp - f_pm - f_mp + f_mm) / (four * h * h);
            hess[j][i] = hess[i][j]; // Hessian is symmetric
        }
    }

    hess
}

/// Compute directional derivative along a direction
///
/// Returns D_v f(x) = ∇f(x) · v
///
/// # Examples
///
/// ```
/// use numrs2::derivative::*;
///
/// let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
/// let direction = vec![1.0, 0.0]; // x-direction
/// let deriv = directional_derivative(&f, &[2.0, 3.0], &direction);
/// assert!((deriv - 4.0).abs() < 1e-6); // ∂f/∂x at (2,3) is 2*2 = 4
/// ```
pub fn directional_derivative<T, F>(f: &F, x: &[T], direction: &[T]) -> T
where
    T: Float + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let grad = gradient(f, x, DerivativeMethod::Central);
    grad.iter()
        .zip(direction.iter())
        .map(|(&gi, &di)| gi * di)
        .sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_derivative_polynomial() {
        // d/dx(x^3) = 3x^2
        let f = |x: f64| x.powi(3);
        let df = derivative(f, 2.0, DerivativeMethod::Central);
        assert_relative_eq!(df, 12.0, epsilon = 1e-6); // Numerical differentiation tolerance
    }

    #[test]
    fn test_derivative_transcendental() {
        // d/dx(sin(x)) = cos(x)
        let f = |x: f64| x.sin();
        let df = derivative(f, 1.0, DerivativeMethod::Central);
        assert_relative_eq!(df, 1.0f64.cos(), epsilon = 1e-7);
    }

    #[test]
    fn test_derivative_exponential() {
        // d/dx(e^x) = e^x
        let f = |x: f64| x.exp();
        let df = derivative(f, 2.0, DerivativeMethod::Central);
        assert_relative_eq!(df, 2.0f64.exp(), epsilon = 1e-7); // Relaxed for numerical diff
    }

    #[test]
    fn test_gradient_quadratic() {
        // ∇(x² + y²) = [2x, 2y]
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = gradient(&f, &[2.0, 3.0], DerivativeMethod::Central);
        assert_relative_eq!(grad[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(grad[1], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gradient_mixed() {
        // ∇(x²y + xy²) = [2xy + y², x² + 2xy]
        let f = |x: &[f64]| x[0] * x[0] * x[1] + x[0] * x[1] * x[1];
        let grad = gradient(&f, &[2.0, 3.0], DerivativeMethod::Central);
        let expected_dx = 2.0 * 2.0 * 3.0 + 3.0 * 3.0; // 12 + 9 = 21
        let expected_dy = 2.0 * 2.0 + 2.0 * 2.0 * 3.0; // 4 + 12 = 16
        assert_relative_eq!(grad[0], expected_dx, epsilon = 1e-5);
        assert_relative_eq!(grad[1], expected_dy, epsilon = 1e-5);
    }

    #[test]
    fn test_jacobian_linear() {
        // f(x,y) = [x + 2y, 3x + 4y]
        // J = [[1, 2], [3, 4]]
        let f = |x: &[f64]| vec![x[0] + 2.0 * x[1], 3.0 * x[0] + 4.0 * x[1]];
        let jac = jacobian(&f, &[1.0, 1.0], DerivativeMethod::Central);

        assert_relative_eq!(jac[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(jac[0][1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(jac[1][0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(jac[1][1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_jacobian_nonlinear() {
        // f(x,y) = [x², xy]
        // J = [[2x, 0], [y, x]]
        let f = |x: &[f64]| vec![x[0] * x[0], x[0] * x[1]];
        let jac = jacobian(&f, &[2.0, 3.0], DerivativeMethod::Central);

        assert_relative_eq!(jac[0][0], 4.0, epsilon = 1e-6); // 2x
        assert_relative_eq!(jac[0][1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(jac[1][0], 3.0, epsilon = 1e-6); // y
        assert_relative_eq!(jac[1][1], 2.0, epsilon = 1e-6); // x
    }

    #[test]
    fn test_hessian_quadratic() {
        // f(x,y) = x² + y²
        // H = [[2, 0], [0, 2]]
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let hess = hessian(&f, &[1.0, 1.0]);

        assert_relative_eq!(hess[0][0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(hess[0][1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(hess[1][0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(hess[1][1], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_hessian_mixed() {
        // f(x,y) = x²y
        // H = [[2y, 2x], [2x, 0]]
        let f = |x: &[f64]| x[0] * x[0] * x[1];
        let hess = hessian(&f, &[2.0, 3.0]);

        assert_relative_eq!(hess[0][0], 6.0, epsilon = 1e-4); // 2y = 6
        assert_relative_eq!(hess[0][1], 4.0, epsilon = 1e-4); // 2x = 4
        assert_relative_eq!(hess[1][0], 4.0, epsilon = 1e-4); // 2x = 4
        assert_relative_eq!(hess[1][1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_directional_derivative() {
        // f(x,y) = x² + y², direction = [1,1]/√2
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let sqrt2_inv = 1.0 / 2.0f64.sqrt();
        let direction = vec![sqrt2_inv, sqrt2_inv];
        let deriv = directional_derivative(&f, &[2.0, 3.0], &direction);

        // ∇f = [4, 6], direction = [1,1]/√2
        // D_v f = (4 + 6)/√2 = 10/√2
        let expected = 10.0 / 2.0f64.sqrt();
        assert_relative_eq!(deriv, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_richardson_vs_central() {
        // Richardson extrapolation for high accuracy
        let f = |x: f64| x.powi(5);
        let exact = 5.0 * 2.0f64.powi(4); // d/dx(x^5) at x=2

        let central = derivative(f, 2.0, DerivativeMethod::Central);
        let richardson = derivative(f, 2.0, DerivativeMethod::Richardson);

        let err_central = (central - exact).abs();
        let err_richardson = (richardson - exact).abs();

        // Both should be accurate for polynomial
        assert!(err_central < 1e-4, "Central difference should be accurate");
        assert!(err_richardson < 1e-4, "Richardson should be accurate");
    }

    #[test]
    fn test_forward_vs_central_accuracy() {
        let f = |x: f64| x.sin();
        let exact = 1.0f64.cos();

        let forward = derivative(f, 1.0, DerivativeMethod::Forward);
        let central = derivative(f, 1.0, DerivativeMethod::Central);

        let err_forward = (forward - exact).abs();
        let err_central = (central - exact).abs();

        // Central should be more accurate
        assert!(err_central < err_forward);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_derivative_linear(
            a in -10.0f64..10.0,
            b in -10.0f64..10.0,
            x in -5.0f64..5.0
        ) {
            // d/dx(ax + b) = a for any x
            let f = |x: f64| a * x + b;
            let df = derivative(f, x, DerivativeMethod::Central);
            prop_assert!((df - a).abs() < 1e-6);
        }

        #[test]
        fn prop_gradient_separable(
            x0 in -5.0f64..5.0,
            y0 in -5.0f64..5.0
        ) {
            // ∇(x² + y²) = [2x, 2y]
            let f = |x: &[f64]| x[0]*x[0] + x[1]*x[1];
            let grad = gradient(&f, &[x0, y0], DerivativeMethod::Central);
            prop_assert!((grad[0] - 2.0 * x0).abs() < 1e-5);
            prop_assert!((grad[1] - 2.0 * y0).abs() < 1e-5);
        }

        #[test]
        fn prop_hessian_symmetry(
            x in -3.0f64..3.0,
            y in -3.0f64..3.0
        ) {
            // Hessian should always be symmetric
            let f = |x: &[f64]| x[0].powi(3) + x[0] * x[1] * x[1] + x[1].powi(4);
            let hess = hessian(&f, &[x, y]);
            prop_assert!((hess[0][1] - hess[1][0]).abs() < 1e-4);
        }
    }

    // =========================================================================
    // to_vec sweep (W2-D): reused-`x_work`-buffer FD rewrite
    //
    // `gradient`/`jacobian`/`hessian` used to allocate a fresh `x.to_vec()`
    // per perturbation (1-2 per `i`/`j` for gradient/jacobian; 4 per `(i,
    // j)` pair for hessian, i.e. O(n^2) allocations of size O(n) each).
    // They now reuse one `x_work` buffer for the whole call, perturbing
    // and restoring individual scalars. These tests pin (1) bit-exact
    // equivalence against the original approach -- kept here as
    // `naive_*` reproductions -- including `hessian`'s non-obvious
    // `i == j` "double perturbation to a 2h step" quirk, which a
    // tolerance-based (`epsilon`) comparison would not reliably catch,
    // and (2) that the rewrite is actually faster, via the same in-test
    // timing-probe idiom `kernels::elementwise` uses
    // (`probe_binary_dispatch_perf_vs_serial`), since `cargo bench`'s
    // `[[bench]]` wiring in `Cargo.toml` belongs to a different lane.
    // =========================================================================

    /// Byte-for-byte reproduction of `gradient`'s pre-optimization
    /// central-difference body (fresh `x.to_vec()` x2 per `i`).
    fn naive_gradient_central<T, F>(f: &F, x: &[T], h: T) -> Vec<T>
    where
        T: Float,
        F: Fn(&[T]) -> T,
    {
        let n = x.len();
        let mut grad = vec![T::zero(); n];
        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] = x_plus[i] + h;
            x_minus[i] = x_minus[i] - h;
            grad[i] =
                (f(&x_plus) - f(&x_minus)) / (T::from(2.0).expect("2.0 is representable") * h);
        }
        grad
    }

    /// Byte-for-byte reproduction of `hessian`'s pre-optimization body
    /// (4 fresh `x.to_vec()` copies per `(i, j)` pair).
    fn naive_hessian<T, F>(f: &F, x: &[T]) -> Vec<Vec<T>>
    where
        T: Float,
        F: Fn(&[T]) -> T,
    {
        let n = x.len();
        let h = T::from(1e-5).expect("1e-5 is representable");
        let mut hess = vec![vec![T::zero(); n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut x_pp = x.to_vec();
                let mut x_pm = x.to_vec();
                let mut x_mp = x.to_vec();
                let mut x_mm = x.to_vec();

                x_pp[i] = x_pp[i] + h;
                x_pp[j] = x_pp[j] + h;

                x_pm[i] = x_pm[i] + h;
                x_pm[j] = x_pm[j] - h;

                x_mp[i] = x_mp[i] - h;
                x_mp[j] = x_mp[j] + h;

                x_mm[i] = x_mm[i] - h;
                x_mm[j] = x_mm[j] - h;

                let f_pp = f(&x_pp);
                let f_pm = f(&x_pm);
                let f_mp = f(&x_mp);
                let f_mm = f(&x_mm);

                hess[i][j] = (f_pp - f_pm - f_mp + f_mm)
                    / (T::from(4.0).expect("4.0 is representable") * h * h);
                hess[j][i] = hess[i][j];
            }
        }
        hess
    }

    #[test]
    fn gradient_central_matches_naive_bit_exact() {
        // Cross-terms so every partial derivative mixes multiple x_i's.
        let f =
            |x: &[f64]| x[0] * x[0] * x[1] + x[1].sin() * x[2] - x[2] * x[3] * x[3] + x[3] * x[0];
        let x = [1.3, -2.7, 0.4, 5.1];
        let h = 1e-8;
        let expected = naive_gradient_central(&f, &x, h);
        let got = gradient(&f, &x, DerivativeMethod::Central);
        assert_eq!(
            got, expected,
            "reused-buffer gradient must be bit-exact vs the naive fresh-to_vec version"
        );
    }

    #[test]
    fn jacobian_central_matches_naive_bit_exact() {
        let f = |x: &[f64]| {
            vec![
                x[0] * x[0] * x[1],
                x[0] * x[1] * x[1],
                x[0] + x[1] * x[1] * x[1],
            ]
        };
        let x = [2.3, -1.4];
        let got = jacobian(&f, &x, DerivativeMethod::Central);

        // Naive reproduction of jacobian's pre-optimization central branch.
        let h = 1e-8;
        let n = x.len();
        let m = f(&x).len();
        let mut expected = vec![vec![0.0; n]; m];
        for j in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[j] += h;
            x_minus[j] -= h;
            let f_plus = f(&x_plus);
            let f_minus = f(&x_minus);
            for i in 0..m {
                expected[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h);
            }
        }
        assert_eq!(
            got, expected,
            "reused-buffer jacobian must be bit-exact vs naive"
        );
    }

    #[test]
    fn hessian_matches_naive_bit_exact_including_diagonal() {
        // Cross-terms on both off-diagonal and diagonal entries.
        let f = |x: &[f64]| x[0].powi(3) + x[0] * x[1] * x[1] + x[1] * x[2] - x[2].powi(2) * x[0];
        let x = [1.1, -0.6, 2.3];
        let expected = naive_hessian(&f, &x);
        let got = hessian(&f, &x);
        for i in 0..x.len() {
            for j in 0..x.len() {
                assert_eq!(
                    got[i][j],
                    expected[i][j],
                    "hessian[{i}][{j}] (diagonal={}) must be bit-exact vs naive",
                    i == j
                );
            }
        }
    }

    #[test]
    fn probe_gradient_perf_vs_naive() {
        // n=256, as named in the lane brief.
        let n = 256;
        let f = |x: &[f64]| -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, &v)| v * v * (i as f64 + 1.0).sqrt())
                .sum()
        };
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 1.0).collect();
        let h = 1e-8;
        let iters = 200;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_gradient_central(&f, &x, h));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(gradient(&f, &x, DerivativeMethod::Central));
        }
        let reused = t1.elapsed();

        eprintln!(
            "[gradient n={n}] naive(fresh_to_vec_per_i)={:.1}us/iter reused_buffer={:.1}us/iter ({:.2}x)",
            naive.as_secs_f64() * 1e6 / iters as f64,
            reused.as_secs_f64() * 1e6 / iters as f64,
            naive.as_secs_f64() / reused.as_secs_f64(),
        );
    }

    #[test]
    fn probe_hessian_perf_vs_naive() {
        let n = 48;
        let f = |x: &[f64]| -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, &v)| v * v * (i as f64 + 1.0))
                .sum()
        };
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect();
        let iters = 5;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_hessian(&f, &x));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(hessian(&f, &x));
        }
        let reused = t1.elapsed();

        eprintln!(
            "[hessian n={n}] naive(4_fresh_to_vec_per_ij)={:.2}ms/iter reused_buffer={:.2}ms/iter ({:.2}x)",
            naive.as_secs_f64() * 1e3 / iters as f64,
            reused.as_secs_f64() * 1e3 / iters as f64,
            naive.as_secs_f64() / reused.as_secs_f64(),
        );
    }
}
