//! Gradient computation utilities, gradient checking, and related types.

use crate::error::{NumRs2Error, Result};
use num_traits::Float;

use super::hyperdual::{float_const, HyperDual};
use crate::autodiff::Dual;

/// Compute gradient of a scalar function using forward-mode AD (Dual numbers).
///
/// For a function `f: R^n -> R`, computes the gradient vector
/// `[df/dx_1, df/dx_2, ..., df/dx_n]` using exact dual number arithmetic.
///
/// # Arguments
///
/// * `f` - Scalar function accepting Dual number inputs
/// * `x` - Point at which to compute the gradient
pub fn compute_gradient_ad<F, T>(f: F, x: &[T]) -> Result<Vec<T>>
where
    F: Fn(&[Dual<T>]) -> Dual<T>,
    T: Float,
{
    let n = x.len();
    let mut grad = Vec::with_capacity(n);

    for i in 0..n {
        let dual_input: Vec<Dual<T>> = (0..n)
            .map(|k| {
                let deriv = if k == i { T::one() } else { T::zero() };
                Dual::new(x[k], deriv)
            })
            .collect();

        let result = f(&dual_input);
        grad.push(result.deriv());
    }

    Ok(grad)
}

/// Compute gradient using HyperDual numbers (extracts first-order information).
///
/// This is useful when you already have a function written for HyperDual
/// inputs and want to compute the gradient without writing a separate Dual version.
pub fn compute_gradient_hyperdual<F, T>(f: &F, x: &[T]) -> Result<Vec<T>>
where
    F: Fn(&[HyperDual<T>]) -> HyperDual<T>,
    T: Float,
{
    let n = x.len();
    let mut grad = Vec::with_capacity(n);

    for i in 0..n {
        let inputs: Vec<HyperDual<T>> = (0..n)
            .map(|k| {
                HyperDual::new(
                    x[k],
                    if k == i { T::one() } else { T::zero() },
                    T::zero(),
                    T::zero(),
                )
            })
            .collect();
        let result = f(&inputs);
        grad.push(result.eps1());
    }

    Ok(grad)
}

/// Compute numerical gradient using central finite differences.
///
/// This is used internally for gradient checking, providing an independent
/// reference gradient computation.
pub(super) fn numerical_gradient_central<T: Float>(
    f: &dyn Fn(&[T]) -> T,
    x: &[T],
) -> Result<Vec<T>> {
    let n = x.len();
    let eps = T::epsilon().sqrt();
    let two = T::one() + T::one();
    let mut grad = Vec::with_capacity(n);

    for i in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] = x_plus[i] + eps;
        x_minus[i] = x_minus[i] - eps;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        grad.push((f_plus - f_minus) / (two * eps));
    }

    Ok(grad)
}

/// Result of a gradient check comparing analytical and numerical gradients.
#[derive(Debug, Clone)]
pub struct GradientCheckResult<T> {
    /// Maximum absolute error across all gradient components
    pub max_abs_error: T,
    /// Maximum relative error across all gradient components
    pub max_rel_error: T,
    /// Whether the gradient check passed (max_rel_error < tolerance)
    pub passed: bool,
    /// Per-component absolute errors
    pub component_errors: Vec<T>,
    /// The analytically computed gradient
    pub analytical: Vec<T>,
    /// The numerically computed gradient
    pub numerical: Vec<T>,
}

/// Check an analytically computed gradient against numerical finite differences.
///
/// Computes a central-difference numerical gradient and compares it against the
/// provided analytical gradient. Returns detailed error information.
///
/// # Arguments
///
/// * `f` - The original scalar function (for numerical gradient computation)
/// * `x` - Point at which the gradient was computed
/// * `analytical_gradient` - The gradient to validate
/// * `tolerance` - Maximum acceptable relative error
///
/// # Returns
///
/// A `GradientCheckResult` containing error metrics and pass/fail status
pub fn gradient_check<F, T>(
    f: F,
    x: &[T],
    analytical_gradient: &[T],
    tolerance: T,
) -> Result<GradientCheckResult<T>>
where
    F: Fn(&[T]) -> T,
    T: Float,
{
    let n = x.len();
    if analytical_gradient.len() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: vec![analytical_gradient.len()],
        });
    }

    let numerical = numerical_gradient_central(&f, x)?;
    let mut max_abs = T::zero();
    let mut max_rel = T::zero();
    let mut component_errors = Vec::with_capacity(n);

    let threshold = T::epsilon() * float_const::<T>(100.0)?;

    for i in 0..n {
        let abs_err = (analytical_gradient[i] - numerical[i]).abs();
        component_errors.push(abs_err);

        if abs_err > max_abs {
            max_abs = abs_err;
        }

        // Relative error: use max magnitude as denominator
        let max_mag = analytical_gradient[i].abs().max(numerical[i].abs());
        let rel_err = if max_mag < threshold {
            abs_err
        } else {
            abs_err / max_mag
        };

        if rel_err > max_rel {
            max_rel = rel_err;
        }
    }

    Ok(GradientCheckResult {
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        passed: max_rel < tolerance,
        component_errors,
        analytical: analytical_gradient.to_vec(),
        numerical,
    })
}

/// Compute AD gradient and validate against numerical gradient.
///
/// Convenience function that computes the gradient using forward-mode AD
/// (Dual numbers) and then checks it against a numerical gradient.
///
/// # Arguments
///
/// * `f_ad` - Function written for Dual numbers (for AD gradient)
/// * `f_numerical` - Plain scalar function (for numerical gradient)
/// * `x` - Point at which to check the gradient
/// * `tolerance` - Maximum acceptable relative error
pub fn gradient_check_ad<F1, F2, T>(
    f_ad: F1,
    f_numerical: F2,
    x: &[T],
    tolerance: T,
) -> Result<GradientCheckResult<T>>
where
    F1: Fn(&[Dual<T>]) -> Dual<T>,
    F2: Fn(&[T]) -> T,
    T: Float,
{
    let ad_grad = compute_gradient_ad(f_ad, x)?;
    gradient_check(f_numerical, x, &ad_grad, tolerance)
}
