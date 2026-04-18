//! Multivariate Taylor series expansion using HyperDual numbers.

use crate::error::{NumRs2Error, Result};
use num_traits::Float;

use super::hyperdual::HyperDual;

/// Second-order multivariate Taylor expansion of a scalar function.
///
/// Stores the function value, gradient, and Hessian at a center point,
/// enabling evaluation of the quadratic approximation at nearby points.
///
/// `f(a + h) ~ f(a) + grad(a) . h + (1/2) h^T H(a) h`
#[derive(Debug, Clone)]
pub struct TaylorExpansion2<T> {
    /// Center point of the expansion
    pub center: Vec<T>,
    /// Function value at the center
    pub value: T,
    /// Gradient at the center
    pub gradient: Vec<T>,
    /// Hessian matrix at the center (stored as flat n*n vector, row-major)
    pub hessian_flat: Vec<T>,
    /// Dimension of the input space
    pub dim: usize,
}

impl<T: Float> TaylorExpansion2<T> {
    /// Evaluate the second-order Taylor approximation at a given point.
    ///
    /// Computes: `f(a) + grad . h + (1/2) h^T H h`
    /// where `h = x - center`.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the approximation
    pub fn evaluate(&self, x: &[T]) -> Result<T> {
        let n = self.dim;
        if x.len() != n {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n],
                actual: vec![x.len()],
            });
        }

        // Displacement from center
        let h: Vec<T> = (0..n).map(|i| x[i] - self.center[i]).collect();

        let mut result = self.value;

        // First-order term: grad . h
        for i in 0..n {
            result = result + self.gradient[i] * h[i];
        }

        // Second-order term: (1/2) h^T H h
        let two = T::one() + T::one();
        for i in 0..n {
            for j in 0..n {
                result = result + self.hessian_flat[i * n + j] * h[i] * h[j] / two;
            }
        }

        Ok(result)
    }

    /// Get the Hessian element at position (i, j)
    pub fn hessian_element(&self, i: usize, j: usize) -> T {
        self.hessian_flat[i * self.dim + j]
    }
}

/// Compute a second-order multivariate Taylor expansion using HyperDual numbers.
///
/// Computes the function value, exact gradient, and exact Hessian at the center
/// point simultaneously using hyper-dual number arithmetic.
///
/// # Arguments
///
/// * `f` - Scalar function accepting HyperDual inputs
/// * `center` - Point around which to expand
///
/// # Returns
///
/// A `TaylorExpansion2` containing the expansion coefficients
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::higher_order::*;
///
/// // f(x,y) = x^2 + y^2
/// let f = |vars: &[HyperDual<f64>]| vars[0] * vars[0] + vars[1] * vars[1];
/// let taylor = multivariate_taylor(f, &[1.0, 1.0]).expect("valid taylor expansion");
///
/// // Evaluate near the center
/// let approx = taylor.evaluate(&[1.1, 1.2]).expect("valid taylor evaluation");
/// // Should be close to f(1.1, 1.2) = 1.21 + 1.44 = 2.65
/// ```
pub fn multivariate_taylor<F, T>(f: F, center: &[T]) -> Result<TaylorExpansion2<T>>
where
    F: Fn(&[HyperDual<T>]) -> HyperDual<T>,
    T: Float,
{
    let n = center.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Center point must be non-empty".to_string(),
        ));
    }

    let mut value = T::zero();
    let mut gradient = vec![T::zero(); n];
    let mut hessian_flat = vec![T::zero(); n * n];

    for i in 0..n {
        for j in i..n {
            let inputs: Vec<HyperDual<T>> = (0..n)
                .map(|k| HyperDual::make_variable(center[k], k == i, k == j))
                .collect();

            let result = f(&inputs);

            // Function value (same from every evaluation)
            if i == 0 && j == 0 {
                value = result.real();
            }

            // Gradient component from diagonal evaluations
            if i == j {
                gradient[i] = result.eps1();
            }

            // Hessian entry (symmetric)
            let h_ij = result.eps1eps2();
            hessian_flat[i * n + j] = h_ij;
            hessian_flat[j * n + i] = h_ij;
        }
    }

    Ok(TaylorExpansion2 {
        center: center.to_vec(),
        value,
        gradient,
        hessian_flat,
        dim: n,
    })
}
