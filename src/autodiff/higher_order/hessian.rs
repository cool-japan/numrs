//! Hessian computation via HyperDual numbers and forward-over-reverse mode.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

use super::hyperdual::HyperDual;
use crate::autodiff::{Tape, Var};

/// Compute the exact Hessian matrix using HyperDual numbers.
///
/// For a scalar function `f: R^n -> R`, computes the n x n Hessian matrix
/// where `H[i,j] = d^2f/(dx_i dx_j)`. Uses hyper-dual numbers for exact
/// analytical second derivatives with no numerical approximation.
///
/// This requires `n*(n+1)/2` function evaluations (exploiting symmetry).
///
/// # Arguments
///
/// * `f` - Scalar function accepting HyperDual number inputs
/// * `x` - Point at which to compute the Hessian
///
/// # Returns
///
/// The n x n symmetric Hessian matrix
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::higher_order::*;
///
/// // f(x,y) = x^2 + y^2, Hessian = [[2,0],[0,2]]
/// fn f(vars: &[HyperDual<f64>]) -> HyperDual<f64> {
///     vars[0] * vars[0] + vars[1] * vars[1]
/// }
///
/// let hess = hessian_exact(f, &[3.0, 4.0]).expect("valid hessian computation");
/// ```
pub fn hessian_exact<F, T>(f: F, x: &[T]) -> Result<Array<T>>
where
    F: Fn(&[HyperDual<T>]) -> HyperDual<T>,
    T: Float,
{
    let n = x.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Input vector must be non-empty for Hessian computation".to_string(),
        ));
    }

    let mut hess_data = vec![T::zero(); n * n];

    for i in 0..n {
        for j in i..n {
            // Set up hyper-dual inputs with perturbations in directions i and j
            let inputs: Vec<HyperDual<T>> = (0..n)
                .map(|k| HyperDual::make_variable(x[k], k == i, k == j))
                .collect();

            let result = f(&inputs);
            let h_ij = result.eps1eps2();

            // Hessian is symmetric: H[i,j] = H[j,i]
            hess_data[i * n + j] = h_ij;
            hess_data[j * n + i] = h_ij;
        }
    }

    Array::from_vec_shape(hess_data, &[n, n])
}

/// Compute Hessian-vector product `H(x) * v` without forming the full Hessian.
///
/// For a scalar function `f: R^n -> R` and vector `v`, computes `H*v` where
/// `H` is the Hessian of `f` at point `x`. This is much more efficient than
/// forming the full Hessian when only the product is needed (O(n) evaluations
/// instead of O(n^2)).
///
/// The computation uses hyper-dual numbers:
/// - Set eps1 = e_i (standard basis), eps2 = v
/// - Then result.eps1eps2 = sum_j (H\[i,j\] * v\[j\]) = (H*v)\[i\]
///
/// # Arguments
///
/// * `f` - Scalar function accepting HyperDual inputs
/// * `x` - Point at which to evaluate
/// * `v` - Vector to multiply with the Hessian
///
/// # Returns
///
/// The vector `H(x) * v`
pub fn hessian_vector_product<F, T>(f: F, x: &[T], v: &[T]) -> Result<Vec<T>>
where
    F: Fn(&[HyperDual<T>]) -> HyperDual<T>,
    T: Float,
{
    let n = x.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Input vector must be non-empty".to_string(),
        ));
    }
    if v.len() != n {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![n],
            actual: vec![v.len()],
        });
    }

    let mut hv = Vec::with_capacity(n);

    for i in 0..n {
        // eps1 direction = e_i (standard basis vector)
        // eps2 direction = v (the given vector)
        let inputs: Vec<HyperDual<T>> = (0..n)
            .map(|k| {
                HyperDual::new(
                    x[k],
                    if k == i { T::one() } else { T::zero() },
                    v[k],
                    T::zero(),
                )
            })
            .collect();

        let result = f(&inputs);
        // result.eps1eps2 = sum_j H[i,j]*v[j] = (H*v)[i]
        hv.push(result.eps1eps2());
    }

    Ok(hv)
}

/// Compute Hessian using forward-over-reverse mode.
///
/// This approach combines reverse-mode AD for exact first derivatives with
/// numerical finite differences for second derivatives. The gradient at each
/// perturbed point is computed exactly via the tape-based reverse mode, and
/// then the Hessian is approximated by finite-differencing the gradient.
///
/// This is more accurate than purely numerical second derivatives because
/// the gradient computation itself is exact.
///
/// Requires `2n` tape constructions and backward passes.
///
/// # Arguments
///
/// * `f` - Function that builds a scalar computation on a Tape
/// * `x` - Point at which to compute the Hessian
///
/// # Returns
///
/// The n x n Hessian matrix (approximately symmetric)
pub fn hessian_forward_over_reverse<F, T>(f: F, x: &[T]) -> Result<Array<T>>
where
    F: Fn(&mut Tape<T>, &[Var]) -> Var,
    T: Float,
{
    let n = x.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Input vector must be non-empty for Hessian computation".to_string(),
        ));
    }

    // Use adaptive step size based on machine epsilon
    let eps = T::epsilon().sqrt();
    let two = T::one() + T::one();

    // Helper: compute exact gradient at a point using reverse mode
    let compute_gradient = |point: &[T]| -> Vec<T> {
        let mut tape = Tape::new();
        let vars: Vec<Var> = point.iter().map(|&v| tape.var(v)).collect();
        let output = f(&mut tape, &vars);
        tape.backward(output);
        vars.iter().map(|&v| tape.grad(v)).collect()
    };

    // Compute Hessian by finite-differencing the exact gradient
    let mut hess_data = Vec::with_capacity(n * n);

    for i in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] = x_plus[i] + eps;
        x_minus[i] = x_minus[i] - eps;

        let grad_plus = compute_gradient(&x_plus);
        let grad_minus = compute_gradient(&x_minus);

        for j in 0..n {
            let h_ij = (grad_plus[j] - grad_minus[j]) / (two * eps);
            hess_data.push(h_ij);
        }
    }

    Array::from_vec_shape(hess_data, &[n, n])
}
