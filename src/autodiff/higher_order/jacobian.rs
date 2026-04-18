//! Jacobian computation via forward and reverse mode AD.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

use crate::autodiff::{Dual, Tape, Var};

/// Compute Jacobian matrix using forward-mode AD (Dual numbers).
///
/// For a function `f: R^n -> R^m`, computes the m x n Jacobian matrix
/// where `J[i,j] = df_i/dx_j`. Requires n forward passes (one per input
/// dimension), so this is efficient when n < m.
///
/// # Arguments
///
/// * `f` - Vector-valued function mapping Dual number inputs to Dual number outputs
/// * `x` - Point at which to evaluate the Jacobian
///
/// # Returns
///
/// The m x n Jacobian matrix as a reshaped `Array<T>`
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::higher_order::*;
/// use numrs2::autodiff::Dual;
///
/// // f(x, y) = [x^2 + y, x*y]
/// let f = |vars: &[Dual<f64>]| -> Vec<Dual<f64>> {
///     vec![vars[0] * vars[0] + vars[1], vars[0] * vars[1]]
/// };
/// let jac = forward_jacobian(f, &[2.0, 3.0]).expect("valid jacobian computation");
/// ```
pub fn forward_jacobian<F, T>(f: F, x: &[T]) -> Result<Array<T>>
where
    F: Fn(&[Dual<T>]) -> Vec<Dual<T>>,
    T: Float,
{
    let n = x.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Input vector must be non-empty".to_string(),
        ));
    }

    // Determine output dimension with a probe evaluation
    let probe_input: Vec<Dual<T>> = x.iter().map(|&v| Dual::variable(v)).collect();
    let probe_output = f(&probe_input);
    let m = probe_output.len();
    if m == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Output vector must be non-empty".to_string(),
        ));
    }

    // Build Jacobian: one forward pass per input variable
    let mut jac_data = vec![T::zero(); m * n];

    for j in 0..n {
        // Set derivative seed: e_j (unit vector in direction j)
        let dual_input: Vec<Dual<T>> = (0..n)
            .map(|k| {
                let deriv = if k == j { T::one() } else { T::zero() };
                Dual::new(x[k], deriv)
            })
            .collect();

        let output = f(&dual_input);
        for i in 0..m {
            // J[i,j] = derivative of output i w.r.t. input j
            jac_data[i * n + j] = output[i].deriv();
        }
    }

    Ok(Array::from_vec(jac_data).reshape(&[m, n]))
}

/// Compute Jacobian matrix using reverse-mode AD (Tape).
///
/// For a function `f: R^n -> R^m`, computes the m x n Jacobian matrix
/// by running m backward passes (one per output dimension). This is
/// efficient when n > m (many inputs, few outputs).
///
/// # Arguments
///
/// * `f` - Function that builds a computation graph on a Tape
/// * `x` - Point at which to evaluate the Jacobian
///
/// # Returns
///
/// The m x n Jacobian matrix as a reshaped `Array<T>`
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::higher_order::*;
/// use numrs2::autodiff::{Tape, Var};
///
/// let jac = reverse_jacobian(
///     |tape: &mut Tape<f64>, vars: &[Var]| {
///         let x_sq = tape.mul(vars[0], vars[0]);
///         let out1 = tape.add(x_sq, vars[1]);
///         let out2 = tape.mul(vars[0], vars[1]);
///         vec![out1, out2]
///     },
///     &[2.0, 3.0],
/// ).expect("valid reverse jacobian computation");
/// ```
pub fn reverse_jacobian<F, T>(f: F, x: &[T]) -> Result<Array<T>>
where
    F: Fn(&mut Tape<T>, &[Var]) -> Vec<Var>,
    T: Float,
{
    let n = x.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Input vector must be non-empty".to_string(),
        ));
    }

    // Build the computation graph once
    let mut tape = Tape::new();
    let vars: Vec<Var> = x.iter().map(|&v| tape.var(v)).collect();
    let outputs = f(&mut tape, &vars);
    let m = outputs.len();
    if m == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Output vector must be non-empty".to_string(),
        ));
    }

    // Compute Jacobian: one backward pass per output
    let mut jac_data = vec![T::zero(); m * n];

    for i in 0..m {
        tape.zero_grad();
        tape.backward(outputs[i]);
        for j in 0..n {
            jac_data[i * n + j] = tape.grad(vars[j]);
        }
    }

    Ok(Array::from_vec(jac_data).reshape(&[m, n]))
}

/// Automatically select forward or reverse mode for Jacobian computation.
///
/// Chooses the more efficient mode based on input/output dimensions:
/// - Forward mode when n <= m (fewer inputs than outputs)
/// - Forward mode is always used (since reverse mode requires a different
///   function signature with Tape operations)
///
/// For cases where reverse mode is preferred, use `reverse_jacobian` directly.
///
/// # Arguments
///
/// * `f` - Vector-valued function using Dual numbers
/// * `x` - Point at which to evaluate the Jacobian
pub fn jacobian_auto<F, T>(f: F, x: &[T]) -> Result<Array<T>>
where
    F: Fn(&[Dual<T>]) -> Vec<Dual<T>>,
    T: Float,
{
    // With Dual number interface, we use forward mode
    // The function probe determines output dimension for documentation
    forward_jacobian(f, x)
}
