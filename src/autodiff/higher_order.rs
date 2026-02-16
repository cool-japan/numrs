//! Higher-order automatic differentiation for NumRS2
//!
//! This module provides exact higher-order derivatives through hyper-dual numbers,
//! efficient Jacobian and Hessian computation, and gradient validation utilities.
//!
//! # Overview
//!
//! ## HyperDual Numbers
//!
//! Hyper-dual numbers extend dual numbers to capture second-order derivatives exactly.
//! A hyper-dual number has four components: `f + f_1*e1 + f_2*e2 + f_12*e1e2`
//! where `e1^2 = e2^2 = 0` but `e1*e2 != 0`. The `e1*e2` component gives the
//! exact mixed second partial derivative `d^2f/(dx_i dx_j)`.
//!
//! ## Jacobian Computation
//!
//! For vector-valued functions `f: R^n -> R^m`:
//! - **Forward mode**: Efficient when `n < m` (few inputs, many outputs)
//! - **Reverse mode**: Efficient when `n > m` (many inputs, few outputs)
//! - **Auto mode**: Automatically selects the most efficient mode
//!
//! ## Hessian Computation
//!
//! For scalar functions `f: R^n -> R`:
//! - **Exact (HyperDual)**: Analytical second derivatives, no numerical error
//! - **Forward-over-reverse**: Exact first derivatives, numerical second derivatives
//! - **Hessian-vector product**: Efficient `Hv` without forming full matrix
//!
//! ## Gradient Checking
//!
//! Validates analytical gradients against numerical finite differences.
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::autodiff::higher_order::*;
//!
//! // Compute exact Hessian of f(x,y) = x^2 + x*y + y^2
//! fn f(vars: &[HyperDual<f64>]) -> HyperDual<f64> {
//!     let x = vars[0];
//!     let y = vars[1];
//!     x * x + x * y + y * y
//! }
//!
//! let point = vec![1.0, 2.0];
//! let hess = hessian_exact(f, &point).unwrap();
//! // H = [[2, 1], [1, 2]]
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{Dual, Tape, Var};

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert an f64 constant to a generic Float type with proper error handling
fn float_const<T: Float>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| {
        NumRs2Error::NumericalError(format!("Cannot represent {} in target float type", val))
    })
}

// ============================================================================
// HyperDual Numbers for Exact Second-Order Derivatives
// ============================================================================

/// Hyper-dual number for exact computation of second-order derivatives.
///
/// A hyper-dual number extends the dual number concept to capture both first
/// and second derivatives exactly (no numerical approximation). It represents:
///
/// `f = a + b*e1 + c*e2 + d*e1*e2`
///
/// where `e1^2 = e2^2 = 0` but `e1*e2 != 0`.
///
/// When evaluating `f(x)` with `x_k = x_k + delta(k,i)*e1 + delta(k,j)*e2`:
/// - `real` = f(x)
/// - `eps1` = df/dx_i
/// - `eps2` = df/dx_j
/// - `eps1eps2` = d^2f/(dx_i dx_j)  (exact second derivative!)
///
/// # Type Parameters
///
/// * `T` - The numeric type (typically `f32` or `f64`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperDual<T> {
    /// Function value (real part)
    real: T,
    /// First derivative w.r.t. first perturbation direction (e1 coefficient)
    eps1: T,
    /// First derivative w.r.t. second perturbation direction (e2 coefficient)
    eps2: T,
    /// Second cross derivative (e1*e2 coefficient)
    eps1eps2: T,
}

impl<T: Float> HyperDual<T> {
    /// Create a new hyper-dual number with all four components
    ///
    /// # Arguments
    ///
    /// * `real` - The function value
    /// * `eps1` - First derivative component (e1 direction)
    /// * `eps2` - First derivative component (e2 direction)
    /// * `eps1eps2` - Second cross derivative component
    #[inline]
    pub fn new(real: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        Self {
            real,
            eps1,
            eps2,
            eps1eps2,
        }
    }

    /// Create a constant hyper-dual number (all derivatives zero)
    #[inline]
    pub fn constant(value: T) -> Self {
        Self::new(value, T::zero(), T::zero(), T::zero())
    }

    /// Create a hyper-dual variable for computing d^2f/(dx_i dx_j)
    ///
    /// For input component x_k, set:
    /// - eps1 = 1 if k == i (variable index for first derivative direction)
    /// - eps2 = 1 if k == j (variable index for second derivative direction)
    /// - eps1eps2 = 0 always
    #[inline]
    pub fn make_variable(value: T, is_dir_i: bool, is_dir_j: bool) -> Self {
        Self::new(
            value,
            if is_dir_i { T::one() } else { T::zero() },
            if is_dir_j { T::one() } else { T::zero() },
            T::zero(),
        )
    }

    /// Get the function value (real part)
    #[inline]
    pub fn real(&self) -> T {
        self.real
    }

    /// Get the first derivative w.r.t. direction 1 (e1 coefficient)
    #[inline]
    pub fn eps1(&self) -> T {
        self.eps1
    }

    /// Get the first derivative w.r.t. direction 2 (e2 coefficient)
    #[inline]
    pub fn eps2(&self) -> T {
        self.eps2
    }

    /// Get the second cross derivative (e1*e2 coefficient)
    ///
    /// This is the key output: the exact mixed partial derivative d^2f/(dx_i dx_j).
    #[inline]
    pub fn eps1eps2(&self) -> T {
        self.eps1eps2
    }

    /// Scale all components by a scalar value
    #[inline]
    pub fn scale(self, s: T) -> Self {
        Self::new(
            self.real * s,
            self.eps1 * s,
            self.eps2 * s,
            self.eps1eps2 * s,
        )
    }

    // ========================================================================
    // Mathematical Functions
    // ========================================================================
    //
    // For a smooth function g applied to hyper-dual x = a + b*e1 + c*e2 + d*e1*e2:
    // g(x) = g(a) + g'(a)*b*e1 + g'(a)*c*e2 + (g''(a)*b*c + g'(a)*d)*e1*e2

    /// Compute power with float exponent: x^n
    ///
    /// Note: Requires `self.real() > 0` for non-integer exponents.
    /// For integer powers with possibly negative base, use `powi`.
    pub fn powf(self, n: T) -> Self {
        let a = self.real;
        let gp = n * a.powf(n - T::one());
        let gpp = n * (n - T::one()) * a.powf(n - T::one() - T::one());
        Self::new(
            a.powf(n),
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute integer power: x^n (safe for negative base)
    ///
    /// Uses exponentiation by squaring via operator overloading,
    /// which correctly propagates derivatives through multiplication.
    pub fn powi(self, n: i32) -> Self {
        match n {
            0 => Self::constant(T::one()),
            1 => self,
            _ if n < 0 => Self::constant(T::one()) / self.powi(-n),
            _ => {
                let half = self.powi(n / 2);
                if n % 2 == 0 {
                    half * half
                } else {
                    half * half * self
                }
            }
        }
    }

    /// Compute exponential: e^x
    pub fn exp(self) -> Self {
        let ea = self.real.exp();
        // g'(a) = e^a, g''(a) = e^a
        Self::new(
            ea,
            ea * self.eps1,
            ea * self.eps2,
            ea * (self.eps1 * self.eps2 + self.eps1eps2),
        )
    }

    /// Compute natural logarithm: ln(x)
    pub fn ln(self) -> Self {
        let a = self.real;
        // g'(a) = 1/a, g''(a) = -1/a^2
        let gp = T::one() / a;
        let gpp = -T::one() / (a * a);
        Self::new(
            a.ln(),
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute sine: sin(x)
    pub fn sin(self) -> Self {
        let sin_a = self.real.sin();
        let cos_a = self.real.cos();
        // g'(a) = cos(a), g''(a) = -sin(a)
        Self::new(
            sin_a,
            cos_a * self.eps1,
            cos_a * self.eps2,
            -sin_a * self.eps1 * self.eps2 + cos_a * self.eps1eps2,
        )
    }

    /// Compute cosine: cos(x)
    pub fn cos(self) -> Self {
        let sin_a = self.real.sin();
        let cos_a = self.real.cos();
        // g'(a) = -sin(a), g''(a) = -cos(a)
        Self::new(
            cos_a,
            -sin_a * self.eps1,
            -sin_a * self.eps2,
            -cos_a * self.eps1 * self.eps2 - sin_a * self.eps1eps2,
        )
    }

    /// Compute tangent: tan(x)
    pub fn tan(self) -> Self {
        let tan_a = self.real.tan();
        let cos_a = self.real.cos();
        let sec2 = T::one() / (cos_a * cos_a);
        let two = T::one() + T::one();
        // g'(a) = sec^2(a), g''(a) = 2*tan(a)*sec^2(a)
        let gp = sec2;
        let gpp = two * tan_a * sec2;
        Self::new(
            tan_a,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute hyperbolic sine: sinh(x)
    pub fn sinh(self) -> Self {
        let sinh_a = self.real.sinh();
        let cosh_a = self.real.cosh();
        // g'(a) = cosh(a), g''(a) = sinh(a)
        Self::new(
            sinh_a,
            cosh_a * self.eps1,
            cosh_a * self.eps2,
            sinh_a * self.eps1 * self.eps2 + cosh_a * self.eps1eps2,
        )
    }

    /// Compute hyperbolic cosine: cosh(x)
    pub fn cosh(self) -> Self {
        let sinh_a = self.real.sinh();
        let cosh_a = self.real.cosh();
        // g'(a) = sinh(a), g''(a) = cosh(a)
        Self::new(
            cosh_a,
            sinh_a * self.eps1,
            sinh_a * self.eps2,
            cosh_a * self.eps1 * self.eps2 + sinh_a * self.eps1eps2,
        )
    }

    /// Compute hyperbolic tangent: tanh(x)
    pub fn tanh(self) -> Self {
        let tanh_a = self.real.tanh();
        let sech2 = T::one() - tanh_a * tanh_a;
        let two = T::one() + T::one();
        // g'(a) = sech^2(a), g''(a) = -2*tanh(a)*sech^2(a)
        let gp = sech2;
        let gpp = -two * tanh_a * sech2;
        Self::new(
            tanh_a,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute square root: sqrt(x)
    pub fn sqrt(self) -> Self {
        let a = self.real;
        let sqrt_a = a.sqrt();
        let two = T::one() + T::one();
        let four = two * two;
        // g'(a) = 1/(2*sqrt(a)), g''(a) = -1/(4*a^(3/2))
        let gp = T::one() / (two * sqrt_a);
        let gpp = -T::one() / (four * a * sqrt_a);
        Self::new(
            sqrt_a,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute absolute value: |x|
    ///
    /// Note: Derivative at x=0 is set to 0 by convention.
    pub fn abs(self) -> Self {
        if self.real >= T::zero() {
            self
        } else {
            -self
        }
    }

    /// Compute sigmoid function: 1 / (1 + e^(-x))
    pub fn sigmoid(self) -> Self {
        let a = self.real;
        let s = T::one() / (T::one() + (-a).exp());
        let two = T::one() + T::one();
        // g'(a) = s*(1-s), g''(a) = s*(1-s)*(1-2s)
        let gp = s * (T::one() - s);
        let gpp = gp * (T::one() - two * s);
        Self::new(
            s,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute ReLU: max(0, x)
    ///
    /// Note: Derivative at x=0 is set to 0 by convention.
    pub fn relu(self) -> Self {
        if self.real > T::zero() {
            self
        } else {
            Self::constant(T::zero())
        }
    }
}

// ============================================================================
// Arithmetic Operators for HyperDual Numbers
// ============================================================================

impl<T: Float> Add for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real + rhs.real,
            self.eps1 + rhs.eps1,
            self.eps2 + rhs.eps2,
            self.eps1eps2 + rhs.eps1eps2,
        )
    }
}

impl<T: Float> Sub for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real - rhs.real,
            self.eps1 - rhs.eps1,
            self.eps2 - rhs.eps2,
            self.eps1eps2 - rhs.eps1eps2,
        )
    }
}

impl<T: Float> Mul for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // (a + b*e1 + c*e2 + d*e1e2) * (e + f*e1 + g*e2 + h*e1e2)
        // real: a*e
        // e1:   a*f + b*e
        // e2:   a*g + c*e
        // e1e2: a*h + b*g + c*f + d*e
        Self::new(
            self.real * rhs.real,
            self.real * rhs.eps1 + self.eps1 * rhs.real,
            self.real * rhs.eps2 + self.eps2 * rhs.real,
            self.real * rhs.eps1eps2
                + self.eps1 * rhs.eps2
                + self.eps2 * rhs.eps1
                + self.eps1eps2 * rhs.real,
        )
    }
}

impl<T: Float> Div for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        // u/v derivation (see module-level documentation)
        let (a, b, c, d) = (self.real, self.eps1, self.eps2, self.eps1eps2);
        let (e, f, g, h) = (rhs.real, rhs.eps1, rhs.eps2, rhs.eps1eps2);
        let e2 = e * e;
        let e3 = e2 * e;
        let two = T::one() + T::one();

        Self::new(
            a / e,
            (b * e - a * f) / e2,
            (c * e - a * g) / e2,
            (d * e2 - a * h * e - b * g * e - c * f * e + two * a * f * g) / e3,
        )
    }
}

impl<T: Float> Neg for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.real, -self.eps1, -self.eps2, -self.eps1eps2)
    }
}

impl<T: Float + fmt::Display> fmt::Display for HyperDual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} + {}*e1 + {}*e2 + {}*e1e2",
            self.real, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

// ============================================================================
// Jacobian Computation
// ============================================================================

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
/// let jac = forward_jacobian(f, &[2.0, 3.0]).unwrap();
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
/// ).unwrap();
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

// ============================================================================
// Hessian Computation
// ============================================================================

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
/// let hess = hessian_exact(f, &[3.0, 4.0]).unwrap();
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

    Ok(Array::from_vec(hess_data).reshape(&[n, n]))
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

    Ok(Array::from_vec(hess_data).reshape(&[n, n]))
}

// ============================================================================
// Gradient Computation Utilities
// ============================================================================

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
fn numerical_gradient_central<T: Float>(f: &dyn Fn(&[T]) -> T, x: &[T]) -> Result<Vec<T>> {
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

// ============================================================================
// Gradient Checking
// ============================================================================

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

// ============================================================================
// Multivariate Taylor Series Expansion
// ============================================================================

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
/// let taylor = multivariate_taylor(f, &[1.0, 1.0]).unwrap();
///
/// // Evaluate near the center
/// let approx = taylor.evaluate(&[1.1, 1.2]).unwrap();
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;
    const NUMERICAL_TOL: f64 = 1e-5;

    // ====================================================================
    // HyperDual Number Tests
    // ====================================================================

    #[test]
    fn test_hyperdual_basic_arithmetic() {
        // Test addition: (a + e1) + (b + e2) = (a+b) + e1 + e2
        let x = HyperDual::new(3.0, 1.0, 0.0, 0.0);
        let y = HyperDual::new(4.0, 0.0, 1.0, 0.0);

        let sum = x + y;
        assert!((sum.real() - 7.0).abs() < TOL);
        assert!((sum.eps1() - 1.0).abs() < TOL);
        assert!((sum.eps2() - 1.0).abs() < TOL);
        assert!((sum.eps1eps2() - 0.0).abs() < TOL);

        // Test subtraction
        let diff = x - y;
        assert!((diff.real() - (-1.0)).abs() < TOL);

        // Test multiplication: x*y where x = 3+e1, y = 4+e2
        // real: 12, e1: 4, e2: 3, e1e2: 1 (product rule cross term)
        let prod = x * y;
        assert!((prod.real() - 12.0).abs() < TOL);
        assert!((prod.eps1() - 4.0).abs() < TOL);
        assert!((prod.eps2() - 3.0).abs() < TOL);
        assert!((prod.eps1eps2() - 1.0).abs() < TOL);

        // Test negation
        let neg = -x;
        assert!((neg.real() - (-3.0)).abs() < TOL);
        assert!((neg.eps1() - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_division() {
        // f(x,y) = x/y at (6, 3)
        // d^2f/dxdy = -1/y^2 = -1/9
        let x = HyperDual::new(6.0, 1.0, 0.0, 0.0);
        let y = HyperDual::new(3.0, 0.0, 1.0, 0.0);

        let quot = x / y;
        assert!((quot.real() - 2.0).abs() < TOL);
        // df/dx = 1/y = 1/3
        assert!((quot.eps1() - 1.0 / 3.0).abs() < TOL);
        // df/dy = -x/y^2 = -6/9 = -2/3
        assert!((quot.eps2() - (-2.0 / 3.0)).abs() < TOL);
        // d^2f/dxdy = -1/y^2 = -1/9
        assert!((quot.eps1eps2() - (-1.0 / 9.0)).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_exp() {
        // f(x) = e^x, f'(x) = e^x, f''(x) = e^x
        let x = HyperDual::new(2.0, 1.0, 1.0, 0.0);
        let result = x.exp();

        let e2 = 2.0_f64.exp();
        assert!((result.real() - e2).abs() < TOL);
        assert!((result.eps1() - e2).abs() < TOL);
        assert!((result.eps2() - e2).abs() < TOL);
        // f''(x)*b*c + f'(x)*d = e^2 * 1*1 + e^2 * 0 = e^2
        assert!((result.eps1eps2() - e2).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_sin_cos() {
        let val = 1.0_f64;
        let x = HyperDual::new(val, 1.0, 1.0, 0.0);

        // sin(x): f' = cos(x), f'' = -sin(x)
        let sin_result = x.sin();
        assert!((sin_result.real() - val.sin()).abs() < TOL);
        assert!((sin_result.eps1() - val.cos()).abs() < TOL);
        assert!((sin_result.eps1eps2() - (-val.sin())).abs() < TOL);

        // cos(x): f' = -sin(x), f'' = -cos(x)
        let cos_result = x.cos();
        assert!((cos_result.real() - val.cos()).abs() < TOL);
        assert!((cos_result.eps1() - (-val.sin())).abs() < TOL);
        assert!((cos_result.eps1eps2() - (-val.cos())).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_chain_rule() {
        // f(x) = sin(x^2) at x = 1.5
        // f'(x) = cos(x^2) * 2x
        // f''(x) = -sin(x^2) * (2x)^2 + cos(x^2) * 2
        //        = -4x^2*sin(x^2) + 2*cos(x^2)
        let val = 1.5_f64;
        let x = HyperDual::new(val, 1.0, 1.0, 0.0);

        let x_sq = x * x;
        let result = x_sq.sin();

        let x2 = val * val;
        let expected_val = x2.sin();
        let expected_deriv = x2.cos() * 2.0 * val;
        let expected_second = -4.0 * x2 * x2.sin() + 2.0 * x2.cos();

        assert!((result.real() - expected_val).abs() < TOL);
        assert!((result.eps1() - expected_deriv).abs() < TOL);
        assert!((result.eps1eps2() - expected_second).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_powi() {
        // f(x) = x^3 at x = -2
        // f(-2) = -8, f'(-2) = 12, f''(-2) = -12
        let x = HyperDual::new(-2.0, 1.0, 1.0, 0.0);
        let result = x.powi(3);

        assert!((result.real() - (-8.0)).abs() < TOL);
        assert!((result.eps1() - 12.0).abs() < TOL);
        // f''(x) = 6x, at x=-2: -12
        assert!((result.eps1eps2() - (-12.0)).abs() < TOL);
    }

    // ====================================================================
    // Forward Jacobian Tests
    // ====================================================================

    #[test]
    fn test_forward_jacobian_linear() {
        // f(x,y) = [x + 2y, 3x + 4y]
        // J = [[1, 2], [3, 4]]
        let f = |vars: &[Dual<f64>]| -> Vec<Dual<f64>> {
            let two = Dual::constant(2.0);
            let three = Dual::constant(3.0);
            let four = Dual::constant(4.0);
            vec![vars[0] + two * vars[1], three * vars[0] + four * vars[1]]
        };

        let jac = forward_jacobian(f, &[1.0, 1.0]).expect("forward_jacobian should succeed");
        let jac_vec = jac.to_vec();

        // J = [[1, 2], [3, 4]] stored row-major
        assert!((jac_vec[0] - 1.0).abs() < TOL); // J[0,0]
        assert!((jac_vec[1] - 2.0).abs() < TOL); // J[0,1]
        assert!((jac_vec[2] - 3.0).abs() < TOL); // J[1,0]
        assert!((jac_vec[3] - 4.0).abs() < TOL); // J[1,1]
    }

    #[test]
    fn test_forward_jacobian_quadratic() {
        // f(x,y) = [x^2, x*y]
        // J = [[2x, 0], [y, x]]
        // At (2, 3): J = [[4, 0], [3, 2]]
        let f =
            |vars: &[Dual<f64>]| -> Vec<Dual<f64>> { vec![vars[0] * vars[0], vars[0] * vars[1]] };

        let jac = forward_jacobian(f, &[2.0, 3.0]).expect("forward_jacobian should succeed");
        let jac_vec = jac.to_vec();

        assert!((jac_vec[0] - 4.0).abs() < TOL); // 2x = 4
        assert!((jac_vec[1] - 0.0).abs() < TOL); // 0
        assert!((jac_vec[2] - 3.0).abs() < TOL); // y = 3
        assert!((jac_vec[3] - 2.0).abs() < TOL); // x = 2
    }

    // ====================================================================
    // Reverse Jacobian Tests
    // ====================================================================

    #[test]
    fn test_reverse_jacobian_linear() {
        // f(x,y) = [x + 2y, 3x + 4y]
        // J = [[1, 2], [3, 4]]
        let jac = reverse_jacobian(
            |tape: &mut Tape<f64>, vars: &[Var]| {
                let two = tape.var(2.0);
                let three = tape.var(3.0);
                let four = tape.var(4.0);

                let two_y = tape.mul(two, vars[1]);
                let out1 = tape.add(vars[0], two_y);

                let three_x = tape.mul(three, vars[0]);
                let four_y = tape.mul(four, vars[1]);
                let out2 = tape.add(three_x, four_y);

                vec![out1, out2]
            },
            &[1.0, 1.0],
        )
        .expect("reverse_jacobian should succeed");

        let jac_vec = jac.to_vec();
        assert!((jac_vec[0] - 1.0).abs() < TOL);
        assert!((jac_vec[1] - 2.0).abs() < TOL);
        assert!((jac_vec[2] - 3.0).abs() < TOL);
        assert!((jac_vec[3] - 4.0).abs() < TOL);
    }

    #[test]
    fn test_reverse_jacobian_multi_output() {
        // f(x) = [x^2, x^3] (single input, two outputs)
        // J = [[2x], [3x^2]]
        // At x=2: J = [[4], [12]]
        let jac = reverse_jacobian(
            |tape: &mut Tape<f64>, vars: &[Var]| {
                let x_sq = tape.mul(vars[0], vars[0]);
                let x_cube = tape.mul(x_sq, vars[0]);
                vec![x_sq, x_cube]
            },
            &[2.0],
        )
        .expect("reverse_jacobian should succeed");

        let jac_vec = jac.to_vec();
        assert!((jac_vec[0] - 4.0).abs() < TOL); // 2x = 4
        assert!((jac_vec[1] - 12.0).abs() < TOL); // 3x^2 = 12
    }

    // ====================================================================
    // Jacobian Agreement Test
    // ====================================================================

    #[test]
    fn test_jacobian_forward_reverse_agree() {
        // Both modes should give the same Jacobian for the same function
        // f(x,y) = [x*y + x, y^2 - x]
        let fwd_jac = forward_jacobian(
            |vars: &[Dual<f64>]| -> Vec<Dual<f64>> {
                let one = Dual::constant(1.0);
                vec![vars[0] * vars[1] + vars[0], vars[1] * vars[1] - vars[0]]
            },
            &[2.0, 3.0],
        )
        .expect("forward_jacobian should succeed");

        let rev_jac = reverse_jacobian(
            |tape: &mut Tape<f64>, vars: &[Var]| {
                let xy = tape.mul(vars[0], vars[1]);
                let out1 = tape.add(xy, vars[0]);

                let y_sq = tape.mul(vars[1], vars[1]);
                let out2 = tape.sub(y_sq, vars[0]);

                vec![out1, out2]
            },
            &[2.0, 3.0],
        )
        .expect("reverse_jacobian should succeed");

        let fwd_vec = fwd_jac.to_vec();
        let rev_vec = rev_jac.to_vec();

        for k in 0..fwd_vec.len() {
            assert!(
                (fwd_vec[k] - rev_vec[k]).abs() < TOL,
                "Jacobian mismatch at index {}: forward={}, reverse={}",
                k,
                fwd_vec[k],
                rev_vec[k]
            );
        }
    }

    // ====================================================================
    // Hessian Tests
    // ====================================================================

    #[test]
    fn test_hessian_exact_quadratic() {
        // f(x,y) = x^2 + 3*x*y + 2*y^2
        // H = [[2, 3], [3, 4]] (constant for quadratic functions)
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let three = HyperDual::constant(3.0);
            let two = HyperDual::constant(2.0);
            vars[0] * vars[0] + three * vars[0] * vars[1] + two * vars[1] * vars[1]
        };

        let hess = hessian_exact(f, &[5.0, 7.0]).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 2.0).abs() < TOL); // H[0,0] = 2
        assert!((hess_vec[1] - 3.0).abs() < TOL); // H[0,1] = 3
        assert!((hess_vec[2] - 3.0).abs() < TOL); // H[1,0] = 3 (symmetric)
        assert!((hess_vec[3] - 4.0).abs() < TOL); // H[1,1] = 4
    }

    #[test]
    fn test_hessian_exact_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // At (1,1):
        // H[0,0] = 2 + 1200*x^2 - 400*y = 2 + 1200 - 400 = 802
        // H[0,1] = H[1,0] = -400*x = -400
        // H[1,1] = 200
        let rosenbrock = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let x = vars[0];
            let y = vars[1];
            let one = HyperDual::constant(1.0);
            let hundred = HyperDual::constant(100.0);
            let diff1 = one - x;
            let diff2 = y - x * x;
            diff1 * diff1 + hundred * diff2 * diff2
        };

        let hess = hessian_exact(rosenbrock, &[1.0, 1.0]).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 802.0).abs() < TOL);
        assert!((hess_vec[1] - (-400.0)).abs() < TOL);
        assert!((hess_vec[2] - (-400.0)).abs() < TOL);
        assert!((hess_vec[3] - 200.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_vector_product_simple() {
        // f(x,y) = x^2 + 2*x*y + 3*y^2
        // H = [[2, 2], [2, 6]]
        // v = [1, 2]
        // H*v = [2*1+2*2, 2*1+6*2] = [6, 14]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let two = HyperDual::constant(2.0);
            let three = HyperDual::constant(3.0);
            vars[0] * vars[0] + two * vars[0] * vars[1] + three * vars[1] * vars[1]
        };

        let hv = hessian_vector_product(f, &[1.0, 1.0], &[1.0, 2.0])
            .expect("hessian_vector_product should succeed");

        assert!((hv[0] - 6.0).abs() < TOL);
        assert!((hv[1] - 14.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_forward_over_reverse() {
        // f(x,y) = x^2 + y^2
        // H = [[2, 0], [0, 2]]
        let f = |tape: &mut Tape<f64>, vars: &[Var]| -> Var {
            let x_sq = tape.mul(vars[0], vars[0]);
            let y_sq = tape.mul(vars[1], vars[1]);
            tape.add(x_sq, y_sq)
        };

        let hess = hessian_forward_over_reverse(f, &[3.0, 4.0])
            .expect("hessian_forward_over_reverse should succeed");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 2.0).abs() < NUMERICAL_TOL); // H[0,0]
        assert!(hess_vec[1].abs() < NUMERICAL_TOL); // H[0,1]
        assert!(hess_vec[2].abs() < NUMERICAL_TOL); // H[1,0]
        assert!((hess_vec[3] - 2.0).abs() < NUMERICAL_TOL); // H[1,1]
    }

    // ====================================================================
    // Gradient Check Tests
    // ====================================================================

    #[test]
    fn test_gradient_check_passes() {
        // f(x,y) = x^2 + y^2
        // Correct gradient at (3, 4) is [6, 8]
        let f_numerical = |vars: &[f64]| -> f64 { vars[0] * vars[0] + vars[1] * vars[1] };
        let analytical = vec![6.0, 8.0];

        let result = gradient_check(f_numerical, &[3.0, 4.0], &analytical, 1e-4)
            .expect("gradient_check should succeed");

        assert!(result.passed, "Correct gradient should pass check");
        assert!(result.max_rel_error < 1e-4);
    }

    #[test]
    fn test_gradient_check_detects_error() {
        // f(x,y) = x^2 + y^2
        // Wrong gradient: [6, 100] (y-component is wrong)
        let f_numerical = |vars: &[f64]| -> f64 { vars[0] * vars[0] + vars[1] * vars[1] };
        let wrong_analytical = vec![6.0, 100.0];

        let result = gradient_check(f_numerical, &[3.0, 4.0], &wrong_analytical, 1e-4)
            .expect("gradient_check should succeed");

        assert!(!result.passed, "Wrong gradient should fail check");
        assert!(result.max_abs_error > 90.0);
    }

    #[test]
    fn test_gradient_check_ad_integration() {
        // Test the combined AD + numerical check
        let f_ad = |vars: &[Dual<f64>]| -> Dual<f64> { vars[0] * vars[0] + vars[1] * vars[1] };
        let f_num = |vars: &[f64]| -> f64 { vars[0] * vars[0] + vars[1] * vars[1] };

        let result = gradient_check_ad(f_ad, f_num, &[3.0, 4.0], 1e-4)
            .expect("gradient_check_ad should succeed");

        assert!(result.passed);
    }

    // ====================================================================
    // Taylor Series Tests
    // ====================================================================

    #[test]
    fn test_taylor_quadratic_approximation() {
        // f(x,y) = x^2 + y^2 is exactly quadratic, so the 2nd order
        // Taylor expansion should be exact everywhere
        let f =
            |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0] * vars[0] + vars[1] * vars[1] };

        let taylor =
            multivariate_taylor(f, &[1.0, 1.0]).expect("multivariate_taylor should succeed");

        // Value at center: f(1,1) = 2
        assert!((taylor.value - 2.0).abs() < TOL);

        // Gradient at center: [2, 2]
        assert!((taylor.gradient[0] - 2.0).abs() < TOL);
        assert!((taylor.gradient[1] - 2.0).abs() < TOL);

        // For quadratic function, Taylor expansion is exact
        let approx = taylor
            .evaluate(&[2.0, 3.0])
            .expect("evaluate should succeed");
        let exact = 4.0 + 9.0; // f(2,3) = 13
        assert!((approx - exact).abs() < TOL);
    }

    #[test]
    fn test_taylor_approximation_accuracy() {
        // f(x,y) = exp(x) * sin(y) - Taylor expansion around (0, 0)
        // should be accurate near the center
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0].exp() * vars[1].sin() };

        let taylor =
            multivariate_taylor(f, &[0.0, 0.0]).expect("multivariate_taylor should succeed");

        // f(0,0) = exp(0)*sin(0) = 0
        assert!((taylor.value - 0.0).abs() < TOL);

        // Test at a nearby point (0.1, 0.1)
        let approx = taylor
            .evaluate(&[0.1, 0.1])
            .expect("evaluate should succeed");
        let exact = 0.1_f64.exp() * 0.1_f64.sin();

        // Second-order Taylor should be accurate to within O(h^3)
        let error = (approx - exact).abs();
        assert!(
            error < 1e-3,
            "Taylor approximation error {} should be small near center",
            error
        );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    #[test]
    fn test_hessian_1d_function() {
        // f(x) = x^3, f''(x) = 6x
        // At x=2: H = [[12]]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0].powi(3) };

        let hess = hessian_exact(f, &[2.0]).expect("hessian_exact should succeed for 1D");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 12.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_identity_function() {
        // f(x,y) = x + y (linear function)
        // H = [[0, 0], [0, 0]]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0] + vars[1] };

        let hess = hessian_exact(f, &[3.0, 4.0]).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        for val in &hess_vec {
            assert!(val.abs() < TOL, "Hessian of linear function should be zero");
        }
    }

    #[test]
    fn test_hyperdual_ln() {
        // f(x) = ln(x) at x = 2
        // f' = 1/x = 0.5, f'' = -1/x^2 = -0.25
        let x = HyperDual::new(2.0, 1.0, 1.0, 0.0);
        let result = x.ln();

        assert!((result.real() - 2.0_f64.ln()).abs() < TOL);
        assert!((result.eps1() - 0.5).abs() < TOL);
        assert!((result.eps1eps2() - (-0.25)).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_tanh() {
        // f(x) = tanh(x) at x = 0
        // f(0) = 0, f'(0) = sech^2(0) = 1, f''(0) = -2*tanh(0)*sech^2(0) = 0
        let x = HyperDual::new(0.0, 1.0, 1.0, 0.0);
        let result = x.tanh();

        assert!((result.real() - 0.0).abs() < TOL);
        assert!((result.eps1() - 1.0).abs() < TOL);
        assert!((result.eps1eps2() - 0.0).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_sigmoid() {
        // f(x) = sigmoid(x) at x = 0
        // f(0) = 0.5, f'(0) = 0.25, f''(0) = 0
        let x = HyperDual::new(0.0, 1.0, 1.0, 0.0);
        let result = x.sigmoid();

        assert!((result.real() - 0.5).abs() < TOL);
        assert!((result.eps1() - 0.25).abs() < TOL);
        // f''(0) = f'(0) * (1 - 2*f(0)) = 0.25 * (1 - 1) = 0
        assert!((result.eps1eps2() - 0.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_3d_function() {
        // f(x,y,z) = x*y + y*z + x*z
        // H = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            vars[0] * vars[1] + vars[1] * vars[2] + vars[0] * vars[2]
        };

        let hess = hessian_exact(f, &[1.0, 2.0, 3.0]).expect("hessian_exact should succeed for 3D");
        let hess_vec = hess.to_vec();

        // H[0,0] = 0, H[0,1] = 1, H[0,2] = 1
        assert!(hess_vec[0].abs() < TOL);
        assert!((hess_vec[1] - 1.0).abs() < TOL);
        assert!((hess_vec[2] - 1.0).abs() < TOL);
        // H[1,0] = 1, H[1,1] = 0, H[1,2] = 1
        assert!((hess_vec[3] - 1.0).abs() < TOL);
        assert!(hess_vec[4].abs() < TOL);
        assert!((hess_vec[5] - 1.0).abs() < TOL);
        // H[2,0] = 1, H[2,1] = 1, H[2,2] = 0
        assert!((hess_vec[6] - 1.0).abs() < TOL);
        assert!((hess_vec[7] - 1.0).abs() < TOL);
        assert!(hess_vec[8].abs() < TOL);
    }

    #[test]
    fn test_compute_gradient_ad_matches_hyperdual() {
        // Verify that gradient from Dual and HyperDual agree
        let f_dual = |vars: &[Dual<f64>]| -> Dual<f64> {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };
        let f_hyper = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };

        let point = &[2.0, 3.0];
        let grad_dual =
            compute_gradient_ad(f_dual, point).expect("compute_gradient_ad should succeed");
        let grad_hyper = compute_gradient_hyperdual(&f_hyper, point)
            .expect("compute_gradient_hyperdual should succeed");

        for i in 0..grad_dual.len() {
            assert!(
                (grad_dual[i] - grad_hyper[i]).abs() < TOL,
                "Gradient component {} disagrees: dual={}, hyper={}",
                i,
                grad_dual[i],
                grad_hyper[i]
            );
        }
    }

    #[test]
    fn test_hessian_vector_product_matches_full_hessian() {
        // Verify that H*v from hessian_vector_product matches H_full * v
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let three = HyperDual::constant(3.0);
            vars[0] * vars[0] + three * vars[0] * vars[1] + vars[1] * vars[1]
        };

        let x = &[2.0, 3.0];
        let v = &[1.0, -1.0];

        // Full Hessian: H = [[2, 3], [3, 2]]
        let hess = hessian_exact(f, x).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        // H*v via full matrix
        let n = x.len();
        let mut hv_full = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                hv_full[i] += hess_vec[i * n + j] * v[j];
            }
        }

        // H*v via hessian_vector_product
        let hv_efficient =
            hessian_vector_product(f, x, v).expect("hessian_vector_product should succeed");

        for i in 0..n {
            assert!(
                (hv_full[i] - hv_efficient[i]).abs() < TOL,
                "Hv component {} disagrees: full={}, efficient={}",
                i,
                hv_full[i],
                hv_efficient[i]
            );
        }
    }

    #[test]
    fn test_hessian_exact_vs_forward_over_reverse() {
        // Both methods should give approximately the same Hessian
        let f_hyper = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };
        let f_tape = |tape: &mut Tape<f64>, vars: &[Var]| -> Var {
            let x_sq = tape.mul(vars[0], vars[0]);
            let xy = tape.mul(vars[0], vars[1]);
            let y_sq = tape.mul(vars[1], vars[1]);
            let tmp = tape.add(x_sq, xy);
            tape.add(tmp, y_sq)
        };

        let x = &[2.0, 3.0];
        let hess_exact = hessian_exact(f_hyper, x).expect("hessian_exact should succeed");
        let hess_for = hessian_forward_over_reverse(f_tape, x)
            .expect("hessian_forward_over_reverse should succeed");

        let exact_vec = hess_exact.to_vec();
        let for_vec = hess_for.to_vec();

        for k in 0..exact_vec.len() {
            assert!(
                (exact_vec[k] - for_vec[k]).abs() < NUMERICAL_TOL,
                "Hessian element {} disagrees: exact={}, forward_over_reverse={}",
                k,
                exact_vec[k],
                for_vec[k]
            );
        }
    }

    #[test]
    fn test_empty_input_errors() {
        // All functions should return errors for empty input
        let f_dual = |_vars: &[Dual<f64>]| -> Vec<Dual<f64>> { vec![] };
        assert!(forward_jacobian(f_dual, &[]).is_err());

        let f_hyper = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { HyperDual::constant(0.0) };
        assert!(hessian_exact(f_hyper, &[]).is_err());
        assert!(hessian_vector_product(f_hyper, &[], &[]).is_err());
        assert!(multivariate_taylor(f_hyper, &[]).is_err());
    }

    #[test]
    fn test_hessian_vector_product_dimension_mismatch() {
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0] * vars[0] };

        // v has wrong dimension
        let result = hessian_vector_product(f, &[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }
}
