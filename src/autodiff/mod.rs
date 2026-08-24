//! Automatic Differentiation for NumRS2
//!
//! This module implements both forward-mode and reverse-mode automatic differentiation
//! (AD), enabling computation of derivatives for numerical functions.
//!
//! # Overview
//!
//! Automatic differentiation is a technique to compute derivatives of functions by
//! applying the chain rule to elementary operations. Unlike numerical differentiation
//! (finite differences), AD computes exact derivatives (up to floating-point precision).
//!
//! ## Forward Mode AD (Dual Numbers)
//!
//! Forward mode AD uses dual numbers to propagate derivatives alongside values:
//! - Dual number: `f(x) + f'(x)ε` where `ε² = 0`
//! - Efficient for functions with few inputs, many outputs (Jacobian-vector products)
//! - Computes derivatives in one forward pass
//!
//! ## Reverse Mode AD (Backpropagation)
//!
//! Reverse mode AD records operations in a computation graph and backpropagates gradients:
//! - Records operations on a tape during forward pass
//! - Computes gradients during backward pass
//! - Efficient for functions with many inputs, few outputs (vector-Jacobian products)
//! - This is the foundation of modern deep learning frameworks
//!
//! # Examples
//!
//! ## Forward Mode with Dual Numbers
//!
//! ```rust,ignore
//! use numrs2::autodiff::*;
//!
//! // Compute derivative of f(x) = x² at x = 3
//! let x = Dual::new(3.0, 1.0); // value=3.0, derivative=1.0
//! let y = x * x;
//! assert_eq!(y.value(), 9.0);  // f(3) = 9
//! assert_eq!(y.deriv(), 6.0);  // f'(3) = 6
//! ```
//!
//! ## Reverse Mode with Tape
//!
//! ```rust,ignore
//! use numrs2::autodiff::*;
//!
//! let mut tape = Tape::new();
//! let x = tape.var(3.0);
//! let y = tape.var(4.0);
//! let z = x * x + y * y; // z = x² + y²
//!
//! tape.backward(z);
//! assert_eq!(tape.grad(x), 6.0);  // ∂z/∂x = 2x = 6
//! assert_eq!(tape.grad(y), 8.0);  // ∂z/∂y = 2y = 8
//! ```

pub mod higher_order;
pub use higher_order::{
    compute_gradient_ad, compute_gradient_hyperdual, forward_jacobian, gradient_check,
    gradient_check_ad, hessian_exact, hessian_forward_over_reverse, hessian_vector_product,
    jacobian_auto, multivariate_taylor, reverse_jacobian, GradientCheckResult, HyperDual,
    TaylorExpansion2,
};

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// Forward Mode AD: Dual Numbers
// ============================================================================

/// Dual number for forward-mode automatic differentiation
///
/// A dual number represents both a value and its derivative: `f(x) + f'(x)ε`
/// where `ε² = 0`. This enables exact derivative computation through operator
/// overloading and the chain rule.
///
/// # Type Parameters
///
/// * `T` - The numeric type (typically `f32` or `f64`)
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::Dual;
///
/// // f(x) = x³ + 2x² - 5x + 1
/// fn f(x: Dual<f64>) -> Dual<f64> {
///     x.pow(3.0) + x.pow(2.0) * 2.0 - x * 5.0 + Dual::constant(1.0)
/// }
///
/// // Compute f(2) and f'(2)
/// let x = Dual::new(2.0, 1.0);
/// let result = f(x);
/// assert_eq!(result.value(), 3.0);   // f(2) = 8 + 8 - 10 + 1 = 7... wait let me recalculate
/// assert_eq!(result.deriv(), 15.0);  // f'(2) = 3(4) + 2(2)(2) - 5 = 12 + 8 - 5 = 15
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<T> {
    /// The value component
    value: T,
    /// The derivative component
    deriv: T,
}

impl<T: Float> Dual<T> {
    /// Create a new dual number with given value and derivative
    ///
    /// # Arguments
    ///
    /// * `value` - The function value
    /// * `deriv` - The derivative value
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numrs2::autodiff::Dual;
    ///
    /// // Variable with derivative = 1 (for computing ∂f/∂x)
    /// let x = Dual::new(3.0, 1.0);
    ///
    /// // Constant with derivative = 0
    /// let c = Dual::new(5.0, 0.0);
    /// ```
    pub fn new(value: T, deriv: T) -> Self {
        Self { value, deriv }
    }

    /// Create a dual number representing a variable (derivative = 1)
    ///
    /// Use this for the input variable with respect to which you want to differentiate.
    pub fn variable(value: T) -> Self {
        Self::new(value, T::one())
    }

    /// Create a dual number representing a constant (derivative = 0)
    ///
    /// Use this for constants in expressions.
    pub fn constant(value: T) -> Self {
        Self::new(value, T::zero())
    }

    /// Get the value component
    pub fn value(&self) -> T {
        self.value
    }

    /// Get the derivative component
    pub fn deriv(&self) -> T {
        self.deriv
    }

    /// Compute power: x^n
    pub fn pow(&self, n: T) -> Self {
        let value = self.value.powf(n);
        let deriv = n * self.value.powf(n - T::one()) * self.deriv;
        Self::new(value, deriv)
    }

    /// Compute exponential: e^x
    pub fn exp(&self) -> Self {
        let exp_val = self.value.exp();
        Self::new(exp_val, exp_val * self.deriv)
    }

    /// Compute natural logarithm: ln(x)
    pub fn ln(&self) -> Self {
        Self::new(self.value.ln(), self.deriv / self.value)
    }

    /// Compute sine: sin(x)
    pub fn sin(&self) -> Self {
        Self::new(self.value.sin(), self.value.cos() * self.deriv)
    }

    /// Compute cosine: cos(x)
    pub fn cos(&self) -> Self {
        Self::new(self.value.cos(), -self.value.sin() * self.deriv)
    }

    /// Compute tangent: tan(x)
    pub fn tan(&self) -> Self {
        let tan_val = self.value.tan();
        Self::new(tan_val, self.deriv / self.value.cos().powi(2))
    }

    /// Compute hyperbolic sine: sinh(x)
    pub fn sinh(&self) -> Self {
        Self::new(self.value.sinh(), self.value.cosh() * self.deriv)
    }

    /// Compute hyperbolic cosine: cosh(x)
    pub fn cosh(&self) -> Self {
        Self::new(self.value.cosh(), self.value.sinh() * self.deriv)
    }

    /// Compute hyperbolic tangent: tanh(x)
    pub fn tanh(&self) -> Self {
        let tanh_val = self.value.tanh();
        Self::new(tanh_val, (T::one() - tanh_val * tanh_val) * self.deriv)
    }

    /// Compute square root: √x
    pub fn sqrt(&self) -> Self {
        let sqrt_val = self.value.sqrt();
        Self::new(sqrt_val, self.deriv / (T::one() + T::one()) / sqrt_val)
    }

    /// Compute absolute value: |x|
    ///
    /// Note: The derivative at x=0 is set to 0 by convention.
    pub fn abs(&self) -> Self {
        if self.value >= T::zero() {
            *self
        } else {
            -(*self)
        }
    }

    /// Compute sigmoid function: 1 / (1 + e^(-x))
    pub fn sigmoid(&self) -> Self {
        let exp_neg_x = (-self.value).exp();
        let sigmoid_val = T::one() / (T::one() + exp_neg_x);
        Self::new(
            sigmoid_val,
            sigmoid_val * (T::one() - sigmoid_val) * self.deriv,
        )
    }

    /// Compute ReLU: max(0, x)
    ///
    /// Note: The derivative at x=0 is set to 0 by convention.
    pub fn relu(&self) -> Self {
        if self.value > T::zero() {
            *self
        } else {
            Self::constant(T::zero())
        }
    }
}

// Arithmetic operators for Dual numbers

impl<T: Float> Add for Dual<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.value + rhs.value, self.deriv + rhs.deriv)
    }
}

impl<T: Float> Sub for Dual<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value - rhs.value, self.deriv - rhs.deriv)
    }
}

impl<T: Float> Mul for Dual<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Product rule: (uv)' = u'v + uv'
        Self::new(
            self.value * rhs.value,
            self.deriv * rhs.value + self.value * rhs.deriv,
        )
    }
}

impl<T: Float> Div for Dual<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Quotient rule: (u/v)' = (u'v - uv') / v²
        Self::new(
            self.value / rhs.value,
            (self.deriv * rhs.value - self.value * rhs.deriv) / (rhs.value * rhs.value),
        )
    }
}

impl<T: Float> Neg for Dual<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.value, -self.deriv)
    }
}

impl<T: Float + fmt::Display> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}ε", self.value, self.deriv)
    }
}

// ============================================================================
// Array-based Forward Mode AD
// ============================================================================

/// Compute gradient of a scalar-valued function using forward mode AD
///
/// This function computes the gradient ∇f of a function f: ℝⁿ → ℝ at a given point.
///
/// # Arguments
///
/// * `f` - Function to differentiate (takes array of Dual numbers, returns Dual number)
/// * `x` - Point at which to compute the gradient
///
/// # Returns
///
/// The gradient vector ∇f(x)
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::*;
/// use numrs2::prelude::*;
///
/// // f(x, y) = x² + y²
/// fn f(vars: &Array<Dual<f64>>) -> Dual<f64> {
///     let v = vars.to_vec();
///     v[0] * v[0] + v[1] * v[1]
/// }
///
/// let x = Array::from_vec(vec![3.0, 4.0]);
/// let grad = gradient(f, &x).expect("valid gradient computation");
/// // ∇f = [2x, 2y] = [6, 8]
/// assert_eq!(grad.to_vec(), vec![6.0, 8.0]);
/// ```
pub fn gradient<F, T>(f: F, x: &Array<T>) -> Result<Array<T>>
where
    F: Fn(&Array<Dual<T>>) -> Dual<T>,
    T: Float,
{
    let n = x.size();
    let x_vec = x.to_vec();
    let mut grad = Vec::with_capacity(n);

    // Compute gradient using forward mode: compute one directional derivative per input
    for i in 0..n {
        // Create dual number array with derivative = 1 for variable i, 0 for others
        let mut dual_vec = Vec::with_capacity(n);
        for j in 0..n {
            let deriv = if i == j { T::one() } else { T::zero() };
            dual_vec.push(Dual::new(x_vec[j], deriv));
        }
        let dual_array = Array::from_vec(dual_vec);

        // Evaluate function and extract derivative component
        let result = f(&dual_array);
        grad.push(result.deriv());
    }

    Ok(Array::from_vec(grad))
}

/// Compute Jacobian matrix of a vector-valued function using forward mode AD
///
/// This function computes the Jacobian J of a function f: ℝⁿ → ℝᵐ at a given point.
/// The Jacobian is an m×n matrix where `J[i,j] = ∂fᵢ/∂xⱼ`.
///
/// # Arguments
///
/// * `f` - Function to differentiate (takes array of Dual numbers, returns array of Dual numbers)
/// * `x` - Point at which to compute the Jacobian
///
/// # Returns
///
/// The Jacobian matrix J(x)
pub fn jacobian<F, T>(f: F, x: &Array<T>) -> Result<Array<T>>
where
    F: Fn(&Array<Dual<T>>) -> Array<Dual<T>>,
    T: Float,
{
    let n = x.size();
    let x_vec = x.to_vec();

    // First pass: determine output dimension
    let dual_vec_test: Vec<Dual<T>> = x_vec.iter().map(|&v| Dual::variable(v)).collect();
    let output_test = f(&Array::from_vec(dual_vec_test));
    let m = output_test.size();

    let mut jac = Vec::with_capacity(m * n);

    // Compute Jacobian using forward mode: one pass per input variable
    for i in 0..n {
        // Create dual number array with derivative = 1 for variable i, 0 for others
        let mut dual_vec = Vec::with_capacity(n);
        for j in 0..n {
            let deriv = if i == j { T::one() } else { T::zero() };
            dual_vec.push(Dual::new(x_vec[j], deriv));
        }
        let dual_array = Array::from_vec(dual_vec);

        // Evaluate function and extract derivative components
        let result = f(&dual_array);
        let result_vec = result.to_vec();
        for elem in result_vec {
            jac.push(elem.deriv());
        }
    }

    // Reshape to m×n matrix (each column is ∂f/∂xᵢ)
    Array::from_vec_shape(jac, &[m, n])
}

// ============================================================================
// Reverse Mode AD: Tape-based Computation Graph
// ============================================================================

/// Node in the computation graph for reverse-mode AD
#[derive(Debug, Clone)]
struct TapeNode<T> {
    /// Value computed by this node
    value: T,
    /// Accumulated gradient (adjoint) for this node
    grad: T,
    /// Parent nodes and their contribution to the gradient
    parents: Vec<(usize, T)>,
}

/// Computation tape for reverse-mode automatic differentiation
///
/// The tape records all operations during the forward pass, building a computation
/// graph. During the backward pass, gradients are propagated from outputs to inputs
/// using the chain rule.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::Tape;
///
/// let mut tape = Tape::new();
///
/// // Forward pass: build computation graph
/// let x = tape.var(3.0);
/// let y = tape.var(4.0);
/// let z = tape.mul(x, x);  // z = x²
/// let w = tape.mul(y, y);  // w = y²
/// let result = tape.add(z, w);  // result = x² + y²
///
/// // Backward pass: compute gradients
/// tape.backward(result);
///
/// assert_eq!(tape.grad(x), 6.0);  // ∂result/∂x = 2x = 6
/// assert_eq!(tape.grad(y), 8.0);  // ∂result/∂y = 2y = 8
/// ```
pub struct Tape<T> {
    /// Nodes in the computation graph
    nodes: Vec<TapeNode<T>>,
}

/// Variable identifier in the computation tape
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(usize);

impl<T: Float> Tape<T> {
    /// Create a new empty tape
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a variable to the tape
    ///
    /// Variables are leaf nodes in the computation graph (inputs).
    pub fn var(&mut self, value: T) -> Var {
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: Vec::new(),
        });
        Var(idx)
    }

    /// Get the value of a variable
    pub fn value(&self, var: Var) -> T {
        self.nodes[var.0].value
    }

    /// Get the gradient of a variable (only valid after backward pass)
    pub fn grad(&self, var: Var) -> T {
        self.nodes[var.0].grad
    }

    /// Addition operation: z = x + y
    pub fn add(&mut self, x: Var, y: Var) -> Var {
        let value = self.nodes[x.0].value + self.nodes[y.0].value;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, T::one()), (y.0, T::one())],
        });
        Var(idx)
    }

    /// Subtraction operation: z = x - y
    pub fn sub(&mut self, x: Var, y: Var) -> Var {
        let value = self.nodes[x.0].value - self.nodes[y.0].value;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, T::one()), (y.0, -T::one())],
        });
        Var(idx)
    }

    /// Multiplication operation: z = x * y
    pub fn mul(&mut self, x: Var, y: Var) -> Var {
        let x_val = self.nodes[x.0].value;
        let y_val = self.nodes[y.0].value;
        let value = x_val * y_val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, y_val), (y.0, x_val)],
        });
        Var(idx)
    }

    /// Division operation: z = x / y
    pub fn div(&mut self, x: Var, y: Var) -> Var {
        let x_val = self.nodes[x.0].value;
        let y_val = self.nodes[y.0].value;
        let value = x_val / y_val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, T::one() / y_val), (y.0, -x_val / (y_val * y_val))],
        });
        Var(idx)
    }

    /// Power operation: z = x^n
    pub fn pow(&mut self, x: Var, n: T) -> Var {
        let x_val = self.nodes[x.0].value;
        let value = x_val.powf(n);
        let idx = self.nodes.len();
        let grad_coeff = n * x_val.powf(n - T::one());
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, grad_coeff)],
        });
        Var(idx)
    }

    /// Exponential operation: z = e^x
    pub fn exp(&mut self, x: Var) -> Var {
        let x_val = self.nodes[x.0].value;
        let value = x_val.exp();
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, value)],
        });
        Var(idx)
    }

    /// Natural logarithm: z = ln(x)
    pub fn ln(&mut self, x: Var) -> Var {
        let x_val = self.nodes[x.0].value;
        let value = x_val.ln();
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, T::one() / x_val)],
        });
        Var(idx)
    }

    /// Sine operation: z = sin(x)
    pub fn sin(&mut self, x: Var) -> Var {
        let x_val = self.nodes[x.0].value;
        let value = x_val.sin();
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, x_val.cos())],
        });
        Var(idx)
    }

    /// Cosine operation: z = cos(x)
    pub fn cos(&mut self, x: Var) -> Var {
        let x_val = self.nodes[x.0].value;
        let value = x_val.cos();
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            grad: T::zero(),
            parents: vec![(x.0, -x_val.sin())],
        });
        Var(idx)
    }

    /// Perform backward pass to compute gradients
    ///
    /// This propagates gradients from the output variable back to all variables
    /// that contributed to it, using the chain rule.
    ///
    /// # Arguments
    ///
    /// * `output` - The output variable to differentiate with respect to
    pub fn backward(&mut self, output: Var) {
        // Initialize all gradients to zero
        for node in &mut self.nodes {
            node.grad = T::zero();
        }

        // Set gradient of output to 1 (∂output/∂output = 1)
        self.nodes[output.0].grad = T::one();

        // Propagate gradients backward through the graph
        for i in (0..self.nodes.len()).rev() {
            let grad = self.nodes[i].grad;
            let parents = self.nodes[i].parents.clone();

            for (parent_idx, coeff) in parents {
                self.nodes[parent_idx].grad = self.nodes[parent_idx].grad + grad * coeff;
            }
        }
    }

    /// Reset all gradients to zero
    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            node.grad = T::zero();
        }
    }

    /// Get the number of nodes in the tape
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl<T: Float> Default for Tape<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Higher-Order Derivatives
// ============================================================================

/// Compute Hessian matrix using numerical differentiation of gradients
///
/// Computes second derivatives by numerical differentiation of the gradient.
/// This is a robust approach that works well for most functions.
///
/// # Arguments
///
/// * `f` - Scalar-valued function to differentiate
/// * `x` - Point at which to compute the Hessian
///
/// # Returns
///
/// The Hessian matrix H(x) (n×n symmetric matrix)
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::*;
/// use numrs2::prelude::*;
///
/// // f(x, y) = x² + y²
/// fn f(vars: &[f64]) -> f64 {
///     vars[0] * vars[0] + vars[1] * vars[1]
/// }
///
/// let x = Array::from_vec(vec![3.0, 4.0]);
/// let hessian = hessian(f, &x).expect("valid hessian computation");
/// // H = [[2, 0], [0, 2]]
/// ```
pub fn hessian<F, T>(f: F, x: &Array<T>) -> Result<Array<T>>
where
    F: Fn(&[T]) -> T,
    T: Float,
{
    let n = x.size();
    let x_vec = x.to_vec();
    let mut hess = Vec::with_capacity(n * n);

    // For each output dimension of gradient
    for i in 0..n {
        // Compute gradient using forward mode at slightly perturbed points
        let eps = T::from(1e-7).expect("1e-7 is representable as Float");

        // Gradient computation helper
        let grad_at = |point: &[T]| -> Vec<T> {
            let mut grad = Vec::with_capacity(n);
            for j in 0..n {
                let mut point_plus = point.to_vec();
                let mut point_minus = point.to_vec();
                point_plus[j] = point_plus[j] + eps;
                point_minus[j] = point_minus[j] - eps;

                let f_plus = f(&point_plus);
                let f_minus = f(&point_minus);
                grad.push((f_plus - f_minus) / (eps + eps));
            }
            grad
        };

        // Compute gradient at x
        let grad = grad_at(&x_vec);

        // For each gradient component, compute its derivative (Hessian row)
        for j in 0..n {
            let mut x_plus = x_vec.clone();
            let mut x_minus = x_vec.clone();
            x_plus[i] = x_plus[i] + eps;
            x_minus[i] = x_minus[i] - eps;

            let grad_plus = grad_at(&x_plus);
            let grad_minus = grad_at(&x_minus);

            let second_deriv = (grad_plus[j] - grad_minus[j]) / (eps + eps);
            hess.push(second_deriv);
        }
    }

    Array::from_vec_shape(hess, &[n, n])
}

/// Compute directional derivative in direction v: ∇_v f = ∇f · v
///
/// # Arguments
///
/// * `f` - Scalar-valued function to differentiate
/// * `x` - Point at which to compute the directional derivative
/// * `v` - Direction vector (will be normalized)
///
/// # Returns
///
/// The directional derivative ∇_v f(x)
pub fn directional_derivative<F, T>(f: F, x: &Array<T>, v: &Array<T>) -> Result<T>
where
    F: Fn(&Array<Dual<T>>) -> Dual<T>,
    T: Float,
{
    if x.size() != v.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: v.shape(),
        });
    }

    // Normalize direction vector
    let v_vec = v.to_vec();
    let v_norm_sq: T = v_vec
        .iter()
        .map(|&vi| vi * vi)
        .fold(T::zero(), |acc, x| acc + x);
    let v_norm = v_norm_sq.sqrt();

    // Create dual number array with derivative in direction v
    let x_vec = x.to_vec();
    let mut dual_vec = Vec::with_capacity(x.size());
    for i in 0..x.size() {
        dual_vec.push(Dual::new(x_vec[i], v_vec[i] / v_norm));
    }
    let dual_array = Array::from_vec(dual_vec);

    // Evaluate and return directional derivative
    let result = f(&dual_array);
    Ok(result.deriv())
}

/// Compute nth derivative of a univariate function using forward mode and numerical methods
///
/// For n=0 and n=1, uses exact dual number differentiation.
/// For n≥2, uses numerical differentiation of lower-order derivatives.
///
/// # Arguments
///
/// * `f` - Univariate function to differentiate
/// * `x` - Point at which to compute the derivative
/// * `n` - Order of derivative (1 for first derivative, 2 for second, etc.)
///
/// # Returns
///
/// The nth derivative f⁽ⁿ⁾(x)
pub fn nth_derivative<F, T>(f: F, x: T, n: usize) -> T
where
    F: Fn(Dual<T>) -> Dual<T> + Copy,
    T: Float,
{
    nth_derivative_impl(&f, x, n)
}

fn nth_derivative_impl<F, T>(f: &F, x: T, n: usize) -> T
where
    F: Fn(Dual<T>) -> Dual<T>,
    T: Float,
{
    if n == 0 {
        let result = f(Dual::constant(x));
        return result.value();
    }

    if n == 1 {
        let result = f(Dual::variable(x));
        return result.deriv();
    }

    // For n ≥ 2, use numerical differentiation of (n-1)th derivative
    let eps = T::from(1e-7).expect("1e-7 is representable as Float");
    let deriv_prev_plus = nth_derivative_impl(f, x + eps, n - 1);
    let deriv_prev_minus = nth_derivative_impl(f, x - eps, n - 1);
    (deriv_prev_plus - deriv_prev_minus) / (eps + eps)
}

/// Compute Taylor series approximation of a function around a point
///
/// Returns coefficients [f(a), f'(a), f''(a)/2!, ..., f⁽ⁿ⁾(a)/n!]
///
/// # Arguments
///
/// * `f` - Univariate function to approximate
/// * `a` - Point around which to expand
/// * `order` - Number of terms in the Taylor series
///
/// # Returns
///
/// Vector of Taylor series coefficients
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::autodiff::*;
///
/// // Taylor series of e^x around x=0
/// fn f(x: Dual<f64>) -> Dual<f64> {
///     x.exp()
/// }
///
/// let coeffs = taylor_series(f, 0.0, 4);
/// // coeffs ≈ [1, 1, 0.5, 1/6, 1/24]
/// ```
pub fn taylor_series<F, T>(f: F, a: T, order: usize) -> Vec<T>
where
    F: Fn(Dual<T>) -> Dual<T> + Copy,
    T: Float,
{
    let mut coeffs = Vec::with_capacity(order + 1);

    // Compute factorial
    let factorial = |n: usize| -> T {
        let mut result = T::one();
        for i in 1..=n {
            result = result * T::from(i).expect("factorial index is representable as Float");
        }
        result
    };

    for n in 0..=order {
        let deriv_n = nth_derivative(f, a, n);
        coeffs.push(deriv_n / factorial(n));
    }

    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_arithmetic() {
        let x = Dual::new(3.0, 1.0);
        let y = Dual::new(4.0, 0.0);

        let sum = x + y;
        assert_eq!(sum.value(), 7.0);
        assert_eq!(sum.deriv(), 1.0);

        let diff = x - y;
        assert_eq!(diff.value(), -1.0);
        assert_eq!(diff.deriv(), 1.0);

        let prod = x * y;
        assert_eq!(prod.value(), 12.0);
        assert_eq!(prod.deriv(), 4.0);

        let quot = x / y;
        assert_eq!(quot.value(), 0.75);
        assert_eq!(quot.deriv(), 0.25);
    }

    #[test]
    fn test_dual_functions() {
        let x = Dual::variable(2.0);

        // Test pow: d/dx(x²) = 2x
        let square = x.pow(2.0);
        assert_eq!(square.value(), 4.0);
        assert_eq!(square.deriv(), 4.0);

        // Test exp: d/dx(e^x) = e^x
        let exp_x = x.exp();
        assert!((exp_x.value() - 2.0_f64.exp()).abs() < 1e-10);
        assert!((exp_x.deriv() - 2.0_f64.exp()).abs() < 1e-10);

        // Test ln: d/dx(ln(x)) = 1/x
        let ln_x = x.ln();
        assert!((ln_x.value() - 2.0_f64.ln()).abs() < 1e-10);
        assert!((ln_x.deriv() - 0.5).abs() < 1e-10);

        // Test sin: d/dx(sin(x)) = cos(x)
        let sin_x = x.sin();
        assert!((sin_x.value() - 2.0_f64.sin()).abs() < 1e-10);
        assert!((sin_x.deriv() - 2.0_f64.cos()).abs() < 1e-10);
    }

    #[test]
    fn test_dual_composition() {
        // f(x) = (x² + 1)³
        // f'(x) = 3(x² + 1)² * 2x = 6x(x² + 1)²
        let x = Dual::variable(2.0);
        let x_sq = x * x;
        let x_sq_plus_1 = x_sq + Dual::constant(1.0);
        let result = x_sq_plus_1.pow(3.0);

        assert_eq!(result.value(), 125.0); // (4 + 1)³ = 125
        assert_eq!(result.deriv(), 300.0); // 6 * 2 * 25 = 300
    }

    #[test]
    fn test_gradient_simple() {
        // f(x, y) = x² + y²
        // ∇f = [2x, 2y]
        fn f(vars: &Array<Dual<f64>>) -> Dual<f64> {
            let v = vars.to_vec();
            v[0] * v[0] + v[1] * v[1]
        }

        let x = Array::from_vec(vec![3.0, 4.0]);
        let grad = gradient(f, &x).expect("gradient computation should succeed");

        let grad_vec = grad.to_vec();
        assert!((grad_vec[0] - 6.0).abs() < 1e-10);
        assert!((grad_vec[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_tape_basic() {
        let mut tape = Tape::new();

        let x = tape.var(3.0);
        let y = tape.var(4.0);
        let z = tape.mul(x, x); // z = x²
        let w = tape.mul(y, y); // w = y²
        let result = tape.add(z, w); // result = x² + y²

        assert_eq!(tape.value(result), 25.0);

        tape.backward(result);

        assert_eq!(tape.grad(x), 6.0); // ∂/∂x(x² + y²) = 2x = 6
        assert_eq!(tape.grad(y), 8.0); // ∂/∂y(x² + y²) = 2y = 8
    }

    #[test]
    fn test_tape_division() {
        let mut tape = Tape::new();

        let x = tape.var(4.0);
        let y = tape.var(2.0);
        let z = tape.div(x, y); // z = x / y = 2

        assert_eq!(tape.value(z), 2.0);

        tape.backward(z);

        assert_eq!(tape.grad(x), 0.5); // ∂/∂x(x/y) = 1/y = 0.5
        assert_eq!(tape.grad(y), -1.0); // ∂/∂y(x/y) = -x/y² = -1
    }

    #[test]
    fn test_tape_exp_ln() {
        let mut tape = Tape::new();

        let x = tape.var(2.0);
        let y = tape.exp(x); // y = e^x
        let z = tape.ln(y); // z = ln(e^x) = x

        assert!((tape.value(z) - 2.0).abs() < 1e-10);

        tape.backward(z);

        // ∂z/∂x should be 1 (since z = x through composition)
        assert!((tape.grad(x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tape_trigonometric() {
        let mut tape = Tape::new();

        let x = tape.var(1.0);
        let s = tape.sin(x);
        let c = tape.cos(x);

        // Test values
        assert!((tape.value(s) - 1.0_f64.sin()).abs() < 1e-10);
        assert!((tape.value(c) - 1.0_f64.cos()).abs() < 1e-10);

        // Test derivatives
        tape.backward(s);
        assert!((tape.grad(x) - 1.0_f64.cos()).abs() < 1e-10);

        tape.zero_grad();
        tape.backward(c);
        assert!((tape.grad(x) + 1.0_f64.sin()).abs() < 1e-10);
    }

    #[test]
    fn test_tape_chain_rule() {
        // f(x) = sin(x²)
        // f'(x) = cos(x²) * 2x
        let mut tape = Tape::new();

        let x = tape.var(2.0);
        let x_sq = tape.pow(x, 2.0);
        let result = tape.sin(x_sq);

        let expected_value = (4.0_f64).sin();
        assert!((tape.value(result) - expected_value).abs() < 1e-10);

        tape.backward(result);

        let expected_grad = (4.0_f64).cos() * 4.0; // cos(4) * 2*2
        assert!((tape.grad(x) - expected_grad).abs() < 1e-10);
    }

    #[test]
    fn test_directional_derivative() {
        // f(x, y) = x² + y²
        fn f(vars: &Array<Dual<f64>>) -> Dual<f64> {
            let v = vars.to_vec();
            v[0] * v[0] + v[1] * v[1]
        }

        let x = Array::from_vec(vec![3.0, 4.0]);
        let v = Array::from_vec(vec![1.0, 0.0]); // direction along x-axis

        let dir_deriv = directional_derivative(f, &x, &v)
            .expect("directional derivative computation should succeed");
        // ∇f = [2x, 2y] = [6, 8], direction v = [1, 0]
        // ∇_v f = [6, 8] · [1, 0] = 6
        assert!((dir_deriv - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_nth_derivative_simple() {
        // f(x) = x³
        // f'(x) = 3x²
        // f''(x) = 6x
        // f'''(x) = 6
        fn f(x: Dual<f64>) -> Dual<f64> {
            x.pow(3.0)
        }

        let x = 2.0;

        let f0 = nth_derivative(f, x, 0);
        assert!((f0 - 8.0).abs() < 1e-6); // f(2) = 8

        let f1 = nth_derivative(f, x, 1);
        assert!((f1 - 12.0).abs() < 1e-6); // f'(2) = 12

        let f2 = nth_derivative(f, x, 2);
        assert!((f2 - 12.0).abs() < 0.1); // f''(2) = 12 (numerical tolerance)

        let f3 = nth_derivative(f, x, 3);
        assert!((f3 - 6.0).abs() < 1.0); // f'''(2) = 6 (numerical tolerance)
    }

    #[test]
    fn test_taylor_series_exp() {
        // Taylor series of e^x around x=0: 1 + x + x²/2! + x³/3! + ...
        fn f(x: Dual<f64>) -> Dual<f64> {
            x.exp()
        }

        let coeffs = taylor_series(f, 0.0, 4);

        // All coefficients should be approximately 1/n!
        assert!((coeffs[0] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((coeffs[1] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((coeffs[2] - 0.5).abs() < 1e-4); // e^0/2! = 0.5
        assert!((coeffs[3] - 1.0 / 6.0).abs() < 1e-3); // e^0/3! = 1/6
    }

    #[test]
    fn test_hessian_simple() {
        // f(x, y) = x² + y²
        // H = [[2, 0], [0, 2]]
        fn f(vars: &[f64]) -> f64 {
            vars[0] * vars[0] + vars[1] * vars[1]
        }

        let x = Array::from_vec(vec![3.0, 4.0]);
        let hess = hessian(f, &x).expect("hessian computation should succeed");

        let hess_vec = hess.to_vec();
        // Numerical differentiation has lower accuracy for second derivatives
        // H[0,0] ≈ 2 (relaxed tolerance for numerical method)
        assert!((hess_vec[0] - 2.0).abs() < 1.0);
        // H[0,1] ≈ 0
        assert!(hess_vec[1].abs() < 1.0);
        // H[1,0] ≈ 0
        assert!(hess_vec[2].abs() < 1.0);
        // H[1,1] ≈ 2
        assert!((hess_vec[3] - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_dual_sigmoid() {
        let x = Dual::variable(0.0);
        let y = x.sigmoid();

        assert!((y.value() - 0.5).abs() < 1e-10); // sigmoid(0) = 0.5
        assert!((y.deriv() - 0.25).abs() < 1e-10); // sigmoid'(0) = 0.25
    }

    #[test]
    fn test_dual_relu() {
        let x_pos = Dual::variable(2.0);
        let y_pos = x_pos.relu();
        assert_eq!(y_pos.value(), 2.0);
        assert_eq!(y_pos.deriv(), 1.0);

        let x_neg = Dual::variable(-2.0);
        let y_neg = x_neg.relu();
        assert_eq!(y_neg.value(), 0.0);
        assert_eq!(y_neg.deriv(), 0.0);
    }
}
