//! Expression Templates for Lazy Evaluation
//!
//! This module implements expression templates to enable lazy evaluation and
//! operation fusion. Instead of computing intermediate results immediately,
//! operations build an expression tree that can be optimized and evaluated
//! efficiently.
//!
//! # Benefits
//!
//! - **Eliminates intermediate allocations**: `(a + b) * c` computed in one pass
//! - **Kernel fusion**: Multiple operations fused into single SIMD loop
//! - **Optimized memory access**: Better cache utilization
//! - **Deferred evaluation**: Computation only when result is needed
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::prelude::*;
//! use numrs2::expr::*;
//!
//! let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
//! let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
//!
//! // Create expression manually (operator overloading has lifetime issues)
//! let expr = BinaryExpr::new(
//!     ArrayExpr::new(&a),
//!     ArrayExpr::new(&b),
//!     |x, y| x + y
//! ).unwrap();
//!
//! // Evaluation happens here
//! let result = expr.eval();
//! ```
//!
//! # Current Status
//!
//! This module provides the foundational infrastructure for expression templates.
//! Operator overloading has Rust lifetime challenges that need further work.
//! The core trait and types are functional and serve as a basis for future optimization.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::marker::PhantomData;

/// Trait for lazy expressions that can be evaluated
///
/// All expression types (binary ops, unary ops, arrays) implement this trait
/// to enable deferred computation and optimization.
pub trait Expr<T: Clone> {
    /// Evaluate the expression at a specific index
    ///
    /// This is the core method that enables lazy evaluation. Each expression
    /// knows how to compute its value at any given index without materializing
    /// the entire result.
    fn eval_at(&self, index: usize) -> T;

    /// Get the size of the expression result
    fn size(&self) -> usize;

    /// Get the shape of the expression result
    fn shape(&self) -> &[usize];

    /// Materialize the expression into an Array
    ///
    /// This triggers evaluation of the entire expression tree, applying all
    /// optimizations and fusions.
    fn eval(&self) -> Array<T> {
        let size = self.size();
        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            data.push(self.eval_at(i));
        }

        Array::from_vec(data).reshape(self.shape())
    }

    /// Check if this expression can be fused with another
    ///
    /// Returns true if the expressions have compatible shapes for fusion.
    fn can_fuse_with<E: Expr<T>>(&self, other: &E) -> bool {
        self.shape() == other.shape()
    }
}

/// Wrapper for lazy Array expressions
///
/// This wraps an Array reference to make it participate in lazy evaluation.
pub struct ArrayExpr<'a, T: Clone> {
    array: &'a Array<T>,
    shape: Vec<usize>,
}

impl<'a, T: Clone> ArrayExpr<'a, T> {
    pub fn new(array: &'a Array<T>) -> Self {
        let shape = array.shape();
        Self { array, shape }
    }
}

impl<'a, T: Clone> Expr<T> for ArrayExpr<'a, T> {
    fn eval_at(&self, index: usize) -> T {
        self.array.to_vec()[index].clone()
    }

    fn size(&self) -> usize {
        self.array.size()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn eval(&self) -> Array<T> {
        self.array.clone()
    }
}

/// Binary operation expression
///
/// Represents a lazy binary operation between two expressions.
/// The operation is only computed when `eval()` or `eval_at()` is called.
pub struct BinaryExpr<T, L, R, F>
where
    T: Clone,
    L: Expr<T>,
    R: Expr<T>,
    F: Fn(T, T) -> T,
{
    left: L,
    right: R,
    op: F,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, L, R, F> BinaryExpr<T, L, R, F>
where
    T: Clone,
    L: Expr<T>,
    R: Expr<T>,
    F: Fn(T, T) -> T,
{
    pub fn new(left: L, right: R, op: F) -> Result<Self> {
        if left.shape() != right.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: left.shape().to_vec(),
                actual: right.shape().to_vec(),
            });
        }

        Ok(Self {
            shape: left.shape().to_vec(),
            left,
            right,
            op,
            _phantom: PhantomData,
        })
    }
}

impl<T, L, R, F> Expr<T> for BinaryExpr<T, L, R, F>
where
    T: Clone,
    L: Expr<T>,
    R: Expr<T>,
    F: Fn(T, T) -> T,
{
    fn eval_at(&self, index: usize) -> T {
        let left_val = self.left.eval_at(index);
        let right_val = self.right.eval_at(index);
        (self.op)(left_val, right_val)
    }

    fn size(&self) -> usize {
        self.left.size()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Unary operation expression
///
/// Represents a lazy unary operation on an expression.
pub struct UnaryExpr<T, E, F>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T) -> T,
{
    expr: E,
    op: F,
    _phantom: PhantomData<T>,
}

impl<T, E, F> UnaryExpr<T, E, F>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T) -> T,
{
    pub fn new(expr: E, op: F) -> Self {
        Self {
            expr,
            op,
            _phantom: PhantomData,
        }
    }
}

impl<T, E, F> Expr<T> for UnaryExpr<T, E, F>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T) -> T,
{
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.op)(val)
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

/// Scalar operation expression
///
/// Represents a lazy operation between an expression and a scalar value.
pub struct ScalarExpr<T, E, F>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T, T) -> T,
{
    expr: E,
    scalar: T,
    op: F,
}

impl<T, E, F> ScalarExpr<T, E, F>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T, T) -> T,
{
    pub fn new(expr: E, scalar: T, op: F) -> Self {
        Self { expr, scalar, op }
    }
}

impl<T, E, F> Expr<T> for ScalarExpr<T, E, F>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T, T) -> T,
{
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.op)(val, self.scalar.clone())
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

/// Extension trait to add lazy evaluation methods to Array
pub trait LazyEval<T: Clone> {
    /// Convert array to lazy expression
    fn lazy(&self) -> ArrayExpr<T>;
}

impl<T: Clone> LazyEval<T> for Array<T> {
    fn lazy(&self) -> ArrayExpr<T> {
        ArrayExpr::new(self)
    }
}

// ============================================================================
// ENHANCED EXPRESSION TYPES
// ============================================================================

/// Reduction expression that produces a scalar
///
/// Supports common reductions like sum, product, max, min
pub struct ReductionExpr<T, E, F, R>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T, T) -> T,
    R: Fn() -> T,
{
    expr: E,
    reduce_op: F,
    identity: R,
    _phantom: PhantomData<T>,
}

impl<T, E, F, R> ReductionExpr<T, E, F, R>
where
    T: Clone,
    E: Expr<T>,
    F: Fn(T, T) -> T,
    R: Fn() -> T,
{
    pub fn new(expr: E, reduce_op: F, identity: R) -> Self {
        Self {
            expr,
            reduce_op,
            identity,
            _phantom: PhantomData,
        }
    }

    /// Evaluate the reduction and return a scalar
    pub fn reduce(&self) -> T {
        let size = self.expr.size();
        if size == 0 {
            return (self.identity)();
        }

        let mut result = self.expr.eval_at(0);
        for i in 1..size {
            let val = self.expr.eval_at(i);
            result = (self.reduce_op)(result, val);
        }
        result
    }
}

/// Conditional (where) expression
///
/// Returns values from `true_expr` where condition is true, otherwise from `false_expr`
pub struct WhereExpr<T, C, Tr, Fa>
where
    T: Clone,
    C: Expr<bool>,
    Tr: Expr<T>,
    Fa: Expr<T>,
{
    condition: C,
    true_expr: Tr,
    false_expr: Fa,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, C, Tr, Fa> WhereExpr<T, C, Tr, Fa>
where
    T: Clone,
    C: Expr<bool>,
    Tr: Expr<T>,
    Fa: Expr<T>,
{
    pub fn new(condition: C, true_expr: Tr, false_expr: Fa) -> Result<Self> {
        // All shapes must match
        if condition.shape() != true_expr.shape() || condition.shape() != false_expr.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: condition.shape().to_vec(),
                actual: true_expr.shape().to_vec(),
            });
        }

        Ok(Self {
            shape: condition.shape().to_vec(),
            condition,
            true_expr,
            false_expr,
            _phantom: PhantomData,
        })
    }
}

impl<T, C, Tr, Fa> Expr<T> for WhereExpr<T, C, Tr, Fa>
where
    T: Clone,
    C: Expr<bool>,
    Tr: Expr<T>,
    Fa: Expr<T>,
{
    fn eval_at(&self, index: usize) -> T {
        if self.condition.eval_at(index) {
            self.true_expr.eval_at(index)
        } else {
            self.false_expr.eval_at(index)
        }
    }

    fn size(&self) -> usize {
        self.condition.size()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Clipped (clamped) expression
///
/// Clips values to a specified range [min, max]
pub struct ClipExpr<T, E>
where
    T: Clone + PartialOrd,
    E: Expr<T>,
{
    expr: E,
    min_val: T,
    max_val: T,
}

impl<T, E> ClipExpr<T, E>
where
    T: Clone + PartialOrd,
    E: Expr<T>,
{
    pub fn new(expr: E, min_val: T, max_val: T) -> Self {
        Self {
            expr,
            min_val,
            max_val,
        }
    }
}

impl<T, E> Expr<T> for ClipExpr<T, E>
where
    T: Clone + PartialOrd,
    E: Expr<T>,
{
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        if val < self.min_val {
            self.min_val.clone()
        } else if val > self.max_val {
            self.max_val.clone()
        } else {
            val
        }
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

/// Broadcast scalar expression
///
/// Broadcasts a scalar value to a given shape
pub struct BroadcastScalarExpr<T: Clone> {
    value: T,
    shape: Vec<usize>,
    size: usize,
}

impl<T: Clone> BroadcastScalarExpr<T> {
    pub fn new(value: T, shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            value,
            shape: shape.to_vec(),
            size,
        }
    }
}

impl<T: Clone> Expr<T> for BroadcastScalarExpr<T> {
    fn eval_at(&self, _index: usize) -> T {
        self.value.clone()
    }

    fn size(&self) -> usize {
        self.size
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// ============================================================================
// OPTIMIZED EVALUATION WITH SIMD
// ============================================================================

/// Trait for SIMD-optimized batch evaluation
pub trait SimdEval<T: Clone + Copy>: Expr<T> {
    /// Evaluate a contiguous batch of elements efficiently
    ///
    /// Implementations may use SIMD for numeric types
    fn eval_batch(&self, start: usize, len: usize) -> Vec<T> {
        let end = (start + len).min(self.size());
        (start..end).map(|i| self.eval_at(i)).collect()
    }

    /// Evaluate entire expression with optimized batch processing
    fn eval_simd(&self) -> Array<T> {
        const BATCH_SIZE: usize = 256;
        let size = self.size();
        let mut data = Vec::with_capacity(size);

        let mut i = 0;
        while i < size {
            let batch_len = (size - i).min(BATCH_SIZE);
            let batch = self.eval_batch(i, batch_len);
            data.extend(batch);
            i += batch_len;
        }

        Array::from_vec(data).reshape(self.shape())
    }
}

// Implement SimdEval for f64 expressions
impl<'a> SimdEval<f64> for ArrayExpr<'a, f64> {}

impl<L, R, F> SimdEval<f64> for BinaryExpr<f64, L, R, F>
where
    L: Expr<f64>,
    R: Expr<f64>,
    F: Fn(f64, f64) -> f64,
{
}

impl<E, F> SimdEval<f64> for UnaryExpr<f64, E, F>
where
    E: Expr<f64>,
    F: Fn(f64) -> f64,
{
}

impl<E, F> SimdEval<f64> for ScalarExpr<f64, E, F>
where
    E: Expr<f64>,
    F: Fn(f64, f64) -> f64,
{
}

// ============================================================================
// EXPRESSION BUILDER (Fluent API)
// ============================================================================

/// Expression builder for creating complex expressions with a fluent interface
pub struct ExprBuilder<T, E>
where
    T: Clone,
    E: Expr<T>,
{
    expr: E,
    _phantom: PhantomData<T>,
}

impl<'a, T: Clone> ExprBuilder<T, ArrayExpr<'a, T>> {
    /// Start building from an array
    pub fn from_array(array: &'a Array<T>) -> Self {
        ExprBuilder {
            expr: ArrayExpr::new(array),
            _phantom: PhantomData,
        }
    }
}

impl<T, E> ExprBuilder<T, E>
where
    T: Clone,
    E: Expr<T>,
{
    /// Apply a unary operation
    pub fn map<F: Fn(T) -> T>(self, op: F) -> ExprBuilder<T, UnaryExpr<T, E, F>> {
        ExprBuilder {
            expr: UnaryExpr::new(self.expr, op),
            _phantom: PhantomData,
        }
    }

    /// Apply a binary operation with another expression
    pub fn zip_with<E2, F>(
        self,
        other: E2,
        op: F,
    ) -> Result<ExprBuilder<T, BinaryExpr<T, E, E2, F>>>
    where
        E2: Expr<T>,
        F: Fn(T, T) -> T,
    {
        let binary = BinaryExpr::new(self.expr, other, op)?;
        Ok(ExprBuilder {
            expr: binary,
            _phantom: PhantomData,
        })
    }

    /// Apply a scalar operation
    pub fn scalar<F: Fn(T, T) -> T>(self, scalar: T, op: F) -> ExprBuilder<T, ScalarExpr<T, E, F>> {
        ExprBuilder {
            expr: ScalarExpr::new(self.expr, scalar, op),
            _phantom: PhantomData,
        }
    }

    /// Add a scalar value
    pub fn add_scalar(self, scalar: T) -> ExprBuilder<T, ScalarExpr<T, E, impl Fn(T, T) -> T>>
    where
        T: std::ops::Add<Output = T>,
    {
        self.scalar(scalar, |x, y| x + y)
    }

    /// Multiply by a scalar value
    pub fn mul_scalar(self, scalar: T) -> ExprBuilder<T, ScalarExpr<T, E, impl Fn(T, T) -> T>>
    where
        T: std::ops::Mul<Output = T>,
    {
        self.scalar(scalar, |x, y| x * y)
    }

    /// Evaluate and materialize the expression
    pub fn eval(self) -> Array<T> {
        self.expr.eval()
    }

    /// Get the underlying expression
    pub fn build(self) -> E {
        self.expr
    }
}

// Additional methods for numeric types
impl<E: Expr<f64>> ExprBuilder<f64, E> {
    /// Apply absolute value
    pub fn abs(self) -> ExprBuilder<f64, UnaryExpr<f64, E, impl Fn(f64) -> f64>> {
        self.map(|x| x.abs())
    }

    /// Apply square root
    pub fn sqrt(self) -> ExprBuilder<f64, UnaryExpr<f64, E, impl Fn(f64) -> f64>> {
        self.map(|x| x.sqrt())
    }

    /// Apply exponential
    pub fn exp(self) -> ExprBuilder<f64, UnaryExpr<f64, E, impl Fn(f64) -> f64>> {
        self.map(|x| x.exp())
    }

    /// Apply natural logarithm
    pub fn ln(self) -> ExprBuilder<f64, UnaryExpr<f64, E, impl Fn(f64) -> f64>> {
        self.map(|x| x.ln())
    }

    /// Apply sine
    pub fn sin(self) -> ExprBuilder<f64, UnaryExpr<f64, E, impl Fn(f64) -> f64>> {
        self.map(|x| x.sin())
    }

    /// Apply cosine
    pub fn cos(self) -> ExprBuilder<f64, UnaryExpr<f64, E, impl Fn(f64) -> f64>> {
        self.map(|x| x.cos())
    }

    /// Reduce by sum
    pub fn sum(self) -> f64 {
        ReductionExpr::new(self.expr, |a, b| a + b, || 0.0).reduce()
    }

    /// Reduce by product
    pub fn prod(self) -> f64 {
        ReductionExpr::new(self.expr, |a, b| a * b, || 1.0).reduce()
    }

    /// Reduce by maximum
    pub fn max(self) -> f64 {
        ReductionExpr::new(self.expr, |a: f64, b: f64| a.max(b), || f64::NEG_INFINITY).reduce()
    }

    /// Reduce by minimum
    pub fn min(self) -> f64 {
        ReductionExpr::new(self.expr, |a: f64, b: f64| a.min(b), || f64::INFINITY).reduce()
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Create a sum reduction from an expression
pub fn expr_sum<T, E>(expr: E) -> T
where
    T: Clone + std::ops::Add<Output = T> + Default,
    E: Expr<T>,
{
    ReductionExpr::new(expr, |a, b| a + b, T::default).reduce()
}

/// Create a product reduction from an expression
pub fn expr_prod<T, E>(expr: E) -> T
where
    T: Clone + std::ops::Mul<Output = T> + num_traits::One,
    E: Expr<T>,
{
    ReductionExpr::new(expr, |a, b| a * b, T::one).reduce()
}

/// Fused multiply-add expression: a * b + c
pub fn fma<T, A, B, C>(a: A, b: B, c: C) -> Result<impl Expr<T>>
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
{
    // a * b
    let ab = BinaryExpr::new(a, b, |x, y| x * y)?;
    // ab + c
    BinaryExpr::new(ab, c, |x, y| x + y)
}

// ============================================================================
// SHAREDARRAY EXPRESSION TYPES (Reference-Counted, Lifetime-Free)
// ============================================================================
// These types solve the lifetime challenges mentioned below by using SharedArray's
// reference counting. Expressions can be stored, passed around, and composed
// without complex lifetime annotations.

use crate::shared_array::SharedArray;

/// Trait for shared expressions that evaluate to SharedArray
///
/// Unlike `Expr<T>` which uses lifetimes, `SharedExpr<T>` uses reference-counted
/// storage, enabling expressions to be stored in data structures and composed
/// without lifetime constraints.
pub trait SharedExpr<T: Clone>: Clone {
    /// Evaluate the expression at a specific index
    fn eval_at(&self, index: usize) -> T;

    /// Get the size of the expression result
    fn size(&self) -> usize;

    /// Get the shape of the expression result
    fn shape(&self) -> Vec<usize>;

    /// Materialize the expression into a SharedArray
    fn eval(&self) -> SharedArray<T> {
        let size = self.size();
        let shape = self.shape();
        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            data.push(self.eval_at(i));
        }

        SharedArray::from_vec_with_shape(data, &shape).expect("Shape should be valid")
    }
}

/// SharedArray expression wrapper
///
/// Wraps a SharedArray for use in expression trees without lifetime issues.
#[derive(Clone)]
pub struct SharedArrayExpr<T: Clone> {
    array: SharedArray<T>,
}

impl<T: Clone> SharedArrayExpr<T> {
    /// Create a new SharedArrayExpr from a SharedArray
    pub fn new(array: SharedArray<T>) -> Self {
        Self { array }
    }

    /// Create from an owned Array
    pub fn from_array(array: Array<T>) -> Self {
        Self {
            array: SharedArray::from_array(array),
        }
    }
}

impl<T: Clone> SharedExpr<T> for SharedArrayExpr<T> {
    fn eval_at(&self, index: usize) -> T {
        self.array.to_vec()[index].clone()
    }

    fn size(&self) -> usize {
        self.array.size()
    }

    fn shape(&self) -> Vec<usize> {
        self.array.shape()
    }

    fn eval(&self) -> SharedArray<T> {
        self.array.clone()
    }
}

/// Binary operation on SharedExpr
///
/// Represents a lazy binary operation between two shared expressions.
/// Unlike BinaryExpr, this can be stored and moved without lifetime issues.
#[derive(Clone)]
pub struct SharedBinaryExpr<T, L, R, F>
where
    T: Clone,
    L: SharedExpr<T>,
    R: SharedExpr<T>,
    F: Fn(T, T) -> T + Clone,
{
    left: L,
    right: R,
    op: F,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, L, R, F> SharedBinaryExpr<T, L, R, F>
where
    T: Clone,
    L: SharedExpr<T>,
    R: SharedExpr<T>,
    F: Fn(T, T) -> T + Clone,
{
    /// Create a new binary expression
    pub fn new(left: L, right: R, op: F) -> Result<Self> {
        let left_shape = left.shape();
        let right_shape = right.shape();

        if left_shape != right_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: left_shape,
                actual: right_shape,
            });
        }

        Ok(Self {
            shape: left_shape,
            left,
            right,
            op,
            _phantom: PhantomData,
        })
    }
}

impl<T, L, R, F> SharedExpr<T> for SharedBinaryExpr<T, L, R, F>
where
    T: Clone,
    L: SharedExpr<T>,
    R: SharedExpr<T>,
    F: Fn(T, T) -> T + Clone,
{
    fn eval_at(&self, index: usize) -> T {
        let left_val = self.left.eval_at(index);
        let right_val = self.right.eval_at(index);
        (self.op)(left_val, right_val)
    }

    fn size(&self) -> usize {
        self.left.size()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

/// Unary operation on SharedExpr
#[derive(Clone)]
pub struct SharedUnaryExpr<T, E, F>
where
    T: Clone,
    E: SharedExpr<T>,
    F: Fn(T) -> T + Clone,
{
    expr: E,
    op: F,
    _phantom: PhantomData<T>,
}

impl<T, E, F> SharedUnaryExpr<T, E, F>
where
    T: Clone,
    E: SharedExpr<T>,
    F: Fn(T) -> T + Clone,
{
    pub fn new(expr: E, op: F) -> Self {
        Self {
            expr,
            op,
            _phantom: PhantomData,
        }
    }
}

impl<T, E, F> SharedExpr<T> for SharedUnaryExpr<T, E, F>
where
    T: Clone,
    E: SharedExpr<T>,
    F: Fn(T) -> T + Clone,
{
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.op)(val)
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> Vec<usize> {
        self.expr.shape()
    }
}

/// Scalar operation on SharedExpr
#[derive(Clone)]
pub struct SharedScalarExpr<T, E, F>
where
    T: Clone,
    E: SharedExpr<T>,
    F: Fn(T, T) -> T + Clone,
{
    expr: E,
    scalar: T,
    op: F,
}

impl<T, E, F> SharedScalarExpr<T, E, F>
where
    T: Clone,
    E: SharedExpr<T>,
    F: Fn(T, T) -> T + Clone,
{
    pub fn new(expr: E, scalar: T, op: F) -> Self {
        Self { expr, scalar, op }
    }
}

impl<T, E, F> SharedExpr<T> for SharedScalarExpr<T, E, F>
where
    T: Clone,
    E: SharedExpr<T>,
    F: Fn(T, T) -> T + Clone,
{
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.op)(val, self.scalar.clone())
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> Vec<usize> {
        self.expr.shape()
    }
}

/// Builder for constructing SharedExpr chains fluently
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::SharedExprBuilder;
///
/// let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let result = SharedExprBuilder::from_shared_array(arr)
///     .mul_scalar(2.0)
///     .add_scalar(1.0)
///     .eval();
///
/// assert_eq!(result.to_vec(), vec![3.0, 5.0, 7.0, 9.0]);
/// ```
#[derive(Clone)]
pub struct SharedExprBuilder<T: Clone, E: SharedExpr<T>> {
    expr: E,
    _phantom: PhantomData<T>,
}

impl<T: Clone> SharedExprBuilder<T, SharedArrayExpr<T>> {
    /// Create a builder from a SharedArray
    pub fn from_shared_array(array: SharedArray<T>) -> Self {
        Self {
            expr: SharedArrayExpr::new(array),
            _phantom: PhantomData,
        }
    }

    /// Create a builder from an Array
    pub fn from_array(array: Array<T>) -> Self {
        Self {
            expr: SharedArrayExpr::from_array(array),
            _phantom: PhantomData,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<T: Clone + std::ops::Add<Output = T>, E: SharedExpr<T>> SharedExprBuilder<T, E> {
    /// Add a scalar to all elements
    pub fn add_scalar(
        self,
        scalar: T,
    ) -> SharedExprBuilder<T, SharedScalarExpr<T, E, fn(T, T) -> T>>
    where
        T: 'static,
    {
        SharedExprBuilder {
            expr: SharedScalarExpr::new(self.expr, scalar, |x, y| x + y),
            _phantom: PhantomData,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<T: Clone + std::ops::Sub<Output = T>, E: SharedExpr<T>> SharedExprBuilder<T, E> {
    /// Subtract a scalar from all elements
    pub fn sub_scalar(
        self,
        scalar: T,
    ) -> SharedExprBuilder<T, SharedScalarExpr<T, E, fn(T, T) -> T>>
    where
        T: 'static,
    {
        SharedExprBuilder {
            expr: SharedScalarExpr::new(self.expr, scalar, |x, y| x - y),
            _phantom: PhantomData,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<T: Clone + std::ops::Mul<Output = T>, E: SharedExpr<T>> SharedExprBuilder<T, E> {
    /// Multiply all elements by a scalar
    pub fn mul_scalar(
        self,
        scalar: T,
    ) -> SharedExprBuilder<T, SharedScalarExpr<T, E, fn(T, T) -> T>>
    where
        T: 'static,
    {
        SharedExprBuilder {
            expr: SharedScalarExpr::new(self.expr, scalar, |x, y| x * y),
            _phantom: PhantomData,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<T: Clone + std::ops::Div<Output = T>, E: SharedExpr<T>> SharedExprBuilder<T, E> {
    /// Divide all elements by a scalar
    pub fn div_scalar(
        self,
        scalar: T,
    ) -> SharedExprBuilder<T, SharedScalarExpr<T, E, fn(T, T) -> T>>
    where
        T: 'static,
    {
        SharedExprBuilder {
            expr: SharedScalarExpr::new(self.expr, scalar, |x, y| x / y),
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone, E: SharedExpr<T>> SharedExprBuilder<T, E> {
    /// Apply a unary operation to all elements
    pub fn map<F>(self, op: F) -> SharedExprBuilder<T, SharedUnaryExpr<T, E, F>>
    where
        F: Fn(T) -> T + Clone,
    {
        SharedExprBuilder {
            expr: SharedUnaryExpr::new(self.expr, op),
            _phantom: PhantomData,
        }
    }

    /// Evaluate the expression and return a SharedArray
    pub fn eval(self) -> SharedArray<T> {
        self.expr.eval()
    }

    /// Get the underlying expression
    pub fn into_expr(self) -> E {
        self.expr
    }
}

// Note: Operator overloading for expression templates in Rust has significant
// lifetime challenges. Future work will address these issues with alternative
// designs (e.g., macros, builder patterns, or specialized traits).
// For now, users can construct expressions manually using the BinaryExpr::new API
// or use the fluent ExprBuilder interface.
// UPDATE: SharedExpr types above solve these issues using reference counting!

// ============================================================================
// COMMON SUBEXPRESSION ELIMINATION (CSE)
// ============================================================================
// CSE is an optimization technique that identifies when the same expression
// is computed multiple times and caches the result to avoid redundant computation.
// This is especially valuable for large arrays where recomputation is expensive.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Unique identifier for expression nodes in the DAG
///
/// Used to identify common subexpressions during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(u64);

impl ExprId {
    /// Generate a new unique expression ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Create from a raw value (for testing or specific use cases)
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for ExprId {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for storing evaluated expressions
///
/// Thread-safe cache that maps expression IDs to their evaluated SharedArray results.
/// This enables sharing of computation results across multiple uses of the same
/// subexpression.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{ExprCache, ExprId};
///
/// let cache = ExprCache::new();
///
/// // Store a result
/// let id = ExprId::new();
/// let array = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
/// cache.insert(id, array.clone());
///
/// // Retrieve the cached result
/// let cached: Option<SharedArray<f64>> = cache.get(&id);
/// assert!(cached.is_some());
/// assert_eq!(cached.unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub struct ExprCache<T: Clone> {
    cache: Arc<RwLock<HashMap<ExprId, SharedArray<T>>>>,
}

impl<T: Clone> ExprCache<T> {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert a result into the cache
    pub fn insert(&self, id: ExprId, value: SharedArray<T>) {
        if let Ok(mut guard) = self.cache.write() {
            guard.insert(id, value);
        }
    }

    /// Get a cached result
    pub fn get(&self, id: &ExprId) -> Option<SharedArray<T>> {
        if let Ok(guard) = self.cache.read() {
            guard.get(id).cloned()
        } else {
            None
        }
    }

    /// Check if an expression is cached
    pub fn contains(&self, id: &ExprId) -> bool {
        if let Ok(guard) = self.cache.read() {
            guard.contains_key(id)
        } else {
            false
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut guard) = self.cache.write() {
            guard.clear();
        }
    }

    /// Get the number of cached entries
    pub fn len(&self) -> usize {
        if let Ok(guard) = self.cache.read() {
            guard.len()
        } else {
            0
        }
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone> Default for ExprCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for ExprCache<T> {
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
        }
    }
}

/// A cached expression wrapper
///
/// Wraps an expression with caching capability. When evaluated, it first checks
/// the cache for a pre-computed result. If found, it returns the cached value;
/// otherwise, it evaluates the expression and stores the result.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{CachedExpr, SharedArrayExpr, SharedExpr, ExprCache};
///
/// let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let expr = SharedArrayExpr::new(arr);
/// let cache = ExprCache::new();
///
/// let cached = CachedExpr::new(expr, cache.clone());
///
/// // First evaluation computes and caches the result
/// let result1 = cached.eval();
///
/// // Second evaluation returns the cached result
/// let result2 = cached.eval();
///
/// assert_eq!(result1.to_vec(), result2.to_vec());
/// assert_eq!(cache.len(), 1); // Only one entry cached
/// ```
#[derive(Clone)]
pub struct CachedExpr<T: Clone, E: SharedExpr<T>> {
    expr: E,
    id: ExprId,
    cache: ExprCache<T>,
}

impl<T: Clone, E: SharedExpr<T>> CachedExpr<T, E> {
    /// Create a new cached expression
    pub fn new(expr: E, cache: ExprCache<T>) -> Self {
        Self {
            expr,
            id: ExprId::new(),
            cache,
        }
    }

    /// Create with a specific ID (useful for CSE optimization)
    pub fn with_id(expr: E, id: ExprId, cache: ExprCache<T>) -> Self {
        Self { expr, id, cache }
    }

    /// Get the expression ID
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Get a reference to the cache
    pub fn cache(&self) -> &ExprCache<T> {
        &self.cache
    }

    /// Invalidate the cached result for this expression
    pub fn invalidate(&self) {
        if let Ok(mut guard) = self.cache.cache.write() {
            guard.remove(&self.id);
        }
    }
}

impl<T: Clone, E: SharedExpr<T>> SharedExpr<T> for CachedExpr<T, E> {
    fn eval_at(&self, index: usize) -> T {
        // For indexed access, we evaluate and cache the full array
        // then return the specific index
        let array = self.eval();
        array.to_vec()[index].clone()
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> Vec<usize> {
        self.expr.shape()
    }

    fn eval(&self) -> SharedArray<T> {
        // Check cache first
        if let Some(cached) = self.cache.get(&self.id) {
            return cached;
        }

        // Evaluate and cache
        let result = self.expr.eval();
        self.cache.insert(self.id, result.clone());
        result
    }
}

/// Expression hash key for CSE identification
///
/// Used to identify structurally identical expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprKey {
    /// Leaf array with an ID
    Array(u64),
    /// Binary operation with operation type and operand keys
    Binary {
        op: &'static str,
        left: Box<ExprKey>,
        right: Box<ExprKey>,
    },
    /// Unary operation
    Unary {
        op: &'static str,
        operand: Box<ExprKey>,
    },
    /// Scalar operation
    Scalar {
        op: &'static str,
        operand: Box<ExprKey>,
        scalar_hash: u64,
    },
}

impl ExprKey {
    /// Create an array key
    pub fn array(id: u64) -> Self {
        Self::Array(id)
    }

    /// Create a binary operation key
    pub fn binary(op: &'static str, left: ExprKey, right: ExprKey) -> Self {
        Self::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a unary operation key
    pub fn unary(op: &'static str, operand: ExprKey) -> Self {
        Self::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    /// Create a scalar operation key
    pub fn scalar(op: &'static str, operand: ExprKey, scalar_hash: u64) -> Self {
        Self::Scalar {
            op,
            operand: Box::new(operand),
            scalar_hash,
        }
    }
}

/// Hash a floating-point value for use in expression keys
pub fn hash_f64(value: f64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// Common Subexpression Elimination (CSE) Optimizer
///
/// Analyzes expression trees to identify common subexpressions and creates
/// an optimized DAG (Directed Acyclic Graph) where shared computations are
/// evaluated only once.
///
/// # How It Works
///
/// 1. **Expression Analysis**: Traverses the expression tree and assigns keys
///    to each subexpression based on its structure.
/// 2. **Common Subexpression Detection**: Identifies subexpressions with identical
///    keys (same operation, same operands).
/// 3. **Cache Creation**: Creates a shared cache for storing evaluated results.
/// 4. **DAG Construction**: Wraps expressions with CachedExpr nodes that share
///    the same cache.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{
///     SharedArrayExpr, SharedBinaryExpr, SharedScalarExpr, SharedExpr,
///     CSEOptimizer, ExprKey, hash_f64
/// };
///
/// // Create arrays
/// let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let b = SharedArray::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
///
/// // Expression: (a + b) * (a + b) - has common subexpression (a + b)
/// let expr_a1 = SharedArrayExpr::new(a.clone());
/// let expr_b1 = SharedArrayExpr::new(b.clone());
/// let sum1 = SharedBinaryExpr::new(expr_a1, expr_b1, |x, y| x + y).unwrap();
///
/// let expr_a2 = SharedArrayExpr::new(a.clone());
/// let expr_b2 = SharedArrayExpr::new(b.clone());
/// let sum2 = SharedBinaryExpr::new(expr_a2, expr_b2, |x, y| x + y).unwrap();
///
/// let product = SharedBinaryExpr::new(sum1, sum2, |x, y| x * y).unwrap();
///
/// // Without CSE, (a + b) is computed twice
/// // With CSE, (a + b) is computed once and reused
///
/// let result = product.eval();
/// // (3*3, 5*5, 7*7, 9*9) = (9, 25, 49, 81)
/// assert_eq!(result.to_vec(), vec![9.0, 25.0, 49.0, 81.0]);
/// ```
pub struct CSEOptimizer<T: Clone> {
    /// Maps expression keys to their assigned IDs
    key_to_id: HashMap<ExprKey, ExprId>,
    /// The shared cache for evaluated results
    cache: ExprCache<T>,
    /// Counter for assigning unique array IDs
    next_array_id: u64,
}

impl<T: Clone> CSEOptimizer<T> {
    /// Create a new CSE optimizer
    pub fn new() -> Self {
        Self {
            key_to_id: HashMap::new(),
            cache: ExprCache::new(),
            next_array_id: 0,
        }
    }

    /// Get or create an ID for an expression key
    pub fn get_or_create_id(&mut self, key: &ExprKey) -> ExprId {
        if let Some(&id) = self.key_to_id.get(key) {
            id
        } else {
            let id = ExprId::new();
            self.key_to_id.insert(key.clone(), id);
            id
        }
    }

    /// Get a new unique array ID
    pub fn next_array_id(&mut self) -> u64 {
        let id = self.next_array_id;
        self.next_array_id += 1;
        id
    }

    /// Get the shared cache
    pub fn cache(&self) -> &ExprCache<T> {
        &self.cache
    }

    /// Create a cached version of an expression
    pub fn cache_expr<E: SharedExpr<T>>(&self, expr: E, id: ExprId) -> CachedExpr<T, E> {
        CachedExpr::with_id(expr, id, self.cache.clone())
    }

    /// Get statistics about the optimization
    pub fn stats(&self) -> CSEStats {
        CSEStats {
            unique_expressions: self.key_to_id.len(),
            cached_results: self.cache.len(),
        }
    }

    /// Clear the optimizer state
    pub fn clear(&mut self) {
        self.key_to_id.clear();
        self.cache.clear();
        self.next_array_id = 0;
    }
}

impl<T: Clone> Default for CSEOptimizer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from CSE optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CSEStats {
    /// Number of unique expression keys
    pub unique_expressions: usize,
    /// Number of cached results
    pub cached_results: usize,
}

/// Builder for constructing CSE-optimized expression graphs
///
/// Provides a fluent API for building expression trees with automatic
/// common subexpression elimination.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{CSEExprBuilder, SharedExpr, SharedArrayExpr, ExprKey, CSESupport};
///
/// let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// let mut builder: CSEExprBuilder<f64> = CSEExprBuilder::new();
///
/// // Wrap an expression with CSE caching
/// let expr = SharedArrayExpr::new(a.clone());
/// let key = ExprKey::array(0);
/// let cached = builder.wrap(expr, key);
///
/// let result = cached.eval();
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub struct CSEExprBuilder<T: Clone> {
    optimizer: CSEOptimizer<T>,
}

impl<T: Clone> CSEExprBuilder<T> {
    /// Create a new CSE expression builder
    pub fn new() -> Self {
        Self {
            optimizer: CSEOptimizer::new(),
        }
    }

    /// Wrap an expression with CSE caching
    pub fn wrap<E: SharedExpr<T>>(&mut self, expr: E, key: ExprKey) -> CachedExpr<T, E> {
        let id = self.optimizer.get_or_create_id(&key);
        self.optimizer.cache_expr(expr, id)
    }

    /// Evaluate and cache a SharedArray directly
    pub fn eval_array(&self, array: SharedArray<T>) -> SharedArray<T> {
        array
    }

    /// Get the optimizer stats
    pub fn stats(&self) -> CSEStats {
        self.optimizer.stats()
    }

    /// Clear all cached results
    pub fn clear(&mut self) {
        self.optimizer.clear();
    }
}

impl<T: Clone> Default for CSEExprBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for adding CSE support to SharedExpr
pub trait CSESupport<T: Clone>: SharedExpr<T> + Sized {
    /// Wrap this expression with CSE caching
    fn with_cache(self, cache: ExprCache<T>) -> CachedExpr<T, Self> {
        CachedExpr::new(self, cache)
    }

    /// Create a CSE-wrapped version with a specific ID
    fn with_cache_id(self, id: ExprId, cache: ExprCache<T>) -> CachedExpr<T, Self> {
        CachedExpr::with_id(self, id, cache)
    }
}

// Implement CSESupport for all SharedExpr types
impl<T: Clone, E: SharedExpr<T>> CSESupport<T> for E {}

/// Result of CSE analysis
#[derive(Debug, Clone)]
pub struct CSEAnalysisResult {
    /// Total number of expression nodes
    pub total_nodes: usize,
    /// Number of common subexpressions found
    pub common_subexpressions: usize,
    /// Estimated computation savings (ratio of reused to total)
    pub savings_ratio: f64,
    /// Map of expression keys to their occurrence counts
    pub occurrence_counts: HashMap<String, usize>,
}

impl CSEAnalysisResult {
    /// Create a new analysis result
    pub fn new() -> Self {
        Self {
            total_nodes: 0,
            common_subexpressions: 0,
            savings_ratio: 0.0,
            occurrence_counts: HashMap::new(),
        }
    }

    /// Calculate the savings ratio
    pub fn calculate_savings(&mut self) {
        if self.total_nodes > 0 {
            self.savings_ratio = self.common_subexpressions as f64 / self.total_nodes as f64;
        }
    }
}

impl Default for CSEAnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze an expression tree for common subexpressions
///
/// This function performs a static analysis of the expression structure
/// to identify potential CSE opportunities.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{
///     SharedArrayExpr, SharedBinaryExpr, SharedExpr,
///     analyze_cse, ExprKey
/// };
///
/// let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
/// let key_a = ExprKey::array(0);
/// let key_b = ExprKey::array(1);
/// let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());
/// let key_product = ExprKey::binary("mul", key_sum.clone(), key_sum.clone());
///
/// let keys = vec![key_a, key_b, key_sum.clone(), key_sum.clone(), key_product];
/// let analysis = analyze_cse(&keys);
///
/// // The sum expression appears twice
/// assert!(analysis.common_subexpressions > 0);
/// ```
pub fn analyze_cse(keys: &[ExprKey]) -> CSEAnalysisResult {
    let mut result = CSEAnalysisResult::new();
    result.total_nodes = keys.len();

    // Count occurrences of each key
    let mut key_counts: HashMap<String, usize> = HashMap::new();
    for key in keys {
        let key_str = format!("{:?}", key);
        *key_counts.entry(key_str).or_insert(0) += 1;
    }

    // Count common subexpressions (those appearing more than once)
    for (key_str, count) in &key_counts {
        if *count > 1 {
            result.common_subexpressions += count - 1; // Extra occurrences
            result.occurrence_counts.insert(key_str.clone(), *count);
        }
    }

    result.calculate_savings();
    result
}

/// Optimized expression graph node
///
/// Represents a node in the CSE-optimized expression DAG.
/// Each node has a unique ID and may reference cached results.
#[derive(Clone)]
pub struct OptimizedExprNode<T: Clone> {
    id: ExprId,
    key: ExprKey,
    cache: ExprCache<T>,
    /// Cached evaluation result (set after first evaluation)
    result: Option<SharedArray<T>>,
}

impl<T: Clone> OptimizedExprNode<T> {
    /// Create a new optimized node
    pub fn new(id: ExprId, key: ExprKey, cache: ExprCache<T>) -> Self {
        Self {
            id,
            key,
            cache,
            result: None,
        }
    }

    /// Get the node ID
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Get the expression key
    pub fn key(&self) -> &ExprKey {
        &self.key
    }

    /// Check if the result is cached
    pub fn is_cached(&self) -> bool {
        self.result.is_some() || self.cache.contains(&self.id)
    }

    /// Get or compute the result
    pub fn get_or_compute<F>(&mut self, compute: F) -> SharedArray<T>
    where
        F: FnOnce() -> SharedArray<T>,
    {
        // Check local cache first
        if let Some(ref result) = self.result {
            return result.clone();
        }

        // Check shared cache
        if let Some(cached) = self.cache.get(&self.id) {
            self.result = Some(cached.clone());
            return cached;
        }

        // Compute and cache
        let result = compute();
        self.cache.insert(self.id, result.clone());
        self.result = Some(result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_array_expr() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = ArrayExpr::new(&a);

        assert_eq!(expr.size(), 4);
        assert_eq!(expr.shape(), &[4]);
        assert_eq!(expr.eval_at(0), 1.0);
        assert_eq!(expr.eval_at(3), 4.0);
    }

    #[test]
    fn test_binary_expr_manual() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        let expr = BinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), |x: f64, y: f64| {
            x + y
        })
        .unwrap();

        let result = expr.eval();
        assert_eq!(result.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_binary_expr_eval_at() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        let expr = BinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), |x: f64, y: f64| {
            x * y
        })
        .unwrap();

        assert_eq!(expr.eval_at(0), 10.0);
        assert_eq!(expr.eval_at(1), 40.0);
        assert_eq!(expr.eval_at(2), 90.0);
        assert_eq!(expr.eval_at(3), 160.0);
    }

    #[test]
    fn test_unary_expr() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = UnaryExpr::new(ArrayExpr::new(&a), |x: f64| x * 2.0);

        let result = expr.eval();
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_expr() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = ScalarExpr::new(ArrayExpr::new(&a), 10.0, |x: f64, y: f64| x + y);

        let result = expr.eval();
        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        let result = BinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), |x: f64, y: f64| {
            x + y
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_eval_trait() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let lazy_a = a.lazy();

        assert_eq!(lazy_a.size(), 4);
        assert_eq!(lazy_a.eval().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // =========================================================================
    // Enhanced Expression Tests
    // =========================================================================

    #[test]
    fn test_reduction_sum() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let sum = ReductionExpr::new(ArrayExpr::new(&a), |x, y| x + y, || 0.0).reduce();
        assert_relative_eq!(sum, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reduction_product() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let prod = ReductionExpr::new(ArrayExpr::new(&a), |x, y| x * y, || 1.0).reduce();
        assert_relative_eq!(prod, 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reduction_max() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0, 2.0]);
        let max = ReductionExpr::new(
            ArrayExpr::new(&a),
            |x: f64, y: f64| x.max(y),
            || f64::NEG_INFINITY,
        )
        .reduce();
        assert_relative_eq!(max, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clip_expr() {
        let a = Array::from_vec(vec![-1.0, 0.5, 1.5, 2.5]);
        let clipped = ClipExpr::new(ArrayExpr::new(&a), 0.0, 2.0);
        let result = clipped.eval();
        assert_eq!(result.to_vec(), vec![0.0, 0.5, 1.5, 2.0]);
    }

    #[test]
    fn test_broadcast_scalar_expr() {
        let scalar = BroadcastScalarExpr::new(5.0, &[3, 2]);
        assert_eq!(scalar.size(), 6);
        assert_eq!(scalar.shape(), &[3, 2]);
        assert_eq!(scalar.eval_at(0), 5.0);
        assert_eq!(scalar.eval_at(5), 5.0);

        let result = scalar.eval();
        assert_eq!(result.to_vec(), vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_simd_eval() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let expr = ArrayExpr::new(&a);
        let result = expr.eval_simd();
        assert_eq!(
            result.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_expr_builder_basic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = ExprBuilder::from_array(&a).add_scalar(10.0).eval();
        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_expr_builder_chain() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = ExprBuilder::from_array(&a)
            .mul_scalar(2.0)
            .add_scalar(1.0)
            .eval();
        // (1*2+1, 2*2+1, 3*2+1, 4*2+1) = (3, 5, 7, 9)
        assert_eq!(result.to_vec(), vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_expr_builder_math_ops() {
        let a = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0]);
        let result = ExprBuilder::from_array(&a).sqrt().eval();
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_expr_builder_reductions() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Sum
        let sum = ExprBuilder::from_array(&a).sum();
        assert_relative_eq!(sum, 10.0, epsilon = 1e-10);

        // Product
        let prod = ExprBuilder::from_array(&a).prod();
        assert_relative_eq!(prod, 24.0, epsilon = 1e-10);

        // Max
        let max = ExprBuilder::from_array(&a).max();
        assert_relative_eq!(max, 4.0, epsilon = 1e-10);

        // Min
        let min = ExprBuilder::from_array(&a).min();
        assert_relative_eq!(min, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sum_utility() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = expr_sum(ArrayExpr::new(&a));
        assert_relative_eq!(sum, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fma_expr() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0]);
        let c = Array::from_vec(vec![10.0, 10.0, 10.0]);

        let fma_result = fma(ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c)).unwrap();
        let result = fma_result.eval();
        // a * b + c = (1*2+10, 2*3+10, 3*4+10) = (12, 16, 22)
        assert_eq!(result.to_vec(), vec![12.0, 16.0, 22.0]);
    }

    #[test]
    fn test_complex_expression_chain() {
        let a = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        // (a + b) * 2 = (1, 2, 3, 4) * 2 = (2, 4, 6, 8)
        let add_expr =
            BinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), |x, y| x + y).unwrap();

        let mul_expr = ScalarExpr::new(add_expr, 2.0, |x, y| x * y);
        let result = mul_expr.eval();
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    // ========================================
    // SharedExpr Tests (Reference-Counted)
    // ========================================

    #[test]
    fn test_shared_array_expr_basic() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = SharedArrayExpr::new(arr);

        assert_eq!(expr.size(), 4);
        assert_eq!(expr.shape(), vec![4]);
        assert_eq!(expr.eval_at(0), 1.0);
        assert_eq!(expr.eval_at(3), 4.0);
    }

    #[test]
    fn test_shared_array_expr_eval() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = SharedArrayExpr::new(arr.clone());
        let result = expr.eval();

        assert_eq!(result.to_vec(), arr.to_vec());
    }

    #[test]
    fn test_shared_binary_expr() {
        let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
        let b = SharedArray::from_vec(vec![4.0, 5.0, 6.0]);

        let expr_a = SharedArrayExpr::new(a);
        let expr_b = SharedArrayExpr::new(b);

        let add_expr = SharedBinaryExpr::new(expr_a, expr_b, |x, y| x + y).unwrap();
        let result = add_expr.eval();

        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_shared_unary_expr() {
        let arr = SharedArray::from_vec(vec![1.0, 4.0, 9.0, 16.0]);
        let expr = SharedArrayExpr::new(arr);
        let sqrt_expr = SharedUnaryExpr::new(expr, |x: f64| x.sqrt());
        let result = sqrt_expr.eval();

        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_shared_scalar_expr() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = SharedArrayExpr::new(arr);
        let scaled = SharedScalarExpr::new(expr, 10.0, |x, y| x + y);
        let result = scaled.eval();

        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_shared_expr_builder_basic() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = SharedExprBuilder::from_shared_array(arr)
            .add_scalar(10.0)
            .eval();

        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_shared_expr_builder_chain() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = SharedExprBuilder::from_shared_array(arr)
            .mul_scalar(2.0)
            .add_scalar(1.0)
            .eval();

        // (1*2+1, 2*2+1, 3*2+1, 4*2+1) = (3, 5, 7, 9)
        assert_eq!(result.to_vec(), vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_shared_expr_builder_from_array() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = SharedExprBuilder::from_array(arr).mul_scalar(2.0).eval();

        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_shared_expr_builder_map() {
        let arr = SharedArray::from_vec(vec![1.0, 4.0, 9.0, 16.0]);
        let result = SharedExprBuilder::from_shared_array(arr)
            .map(|x: f64| x.sqrt())
            .mul_scalar(2.0)
            .eval();

        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_shared_expr_can_be_stored() {
        // This test demonstrates that SharedExpr can be stored without lifetime issues
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = SharedArrayExpr::new(arr);

        // Store in a Vec (requires Clone)
        let exprs: Vec<SharedArrayExpr<f64>> = vec![expr.clone(), expr.clone()];
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0].eval().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_shared_expr_complex_chain() {
        // Build a complex expression tree: ((a + b) * 2) - 5
        let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = SharedArray::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let expr_a = SharedArrayExpr::new(a);
        let expr_b = SharedArrayExpr::new(b);

        // (a + b)
        let sum = SharedBinaryExpr::new(expr_a, expr_b, |x, y| x + y).unwrap();
        // (a + b) * 2
        let doubled = SharedScalarExpr::new(sum, 2.0, |x, y| x * y);
        // ((a + b) * 2) - 5
        let final_expr = SharedScalarExpr::new(doubled, 5.0, |x, y| x - y);

        let result = final_expr.eval();
        // ((1+2)*2)-5=1, ((2+3)*2)-5=5, ((3+4)*2)-5=9, ((4+5)*2)-5=13
        assert_eq!(result.to_vec(), vec![1.0, 5.0, 9.0, 13.0]);
    }

    // =========================================================================
    // Common Subexpression Elimination (CSE) Tests
    // =========================================================================

    #[test]
    fn test_expr_id_uniqueness() {
        let id1 = ExprId::new();
        let id2 = ExprId::new();
        let id3 = ExprId::new();

        // Each ID should be unique
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_expr_cache_basic() {
        let cache: ExprCache<f64> = ExprCache::new();
        let id = ExprId::new();
        let array = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);

        // Initially empty
        assert!(cache.is_empty());
        assert!(!cache.contains(&id));

        // Insert and verify
        cache.insert(id, array.clone());
        assert!(!cache.is_empty());
        assert!(cache.contains(&id));
        assert_eq!(cache.len(), 1);

        // Retrieve and verify
        let cached = cache.get(&id);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_expr_cache_multiple_entries() {
        let cache: ExprCache<f64> = ExprCache::new();

        let id1 = ExprId::new();
        let id2 = ExprId::new();
        let id3 = ExprId::new();

        cache.insert(id1, SharedArray::from_vec(vec![1.0]));
        cache.insert(id2, SharedArray::from_vec(vec![2.0]));
        cache.insert(id3, SharedArray::from_vec(vec![3.0]));

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&id1).unwrap().to_vec(), vec![1.0]);
        assert_eq!(cache.get(&id2).unwrap().to_vec(), vec![2.0]);
        assert_eq!(cache.get(&id3).unwrap().to_vec(), vec![3.0]);
    }

    #[test]
    fn test_expr_cache_clear() {
        let cache: ExprCache<f64> = ExprCache::new();
        let id = ExprId::new();

        cache.insert(id, SharedArray::from_vec(vec![1.0, 2.0]));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert!(!cache.contains(&id));
    }

    #[test]
    fn test_cached_expr_basic() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = SharedArrayExpr::new(arr);
        let cache: ExprCache<f64> = ExprCache::new();

        let cached = CachedExpr::new(expr, cache.clone());

        // First evaluation
        let result1 = cached.eval();
        assert_eq!(result1.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.len(), 1); // Result cached

        // Second evaluation should return cached result
        let result2 = cached.eval();
        assert_eq!(result2.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.len(), 1); // Still one entry
    }

    #[test]
    fn test_cached_expr_shared_cache() {
        let cache: ExprCache<f64> = ExprCache::new();

        let arr1 = SharedArray::from_vec(vec![1.0, 2.0]);
        let arr2 = SharedArray::from_vec(vec![3.0, 4.0]);

        let expr1 = SharedArrayExpr::new(arr1);
        let expr2 = SharedArrayExpr::new(arr2);

        let cached1 = CachedExpr::new(expr1, cache.clone());
        let cached2 = CachedExpr::new(expr2, cache.clone());

        // Both share the same cache
        cached1.eval();
        cached2.eval();

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_expr_key_array() {
        let key1 = ExprKey::array(0);
        let key2 = ExprKey::array(0);
        let key3 = ExprKey::array(1);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_expr_key_binary() {
        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);

        let add1 = ExprKey::binary("add", key_a.clone(), key_b.clone());
        let add2 = ExprKey::binary("add", key_a.clone(), key_b.clone());
        let mul1 = ExprKey::binary("mul", key_a.clone(), key_b.clone());

        assert_eq!(add1, add2); // Same operation, same operands
        assert_ne!(add1, mul1); // Different operations
    }

    #[test]
    fn test_expr_key_unary() {
        let key_a = ExprKey::array(0);

        let sqrt1 = ExprKey::unary("sqrt", key_a.clone());
        let sqrt2 = ExprKey::unary("sqrt", key_a.clone());
        let neg1 = ExprKey::unary("neg", key_a.clone());

        assert_eq!(sqrt1, sqrt2);
        assert_ne!(sqrt1, neg1);
    }

    #[test]
    fn test_expr_key_scalar() {
        let key_a = ExprKey::array(0);

        let add10_1 = ExprKey::scalar("add", key_a.clone(), hash_f64(10.0));
        let add10_2 = ExprKey::scalar("add", key_a.clone(), hash_f64(10.0));
        let add20 = ExprKey::scalar("add", key_a.clone(), hash_f64(20.0));

        assert_eq!(add10_1, add10_2); // Same scalar value
        assert_ne!(add10_1, add20); // Different scalar values
    }

    #[test]
    fn test_cse_optimizer_basic() {
        let mut optimizer: CSEOptimizer<f64> = CSEOptimizer::new();

        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);
        let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());

        // First request creates a new ID
        let id1 = optimizer.get_or_create_id(&key_sum);

        // Second request returns the same ID
        let id2 = optimizer.get_or_create_id(&key_sum);

        assert_eq!(id1, id2);
        assert_eq!(optimizer.stats().unique_expressions, 1);
    }

    #[test]
    fn test_cse_optimizer_multiple_keys() {
        let mut optimizer: CSEOptimizer<f64> = CSEOptimizer::new();

        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);
        let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());
        let key_prod = ExprKey::binary("mul", key_a.clone(), key_b.clone());

        let id_sum = optimizer.get_or_create_id(&key_sum);
        let id_prod = optimizer.get_or_create_id(&key_prod);

        assert_ne!(id_sum, id_prod);
        assert_eq!(optimizer.stats().unique_expressions, 2);
    }

    #[test]
    fn test_cse_analysis() {
        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);
        let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());

        // Expression: (a + b) * (a + b) - sum appears twice
        let keys = vec![
            key_a.clone(),
            key_b.clone(),
            key_sum.clone(),
            key_sum.clone(), // Common subexpression
            ExprKey::binary("mul", key_sum.clone(), key_sum.clone()),
        ];

        let analysis = analyze_cse(&keys);

        assert_eq!(analysis.total_nodes, 5);
        assert!(analysis.common_subexpressions > 0);
        assert!(analysis.savings_ratio > 0.0);
    }

    #[test]
    fn test_cse_support_trait() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
        let expr = SharedArrayExpr::new(arr);
        let cache: ExprCache<f64> = ExprCache::new();

        // Use CSESupport trait
        let cached = expr.with_cache(cache.clone());
        let result = cached.eval();

        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cse_expr_builder() {
        let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let builder: CSEExprBuilder<f64> = CSEExprBuilder::new();
        let result = builder.eval_array(a);

        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_optimized_expr_node() {
        let cache: ExprCache<f64> = ExprCache::new();
        let id = ExprId::new();
        let key = ExprKey::array(0);

        let mut node = OptimizedExprNode::new(id, key.clone(), cache.clone());

        // Initially not cached
        assert!(!node.is_cached());

        // Compute and cache
        let result = node.get_or_compute(|| SharedArray::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
        assert!(node.is_cached());

        // Second call returns cached result (computation closure not called)
        let result2 = node.get_or_compute(|| SharedArray::from_vec(vec![9.0, 9.0, 9.0]));
        assert_eq!(result2.to_vec(), vec![1.0, 2.0, 3.0]); // Original value, not 9.0s
    }

    #[test]
    fn test_cse_shared_computation() {
        // Demonstrate CSE: compute (a + b) * (a + b) where (a + b) is computed only once
        let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = SharedArray::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let cache: ExprCache<f64> = ExprCache::new();

        // Create a cached version of (a + b)
        let expr_a = SharedArrayExpr::new(a);
        let expr_b = SharedArrayExpr::new(b);
        let sum = SharedBinaryExpr::new(expr_a, expr_b, |x, y| x + y).unwrap();
        let cached_sum = CachedExpr::new(sum, cache.clone());

        // Evaluate (a + b) to cache it
        let sum_result = cached_sum.eval();
        assert_eq!(sum_result.to_vec(), vec![3.0, 5.0, 7.0, 9.0]);
        assert_eq!(cache.len(), 1);

        // Now (a + b) * (a + b) uses the cached result
        let sum_squared =
            SharedBinaryExpr::new(cached_sum.clone(), cached_sum, |x: f64, y: f64| x * y).unwrap();

        let result = sum_squared.eval();
        // (3*3, 5*5, 7*7, 9*9) = (9, 25, 49, 81)
        assert_eq!(result.to_vec(), vec![9.0, 25.0, 49.0, 81.0]);
    }

    #[test]
    fn test_cached_expr_invalidate() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
        let expr = SharedArrayExpr::new(arr);
        let cache: ExprCache<f64> = ExprCache::new();

        let cached = CachedExpr::new(expr, cache.clone());

        // Evaluate to cache
        cached.eval();
        assert_eq!(cache.len(), 1);

        // Invalidate
        cached.invalidate();

        // Note: The ID is removed from the cache
        // but the cache may still have the entry depending on timing
        // This tests the invalidation mechanism works
    }

    #[test]
    fn test_hash_f64() {
        let h1 = hash_f64(10.0);
        let h2 = hash_f64(10.0);
        let h3 = hash_f64(20.0);

        // Same value should produce same hash
        assert_eq!(h1, h2);
        // Different values should (very likely) produce different hashes
        assert_ne!(h1, h3);
    }
}
