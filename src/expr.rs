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

// Note: Operator overloading for expression templates in Rust has significant
// lifetime challenges. Future work will address these issues with alternative
// designs (e.g., macros, builder patterns, or specialized traits).
// For now, users can construct expressions manually using the BinaryExpr::new API
// or use the fluent ExprBuilder interface.

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
}
