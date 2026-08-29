//! Core expression traits and types for lazy evaluation
//!
//! This module provides the foundational building blocks for expression templates:
//! - `Expr<T>` trait for lazy evaluation
//! - Basic expression types: `ArrayExpr`, `BinaryExpr`, `UnaryExpr`, `ScalarExpr`
//! - `LazyEval` trait for converting arrays to lazy expressions

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

        Array::from_vec_shape(data, self.shape()).unwrap_or_else(|e| panic!("{e}"))
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
    /// Create a new ArrayExpr from an array reference
    pub fn new(array: &'a Array<T>) -> Self {
        let shape = array.shape();
        Self { array, shape }
    }
}

impl<'a, T: Clone> Expr<T> for ArrayExpr<'a, T> {
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        // CRITICAL FIX: Direct index access instead of to_vec() for every element
        // This changes complexity from O(n²) to O(1) per access
        self.array.get_flat(index).expect("Index out of bounds")
    }

    #[inline]
    fn size(&self) -> usize {
        self.array.size()
    }

    #[inline]
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
    /// Create a new binary expression
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
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let left_val = self.left.eval_at(index);
        let right_val = self.right.eval_at(index);
        (self.op)(left_val, right_val)
    }

    #[inline]
    fn size(&self) -> usize {
        self.left.size()
    }

    #[inline]
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
    /// Create a new unary expression
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
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.op)(val)
    }

    #[inline]
    fn size(&self) -> usize {
        self.expr.size()
    }

    #[inline]
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
    /// Create a new scalar expression
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
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.op)(val, self.scalar.clone())
    }

    #[inline]
    fn size(&self) -> usize {
        self.expr.size()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

/// Extension trait to add lazy evaluation methods to Array
pub trait LazyEval<T: Clone> {
    /// Convert array to lazy expression
    fn lazy(&self) -> ArrayExpr<'_, T>;
}

impl<T: Clone> LazyEval<T> for Array<T> {
    fn lazy(&self) -> ArrayExpr<'_, T> {
        ArrayExpr::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        .expect("Binary expression creation should succeed");

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
        .expect("Binary expression creation should succeed");

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
}
