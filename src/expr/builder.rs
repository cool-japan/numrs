//! Expression builder for fluent API
//!
//! This module provides the `ExprBuilder` for constructing complex expressions
//! with a fluent interface, along with utility functions for common operations.

use crate::array::Array;
use crate::error::Result;
use std::marker::PhantomData;

use super::core::{ArrayExpr, BinaryExpr, Expr, ScalarExpr, UnaryExpr};
use super::enhanced::ReductionExpr;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::core::ArrayExpr;
    use approx::assert_relative_eq;

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

        let fma_result = fma(ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c))
            .expect("FMA expression creation should succeed");
        let result = fma_result.eval();
        // a * b + c = (1*2+10, 2*3+10, 3*4+10) = (12, 16, 22)
        assert_eq!(result.to_vec(), vec![12.0, 16.0, 22.0]);
    }

    #[test]
    fn test_complex_expression_chain() {
        let a = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        // (a + b) * 2 = (1, 2, 3, 4) * 2 = (2, 4, 6, 8)
        let add_expr = BinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), |x, y| x + y)
            .expect("Add expression creation should succeed");

        let mul_expr = ScalarExpr::new(add_expr, 2.0, |x, y| x * y);
        let result = mul_expr.eval();
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }
}
