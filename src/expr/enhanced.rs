//! Enhanced expression types for advanced operations
//!
//! This module provides additional expression types beyond the basic ones:
//! - `ReductionExpr` - Reduction operations (sum, product, max, min)
//! - `WhereExpr` - Conditional selection
//! - `ClipExpr` - Value clamping
//! - `BroadcastScalarExpr` - Broadcasting scalar values

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::marker::PhantomData;

use super::core::Expr;

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
    /// Create a new reduction expression
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
    /// Create a new conditional expression
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
    /// Create a new clip expression
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
    /// Create a new broadcast scalar expression
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::core::ArrayExpr;
    use approx::assert_relative_eq;

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
}
