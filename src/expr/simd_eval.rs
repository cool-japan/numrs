//! SIMD-optimized batch evaluation for expressions
//!
//! This module provides the `SimdEval` trait for efficient batch evaluation
//! of expression templates using SIMD operations where applicable.

use crate::array::Array;

use super::core::{ArrayExpr, BinaryExpr, Expr, ScalarExpr, UnaryExpr};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
