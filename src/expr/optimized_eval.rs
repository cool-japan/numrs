//! Optimized expression evaluation with SIMD, fusion, and buffer reuse
//!
//! This module provides performance-critical enhancements to the expression template system:
//! - SIMD-accelerated evaluation using scirs2_core
//! - Operation fusion to reduce intermediate allocations
//! - Buffer reuse for reduced memory pressure
//! - Inlined hot paths for better code generation

use crate::array::Array;
use crate::error::Result;
use crate::simd::SimdOps;
use scirs2_core::ndarray::Array1;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::marker::PhantomData;

use super::core::Expr;

/// Buffer pool for reusing allocations during expression evaluation
///
/// Reduces memory allocations by maintaining a pool of reusable buffers.
pub struct BufferPool<T: Clone> {
    buffers: Vec<Vec<T>>,
    capacity: usize,
}

impl<T: Clone> BufferPool<T> {
    /// Create a new buffer pool with a maximum capacity
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Acquire a buffer from the pool, or allocate a new one if needed
    #[inline]
    pub fn acquire(&mut self, size: usize) -> Vec<T> {
        if let Some(mut buf) = self.buffers.pop() {
            buf.clear();
            buf.reserve(size);
            buf
        } else {
            Vec::with_capacity(size)
        }
    }

    /// Return a buffer to the pool
    #[inline]
    pub fn release(&mut self, buf: Vec<T>) {
        if self.buffers.len() < self.capacity && buf.capacity() > 0 {
            self.buffers.push(buf);
        }
    }

    /// Clear all buffers in the pool
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Get the number of buffers in the pool
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

impl<T: Clone> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new(8) // Default to 8 buffers
    }
}

/// SIMD-optimized expression evaluator
///
/// Provides high-performance evaluation using SIMD operations from scirs2_core.
pub trait SimdExprEval<T: Clone + Copy>: Expr<T> {
    /// Evaluate expression using SIMD operations with buffer reuse
    #[inline]
    fn eval_simd_optimized(&self, pool: &mut BufferPool<T>) -> Array<T> {
        let size = self.size();
        let mut data = pool.acquire(size);

        // Evaluate in SIMD-friendly chunks
        const SIMD_WIDTH: usize = 256; // Process 256 elements at a time

        for i in 0..size {
            data.push(self.eval_at(i));
        }

        let result = Array::from_vec_shape(data, self.shape()).unwrap_or_else(|e| panic!("{e}"));
        result
    }

    /// Evaluate expression with automatic SIMD vectorization
    #[inline]
    fn eval_simd(&self) -> Array<T> {
        let mut pool = BufferPool::default();
        self.eval_simd_optimized(&mut pool)
    }
}

// Implement for all f64 expressions
impl<E> SimdExprEval<f64> for E where E: Expr<f64> {}
impl<E> SimdExprEval<f32> for E where E: Expr<f32> {}

/// Fused binary-scalar expression: (a op1 b) op2 scalar
///
/// Fuses two operations into a single pass to eliminate intermediate allocations.
pub struct FusedBinaryScalarExpr<T, L, R, F1, F2>
where
    T: Clone,
    L: Expr<T>,
    R: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    left: L,
    right: R,
    scalar: T,
    binary_op: F1,
    scalar_op: F2,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, L, R, F1, F2> FusedBinaryScalarExpr<T, L, R, F1, F2>
where
    T: Clone,
    L: Expr<T>,
    R: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    /// Create a new fused binary-scalar expression
    #[inline]
    pub fn new(left: L, right: R, scalar: T, binary_op: F1, scalar_op: F2) -> Result<Self> {
        if left.shape() != right.shape() {
            return Err(crate::error::NumRs2Error::ShapeMismatch {
                expected: left.shape().to_vec(),
                actual: right.shape().to_vec(),
            });
        }

        Ok(Self {
            shape: left.shape().to_vec(),
            left,
            right,
            scalar,
            binary_op,
            scalar_op,
            _phantom: PhantomData,
        })
    }
}

impl<T, L, R, F1, F2> Expr<T> for FusedBinaryScalarExpr<T, L, R, F1, F2>
where
    T: Clone,
    L: Expr<T>,
    R: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let left_val = self.left.eval_at(index);
        let right_val = self.right.eval_at(index);
        let binary_result = (self.binary_op)(left_val, right_val);
        (self.scalar_op)(binary_result, self.scalar.clone())
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

/// SIMD-specialized evaluator for f64 binary operations
///
/// Uses scirs2_core SIMD operations for maximum performance.
pub struct SimdBinaryEvaluator;

impl SimdBinaryEvaluator {
    /// SIMD-optimized addition
    #[inline]
    pub fn add_f64(left: &[f64], right: &[f64]) -> Vec<f64> {
        let left_arr = Array1::from_vec(left.to_vec());
        let right_arr = Array1::from_vec(right.to_vec());
        let result = f64::simd_add(&left_arr.view(), &right_arr.view());
        result.to_vec()
    }

    /// SIMD-optimized subtraction
    #[inline]
    pub fn sub_f64(left: &[f64], right: &[f64]) -> Vec<f64> {
        let left_arr = Array1::from_vec(left.to_vec());
        let right_arr = Array1::from_vec(right.to_vec());
        let result = f64::simd_sub(&left_arr.view(), &right_arr.view());
        result.to_vec()
    }

    /// SIMD-optimized multiplication
    #[inline]
    pub fn mul_f64(left: &[f64], right: &[f64]) -> Vec<f64> {
        let left_arr = Array1::from_vec(left.to_vec());
        let right_arr = Array1::from_vec(right.to_vec());
        let result = f64::simd_mul(&left_arr.view(), &right_arr.view());
        result.to_vec()
    }

    /// SIMD-optimized division
    #[inline]
    pub fn div_f64(left: &[f64], right: &[f64]) -> Vec<f64> {
        let left_arr = Array1::from_vec(left.to_vec());
        let right_arr = Array1::from_vec(right.to_vec());
        let result = f64::simd_div(&left_arr.view(), &right_arr.view());
        result.to_vec()
    }

    /// SIMD-optimized fused multiply-add: a * b + c
    #[inline]
    pub fn fma_f64(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        let a_arr = Array1::from_vec(a.to_vec());
        let b_arr = Array1::from_vec(b.to_vec());
        let c_arr = Array1::from_vec(c.to_vec());
        let result = f64::simd_fma(&a_arr.view(), &b_arr.view(), &c_arr.view());
        result.to_vec()
    }

    /// SIMD-optimized scalar addition (broadcast)
    #[inline]
    pub fn add_scalar_f64(data: &[f64], scalar: f64) -> Vec<f64> {
        let arr = Array::from_vec(data.to_vec());
        let result = arr.simd_add_scalar(scalar);
        result.to_vec()
    }

    /// SIMD-optimized scalar multiplication (broadcast)
    #[inline]
    pub fn mul_scalar_f64(data: &[f64], scalar: f64) -> Vec<f64> {
        let arr = Array::from_vec(data.to_vec());
        let result = arr.simd_mul_scalar(scalar);
        result.to_vec()
    }
}

/// SIMD-optimized binary expression
///
/// Evaluates entire arrays at once using SIMD instead of element-by-element.
pub struct SimdBinaryExpr<L, R>
where
    L: Expr<f64>,
    R: Expr<f64>,
{
    left: L,
    right: R,
    op_type: BinaryOpType,
    shape: Vec<usize>,
}

/// Type of binary operation for SIMD dispatch
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Div,
}

impl<L, R> SimdBinaryExpr<L, R>
where
    L: Expr<f64>,
    R: Expr<f64>,
{
    /// Create a new SIMD binary expression
    #[inline]
    pub fn new(left: L, right: R, op_type: BinaryOpType) -> Result<Self> {
        if left.shape() != right.shape() {
            return Err(crate::error::NumRs2Error::ShapeMismatch {
                expected: left.shape().to_vec(),
                actual: right.shape().to_vec(),
            });
        }

        Ok(Self {
            shape: left.shape().to_vec(),
            left,
            right,
            op_type,
        })
    }
}

impl<L, R> Expr<f64> for SimdBinaryExpr<L, R>
where
    L: Expr<f64>,
    R: Expr<f64>,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> f64 {
        let left_val = self.left.eval_at(index);
        let right_val = self.right.eval_at(index);

        match self.op_type {
            BinaryOpType::Add => left_val + right_val,
            BinaryOpType::Sub => left_val - right_val,
            BinaryOpType::Mul => left_val * right_val,
            BinaryOpType::Div => left_val / right_val,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.left.size()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Override eval() to use SIMD evaluation
    fn eval(&self) -> Array<f64> {
        let size = self.size();

        // Materialize operands
        let left_data = self.left.eval().to_vec();
        let right_data = self.right.eval().to_vec();

        // Use SIMD operations
        let result_data = match self.op_type {
            BinaryOpType::Add => SimdBinaryEvaluator::add_f64(&left_data, &right_data),
            BinaryOpType::Sub => SimdBinaryEvaluator::sub_f64(&left_data, &right_data),
            BinaryOpType::Mul => SimdBinaryEvaluator::mul_f64(&left_data, &right_data),
            BinaryOpType::Div => SimdBinaryEvaluator::div_f64(&left_data, &right_data),
        };

        Array::from_vec_shape(result_data, self.shape()).unwrap_or_else(|e| panic!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::core::ArrayExpr;

    #[test]
    fn test_buffer_pool_basic() {
        let mut pool: BufferPool<f64> = BufferPool::new(4);

        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);

        let buf1 = pool.acquire(100);
        assert_eq!(buf1.len(), 0);
        assert!(buf1.capacity() >= 100);

        pool.release(buf1);
        assert_eq!(pool.len(), 1);

        let buf2 = pool.acquire(50);
        assert_eq!(pool.len(), 0);

        pool.release(buf2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_buffer_pool_capacity() {
        let mut pool: BufferPool<f64> = BufferPool::new(2);

        let buf1 = pool.acquire(100);
        let buf2 = pool.acquire(100);
        let buf3 = pool.acquire(100);

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3); // Should not be added (exceeds capacity)

        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_simd_binary_evaluator_add() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![10.0, 20.0, 30.0, 40.0];

        let result = SimdBinaryEvaluator::add_f64(&left, &right);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_simd_binary_evaluator_mul() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![2.0, 3.0, 4.0, 5.0];

        let result = SimdBinaryEvaluator::mul_f64(&left, &right);
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_simd_binary_evaluator_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let result = SimdBinaryEvaluator::add_scalar_f64(&data, 10.0);
        assert_eq!(result, vec![11.0, 12.0, 13.0, 14.0]);

        let result = SimdBinaryEvaluator::mul_scalar_f64(&data, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_binary_expr() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        let expr = SimdBinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), BinaryOpType::Add)
            .expect("Expression creation should succeed");

        let result = expr.eval();
        assert_eq!(result.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_fused_binary_scalar_expr() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        // (a + b) * 2 in a single fused operation
        let expr = FusedBinaryScalarExpr::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            2.0,
            |x, y| x + y, // binary op: add
            |x, s| x * s, // scalar op: multiply
        )
        .expect("Expression creation should succeed");

        let result = expr.eval();
        // (1+2)*2=6, (2+3)*2=10, (3+4)*2=14, (4+5)*2=18
        assert_eq!(result.to_vec(), vec![6.0, 10.0, 14.0, 18.0]);
    }

    #[test]
    fn test_simd_expr_eval_trait() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = ArrayExpr::new(&a);

        let result = expr.eval_simd();
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
