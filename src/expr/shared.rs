//! SharedArray expression types (Reference-Counted, Lifetime-Free)
//!
//! These types solve the lifetime challenges of regular expressions by using SharedArray's
//! reference counting. Expressions can be stored, passed around, and composed
//! without complex lifetime annotations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::shared_array::SharedArray;
use std::marker::PhantomData;

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
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        // OPTIMIZATION: Use get_flat instead of to_vec() to avoid O(n) copy
        // This changes complexity from O(n²) to O(1) per access
        self.array.get_flat(index).expect("Index out of bounds")
    }

    #[inline]
    fn size(&self) -> usize {
        self.array.size()
    }

    #[inline]
    fn shape(&self) -> Vec<usize> {
        self.array.shape()
    }

    #[inline]
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
    /// Create a new unary expression
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
    /// Create a new scalar expression
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

#[cfg(test)]
mod tests {
    use super::*;

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

        let add_expr = SharedBinaryExpr::new(expr_a, expr_b, |x, y| x + y)
            .expect("Binary expression creation should succeed");
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
        let sum = SharedBinaryExpr::new(expr_a, expr_b, |x, y| x + y)
            .expect("Binary expression creation should succeed");
        // (a + b) * 2
        let doubled = SharedScalarExpr::new(sum, 2.0, |x, y| x * y);
        // ((a + b) * 2) - 5
        let final_expr = SharedScalarExpr::new(doubled, 5.0, |x, y| x - y);

        let result = final_expr.eval();
        // ((1+2)*2)-5=1, ((2+3)*2)-5=5, ((3+4)*2)-5=9, ((4+5)*2)-5=13
        assert_eq!(result.to_vec(), vec![1.0, 5.0, 9.0, 13.0]);
    }
}
