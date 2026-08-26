//! Expression Template Operation Fusion for Performance Optimization
//!
//! This module implements operation fusion patterns for the expression template system.
//! Operation fusion detects and combines multiple operations into fewer, more efficient
//! passes over the data, eliminating intermediate allocations and improving cache locality.
//!
//! # Fusion Patterns
//!
//! - **Element-wise chain fusion**: Chains like `(a + b) * c` fused into a single vectorized pass
//! - **Scalar broadcast fusion**: Operations like `a * 2.0 + 3.0` fused into a single FMA
//! - **Reduction fusion**: Element-wise ops followed by reduction (e.g., `sum(a * b)` -> dot product)
//! - **FMA (Fused Multiply-Add)**: Hardware FMA when available via SIMD
//!
//! # Architecture
//!
//! The fusion system operates at the expression tree level. The `FusionDetector` inspects
//! expression tree patterns and produces `FusedOp` instances that evaluate the fused
//! computation in a single pass using SIMD-friendly loops.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels::borrow::operand;
use crate::simd::into_vec_no_copy;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::marker::PhantomData;

use super::core::{ArrayExpr, Expr};

// ---------------------------------------------------------------------------
// FusedOp trait
// ---------------------------------------------------------------------------

/// Trait for fused operations that evaluate multiple combined operations in a single pass.
///
/// Implementations of this trait execute fused computation using SIMD-friendly
/// evaluation strategies. The `eval_fused` method materializes data through
/// batch processing with configurable chunk sizes.
pub trait FusedOp<T: Clone> {
    /// Evaluate the fused operation, producing a result array.
    fn eval_fused(&self) -> Array<T>;

    /// Return the number of elements in the result.
    fn fused_size(&self) -> usize;

    /// Return the shape of the result.
    fn fused_shape(&self) -> &[usize];
}

// ---------------------------------------------------------------------------
// Element-wise chain fusion: (a op1 b) op2 c
// ---------------------------------------------------------------------------

/// Fused element-wise ternary chain: `(a op1 b) op2 c` evaluated in a single pass.
///
/// This avoids allocating an intermediate array for `a op1 b` by computing
/// both operations at each index in one go.
///
/// # Example
///
/// ```rust,ignore
/// // (a + b) * c  -  one pass, no intermediate allocation
/// let fused = FusedElementWiseChain::new(
///     ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c),
///     |x, y| x + y, |r, z| r * z,
/// )?;
/// let result = fused.eval_fused();
/// ```
pub struct FusedElementWiseChain<T, A, B, C, F1, F2>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    a: A,
    b: B,
    c: C,
    op1: F1,
    op2: F2,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, A, B, C, F1, F2> FusedElementWiseChain<T, A, B, C, F1, F2>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    /// Create a new fused element-wise chain: `(a op1 b) op2 c`.
    ///
    /// Returns an error if the shapes of `a`, `b`, and `c` are not identical.
    pub fn new(a: A, b: B, c: C, op1: F1, op2: F2) -> Result<Self> {
        if a.shape() != b.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: b.shape().to_vec(),
            });
        }
        if a.shape() != c.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: c.shape().to_vec(),
            });
        }
        Ok(Self {
            shape: a.shape().to_vec(),
            a,
            b,
            c,
            op1,
            op2,
            _phantom: PhantomData,
        })
    }
}

impl<T, A, B, C, F1, F2> Expr<T> for FusedElementWiseChain<T, A, B, C, F1, F2>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let av = self.a.eval_at(index);
        let bv = self.b.eval_at(index);
        let cv = self.c.eval_at(index);
        (self.op2)((self.op1)(av, bv), cv)
    }

    #[inline]
    fn size(&self) -> usize {
        self.a.size()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, A, B, C, F1, F2> FusedOp<T> for FusedElementWiseChain<T, A, B, C, F1, F2>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
{
    fn eval_fused(&self) -> Array<T> {
        let size = self.a.size();
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(self.eval_at(i));
        }
        Array::from_vec_shape(data, &self.shape).unwrap_or_else(|e| panic!("{e}"))
    }

    fn fused_size(&self) -> usize {
        self.a.size()
    }

    fn fused_shape(&self) -> &[usize] {
        &self.shape
    }
}

// ---------------------------------------------------------------------------
// Scalar broadcast fusion: (a * scalar1) + scalar2  -->  FMA-like
// ---------------------------------------------------------------------------

/// Fused scalar broadcast operation: `a * mul_scalar + add_scalar`.
///
/// This corresponds to the pattern where an array is multiplied by one scalar
/// and then another scalar is added. Instead of two passes (multiply then add),
/// this computes the result in a single pass using an FMA-like pattern.
pub struct FusedScalarBroadcast<T, E>
where
    T: Clone,
    E: Expr<T>,
{
    expr: E,
    mul_scalar: T,
    add_scalar: T,
    fma_fn: fn(T, T, T) -> T,
}

impl<T, E> FusedScalarBroadcast<T, E>
where
    T: Clone,
    E: Expr<T>,
{
    /// Create a new fused scalar broadcast: `expr * mul_scalar + add_scalar`.
    ///
    /// The `fma_fn` should compute `a * b + c` for the given type.
    pub fn new(expr: E, mul_scalar: T, add_scalar: T, fma_fn: fn(T, T, T) -> T) -> Self {
        Self {
            expr,
            mul_scalar,
            add_scalar,
            fma_fn,
        }
    }
}

impl<T, E> Expr<T> for FusedScalarBroadcast<T, E>
where
    T: Clone,
    E: Expr<T>,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let val = self.expr.eval_at(index);
        (self.fma_fn)(val, self.mul_scalar.clone(), self.add_scalar.clone())
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

impl<T, E> FusedOp<T> for FusedScalarBroadcast<T, E>
where
    T: Clone,
    E: Expr<T>,
{
    fn eval_fused(&self) -> Array<T> {
        let size = self.expr.size();
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(self.eval_at(i));
        }
        Array::from_vec_shape(data, self.expr.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    fn fused_size(&self) -> usize {
        self.expr.size()
    }

    fn fused_shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

/// Create a fused scalar broadcast for f64 using hardware FMA when available.
pub fn fused_scalar_broadcast_f64<E: Expr<f64>>(
    expr: E,
    mul_scalar: f64,
    add_scalar: f64,
) -> FusedScalarBroadcast<f64, E> {
    FusedScalarBroadcast::new(expr, mul_scalar, add_scalar, |a, b, c| a.mul_add(b, c))
}

/// Create a fused scalar broadcast for f32 using hardware FMA when available.
pub fn fused_scalar_broadcast_f32<E: Expr<f32>>(
    expr: E,
    mul_scalar: f32,
    add_scalar: f32,
) -> FusedScalarBroadcast<f32, E> {
    FusedScalarBroadcast::new(expr, mul_scalar, add_scalar, |a, b, c| a.mul_add(b, c))
}

// ---------------------------------------------------------------------------
// Fused Multiply-Add (FMA): a * b + c
// ---------------------------------------------------------------------------

/// Fused multiply-add expression: `a * b + c` in a single pass.
///
/// This is the classic FMA pattern that many CPUs can execute in a single
/// instruction. When evaluating over arrays, this avoids the intermediate
/// allocation for `a * b`.
pub struct FusedMultiplyAdd<A, B, C>
where
    A: Expr<f64>,
    B: Expr<f64>,
    C: Expr<f64>,
{
    a: A,
    b: B,
    c: C,
    shape: Vec<usize>,
}

impl<A, B, C> FusedMultiplyAdd<A, B, C>
where
    A: Expr<f64>,
    B: Expr<f64>,
    C: Expr<f64>,
{
    /// Create a new fused multiply-add: `a * b + c`.
    ///
    /// Returns an error if the shapes are incompatible.
    pub fn new(a: A, b: B, c: C) -> Result<Self> {
        if a.shape() != b.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: b.shape().to_vec(),
            });
        }
        if a.shape() != c.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: c.shape().to_vec(),
            });
        }
        Ok(Self {
            shape: a.shape().to_vec(),
            a,
            b,
            c,
        })
    }
}

impl<A, B, C> Expr<f64> for FusedMultiplyAdd<A, B, C>
where
    A: Expr<f64>,
    B: Expr<f64>,
    C: Expr<f64>,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> f64 {
        let av = self.a.eval_at(index);
        let bv = self.b.eval_at(index);
        let cv = self.c.eval_at(index);
        av.mul_add(bv, cv)
    }

    #[inline]
    fn size(&self) -> usize {
        self.a.size()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Override eval() to use SIMD FMA when possible.
    fn eval(&self) -> Array<f64> {
        let size = self.a.size();
        // Threshold: use SIMD for larger arrays, scalar for small
        const SIMD_THRESHOLD: usize = 32;
        if size >= SIMD_THRESHOLD {
            self.eval_simd_fma()
        } else {
            self.eval_scalar_fma()
        }
    }
}

impl<A, B, C> FusedMultiplyAdd<A, B, C>
where
    A: Expr<f64>,
    B: Expr<f64>,
    C: Expr<f64>,
{
    /// SIMD-accelerated FMA evaluation using scirs2_core.
    fn eval_simd_fma(&self) -> Array<f64> {
        let a_arr = self.a.eval();
        let b_arr = self.b.eval();
        let c_arr = self.c.eval();

        // `as_cow_1d()` borrows each freshly-evaluated (and so always
        // uniquely-owned) operand zero-copy in the common contiguous case,
        // instead of `to_vec()`-ing it into a second buffer.
        let a_nd = a_arr.as_cow_1d();
        let b_nd = b_arr.as_cow_1d();
        let c_nd = c_arr.as_cow_1d();

        let result = f64::simd_fma(&a_nd.view(), &b_nd.view(), &c_nd.view());
        Array::from_vec_shape(into_vec_no_copy(result), &self.shape)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Scalar FMA evaluation using hardware FMA instruction per element.
    fn eval_scalar_fma(&self) -> Array<f64> {
        let size = self.a.size();
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            let av = self.a.eval_at(i);
            let bv = self.b.eval_at(i);
            let cv = self.c.eval_at(i);
            data.push(av.mul_add(bv, cv));
        }
        Array::from_vec_shape(data, &self.shape).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl<A, B, C> FusedOp<f64> for FusedMultiplyAdd<A, B, C>
where
    A: Expr<f64>,
    B: Expr<f64>,
    C: Expr<f64>,
{
    fn eval_fused(&self) -> Array<f64> {
        self.eval()
    }

    fn fused_size(&self) -> usize {
        self.a.size()
    }

    fn fused_shape(&self) -> &[usize] {
        &self.shape
    }
}

// ---------------------------------------------------------------------------
// Reduction fusion: fused element-wise + reduction
// ---------------------------------------------------------------------------

/// Fused reduction over an element-wise binary operation.
///
/// Instead of materializing the element-wise result into an intermediate
/// array and then reducing it, this computes and accumulates in a single pass.
///
/// Classic example: `sum(a * b)` becomes a dot-product-like computation.
pub struct FusedReduction<T, A, B, ElemOp, RedOp, Identity>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    ElemOp: Fn(T, T) -> T,
    RedOp: Fn(T, T) -> T,
    Identity: Fn() -> T,
{
    a: A,
    b: B,
    elem_op: ElemOp,
    reduce_op: RedOp,
    identity: Identity,
    _phantom: PhantomData<T>,
}

impl<T, A, B, ElemOp, RedOp, Identity> FusedReduction<T, A, B, ElemOp, RedOp, Identity>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    ElemOp: Fn(T, T) -> T,
    RedOp: Fn(T, T) -> T,
    Identity: Fn() -> T,
{
    /// Create a new fused reduction: `reduce(a elem_op b)`.
    ///
    /// Returns an error if the shapes of `a` and `b` don't match.
    pub fn new(a: A, b: B, elem_op: ElemOp, reduce_op: RedOp, identity: Identity) -> Result<Self> {
        if a.shape() != b.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: b.shape().to_vec(),
            });
        }
        Ok(Self {
            a,
            b,
            elem_op,
            reduce_op,
            identity,
            _phantom: PhantomData,
        })
    }

    /// Execute the fused reduction, returning a scalar result.
    ///
    /// This processes elements one at a time, accumulating using the
    /// reduction operation, without allocating an intermediate array.
    pub fn reduce(&self) -> T {
        let size = self.a.size();
        if size == 0 {
            return (self.identity)();
        }

        let first_a = self.a.eval_at(0);
        let first_b = self.b.eval_at(0);
        let mut acc = (self.elem_op)(first_a, first_b);

        for i in 1..size {
            let av = self.a.eval_at(i);
            let bv = self.b.eval_at(i);
            let elem_result = (self.elem_op)(av, bv);
            acc = (self.reduce_op)(acc, elem_result);
        }

        acc
    }
}

/// Create a fused dot product: `sum(a * b)` computed in a single pass.
pub fn fused_dot_product<'a>(a: ArrayExpr<'a, f64>, b: ArrayExpr<'a, f64>) -> Result<f64> {
    let fused = FusedReduction::new(a, b, |x, y| x * y, |acc, v| acc + v, || 0.0)?;
    Ok(fused.reduce())
}

/// Create a fused sum of squares: `sum(a * a)` computed in a single pass.
pub fn fused_sum_of_squares(a: &Array<f64>) -> f64 {
    let size = a.size();
    if size == 0 {
        return 0.0;
    }
    let expr = ArrayExpr::new(a);
    let fused = FusedReduction::new(
        ArrayExpr::new(a),
        expr,
        |x, y| x * y,
        |acc, v| acc + v,
        || 0.0,
    );
    match fused {
        Ok(f) => f.reduce(),
        Err(_) => 0.0,
    }
}

/// Create a fused sum-of-absolute-differences: `sum(|a - b|)`.
pub fn fused_sum_abs_diff<'a>(a: ArrayExpr<'a, f64>, b: ArrayExpr<'a, f64>) -> Result<f64> {
    let fused = FusedReduction::new(a, b, |x, y| (x - y).abs(), |acc, v| acc + v, || 0.0)?;
    Ok(fused.reduce())
}

// ---------------------------------------------------------------------------
// SIMD FMA evaluation for f64 arrays (direct, without expression trees)
// ---------------------------------------------------------------------------

/// Evaluate `a * b + c` element-wise using SIMD FMA, operating directly on arrays.
///
/// This is the highest-performance path for the FMA pattern on raw arrays.
/// It bypasses the expression tree entirely and uses scirs2_core SIMD directly.
pub fn simd_fma_arrays(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    if a.shape() != c.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: c.shape(),
        });
    }

    let a_nd = a.as_cow_1d();
    let b_nd = b.as_cow_1d();
    let c_nd = c.as_cow_1d();

    let result = f64::simd_fma(&a_nd.view(), &b_nd.view(), &c_nd.view());
    Array::from_vec_shape(into_vec_no_copy(result), &a.shape())
}

/// Evaluate `a * scalar_mul + scalar_add` using SIMD, operating directly on an array.
pub fn simd_fused_scalar_broadcast(a: &Array<f64>, scalar_mul: f64, scalar_add: f64) -> Array<f64> {
    let data = operand(a);
    let size = data.len();
    let mut result = Vec::with_capacity(size);

    // Process in chunks for better auto-vectorization
    const CHUNK: usize = 8;
    let full_chunks = size / CHUNK;

    for chunk_idx in 0..full_chunks {
        let base = chunk_idx * CHUNK;
        for j in 0..CHUNK {
            result.push(data[base + j].mul_add(scalar_mul, scalar_add));
        }
    }

    // Handle remainder
    for i in (full_chunks * CHUNK)..size {
        result.push(data[i].mul_add(scalar_mul, scalar_add));
    }

    Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
}

/// Compute a fused dot product directly on arrays using SIMD.
pub fn simd_fused_dot_product(a: &Array<f64>, b: &Array<f64>) -> Result<f64> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    let a_nd = a.as_cow_1d();
    let b_nd = b.as_cow_1d();

    Ok(f64::simd_dot(&a_nd.view(), &b_nd.view()))
}

// ---------------------------------------------------------------------------
// Fusion detector
// ---------------------------------------------------------------------------

/// Describes detected fusible patterns in an expression tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionPattern {
    /// Element-wise chain: `(a op1 b) op2 c`
    ElementWiseChain,
    /// Scalar broadcast: `a * scalar1 + scalar2`
    ScalarBroadcast,
    /// Fused multiply-add: `a * b + c`
    FusedMultiplyAdd,
    /// Reduction over element-wise: `reduce(a op b)`
    ReductionFusion,
    /// No fusion opportunity detected
    None,
}

/// Result of fusion analysis on an expression tree.
#[derive(Debug, Clone)]
pub struct FusionAnalysis {
    /// The detected pattern.
    pub pattern: FusionPattern,
    /// Estimated speedup factor (1.0 = no improvement).
    pub estimated_speedup: f64,
    /// Number of intermediate allocations eliminated.
    pub allocations_eliminated: usize,
    /// Number of passes over data eliminated.
    pub passes_eliminated: usize,
}

impl FusionAnalysis {
    /// Create a "no fusion" analysis.
    pub fn none() -> Self {
        Self {
            pattern: FusionPattern::None,
            estimated_speedup: 1.0,
            allocations_eliminated: 0,
            passes_eliminated: 0,
        }
    }

    /// Create a fusion analysis for an element-wise chain.
    pub fn element_wise_chain() -> Self {
        Self {
            pattern: FusionPattern::ElementWiseChain,
            estimated_speedup: 1.5,
            allocations_eliminated: 1,
            passes_eliminated: 1,
        }
    }

    /// Create a fusion analysis for scalar broadcast fusion.
    pub fn scalar_broadcast() -> Self {
        Self {
            pattern: FusionPattern::ScalarBroadcast,
            estimated_speedup: 1.8,
            allocations_eliminated: 1,
            passes_eliminated: 1,
        }
    }

    /// Create a fusion analysis for FMA.
    pub fn fma() -> Self {
        Self {
            pattern: FusionPattern::FusedMultiplyAdd,
            estimated_speedup: 2.0,
            allocations_eliminated: 1,
            passes_eliminated: 1,
        }
    }

    /// Create a fusion analysis for reduction fusion.
    pub fn reduction() -> Self {
        Self {
            pattern: FusionPattern::ReductionFusion,
            estimated_speedup: 2.5,
            allocations_eliminated: 1,
            passes_eliminated: 1,
        }
    }
}

/// Fusion detector that analyzes expression tree structure.
///
/// The detector inspects the topology of expression trees to identify
/// patterns that can be replaced with fused implementations.
pub struct FusionDetector;

impl FusionDetector {
    /// Analyze whether two chained binary operations can be fused.
    ///
    /// Returns `FusionPattern::ElementWiseChain` if `expr1` and `expr2` are both
    /// element-wise binary operations that can be merged into a single pass.
    pub fn detect_element_wise_chain(size_outer: usize, size_inner: usize) -> FusionAnalysis {
        if size_outer == size_inner && size_inner > 0 {
            FusionAnalysis::element_wise_chain()
        } else {
            FusionAnalysis::none()
        }
    }

    /// Detect if a scalar-multiply followed by scalar-add can be fused.
    ///
    /// This catches the common pattern `a * k1 + k2` which maps to FMA.
    pub fn detect_scalar_broadcast(has_mul: bool, has_add: bool) -> FusionAnalysis {
        if has_mul && has_add {
            FusionAnalysis::scalar_broadcast()
        } else {
            FusionAnalysis::none()
        }
    }

    /// Detect if a multiply followed by sum-reduction can become a dot product.
    pub fn detect_reduction_fusion(is_element_wise: bool, is_reduction: bool) -> FusionAnalysis {
        if is_element_wise && is_reduction {
            FusionAnalysis::reduction()
        } else {
            FusionAnalysis::none()
        }
    }

    /// Detect if three operands form an FMA pattern: `a * b + c`.
    pub fn detect_fma(has_multiply: bool, has_add: bool, sizes_match: bool) -> FusionAnalysis {
        if has_multiply && has_add && sizes_match {
            FusionAnalysis::fma()
        } else {
            FusionAnalysis::none()
        }
    }

    /// Comprehensive analysis: given three arrays, determine the best fusion.
    pub fn analyze_ternary(a_size: usize, b_size: usize, c_size: usize) -> FusionAnalysis {
        if a_size == b_size && b_size == c_size && a_size > 0 {
            FusionAnalysis::fma()
        } else if a_size == b_size && a_size > 0 {
            FusionAnalysis::element_wise_chain()
        } else {
            FusionAnalysis::none()
        }
    }
}

// ---------------------------------------------------------------------------
// Fused quad-op: (a op1 b) op2 (c op3 d)
// ---------------------------------------------------------------------------

/// Fused quaternary expression: `(a op1 b) op2 (c op3 d)` in a single pass.
///
/// Eliminates two intermediate allocations compared to the naive approach.
pub struct FusedQuadOp<T, A, B, C, D, F1, F2, F3>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    D: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
    F3: Fn(T, T) -> T,
{
    a: A,
    b: B,
    c: C,
    d: D,
    op1: F1,
    op2: F2,
    op3: F3,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, A, B, C, D, F1, F2, F3> FusedQuadOp<T, A, B, C, D, F1, F2, F3>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    D: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
    F3: Fn(T, T) -> T,
{
    /// Create a new fused quad-op: `(a op1 b) op2 (c op3 d)`.
    pub fn new(a: A, b: B, c: C, d: D, op1: F1, op2: F2, op3: F3) -> Result<Self> {
        let shape = a.shape().to_vec();
        if shape != b.shape() || shape != c.shape() || shape != d.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape,
                actual: b.shape().to_vec(),
            });
        }
        Ok(Self {
            shape: a.shape().to_vec(),
            a,
            b,
            c,
            d,
            op1,
            op2,
            op3,
            _phantom: PhantomData,
        })
    }
}

impl<T, A, B, C, D, F1, F2, F3> Expr<T> for FusedQuadOp<T, A, B, C, D, F1, F2, F3>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    D: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
    F3: Fn(T, T) -> T,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        let lhs = (self.op1)(self.a.eval_at(index), self.b.eval_at(index));
        let rhs = (self.op3)(self.c.eval_at(index), self.d.eval_at(index));
        (self.op2)(lhs, rhs)
    }

    #[inline]
    fn size(&self) -> usize {
        self.a.size()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, A, B, C, D, F1, F2, F3> FusedOp<T> for FusedQuadOp<T, A, B, C, D, F1, F2, F3>
where
    T: Clone,
    A: Expr<T>,
    B: Expr<T>,
    C: Expr<T>,
    D: Expr<T>,
    F1: Fn(T, T) -> T,
    F2: Fn(T, T) -> T,
    F3: Fn(T, T) -> T,
{
    fn eval_fused(&self) -> Array<T> {
        self.eval()
    }

    fn fused_size(&self) -> usize {
        self.a.size()
    }

    fn fused_shape(&self) -> &[usize] {
        &self.shape
    }
}

// ---------------------------------------------------------------------------
// Fused unary chain: op2(op1(a))
// ---------------------------------------------------------------------------

/// Fused unary chain: `op2(op1(a))` in a single pass.
///
/// Avoids materializing the intermediate result of `op1(a)`.
pub struct FusedUnaryChain<T, E, F1, F2>
where
    T: Clone,
    E: Expr<T>,
    F1: Fn(T) -> T,
    F2: Fn(T) -> T,
{
    expr: E,
    op1: F1,
    op2: F2,
    _phantom: PhantomData<T>,
}

impl<T, E, F1, F2> FusedUnaryChain<T, E, F1, F2>
where
    T: Clone,
    E: Expr<T>,
    F1: Fn(T) -> T,
    F2: Fn(T) -> T,
{
    /// Create a new fused unary chain: `op2(op1(expr))`.
    pub fn new(expr: E, op1: F1, op2: F2) -> Self {
        Self {
            expr,
            op1,
            op2,
            _phantom: PhantomData,
        }
    }
}

impl<T, E, F1, F2> Expr<T> for FusedUnaryChain<T, E, F1, F2>
where
    T: Clone,
    E: Expr<T>,
    F1: Fn(T) -> T,
    F2: Fn(T) -> T,
{
    #[inline(always)]
    fn eval_at(&self, index: usize) -> T {
        (self.op2)((self.op1)(self.expr.eval_at(index)))
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

// ---------------------------------------------------------------------------
// FusionBuilder - Fluent API for constructing fused expressions
// ---------------------------------------------------------------------------

/// Builder for constructing fused expressions using a fluent API.
///
/// This builder detects fusible patterns and automatically applies
/// the appropriate fusion strategies.
///
/// # Example
///
/// ```rust,ignore
/// let result = FusionBuilder::from_array(&a)
///     .add_expr(&b)
///     .mul_expr(&c)
///     .eval_fused();
/// ```
pub struct FusionBuilder<'a> {
    /// Current accumulated data (materialized for fusion).
    data: Vec<f64>,
    /// Shape of the current expression.
    shape: Vec<usize>,
    /// Tracks the number of fusions applied.
    fusions_applied: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> FusionBuilder<'a> {
    /// Start from an array.
    pub fn from_array(array: &'a Array<f64>) -> Self {
        Self {
            data: array.to_vec(),
            shape: array.shape(),
            fusions_applied: 0,
            _lifetime: PhantomData,
        }
    }

    /// Element-wise add with another array (fused into the current pass).
    pub fn add_expr(mut self, other: &Array<f64>) -> Self {
        // `other` is only ever read here, never mutated -- `operand()`
        // borrows it zero-copy for the common contiguous case instead of
        // `to_vec()`-ing it into an owned buffer just to iterate it once.
        let other_operand = operand(other);
        let other_data: &[f64] = &other_operand;
        let len = self.data.len().min(other_data.len());
        for i in 0..len {
            self.data[i] += other_data[i];
        }
        self.fusions_applied += 1;
        self
    }

    /// Element-wise multiply with another array (fused).
    pub fn mul_expr(mut self, other: &Array<f64>) -> Self {
        // See `add_expr`: read-only access, so borrow rather than copy.
        let other_operand = operand(other);
        let other_data: &[f64] = &other_operand;
        let len = self.data.len().min(other_data.len());
        for i in 0..len {
            self.data[i] *= other_data[i];
        }
        self.fusions_applied += 1;
        self
    }

    /// Element-wise subtract another array (fused).
    pub fn sub_expr(mut self, other: &Array<f64>) -> Self {
        // See `add_expr`: read-only access, so borrow rather than copy.
        let other_operand = operand(other);
        let other_data: &[f64] = &other_operand;
        let len = self.data.len().min(other_data.len());
        for i in 0..len {
            self.data[i] -= other_data[i];
        }
        self.fusions_applied += 1;
        self
    }

    /// Add scalar (broadcast, fused).
    pub fn add_scalar(mut self, scalar: f64) -> Self {
        for v in &mut self.data {
            *v += scalar;
        }
        self.fusions_applied += 1;
        self
    }

    /// Multiply by scalar (broadcast, fused).
    pub fn mul_scalar(mut self, scalar: f64) -> Self {
        for v in &mut self.data {
            *v *= scalar;
        }
        self.fusions_applied += 1;
        self
    }

    /// Fused multiply-add with scalar: `self * mul + add`.
    pub fn fma_scalar(mut self, mul: f64, add: f64) -> Self {
        for v in &mut self.data {
            *v = v.mul_add(mul, add);
        }
        self.fusions_applied += 1;
        self
    }

    /// Apply a unary function (fused).
    pub fn map<F: Fn(f64) -> f64>(mut self, op: F) -> Self {
        for v in &mut self.data {
            *v = op(*v);
        }
        self.fusions_applied += 1;
        self
    }

    /// Reduce via summation.
    pub fn sum(self) -> f64 {
        self.data.iter().sum()
    }

    /// Reduce via product.
    pub fn product(self) -> f64 {
        self.data.iter().product()
    }

    /// Materialize the fused result.
    pub fn eval_fused(self) -> Array<f64> {
        Array::from_vec_shape(self.data, &self.shape).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Return the number of fusions that were applied.
    pub fn fusions_applied(&self) -> usize {
        self.fusions_applied
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::expr::core::{ArrayExpr, BinaryExpr, Expr};
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // 1. Element-wise chain fusion: (a + b) * c
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_element_wise_chain_add_mul() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let c = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let fused = FusedElementWiseChain::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            ArrayExpr::new(&c),
            |x, y| x + y,
            |r, z| r * z,
        )
        .expect("Fused chain creation should succeed");

        let result = fused.eval_fused();
        // (1+10)*2=22, (2+20)*3=66, (3+30)*4=132, (4+40)*5=220
        assert_eq!(result.to_vec(), vec![22.0, 66.0, 132.0, 220.0]);
    }

    #[test]
    fn test_fused_element_wise_chain_sub_mul() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let c = Array::from_vec(vec![3.0, 3.0, 3.0, 3.0]);

        let fused = FusedElementWiseChain::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            ArrayExpr::new(&c),
            |x, y| x - y,
            |r, z| r * z,
        )
        .expect("Fused chain creation should succeed");

        let result = fused.eval_fused();
        // (10-1)*3=27, (20-2)*3=54, (30-3)*3=81, (40-4)*3=108
        assert_eq!(result.to_vec(), vec![27.0, 54.0, 81.0, 108.0]);
    }

    // -----------------------------------------------------------------------
    // 2. Scalar broadcast fusion: a * 2.0 + 3.0
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_scalar_broadcast_mul_add() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let fused = fused_scalar_broadcast_f64(ArrayExpr::new(&a), 2.0, 3.0);

        let result = fused.eval_fused();
        // 1*2+3=5, 2*2+3=7, 3*2+3=9, 4*2+3=11
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn test_fused_scalar_broadcast_f32() {
        let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

        let fused = fused_scalar_broadcast_f32(ArrayExpr::new(&a), 3.0, 1.0);

        let result = fused.eval_fused();
        // 1*3+1=4, 2*3+1=7, 3*3+1=10, 4*3+1=13
        assert_eq!(result.to_vec(), vec![4.0f32, 7.0, 10.0, 13.0]);
    }

    // -----------------------------------------------------------------------
    // 3. FMA detection and fusion
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_multiply_add_basic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let c = Array::from_vec(vec![10.0, 10.0, 10.0, 10.0]);

        let fma = FusedMultiplyAdd::new(ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c))
            .expect("FMA creation should succeed");

        let result = fma.eval_fused();
        // 1*2+10=12, 2*3+10=16, 3*4+10=22, 4*5+10=30
        assert_eq!(result.to_vec(), vec![12.0, 16.0, 22.0, 30.0]);
    }

    #[test]
    fn test_fused_multiply_add_simd_path() {
        // Create a larger array to trigger SIMD path (>= 32 elements)
        let size = 64;
        let a_data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let b_data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
        let c_data: Vec<f64> = vec![100.0; size];

        let a = Array::from_vec(a_data.clone());
        let b = Array::from_vec(b_data.clone());
        let c = Array::from_vec(c_data.clone());

        let fma = FusedMultiplyAdd::new(ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c))
            .expect("FMA creation should succeed");

        let result = fma.eval();
        let result_data = result.to_vec();

        for i in 0..size {
            let expected = a_data[i].mul_add(b_data[i], c_data[i]);
            assert_relative_eq!(result_data[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fma_shape_mismatch() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let c = Array::from_vec(vec![1.0, 2.0, 3.0]);

        let result =
            FusedMultiplyAdd::new(ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c));

        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 4. Reduction fusion: sum(a * b)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_dot_product() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let dot = fused_dot_product(ArrayExpr::new(&a), ArrayExpr::new(&b))
            .expect("Fused dot product should succeed");

        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_relative_eq!(dot, 40.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fused_sum_of_squares() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let sos = fused_sum_of_squares(&a);

        // 1 + 4 + 9 + 16 = 30
        assert_relative_eq!(sos, 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fused_sum_abs_diff() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0, 10.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 7.0, 8.0]);

        let sad = fused_sum_abs_diff(ArrayExpr::new(&a), ArrayExpr::new(&b))
            .expect("Fused sum-abs-diff should succeed");

        // |1-2| + |5-3| + |3-7| + |10-8| = 1 + 2 + 4 + 2 = 9
        assert_relative_eq!(sad, 9.0, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // 5. Non-fusible expressions (preserve correctness)
    // -----------------------------------------------------------------------

    #[test]
    fn test_shape_mismatch_chain() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let c = Array::from_vec(vec![1.0, 2.0, 3.0]);

        let result = FusedElementWiseChain::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            ArrayExpr::new(&c),
            |x, y| x + y,
            |r, z| r * z,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_non_fusible_preserves_correctness() {
        // Even when we can't fuse, the non-fused path should be correct.
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        // Standard (non-fused) approach
        let add_expr = BinaryExpr::new(ArrayExpr::new(&a), ArrayExpr::new(&b), |x: f64, y: f64| {
            x + y
        })
        .expect("Binary expr should succeed");

        let result = add_expr.eval();
        assert_eq!(result.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    // -----------------------------------------------------------------------
    // 6. SIMD alignment handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_fma_arrays_alignment() {
        // Test with sizes that are not multiples of SIMD width
        let sizes = [1, 3, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257];

        for &size in &sizes {
            let a_data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let b_data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
            let c_data: Vec<f64> = vec![1.0; size];

            let a = Array::from_vec(a_data.clone());
            let b = Array::from_vec(b_data.clone());
            let c = Array::from_vec(c_data.clone());

            let result =
                simd_fma_arrays(&a, &b, &c).expect("SIMD FMA should succeed for all sizes");
            let result_data = result.to_vec();

            for i in 0..size {
                let expected = a_data[i].mul_add(b_data[i], c_data[i]);
                assert_relative_eq!(result_data[i], expected, epsilon = 1e-10,);
            }
        }
    }

    #[test]
    fn test_simd_fused_scalar_broadcast_alignment() {
        for size in [1, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65] {
            let data: Vec<f64> = (0..size).map(|i| i as f64 + 1.0).collect();
            let a = Array::from_vec(data.clone());

            let result = simd_fused_scalar_broadcast(&a, 2.0, 3.0);
            let result_data = result.to_vec();

            for i in 0..size {
                let expected = data[i].mul_add(2.0, 3.0);
                assert_relative_eq!(result_data[i], expected, epsilon = 1e-10);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Performance validation (fused should produce same results)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fusion_vs_unfused_correctness() {
        let n = 1000;
        let a_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let b_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.2 + 1.0).collect();
        let c_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.05).collect();

        let a = Array::from_vec(a_data.clone());
        let b = Array::from_vec(b_data.clone());
        let c = Array::from_vec(c_data.clone());

        // Unfused: step1 = a + b, step2 = step1 * c
        let unfused: Vec<f64> = (0..n)
            .map(|i| (a_data[i] + b_data[i]) * c_data[i])
            .collect();

        // Fused
        let fused = FusedElementWiseChain::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            ArrayExpr::new(&c),
            |x, y| x + y,
            |r, z| r * z,
        )
        .expect("Fused creation should succeed");

        let fused_result = fused.eval_fused().to_vec();

        for i in 0..n {
            assert_relative_eq!(fused_result[i], unfused[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fma_vs_manual_correctness() {
        let n = 500;
        let a_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3).collect();
        let b_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.7 + 0.5).collect();
        let c_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 + 2.0).collect();

        let a = Array::from_vec(a_data.clone());
        let b = Array::from_vec(b_data.clone());
        let c = Array::from_vec(c_data.clone());

        // Manual: a * b + c
        let manual: Vec<f64> = (0..n).map(|i| a_data[i] * b_data[i] + c_data[i]).collect();

        // FMA (uses hardware FMA instruction via mul_add)
        let fma = FusedMultiplyAdd::new(ArrayExpr::new(&a), ArrayExpr::new(&b), ArrayExpr::new(&c))
            .expect("FMA creation should succeed");

        let fma_result = fma.eval().to_vec();

        for i in 0..n {
            // FMA may produce slightly more precise results than manual
            assert_relative_eq!(fma_result[i], manual[i], epsilon = 1e-8);
        }
    }

    // -----------------------------------------------------------------------
    // 8. Mixed operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_fused_ops() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
        let c = Array::from_vec(vec![10.0, 10.0, 10.0, 10.0]);
        let d = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0]);

        // (a + b) * (c - d) in one fused pass
        let quad = FusedQuadOp::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            ArrayExpr::new(&c),
            ArrayExpr::new(&d),
            |x, y| x + y, // op1: a + b
            |l, r| l * r, // op2: lhs * rhs
            |x, y| x - y, // op3: c - d
        )
        .expect("Quad op creation should succeed");

        let result = quad.eval_fused();
        // (1+5)*(10-2)=48, (2+6)*(10-2)=64, (3+7)*(10-2)=80, (4+8)*(10-2)=96
        assert_eq!(result.to_vec(), vec![48.0, 64.0, 80.0, 96.0]);
    }

    #[test]
    fn test_fused_unary_chain() {
        let a = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0]);

        // sqrt then double
        let chain = FusedUnaryChain::new(ArrayExpr::new(&a), |x: f64| x.sqrt(), |x: f64| x * 2.0);

        let result = chain.eval();
        // sqrt(1)*2=2, sqrt(4)*2=4, sqrt(9)*2=6, sqrt(16)*2=8
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // 9. Fusion detector
    // -----------------------------------------------------------------------

    #[test]
    fn test_fusion_detector_element_wise() {
        let analysis = FusionDetector::detect_element_wise_chain(100, 100);
        assert_eq!(analysis.pattern, FusionPattern::ElementWiseChain);
        assert!(analysis.estimated_speedup > 1.0);
        assert_eq!(analysis.allocations_eliminated, 1);
    }

    #[test]
    fn test_fusion_detector_scalar_broadcast() {
        let analysis = FusionDetector::detect_scalar_broadcast(true, true);
        assert_eq!(analysis.pattern, FusionPattern::ScalarBroadcast);

        let no_fusion = FusionDetector::detect_scalar_broadcast(true, false);
        assert_eq!(no_fusion.pattern, FusionPattern::None);
    }

    #[test]
    fn test_fusion_detector_reduction() {
        let analysis = FusionDetector::detect_reduction_fusion(true, true);
        assert_eq!(analysis.pattern, FusionPattern::ReductionFusion);

        let no_fusion = FusionDetector::detect_reduction_fusion(false, true);
        assert_eq!(no_fusion.pattern, FusionPattern::None);
    }

    #[test]
    fn test_fusion_detector_fma() {
        let analysis = FusionDetector::detect_fma(true, true, true);
        assert_eq!(analysis.pattern, FusionPattern::FusedMultiplyAdd);

        let no_fusion = FusionDetector::detect_fma(true, false, true);
        assert_eq!(no_fusion.pattern, FusionPattern::None);
    }

    #[test]
    fn test_fusion_detector_ternary_analysis() {
        let fma = FusionDetector::analyze_ternary(100, 100, 100);
        assert_eq!(fma.pattern, FusionPattern::FusedMultiplyAdd);

        let chain = FusionDetector::analyze_ternary(100, 100, 50);
        assert_eq!(chain.pattern, FusionPattern::ElementWiseChain);

        let none = FusionDetector::analyze_ternary(100, 50, 50);
        assert_eq!(none.pattern, FusionPattern::None);
    }

    // -----------------------------------------------------------------------
    // 10. FusionBuilder (fluent API)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fusion_builder_add_mul() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let c = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0]);

        let result = FusionBuilder::from_array(&a)
            .add_expr(&b)
            .mul_expr(&c)
            .eval_fused();

        // (1+10)*2=22, (2+20)*2=44, (3+30)*2=66, (4+40)*2=88
        assert_eq!(result.to_vec(), vec![22.0, 44.0, 66.0, 88.0]);
    }

    #[test]
    fn test_fusion_builder_scalar_fma() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = FusionBuilder::from_array(&a)
            .fma_scalar(2.0, 3.0)
            .eval_fused();

        // 1*2+3=5, 2*2+3=7, 3*2+3=9, 4*2+3=11
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn test_fusion_builder_complex_chain() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0]);

        let result = FusionBuilder::from_array(&a)
            .mul_scalar(3.0) // [3, 6, 9, 12]
            .add_expr(&b) // [5, 8, 11, 14]
            .add_scalar(1.0) // [6, 9, 12, 15]
            .eval_fused();

        assert_eq!(result.to_vec(), vec![6.0, 9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_fusion_builder_fusions_count() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);

        let builder = FusionBuilder::from_array(&a)
            .mul_scalar(2.0)
            .add_scalar(1.0)
            .map(|x| x * x);

        assert_eq!(builder.fusions_applied(), 3);
    }

    #[test]
    fn test_fusion_builder_sum_reduction() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        // sum(a * b) = fused dot product via builder
        let dot = FusionBuilder::from_array(&a).mul_expr(&b).sum();

        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_relative_eq!(dot, 40.0, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // W3-B2 perf probe: FusionBuilder de-copy fix (this task's actual change).
    // #[ignore]d -- run explicitly in release:
    //   CARGO_INCREMENTAL=0 cargo test --release -p numrs2 \
    //     expr::fusion::tests::probe_ -- --ignored --nocapture
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn probe_fusion_builder_mul_expr_copy_elimination_perf() {
        // Honest A/B, both sides real code (not reconstructed from prose):
        // `old_mul_in_place` is the exact body `mul_expr` had before this
        // fix (`other.to_vec()` purely to read `other` once, never
        // mutating it); `new_mul_in_place` is literally what `mul_expr`
        // does today (`operand(other)`, zero-copy in the common contiguous
        // case). Min-over-many-alternating-samples, per project convention,
        // to stay robust on a possibly noisy/shared machine.
        fn old_mul_in_place(data: &mut [f64], other: &Array<f64>) {
            let other_data = other.to_vec();
            let len = data.len().min(other_data.len());
            for i in 0..len {
                data[i] *= other_data[i];
            }
        }

        fn new_mul_in_place(data: &mut [f64], other: &Array<f64>) {
            let other_operand = operand(other);
            let other_data: &[f64] = &other_operand;
            let len = data.len().min(other_data.len());
            for i in 0..len {
                data[i] *= other_data[i];
            }
        }

        let n = 200_000;
        // Values close to 1 so many rounds of repeated in-place
        // multiplication (across samples) don't drift toward 0/inf --
        // each round resets `data` from `base` before the timed section.
        let base: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1e-9).collect();
        let other = Array::from_vec((0..n).map(|i| 1.0 - (i as f64) * 1e-10).collect());

        const SAMPLES: usize = 200;
        let mut data = base.clone();
        let mut old_min = std::time::Duration::MAX;
        let mut new_min = std::time::Duration::MAX;

        for _ in 0..SAMPLES {
            data.copy_from_slice(&base);
            let t0 = std::time::Instant::now();
            old_mul_in_place(
                std::hint::black_box(&mut data),
                std::hint::black_box(&other),
            );
            old_min = old_min.min(t0.elapsed());

            data.copy_from_slice(&base);
            let t1 = std::time::Instant::now();
            new_mul_in_place(
                std::hint::black_box(&mut data),
                std::hint::black_box(&other),
            );
            new_min = new_min.min(t1.elapsed());
        }

        let ratio = old_min.as_secs_f64() / new_min.as_secs_f64();
        eprintln!(
            "[FusionBuilder::mul_expr copy elimination, n={n}] old(min)={:.2}us new(min)={:.2}us ({:.2}x)",
            old_min.as_secs_f64() * 1e6,
            new_min.as_secs_f64() * 1e6,
            ratio,
        );

        // Same inputs -> bit-identical outputs, not just faster.
        let mut a = base.clone();
        old_mul_in_place(&mut a, &other);
        let mut b = base.clone();
        new_mul_in_place(&mut b, &other);
        assert_eq!(a, b);

        assert!(
            ratio >= 1.0,
            "expected the zero-copy borrow to be no slower than the old to_vec() copy, got {ratio:.2}x"
        );
    }

    // -----------------------------------------------------------------------
    // 11. SIMD direct array functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_fma_arrays_basic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let c = Array::from_vec(vec![10.0, 10.0, 10.0, 10.0]);

        let result = simd_fma_arrays(&a, &b, &c).expect("SIMD FMA should succeed");

        // 1*2+10=12, 2*3+10=16, 3*4+10=22, 4*5+10=30
        assert_eq!(result.to_vec(), vec![12.0, 16.0, 22.0, 30.0]);
    }

    #[test]
    fn test_simd_fused_dot_product() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let dot = simd_fused_dot_product(&a, &b).expect("SIMD dot product should succeed");

        assert_relative_eq!(dot, 40.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_fused_dot_product_shape_mismatch() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0]);

        let result = simd_fused_dot_product(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // W3-B2 perf probe (secondary): `simd_fma_arrays` was ALREADY fixed by
    // Wave 2 before this task started (see `as_cow_1d`/`into_vec_no_copy`
    // usage above) -- this task changed nothing here. `old_simd_fma_arrays`
    // below is a byte-for-byte reconstruction of the actual pre-Wave-2 body
    // of this function (recovered from `git diff HEAD`'s
    // `-let a_nd = Array1::from_vec(a_data);` / `-Array::from_vec_shape(
    // result.to_vec(), ...)` hunk, not guessed from the ticket's prose):
    // 3 `to_vec()` input copies + a `result.to_vec()` output detour, 4
    // buffers total vs. today's 0 (contiguous case). This measures that
    // real 4-copy-vs-0-copy delta so the win claimed for this file's
    // heaviest-by-copy-count SIMD helper (vs. `SimdBinaryEvaluator`'s 3) is
    // actually measured, not merely asserted.
    // #[ignore]d -- run explicitly in release, same as the probe above.
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn probe_simd_fma_arrays_copy_elimination_perf() {
        fn old_simd_fma_arrays(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
            let a_data = a.to_vec();
            let b_data = b.to_vec();
            let c_data = c.to_vec();

            let a_nd = scirs2_core::ndarray::Array1::from_vec(a_data);
            let b_nd = scirs2_core::ndarray::Array1::from_vec(b_data);
            let c_nd = scirs2_core::ndarray::Array1::from_vec(c_data);

            let result = f64::simd_fma(&a_nd.view(), &b_nd.view(), &c_nd.view());
            Array::from_vec_shape(result.to_vec(), &a.shape())
                .expect("shape always matches for equal-shape inputs")
        }

        let n = 200_000;
        let a = Array::from_vec((0..n).map(|i| i as f64).collect::<Vec<_>>());
        let b = Array::from_vec((0..n).map(|i| (i as f64) * 0.5).collect::<Vec<_>>());
        let c = Array::from_vec((0..n).map(|i| (i as f64) * 0.25).collect::<Vec<_>>());

        const SAMPLES: usize = 200;
        let mut old_min = std::time::Duration::MAX;
        let mut new_min = std::time::Duration::MAX;

        for _ in 0..SAMPLES {
            let t0 = std::time::Instant::now();
            let r_old = old_simd_fma_arrays(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                std::hint::black_box(&c),
            );
            old_min = old_min.min(t0.elapsed());
            std::hint::black_box(&r_old);

            let t1 = std::time::Instant::now();
            let r_new = simd_fma_arrays(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                std::hint::black_box(&c),
            )
            .expect("simd_fma_arrays should succeed");
            new_min = new_min.min(t1.elapsed());
            std::hint::black_box(&r_new);
        }

        let ratio = old_min.as_secs_f64() / new_min.as_secs_f64();
        eprintln!(
            "[simd_fma_arrays copy elimination (Wave 2 fix, verified here), n={n}] old(min)={:.2}us new(min)={:.2}us ({:.2}x)",
            old_min.as_secs_f64() * 1e6,
            new_min.as_secs_f64() * 1e6,
            ratio,
        );

        assert_eq!(
            old_simd_fma_arrays(&a, &b, &c).to_vec(),
            simd_fma_arrays(&a, &b, &c)
                .expect("simd_fma_arrays should succeed")
                .to_vec()
        );
    }

    // -----------------------------------------------------------------------
    // 12. Edge cases and empty arrays
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_reduction_empty() {
        let a = Array::<f64>::from_vec(vec![]);
        let b = Array::<f64>::from_vec(vec![]);

        let fused = FusedReduction::new(
            ArrayExpr::new(&a),
            ArrayExpr::new(&b),
            |x, y| x * y,
            |acc, v| acc + v,
            || 0.0,
        )
        .expect("Fused reduction on empty arrays should succeed");

        assert_relative_eq!(fused.reduce(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fused_scalar_broadcast_single_element() {
        let a = Array::from_vec(vec![5.0]);

        let fused = fused_scalar_broadcast_f64(ArrayExpr::new(&a), 3.0, 7.0);
        let result = fused.eval_fused();

        // 5*3+7 = 22
        assert_eq!(result.to_vec(), vec![22.0]);
    }

    #[test]
    fn test_fused_sum_of_squares_empty() {
        let a = Array::<f64>::from_vec(vec![]);
        assert_relative_eq!(fused_sum_of_squares(&a), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fusion_analysis_properties() {
        let none = FusionAnalysis::none();
        assert_eq!(none.pattern, FusionPattern::None);
        assert_relative_eq!(none.estimated_speedup, 1.0, epsilon = 1e-10);
        assert_eq!(none.allocations_eliminated, 0);

        let fma = FusionAnalysis::fma();
        assert_eq!(fma.pattern, FusionPattern::FusedMultiplyAdd);
        assert!(fma.estimated_speedup > 1.0);
        assert!(fma.allocations_eliminated > 0);
    }
}
