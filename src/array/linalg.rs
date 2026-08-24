//! Linear algebra operations
//!
//! This module contains linear algebra methods:
//! - Matrix multiplication (matmul, dot)
//! - SIMD-optimized operations (dot_simd, norm_l2_simd, norm_l1_simd)
//! - Condition number and related functions (cond, rcond)
//! - Least squares (lstsq)

#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use super::core::LstsqResult;
use super::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels;
use num_traits::{Float, Zero};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::borrow::Cow;
use std::fmt;
use std::fmt::Debug;
use std::ops::{Add, Mul};

/// Batched GEMM over `batch` row-major panels laid out back to back.
///
/// `a` is `[batch, m, k]`, `b` is `[batch, k, n]`, `c` is `[batch, m, n]`,
/// all flattened in row-major (C) order, so panel `t` of each occupies
/// exactly one contiguous run: `a[t*m*k .. (t+1)*m*k]`, and likewise for
/// `b`/`c`. Each panel is dispatched through [`kernels::gemm::gemm_2d`],
/// which picks its own f64/f32 SIMD or generic tier.
///
/// # Parallelism, and why it is f64/f32-only
///
/// Panels are disjoint (no cross-panel accumulation), so splitting the
/// batch across threads changes nothing about the arithmetic -- only which
/// thread evaluates which panel -- exactly the argument
/// `kernels::gemm`'s own row split makes for the `M` axis. Parallelizing a
/// *generic* `&mut [T]` would nonetheless require `T: Send + Sync`, and
/// `Array::matmul` is called from ~50 sites across this crate on generic
/// `T: Float + ...` parameters that carry no such bound; adding it to the
/// public signature is a far wider cascade than the `'static` this module
/// already takes on. Instead the parallel branch reinterprets the flat
/// operands as concrete `&[f64]`/`&mut [f64]` (or `f32`) via
/// [`kernels::cast`] -- which the `'static` bound already permits, and
/// which yields slices that are unconditionally `Send + Sync` -- and
/// parallelizes those. The f64/f32 tiers are exactly the ones where the
/// per-panel work is fast enough for scheduling overhead to matter, so
/// nothing is lost by leaving the generic tier sequential.
///
/// `total_flops >= GEMM_PARALLEL_MIN_FLOPS` forces `batch`, `m`, `n`, `k`
/// all `> 0`, so the `par_chunks_mut(m * n)` chunk size below is never
/// zero (rayon panics on a zero chunk size).
///
/// If a panel is *itself* large enough to clear `GEMM_PARALLEL_MIN_FLOPS`,
/// `gemm_2d`'s row split nests inside this batch split. Rayon supports
/// that (the inner split just joins within the same pool); it is not a
/// deadlock, only a mild scheduling cost in a regime where each panel is
/// already tens of milliseconds of work.
fn batched_gemm<T>(batch: usize, m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero + 'static,
{
    if batch == 0 || m == 0 || n == 0 {
        return;
    }

    // Defensive: every caller is a shape-validated public entry point one
    // layer up, so these always match in practice. Mirrors
    // `kernels::gemm::gemm_generic`'s convention for a violated
    // precondition -- leave `c` as the caller allocated it (already
    // zeroed) rather than indexing past what was actually passed.
    if a.len() != batch * m * k || b.len() != batch * k * n || c.len() != batch * m * n {
        return;
    }

    let total_flops = 2usize
        .saturating_mul(batch)
        .saturating_mul(m)
        .saturating_mul(n)
        .saturating_mul(k);

    if batch > 1 && total_flops >= kernels::GEMM_PARALLEL_MIN_FLOPS {
        if let (Some(a64), Some(b64)) = (kernels::cast::as_f64(a), kernels::cast::as_f64(b)) {
            if let Some(c64) = kernels::cast::as_f64_mut(c) {
                par_batch_f64(m, k, n, a64, b64, c64);
                return;
            }
        }
        if let (Some(a32), Some(b32)) = (kernels::cast::as_f32(a), kernels::cast::as_f32(b)) {
            if let Some(c32) = kernels::cast::as_f32_mut(c) {
                par_batch_f32(m, k, n, a32, b32, c32);
                return;
            }
        }
    }

    for t in 0..batch {
        kernels::gemm::gemm_2d(
            m,
            k,
            n,
            &a[t * m * k..(t + 1) * m * k],
            &b[t * k * n..(t + 1) * k * n],
            &mut c[t * m * n..(t + 1) * m * n],
        );
    }
}

/// f64 batch split for [`batched_gemm`]: one panel per rayon task.
fn par_batch_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    c.par_chunks_mut(m * n)
        .enumerate()
        .for_each(|(t, c_panel)| {
            kernels::gemm::gemm_2d(
                m,
                k,
                n,
                &a[t * m * k..(t + 1) * m * k],
                &b[t * k * n..(t + 1) * k * n],
                c_panel,
            );
        });
}

/// `f32` twin of [`par_batch_f64`].
fn par_batch_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_chunks_mut(m * n)
        .enumerate()
        .for_each(|(t, c_panel)| {
            kernels::gemm::gemm_2d(
                m,
                k,
                n,
                &a[t * m * k..(t + 1) * m * k],
                &b[t * k * n..(t + 1) * k * n],
                c_panel,
            );
        });
}

// Matrix multiplication.
//
// Split out of the block holding `dot` below purely so `matmul`/`matmul_2d`
// can carry the extra `+ 'static` bound that `kernels::gemm::gemm_2d`
// needs to `TypeId`-dispatch `T` onto its f64/f32 SIMD tiers. `dot` needs
// no dispatch and keeps its historical bounds unchanged.
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero + 'static,
{
    /// Perform matrix multiplication using BLAS if available
    ///
    /// Enhanced version with support for broadcasting and stacked matrices.
    /// If arrays have more than 2 dimensions, they are treated as stacks of matrices
    /// and broadcasting rules are applied to stack dimensions.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Handle the basic 2D case directly
        if a_shape.len() == 2 && b_shape.len() == 2 {
            return self.matmul_2d(other);
        }

        // For higher dimensions, we need to handle broadcasting.
        // Ensure both arrays have at least 2 dimensions. Only the 1-D
        // promotion actually needs to build a new array; every other
        // operand is *borrowed* through the `Cow` (the previous
        // implementation `clone()`d both operands unconditionally here,
        // copying every element of both inputs before a single multiply
        // had happened).
        let a: Cow<'_, Self> = if a_shape.len() == 1 {
            Cow::Owned(self.try_reshape(&[1, a_shape[0]])?)
        } else {
            Cow::Borrowed(self)
        };

        let b: Cow<'_, Self> = if b_shape.len() == 1 {
            Cow::Owned(other.try_reshape(&[b_shape[0], 1])?)
        } else {
            Cow::Borrowed(other)
        };

        let a_shape = a.shape();
        let b_shape = b.shape();

        // A 0-D operand has no core dimensions to slice off; return a
        // dimension error rather than underflowing `len() - 2` below.
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "matmul requires operands with at least 1 dimension".to_string(),
            ));
        }

        // Extract core dimensions (last 2 of each array)
        let a_core_shape = &a_shape[a_shape.len() - 2..];
        let b_core_shape = &b_shape[b_shape.len() - 2..];

        // Check if core dimensions are compatible for matrix multiplication
        if a_core_shape[1] != b_core_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_core_shape[0], b_core_shape[1]],
                actual: vec![a_core_shape[0], a_core_shape[1]],
            });
        }

        // Calculate batch dimensions (all but the last 2 of each array)
        let a_batch_shape = &a_shape[..a_shape.len() - 2];
        let b_batch_shape = &b_shape[..b_shape.len() - 2];

        // Calculate broadcast batch shape
        let broadcast_batch_shape = if a_batch_shape.is_empty() && b_batch_shape.is_empty() {
            vec![]
        } else if a_batch_shape.is_empty() {
            b_batch_shape.to_vec()
        } else if b_batch_shape.is_empty() {
            a_batch_shape.to_vec()
        } else {
            // Use broadcasting rules to get common batch shape
            Self::broadcast_shape(a_batch_shape, b_batch_shape)?
        };

        // Reshape arrays to broadcast batch dimensions
        let a_broadcast_shape = [&broadcast_batch_shape, a_core_shape].concat();
        let b_broadcast_shape = [&broadcast_batch_shape, b_core_shape].concat();

        let a_broadcast: Cow<'_, Self> = if a_shape == a_broadcast_shape {
            a
        } else {
            Cow::Owned(a.broadcast_to(&a_broadcast_shape)?)
        };

        let b_broadcast: Cow<'_, Self> = if b_shape == b_broadcast_shape {
            b
        } else {
            Cow::Owned(b.broadcast_to(&b_broadcast_shape)?)
        };

        let m = a_core_shape[0];
        let k = a_core_shape[1];
        let n = b_core_shape[1];

        // Calculate output shape
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        // Calculate total batch size (`1` when there are no batch axes,
        // i.e. the plain 2-D-after-promotion case)
        let batch_size: usize = broadcast_batch_shape.iter().product();

        // Both operands are now shaped `[batch.., m, k]` / `[batch.., k, n]`.
        // `operand()` hands back their data as one flat slice in *logical*
        // (row-major) order -- borrowed outright when the array is
        // contiguous (which every `broadcast_to` output is, since it is a
        // freshly owned standard-layout array), materialized only for a
        // non-contiguous operand that needed no broadcasting. In that
        // layout panel `t` is a contiguous run, so the whole batch reduces
        // to flat-index panel arithmetic feeding `kernels::gemm::gemm_2d`
        // -- replacing the previous per-element `IxDyn` `get`/`set` pair,
        // which rebuilt an index `Vec` and walked strides for every single
        // multiply-add.
        let a_flat = kernels::borrow::operand(&a_broadcast);
        let b_flat = kernels::borrow::operand(&b_broadcast);
        let mut c_data = vec![T::zero(); batch_size * m * n];
        batched_gemm(batch_size, m, k, n, &a_flat, &b_flat, &mut c_data);

        Self::from_vec_shape(c_data, &output_shape)
    }

    /// Basic 2D matrix multiplication (no broadcasting)
    fn matmul_2d(&self, other: &Self) -> Result<Self> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Check dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "matmul_2d requires 2D arrays".to_string(),
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0], b_shape[1]],
                actual: vec![a_shape[0], a_shape[1]],
            });
        }

        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        // Hand both operands to the crate's dtype-dispatched GEMM kernel:
        // `gemm_2d` routes f64/f32 onto `scirs2-core`'s blocked SIMD
        // matmul (row-split across threads once the FLOP count clears
        // `kernels::GEMM_PARALLEL_MIN_FLOPS`) and everything else onto the
        // same `BLOCK_SIZE = 64` blocked i-k-j triple loop that used to be
        // inlined right here.
        //
        // `operand()` borrows a contiguous operand outright; only a
        // non-contiguous layout (e.g. a permuted-axes view) materializes a
        // logically-ordered copy. The result is built with
        // `from_vec_shape`, which *consumes* the output buffer -- the
        // previous `from_vec(..).reshape(&[m, n])` had to clone the whole
        // buffer, because `reshape` takes `&self`.
        let a = kernels::borrow::operand(self);
        let b = kernels::borrow::operand(other);
        let mut c = vec![T::zero(); m * n];
        kernels::gemm::gemm_2d(m, k, n, &a, &b, &mut c);

        Self::from_vec_shape(c, &[m, n])
    }
}

// `dot` deliberately stays in its own impl block, on the *original*
// bounds: it needs no dtype dispatch, so there is no reason to make it
// demand `T: 'static` alongside `matmul`/`matmul_2d` above.
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    /// Compute the dot product of two vectors
    pub fn dot(&self, other: &Self) -> Result<T> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Check dimensions
        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "dot product requires 1D arrays".to_string(),
            ));
        }

        if a_shape[0] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a_shape,
                actual: b_shape,
            });
        }

        // Compute dot product via zero-copy iteration over both operands
        let result = self
            .array()
            .iter()
            .zip(other.array().iter())
            .fold(T::zero(), |acc, (a, b)| acc + a.clone() * b.clone());

        Ok(result)
    }
}

// SIMD-optimized operations for f64 using SimdUnifiedOps
impl Array<f64> {
    /// Compute SIMD-optimized dot product of two f64 vectors
    /// Uses SimdUnifiedOps for automatic platform detection (AVX-512, AVX2, NEON)
    pub fn dot_simd(&self, other: &Self) -> Result<f64> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Check dimensions
        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "dot product requires 1D arrays".to_string(),
            ));
        }

        if a_shape[0] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a_shape,
                actual: b_shape,
            });
        }

        // Use SimdUnifiedOps for platform-independent SIMD acceleration.
        // `as_cow_1d` borrows contiguous operands without copying.
        let a_nd = self.as_cow_1d();
        let b_nd = other.as_cow_1d();
        Ok(f64::simd_dot(&a_nd.view(), &b_nd.view()))
    }

    /// Compute SIMD-optimized L2 norm (Euclidean norm)
    /// Uses SimdUnifiedOps for automatic platform detection
    pub fn norm_l2_simd(&self) -> f64 {
        let nd_array = self.as_cow_1d();
        f64::simd_norm(&nd_array.view())
    }

    /// Compute SIMD-optimized L1 norm (Manhattan norm)
    /// Uses SimdUnifiedOps for automatic platform detection
    pub fn norm_l1_simd(&self) -> f64 {
        let nd_array = self.as_cow_1d();
        f64::simd_norm_l1(&nd_array.view())
    }
}

// Additional linear algebra methods for Array
impl<
        T: Float
            + Clone
            + fmt::Debug
            + std::ops::AddAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::SubAssign
            + std::fmt::Display
            + 'static,
    > Array<T>
{
    /// Compute the condition number of a matrix
    ///
    /// The condition number is the ratio of the largest to smallest singular value.
    /// A well-conditioned matrix has a condition number close to 1, while
    /// an ill-conditioned matrix has a large condition number.
    ///
    /// # Returns
    ///
    /// The condition number (L2 norm)
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    pub fn cond(&self) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        crate::new_modules::matrix_decomp::condition_number(self)
    }

    /// Compute the condition number of a matrix (fallback implementation)
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    pub fn cond(&self) -> Option<T> {
        // Check if matrix is square
        let shape = self.shape();
        if shape.len() != 2 {
            return None;
        }

        // Simple placeholder for when advanced features are not available
        Some(T::one())
    }

    /// Compute the reciprocal condition number
    ///
    /// This is 1/cond(matrix), which is more numerically stable
    /// for matrices with large condition numbers.
    ///
    /// # Returns
    ///
    /// The reciprocal condition number
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    pub fn rcond(&self) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        crate::new_modules::matrix_decomp::rcond(self)
    }

    /// Compute the reciprocal condition number (fallback implementation)
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    pub fn rcond(&self) -> Option<T> {
        self.cond().map(|c| T::one() / c)
    }

    /// Check if a matrix is well-conditioned
    ///
    /// A matrix is considered well-conditioned if its condition number
    /// is below a reasonable threshold (typically 1e12 for double precision).
    ///
    /// # Returns
    ///
    /// True if the matrix is well-conditioned, false otherwise
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    pub fn is_well_conditioned(&self) -> Result<bool>
    where
        T: Float + Clone + Debug,
    {
        let cond = crate::new_modules::matrix_decomp::condition_number(self)?;
        let threshold = T::from(1e4_f64)
            .unwrap_or_else(|| T::from(1e3_f64).expect("1e3 should be representable"));
        Ok(cond < threshold)
    }

    /// Check if a matrix is well-conditioned (fallback implementation)
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    pub fn is_well_conditioned(&self) -> bool {
        match self.cond() {
            Some(cond_num) => {
                let threshold = T::from(1e4)
                    .unwrap_or(T::from(1000.0).expect("1000.0 should be representable"));
                cond_num < threshold
            }
            None => false,
        }
    }

    /// Compute the sign and log determinant of the matrix
    ///
    /// This is a numerically stable way to compute the determinant for large matrices
    /// where the determinant might overflow or underflow.
    ///
    /// # Returns
    ///
    /// A tuple (sign, logdet) where sign is -1, 0, or 1, and logdet is the natural
    /// logarithm of the absolute value of the determinant.
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    pub fn slogdet(&self) -> Result<(i8, T)>
    where
        T: Float + Clone + Debug,
    {
        crate::new_modules::matrix_decomp::slogdet(self)
    }

    /// Solve a linear least-squares problem
    ///
    /// Computes the least-squares solution to the linear system Ax = b.
    /// If the system is over-determined, this finds the solution that minimizes ||Ax - b||_2.
    /// If the system is under-determined, this finds the minimum-norm solution.
    ///
    /// # Arguments
    /// * `b` - Right-hand side vector or matrix
    /// * `rcond` - Cutoff for small singular values. If None, uses machine precision.
    ///
    /// # Returns
    /// A tuple (x, residuals, rank, singular_values)
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    pub fn lstsq(&self, b: &Array<T>, rcond: Option<T>) -> LstsqResult<T>
    where
        T: Float + Clone + Debug,
    {
        crate::new_modules::matrix_decomp::lstsq(self, b, rcond)
    }
}
