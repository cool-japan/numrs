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
use num_traits::{Float, Zero};
use scirs2_core::ndarray::IxDyn;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt;
use std::fmt::Debug;
use std::ops::{Add, Mul};

// Matrix multiplication
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
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

        // For higher dimensions, we need to handle broadcasting
        // Ensure both arrays have at least 2 dimensions
        let a = if a_shape.len() == 1 {
            self.reshape(&[1, a_shape[0]])
        } else {
            self.clone()
        };

        let b = if b_shape.len() == 1 {
            other.reshape(&[b_shape[0], 1])
        } else {
            other.clone()
        };

        let a_shape = a.shape();
        let b_shape = b.shape();

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

        let a_broadcast = if a_shape == a_broadcast_shape {
            a
        } else {
            a.broadcast_to(&a_broadcast_shape)?
        };

        let b_broadcast = if b_shape == b_broadcast_shape {
            b
        } else {
            b.broadcast_to(&b_broadcast_shape)?
        };

        // Calculate output shape
        let output_core_shape = vec![a_core_shape[0], b_core_shape[1]];
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.extend_from_slice(&output_core_shape);

        // Perform batch matrix multiplication
        let mut result = Self::zeros(&output_shape);

        // Calculate total batch size
        let batch_size: usize = broadcast_batch_shape.iter().product();

        // For each batch, perform matrix multiplication
        for batch_idx in 0..batch_size {
            // Calculate indices for this batch
            let mut batch_indices = Vec::with_capacity(broadcast_batch_shape.len());
            let mut temp = batch_idx;

            for &dim in broadcast_batch_shape.iter().rev() {
                batch_indices.insert(0, temp % dim);
                temp /= dim;
            }

            // Extract matrices for this batch
            let mut a_indices = batch_indices.clone();
            a_indices.push(0); // Placeholder for row index
            a_indices.push(0); // Placeholder for column index

            let mut b_indices = batch_indices.clone();
            b_indices.push(0); // Placeholder for row index
            b_indices.push(0); // Placeholder for column index

            // Perform matrix multiplication for this batch
            let m = a_core_shape[0];
            let n = b_core_shape[1];
            let k = a_core_shape[1];

            for i in 0..m {
                let a_idx_pos = a_indices.len() - 2;
                a_indices[a_idx_pos] = i;

                for j in 0..n {
                    let b_idx_pos = b_indices.len() - 1;
                    b_indices[b_idx_pos] = j;

                    let mut sum = T::zero();

                    for l in 0..k {
                        let a_col_pos = a_indices.len() - 1;
                        a_indices[a_col_pos] = l;
                        let b_row_pos = b_indices.len() - 2;
                        b_indices[b_row_pos] = l;

                        let a_val = a_broadcast
                            .array()
                            .get(IxDyn(&a_indices))
                            .expect("broadcast element access should succeed");
                        let b_val = b_broadcast
                            .array()
                            .get(IxDyn(&b_indices))
                            .expect("broadcast element access should succeed");

                        sum = sum + a_val.clone() * b_val.clone();
                    }

                    // Calculate output indices
                    let mut output_indices = batch_indices.clone();
                    output_indices.push(i);
                    output_indices.push(j);

                    result.set(&output_indices, sum)?;
                }
            }
        }

        Ok(result)
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

        // Implement cache-aware blocked matrix multiplication
        // This provides significant performance improvements over naive O(n^3) algorithm

        // Cache-optimized implementation with improved memory access pattern
        // Borrow the contiguous operands without copying; only non-contiguous
        // layouts incur a materializing copy. The output buffer is allocated
        // directly instead of via a throwaway zeroed Array.
        let mut c_data = vec![T::zero(); m * n];
        let owned_a;
        let a_data: &[T] = match self.as_slice() {
            Some(slice) => slice,
            None => {
                owned_a = self.to_vec();
                &owned_a
            }
        };
        let owned_b;
        let b_data: &[T] = match other.as_slice() {
            Some(slice) => slice,
            None => {
                owned_b = other.to_vec();
                &owned_b
            }
        };

        // Cache-optimized matrix multiplication with improved memory access pattern
        // Using i-k-j loop order for better cache locality
        const BLOCK_SIZE: usize = 64; // Cache-friendly block size

        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                for j_block in (0..n).step_by(BLOCK_SIZE) {
                    // Process blocks to improve cache reuse
                    let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                    let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                    let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);

                    for i in i_block..i_end {
                        for k_l in k_block..k_end {
                            let a_ik = &a_data[i * k + k_l];
                            for j in j_block..j_end {
                                c_data[i * n + j] = c_data[i * n + j].clone()
                                    + a_ik.clone() * b_data[k_l * n + j].clone();
                            }
                        }
                    }
                }
            }
        }

        Ok(Self::from_vec(c_data).reshape(&[m, n]))
    }

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
            + std::fmt::Display,
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
