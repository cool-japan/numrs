//! Distributed Linear Algebra Operations
//!
//! This module provides distributed implementations of common linear algebra operations
//! for large-scale matrix computations across multiple processes.
//!
//! # Operations
//!
//! - Matrix multiplication (SUMMA algorithm)
//! - Dot product
//! - Matrix-vector operations
//! - Matrix decompositions (SVD, QR, Cholesky)
//! - Linear system solving
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::linalg::*;
//! use numrs2::distributed::array::*;
//! use numrs2::distributed::process::*;
//!
//! # async fn example() -> Result<(), DistributedLinalgError> {
//! let world = init().await.map_err(|e| DistributedLinalgError::LinalgError(e.to_string()))?;
//!
//! // Create distributed matrices
//! let local_a = vec![1.0_f64; 100];
//! let dist_a = DistributedArray::from_local(
//!     local_a,
//!     DistributionStrategy::Block,
//!     400,
//!     &world
//! )?;
//!
//! let local_b = vec![2.0_f64; 100];
//! let dist_b = DistributedArray::from_local(
//!     local_b,
//!     DistributionStrategy::Block,
//!     400,
//!     &world
//! )?;
//!
//! // Distributed dot product
//! let result = distributed_dot(&dist_a, &dist_b).await?;
//! if world.is_root() {
//!     println!("Dot product: {}", result);
//! }
//!
//! finalize(world).await.map_err(|e| DistributedLinalgError::LinalgError(e.to_string()))?;
//! # Ok(())
//! # }
//! ```

use super::array::{DistributedArray, DistributedArrayError, DistributionStrategy};
use super::collective::{allreduce, CollectiveError, ReduceOp};
use super::process::Communicator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::{qr, svd};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use thiserror::Error;

/// Errors that can occur during distributed linear algebra operations
#[derive(Error, Debug)]
pub enum DistributedLinalgError {
    #[error("Distributed array error: {0}")]
    Array(#[from] DistributedArrayError),

    #[error("Collective operation error: {0}")]
    Collective(#[from] CollectiveError),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid matrix dimensions: rows={rows}, cols={cols}")]
    InvalidDimensions { rows: usize, cols: usize },

    #[error("Singular matrix")]
    SingularMatrix,

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),

    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    #[error("Not yet implemented: {0}")]
    NotImplemented(String),
}

/// Distributed dot product of two vectors
///
/// Computes the dot product of two distributed vectors using parallel reduction.
///
/// # Arguments
///
/// * `x` - First distributed vector
/// * `y` - Second distributed vector
///
/// # Returns
///
/// The scalar dot product result (same value on all processes)
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(x: &DistributedArray<f64>, y: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let dot_product = distributed_dot(x, y).await?;
/// println!("Dot product: {}", dot_product);
/// # Ok(())
/// # }
/// ```
pub async fn distributed_dot<T>(
    x: &DistributedArray<T>,
    y: &DistributedArray<T>,
) -> Result<T, DistributedLinalgError>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Add<Output = T>
        + Mul<Output = T>
        + PartialOrd
        + Send
        + 'static,
    T: std::iter::Sum,
{
    // Check dimensions
    if x.global_size() != y.global_size() {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "Vector sizes don't match: {} vs {}",
            x.global_size(),
            y.global_size()
        )));
    }

    // Compute local dot product
    let local_x = x.local_data();
    let local_y = y.local_data();

    let local_result = local_x
        .iter()
        .zip(local_y.iter())
        .map(|(a, b)| a.clone() * b.clone())
        .sum::<T>();

    // Global reduction (sum)
    let global_result = allreduce(&[local_result], ReduceOp::Sum, x.comm()).await?;

    global_result
        .into_iter()
        .next()
        .ok_or_else(|| DistributedLinalgError::LinalgError("Empty reduction result".to_string()))
}

/// Distributed matrix-vector multiplication
///
/// Computes y = A * x where A is a distributed matrix and x is a distributed vector.
///
/// # Arguments
///
/// * `a` - Distributed matrix (row-distributed)
/// * `x` - Distributed vector
///
/// # Returns
///
/// Distributed result vector y
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>, x: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let y = distributed_matvec(a, x).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_matvec<T>(
    _a: &DistributedArray<T>,
    _x: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Add<Output = T>
        + Mul<Output = T>
        + PartialOrd
        + Send
        + 'static,
{
    // Placeholder implementation
    // Real implementation would perform distributed matrix-vector multiply
    Err(DistributedLinalgError::NotImplemented(
        "Distributed matrix-vector multiplication".to_string(),
    ))
}

/// Distributed matrix multiplication using SUMMA algorithm
///
/// Computes C = A * B where A and B are distributed matrices.
/// Uses the Scalable Universal Matrix Multiplication Algorithm (SUMMA).
///
/// # Arguments
///
/// * `a` - First distributed matrix
/// * `b` - Second distributed matrix
///
/// # Returns
///
/// Distributed result matrix C
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>, b: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let c = distributed_matmul(a, b).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_matmul<T>(
    _a: &DistributedArray<T>,
    _b: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Add<Output = T>
        + Mul<Output = T>
        + PartialOrd
        + Send
        + 'static,
{
    // Placeholder implementation
    // Real implementation would use SUMMA algorithm:
    // 1. Partition matrices into blocks
    // 2. Broadcast blocks along rows/columns
    // 3. Perform local matrix multiplications
    // 4. Accumulate results
    Err(DistributedLinalgError::NotImplemented(
        "Distributed matrix multiplication (SUMMA)".to_string(),
    ))
}

/// Distributed Singular Value Decomposition (SVD)
///
/// Computes A = U * Σ * V^T where A is a distributed matrix.
///
/// # Arguments
///
/// * `a` - Distributed matrix to decompose
///
/// # Returns
///
/// Tuple of (U, singular_values, Vt) as distributed arrays
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let (u, s, vt) = distributed_svd(a).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_svd<T>(
    _a: &DistributedArray<T>,
) -> Result<(DistributedArray<T>, Vec<T>, DistributedArray<T>), DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    // Placeholder implementation
    // Real implementation would use distributed SVD algorithms:
    // - Tall-skinny SVD for tall matrices
    // - Block-based algorithms for large matrices
    // - Iterative methods for largest singular values
    Err(DistributedLinalgError::NotImplemented(
        "Distributed SVD".to_string(),
    ))
}

/// Distributed QR decomposition
///
/// Computes A = Q * R where Q is orthogonal and R is upper triangular.
///
/// # Arguments
///
/// * `a` - Distributed matrix to decompose
///
/// # Returns
///
/// Tuple of (Q, R) as distributed arrays
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let (q, r) = distributed_qr(a).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_qr<T>(
    _a: &DistributedArray<T>,
) -> Result<(DistributedArray<T>, DistributedArray<T>), DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    // Placeholder implementation
    // Real implementation would use:
    // - Block Householder QR
    // - Communication-avoiding QR (CAQR)
    // - Tree-based QR for tall-skinny matrices
    Err(DistributedLinalgError::NotImplemented(
        "Distributed QR decomposition".to_string(),
    ))
}

/// Distributed Cholesky factorization
///
/// Computes A = L * L^T for symmetric positive definite matrix A.
///
/// # Arguments
///
/// * `a` - Distributed symmetric positive definite matrix
///
/// # Returns
///
/// Lower triangular factor L as distributed array
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let l = distributed_cholesky(a).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_cholesky<T>(
    _a: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    // Placeholder implementation
    // Real implementation would use block Cholesky with broadcasts
    Err(DistributedLinalgError::NotImplemented(
        "Distributed Cholesky factorization".to_string(),
    ))
}

/// Distributed linear system solver
///
/// Solves A * x = b for x where A is a distributed matrix and b is a distributed vector.
///
/// # Arguments
///
/// * `a` - Distributed coefficient matrix
/// * `b` - Distributed right-hand side vector
///
/// # Returns
///
/// Distributed solution vector x
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>, b: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let x = distributed_solve(a, b).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_solve<T>(
    _a: &DistributedArray<T>,
    _b: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    // Placeholder implementation
    // Real implementation would use:
    // - Distributed LU factorization + triangular solves
    // - Iterative methods (CG, GMRES) for large sparse systems
    // - Block algorithms for dense systems
    Err(DistributedLinalgError::NotImplemented(
        "Distributed linear system solve".to_string(),
    ))
}

/// Distributed norm computation
///
/// Computes various norms of a distributed vector.
pub async fn distributed_norm<T>(
    x: &DistributedArray<T>,
    p: f64,
) -> Result<f64, DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
    T: Into<f64> + Copy,
{
    let local_x = x.local_data();

    let local_sum = if p == f64::INFINITY {
        // Infinity norm: max absolute value
        local_x
            .iter()
            .map(|&v| Into::<f64>::into(v).abs())
            .fold(0.0, f64::max)
    } else if p == 2.0 {
        // L2 norm: sqrt(sum of squares)
        local_x
            .iter()
            .map(|&v| {
                let val = Into::<f64>::into(v);
                val * val
            })
            .sum::<f64>()
    } else {
        // Lp norm: (sum of |x|^p)^(1/p)
        local_x
            .iter()
            .map(|&v| Into::<f64>::into(v).abs().powf(p))
            .sum::<f64>()
    };

    // Global reduction
    let global_sum = if p == f64::INFINITY {
        // Use max reduction for infinity norm
        let result = allreduce(&[local_sum], ReduceOp::Max, x.comm()).await?;
        result[0]
    } else {
        // Use sum reduction for other norms
        let result = allreduce(&[local_sum], ReduceOp::Sum, x.comm()).await?;
        if p == 2.0 {
            result[0].sqrt()
        } else {
            result[0].powf(1.0 / p)
        }
    };

    Ok(global_sum)
}

/// Helper structure for matrix dimensions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatrixDims {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl MatrixDims {
    /// Create new matrix dimensions
    pub fn new(rows: usize, cols: usize) -> Result<Self, DistributedLinalgError> {
        if rows == 0 || cols == 0 {
            return Err(DistributedLinalgError::InvalidDimensions { rows, cols });
        }
        Ok(Self { rows, cols })
    }

    /// Check if dimensions are compatible for matrix multiplication
    pub fn can_multiply(&self, other: &MatrixDims) -> bool {
        self.cols == other.rows
    }

    /// Get result dimensions for matrix multiplication
    pub fn multiply_result(&self, other: &MatrixDims) -> Option<MatrixDims> {
        if self.can_multiply(other) {
            Some(MatrixDims {
                rows: self.rows,
                cols: other.cols,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_dims() {
        let dims = MatrixDims::new(3, 4).expect("Valid dimensions");
        assert_eq!(dims.rows, 3);
        assert_eq!(dims.cols, 4);
    }

    #[test]
    fn test_matrix_dims_invalid() {
        assert!(MatrixDims::new(0, 4).is_err());
        assert!(MatrixDims::new(3, 0).is_err());
    }

    #[test]
    fn test_matrix_dims_can_multiply() {
        let a = MatrixDims::new(3, 4).expect("Valid");
        let b = MatrixDims::new(4, 5).expect("Valid");
        let c = MatrixDims::new(5, 2).expect("Valid");

        assert!(a.can_multiply(&b));
        assert!(b.can_multiply(&c));
        assert!(!a.can_multiply(&c));
    }

    #[test]
    fn test_matrix_dims_multiply_result() {
        let a = MatrixDims::new(3, 4).expect("Valid");
        let b = MatrixDims::new(4, 5).expect("Valid");

        let result = a.multiply_result(&b).expect("Compatible");
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 5);
    }

    #[test]
    fn test_matrix_dims_multiply_incompatible() {
        let a = MatrixDims::new(3, 4).expect("Valid");
        let b = MatrixDims::new(5, 2).expect("Valid");

        assert!(a.multiply_result(&b).is_none());
    }
}
