//! Randomized Linear Algebra Algorithms
//!
//! This module provides randomized algorithms for large-scale linear algebra computations.
//! These algorithms trade exact computation for speed, providing approximate solutions that
//! are often sufficient for many applications.
//!
//! ## Key Algorithms
//!
//! - **Randomized SVD**: Fast approximate singular value decomposition
//! - **Random Projections**: Dimensionality reduction via random matrices
//! - **Randomized Range Finder**: Basis for column space approximation
//! - **Randomized Power Iteration**: Enhanced spectral approximation
//!
//! ## Use Cases
//!
//! - Large-scale dimensionality reduction
//! - Low-rank matrix approximation
//! - Principal Component Analysis (PCA)
//! - Data sketching and streaming algorithms
//! - Fast approximate matrix decompositions
//!
//! ## References
//!
//! - Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). "Finding structure with
//!   randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
//!   SIAM review, 53(2), 217-288.
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::prelude::*;
//! use numrs2::linalg::randomized::*;
//!
//! // Create a large matrix
//! let m = 1000;
//! let n = 500;
//! let a = Array::random(&[m, n]);
//!
//! // Compute randomized SVD with target rank 50
//! let (u, s, vt) = randomized_svd(&a, 50, None, 2)?;
//!
//! // Compute random projection to 100 dimensions
//! let projected = random_projection(&a, 100, ProjectionType::Gaussian)?;
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use scirs2_core::random::{seeded_rng, Distribution, StandardNormal, Uniform};
use std::ops::{Add, Div, Mul, Sub};

/// Type of random projection matrix to use
#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    /// Gaussian random projection (entries drawn from N(0, 1))
    Gaussian,
    /// Sparse random projection (entries are -1, 0, or 1)
    Sparse,
    /// Rademacher random projection (entries are ±1)
    Rademacher,
}

/// Randomized SVD algorithm
///
/// Computes an approximate low-rank SVD of matrix A using randomized algorithms.
/// This is much faster than exact SVD for large matrices when only top-k singular
/// values/vectors are needed.
///
/// # Arguments
///
/// * `a` - Input matrix (m × n)
/// * `rank` - Target rank for the approximation
/// * `n_oversamples` - Number of additional samples for accuracy (default: 10)
/// * `n_iter` - Number of power iterations for better accuracy (default: 0)
///
/// # Returns
///
/// * `(U, S, Vt)` where A ≈ U @ diag(S) @ Vt
///   - U: m × rank orthonormal matrix
///   - S: rank-length vector of singular values
///   - Vt: rank × n orthonormal matrix
///
/// # Algorithm
///
/// 1. Generate random matrix Ω of size n × (rank + n_oversamples)
/// 2. Compute Y = A @ Ω
/// 3. Optionally perform power iterations: Y = (A @ A.T)^n_iter @ Y
/// 4. Orthonormalize Y to get Q via QR decomposition
/// 5. Compute B = Q.T @ A
/// 6. Compute exact SVD of small matrix B
/// 7. Return U = Q @ U_B, S, Vt = Vt_B
pub fn randomized_svd<T>(
    a: &Array<T>,
    rank: usize,
    n_oversamples: Option<usize>,
    n_iter: usize,
) -> Result<(Array<T>, Array<T>, Array<T>)>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::fmt::Debug
        + 'static,
{
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Randomized SVD requires 2D matrix".to_string(),
        ));
    }

    let m = shape[0];
    let n = shape[1];

    if rank > std::cmp::min(m, n) {
        return Err(NumRs2Error::ValueError(format!(
            "Rank {} exceeds matrix dimensions {}x{}",
            rank, m, n
        )));
    }

    let n_oversamples = n_oversamples.unwrap_or(10);
    let n_random = std::cmp::min(rank + n_oversamples, std::cmp::min(m, n));

    // Step 1: Generate random matrix Ω (n × n_random)
    let omega = generate_random_gaussian_matrix(n, n_random)?;

    // Step 2: Compute Y = A @ Ω
    let mut y = a.matmul(&omega)?;

    // Step 3: Power iterations (optional, for better accuracy)
    for _ in 0..n_iter {
        // Y = A @ (A.T @ Y)
        let aty = a.transpose().matmul(&y)?;
        y = a.matmul(&aty)?;
    }

    // Step 4: Orthonormalize Y to get Q via QR decomposition
    let (q, _r) = qr_decomposition(&y)?;

    // Step 5: Compute B = Q.T @ A
    let b = q.transpose().matmul(a)?;

    // Step 6: Compute exact SVD of small matrix B (n_random × n)
    let (u_b, s_full, vt_b) = compute_svd(&b)?;

    // Step 7: Extract top 'rank' components
    let u_b_rank = extract_columns(&u_b, rank)?;
    let s_rank = extract_elements(&s_full, rank)?;
    let vt_rank = extract_rows(&vt_b, rank)?;

    // U = Q @ U_B
    let u = q.matmul(&u_b_rank)?;

    Ok((u, s_rank, vt_rank))
}

/// Randomized range finder
///
/// Finds an orthonormal matrix Q whose range approximates the range of A.
/// This is the core subroutine for many randomized matrix algorithms.
///
/// # Arguments
///
/// * `a` - Input matrix (m × n)
/// * `rank` - Target rank
/// * `n_iter` - Number of power iterations (default: 0)
///
/// # Returns
///
/// * Q: m × rank orthonormal matrix where range(Q) ≈ range(A)
pub fn randomized_range_finder<T>(a: &Array<T>, rank: usize, n_iter: usize) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static,
{
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Range finder requires 2D matrix".to_string(),
        ));
    }

    let n = shape[1];

    // Generate random matrix
    let omega = generate_random_gaussian_matrix(n, rank)?;

    // Compute Y = A @ Ω
    let mut y = a.matmul(&omega)?;

    // Power iterations
    for _ in 0..n_iter {
        let aty = a.transpose().matmul(&y)?;
        y = a.matmul(&aty)?;
    }

    // Orthonormalize via QR
    let (q, _) = qr_decomposition(&y)?;

    Ok(q)
}

/// Random projection for dimensionality reduction
///
/// Projects high-dimensional data to a lower-dimensional space using a random matrix.
/// This preserves approximate distances (Johnson-Lindenstrauss lemma).
///
/// # Arguments
///
/// * `a` - Input matrix (m × n)
/// * `target_dim` - Target dimensionality
/// * `projection_type` - Type of random projection to use
///
/// # Returns
///
/// * Projected matrix (m × target_dim)
pub fn random_projection<T>(
    a: &Array<T>,
    target_dim: usize,
    projection_type: ProjectionType,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static,
{
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Random projection requires 2D matrix".to_string(),
        ));
    }

    let n = shape[1];

    if target_dim > n {
        return Err(NumRs2Error::ValueError(format!(
            "Target dimension {} exceeds input dimension {}",
            target_dim, n
        )));
    }

    // Generate projection matrix
    let proj_matrix = match projection_type {
        ProjectionType::Gaussian => generate_random_gaussian_matrix(n, target_dim)?,
        ProjectionType::Sparse => generate_sparse_random_matrix(n, target_dim)?,
        ProjectionType::Rademacher => generate_rademacher_matrix(n, target_dim)?,
    };

    // Apply projection: A_proj = A @ P
    let projected = a.matmul(&proj_matrix)?;

    // Scale by sqrt(n / target_dim) for variance preservation
    let scale = T::from((n as f64) / (target_dim as f64))
        .expect("ratio of positive integers is a valid f64")
        .sqrt();
    let scaled = projected.multiply_scalar(scale);

    Ok(scaled)
}

/// Low-rank approximation using randomized algorithms
///
/// Computes A ≈ Q @ Q.T @ A where Q is an orthonormal basis for the range of A.
///
/// # Arguments
///
/// * `a` - Input matrix (m × n)
/// * `rank` - Target rank
/// * `n_iter` - Number of power iterations
///
/// # Returns
///
/// * Approximation of A with rank approximately equal to `rank`
pub fn randomized_low_rank_approximation<T>(
    a: &Array<T>,
    rank: usize,
    n_iter: usize,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static,
{
    // Find orthonormal basis Q
    let q = randomized_range_finder(a, rank, n_iter)?;

    // Compute Q.T @ A
    let qt_a = q.transpose().matmul(a)?;

    // Reconstruct: A_approx = Q @ (Q.T @ A)
    let a_approx = q.matmul(&qt_a)?;

    Ok(a_approx)
}

// ===================================================================
// Helper Functions
// ===================================================================

/// Generate random Gaussian matrix with entries from N(0, 1)
fn generate_random_gaussian_matrix<T>(rows: usize, cols: usize) -> Result<Array<T>>
where
    T: Float + Clone + Default,
{
    let mut rng = seeded_rng(42);
    let dist = StandardNormal;

    let data: Vec<T> = (0..rows * cols)
        .map(|_| {
            let sample: f64 = dist.sample(&mut rng);
            T::from(sample).expect("standard normal sample is a valid f64")
        })
        .collect();

    Array::from_vec_shape(data, &[rows, cols])
}

/// Generate sparse random matrix (entries are -1, 0, 1 with specific probabilities)
fn generate_sparse_random_matrix<T>(rows: usize, cols: usize) -> Result<Array<T>>
where
    T: Float + Clone + Default + Zero + One,
{
    let mut rng = seeded_rng(42);
    let dist = Uniform::new(0.0, 1.0).expect("uniform distribution with valid range");

    let data: Vec<T> = (0..rows * cols)
        .map(|_| {
            let p: f64 = dist.sample(&mut rng);
            if p < 1.0 / 6.0 {
                T::one()
            } else if p < 2.0 / 6.0 {
                -T::one()
            } else {
                T::zero()
            }
        })
        .collect();

    Array::from_vec_shape(data, &[rows, cols])
}

/// Generate Rademacher matrix (entries are ±1 with equal probability)
fn generate_rademacher_matrix<T>(rows: usize, cols: usize) -> Result<Array<T>>
where
    T: Float + Clone + Default + One,
{
    let mut rng = seeded_rng(42);
    let dist = Uniform::new(0.0, 1.0).expect("uniform distribution with valid range");

    let data: Vec<T> = (0..rows * cols)
        .map(|_| {
            let p: f64 = dist.sample(&mut rng);
            let threshold = T::from(0.5).expect("0.5 is a valid f64 constant");
            let p_t = T::from(p).expect("uniform sample is a valid f64");
            if p_t < threshold {
                T::one()
            } else {
                -T::one()
            }
        })
        .collect();

    Array::from_vec_shape(data, &[rows, cols])
}

/// Simplified QR decomposition using Gram-Schmidt
fn qr_decomposition<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>)>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];

    let mut q_data = vec![T::zero(); m * n];
    let mut r_data = vec![T::zero(); n * n];

    // Modified Gram-Schmidt process
    for j in 0..n {
        // Start with column j of A
        let mut v: Vec<T> = (0..m)
            .map(|i| a.get(&[i, j]).expect("array indices are valid"))
            .collect();

        // Orthogonalize against previous columns
        for i in 0..j {
            // r_ij = <q_i, v>
            let mut dot = T::zero();
            for k in 0..m {
                dot = dot + q_data[k * n + i] * v[k];
            }
            r_data[i * n + j] = dot;

            // v = v - r_ij * q_i
            for k in 0..m {
                v[k] = v[k] - dot * q_data[k * n + i];
            }
        }

        // Compute norm
        let mut norm = T::zero();
        for k in 0..m {
            norm = norm + v[k] * v[k];
        }
        norm = norm.sqrt();

        r_data[j * n + j] = norm;

        // Normalize
        if norm > T::zero() {
            for k in 0..m {
                q_data[k * n + j] = v[k] / norm;
            }
        }
    }

    let q = Array::from_vec_shape(q_data, &[m, n])?;
    let r = Array::from_vec_shape(r_data, &[n, n])?;

    Ok((q, r))
}

/// Simplified SVD computation (delegates to matrix_decomp implementation)
fn compute_svd<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>, Array<T>)>
where
    T: Float + Clone + Default + std::fmt::Debug,
{
    #[cfg(feature = "lapack")]
    {
        use crate::new_modules::matrix_decomp::svd;
        svd(a)
    }
    #[cfg(not(feature = "lapack"))]
    {
        let _ = a;
        Err(NumRs2Error::FeatureNotEnabled(
            "lapack feature required for SVD in randomized algorithms".to_string(),
        ))
    }
}

/// Extract first k columns of a matrix
fn extract_columns<T>(a: &Array<T>, k: usize) -> Result<Array<T>>
where
    T: Clone + Default + Zero,
{
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];

    if k > n {
        return Err(NumRs2Error::ValueError(format!(
            "Cannot extract {} columns from matrix with {} columns",
            k, n
        )));
    }

    let mut data = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            data.push(a.get(&[i, j])?);
        }
    }

    Array::from_vec_shape(data, &[m, k])
}

/// Extract first k rows of a matrix
fn extract_rows<T>(a: &Array<T>, k: usize) -> Result<Array<T>>
where
    T: Clone + Default + Zero,
{
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];

    if k > m {
        return Err(NumRs2Error::ValueError(format!(
            "Cannot extract {} rows from matrix with {} rows",
            k, m
        )));
    }

    let mut data = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            data.push(a.get(&[i, j])?);
        }
    }

    Array::from_vec_shape(data, &[k, n])
}

/// Extract first k elements of a vector
fn extract_elements<T>(a: &Array<T>, k: usize) -> Result<Array<T>>
where
    T: Clone + Default + Zero,
{
    let shape = a.shape();

    if shape.len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "extract_elements requires 1D array".to_string(),
        ));
    }

    let n = shape[0];
    if k > n {
        return Err(NumRs2Error::ValueError(format!(
            "Cannot extract {} elements from array with {} elements",
            k, n
        )));
    }

    let mut data = Vec::with_capacity(k);
    for i in 0..k {
        data.push(a.get(&[i])?);
    }

    Ok(Array::from_vec(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_generate_gaussian_matrix() {
        let mat: Array<f64> =
            generate_random_gaussian_matrix(10, 5).expect("matrix generation should succeed");
        assert_eq!(mat.shape(), vec![10, 5]);
        // Check that values are reasonable (not all zeros)
        let sum: f64 = mat.to_vec().iter().map(|&x| x.abs()).sum();
        assert!(sum > 0.1); // Should have some non-zero values
    }

    #[test]
    fn test_generate_sparse_matrix() {
        let mat: Array<f64> =
            generate_sparse_random_matrix(100, 50).expect("matrix generation should succeed");
        assert_eq!(mat.shape(), vec![100, 50]);

        // Check sparsity (approximately 2/3 should be zero)
        let data = mat.to_vec();
        let zero_count = data.iter().filter(|&&x| x.abs() < 1e-10).count();
        let total = data.len();
        let zero_ratio = zero_count as f64 / total as f64;

        // Should be approximately 2/3 zeros (allow some variance)
        assert!(zero_ratio > 0.5 && zero_ratio < 0.8);
    }

    #[test]
    fn test_generate_rademacher_matrix() {
        let mat: Array<f64> =
            generate_rademacher_matrix(10, 5).expect("matrix generation should succeed");
        assert_eq!(mat.shape(), vec![10, 5]);

        // All values should be +1 or -1
        let data = mat.to_vec();
        for &val in &data {
            assert!((val - 1.0).abs() < 1e-10 || (val + 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qr_decomposition_small() {
        // Create a simple 3x2 matrix
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);

        let (q, r) = qr_decomposition(&a).expect("QR decomposition should succeed");

        // Check shapes
        assert_eq!(q.shape(), vec![3, 2]);
        assert_eq!(r.shape(), vec![2, 2]);

        // Check Q is orthonormal: Q.T @ Q should be approximately I
        let qtq = q.transpose().matmul(&q).expect("matmul should succeed");
        for i in 0..2 {
            for j in 0..2 {
                let val = qtq.get(&[i, j]).expect("valid index");
                if i == j {
                    assert_relative_eq!(val, 1.0, epsilon = 1e-6);
                } else {
                    assert_relative_eq!(val, 0.0, epsilon = 1e-6);
                }
            }
        }

        // Check R is upper triangular
        assert!(r.get(&[1, 0]).expect("valid index").abs() < 1e-6);
    }

    #[test]
    fn test_extract_columns() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

        let extracted = extract_columns(&a, 2).expect("extract columns should succeed");
        assert_eq!(extracted.shape(), vec![2, 2]);
        assert_relative_eq!(extracted.get(&[0, 0]).expect("valid index"), 1.0);
        assert_relative_eq!(extracted.get(&[0, 1]).expect("valid index"), 2.0);
        assert_relative_eq!(extracted.get(&[1, 0]).expect("valid index"), 4.0);
        assert_relative_eq!(extracted.get(&[1, 1]).expect("valid index"), 5.0);
    }

    #[test]
    fn test_extract_rows() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);

        let extracted = extract_rows(&a, 2).expect("extract rows should succeed");
        assert_eq!(extracted.shape(), vec![2, 2]);
        assert_relative_eq!(extracted.get(&[0, 0]).expect("valid index"), 1.0);
        assert_relative_eq!(extracted.get(&[0, 1]).expect("valid index"), 2.0);
        assert_relative_eq!(extracted.get(&[1, 0]).expect("valid index"), 3.0);
        assert_relative_eq!(extracted.get(&[1, 1]).expect("valid index"), 4.0);
    }

    #[test]
    fn test_extract_elements() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let extracted = extract_elements(&a, 3).expect("extract elements should succeed");
        assert_eq!(extracted.shape(), vec![3]);
        assert_relative_eq!(extracted.get(&[0]).expect("valid index"), 1.0);
        assert_relative_eq!(extracted.get(&[1]).expect("valid index"), 2.0);
        assert_relative_eq!(extracted.get(&[2]).expect("valid index"), 3.0);
    }

    #[test]
    fn test_randomized_range_finder_shape() {
        let a = Array::from_vec((0..100).map(|x| x as f64).collect()).reshape(&[10, 10]);

        let q = randomized_range_finder(&a, 5, 0).expect("range finder should succeed");
        assert_eq!(q.shape(), vec![10, 5]);
    }

    #[test]
    fn test_random_projection_gaussian() {
        let a = Array::from_vec((0..100).map(|x| x as f64).collect()).reshape(&[10, 10]);

        let projected =
            random_projection(&a, 5, ProjectionType::Gaussian).expect("projection should succeed");
        assert_eq!(projected.shape(), vec![10, 5]);
    }

    #[test]
    fn test_random_projection_sparse() {
        let a = Array::from_vec((0..100).map(|x| x as f64).collect()).reshape(&[10, 10]);

        let projected =
            random_projection(&a, 5, ProjectionType::Sparse).expect("projection should succeed");
        assert_eq!(projected.shape(), vec![10, 5]);
    }

    #[test]
    fn test_random_projection_rademacher() {
        let a = Array::from_vec((0..100).map(|x| x as f64).collect()).reshape(&[10, 10]);

        let projected = random_projection(&a, 5, ProjectionType::Rademacher)
            .expect("projection should succeed");
        assert_eq!(projected.shape(), vec![10, 5]);
    }
}
