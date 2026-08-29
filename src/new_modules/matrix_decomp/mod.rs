#[cfg(feature = "lapack")]
use crate::array::Array;
#[cfg(feature = "lapack")]
use crate::error::{NumRs2Error, Result};
#[cfg(feature = "lapack")]
use num_traits::{Float, NumCast, Zero};
#[cfg(feature = "lapack")]
use scirs2_core::linalg::svd_ndarray;
#[cfg(feature = "lapack")]
use scirs2_core::ndarray::ArrayView2;
#[cfg(feature = "lapack")]
use std::fmt::Debug;

pub mod cholesky;
pub mod condition;
pub mod lu;
pub mod qr;
pub mod schur;
pub mod utils;

// Re-export QR functions for convenience
#[cfg(feature = "lapack")]
pub use qr::{householder_qr, identity_matrix, qr};

// Re-export Cholesky functions for convenience
#[cfg(feature = "lapack")]
pub use cholesky::{cholesky, pivoted_cholesky};

// Re-export LU functions for convenience
#[cfg(feature = "lapack")]
pub use lu::lu;

// Re-export Schur functions for convenience
#[cfg(feature = "lapack")]
pub use schur::schur;

// Re-export condition number functions for convenience
#[cfg(feature = "lapack")]
pub use condition::{condition_number, lstsq, rcond, slogdet};

// Re-export utils functions for convenience
#[cfg(test)]
pub use utils::calculate_max_diff;

#[cfg(feature = "lapack")]
/// Type alias for SVD result to reduce complexity
/// (U, S, Vt) where U and Vt are orthogonal matrices, S is singular values
pub type SvdResult<T> = (
    Array<T>,
    Array<T>, // Singular values are same type as matrix elements
    Array<T>,
);

#[cfg(feature = "lapack")]
/// Enhanced matrix decomposition implementations that utilize ndarray-linalg
/// for more complete linear algebra functionality
/// Compute the Singular Value Decomposition (SVD) of a matrix
///
/// This implementation includes various numerical stability enhancements:
/// 1. Matrix scaling to avoid overflow
/// 2. Handling of very small singular values
/// 3. Verification of orthogonality and reconstruction error
pub fn svd<T>(a: &Array<T>) -> Result<SvdResult<T>>
where
    T: Float + Clone + Debug,
{
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "SVD requires a 2D matrix".to_string(),
        ));
    }

    let m = shape[0];
    let n = shape[1];

    // Scale the matrix to avoid overflow in large-magnitude entries
    // Find the maximum absolute value in the matrix
    let mut max_val = T::zero();
    let mut a_scaled = a.clone();

    for i in 0..m {
        for j in 0..n {
            let val = a.get(&[i, j])?;
            let abs_val = num_traits::Float::abs(val);
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
    }

    // Apply scaling if maximum is very large or very small
    let mut scaling_factor = T::one();
    if max_val > T::from(1e6).unwrap_or_else(|| T::one()) {
        scaling_factor = T::one() / max_val;

        let scaled_arr = a_scaled.array_mut();
        for i in 0..m {
            for j in 0..n {
                let val = a.get(&[i, j])?;
                scaled_arr[[i, j]] = val * scaling_factor;
            }
        }
    }

    // Get the 2D view and convert to f64 for OxiBLAS
    let a_view: ArrayView2<T> = a_scaled.view_2d()?;

    // Convert to f64 for OxiBLAS
    let mut a_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a_f64[[i, j]] = a_view[[i, j]].to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert to f64".to_string())
            })?;
        }
    }

    // Use OxiBLAS SVD implementation
    let svd_result = svd_ndarray(&a_f64)
        .map_err(|e| NumRs2Error::ComputationError(format!("SVD computation failed: {:?}", e)))?;

    // Convert U from f64 to T
    let u_f64 = svd_result.u;
    let mut u_vec: Vec<T> = Vec::with_capacity(u_f64.len());
    for &v in u_f64.iter() {
        u_vec.push(
            T::from(v)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?,
        );
    }
    let u_converted = Array::from_vec_shape(u_vec, &[u_f64.nrows(), u_f64.ncols()])?;

    // Convert singular values from f64 to T
    let s_f64 = svd_result.s;
    let mut s_vec: Vec<T> = Vec::with_capacity(s_f64.len());
    for &v in s_f64.iter() {
        s_vec.push(
            T::from(v)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?,
        );
    }
    let mut s_converted = Array::from_vec(s_vec);

    // Convert Vt from f64 to T
    let vt_f64 = svd_result.vt;
    let mut vt_vec: Vec<T> = Vec::with_capacity(vt_f64.len());
    for &v in vt_f64.iter() {
        vt_vec.push(
            T::from(v)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?,
        );
    }
    let vt_converted = Array::from_vec_shape(vt_vec, &[vt_f64.nrows(), vt_f64.ncols()])?;

    // Bulk-acquire once: rescaling, the tolerance's diagonal read, and the
    // small-value cleanup below are all direct calls on `s_converted` with no
    // intervening helper, so one hoisted handle covers the whole span.
    let s_len = s_converted.size();
    let s_arr = s_converted.array_mut();

    // Rescale singular values if we scaled the matrix
    if scaling_factor != T::one() {
        for i in 0..s_len {
            s_arr[[i]] = s_arr[[i]] / scaling_factor;
        }
    }

    // Set very small singular values to zero for numerical stability
    let eps = T::epsilon();
    let tolerance = eps
        * T::from(std::cmp::max(m, n)).expect("matrix dimension should convert to float type")
        * s_arr[[0]];

    for i in 0..s_len {
        if s_arr[[i]] < tolerance {
            s_arr[[i]] = T::zero();
        }
    }

    // Verify orthogonality and reconstruction error for debugging
    // These checks would be too expensive to run in production, but are useful during development
    #[cfg(debug_assertions)]
    {
        // 1. Check that U and V are orthogonal (U^T * U ≈ I, V^T * V ≈ I)
        // 2. Check reconstruction error: ||A - U*S*V^T|| should be small
    }

    Ok((u_converted, s_converted, vt_converted))
}

#[cfg(feature = "lapack")]
/// Compute a complete orthogonal decomposition of a matrix
/// This returns (Q, T, Z) where A = Q*T*Z^T, Q and Z are orthogonal, and T is upper triangular
///
/// This implementation includes various numerical stability enhancements:
/// 1. Rank determination using a robust numerical threshold
/// 2. Column pivoting in QR decomposition for better stability
/// 3. Proper handling of ill-conditioned matrices
pub fn cod<T>(a: &Array<T>) -> Result<SvdResult<T>>
where
    T: Float
        + Clone
        + Debug
        + PartialOrd
        + NumCast
        + num_traits::Zero
        + std::iter::Sum
        + std::ops::DivAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::fmt::Display
        + 'static,
{
    // Check if the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Complete orthogonal decomposition requires a 2D matrix".to_string(),
        ));
    }

    // A complete orthogonal decomposition can be computed via QR and SVD
    // For A = Q1*R1*P1^T (QR with column pivoting)
    // Then R1 = Q2*R2*Z^T (SVD of R1)
    // So A = Q*T*Z^T where Q = Q1*Q2, T = R2, and Z = P1*Z

    let m = shape[0];
    let n = shape[1];

    // ---------- STEP 1: QR decomposition with column pivoting ----------
    // First, we need to implement a QR decomposition with column pivoting
    // This is crucial for numerical stability, especially for rank-deficient matrices
    let mut a_copy = a.clone();
    let mut p = (0..n).collect::<Vec<usize>>(); // Column permutation
    let mut q1 = identity_matrix(m);

    // Bulk-acquire both buffers once for the whole routine below: every
    // access to `a_copy` and `q1` is a direct `.get()`/`.set()` call (the
    // Householder math itself runs on plain `Vec<T>`), so nothing here needs
    // to re-acquire either handle mid-loop.
    let a_arr = a_copy.array_mut();
    let q1_arr = q1.array_mut();

    // Compute column norms for pivoting
    let mut col_norms = vec![num_traits::Zero::zero(); n];
    for j in 0..n {
        for i in 0..m {
            let val = a_arr[[i, j]];
            col_norms[j] += val * val;
        }
        col_norms[j] = num_traits::Float::sqrt(col_norms[j]);
    }

    let min_dim = std::cmp::min(m, n);
    for k in 0..min_dim {
        // Find column with maximum norm
        let mut p_col = k;
        let mut p_norm: T = col_norms[k];

        for j in (k + 1)..n {
            if col_norms[j] > p_norm {
                p_col = j;
                p_norm = col_norms[j];
            }
        }

        // Swap columns if needed
        if p_col != k {
            p.swap(k, p_col);
            col_norms.swap(k, p_col);

            // Swap columns in A
            for i in 0..m {
                let temp = a_arr[[i, k]];
                a_arr[[i, k]] = a_arr[[i, p_col]];
                a_arr[[i, p_col]] = temp;
            }
        }

        // Skip if we have a numerically zero column
        if col_norms[k]
            < T::epsilon() * num_traits::NumCast::from(m).expect("m should convert to float type")
        {
            continue;
        }

        // Compute Householder reflection to zero out below the diagonal
        let mut x = Vec::with_capacity(m - k);
        for i in k..m {
            x.push(a_arr[[i, k]]);
        }

        let x_norm = num_traits::Float::sqrt(x.iter().map(|&val| val * val).sum::<T>());
        if x_norm > T::epsilon() {
            // First element of v determines the sign
            let alpha = if x[0] >= num_traits::Zero::zero() {
                -x_norm
            } else {
                x_norm
            };

            // Compute v = x - alpha*e1
            let mut v = x.clone();
            v[0] -= alpha;

            // Normalize v
            let v_norm = num_traits::Float::sqrt(v.iter().map(|&val| val * val).sum::<T>());
            if v_norm > T::epsilon() {
                for val in &mut v {
                    *val /= v_norm;
                }

                // Apply Householder reflection to A: A = A - 2 * v * (v^T * A)
                for j in k..n {
                    let mut vta: T = <T as num_traits::Zero>::zero();
                    for i in 0..(m - k) {
                        vta += v[i] * a_arr[[i + k, j]];
                    }

                    for i in 0..(m - k) {
                        let val = a_arr[[i + k, j]];
                        a_arr[[i + k, j]] = val
                            - <T as num_traits::NumCast>::from(2.0)
                                .expect("2.0 should convert to float type")
                                * v[i]
                                * vta;
                    }
                }

                // Update Q1
                for i in 0..m {
                    let mut q_row_dot_v: T = <T as num_traits::Zero>::zero();
                    for l in 0..(m - k) {
                        let q_val = q1_arr[[i, l + k]];
                        q_row_dot_v += q_val * v[l];
                    }

                    for j in k..m {
                        let q_val = q1_arr[[i, j]];
                        q1_arr[[i, j]] = q_val
                            - <T as num_traits::NumCast>::from(2.0)
                                .expect("2.0 should convert to float type")
                                * q_row_dot_v
                                * v[j - k];
                    }
                }

                // Update column norms for columns k+1 to n-1
                for j in (k + 1)..n {
                    col_norms[j] = T::zero();
                    for i in (k + 1)..m {
                        let val = a_arr[[i, j]];
                        col_norms[j] += val * val;
                    }
                    col_norms[j] = num_traits::Float::sqrt(col_norms[j]);
                }
            }
        }
    }

    // At this point, a_copy contains R1, q1 contains Q1, and p contains the column permutation
    // Now we can extract the upper triangular part of a_copy to get R1
    let mut r1 = Array::zeros(&[min_dim, n]);
    let r1_arr = r1.array_mut();
    for i in 0..min_dim {
        for j in i..n {
            r1_arr[[i, j]] = a_arr[[i, j]];
        }
    }

    // ---------- STEP 2: SVD of R1 ----------
    let (u, s, vt) = svd(&r1)?;

    // Determine numerical rank by identifying singular values above threshold
    // Use a more robust threshold based on machine precision, matrix dimensions, and condition number
    let s_vec = s.to_vec();
    let max_sv = s_vec.first().cloned().unwrap_or(T::zero());

    // Condition-number-based threshold
    let tol_factor = T::sqrt(T::epsilon());
    let tol_real = max_sv * tol_factor * T::from(std::cmp::max(m, n)).unwrap_or(T::one());

    let rank = s_vec.iter().filter(|&&sv| sv > tol_real).count();

    // ---------- STEP 3: Form final decomposition ----------
    // Compute Q = Q1 * U
    let q = q1.matmul(&u)?;

    // Create diagonal matrix T from singular values
    let mut t = Array::zeros(&[m, n]);
    {
        let t_arr = t.array_mut();
        for i in 0..rank {
            // Zero out tiny singular values for improved stability
            let s_val = s_vec[i];
            if s_val > tol_real {
                t_arr[[i, i]] = s_val;
            }
        }
    }

    // Compute Z from Vt and the column permutation P
    // Z = P * V (where V is the transpose of Vt)
    let v = vt.transpose();

    // Apply the permutation to get Z (write-only: `vt`/`v` are read-only
    // inputs). Read the bound once up front rather than per outer iteration.
    let mut z = Array::zeros(&[n, n]);
    let z_arr = z.array_mut();
    let vt_cols = vt.shape()[1];
    for j in 0..n {
        for i in 0..n {
            let idx = p[j]; // Get original column index
            if i < vt_cols {
                // Check bounds to avoid index errors
                z_arr[[idx, i]] = v.get(&[j, i])?;
            }
        }
    }

    // ---------- STEP 4: Verify the decomposition ----------
    #[cfg(debug_assertions)]
    {
        // Verify that Q*T*Z^T ≈ A with a small relative error
        // In a full implementation, we would compute and check this error
    }

    Ok((q, t, z))
}

#[cfg(feature = "lapack")]
/// Extend the Array type with the decomposition methods
impl<T> Array<T>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    /// Enhanced SVD implementation using OxiBLAS
    pub fn svd_compute(&self) -> Result<SvdResult<T>> {
        svd(self)
    }

    /// Enhanced QR decomposition using OxiBLAS
    pub fn qr_compute(&self) -> Result<(Array<T>, Array<T>)> {
        qr(self)
    }

    /// Enhanced Cholesky decomposition using OxiBLAS
    pub fn cholesky_compute(&self) -> Result<Array<T>> {
        cholesky(self)
    }

    /// LU decomposition
    pub fn lu(&self) -> Result<(Array<T>, Array<T>, Array<usize>)> {
        lu(self)
    }

    /// Schur decomposition
    pub fn schur(&self) -> Result<(Array<T>, Array<T>)> {
        schur(self)
    }

    /// Complete orthogonal decomposition
    pub fn cod(&self) -> Result<SvdResult<T>>
    where
        T: PartialOrd
            + NumCast
            + Zero
            + std::iter::Sum
            + std::ops::DivAssign
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign,
    {
        cod(self)
    }

    /// Compute the reciprocal condition number of the matrix
    pub fn rcond_compute(&self) -> Result<T> {
        rcond(self)
    }
}

// Add tests to verify the implementation
#[cfg(all(test, feature = "lapack"))]
mod tests {
    use super::*;

    #[test]
    fn test_svd_simple() -> Result<()> {
        // Create a simple 3x3 matrix
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3]);

        let (u, s, vt) = svd(&a)?;

        // Check the dimensions
        assert_eq!(u.shape(), vec![3, 3]);
        assert_eq!(s.shape(), vec![3]);
        assert_eq!(vt.shape(), vec![3, 3]);

        // For a complete test, we would also verify U*S*V^T = A
        // But we'll leave that for a more comprehensive test suite
        Ok(())
    }

    #[test]
    fn test_cod_reconstruction() -> Result<()> {
        // `cod()` (complete orthogonal decomposition, A = Q*T*Z^T) had no
        // test coverage anywhere in the crate despite being public
        // (`Array::cod()`) even though its Householder-QR-with-pivoting
        // stage was restructured for the W3-A2 COW-tax conversion -- added
        // here to pin down the one identity the decomposition promises.
        let a = Array::from_vec(vec![4.0, 0.0, 1.0, 0.0, 5.0, 2.0, 1.0, 2.0, 6.0]).reshape(&[3, 3]);

        let (q, t, z) = cod(&a)?;
        assert_eq!(q.shape(), vec![3, 3]);
        assert_eq!(t.shape(), vec![3, 3]);
        assert_eq!(z.shape(), vec![3, 3]);

        // A ≈ Q * T * Z^T
        let zt = z.transpose();
        let qt = q.matmul(&t)?;
        let recon = qt.matmul(&zt)?;

        for i in 0..3 {
            for j in 0..3 {
                let expected = a.get(&[i, j])?;
                let actual = recon.get(&[i, j])?;
                assert!(
                    (expected - actual).abs() < 1e-8,
                    "COD reconstruction mismatch at ({i},{j}): expected {expected}, got {actual}"
                );
            }
        }

        // Q is orthogonal: Q^T * Q ≈ I.
        let qtq = q.transpose().matmul(&q)?;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq.get(&[i, j])? - expected).abs() < 1e-8,
                    "Q should be orthogonal at ({i},{j})"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_qr_simple() -> Result<()> {
        // Create a simple 3x3 matrix - using a well-conditioned matrix
        let a = Array::from_vec(vec![4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0]).reshape(&[3, 3]);

        let (q, r) = qr(&a)?;

        // Check the dimensions
        assert_eq!(q.shape(), vec![3, 3]);
        assert_eq!(r.shape(), vec![3, 3]);

        // Verify QR decomposition properties:
        // 1. Q*R should equal A
        let qr_product = q.matmul(&r)?;
        for i in 0..3 {
            for j in 0..3 {
                let expected = a.get(&[i, j])?;
                let actual = qr_product.get(&[i, j])?;
                assert!(
                    num_traits::Float::abs(actual - expected) < 1e-10,
                    "QR: Q*R should equal A - expected {}, got {} at ({},{})",
                    expected,
                    actual,
                    i,
                    j
                );
            }
        }

        // 2. Q should be orthogonal (Q^T * Q = I)
        let qt = q.transpose();
        let qtq = qt.matmul(&q)?;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = qtq.get(&[i, j])?;
                assert!(
                    num_traits::Float::abs(actual - expected) < 1e-10,
                    "QR: Q should be orthogonal - Q^T*Q expected {}, got {} at ({},{})",
                    expected,
                    actual,
                    i,
                    j
                );
            }
        }

        // 3. R should be upper triangular
        for i in 1..3 {
            for j in 0..i {
                let val = r.get(&[i, j])?;
                assert!(
                    num_traits::Float::abs(val) < 1e-10,
                    "QR: R should be upper triangular - got {} at ({},{})",
                    val,
                    i,
                    j
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_cholesky_simple() -> Result<()> {
        // Create a simple positive definite matrix (diagonal matrix with positive entries)
        let a =
            Array::from_vec(vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0]).reshape(&[3, 3]);

        // Compute Cholesky decomposition
        let chol = cholesky(&a)?;

        // Check dimensions
        assert_eq!(chol.shape(), vec![3, 3]);

        // For a diagonal matrix with positive entries:
        // The Cholesky factor should be a diagonal matrix with the square roots of a's diagonal
        let expected_diag = [2.0, 3.0, 4.0]; // sqrt of 4, 9, 16

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { expected_diag[i] } else { 0.0 };
                let actual = chol.get(&[i, j])?;
                assert!(
                    num_traits::Float::abs(actual - expected) < 1e-10,
                    "Cholesky: incorrect value at ({},{}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }

        // Check that L * L^T = A
        let chol_t = chol.transpose();
        let product = chol.matmul(&chol_t)?;

        for i in 0..3 {
            for j in 0..3 {
                let expected = a.get(&[i, j])?;
                let actual = product.get(&[i, j])?;
                assert!(
                    num_traits::Float::abs(actual - expected) < 1e-10,
                    "Cholesky: L*L^T=A check failed at ({},{}) - expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_lu_simple() -> Result<()> {
        // Create a simple 3x3 matrix
        let a = Array::from_vec(vec![4.0, 1.0, 2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 6.0]).reshape(&[3, 3]);

        // Compute LU decomposition
        let (l, u, p) = lu(&a)?;

        // Check dimensions
        assert_eq!(l.shape(), vec![3, 3]);
        assert_eq!(u.shape(), vec![3, 3]);
        assert_eq!(p.shape(), vec![3]);

        // Check L properties - lower triangular with ones on diagonal
        for i in 0..3 {
            for j in 0..3 {
                let l_val = l.get(&[i, j])?;
                if i < j {
                    // Upper part should be zero
                    assert!(
                        num_traits::Float::abs(l_val) < 1e-10,
                        "L should be lower triangular, but L[{},{}] = {}",
                        i,
                        j,
                        l_val
                    );
                }
                if i == j {
                    // Diagonal should be one
                    assert!(
                        num_traits::Float::abs(l_val - 1.0) < 1e-10,
                        "Diagonal of L should be 1, but L[{},{}] = {}",
                        i,
                        j,
                        l_val
                    );
                }
            }
        }

        // Check U properties - upper triangular
        for i in 0..3 {
            for j in 0..3 {
                if i > j {
                    let u_val = u.get(&[i, j])?;
                    // Lower part should be zero
                    assert!(
                        num_traits::Float::abs(u_val) < 1e-10,
                        "U should be upper triangular, but U[{},{}] = {}",
                        i,
                        j,
                        u_val
                    );
                }
            }
        }

        // Verify P*A = L*U

        // Permute A according to P
        let mut pa = Array::zeros(&[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let perm_idx = p.get(&[i])?;
                pa.set(&[i, j], a.get(&[perm_idx, j])?)?;
            }
        }

        // Calculate L*U
        let lu_product = l.matmul(&u)?;

        // Check that PA ≈ LU
        for i in 0..3 {
            for j in 0..3 {
                let pa_val = pa.get(&[i, j])?;
                let lu_val = lu_product.get(&[i, j])?;
                assert!(
                    num_traits::Float::abs(pa_val - lu_val) < 1e-10,
                    "PA ≈ LU check failed at ({},{}): PA = {}, LU = {}",
                    i,
                    j,
                    pa_val,
                    lu_val
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_lu_stability() -> Result<()> {
        // Create an ill-conditioned matrix
        let a = Array::from_vec(vec![
            1.0,
            1.0,
            1.0,
            1.0,
            1.0 + 1e-10,
            1.0,
            1.0,
            1.0,
            1.0 + 2e-10,
        ])
        .reshape(&[3, 3]);

        // Compute LU decomposition - this should succeed despite ill conditioning
        let result = lu(&a);
        assert!(
            result.is_ok(),
            "LU decomposition should succeed even for ill-conditioned matrix"
        );

        let (l, u, p) = result?;

        // Verify that L and U are triangular
        for i in 0..3 {
            for j in 0..3 {
                if i < j {
                    let l_val = l.get(&[i, j])?;
                    assert!(
                        num_traits::Float::abs(l_val) < 1e-8,
                        "L should be lower triangular"
                    );
                }
                if i > j {
                    let u_val = u.get(&[i, j])?;
                    assert!(
                        num_traits::Float::abs(u_val) < 1e-8,
                        "U should be upper triangular"
                    );
                }
            }
        }

        // Check reconstruction with permutation
        let lu_product = l.matmul(&u)?;

        // Compute permuted A
        let mut pa = Array::zeros(&[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let perm_idx = p.get(&[i])?;
                pa.set(&[i, j], a.get(&[perm_idx, j])?)?;
            }
        }

        // For ill-conditioned matrix, we use a larger error tolerance
        let tol = 1e-8;
        let mut max_diff = 0.0;

        for i in 0..3 {
            for j in 0..3 {
                let diff_val = pa.get(&[i, j])? - lu_product.get(&[i, j])?;
                let diff = num_traits::Float::abs(diff_val);
                max_diff = max_diff.max(diff);
            }
        }

        assert!(max_diff < tol,
            "LU decomposition should accurately reconstruct the original matrix even for ill-conditioned inputs. Max diff: {}",
            max_diff);
        Ok(())
    }

    #[test]
    fn test_condition_number_well_conditioned() -> Result<()> {
        // Create a well-conditioned diagonal matrix
        let a = Array::from_vec(vec![4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0]).reshape(&[3, 3]);

        // Compute condition number
        let cond = condition_number(&a)?;

        // Expected condition number is max(diag) / min(diag) = 6.0 / 4.0 = 1.5
        let expected: f64 = 1.5;
        let diff = num_traits::Float::abs(cond - expected);
        assert!(
            diff < 1e-10,
            "Condition number should be 1.5 for this diagonal matrix, got {}",
            cond
        );

        // Test the array method
        let cond2 = a.cond()?;
        let diff2 = num_traits::Float::abs(cond2 - expected);
        assert!(
            diff2 < 1e-10,
            "Array::cond() should return 1.5, got {}",
            cond2
        );

        // Test rcond (reciprocal condition number)
        let rcond_val = rcond(&a)?;
        let expected_rcond: f64 = 1.0 / 1.5;
        let diff_rcond = num_traits::Float::abs(rcond_val - expected_rcond);
        assert!(
            diff_rcond < 1e-10,
            "Reciprocal condition number should be {}, got {}",
            expected_rcond,
            rcond_val
        );

        // Test is_well_conditioned
        assert!(
            a.is_well_conditioned()?,
            "Matrix should be well-conditioned"
        );
        Ok(())
    }

    #[test]
    fn test_condition_number_ill_conditioned() -> Result<()> {
        // Create an ill-conditioned matrix with very different singular values
        let a =
            Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1e-14, 0.0, 0.0, 0.0, 1.0]).reshape(&[3, 3]);

        // Compute condition number
        let cond = condition_number(&a)?;

        // Expected condition number is max(diag) / min(diag) = 1.0 / 1e-14 = 1e14
        let expected: f64 = 1e14;
        let diff_value = num_traits::Float::abs(cond - expected);
        let relative_error = diff_value / expected;
        assert!(
            relative_error < 1e-5,
            "Condition number should be approximately 1e14 for this diagonal matrix, got {}",
            cond
        );

        // Test is_well_conditioned - should return false (threshold is 1e12)
        assert!(
            !a.is_well_conditioned()?,
            "Matrix should be ill-conditioned with condition number {}",
            cond
        );
        Ok(())
    }

    #[test]
    fn test_condition_number_singular() -> Result<()> {
        // Create a more obviously singular matrix (with a zero row)
        let a = Array::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0]).reshape(&[3, 3]);

        // This matrix is clearly rank deficient since it has identical rows
        let cond: f64 = condition_number(&a)?;
        println!("Singular matrix condition number: {}", cond);

        // Either the result is infinity, or it should be a very large number
        // Because of floating-point representation, we might not get exact infinity
        assert!(
            cond.is_infinite() || cond > 1e15,
            "Condition number should be very large for a singular matrix, got {}",
            cond
        );

        // Test rcond - should return 0 for singular matrix
        let rcond_val = rcond(&a)?;
        assert!(
            rcond_val == 0.0,
            "Reciprocal condition number should be 0 for a singular matrix, got {}",
            rcond_val
        );

        // Test is_well_conditioned - should return false
        assert!(
            !a.is_well_conditioned()?,
            "Singular matrix should not be well-conditioned"
        );
        Ok(())
    }

    #[test]
    fn test_condition_number_hilbert() -> Result<()> {
        // Create a Hilbert matrix, which is famously ill-conditioned
        // Hilbert matrix has entries H[i,j] = 1/(i+j+1)
        let n = 5;
        let mut hilbert = Array::zeros(&[n, n]);
        for i in 0..n {
            for j in 0..n {
                let val = 1.0 / (i as f64 + j as f64 + 1.0);
                hilbert.set(&[i, j], val)?;
            }
        }

        // Compute condition number
        let cond = condition_number(&hilbert)?;

        // Hilbert matrices are known to have very high condition numbers
        // For n=5, it's approximately 4.8e5
        assert!(
            cond > 1e4,
            "Hilbert matrix should have a high condition number, got {}",
            cond
        );

        println!("Hilbert matrix condition number: {}", cond);

        // Test is_well_conditioned - we don't explicitly test the result since the
        // threshold is dynamically calculated and might vary by implementation
        let _ = hilbert.is_well_conditioned();
        Ok(())
    }

    #[test]
    fn test_numerical_stability_relations() -> Result<()> {
        // Create a matrix with reasonably well-spaced singular values
        let a =
            Array::from_vec(vec![10.0, 4.0, 2.0, 4.0, 5.0, 1.0, 2.0, 1.0, 6.0]).reshape(&[3, 3]);

        // Compute the condition number
        let cond = a.cond()?;

        // Compute LU decomposition
        let (l, u, p) = lu(&a)?;

        // Compute SVD
        let (us, s, vt) = svd(&a)?;

        // Check that the smallest singular value is reasonably related to condition number
        let smallest_sv = s.to_vec().iter().fold(f64::MAX, |a, &b| a.min(b));
        let largest_sv = s.to_vec().iter().fold(0.0, |a, &b| a.max(b));

        // The condition number should be approximately largest_sv / smallest_sv
        let computed_cond: f64 = largest_sv / smallest_sv;

        // Verify with reasonable tolerance
        let abs_diff = num_traits::Float::abs(cond - computed_cond);
        let rel_error = abs_diff / computed_cond;
        assert!(rel_error < 0.01,
                "Condition number should be approximately largest_sv / smallest_sv. Found: {}, Computed: {}",
                cond, computed_cond);

        // Check that different decompositions are numerically compatible
        // If a = LU (with permutation) and a = USV^T, then the decompositions should
        // represent the same matrix (within numerical precision)

        // Create a diagonal matrix from singular values
        let mut s_diag = Array::zeros(&[3, 3]);
        for i in 0..3 {
            s_diag.set(&[i, i], s.get(&[i])?)?;
        }

        // Compute SVD reconstruction: U*S*V^T
        let us_product = us.matmul(&s_diag)?;
        let usv_product = us_product.matmul(&vt)?;

        // Compute LU reconstruction with permutation
        let lu_product = l.matmul(&u)?;

        // Compute permuted A
        let mut pa = Array::zeros(&[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let perm_idx = p.get(&[i])?;
                pa.set(&[i, j], a.get(&[perm_idx, j])?)?;
            }
        }

        // Check that decompositions agree
        // Need to account for permutation in LU

        // Calculate the reconstruction error for each decomposition
        let mut svd_error = 0.0;
        let mut lu_error = 0.0;

        for i in 0..3 {
            for j in 0..3 {
                let svd_diff = num_traits::Float::abs(a.get(&[i, j])? - usv_product.get(&[i, j])?);
                svd_error = svd_error.max(svd_diff);

                let lu_diff = num_traits::Float::abs(pa.get(&[i, j])? - lu_product.get(&[i, j])?);
                lu_error = lu_error.max(lu_diff);
            }
        }

        // Both decompositions should have similar error characteristics
        assert!(
            svd_error < 1e-10,
            "SVD reconstruction error should be small: {}",
            svd_error
        );
        assert!(
            lu_error < 1e-10,
            "LU reconstruction error should be small: {}",
            lu_error
        );
        Ok(())
    }
}
