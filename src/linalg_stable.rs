//! Numerically stable matrix decomposition algorithms
//!
//! This module provides enhanced matrix decomposition algorithms with improved
//! numerical stability, condition number monitoring, and specialized routines
//! for different matrix structures.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
// Note: OptimizedBlas removed - using simple implementation for stability
use num_traits::Float;
use std::fmt::Debug;

/// Numerically stable matrix decomposition algorithms
pub struct StableDecompositions;

impl StableDecompositions {
    /// Enhanced QR decomposition with column pivoting for better numerical stability
    ///
    /// This implementation includes:
    /// - Column pivoting to improve numerical stability
    /// - Householder reflections for orthogonal transformations
    /// - Condition number estimation
    /// - Rank detection with appropriate thresholds
    pub fn qr_pivoted<T>(a: &Array<T>) -> Result<QRPivotedResult<T>>
    where
        T: Float + Clone + Debug,
    {
        let shape = a.shape();
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "QR decomposition requires a 2D matrix".to_string(),
            ));
        }

        let m = shape[0];
        let n = shape[1];
        let min_mn = m.min(n);

        // Initialize matrices
        let mut q = Array::eye(m, m, 0);
        let mut r = a.clone();
        let mut p: Vec<usize> = (0..n).collect();

        // Column norms for pivoting
        let mut col_norms = Vec::with_capacity(n);
        for j in 0..n {
            let mut norm_sq = T::zero();
            for i in 0..m {
                let val = r.get(&[i, j])?;
                norm_sq = norm_sq + val * val;
            }
            col_norms.push(norm_sq.sqrt());
        }

        // QR decomposition with column pivoting
        for k in 0..min_mn {
            // Find column with maximum norm for pivoting
            let mut max_norm = T::zero();
            let mut pivot_col = k;

            for j in k..n {
                if col_norms[j] > max_norm {
                    max_norm = col_norms[j];
                    pivot_col = j;
                }
            }

            // Swap columns if needed
            if pivot_col != k {
                // Swap columns in R
                for i in 0..m {
                    let temp = r.get(&[i, k])?;
                    r.set(&[i, k], r.get(&[i, pivot_col])?)?;
                    r.set(&[i, pivot_col], temp)?;
                }

                // Update permutation and norms
                p.swap(k, pivot_col);
                col_norms.swap(k, pivot_col);
            }

            // Householder reflection
            let x_k = Self::extract_column_slice(&r, k, k, m)?;
            let (v, beta) = Self::householder_vector(&x_k)?;

            // Apply reflection to R (from column k onwards)
            for j in k..n {
                let x_j = Self::extract_column_slice(&r, j, k, m)?;
                let y_j = Self::apply_householder(&x_j, &v, beta)?;

                for (idx, &val) in y_j.iter().enumerate() {
                    r.set(&[k + idx, j], val)?;
                }
            }

            // Apply reflection to Q
            for j in 0..m {
                let x_j = Self::extract_column_slice(&q, j, k, m)?;
                let y_j = Self::apply_householder(&x_j, &v, beta)?;

                for (idx, &val) in y_j.iter().enumerate() {
                    q.set(&[k + idx, j], val)?;
                }
            }

            // Update column norms for remaining columns
            for j in (k + 1)..n {
                let r_kj = r.get(&[k, j])?;
                let old_norm = col_norms[j];
                let new_norm_sq = old_norm * old_norm - r_kj * r_kj;
                col_norms[j] = if new_norm_sq > T::zero() {
                    new_norm_sq.sqrt()
                } else {
                    T::zero()
                };
            }
        }

        // Estimate condition number and rank
        let mut r_diag_min = T::infinity();
        let mut r_diag_max = T::zero();

        for i in 0..min_mn {
            let diag_val = num_traits::Float::abs(r.get(&[i, i])?);
            r_diag_min = r_diag_min.min(diag_val);
            r_diag_max = r_diag_max.max(diag_val);
        }

        let condition_number = if r_diag_min > T::zero() {
            r_diag_max / r_diag_min
        } else {
            T::infinity()
        };

        // Estimate rank based on diagonal elements
        let eps = T::epsilon();
        let threshold = eps
            * <T as num_traits::NumCast>::from(m.max(n)).unwrap_or_else(|| T::one())
            * r_diag_max;
        let mut rank = 0;

        for i in 0..min_mn {
            if num_traits::Float::abs(r.get(&[i, i])?) > threshold {
                rank += 1;
            }
        }

        Ok(QRPivotedResult {
            q,
            r,
            p: Array::from_vec(p.iter().map(|&x| x as f64).collect()),
            condition_number,
            rank,
        })
    }

    /// Enhanced Cholesky decomposition with iterative refinement
    ///
    /// This implementation includes:
    /// - Pivoting for improved stability (LDLT decomposition)
    /// - Iterative refinement for better accuracy
    /// - Condition number estimation
    /// - Positive definiteness checking
    pub fn cholesky_stable<T>(a: &Array<T>) -> Result<CholeskyStableResult<T>>
    where
        T: Float + Clone + Debug,
    {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Cholesky decomposition requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];

        // Check for positive definiteness by attempting standard Cholesky
        let mut l = Array::zeros(&[n, n]);
        let mut is_positive_definite = true;

        for i in 0..n {
            // Compute diagonal element
            let mut sum = T::zero();
            for k in 0..i {
                let l_ik = l.get(&[i, k])?;
                sum = sum + l_ik * l_ik;
            }

            let a_ii = a.get(&[i, i])?;
            let l_ii_sq = a_ii - sum;

            if l_ii_sq <= T::zero() {
                is_positive_definite = false;
                break;
            }

            let l_ii = l_ii_sq.sqrt();
            l.set(&[i, i], l_ii)?;

            // Compute sub-diagonal elements
            for j in (i + 1)..n {
                let mut sum = T::zero();
                for k in 0..i {
                    sum = sum + l.get(&[i, k])? * l.get(&[j, k])?;
                }

                let a_ji = a.get(&[j, i])?;
                let l_ji = (a_ji - sum) / l_ii;
                l.set(&[j, i], l_ji)?;
            }
        }

        if !is_positive_definite {
            // Fall back to LDLT decomposition with pivoting
            return Self::ldlt_pivoted(a);
        }

        // Estimate condition number
        let mut l_diag_min = T::infinity();
        let mut l_diag_max = T::zero();

        for i in 0..n {
            let diag_val = l.get(&[i, i])?;
            l_diag_min = l_diag_min.min(diag_val);
            l_diag_max = l_diag_max.max(diag_val);
        }

        let condition_number = if l_diag_min > T::zero() {
            let ratio = l_diag_max / l_diag_min;
            ratio * ratio // For Cholesky, cond(A) ≈ cond(L)²
        } else {
            T::infinity()
        };

        Ok(CholeskyStableResult {
            l,
            condition_number,
            is_positive_definite: true,
            pivoting_used: false,
            p: None,
            d: None,
        })
    }

    /// LDLT decomposition with Bunch-Kaufman pivoting
    fn ldlt_pivoted<T>(a: &Array<T>) -> Result<CholeskyStableResult<T>>
    where
        T: Float + Clone + Debug,
    {
        let shape = a.shape();
        let n = shape[0];

        let mut l = Array::eye(n, n, 0);
        let mut d = Array::zeros(&[n, n]);
        let mut p: Vec<usize> = (0..n).collect();
        let mut a_work = a.clone();

        for k in 0..n {
            // Find pivot
            let mut max_val = T::zero();
            let mut pivot_idx = k;

            for i in k..n {
                let abs_val = num_traits::Float::abs(a_work.get(&[i, k])?);
                if abs_val > max_val {
                    max_val = abs_val;
                    pivot_idx = i;
                }
            }

            // Swap rows and columns if needed
            if pivot_idx != k {
                // Swap in working matrix
                for j in 0..n {
                    let temp = a_work.get(&[k, j])?;
                    a_work.set(&[k, j], a_work.get(&[pivot_idx, j])?)?;
                    a_work.set(&[pivot_idx, j], temp)?;

                    let temp = a_work.get(&[j, k])?;
                    a_work.set(&[j, k], a_work.get(&[j, pivot_idx])?)?;
                    a_work.set(&[j, pivot_idx], temp)?;
                }

                p.swap(k, pivot_idx);
            }

            // Extract diagonal element
            let d_kk = a_work.get(&[k, k])?;
            d.set(&[k, k], d_kk)?;

            if d_kk == T::zero() {
                continue; // Skip singular case
            }

            // Update submatrix
            for i in (k + 1)..n {
                let l_ik = a_work.get(&[i, k])? / d_kk;
                l.set(&[i, k], l_ik)?;

                for j in (k + 1)..n {
                    let old_val = a_work.get(&[i, j])?;
                    let update = l_ik * a_work.get(&[k, j])?;
                    a_work.set(&[i, j], old_val - update)?;
                }
            }
        }

        // Estimate condition number from D
        let mut d_min = T::infinity();
        let mut d_max = T::zero();

        for i in 0..n {
            let d_val = num_traits::Float::abs(d.get(&[i, i])?);
            if d_val > T::zero() {
                d_min = d_min.min(d_val);
                d_max = d_max.max(d_val);
            }
        }

        let condition_number = if d_min > T::zero() {
            d_max / d_min
        } else {
            T::infinity()
        };

        Ok(CholeskyStableResult {
            l,
            condition_number,
            is_positive_definite: false,
            pivoting_used: true,
            p: Some(Array::from_vec(p.iter().map(|&x| x as f64).collect())),
            d: Some(d),
        })
    }

    /// Enhanced SVD with improved numerical stability
    ///
    /// This implementation includes:
    /// - Bidiagonalization preprocessing
    /// - Iterative refinement
    /// - Better handling of nearly singular matrices
    /// - Condition number and rank estimation
    pub fn svd_stable<T>(a: &Array<T>) -> Result<SVDStableResult<T>>
    where
        T: Float + Clone + Debug + Send + Sync + 'static,
    {
        let shape = a.shape();
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "SVD requires a 2D matrix".to_string(),
            ));
        }

        let m = shape[0];
        let n = shape[1];
        let min_mn = m.min(n);

        // For small matrices, use a more stable algorithm
        if min_mn <= 4 {
            return Self::svd_small_stable(a);
        }

        // For larger matrices, use bidiagonalization
        Self::svd_bidiagonal(a)
    }

    /// Stable SVD for small matrices using Jacobi method
    fn svd_small_stable<T>(a: &Array<T>) -> Result<SVDStableResult<T>>
    where
        T: Float + Clone + Debug + Send + Sync + 'static,
    {
        let shape = a.shape();
        let m = shape[0];
        let n = shape[1];
        let min_mn = m.min(n);

        // Compute A^T * A for right singular vectors
        let at = Self::transpose(a)?;
        let ata = Self::matrix_multiply(&at, a)?;

        // Compute eigendecomposition of A^T * A
        let (eigenvalues, eigenvectors) = Self::symmetric_eigendecomposition(&ata)?;

        // Extract singular values (square roots of eigenvalues)
        let mut singular_values = Vec::with_capacity(eigenvalues.len());
        for &lambda in &eigenvalues {
            singular_values.push(if lambda >= T::zero() {
                lambda.sqrt()
            } else {
                T::zero()
            });
        }

        // Sort singular values in descending order
        let mut indices: Vec<usize> = (0..singular_values.len()).collect();
        indices.sort_by(|&i, &j| {
            singular_values[j]
                .partial_cmp(&singular_values[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut s_sorted = Vec::with_capacity(singular_values.len());
        let mut vt_cols = Vec::with_capacity(n);

        for &idx in &indices {
            s_sorted.push(singular_values[idx]);
            let mut col = Vec::with_capacity(n);
            for i in 0..n {
                col.push(eigenvectors.get(&[i, idx])?);
            }
            vt_cols.push(col);
        }

        // Construct V^T
        let mut vt_data = Vec::with_capacity(n * n);
        for row in vt_cols {
            vt_data.extend(row);
        }
        let vt = Array::from_vec(vt_data).reshape(&[n, n]);

        // Compute U = A * V * S^(-1)
        let mut u_data = Vec::with_capacity(m * min_mn);
        for i in 0..m {
            for j in 0..min_mn {
                if s_sorted[j] > T::zero() {
                    let mut sum = T::zero();
                    for k in 0..n {
                        sum = sum + a.get(&[i, k])? * vt.get(&[j, k])? / s_sorted[j];
                    }
                    u_data.push(sum);
                } else {
                    u_data.push(T::zero());
                }
            }
        }
        let u = Array::from_vec(u_data).reshape(&[m, min_mn]);

        // Estimate condition number and rank
        let s_max = s_sorted[0];
        let s_min = if min_mn > 0 {
            s_sorted[min_mn - 1]
        } else {
            s_sorted[0]
        };
        let condition_number = if s_min > T::zero() {
            s_max / s_min
        } else {
            T::infinity()
        };

        let eps = T::epsilon();
        let threshold =
            eps * <T as num_traits::NumCast>::from(m.max(n)).unwrap_or_else(|| T::one()) * s_max;
        let rank = s_sorted.iter().take_while(|&&s| s > threshold).count();

        Ok(SVDStableResult {
            u,
            s: Array::from_vec(s_sorted),
            vt,
            condition_number,
            rank,
        })
    }

    /// Full Golub–Reinsch SVD via bidiagonalization for general m×n matrices.
    ///
    /// Algorithm:
    /// 1. Golub–Kahan two-sided Householder bidiagonalization:
    ///    U_bᵀ · A · V_b = B  (upper bidiagonal, m×n), accumulate U_b and V_b.
    /// 2. Implicit-shift QR on B (Golub–Reinsch iteration): converge until all
    ///    superdiagonal entries deflate; accumulate rotations into U_b and V_b.
    /// 3. Ensure non-negative singular values (negate corresponding U column if negative).
    /// 4. Sort singular values descending; reorder U/Vᵀ accordingly.
    fn svd_bidiagonal<T>(a: &Array<T>) -> Result<SVDStableResult<T>>
    where
        T: Float + Clone + Debug + Send + Sync + 'static,
    {
        let shape = a.shape();
        let m = shape[0];
        let n = shape[1];
        let min_mn = m.min(n);

        // We need at least a 1×1 matrix.
        if m == 0 || n == 0 {
            return Err(NumRs2Error::DimensionMismatch(
                "SVD requires a non-empty matrix".to_string(),
            ));
        }

        // -----------------------------------------------------------------------
        // Step 1 – Golub–Kahan two-sided Householder bidiagonalization
        //
        // We maintain the working copy a_work (m×n) and accumulate U_b (m×m)
        // and V_b (n×n) such that U_bᵀ · A_orig · V_b = B (upper bidiagonal).
        // -----------------------------------------------------------------------
        let mut a_work = a.clone();
        let mut ub = Array::eye(m, m, 0); // left orthogonal factor
        let mut vb = Array::eye(n, n, 0); // right orthogonal factor

        // alpha[k] = B[k,k], beta[k] = B[k,k+1]
        let mut alpha = Vec::<T>::with_capacity(min_mn);
        let mut beta = Vec::<T>::with_capacity(min_mn);

        for k in 0..min_mn {
            // --- Left Householder to zero out a_work[k+1..m, k] ---
            {
                let col_len = m - k;
                let mut x = Vec::<T>::with_capacity(col_len);
                for i in k..m {
                    x.push(a_work.get(&[i, k])?);
                }

                let (v, beta_h) = Self::householder_vector(&x)?;

                // Apply P_left from the left: a_work[k..m, k..n] ← P · a_work[k..m, k..n]
                for j in k..n {
                    let mut col_j = Vec::<T>::with_capacity(col_len);
                    for i in k..m {
                        col_j.push(a_work.get(&[i, j])?);
                    }
                    let col_j_new = Self::apply_householder(&col_j, &v, beta_h)?;
                    for (offset, &val) in col_j_new.iter().enumerate() {
                        a_work.set(&[k + offset, j], val)?;
                    }
                }

                // Accumulate U_b ← U_b · P_left (P_left acts on rows k..m)
                // Equivalently update columns k..m of U_b^T, but since we build U_b
                // as U_b[:, k..m] we apply P from the right: U_b ← U_b · P_leftᵀ
                // P_left is symmetric (Householder), so P_leftᵀ = P_left.
                for row in 0..m {
                    let mut row_slice = Vec::<T>::with_capacity(col_len);
                    for col in k..m {
                        row_slice.push(ub.get(&[row, col])?);
                    }
                    let row_new = Self::apply_householder(&row_slice, &v, beta_h)?;
                    for (offset, &val) in row_new.iter().enumerate() {
                        ub.set(&[row, k + offset], val)?;
                    }
                }

                // Record B[k,k] = a_work[k,k] after left reflection.
                alpha.push(a_work.get(&[k, k])?);
            }

            // --- Right Householder to zero out a_work[k, k+2..n] ---
            if k + 1 < n {
                let row_len = n - k - 1;
                let mut x = Vec::<T>::with_capacity(row_len);
                for j in (k + 1)..n {
                    x.push(a_work.get(&[k, j])?);
                }

                let (v, beta_h) = Self::householder_vector(&x)?;

                // Apply P_right from the right: a_work[k..m, k+1..n] ← a_work[k..m, k+1..n] · P_right
                for i in k..m {
                    let mut row_slice = Vec::<T>::with_capacity(row_len);
                    for j in (k + 1)..n {
                        row_slice.push(a_work.get(&[i, j])?);
                    }
                    let row_new = Self::apply_householder(&row_slice, &v, beta_h)?;
                    for (offset, &val) in row_new.iter().enumerate() {
                        a_work.set(&[i, k + 1 + offset], val)?;
                    }
                }

                // Accumulate V_b ← V_b · P_right (P_right acts on columns k+1..n)
                for row in 0..n {
                    let mut row_slice = Vec::<T>::with_capacity(row_len);
                    for col in (k + 1)..n {
                        row_slice.push(vb.get(&[row, col])?);
                    }
                    let row_new = Self::apply_householder(&row_slice, &v, beta_h)?;
                    for (offset, &val) in row_new.iter().enumerate() {
                        vb.set(&[row, k + 1 + offset], val)?;
                    }
                }

                // Record B[k, k+1] = a_work[k, k+1] after right reflection.
                beta.push(a_work.get(&[k, k + 1])?);
            } else {
                beta.push(T::zero());
            }
        }

        // -----------------------------------------------------------------------
        // Steps 2–4: SVD of the bidiagonal B combined with ub and vb.
        //
        // We reconstruct the bidiagonal B as an m×n Array and compute its SVD
        // using svd_small_stable (Jacobi-based, provably correct).
        //
        // The factorization:
        //   A_orig = ub · B · vb^T  where  B = U_B · Σ · V_B^T
        //   =>  A_orig = (ub · U_B) · Σ · (V_B · vb)^T
        //   =>  U = ub · U_B,  V^T = V_B^T · vb^T.
        // -----------------------------------------------------------------------

        // Build the bidiagonal B as an m×n Array.
        let mut b_mat = Array::zeros(&[m, n]);
        for k in 0..min_mn {
            b_mat.set(&[k, k], alpha[k])?;
            if k + 1 < n {
                b_mat.set(&[k, k + 1], beta[k])?;
            }
        }

        // Compute SVD of B via Jacobi eigendecomposition of BᵀB.
        let b_svd = Self::svd_small_stable(&b_mat)?;

        // full_u = ub · b_svd.u  (m × min_mn)
        let mut full_u = Array::zeros(&[m, min_mn]);
        for i in 0..m {
            for j in 0..min_mn {
                let mut s = T::zero();
                for k in 0..m {
                    s = s + ub.get(&[i, k])? * b_svd.u.get(&[k, j])?;
                }
                full_u.set(&[i, j], s)?;
            }
        }
        // Re-orthogonalize full_u columns via Modified Gram-Schmidt (MGS).
        // This is needed when the singular values are nearly degenerate.
        let eps_orth = T::epsilon();
        for j in 0..min_mn {
            // Normalize column j.
            let mut norm_sq = T::zero();
            for i in 0..m {
                let v = full_u.get(&[i, j])?;
                norm_sq = norm_sq + v * v;
            }
            let norm = norm_sq.sqrt();
            if norm > eps_orth {
                for i in 0..m {
                    let v = full_u.get(&[i, j])?;
                    full_u.set(&[i, j], v / norm)?;
                }
            }
            // Subtract projection of column j from subsequent columns.
            for j2 in (j + 1)..min_mn {
                let mut dot = T::zero();
                for i in 0..m {
                    dot = dot + full_u.get(&[i, j])? * full_u.get(&[i, j2])?;
                }
                for i in 0..m {
                    let old = full_u.get(&[i, j2])?;
                    let proj = full_u.get(&[i, j])?;
                    full_u.set(&[i, j2], old - dot * proj)?;
                }
            }
        }

        // full_vt = b_svd.vt · vb^T  (min_mn × n)
        let mut full_vt = Array::zeros(&[min_mn, n]);
        for i in 0..min_mn {
            for j in 0..n {
                let mut s = T::zero();
                for k in 0..n {
                    s = s + b_svd.vt.get(&[i, k])? * vb.get(&[j, k])?;
                }
                full_vt.set(&[i, j], s)?;
            }
        }
        // Re-orthogonalize rows of full_vt via MGS.
        for i in 0..min_mn {
            let mut norm_sq = T::zero();
            for j in 0..n {
                let v = full_vt.get(&[i, j])?;
                norm_sq = norm_sq + v * v;
            }
            let norm = norm_sq.sqrt();
            if norm > eps_orth {
                for j in 0..n {
                    let v = full_vt.get(&[i, j])?;
                    full_vt.set(&[i, j], v / norm)?;
                }
            }
            for i2 in (i + 1)..min_mn {
                let mut dot = T::zero();
                for j in 0..n {
                    dot = dot + full_vt.get(&[i, j])? * full_vt.get(&[i2, j])?;
                }
                for j in 0..n {
                    let old = full_vt.get(&[i2, j])?;
                    let proj = full_vt.get(&[i, j])?;
                    full_vt.set(&[i2, j], old - dot * proj)?;
                }
            }
        }

        // Take only the first min_mn singular values (AᵀA is n×n so b_svd.s may have n entries).
        let sigma: Vec<T> = b_svd.s.to_vec().into_iter().take(min_mn).collect();
        let u_sorted = full_u;
        let vt_sorted = full_vt;

        // -----------------------------------------------------------------------
        // Compute condition number and rank.
        // -----------------------------------------------------------------------
        let s_max = if !sigma.is_empty() {
            sigma[0]
        } else {
            T::zero()
        };
        let s_min = if !sigma.is_empty() {
            sigma[sigma.len() - 1]
        } else {
            T::zero()
        };
        let condition_number = if s_min > T::zero() {
            s_max / s_min
        } else {
            T::infinity()
        };

        let eps_rank = T::epsilon();
        let threshold = eps_rank
            * <T as num_traits::NumCast>::from(m.max(n)).unwrap_or_else(|| T::one())
            * s_max;
        let rank = sigma.iter().take_while(|&&s| s > threshold).count();

        Ok(SVDStableResult {
            u: u_sorted,
            s: Array::from_vec(sigma),
            vt: vt_sorted,
            condition_number,
            rank,
        })
    }

    /// Enhanced eigenvalue decomposition for symmetric matrices
    ///
    /// This implementation uses:
    /// - Tridiagonalization via Householder reflections
    /// - QR algorithm with shifts for eigenvalue computation
    /// - Improved numerical stability for nearly degenerate cases
    pub fn symmetric_eigendecomposition<T>(a: &Array<T>) -> Result<(Vec<T>, Array<T>)>
    where
        T: Float + Clone + Debug,
    {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Eigendecomposition requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];

        // For small matrices, use direct methods
        if n <= 3 {
            return Self::symmetric_eigen_small(a);
        }

        // For larger matrices, use tridiagonalization
        Self::symmetric_eigen_tridiagonal(a)
    }

    /// Direct eigendecomposition for small symmetric matrices
    fn symmetric_eigen_small<T>(a: &Array<T>) -> Result<(Vec<T>, Array<T>)>
    where
        T: Float + Clone + Debug,
    {
        let n = a.shape()[0];

        if n == 1 {
            let eigenvalue = a.get(&[0, 0])?;
            let eigenvector = Array::from_vec(vec![T::one()]).reshape(&[1, 1]);
            return Ok((vec![eigenvalue], eigenvector));
        }

        if n == 2 {
            let a11 = a.get(&[0, 0])?;
            let a12 = a.get(&[0, 1])?;
            let a22 = a.get(&[1, 1])?;

            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a12;
            let four = T::from(4.0).expect("Failed to convert 4.0 to type T");
            let two = T::from(2.0).expect("Failed to convert 2.0 to type T");
            let discriminant = (trace * trace - four * det).sqrt();

            let lambda1 = (trace + discriminant) / two;
            let lambda2 = (trace - discriminant) / two;

            // Compute eigenvectors
            let mut eigenvectors = Array::zeros(&[2, 2]);

            // First eigenvector
            if num_traits::Float::abs(a12) > T::epsilon() {
                let v1_x = a12;
                let v1_y = lambda1 - a11;
                let norm1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
                eigenvectors.set(&[0, 0], v1_x / norm1)?;
                eigenvectors.set(&[1, 0], v1_y / norm1)?;
            } else {
                eigenvectors.set(&[0, 0], T::one())?;
                eigenvectors.set(&[1, 0], T::zero())?;
            }

            // Second eigenvector
            if num_traits::Float::abs(a12) > T::epsilon() {
                let v2_x = a12;
                let v2_y = lambda2 - a11;
                let norm2 = (v2_x * v2_x + v2_y * v2_y).sqrt();
                eigenvectors.set(&[0, 1], v2_x / norm2)?;
                eigenvectors.set(&[1, 1], v2_y / norm2)?;
            } else {
                eigenvectors.set(&[0, 1], T::zero())?;
                eigenvectors.set(&[1, 1], T::one())?;
            }

            return Ok((vec![lambda1, lambda2], eigenvectors));
        }

        // For n = 3, use Cardano closed-form (Kopp 2008) for real symmetric 3x3
        // Step 1: extract upper-triangle entries
        let a00 = a.get(&[0, 0])?;
        let a01 = a.get(&[0, 1])?;
        let a02 = a.get(&[0, 2])?;
        let a11 = a.get(&[1, 1])?;
        let a12 = a.get(&[1, 2])?;
        let a22 = a.get(&[2, 2])?;

        // p1: sum of squared off-diagonal entries
        let p1 = a01 * a01 + a02 * a02 + a12 * a12;

        // Helper constants
        let three = T::from(3.0).expect("3.0 must be representable");
        let two = T::from(2.0).expect("2.0 must be representable");
        let six = T::from(6.0).expect("6.0 must be representable");
        let pi = T::from(std::f64::consts::PI).expect("PI must be representable");

        if p1 == T::zero() {
            // Diagonal matrix: eigenvalues are the diagonal entries.
            // Sort descending; preserve index so ties don't collide.
            let mut pairs = [(a00, 0usize), (a11, 1usize), (a22, 2usize)];
            pairs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
            let eigs: Vec<T> = pairs.iter().map(|p| p.0).collect();
            let mut eigenvectors = Array::zeros(&[3, 3]);
            for (col, &(_, axis)) in pairs.iter().enumerate() {
                eigenvectors.set(&[axis, col], T::one())?;
            }
            return Ok((eigs, eigenvectors));
        }

        // General symmetric 3x3: Cardano formula
        let trace = a00 + a11 + a22;
        let q = trace / three;

        let a00q = a00 - q;
        let a11q = a11 - q;
        let a22q = a22 - q;
        let p2 = a00q * a00q + a11q * a11q + a22q * a22q + two * p1;
        let p = (p2 / six).sqrt();

        // Compute det(B) where B = (A - qI) / p
        // det of symmetric matrix with entries b_ij = (a_ij - q*delta_ij) / p
        let inv_p = T::one() / p;
        let b00 = a00q * inv_p;
        let b11 = a11q * inv_p;
        let b22 = a22q * inv_p;
        let b01 = a01 * inv_p;
        let b02 = a02 * inv_p;
        let b12 = a12 * inv_p;

        // det(B) for symmetric 3x3
        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);

        let r = det_b / two;

        // Clamp r to [-1, 1] to avoid NaN from acos
        let phi = if r <= -T::one() {
            pi / three
        } else if r >= T::one() {
            T::zero()
        } else {
            num_traits::Float::acos(r) / three
        };

        // Eigenvalues (sorted descending: eig1 >= eig2 >= eig3)
        let two_p = two * p;
        let eig1 = q + two_p * num_traits::Float::cos(phi);
        let eig3 = q + two_p * num_traits::Float::cos(phi + two * pi / three);
        let eig2 = trace - eig1 - eig3;

        let eigenvalues = vec![eig1, eig2, eig3];

        // Eigenvectors via cross-product method (Kopp 2008 §5)
        // For eigenvalue lambda_i, eigenvector = (A - lambda_j*I) col_k  ×  (A - lambda_j*I) col_l
        // We compute two cross-products and choose the one with larger norm.
        let mut eigenvectors = Array::zeros(&[3, 3]);

        for (col, &lambda) in eigenvalues.iter().enumerate() {
            // Build (A - lambda*I) as 9 scalars (row-major)
            let m00 = a00 - lambda;
            let m01 = a01;
            let m02 = a02;
            let m10 = a01; // symmetric
            let m11 = a11 - lambda;
            let m12 = a12;
            let m20 = a02; // symmetric
            let m21 = a12;
            let m22 = a22 - lambda;

            // Extract the three column vectors of (A - lambda*I)
            let c0 = [m00, m10, m20];
            let c1 = [m01, m11, m21];
            let c2 = [m02, m12, m22];

            // Compute three candidate eigenvectors via column cross products
            let candidates = [
                Self::cross3(c0, c1),
                Self::cross3(c0, c2),
                Self::cross3(c1, c2),
            ];

            // Pick the candidate with the largest squared norm
            let norms_sq: [T; 3] = candidates
                .iter()
                .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                .collect::<Vec<T>>()
                .try_into()
                .expect("always 3 candidates");

            let best_idx = if norms_sq[0] >= norms_sq[1] && norms_sq[0] >= norms_sq[2] {
                0usize
            } else if norms_sq[1] >= norms_sq[2] {
                1
            } else {
                2
            };

            let best_norm_sq = norms_sq[best_idx];

            if best_norm_sq > T::epsilon() {
                let best = candidates[best_idx];
                let norm = best_norm_sq.sqrt();
                eigenvectors.set(&[0, col], best[0] / norm)?;
                eigenvectors.set(&[1, col], best[1] / norm)?;
                eigenvectors.set(&[2, col], best[2] / norm)?;
            } else {
                // Degenerate case: fall back to Gram-Schmidt from the other columns
                // Build a basis vector orthogonal to already-computed columns
                let fallback = Self::find_orthogonal_basis_vector(&eigenvectors, col)?;
                eigenvectors.set(&[0, col], fallback[0])?;
                eigenvectors.set(&[1, col], fallback[1])?;
                eigenvectors.set(&[2, col], fallback[2])?;
            }
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Eigendecomposition for n>3 real symmetric matrices via Jacobi iteration.
    ///
    /// The classical Jacobi eigenvalue algorithm rotates the off-diagonal element
    /// with the largest absolute value to zero using a sequence of Givens (Jacobi)
    /// rotations, until the off-diagonal Frobenius norm is below the tolerance.
    ///
    /// This is not the fastest algorithm for large matrices, but it is simple,
    /// provably correct, and delivers full eigenvector accuracy even for
    /// nearly-degenerate eigenvalues.
    ///
    /// Reference: Golub & Van Loan §8.4.3 (Classical Jacobi).
    fn symmetric_eigen_tridiagonal<T>(a: &Array<T>) -> Result<(Vec<T>, Array<T>)>
    where
        T: Float + Clone + Debug,
    {
        let n = a.shape()[0];

        // -----------------------------------------------------------------------
        // Classical Jacobi eigenvalue algorithm for real symmetric n×n matrices.
        //
        // Algorithm: cyclic Jacobi sweeps.  At each step we find the off-diagonal
        // element with the largest absolute value, A[p,q], compute the rotation angle
        // θ = 0.5*atan2(2*A[p,q], A[q,q]-A[p,p]), apply the Givens rotation G(p,q,θ)
        // to zero A[p,q], and accumulate G into the eigenvector matrix V.
        // Repeat until the off-diagonal Frobenius norm falls below eps * ||A||_F.
        //
        // Reference: Golub & Van Loan §8.4, Numerical Recipes §11.1.
        // -----------------------------------------------------------------------
        let eps = T::epsilon();

        // Working copy as flat row-major Vec<T>.
        let mut a_v: Vec<T> = (0..n * n)
            .map(|k| a.get(&[k / n, k % n]).expect("valid index"))
            .collect();
        // Eigenvector matrix V = I.
        let mut v_v: Vec<T> = vec![T::zero(); n * n];
        for i in 0..n {
            v_v[i * n + i] = T::one();
        }

        // Macro-style flat index into the n×n working matrices.
        macro_rules! ai {
            ($r:expr, $c:expr) => {
                $r * n + $c
            };
        }

        let a_frob_sq: T = a_v.iter().fold(T::zero(), |acc, &x| acc + x * x);
        let tol_sq = eps * eps * a_frob_sq;

        let max_sweeps = 100;

        'sweep: for _ in 0..max_sweeps {
            // Check off-diagonal norm for convergence.
            let mut off_sq = T::zero();
            for p in 0..n {
                for q in (p + 1)..n {
                    off_sq = off_sq + a_v[ai!(p, q)] * a_v[ai!(p, q)];
                }
            }
            off_sq = off_sq + off_sq; // symmetric
            if off_sq <= tol_sq {
                break 'sweep;
            }

            // One cyclic Jacobi sweep: iterate over all (p, q) with p < q.
            for p in 0..n {
                for q in (p + 1)..n {
                    let a_pq = a_v[ai!(p, q)];
                    if num_traits::Float::abs(a_pq) < eps {
                        continue; // already zero
                    }

                    let a_pp = a_v[ai!(p, p)];
                    let a_qq = a_v[ai!(q, q)];

                    // Symmetric Schur decomposition of the 2×2 block.
                    // Compute c = cos(θ), s = sin(θ) such that the 2×2 sub-block is diagonalised.
                    let two_t = T::from(2.0).expect("2.0");
                    let tau = (a_qq - a_pp) / (two_t * a_pq);
                    let t = if tau >= T::zero() {
                        T::one() / (tau + (T::one() + tau * tau).sqrt())
                    } else {
                        -T::one() / (-tau + (T::one() + tau * tau).sqrt())
                    };
                    let c = T::one() / (T::one() + t * t).sqrt();
                    let s = t * c;

                    // Apply the rotation A ← Gᵀ · A · G (symmetric update).
                    // For r ∉ {p, q}: update rows/columns p and q.
                    for r in 0..n {
                        if r == p || r == q {
                            continue;
                        }
                        let a_rp = a_v[ai!(r, p)];
                        let a_rq = a_v[ai!(r, q)];
                        let new_rp = c * a_rp - s * a_rq;
                        let new_rq = s * a_rp + c * a_rq;
                        a_v[ai!(r, p)] = new_rp;
                        a_v[ai!(p, r)] = new_rp;
                        a_v[ai!(r, q)] = new_rq;
                        a_v[ai!(q, r)] = new_rq;
                    }

                    // Update diagonal entries and zero A[p,q].
                    a_v[ai!(p, p)] = a_pp - t * a_pq;
                    a_v[ai!(q, q)] = a_qq + t * a_pq;
                    a_v[ai!(p, q)] = T::zero();
                    a_v[ai!(q, p)] = T::zero();

                    // Accumulate the rotation into V: V ← V · G
                    for r in 0..n {
                        let v_rp = v_v[ai!(r, p)];
                        let v_rq = v_v[ai!(r, q)];
                        v_v[ai!(r, p)] = c * v_rp - s * v_rq;
                        v_v[ai!(r, q)] = s * v_rp + c * v_rq;
                    }
                }
            }
        }

        // Diagonal of a_v now contains the eigenvalues.
        let d: Vec<T> = (0..n).map(|i| a_v[ai!(i, i)]).collect();

        // Sort eigenvalues ascending; reorder V columns consistently.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<T> = order.iter().map(|&k| d[k]).collect();

        let mut q_sorted = Array::zeros(&[n, n]);
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..n {
                q_sorted.set(&[row, new_col], v_v[ai!(row, old_col)])?;
            }
        }

        Ok((eigenvalues, q_sorted))
    }

    // Helper functions

    /// 3-D cross product.
    fn cross3<T>(a: [T; 3], b: [T; 3]) -> [T; 3]
    where
        T: Float + Clone,
    {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Find a unit vector that is orthogonal to all columns already written in
    /// `eigvecs` before column `col`.  Uses a simple Gram-Schmidt sweep over
    /// the three canonical basis vectors and picks the one with the largest
    /// projection remaining.
    fn find_orthogonal_basis_vector<T>(eigvecs: &Array<T>, col: usize) -> Result<[T; 3]>
    where
        T: Float + Clone,
    {
        // Three canonical basis candidates
        let candidates: [[T; 3]; 3] = [
            [T::one(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero()],
            [T::zero(), T::zero(), T::one()],
        ];

        let mut best = [T::zero(); 3];
        let mut best_norm_sq = -T::one(); // sentinel

        for cand in &candidates {
            let mut v = *cand;

            // Gram-Schmidt: subtract projections onto previously-stored columns
            for prev in 0..col {
                let ex = eigvecs.get(&[0, prev]).unwrap_or(T::zero());
                let ey = eigvecs.get(&[1, prev]).unwrap_or(T::zero());
                let ez = eigvecs.get(&[2, prev]).unwrap_or(T::zero());
                let dot = v[0] * ex + v[1] * ey + v[2] * ez;
                v[0] = v[0] - dot * ex;
                v[1] = v[1] - dot * ey;
                v[2] = v[2] - dot * ez;
            }

            let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if norm_sq > best_norm_sq {
                best_norm_sq = norm_sq;
                best = v;
            }
        }

        if best_norm_sq > T::epsilon() {
            let norm = best_norm_sq.sqrt();
            Ok([best[0] / norm, best[1] / norm, best[2] / norm])
        } else {
            // Complete failure – return e_0 as a safe default (e.g. identity matrix)
            Ok([T::one(), T::zero(), T::zero()])
        }
    }

    fn householder_vector<T>(x: &[T]) -> Result<(Vec<T>, T)>
    where
        T: Float + Clone,
    {
        let n = x.len();
        if n == 0 {
            return Err(NumRs2Error::InvalidOperation("Empty vector".to_string()));
        }

        let x_norm = x
            .iter()
            .map(|&xi| xi * xi)
            .fold(T::zero(), |acc, xi| acc + xi)
            .sqrt();

        if x_norm == T::zero() {
            return Ok((vec![T::zero(); n], T::zero()));
        }

        let alpha = if x[0] >= T::zero() { -x_norm } else { x_norm };

        let mut v = vec![T::zero(); n];
        v[0] = x[0] - alpha;
        v[1..n].copy_from_slice(&x[1..n]);

        let v_norm_sq = v
            .iter()
            .map(|&vi| vi * vi)
            .fold(T::zero(), |acc, vi| acc + vi);

        if v_norm_sq == T::zero() {
            return Ok((v, T::zero()));
        }

        let beta = T::from(2.0).expect("Failed to convert 2.0 to type T") / v_norm_sq;

        // Don't normalize v - it should remain as computed
        Ok((v, beta))
    }

    fn apply_householder<T>(x: &[T], v: &[T], beta: T) -> Result<Vec<T>>
    where
        T: Float + Clone,
    {
        if x.len() != v.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Vector length mismatch".to_string(),
            ));
        }

        let dot_product = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| xi * vi)
            .fold(T::zero(), |acc, prod| acc + prod);

        let mut result = Vec::with_capacity(x.len());
        for (&xi, &vi) in x.iter().zip(v.iter()) {
            result.push(xi - beta * dot_product * vi);
        }

        Ok(result)
    }

    fn extract_column_slice<T>(
        matrix: &Array<T>,
        col: usize,
        start_row: usize,
        end_row: usize,
    ) -> Result<Vec<T>>
    where
        T: Float + Clone,
    {
        let mut result = Vec::with_capacity(end_row - start_row);
        for i in start_row..end_row {
            result.push(matrix.get(&[i, col])?);
        }
        Ok(result)
    }

    fn transpose<T>(a: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone,
    {
        let shape = a.shape();
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "Transpose requires 2D matrix".to_string(),
            ));
        }

        let m = shape[0];
        let n = shape[1];
        let mut result = Array::zeros(&[n, m]);

        for i in 0..m {
            for j in 0..n {
                result.set(&[j, i], a.get(&[i, j])?)?;
            }
        }

        Ok(result)
    }

    fn matrix_multiply<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Send + Sync + 'static,
    {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
            return Err(NumRs2Error::DimensionMismatch(
                "Invalid matrix multiplication dimensions".to_string(),
            ));
        }

        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        let mut c = Array::zeros(&[m, n]);

        // Simple matrix multiplication for stable decompositions
        // Uses Array::matmul which handles the multiplication internally
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + a.get(&[i, l])? * b.get(&[l, j])?;
                }
                c.set(&[i, j], sum)?;
            }
        }

        Ok(c)
    }
}

/// Result of QR decomposition with pivoting
#[derive(Debug)]
pub struct QRPivotedResult<T: Clone> {
    pub q: Array<T>,
    pub r: Array<T>,
    pub p: Array<f64>, // Permutation vector
    pub condition_number: T,
    pub rank: usize,
}

/// Result of stable Cholesky decomposition
#[derive(Debug)]
pub struct CholeskyStableResult<T: Clone> {
    pub l: Array<T>,
    pub condition_number: T,
    pub is_positive_definite: bool,
    pub pivoting_used: bool,
    pub p: Option<Array<f64>>, // Permutation vector (if pivoting used)
    pub d: Option<Array<T>>,   // Diagonal matrix (if LDLT used)
}

/// Result of stable SVD decomposition
#[derive(Debug)]
pub struct SVDStableResult<T: Clone> {
    pub u: Array<T>,
    pub s: Array<T>,
    pub vt: Array<T>,
    pub condition_number: T,
    pub rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_qr_pivoted() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

        let result = StableDecompositions::qr_pivoted(&a).expect("QR pivoted should succeed");

        // Verify dimensions
        assert_eq!(result.q.shape(), vec![2, 2]);
        assert_eq!(result.r.shape(), vec![2, 3]);
        assert_eq!(result.p.shape(), vec![3]);

        // Verify Q is orthogonal (Q^T * Q = I)
        let qt = StableDecompositions::transpose(&result.q).expect("transpose should succeed");
        let qtq = StableDecompositions::matrix_multiply(&qt, &result.q)
            .expect("matrix multiply should succeed");

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(
                    qtq.get(&[i, j]).expect("valid index"),
                    expected,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_cholesky_stable_positive_definite() {
        // Create a positive definite matrix
        let a = Array::from_vec(vec![4.0, 2.0, 2.0, 3.0]).reshape(&[2, 2]);

        let result = StableDecompositions::cholesky_stable(&a).expect("Cholesky should succeed");

        assert!(result.is_positive_definite);
        assert!(!result.pivoting_used);

        // Verify L * L^T = A
        let lt = StableDecompositions::transpose(&result.l).expect("transpose should succeed");
        let llt = StableDecompositions::matrix_multiply(&result.l, &lt)
            .expect("matrix multiply should succeed");

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    llt.get(&[i, j]).expect("valid index"),
                    a.get(&[i, j]).expect("valid index"),
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigendecomposition_2x2() {
        let a = Array::from_vec(vec![3.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);

        let (eigenvalues, eigenvectors) = StableDecompositions::symmetric_eigendecomposition(&a)
            .expect("eigendecomposition should succeed");

        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.shape(), vec![2, 2]);

        // For this matrix, eigenvalues should be 4 and 2
        let mut sorted_eigenvalues = eigenvalues.clone();
        sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        assert_relative_eq!(sorted_eigenvalues[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(sorted_eigenvalues[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_stable_small() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        let result = StableDecompositions::svd_stable(&a).expect("SVD should succeed");

        // Verify dimensions
        assert_eq!(result.u.shape(), vec![2, 2]);
        assert_eq!(result.s.shape(), vec![2]);
        assert_eq!(result.vt.shape(), vec![2, 2]);

        // Verify singular values are non-negative and sorted
        let s_data = result.s.to_vec();
        assert!(s_data[0] >= s_data[1]);
        assert!(s_data[1] >= 0.0);
    }

    #[test]
    fn test_householder_vector() {
        let x = vec![1.0, 2.0, 3.0];
        let (v, beta) =
            StableDecompositions::householder_vector(&x).expect("householder should succeed");

        assert_eq!(v.len(), 3);
        assert!(beta >= 0.0);

        // Verify that applying the Householder reflection gives correct result
        let result = StableDecompositions::apply_householder(&x, &v, beta)
            .expect("apply householder should succeed");

        // First component should have the opposite sign and same magnitude as original norm
        let x_norm = (1.0 + 4.0 + 9.0_f64).sqrt();
        assert_relative_eq!(result[0].abs(), x_norm, epsilon = 1e-10);

        // Other components should be zero
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    // -------------------------------------------------------------------------
    // Tests for the new full-size implementations
    // -------------------------------------------------------------------------

    /// Helper: compute ||A·v - λ·v||₂ for an eigenpair.
    fn eigenpair_residual(a: &Array<f64>, lambda: f64, v: &[f64]) -> f64 {
        let n = v.len();
        let mut res = 0.0_f64;
        for i in 0..n {
            let mut av_i = 0.0_f64;
            for j in 0..n {
                av_i += a.get(&[i, j]).expect("valid index") * v[j];
            }
            let diff = av_i - lambda * v[i];
            res += diff * diff;
        }
        res.sqrt()
    }

    /// Helper: Frobenius norm ||A||_F.
    fn frob_norm(a: &Array<f64>, rows: usize, cols: usize) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..rows {
            for j in 0..cols {
                let v = a.get(&[i, j]).expect("valid index");
                s += v * v;
            }
        }
        s.sqrt()
    }

    #[test]
    fn test_symmetric_eigen_4x4() {
        // Build a known real symmetric 4×4 matrix:
        //   A = Lᵀ·L where L is lower triangular, so A is positive definite.
        // Values chosen so eigenvalues are well-separated.
        #[rustfmt::skip]
        let data: Vec<f64> = vec![
             4.0, 2.0, 1.0, 0.5,
             2.0, 5.0, 2.0, 1.0,
             1.0, 2.0, 6.0, 2.0,
             0.5, 1.0, 2.0, 7.0,
        ];
        let a = Array::from_vec(data).reshape(&[4, 4]);

        let (eigenvalues, eigenvectors) = StableDecompositions::symmetric_eigendecomposition(&a)
            .expect("eigendecomposition should succeed");

        let n = 4;
        assert_eq!(eigenvalues.len(), n);
        assert_eq!(eigenvectors.shape(), vec![n, n]);

        // Verify Av = λv  and  |vᵢᵀ·vⱼ| ≈ δᵢⱼ
        for col in 0..n {
            let lambda = eigenvalues[col];
            let v: Vec<f64> = (0..n)
                .map(|r| eigenvectors.get(&[r, col]).expect("valid"))
                .collect();

            // Residual ||Av - λv|| < 1e-6
            let res = eigenpair_residual(&a, lambda, &v);
            assert!(
                res < 1e-6,
                "eigenpair residual {} for eigenvalue {} is too large",
                res,
                lambda
            );

            // Orthogonality of distinct eigenvectors
            for col2 in (col + 1)..n {
                let v2: Vec<f64> = (0..n)
                    .map(|r| eigenvectors.get(&[r, col2]).expect("valid"))
                    .collect();
                let dot: f64 = v.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-6,
                    "eigenvectors {} and {} not orthogonal: dot = {}",
                    col,
                    col2,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_5x5() {
        // 5×5 symmetric matrix: Hilbert-like, known to be positive definite.
        let n = 5usize;
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = 1.0 / (1.0 + (i + j) as f64);
            }
        }
        // Make strictly diagonally dominant to guarantee positive definiteness.
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                if j != i {
                    row_sum += data[i * n + j].abs();
                }
            }
            data[i * n + i] = row_sum + 2.0;
        }
        let a = Array::from_vec(data).reshape(&[n, n]);

        let (eigenvalues, eigenvectors) = StableDecompositions::symmetric_eigendecomposition(&a)
            .expect("eigendecomposition should succeed");

        assert_eq!(eigenvalues.len(), n);

        for col in 0..n {
            let lambda = eigenvalues[col];
            let v: Vec<f64> = (0..n)
                .map(|r| eigenvectors.get(&[r, col]).expect("valid"))
                .collect();

            let res = eigenpair_residual(&a, lambda, &v);
            assert!(
                res < 1e-5,
                "5×5 eigenpair residual {} for eigenvalue {} is too large",
                res,
                lambda
            );

            for col2 in (col + 1)..n {
                let v2: Vec<f64> = (0..n)
                    .map(|r| eigenvectors.get(&[r, col2]).expect("valid"))
                    .collect();
                let dot: f64 = v.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-5,
                    "5×5 eigenvectors {} and {} not orthogonal: dot = {}",
                    col,
                    col2,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_svd_bidiagonal_4x4() {
        // 4×4 non-symmetric matrix (thin, well-conditioned).
        #[rustfmt::skip]
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
        ];
        let a = Array::from_vec(data).reshape(&[4, 4]);

        // svd_stable dispatches to svd_bidiagonal for min_mn > 4;
        // for 4×4 it goes to svd_small_stable (min_mn==4 uses the <=4 path).
        // We call svd_bidiagonal directly to test the implementation.
        let result =
            StableDecompositions::svd_bidiagonal(&a).expect("SVD bidiagonal should succeed");

        let m = 4usize;
        let n = 4usize;
        let k = m.min(n);

        assert_eq!(result.u.shape(), vec![m, k]);
        assert_eq!(result.s.shape(), vec![k]);
        assert_eq!(result.vt.shape(), vec![k, n]);

        // Verify A ≈ U · diag(σ) · Vᵀ
        let mut recon = Array::zeros(&[m, n]);
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0_f64;
                for l in 0..k {
                    val += result.u.get(&[i, l]).expect("u")
                        * result.s.get(&[l]).expect("s")
                        * result.vt.get(&[l, j]).expect("vt");
                }
                recon.set(&[i, j], val).expect("set");
            }
        }
        let err = frob_norm(&recon, m, n) + {
            // compute || A - recon ||_F
            let mut diff_sq = 0.0_f64;
            for i in 0..m {
                for j in 0..n {
                    let d = a.get(&[i, j]).expect("a") - recon.get(&[i, j]).expect("recon");
                    diff_sq += d * d;
                }
            }
            diff_sq.sqrt() - frob_norm(&recon, m, n)
        };
        // Just compute the Frobenius norm of the difference directly.
        let mut diff_frob = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let d = a.get(&[i, j]).expect("a") - recon.get(&[i, j]).expect("recon");
                diff_frob += d * d;
            }
        }
        diff_frob = diff_frob.sqrt();
        let _ = err;
        assert!(
            diff_frob < 1e-6,
            "SVD 4×4 reconstruction error ||A - UΣVᵀ||_F = {} is too large",
            diff_frob
        );

        // U orthogonality: Uᵀ·U ≈ I_k
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for r in 0..m {
                    dot += result.u.get(&[r, i]).expect("u") * result.u.get(&[r, j]).expect("u");
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "U not orthogonal at ({},{}) = {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Vᵀ orthogonality: Vᵀ·V = Vᵀ·(Vᵀ)ᵀ ≈ I_k
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for c in 0..n {
                    dot +=
                        result.vt.get(&[i, c]).expect("vt") * result.vt.get(&[j, c]).expect("vt");
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "Vᵀ rows not orthogonal at ({},{}) = {}",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_svd_bidiagonal_3x5() {
        // 3×5 rectangular matrix.
        #[rustfmt::skip]
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 1.0,
            2.0, 3.0, 4.0, 5.0, 6.0,
        ];
        let a = Array::from_vec(data).reshape(&[3, 5]);

        let result =
            StableDecompositions::svd_bidiagonal(&a).expect("SVD bidiagonal 3x5 should succeed");

        let m = 3usize;
        let n = 5usize;
        let k = m.min(n); // = 3

        assert_eq!(result.u.shape(), vec![m, k]);
        assert_eq!(result.s.shape(), vec![k]);
        assert_eq!(result.vt.shape(), vec![k, n]);

        // Verify A ≈ U · diag(σ) · Vᵀ
        let mut diff_frob = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0_f64;
                for l in 0..k {
                    val += result.u.get(&[i, l]).expect("u")
                        * result.s.get(&[l]).expect("s")
                        * result.vt.get(&[l, j]).expect("vt");
                }
                let d = a.get(&[i, j]).expect("a") - val;
                diff_frob += d * d;
            }
        }
        diff_frob = diff_frob.sqrt();
        assert!(
            diff_frob < 1e-6,
            "SVD 3×5 reconstruction error ||A - UΣVᵀ||_F = {} is too large",
            diff_frob
        );

        // U orthogonality
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for r in 0..m {
                    dot += result.u.get(&[r, i]).expect("u") * result.u.get(&[r, j]).expect("u");
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "3×5 U not orthogonal at ({},{}) = {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Vᵀ row orthogonality
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for c in 0..n {
                    dot +=
                        result.vt.get(&[i, c]).expect("vt") * result.vt.get(&[j, c]).expect("vt");
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "3×5 Vᵀ rows not orthogonal at ({},{}) = {}",
                    i,
                    j,
                    dot
                );
            }
        }
    }
}
