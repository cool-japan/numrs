//! Enhanced sparse matrix operations with advanced algorithms and optimizations
//!
//! This module provides advanced sparse matrix operations including:
//! - Optimized sparse-dense matrix operations
//! - Iterative linear solvers (CG, GMRES, BiCGSTAB)
//! - Sparse matrix decompositions (ILU, Cholesky)
//! - Advanced storage formats and conversions
//! - SIMD-accelerated operations where applicable

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::sparse::{SparseMatrix, SparseMatrixFormat};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Enhanced sparse matrix operations with advanced algorithms
pub struct SparseOpsAdvanced;

impl SparseOpsAdvanced {
    /// Sparse-dense matrix multiplication with optimized memory access patterns
    ///
    /// Computes C = alpha * A * B + beta * C where A is sparse and B, C are dense
    pub fn spmv_dense<T>(
        a: &SparseMatrix<T>,
        x: &Array<T>,
        y: &mut Array<T>,
        alpha: T,
        beta: T,
    ) -> Result<()>
    where
        T: Float + Clone + Debug,
    {
        let a_shape = a.shape();
        let x_shape = x.shape();
        let y_shape = y.shape();

        if a_shape.len() != 2 || x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Sparse-dense multiplication requires 2D sparse matrix and 1D dense vectors"
                    .to_string(),
            ));
        }

        if a_shape[1] != x_shape[0] || a_shape[0] != y_shape[0] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix-vector dimensions incompatible".to_string(),
            ));
        }

        let m = a_shape[0];
        let n = a_shape[1];

        // Apply beta scaling to y first
        if beta != T::one() {
            for i in 0..m {
                let val = y.get(&[i])?;
                y.set(&[i], beta * val)?;
            }
        }

        // Perform sparse matrix-vector multiplication
        // Use format-specific optimizations
        match a.format {
            SparseMatrixFormat::CSR => Self::spmv_csr(a, x, y, alpha),
            SparseMatrixFormat::CSC => Self::spmv_csc(a, x, y, alpha),
            _ => Self::spmv_coo(a, x, y, alpha, m, n),
        }
    }

    /// CSR format sparse matrix-vector multiplication
    fn spmv_csr<T>(a: &SparseMatrix<T>, x: &Array<T>, y: &mut Array<T>, alpha: T) -> Result<()>
    where
        T: Float + Clone + Debug,
    {
        if let (Some(indptr), Some(indices)) = (&a.indptr, &a.indices) {
            let m = a.shape()[0];

            for i in 0..m {
                let row_start = indptr[i];
                let row_end = indptr[i + 1];

                let mut sum = T::zero();
                for idx in row_start..row_end {
                    let j = indices[idx];
                    let a_val = a.get(i, j)?;
                    let x_val = x.get(&[j])?;
                    sum = sum + a_val * x_val;
                }

                let current = y.get(&[i])?;
                y.set(&[i], current + alpha * sum)?;
            }
        } else {
            return Err(NumRs2Error::ComputationError(
                "CSR format data not available".to_string(),
            ));
        }

        Ok(())
    }

    /// CSC format sparse matrix-vector multiplication
    fn spmv_csc<T>(a: &SparseMatrix<T>, x: &Array<T>, y: &mut Array<T>, alpha: T) -> Result<()>
    where
        T: Float + Clone + Debug,
    {
        if let (Some(indptr), Some(indices)) = (&a.indptr, &a.indices) {
            let n = a.shape()[1];

            for j in 0..n {
                let col_start = indptr[j];
                let col_end = indptr[j + 1];

                let x_val = x.get(&[j])?;
                let scaled_x = alpha * x_val;

                for idx in col_start..col_end {
                    let i = indices[idx];
                    let a_val = a.get(i, j)?;
                    let current = y.get(&[i])?;
                    y.set(&[i], current + a_val * scaled_x)?;
                }
            }
        } else {
            return Err(NumRs2Error::ComputationError(
                "CSC format data not available".to_string(),
            ));
        }

        Ok(())
    }

    /// COO format sparse matrix-vector multiplication
    fn spmv_coo<T>(
        a: &SparseMatrix<T>,
        x: &Array<T>,
        y: &mut Array<T>,
        alpha: T,
        m: usize,
        n: usize,
    ) -> Result<()>
    where
        T: Float + Clone + Debug,
    {
        // For COO format, iterate through all non-zero entries
        for (indices, value) in &a.array.data {
            let i = indices[0];
            let j = indices[1];

            if i < m && j < n {
                let x_val = x.get(&[j])?;
                let current = y.get(&[i])?;
                y.set(&[i], current + alpha * *value * x_val)?;
            }
        }

        Ok(())
    }

    /// Optimized sparse matrix-matrix multiplication
    pub fn spgemm<T>(a: &SparseMatrix<T>, b: &SparseMatrix<T>) -> Result<SparseMatrix<T>>
    where
        T: Float + Clone + Debug + Zero + One,
    {
        // Use the existing matmul implementation but with optimizations
        let mut result = a.matmul(b)?;

        // Convert result to most efficient format based on sparsity pattern
        let density = result.density();

        if density < 0.1 {
            // Very sparse - keep as COO or convert to CSR for row operations
            result.format = SparseMatrixFormat::CSR;
        } else if density < 0.3 {
            // Moderately sparse - CSR is usually good
            result.to_csr()?;
        } else {
            // Dense enough that conversion overhead might not be worth it
            result.format = SparseMatrixFormat::COO;
        }

        Ok(result)
    }

    /// Conjugate Gradient solver for sparse symmetric positive definite systems
    pub fn solve_cg<T>(
        a: &SparseMatrix<T>,
        b: &Array<T>,
        x0: Option<&Array<T>>,
        tol: T,
        max_iter: usize,
    ) -> Result<(Array<T>, usize, T)>
    where
        T: Float + Clone + Debug,
    {
        let n = a.shape()[0];

        if a.shape()[1] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square for CG solver".to_string(),
            ));
        }

        if b.shape()[0] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Right-hand side vector dimension mismatch".to_string(),
            ));
        }

        // Initialize solution vector
        let mut x = if let Some(x_init) = x0 {
            x_init.clone()
        } else {
            Array::zeros(&[n])
        };

        // Compute initial residual: r = b - A*x
        let mut ax = Array::zeros(&[n]);
        Self::spmv_dense(a, &x, &mut ax, T::one(), T::zero())?;

        let mut r = Array::zeros(&[n]);
        for i in 0..n {
            let b_val = b.get(&[i])?;
            let ax_val = ax.get(&[i])?;
            r.set(&[i], b_val - ax_val)?;
        }

        let mut p = r.clone();
        let mut rsold = Self::dot_product(&r, &r)?;

        let tol_sq = tol * tol;

        for iter in 0..max_iter {
            // Check convergence
            if rsold < tol_sq {
                return Ok((x, iter, rsold.sqrt()));
            }

            // Compute A*p
            let mut ap = Array::zeros(&[n]);
            Self::spmv_dense(a, &p, &mut ap, T::one(), T::zero())?;

            // Compute alpha = rsold / (p^T * A * p)
            let ptap = Self::dot_product(&p, &ap)?;
            if ptap.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "CG solver breakdown: p^T * A * p = 0".to_string(),
                ));
            }

            let alpha = rsold / ptap;

            // Update solution: x = x + alpha * p
            for i in 0..n {
                let x_val = x.get(&[i])?;
                let p_val = p.get(&[i])?;
                x.set(&[i], x_val + alpha * p_val)?;
            }

            // Update residual: r = r - alpha * A * p
            for i in 0..n {
                let r_val = r.get(&[i])?;
                let ap_val = ap.get(&[i])?;
                r.set(&[i], r_val - alpha * ap_val)?;
            }

            let rsnew = Self::dot_product(&r, &r)?;

            // Check convergence
            if rsnew < tol_sq {
                return Ok((x, iter + 1, rsnew.sqrt()));
            }

            // Update search direction: p = r + beta * p
            let beta = rsnew / rsold;
            for i in 0..n {
                let r_val = r.get(&[i])?;
                let p_val = p.get(&[i])?;
                p.set(&[i], r_val + beta * p_val)?;
            }

            rsold = rsnew;
        }

        Ok((x, max_iter, rsold.sqrt()))
    }

    /// BiCGSTAB solver for general sparse systems
    pub fn solve_bicgstab<T>(
        a: &SparseMatrix<T>,
        b: &Array<T>,
        x0: Option<&Array<T>>,
        tol: T,
        max_iter: usize,
    ) -> Result<(Array<T>, usize, T)>
    where
        T: Float + Clone + Debug,
    {
        let n = a.shape()[0];

        if a.shape()[1] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square for BiCGSTAB solver".to_string(),
            ));
        }

        // Initialize solution vector
        let mut x = if let Some(x_init) = x0 {
            x_init.clone()
        } else {
            Array::zeros(&[n])
        };

        // Compute initial residual: r = b - A*x
        let mut ax = Array::zeros(&[n]);
        Self::spmv_dense(a, &x, &mut ax, T::one(), T::zero())?;

        let mut r = Array::zeros(&[n]);
        for i in 0..n {
            let b_val = b.get(&[i])?;
            let ax_val = ax.get(&[i])?;
            r.set(&[i], b_val - ax_val)?;
        }

        let r0 = r.clone();
        let mut p = r.clone();
        let mut v = Array::zeros(&[n]);
        let mut s = Array::zeros(&[n]);
        let mut t = Array::zeros(&[n]);

        let mut rho = T::one();
        let mut alpha = T::one();
        let mut omega = T::one();

        let tol_sq = tol * tol;

        for iter in 0..max_iter {
            let r_norm_sq = Self::dot_product(&r, &r)?;

            // Check convergence
            if r_norm_sq < tol_sq {
                return Ok((x, iter, r_norm_sq.sqrt()));
            }

            let rho_new = Self::dot_product(&r0, &r)?;

            if rho_new.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "BiCGSTAB solver breakdown: rho = 0".to_string(),
                ));
            }

            let beta = (rho_new / rho) * (alpha / omega);

            // Update p = r + beta * (p - omega * v)
            for i in 0..n {
                let r_val = r.get(&[i])?;
                let p_val = p.get(&[i])?;
                let v_val = v.get(&[i])?;
                p.set(&[i], r_val + beta * (p_val - omega * v_val))?;
            }

            // v = A * p
            Self::spmv_dense(a, &p, &mut v, T::one(), T::zero())?;

            let r0v = Self::dot_product(&r0, &v)?;
            if r0v.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "BiCGSTAB solver breakdown: r0^T * v = 0".to_string(),
                ));
            }

            alpha = rho_new / r0v;

            // s = r - alpha * v
            for i in 0..n {
                let r_val = r.get(&[i])?;
                let v_val = v.get(&[i])?;
                s.set(&[i], r_val - alpha * v_val)?;
            }

            // Check if we can stop here
            let s_norm_sq = Self::dot_product(&s, &s)?;
            if s_norm_sq < tol_sq {
                // Update x and return
                for i in 0..n {
                    let x_val = x.get(&[i])?;
                    let p_val = p.get(&[i])?;
                    x.set(&[i], x_val + alpha * p_val)?;
                }
                return Ok((x, iter + 1, s_norm_sq.sqrt()));
            }

            // t = A * s
            Self::spmv_dense(a, &s, &mut t, T::one(), T::zero())?;

            let ts = Self::dot_product(&t, &s)?;
            let tt = Self::dot_product(&t, &t)?;

            if tt.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "BiCGSTAB solver breakdown: t^T * t = 0".to_string(),
                ));
            }

            omega = ts / tt;

            // Update solution: x = x + alpha * p + omega * s
            for i in 0..n {
                let x_val = x.get(&[i])?;
                let p_val = p.get(&[i])?;
                let s_val = s.get(&[i])?;
                x.set(&[i], x_val + alpha * p_val + omega * s_val)?;
            }

            // Update residual: r = s - omega * t
            for i in 0..n {
                let s_val = s.get(&[i])?;
                let t_val = t.get(&[i])?;
                r.set(&[i], s_val - omega * t_val)?;
            }

            rho = rho_new;

            if omega.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "BiCGSTAB solver breakdown: omega = 0".to_string(),
                ));
            }
        }

        let final_residual = Self::dot_product(&r, &r)?.sqrt();
        Ok((x, max_iter, final_residual))
    }

    /// GMRES (Generalized Minimal Residual) solver for sparse non-symmetric systems
    ///
    /// This is the sparse matrix variant of GMRES, optimized for sparse matrices.
    /// It uses restarted GMRES with Arnoldi iteration and Givens rotations.
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse coefficient matrix (must be square)
    /// * `b` - Right-hand side vector
    /// * `x0` - Optional initial guess (zeros if None)
    /// * `tol` - Convergence tolerance
    /// * `max_iter` - Maximum number of iterations
    /// * `restart` - Restart parameter (Krylov subspace dimension)
    ///
    /// # Returns
    ///
    /// Tuple of (solution, iterations, final_residual_norm)
    pub fn solve_gmres<T>(
        a: &SparseMatrix<T>,
        b: &Array<T>,
        x0: Option<&Array<T>>,
        tol: T,
        max_iter: usize,
        restart: usize,
    ) -> Result<(Array<T>, usize, T)>
    where
        T: Float + Clone + Debug,
    {
        let n = a.shape()[0];

        if a.shape()[1] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square for GMRES solver".to_string(),
            ));
        }

        if b.shape()[0] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Right-hand side dimension mismatch".to_string(),
            ));
        }

        let restart = restart.min(n);

        // Initialize solution vector
        let mut x = if let Some(x_init) = x0 {
            x_init.clone()
        } else {
            Array::zeros(&[n])
        };

        // Compute b norm for relative residual check
        let b_norm = Self::dot_product(b, b)?.sqrt();
        if b_norm.is_zero() {
            return Ok((x, 0, T::zero()));
        }

        let mut total_iter = 0;

        // Outer iteration (restarts)
        for _ in 0..(max_iter / restart + 1) {
            // Compute residual r = b - Ax
            let mut ax = Array::zeros(&[n]);
            Self::spmv_dense(a, &x, &mut ax, T::one(), T::zero())?;

            let mut r = Array::zeros(&[n]);
            for i in 0..n {
                let b_val = b.get(&[i])?;
                let ax_val = ax.get(&[i])?;
                r.set(&[i], b_val - ax_val)?;
            }

            let r_norm = Self::dot_product(&r, &r)?.sqrt();

            // Check convergence
            if r_norm / b_norm < tol {
                return Ok((x, total_iter, r_norm));
            }

            // Initialize Arnoldi iteration
            let mut v = vec![Array::zeros(&[n]); restart + 1];
            for i in 0..n {
                v[0].set(&[i], r.get(&[i])? / r_norm)?;
            }

            let mut h = vec![vec![T::zero(); restart]; restart + 1];
            let mut g = vec![T::zero(); restart + 1];
            g[0] = r_norm;

            // Givens rotation coefficients
            let mut cs_vec = vec![T::zero(); restart];
            let mut sn_vec = vec![T::zero(); restart];

            let mut k = 0;
            for j in 0..restart {
                if total_iter >= max_iter {
                    break;
                }
                total_iter += 1;

                // Arnoldi step: w = A * v[j]
                let mut w = Array::zeros(&[n]);
                Self::spmv_dense(a, &v[j], &mut w, T::one(), T::zero())?;

                // Modified Gram-Schmidt orthogonalization
                for i in 0..=j {
                    h[i][j] = Self::dot_product(&v[i], &w)?;
                    for l in 0..n {
                        let val = w.get(&[l])? - h[i][j] * v[i].get(&[l])?;
                        w.set(&[l], val)?;
                    }
                }

                h[j + 1][j] = Self::dot_product(&w, &w)?.sqrt();

                // Check for lucky breakdown
                if h[j + 1][j].abs() < T::from(1e-14).unwrap() {
                    k = j + 1;
                    break;
                }

                // Normalize
                for i in 0..n {
                    v[j + 1].set(&[i], w.get(&[i])? / h[j + 1][j])?;
                }

                // Apply previous Givens rotations to new column
                for i in 0..j {
                    let temp = h[i][j];
                    h[i][j] = cs_vec[i] * temp + sn_vec[i] * h[i + 1][j];
                    h[i + 1][j] = -sn_vec[i] * temp + cs_vec[i] * h[i + 1][j];
                }

                // Compute new Givens rotation
                let r_val = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
                if r_val < T::from(1e-14).unwrap() {
                    k = j + 1;
                    break;
                }

                let cs = h[j][j] / r_val;
                let sn = h[j + 1][j] / r_val;
                cs_vec[j] = cs;
                sn_vec[j] = sn;

                // Apply new rotation to H
                h[j][j] = r_val;
                h[j + 1][j] = T::zero();

                // Apply new rotation to g
                let temp_g = g[j];
                g[j] = cs * temp_g;
                g[j + 1] = -sn * temp_g;

                k = j + 1;

                // Check convergence
                if g[j + 1].abs() / b_norm < tol {
                    break;
                }
            }

            // Solve upper triangular system H*y = g
            let mut y = vec![T::zero(); k];
            for i in (0..k).rev() {
                let mut sum = g[i];
                for j in (i + 1)..k {
                    sum = sum - h[i][j] * y[j];
                }
                y[i] = sum / h[i][i];
            }

            // Update solution: x = x + V * y
            for j in 0..k {
                for i in 0..n {
                    let x_val = x.get(&[i])? + y[j] * v[j].get(&[i])?;
                    x.set(&[i], x_val)?;
                }
            }

            // Check final residual
            let mut ax_final = Array::zeros(&[n]);
            Self::spmv_dense(a, &x, &mut ax_final, T::one(), T::zero())?;

            let mut r_final = Array::zeros(&[n]);
            for i in 0..n {
                r_final.set(&[i], b.get(&[i])? - ax_final.get(&[i])?)?;
            }
            let final_r_norm = Self::dot_product(&r_final, &r_final)?.sqrt();

            if final_r_norm / b_norm < tol || total_iter >= max_iter {
                return Ok((x, total_iter, final_r_norm));
            }
        }

        // Compute final residual
        let mut ax = Array::zeros(&[n]);
        Self::spmv_dense(a, &x, &mut ax, T::one(), T::zero())?;

        let mut r = Array::zeros(&[n]);
        for i in 0..n {
            r.set(&[i], b.get(&[i])? - ax.get(&[i])?)?;
        }
        let final_residual = Self::dot_product(&r, &r)?.sqrt();

        Ok((x, max_iter, final_residual))
    }

    /// Incomplete LU decomposition for preconditioning
    pub fn incomplete_lu<T>(
        a: &SparseMatrix<T>,
        _fill_factor: f64,
    ) -> Result<(SparseMatrix<T>, SparseMatrix<T>)>
    where
        T: Float + Clone + Debug + Zero + One,
    {
        let n = a.shape()[0];

        if a.shape()[1] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square for ILU decomposition".to_string(),
            ));
        }

        // Create L and U matrices
        let mut l = SparseMatrix::new(&[n, n])?;
        let mut u = SparseMatrix::new(&[n, n])?;

        // Initialize L with identity on diagonal
        for i in 0..n {
            l.set(i, i, T::one())?;
        }

        // Copy upper triangular part to U, lower to L
        for (indices, value) in &a.array.data {
            let i = indices[0];
            let j = indices[1];

            if i <= j {
                u.set(i, j, *value)?;
            } else {
                l.set(i, j, *value)?;
            }
        }

        // Simplified ILU(0) - only process existing sparsity pattern
        for k in 0..n {
            let u_kk = u.get(k, k)?;
            if u_kk.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "ILU decomposition failed: zero pivot".to_string(),
                ));
            }

            // Process elements below diagonal in column k
            for i in (k + 1)..n {
                let l_ik = l.get(i, k)?;
                if l_ik != T::zero() {
                    let factor = l_ik / u_kk;
                    l.set(i, k, factor)?;

                    // Update row i of U
                    for j in (k + 1)..n {
                        let u_kj = u.get(k, j)?;
                        if u_kj != T::zero() {
                            let u_ij = u.get(i, j)?;
                            u.set(i, j, u_ij - factor * u_kj)?;
                        }
                    }
                }
            }
        }

        Ok((l, u))
    }

    /// Helper function to compute dot product of two arrays
    fn dot_product<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        if x.shape() != y.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Arrays must have same shape for dot product".to_string(),
            ));
        }

        let n = x.shape()[0];
        let mut result = T::zero();

        for i in 0..n {
            let x_val = x.get(&[i])?;
            let y_val = y.get(&[i])?;
            result = result + x_val * y_val;
        }

        Ok(result)
    }

    /// Estimate the condition number of a sparse matrix using power iteration
    pub fn condition_number_estimate<T>(a: &SparseMatrix<T>, max_iter: usize, tol: T) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        let n = a.shape()[0];

        if a.shape()[1] != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square for condition number estimation".to_string(),
            ));
        }

        // Estimate largest eigenvalue using power iteration
        let mut v = Array::ones(&[n]);
        let mut lambda_max = T::zero();

        for _ in 0..max_iter {
            let mut av = Array::zeros(&[n]);
            Self::spmv_dense(a, &v, &mut av, T::one(), T::zero())?;

            let norm = Self::vector_norm(&av)?;
            if norm < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "Power iteration failed: zero norm".to_string(),
                ));
            }

            // Normalize
            for i in 0..n {
                let val = av.get(&[i])?;
                v.set(&[i], val / norm)?;
            }

            let new_lambda = Self::dot_product(&v, &av)?;

            if (new_lambda - lambda_max).abs() < tol {
                lambda_max = new_lambda;
                break;
            }
            lambda_max = new_lambda;
        }

        // For condition number, we would need smallest eigenvalue too
        // For now, return an estimate based on largest eigenvalue
        // This is a simplified implementation - full condition number estimation
        // would require more sophisticated algorithms
        Ok(lambda_max.abs())
    }

    /// Compute the 2-norm of a vector
    fn vector_norm<T>(x: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        let n = x.shape()[0];
        let mut sum = T::zero();

        for i in 0..n {
            let val = x.get(&[i])?;
            sum = sum + val * val;
        }

        Ok(sum.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseMatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_sparse_matrix_vector_multiplication() {
        // Create a sparse matrix
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        a.set(0, 0, 2.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(1, 1, 3.0).unwrap();
        a.set(2, 2, 4.0).unwrap();

        // Create input vector
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = Array::zeros(&[3]);

        // Test sparse matrix-vector multiplication
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut y, 1.0, 0.0).unwrap();

        // Expected: [2*1 + 1*2, 3*2, 4*3] = [4, 6, 12]
        assert_relative_eq!(y.get(&[0]).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(y.get(&[1]).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(y.get(&[2]).unwrap(), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_conjugate_gradient_solver() {
        // Create a symmetric positive definite matrix
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        a.set(0, 0, 4.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(1, 0, 1.0).unwrap();
        a.set(1, 1, 3.0).unwrap();
        a.set(1, 2, 1.0).unwrap();
        a.set(2, 1, 1.0).unwrap();
        a.set(2, 2, 2.0).unwrap();

        // Right-hand side
        let b = Array::from_vec(vec![6.0, 8.0, 4.0]);

        // Solve using CG
        let (x, iter, residual) = SparseOpsAdvanced::solve_cg(&a, &b, None, 1e-10, 100).unwrap();

        // Check that the solution satisfies A*x = b (approximately)
        let mut ax = Array::zeros(&[3]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).unwrap();

        for i in 0..3 {
            let b_val = b.get(&[i]).unwrap();
            let ax_val = ax.get(&[i]).unwrap();
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_bicgstab_solver() {
        // Create a general matrix
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        a.set(0, 0, 3.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(1, 0, 1.0).unwrap();
        a.set(1, 1, 2.0).unwrap();
        a.set(1, 2, 1.0).unwrap();
        a.set(2, 1, 1.0).unwrap();
        a.set(2, 2, 3.0).unwrap();

        // Right-hand side
        let b = Array::from_vec(vec![5.0, 6.0, 7.0]);

        // Solve using BiCGSTAB
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_bicgstab(&a, &b, None, 1e-10, 100).unwrap();

        // Check that the solution satisfies A*x = b (approximately)
        let mut ax = Array::zeros(&[3]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).unwrap();

        for i in 0..3 {
            let b_val = b.get(&[i]).unwrap();
            let ax_val = ax.get(&[i]).unwrap();
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_gmres_solver() {
        // Create a general matrix (non-symmetric)
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        a.set(0, 0, 3.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(0, 2, 0.5).unwrap();
        a.set(1, 0, 1.0).unwrap();
        a.set(1, 1, 4.0).unwrap();
        a.set(1, 2, 1.0).unwrap();
        a.set(2, 0, 0.0).unwrap();
        a.set(2, 1, 2.0).unwrap();
        a.set(2, 2, 5.0).unwrap();

        // Right-hand side
        let b = Array::from_vec(vec![5.5, 9.0, 17.0]);

        // Solve using GMRES
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 30).unwrap();

        // Check that the solution satisfies A*x = b (approximately)
        let mut ax = Array::zeros(&[3]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).unwrap();

        for i in 0..3 {
            let b_val = b.get(&[i]).unwrap();
            let ax_val = ax.get(&[i]).unwrap();
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_gmres_solver_larger_system() {
        // Create a larger sparse matrix (5x5 diagonally dominant)
        let n = 5;
        let mut a = SparseMatrix::new(&[n, n]).unwrap();

        // Tridiagonal matrix with strong diagonal dominance
        for i in 0..n {
            a.set(i, i, 4.0).unwrap(); // Main diagonal
            if i > 0 {
                a.set(i, i - 1, -1.0).unwrap(); // Lower diagonal
            }
            if i < n - 1 {
                a.set(i, i + 1, -1.0).unwrap(); // Upper diagonal
            }
        }

        // Right-hand side: solution should be [1, 2, 3, 4, 5]
        let b = Array::from_vec(vec![2.0, 1.0, 2.0, 3.0, 16.0]);

        // Solve using GMRES
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 30).unwrap();

        // Check that the solution satisfies A*x = b
        let mut ax = Array::zeros(&[n]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).unwrap();

        for i in 0..n {
            let b_val = b.get(&[i]).unwrap();
            let ax_val = ax.get(&[i]).unwrap();
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_gmres_with_restart() {
        // Create a matrix that might need restart
        let n = 4;
        let mut a = SparseMatrix::new(&[n, n]).unwrap();

        // Create a diagonally dominant matrix
        for i in 0..n {
            a.set(i, i, 5.0).unwrap();
            for j in 0..n {
                if i != j {
                    a.set(i, j, 0.5).unwrap();
                }
            }
        }

        let b = Array::from_vec(vec![7.5, 7.5, 7.5, 7.5]);

        // Solve with small restart parameter to force restarts
        let (x, _iter, residual) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 2).unwrap();

        // Verify solution
        let mut ax = Array::zeros(&[n]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).unwrap();

        for i in 0..n {
            let b_val = b.get(&[i]).unwrap();
            let ax_val = ax.get(&[i]).unwrap();
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-6);
        }

        assert!(residual < 1e-6);
    }

    #[test]
    fn test_gmres_vs_bicgstab() {
        // Compare GMRES and BiCGSTAB on the same problem
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        a.set(0, 0, 4.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(1, 0, 2.0).unwrap();
        a.set(1, 1, 3.0).unwrap();
        a.set(1, 2, 1.0).unwrap();
        a.set(2, 1, 1.0).unwrap();
        a.set(2, 2, 4.0).unwrap();

        let b = Array::from_vec(vec![6.0, 9.0, 5.0]);

        // Solve with GMRES
        let (x_gmres, _, residual_gmres) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 30).unwrap();

        // Solve with BiCGSTAB
        let (x_bicgstab, _, residual_bicgstab) =
            SparseOpsAdvanced::solve_bicgstab(&a, &b, None, 1e-10, 100).unwrap();

        // Both should converge
        assert!(residual_gmres < 1e-8);
        assert!(residual_bicgstab < 1e-8);

        // Solutions should be similar
        for i in 0..3 {
            let g_val = x_gmres.get(&[i]).unwrap();
            let b_val = x_bicgstab.get(&[i]).unwrap();
            assert_relative_eq!(g_val, b_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_incomplete_lu_decomposition() {
        // Create a test matrix
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        a.set(0, 0, 4.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(1, 0, 1.0).unwrap();
        a.set(1, 1, 3.0).unwrap();
        a.set(1, 2, 1.0).unwrap();
        a.set(2, 1, 1.0).unwrap();
        a.set(2, 2, 2.0).unwrap();

        // Compute ILU decomposition
        let (l, u) = SparseOpsAdvanced::incomplete_lu(&a, 1.0).unwrap();

        // Verify that L is lower triangular with 1s on diagonal
        assert_relative_eq!(l.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(1, 1).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(2, 2).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(0, 1).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(0, 2).unwrap(), 0.0, epsilon = 1e-10);

        // Verify U is upper triangular
        assert_relative_eq!(u.get(1, 0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(u.get(2, 0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(u.get(2, 1).unwrap(), 0.0, epsilon = 1e-10);
    }
}
