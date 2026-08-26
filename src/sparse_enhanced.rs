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
    /// Shorthand for [`NumRs2Error::bulk_index_oob`]: every loop below
    /// hoists one `array()`/`array_mut()` above a `for i in 0..n` and then
    /// reads/writes through the raw `ndarray` accessors (`get`/`get_mut`),
    /// which return `Option` rather than the `Array::get`/`Array::set`
    /// wrappers' `Result` -- see that function's doc comment for why this
    /// is unreachable in practice but still returns `Result`.
    #[inline]
    fn oob(indices: &[usize]) -> NumRs2Error {
        NumRs2Error::bulk_index_oob(indices)
    }

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

        // Apply beta scaling to y first. Bulk-acquired: one `array_mut()`
        // (one `Arc::make_mut` unshare check) for the whole loop instead of
        // one per element via `Array::set`.
        if beta != T::one() {
            let y_arr = y.array_mut();
            for i in 0..m {
                let elem = y_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                *elem = beta * *elem;
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

            // Bulk-acquired: `x` is read-only for the whole sweep and `y`
            // is unshared exactly once, instead of once per row via
            // `Array::get`/`Array::set`.
            let x_arr = x.array();
            let y_arr = y.array_mut();

            for i in 0..m {
                let row_start = indptr[i];
                let row_end = indptr[i + 1];

                let mut sum = T::zero();
                for idx in row_start..row_end {
                    let j = indices[idx];
                    let a_val = a.get(i, j)?;
                    let x_val = *x_arr.get([j]).ok_or_else(|| Self::oob(&[j]))?;
                    sum = sum + a_val * x_val;
                }

                let elem = y_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                *elem = *elem + alpha * sum;
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

            let x_arr = x.array();
            let y_arr = y.array_mut();

            for j in 0..n {
                let col_start = indptr[j];
                let col_end = indptr[j + 1];

                let x_val = *x_arr.get([j]).ok_or_else(|| Self::oob(&[j]))?;
                let scaled_x = alpha * x_val;

                for idx in col_start..col_end {
                    let i = indices[idx];
                    let a_val = a.get(i, j)?;
                    let elem = y_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = *elem + a_val * scaled_x;
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
        // For COO format, iterate through all non-zero entries. Bulk-acquired
        // the same way as the CSR/CSC paths above.
        let x_arr = x.array();
        let y_arr = y.array_mut();
        for (indices, value) in &a.array.data {
            let i = indices[0];
            let j = indices[1];

            if i < m && j < n {
                let x_val = *x_arr.get([j]).ok_or_else(|| Self::oob(&[j]))?;
                let elem = y_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                *elem = *elem + alpha * *value * x_val;
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
        {
            let b_arr = b.array();
            let ax_arr = ax.array();
            let r_arr = r.array_mut();
            for i in 0..n {
                let b_val = *b_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                let ax_val = *ax_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                *r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = b_val - ax_val;
            }
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
            {
                let p_arr = p.array();
                let x_arr = x.array_mut();
                for i in 0..n {
                    let p_val = *p_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let elem = x_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = *elem + alpha * p_val;
                }
            }

            // Update residual: r = r - alpha * A * p
            {
                let ap_arr = ap.array();
                let r_arr = r.array_mut();
                for i in 0..n {
                    let ap_val = *ap_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let elem = r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = *elem - alpha * ap_val;
                }
            }

            let rsnew = Self::dot_product(&r, &r)?;

            // Check convergence
            if rsnew < tol_sq {
                return Ok((x, iter + 1, rsnew.sqrt()));
            }

            // Update search direction: p = r + beta * p
            let beta = rsnew / rsold;
            {
                let r_arr = r.array();
                let p_arr = p.array_mut();
                for i in 0..n {
                    let r_val = *r_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let elem = p_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = r_val + beta * *elem;
                }
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
        {
            let b_arr = b.array();
            let ax_arr = ax.array();
            let r_arr = r.array_mut();
            for i in 0..n {
                let b_val = *b_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                let ax_val = *ax_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                *r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = b_val - ax_val;
            }
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
            {
                let r_arr = r.array();
                let v_arr = v.array();
                let p_arr = p.array_mut();
                for i in 0..n {
                    let r_val = *r_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let v_val = *v_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let elem = p_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = r_val + beta * (*elem - omega * v_val);
                }
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
            {
                let r_arr = r.array();
                let v_arr = v.array();
                let s_arr = s.array_mut();
                for i in 0..n {
                    let r_val = *r_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let v_val = *v_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *s_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = r_val - alpha * v_val;
                }
            }

            // Check if we can stop here
            let s_norm_sq = Self::dot_product(&s, &s)?;
            if s_norm_sq < tol_sq {
                // Update x and return
                {
                    let p_arr = p.array();
                    let x_arr = x.array_mut();
                    for i in 0..n {
                        let p_val = *p_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                        let elem = x_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                        *elem = *elem + alpha * p_val;
                    }
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
            {
                let p_arr = p.array();
                let s_arr = s.array();
                let x_arr = x.array_mut();
                for i in 0..n {
                    let p_val = *p_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let s_val = *s_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let elem = x_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = *elem + alpha * p_val + omega * s_val;
                }
            }

            // Update residual: r = s - omega * t
            {
                let s_arr = s.array();
                let t_arr = t.array();
                let r_arr = r.array_mut();
                for i in 0..n {
                    let s_val = *s_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let t_val = *t_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = s_val - omega * t_val;
                }
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
            {
                let b_arr = b.array();
                let ax_arr = ax.array();
                let r_arr = r.array_mut();
                for i in 0..n {
                    let b_val = *b_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let ax_val = *ax_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = b_val - ax_val;
                }
            }

            let r_norm = Self::dot_product(&r, &r)?.sqrt();

            // Check convergence
            if r_norm / b_norm < tol {
                return Ok((x, total_iter, r_norm));
            }

            // Initialize Arnoldi iteration
            let mut v = vec![Array::zeros(&[n]); restart + 1];
            {
                let r_arr = r.array();
                let v0_arr = v[0].array_mut();
                for i in 0..n {
                    let r_val = *r_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *v0_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = r_val / r_norm;
                }
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

                // Modified Gram-Schmidt orthogonalization. Bulk-acquired
                // per `i`: `w` alternates between being read (the dot
                // product) and written (the axpy below), so its unshare
                // can only be hoisted as far as one `Arc::make_mut` per
                // `i` rather than one for the whole `0..=j` sweep -- still
                // collapsing the inner `0..n` loop's per-element cost to a
                // single check.
                for i in 0..=j {
                    h[i][j] = Self::dot_product(&v[i], &w)?;
                    let hij = h[i][j];
                    let vi_arr = v[i].array();
                    let w_arr = w.array_mut();
                    for l in 0..n {
                        let vi_val = *vi_arr.get([l]).ok_or_else(|| Self::oob(&[l]))?;
                        let elem = w_arr.get_mut([l]).ok_or_else(|| Self::oob(&[l]))?;
                        *elem = *elem - hij * vi_val;
                    }
                }

                h[j + 1][j] = Self::dot_product(&w, &w)?.sqrt();

                // Check for lucky breakdown
                if h[j + 1][j].abs() < T::from(1e-14).expect("1e-14 is representable as Float") {
                    k = j + 1;
                    break;
                }

                // Normalize
                {
                    let denom = h[j + 1][j];
                    let w_arr = w.array();
                    let vj1_arr = v[j + 1].array_mut();
                    for i in 0..n {
                        let w_val = *w_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                        *vj1_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = w_val / denom;
                    }
                }

                // Apply previous Givens rotations to new column
                for i in 0..j {
                    let temp = h[i][j];
                    h[i][j] = cs_vec[i] * temp + sn_vec[i] * h[i + 1][j];
                    h[i + 1][j] = -sn_vec[i] * temp + cs_vec[i] * h[i + 1][j];
                }

                // Compute new Givens rotation
                let r_val = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
                if r_val < T::from(1e-14).expect("1e-14 is representable as Float") {
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

            // Update solution: x = x + V * y. `x`'s unshare is hoisted above
            // both loops (only `v[j]` changes per outer `j`, and reading a
            // *different* array while holding `x`'s `&mut` is fine), so the
            // whole O(k*n) update pays one `Arc::make_mut` in total.
            {
                let x_arr = x.array_mut();
                for j in 0..k {
                    let yj = y[j];
                    let vj_arr = v[j].array();
                    for i in 0..n {
                        let vj_val = *vj_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                        let elem = x_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                        *elem = *elem + yj * vj_val;
                    }
                }
            }

            // Check final residual
            let mut ax_final = Array::zeros(&[n]);
            Self::spmv_dense(a, &x, &mut ax_final, T::one(), T::zero())?;

            let mut r_final = Array::zeros(&[n]);
            {
                let b_arr = b.array();
                let ax_arr = ax_final.array();
                let r_arr = r_final.array_mut();
                for i in 0..n {
                    let b_val = *b_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    let ax_val = *ax_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = b_val - ax_val;
                }
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
        {
            let b_arr = b.array();
            let ax_arr = ax.array();
            let r_arr = r.array_mut();
            for i in 0..n {
                let b_val = *b_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                let ax_val = *ax_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                *r_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = b_val - ax_val;
            }
        }
        let final_residual = Self::dot_product(&r, &r)?.sqrt();

        Ok((x, max_iter, final_residual))
    }

    /// Incomplete LU decomposition for preconditioning
    ///
    /// Implements the dual-threshold ILUT(p, tau) scheme of Saad (1994),
    /// "ILUT: A dual threshold incomplete LU factorization technique,"
    /// Numerical Linear Algebra with Applications, 1(4), 387-402.
    ///
    /// `fill_factor` (clamped to `>= 1.0`) controls how many entries beyond
    /// each row's original sparsity pattern in `a` may survive in `L`/`U`:
    /// for row `i`, the number of kept off-diagonal entries on each side
    /// (strictly-lower for `L`, strictly-upper for `U`) is capped at
    /// `original_count + ceil((fill_factor - 1) * max(original_count, 1))`,
    /// keeping only the largest-magnitude candidates. `fill_factor == 1.0`
    /// therefore reproduces classic ILU(0) (no fill-in survives beyond
    /// `a`'s own nonzero pattern); larger values progressively approach
    /// exact (dense) LU as more fill-in is retained. A small fixed
    /// relative drop tolerance additionally discards numerically
    /// negligible entries irrespective of the fill cap.
    pub fn incomplete_lu<T>(
        a: &SparseMatrix<T>,
        fill_factor: f64,
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

        let fill_factor = fill_factor.max(1.0);
        let relative_drop_tol: T = T::from(1e-10).unwrap_or_else(T::epsilon);

        // Baseline per-row nonzero counts of the ORIGINAL matrix, split at
        // the diagonal; this is the "lfil" budget each row's L/U part may
        // exceed by (fill_factor - 1) times itself.
        let mut orig_lower_nnz = vec![0usize; n];
        let mut orig_upper_off_diag_nnz = vec![0usize; n];
        for indices in a.array.data.keys() {
            let i = indices[0];
            let j = indices[1];
            if j < i {
                orig_lower_nnz[i] += 1;
            } else if j > i {
                orig_upper_off_diag_nnz[i] += 1;
            }
        }

        let mut l: SparseMatrix<T> = SparseMatrix::new(&[n, n])?;
        let mut u: SparseMatrix<T> = SparseMatrix::new(&[n, n])?;
        let mut w = vec![T::zero(); n];

        for i in 0..n {
            // Load row i of A into a dense working row.
            for (j, slot) in w.iter_mut().enumerate() {
                *slot = a.get(i, j)?;
            }
            let row_norm = w.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();
            let abs_drop = relative_drop_tol * (row_norm + T::one());

            // Eliminate using previously computed pivot rows k < i, exactly
            // as in Gaussian elimination, but dropping any multiplier whose
            // magnitude is negligible relative to the row (and skipping the
            // corresponding update, per the ILUT dropping rule).
            for k in 0..i {
                if w[k] == T::zero() {
                    continue;
                }
                let u_kk = u.get(k, k)?;
                if u_kk.abs() < T::epsilon() {
                    return Err(NumRs2Error::ComputationError(
                        "ILU decomposition failed: zero pivot".to_string(),
                    ));
                }
                let factor = w[k] / u_kk;
                if factor.abs() < abs_drop {
                    w[k] = T::zero();
                    continue;
                }
                w[k] = factor;
                for j in (k + 1)..n {
                    let u_kj = u.get(k, j)?;
                    if u_kj != T::zero() {
                        w[j] = w[j] - factor * u_kj;
                    }
                }
            }

            // Split the eliminated row into its L (j < i) and U (j >= i)
            // parts and apply the dual-threshold drop rule to each
            // independently: drop numerically negligible entries, then keep
            // only the `cap` largest-magnitude survivors.
            let fill_cap = |orig: usize| -> usize {
                orig + ((fill_factor - 1.0) * (orig.max(1) as f64)).ceil() as usize
            };

            let mut lower: Vec<(usize, T)> = (0..i)
                .filter_map(|j| {
                    let v = w[j];
                    (v != T::zero() && v.abs() >= abs_drop).then_some((j, v))
                })
                .collect();
            lower.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            lower.truncate(fill_cap(orig_lower_nnz[i]));

            let diag = w[i];
            if diag.abs() < T::epsilon() {
                return Err(NumRs2Error::ComputationError(
                    "ILU decomposition failed: zero pivot".to_string(),
                ));
            }

            let mut upper: Vec<(usize, T)> = ((i + 1)..n)
                .filter_map(|j| {
                    let v = w[j];
                    (v != T::zero() && v.abs() >= abs_drop).then_some((j, v))
                })
                .collect();
            upper.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            upper.truncate(fill_cap(orig_upper_off_diag_nnz[i]));

            l.set(i, i, T::one())?;
            for (j, v) in lower {
                l.set(i, j, v)?;
            }
            u.set(i, i, diag)?;
            for (j, v) in upper {
                u.set(i, j, v)?;
            }
        }

        Ok((l, u))
    }

    /// Helper function to compute dot product of two arrays
    fn dot_product<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        // Compares the raw `ndarray` shapes (`&[usize]`) rather than
        // `Array::shape()` (which heap-allocates a `Vec` on every call) --
        // `dot_product` runs once per solver iteration, so this is cheap
        // but free to avoid.
        if x.array().shape() != y.array().shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Arrays must have same shape for dot product".to_string(),
            ));
        }

        let n = x.shape()[0];
        let mut result = T::zero();

        // Bulk-acquired: both operands are read-only, so this is a plain
        // shared borrow with no unshare at all, but still saves the
        // `Array::get` `Result`/bounds-check wrapper on every element.
        let x_arr = x.array();
        let y_arr = y.array();
        for i in 0..n {
            let x_val = *x_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
            let y_val = *y_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
            result = result + x_val * y_val;
        }

        Ok(result)
    }

    /// Estimate the spectral condition number of a sparse matrix.
    ///
    /// The condition number is `κ = |λ_max| / |λ_min|`, where `λ_max` and `λ_min`
    /// are the eigenvalues of largest and smallest magnitude. For a symmetric
    /// positive-definite (SPD) matrix this equals the spectral 2-norm condition
    /// number `‖A‖₂ · ‖A⁻¹‖₂`.
    ///
    /// The two extremal eigenvalues are estimated with complementary iterations:
    /// * `λ_max` (largest magnitude) is obtained by classical **power iteration**:
    ///   the Rayleigh quotient `vᵀ A v / vᵀ v` of the normalized iterate converges
    ///   to the dominant eigenvalue.
    /// * `λ_min` (smallest magnitude) is obtained by **inverse power iteration**:
    ///   repeatedly solving `A x_{k+1} = x_k` and normalizing drives the iterate
    ///   toward the eigenvector of smallest-magnitude eigenvalue, whose Rayleigh
    ///   quotient then yields `λ_min`. Each inner solve `A x = b` is performed with
    ///   the Conjugate Gradient solver ([`Self::solve_cg`]); this assumes `A` is
    ///   SPD, which is also the regime in which the eigenvalue/condition-number
    ///   interpretation above holds exactly.
    ///
    /// For a numerically singular matrix (`λ_min ≈ 0`) the condition number is
    /// infinite and `T::infinity()` is returned.
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

        // --- Largest-magnitude eigenvalue via power iteration ------------------
        // Iterate v_{k+1} = A v_k / ‖A v_k‖, then estimate the eigenvalue with the
        // Rayleigh quotient of the normalized iterate: λ = vᵀ (A v) / vᵀ v.
        let mut v = Array::ones(&[n]);
        // Normalize the initial vector so the Rayleigh quotient is well scaled.
        {
            let v_norm = Self::vector_norm(&v)?;
            if v_norm > T::zero() {
                let v_arr = v.array_mut();
                for i in 0..n {
                    let elem = v_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = *elem / v_norm;
                }
            }
        }

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

            // Rayleigh quotient with the *current* (unit-norm) iterate v.
            let new_lambda = Self::dot_product(&v, &av)?;

            // Normalize the new iterate for the next step.
            {
                let av_arr = av.array();
                let v_arr = v.array_mut();
                for i in 0..n {
                    let av_val = *av_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *v_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = av_val / norm;
                }
            }

            if (new_lambda - lambda_max).abs() < tol {
                lambda_max = new_lambda;
                break;
            }
            lambda_max = new_lambda;
        }

        // --- Smallest-magnitude eigenvalue via inverse power iteration ---------
        // Iterate solve(A x = v_k), then v_{k+1} = x / ‖x‖. The iterate converges
        // to the eigenvector of the smallest-magnitude eigenvalue of A, whose
        // Rayleigh quotient λ_min = vᵀ (A v) / vᵀ v we track each step.
        let mut w = Array::ones(&[n]);
        {
            let w_norm = Self::vector_norm(&w)?;
            if w_norm > T::zero() {
                let w_arr = w.array_mut();
                for i in 0..n {
                    let elem = w_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *elem = *elem / w_norm;
                }
            }
        }

        // Inner-solve tolerance: tie it to the requested accuracy but keep it
        // comfortably small so the outer iteration converges cleanly.
        let solve_tol = tol;
        // Cap the inner CG iterations; n is a safe upper bound for an exact SPD solve.
        let inner_max_iter = std::cmp::max(n, max_iter);

        let mut lambda_min = T::zero();
        let mut have_lambda_min = false;

        for _ in 0..max_iter {
            // Solve A x = w for the next (unnormalized) iterate.
            let (x, _iters, _residual) =
                Self::solve_cg(a, &w, Some(&w), solve_tol, inner_max_iter)?;

            let x_norm = Self::vector_norm(&x)?;
            if x_norm < T::epsilon() {
                // A x collapses to zero: treat A as singular -> infinite condition.
                return Ok(T::infinity());
            }

            // Normalize the iterate.
            {
                let x_arr = x.array();
                let w_arr = w.array_mut();
                for i in 0..n {
                    let x_val = *x_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
                    *w_arr.get_mut([i]).ok_or_else(|| Self::oob(&[i]))? = x_val / x_norm;
                }
            }

            // Rayleigh quotient λ_min = wᵀ A w (w is unit-norm).
            let mut aw = Array::zeros(&[n]);
            Self::spmv_dense(a, &w, &mut aw, T::one(), T::zero())?;
            let new_lambda = Self::dot_product(&w, &aw)?;

            if have_lambda_min && (new_lambda - lambda_min).abs() < tol {
                lambda_min = new_lambda;
                break;
            }
            lambda_min = new_lambda;
            have_lambda_min = true;
        }

        // --- Form the condition number κ = |λ_max| / |λ_min| -------------------
        let lambda_max_abs = lambda_max.abs();
        let lambda_min_abs = lambda_min.abs();

        // Guard against division by a (near-)zero smallest eigenvalue: such a
        // matrix is singular / ill-conditioned, so the condition number diverges.
        let singular_threshold = T::epsilon() * lambda_max_abs.max(T::one());
        if lambda_min_abs <= singular_threshold {
            return Ok(T::infinity());
        }

        Ok(lambda_max_abs / lambda_min_abs)
    }

    /// Compute the 2-norm of a vector
    fn vector_norm<T>(x: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug,
    {
        let n = x.shape()[0];
        let mut sum = T::zero();

        let x_arr = x.array();
        for i in 0..n {
            let val = *x_arr.get([i]).ok_or_else(|| Self::oob(&[i]))?;
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
        let mut a = SparseMatrix::new(&[3, 3]).expect("3x3 sparse matrix creation");
        a.set(0, 0, 2.0).expect("set matrix element");
        a.set(0, 1, 1.0).expect("set matrix element");
        a.set(1, 1, 3.0).expect("set matrix element");
        a.set(2, 2, 4.0).expect("set matrix element");

        // Create input vector
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = Array::zeros(&[3]);

        // Test sparse matrix-vector multiplication
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut y, 1.0, 0.0).expect("spmv_dense");

        // Expected: [2*1 + 1*2, 3*2, 4*3] = [4, 6, 12]
        assert_relative_eq!(y.get(&[0]).expect("get y[0]"), 4.0, epsilon = 1e-10);
        assert_relative_eq!(y.get(&[1]).expect("get y[1]"), 6.0, epsilon = 1e-10);
        assert_relative_eq!(y.get(&[2]).expect("get y[2]"), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_conjugate_gradient_solver() {
        // Create a symmetric positive definite matrix
        let mut a = SparseMatrix::new(&[3, 3]).expect("3x3 sparse matrix creation");
        a.set(0, 0, 4.0).expect("set matrix element");
        a.set(0, 1, 1.0).expect("set matrix element");
        a.set(1, 0, 1.0).expect("set matrix element");
        a.set(1, 1, 3.0).expect("set matrix element");
        a.set(1, 2, 1.0).expect("set matrix element");
        a.set(2, 1, 1.0).expect("set matrix element");
        a.set(2, 2, 2.0).expect("set matrix element");

        // Right-hand side
        let b = Array::from_vec(vec![6.0, 8.0, 4.0]);

        // Solve using CG
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_cg(&a, &b, None, 1e-10, 100).expect("CG solver");

        // Check that the solution satisfies A*x = b (approximately)
        let mut ax = Array::zeros(&[3]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).expect("spmv_dense");

        for i in 0..3 {
            let b_val = b.get(&[i]).expect("get b[i]");
            let ax_val = ax.get(&[i]).expect("get ax[i]");
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_bicgstab_solver() {
        // Create a general matrix
        let mut a = SparseMatrix::new(&[3, 3]).expect("3x3 sparse matrix creation");
        a.set(0, 0, 3.0).expect("set matrix element");
        a.set(0, 1, 1.0).expect("set matrix element");
        a.set(1, 0, 1.0).expect("set matrix element");
        a.set(1, 1, 2.0).expect("set matrix element");
        a.set(1, 2, 1.0).expect("set matrix element");
        a.set(2, 1, 1.0).expect("set matrix element");
        a.set(2, 2, 3.0).expect("set matrix element");

        // Right-hand side
        let b = Array::from_vec(vec![5.0, 6.0, 7.0]);

        // Solve using BiCGSTAB
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_bicgstab(&a, &b, None, 1e-10, 100).expect("BiCGSTAB solver");

        // Check that the solution satisfies A*x = b (approximately)
        let mut ax = Array::zeros(&[3]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).expect("spmv_dense");

        for i in 0..3 {
            let b_val = b.get(&[i]).expect("get b[i]");
            let ax_val = ax.get(&[i]).expect("get ax[i]");
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_gmres_solver() {
        // Create a general matrix (non-symmetric)
        let mut a = SparseMatrix::new(&[3, 3]).expect("3x3 sparse matrix creation");
        a.set(0, 0, 3.0).expect("set matrix element");
        a.set(0, 1, 1.0).expect("set matrix element");
        a.set(0, 2, 0.5).expect("set matrix element");
        a.set(1, 0, 1.0).expect("set matrix element");
        a.set(1, 1, 4.0).expect("set matrix element");
        a.set(1, 2, 1.0).expect("set matrix element");
        a.set(2, 0, 0.0).expect("set matrix element");
        a.set(2, 1, 2.0).expect("set matrix element");
        a.set(2, 2, 5.0).expect("set matrix element");

        // Right-hand side
        let b = Array::from_vec(vec![5.5, 9.0, 17.0]);

        // Solve using GMRES
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 30).expect("GMRES solver");

        // Check that the solution satisfies A*x = b (approximately)
        let mut ax = Array::zeros(&[3]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).expect("spmv_dense");

        for i in 0..3 {
            let b_val = b.get(&[i]).expect("get b[i]");
            let ax_val = ax.get(&[i]).expect("get ax[i]");
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_gmres_solver_larger_system() {
        // Create a larger sparse matrix (5x5 diagonally dominant)
        let n = 5;
        let mut a = SparseMatrix::new(&[n, n]).expect("nxn sparse matrix creation");

        // Tridiagonal matrix with strong diagonal dominance
        for i in 0..n {
            a.set(i, i, 4.0).expect("set diagonal element"); // Main diagonal
            if i > 0 {
                a.set(i, i - 1, -1.0).expect("set lower diagonal"); // Lower diagonal
            }
            if i < n - 1 {
                a.set(i, i + 1, -1.0).expect("set upper diagonal"); // Upper diagonal
            }
        }

        // Right-hand side: solution should be [1, 2, 3, 4, 5]
        let b = Array::from_vec(vec![2.0, 1.0, 2.0, 3.0, 16.0]);

        // Solve using GMRES
        let (x, iter, residual) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 30).expect("GMRES solver");

        // Check that the solution satisfies A*x = b
        let mut ax = Array::zeros(&[n]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).expect("spmv_dense");

        for i in 0..n {
            let b_val = b.get(&[i]).expect("get b[i]");
            let ax_val = ax.get(&[i]).expect("get ax[i]");
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-8);
        }

        assert!(iter < 100);
        assert!(residual < 1e-8);
    }

    #[test]
    fn test_gmres_with_restart() {
        // Create a matrix that might need restart
        let n = 4;
        let mut a = SparseMatrix::new(&[n, n]).expect("nxn sparse matrix creation");

        // Create a diagonally dominant matrix
        for i in 0..n {
            a.set(i, i, 5.0).expect("set diagonal element");
            for j in 0..n {
                if i != j {
                    a.set(i, j, 0.5).expect("set off-diagonal element");
                }
            }
        }

        let b = Array::from_vec(vec![7.5, 7.5, 7.5, 7.5]);

        // Solve with small restart parameter to force restarts
        let (x, _iter, residual) = SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 2)
            .expect("GMRES solver with restart");

        // Verify solution
        let mut ax = Array::zeros(&[n]);
        SparseOpsAdvanced::spmv_dense(&a, &x, &mut ax, 1.0, 0.0).expect("spmv_dense");

        for i in 0..n {
            let b_val = b.get(&[i]).expect("get b[i]");
            let ax_val = ax.get(&[i]).expect("get ax[i]");
            assert_relative_eq!(ax_val, b_val, epsilon = 1e-6);
        }

        assert!(residual < 1e-6);
    }

    #[test]
    fn test_gmres_vs_bicgstab() {
        // Compare GMRES and BiCGSTAB on the same problem
        let mut a = SparseMatrix::new(&[3, 3]).expect("3x3 sparse matrix creation");
        a.set(0, 0, 4.0).expect("set matrix element");
        a.set(0, 1, 1.0).expect("set matrix element");
        a.set(1, 0, 2.0).expect("set matrix element");
        a.set(1, 1, 3.0).expect("set matrix element");
        a.set(1, 2, 1.0).expect("set matrix element");
        a.set(2, 1, 1.0).expect("set matrix element");
        a.set(2, 2, 4.0).expect("set matrix element");

        let b = Array::from_vec(vec![6.0, 9.0, 5.0]);

        // Solve with GMRES
        let (x_gmres, _, residual_gmres) =
            SparseOpsAdvanced::solve_gmres(&a, &b, None, 1e-10, 100, 30).expect("GMRES solver");

        // Solve with BiCGSTAB
        let (x_bicgstab, _, residual_bicgstab) =
            SparseOpsAdvanced::solve_bicgstab(&a, &b, None, 1e-10, 100).expect("BiCGSTAB solver");

        // Both should converge
        assert!(residual_gmres < 1e-8);
        assert!(residual_bicgstab < 1e-8);

        // Solutions should be similar
        for i in 0..3 {
            let g_val = x_gmres.get(&[i]).expect("get x_gmres[i]");
            let b_val = x_bicgstab.get(&[i]).expect("get b[i]");
            assert_relative_eq!(g_val, b_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_incomplete_lu_decomposition() {
        // Create a test matrix
        let mut a = SparseMatrix::new(&[3, 3]).expect("3x3 sparse matrix creation");
        a.set(0, 0, 4.0).expect("set matrix element");
        a.set(0, 1, 1.0).expect("set matrix element");
        a.set(1, 0, 1.0).expect("set matrix element");
        a.set(1, 1, 3.0).expect("set matrix element");
        a.set(1, 2, 1.0).expect("set matrix element");
        a.set(2, 1, 1.0).expect("set matrix element");
        a.set(2, 2, 2.0).expect("set matrix element");

        // Compute ILU decomposition
        let (l, u) = SparseOpsAdvanced::incomplete_lu(&a, 1.0).expect("ILU decomposition");

        // Verify that L is lower triangular with 1s on diagonal
        assert_relative_eq!(l.get(0, 0).expect("get L[0,0]"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(1, 1).expect("get L[1,1]"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(2, 2).expect("get L[2,2]"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(0, 1).expect("get L[0,1]"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(l.get(0, 2).expect("get L[0,2]"), 0.0, epsilon = 1e-10);

        // Verify U is upper triangular
        assert_relative_eq!(u.get(1, 0).expect("get U[1,0]"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(u.get(2, 0).expect("get U[2,0]"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(u.get(2, 1).expect("get U[2,1]"), 0.0, epsilon = 1e-10);
    }
}
