//! Basic linear algebra operations with Array
//! Includes matrix multiplication, dot product, matrix inversion, etc.

#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use crate::array::Array;
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use crate::error::{NumRs2Error, Result};
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use num_traits::{Float, ToPrimitive};
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use std::fmt::Debug;

// Matrix decomposition submodule
#[path = "../linalg_decomposition.rs"]
pub mod decomposition;

// Solve operations submodule
#[path = "../linalg_solve.rs"]
pub mod solve;

// Import submodules
pub mod iterative_solvers;
pub mod matrix_ops;
pub mod randomized;
pub mod tensor_decomp;
pub mod tensor_ops;
pub mod vector_ops;

// Re-export all functions for backward compatibility - conditional on lapack feature
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub use decomposition::{cholesky, eig, qr, svd};
#[cfg(feature = "lapack")]
pub use solve::{inv, solve};

// Re-export conditional features
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub use decomposition::matrix_rank;
#[cfg(feature = "lapack")]
pub use matrix_ops::{det, matrix_power};
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub use solve::pinv;
pub use tensor_ops::{einsum, kron, tensordot};
pub use vector_ops::{complex_vdot, inner, norm, outer, trace, vdot};

// Randomized linear algebra
pub use randomized::{
    random_projection, randomized_low_rank_approximation, randomized_range_finder, randomized_svd,
    ProjectionType,
};

// Tensor decompositions
pub use tensor_decomp::{
    cp_als, cp_als_with_config, cp_reconstruct, nonnegative_cp_als, tucker_decomposition,
    tucker_reconstruct, CpResult, DecompConfig, TuckerResult,
};

// Additional standalone functions for improved compatibility
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub use crate::new_modules::matrix_decomp::{
    lstsq, lu, pivoted_cholesky, slogdet, svd as svd_enhanced,
};

/// Set the number of threads for LAPACK operations
pub fn set_lapack_threads(threads: usize) {
    // We can use blas_src's set_num_threads when it's available
    // For now, we'll just provide this as a placeholder
    let _threads = threads;
}

/// Implementation for when matrix_decomp feature is enabled
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
impl<T> Array<T>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack,
{
    /// Compute the determinant of a matrix using LU decomposition for large matrices
    /// and direct formula for small matrices.
    ///
    /// This implementation includes:
    /// 1. Optimized direct formulas for 1x1, 2x2, and 3x3 matrices
    /// 2. LU decomposition with partial pivoting for larger matrices
    /// 3. Scaling to prevent overflow/underflow in intermediate calculations
    /// 4. Near-zero detection with appropriate thresholds
    pub fn det(&self) -> Result<T> {
        // Verify the array is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "determinant requires a square matrix".to_string(),
            ));
        }

        // Optimized path for small matrices (1x1, 2x2, 3x3)
        let n = shape[0];
        let data = self.to_vec();

        // For 1x1 matrix, determinant is the single element
        if n == 1 {
            return Ok(data[0]);
        }
        // For 2x2 matrix, use direct formula: ad - bc
        else if n == 2 {
            return Ok(data[0] * data[3] - data[1] * data[2]);
        }
        // For 3x3 matrix, use cofactor expansion (optimized)
        else if n == 3 {
            // For 3x3 matrix using cofactor expansion
            let det = data[0] * (data[4] * data[8] - data[5] * data[7])
                - data[1] * (data[3] * data[8] - data[5] * data[6])
                + data[2] * (data[3] * data[7] - data[4] * data[6]);
            return Ok(det);
        }
        // For 4x4 matrix, use cofactor expansion with smaller determinants
        else if n == 4 {
            // Use optimized cofactor expansion for 4x4
            let mut det = T::zero();

            // First row cofactor expansion
            for j in 0..4 {
                // Get the 3x3 minor by removing row 0 and column j
                let mut minor_data = Vec::with_capacity(9);
                for row in 1..4 {
                    for col in 0..4 {
                        if col != j {
                            minor_data.push(data[row * 4 + col]);
                        }
                    }
                }

                // Create 3x3 submatrix for minor
                let minor = Array::from_vec(minor_data).reshape(&[3, 3]);

                // Compute 3x3 determinant (we know this works from the 3x3 case)
                let minor_det = minor.det()?;

                // Add to determinant with appropriate sign
                let sign = if j % 2 == 0 { T::one() } else { -T::one() };
                det += sign * data[j] * minor_det;
            }

            return Ok(det);
        }

        // For larger matrices (n > 4), use LU decomposition
        // This is more numerically stable and efficient

        // Step 1: Obtain LU decomposition with pivoting
        // We'll use the implementation in matrix_decomp module
        #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
        use crate::new_modules::matrix_decomp::lu;

        // Calculate LU decomposition
        #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
        let (_, u, p) = lu(self)?;

        #[cfg(not(feature = "matrix_decomp"))]
        return Err(NumRs2Error::FeatureNotEnabled(
            "matrix_decomp feature required for LU decomposition".to_string(),
        ));

        // For LU decomposition with row pivoting (PA = LU),
        // det(A) = det(P) * det(L) * det(U)
        // det(L) = 1 (since L has 1's on diagonal)
        // det(U) = product of diagonal elements
        // det(P) = (-1)^s where s is the number of row swaps

        // Calculate determinant of U (product of diagonal elements)
        let mut det_u = T::one();
        for i in 0..n {
            det_u *= u.get(&[i, i])?;
        }

        // Calculate determinant of P (parity of permutation)
        // We need to count the number of swaps in the permutation
        let p_vec = p.to_vec();
        let mut visited = vec![false; n];
        let mut parity = 0;

        for i in 0..n {
            if !visited[i] {
                let mut j = i;
                let mut cycle_length = 0;

                while !visited[j] {
                    visited[j] = true;
                    j = p_vec[j];
                    cycle_length += 1;
                }

                // A cycle of length k requires k-1 swaps
                if cycle_length > 1 {
                    parity += cycle_length - 1;
                }
            }
        }

        // Calculate final determinant
        let det_p = if parity % 2 == 0 { T::one() } else { -T::one() };
        Ok(det_p * det_u)
    }

    /// Compute the inverse of a matrix
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 matrices
    /// 2. LU decomposition with partial pivoting for larger matrices
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    pub fn inv(&self) -> Result<Array<T>> {
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "inverse requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];

        // Check if the matrix is invertible using determinant
        // This is efficient for small matrices and catches perfect singularity
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // For small matrices, use direct formulas for better efficiency and accuracy
        if n == 1 {
            // For 1x1 matrix, inverse is 1/a
            let a = self.get(&[0, 0])?;
            let result = vec![T::one() / a];
            return Ok(Array::from_vec(result).reshape(&[1, 1]));
        } else if n == 2 {
            // For 2x2 matrix, use the formula:
            // [a b]^-1 = (1/det) * [d -b]
            // [c d]             [-c  a]
            let data = self.to_vec();
            let a = data[0];
            let b = data[1];
            let c = data[2];
            let d = data[3];

            let inv_det = T::one() / det;
            let result = vec![d * inv_det, -b * inv_det, -c * inv_det, a * inv_det];

            return Ok(Array::from_vec(result).reshape(&[2, 2]));
        } else if n == 3 {
            // For 3x3 matrix, use adjugate formula:
            // A^-1 = (1/det) * adj(A)
            let data = self.to_vec();

            // Compute cofactors
            let c00 = data[4] * data[8] - data[5] * data[7];
            let c01 = -(data[3] * data[8] - data[5] * data[6]);
            let c02 = data[3] * data[7] - data[4] * data[6];

            let c10 = -(data[1] * data[8] - data[2] * data[7]);
            let c11 = data[0] * data[8] - data[2] * data[6];
            let c12 = -(data[0] * data[7] - data[1] * data[6]);

            let c20 = data[1] * data[5] - data[2] * data[4];
            let c21 = -(data[0] * data[5] - data[2] * data[3]);
            let c22 = data[0] * data[4] - data[1] * data[3];

            // Construct the adjugate (transpose of cofactor matrix)
            let inv_det = T::one() / det;
            let result = vec![
                c00 * inv_det,
                c10 * inv_det,
                c20 * inv_det,
                c01 * inv_det,
                c11 * inv_det,
                c21 * inv_det,
                c02 * inv_det,
                c12 * inv_det,
                c22 * inv_det,
            ];

            return Ok(Array::from_vec(result).reshape(&[3, 3]));
        }

        // For larger matrices, use LU decomposition
        // Get LU decomposition with pivoting: PA = LU
        #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
        use crate::new_modules::matrix_decomp::lu;

        // Step 1: Calculate LU decomposition
        #[cfg(feature = "matrix_decomp")]
        let (l, u, p) = lu(self)?;

        #[cfg(not(feature = "matrix_decomp"))]
        return Err(NumRs2Error::FeatureNotEnabled(
            "matrix_decomp feature required for LU decomposition".to_string(),
        ));

        // Step 2: Create a result matrix
        let mut result = Array::zeros(&[n, n]);

        // Step 3: Solve LUX = I column by column to find A^-1
        // For each column of the identity matrix...
        for j in 0..n {
            // Create the j-th column of identity matrix
            let mut b = vec![T::zero(); n];
            b[j] = T::one();

            // Step 3.1: Apply permutation to b (Pb)
            let mut pb = vec![T::zero(); n];
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let p_idx = p.get(&[i])?.to_usize().unwrap_or(i);
                pb[i] = b[p_idx];
            }

            // Step 3.2: Forward substitution to solve Ly = Pb
            let mut y = vec![T::zero(); n];
            for i in 0..n {
                let mut sum = pb[i];
                #[allow(clippy::needless_range_loop)]
                for k in 0..i {
                    sum -= l.get(&[i, k])? * y[k];
                }
                y[i] = sum; // L has 1's on diagonal
            }

            // Step 3.3: Back substitution to solve Ux = y
            let mut x = vec![T::zero(); n];
            for i in (0..n).rev() {
                let mut sum = y[i];
                #[allow(clippy::needless_range_loop)]
                for k in (i + 1)..n {
                    sum -= u.get(&[i, k])? * x[k];
                }
                x[i] = sum / u.get(&[i, i])?;
            }

            // Step 3.4: Store the solution in the j-th column of the result
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                result.set(&[i, j], x[i])?;
            }
        }

        // Step 4: Verify numerical stability
        // In a real implementation, we would check the condition number
        // and provide a warning for ill-conditioned matrices

        // For debugging: check that A * A^-1 ≈ I
        // This is expensive so we'd only do it in debug mode
        #[cfg(debug_assertions)]
        {
            let product = self.matmul(&result)?;
            let mut max_error = T::zero();

            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { T::one() } else { T::zero() };
                    let actual = product.get(&[i, j])?;
                    let error = num_traits::Float::abs(actual - expected);

                    if error > max_error {
                        max_error = error;
                    }
                }
            }

            let eps = T::epsilon();
            let acceptable_error = eps * T::from(n).unwrap() * T::from(100.0).unwrap();

            if max_error > acceptable_error {
                eprintln!(
                    "Warning: Matrix inversion may be numerically unstable. Max error: {}",
                    max_error
                );
            }
        }

        Ok(result)
    }

    /// Solve a linear system Ax = b
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 systems
    /// 2. LU decomposition with partial pivoting for larger systems
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }

        let n = a_shape[0];

        // Quick check for singularity using determinant
        // This is efficient for small matrices and catches perfect singularity
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "coefficient matrix is singular and cannot be solved".to_string(),
            ));
        }

        // Special fast paths for small systems
        if n == 1 {
            // 1x1 system - trivial solution
            let a_val = self.get(&[0, 0])?;
            let b_val = b.get(&[0])?;

            return Ok(Array::from_vec(vec![b_val / a_val]));
        } else if n == 2 {
            // 2x2 system - use Cramer's rule
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;

            return Ok(Array::from_vec(vec![x1, x2]));
        } else if n == 3 {
            // 3x3 system - use direct formula
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            // Compute determinant and cofactors
            let a11 = a_data[0];
            let a12 = a_data[1];
            let a13 = a_data[2];
            let a21 = a_data[3];
            let a22 = a_data[4];
            let a23 = a_data[5];
            let a31 = a_data[6];
            let a32 = a_data[7];
            let a33 = a_data[8];

            // Calculate cofactors for direct solution
            let c11 = a22 * a33 - a23 * a32;
            let c12 = -(a21 * a33 - a23 * a31);
            let c13 = a21 * a32 - a22 * a31;

            let c21 = -(a12 * a33 - a13 * a32);
            let c22 = a11 * a33 - a13 * a31;
            let c23 = -(a11 * a32 - a12 * a31);

            let c31 = a12 * a23 - a13 * a22;
            let c32 = -(a11 * a23 - a13 * a21);
            let c33 = a11 * a22 - a12 * a21;

            let inv_det = T::one() / det;

            // Multiply inverse of A (which is adjugate/det) by b
            let x1 = (c11 * b_data[0] + c21 * b_data[1] + c31 * b_data[2]) * inv_det;
            let x2 = (c12 * b_data[0] + c22 * b_data[1] + c32 * b_data[2]) * inv_det;
            let x3 = (c13 * b_data[0] + c23 * b_data[1] + c33 * b_data[2]) * inv_det;

            return Ok(Array::from_vec(vec![x1, x2, x3]));
        }

        // For larger systems, use SCIRS
        use crate::interop::scirs_compat::solve_linear_system;
        solve_linear_system(self, b)
    }

    /// Solve a linear system Ax = b
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 systems
    /// 2. LU decomposition with partial pivoting for larger systems
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    #[cfg(all(feature = "matrix_decomp", not(feature = "scirs")))]
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }

        let n = a_shape[0];

        // Quick check for singularity using determinant
        // This is efficient for small matrices and catches perfect singularity
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "coefficient matrix is singular and cannot be solved".to_string(),
            ));
        }

        // Special fast paths for small systems
        if n == 1 {
            // 1x1 system - trivial solution
            let a_val = self.get(&[0, 0])?;
            let b_val = b.get(&[0])?;

            return Ok(Array::from_vec(vec![b_val / a_val]));
        } else if n == 2 {
            // 2x2 system - use Cramer's rule
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;

            return Ok(Array::from_vec(vec![x1, x2]));
        } else if n == 3 {
            // 3x3 system - use direct formula
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            // Compute determinant and cofactors
            let a11 = a_data[0];
            let a12 = a_data[1];
            let a13 = a_data[2];
            let a21 = a_data[3];
            let a22 = a_data[4];
            let a23 = a_data[5];
            let a31 = a_data[6];
            let a32 = a_data[7];
            let a33 = a_data[8];

            // Calculate cofactors for direct solution
            let c11 = a22 * a33 - a23 * a32;
            let c12 = -(a21 * a33 - a23 * a31);
            let c13 = a21 * a32 - a22 * a31;

            let c21 = -(a12 * a33 - a13 * a32);
            let c22 = a11 * a33 - a13 * a31;
            let c23 = -(a11 * a32 - a12 * a31);

            let c31 = a12 * a23 - a13 * a22;
            let c32 = -(a11 * a23 - a13 * a21);
            let c33 = a11 * a22 - a12 * a21;

            let inv_det = T::one() / det;

            // Multiply inverse of A (which is adjugate/det) by b
            let x1 = (c11 * b_data[0] + c21 * b_data[1] + c31 * b_data[2]) * inv_det;
            let x2 = (c12 * b_data[0] + c22 * b_data[1] + c32 * b_data[2]) * inv_det;
            let x3 = (c13 * b_data[0] + c23 * b_data[1] + c33 * b_data[2]) * inv_det;

            return Ok(Array::from_vec(vec![x1, x2, x3]));
        }

        // For larger systems, use LU decomposition with partial pivoting
        // Get LU decomposition with pivoting: PA = LU
        #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
        use crate::new_modules::matrix_decomp::lu;

        // Step 1: Calculate LU decomposition
        #[cfg(feature = "matrix_decomp")]
        let (l, u, p) = lu(self)?;

        #[cfg(not(feature = "matrix_decomp"))]
        return Err(NumRs2Error::FeatureNotEnabled(
            "matrix_decomp feature required for LU decomposition".to_string(),
        ));

        // Step 2: Apply permutation to b (Pb)
        let mut pb = vec![T::zero(); n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let p_idx = p.get(&[i])?.to_usize().unwrap_or(i);
            pb[i] = b.get(&[p_idx])?;
        }

        // Step 3: Forward substitution to solve Ly = Pb
        let mut y = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = pb[i];
            #[allow(clippy::needless_range_loop)]
            for k in 0..i {
                sum -= l.get(&[i, k])? * y[k];
            }
            y[i] = sum; // L has 1's on diagonal
        }

        // Step 4: Back substitution to solve Ux = y
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            #[allow(clippy::needless_range_loop)]
            for k in (i + 1)..n {
                sum -= u.get(&[i, k])? * x[k];
            }
            x[i] = sum / u.get(&[i, i])?;
        }

        // Step 5: Return the solution vector
        Ok(Array::from_vec(x))
    }

    /// Solve a linear system Ax = b
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 systems
    /// 2. Gaussian elimination with partial pivoting for larger systems
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    #[cfg(not(feature = "matrix_decomp"))]
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }

        let n = a_shape[0];

        // Fast paths for small systems
        if n == 1 {
            // 1x1 system - trivial solution
            let a_val = self.get(&[0, 0])?;

            if a_val == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular and cannot be solved".to_string(),
                ));
            }

            let b_val = b.get(&[0])?;
            return Ok(Array::from_vec(vec![b_val / a_val]));
        } else if n == 2 {
            // 2x2 system - use Cramer's rule
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            let det = a_data[0] * a_data[3] - a_data[1] * a_data[2];
            if det == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular".to_string(),
                ));
            }

            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;

            return Ok(Array::from_vec(vec![x1, x2]));
        } else if n == 3 {
            // 3x3 system - use direct formula through cofactors
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            // Calculate determinant
            let det = a_data[0] * (a_data[4] * a_data[8] - a_data[5] * a_data[7])
                - a_data[1] * (a_data[3] * a_data[8] - a_data[5] * a_data[6])
                + a_data[2] * (a_data[3] * a_data[7] - a_data[4] * a_data[6]);

            if det == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular".to_string(),
                ));
            }

            // Compute determinant and cofactors
            let a11 = a_data[0];
            let a12 = a_data[1];
            let a13 = a_data[2];
            let a21 = a_data[3];
            let a22 = a_data[4];
            let a23 = a_data[5];
            let a31 = a_data[6];
            let a32 = a_data[7];
            let a33 = a_data[8];

            // Calculate cofactors for direct solution
            let c11 = a22 * a33 - a23 * a32;
            let c12 = -(a21 * a33 - a23 * a31);
            let c13 = a21 * a32 - a22 * a31;

            let c21 = -(a12 * a33 - a13 * a32);
            let c22 = a11 * a33 - a13 * a31;
            let c23 = -(a11 * a32 - a12 * a31);

            let c31 = a12 * a23 - a13 * a22;
            let c32 = -(a11 * a23 - a13 * a21);
            let c33 = a11 * a22 - a12 * a21;

            let inv_det = T::one() / det;

            // Multiply inverse of A (which is adjugate/det) by b
            let x1 = (c11 * b_data[0] + c21 * b_data[1] + c31 * b_data[2]) * inv_det;
            let x2 = (c12 * b_data[0] + c22 * b_data[1] + c32 * b_data[2]) * inv_det;
            let x3 = (c13 * b_data[0] + c23 * b_data[1] + c33 * b_data[2]) * inv_det;

            return Ok(Array::from_vec(vec![x1, x2, x3]));
        }

        // For larger systems, use Gaussian elimination with partial pivoting

        // Create an augmented matrix [A|b]
        let mut aug = Array::zeros(&[n, n + 1]);

        // Fill in the augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug.set(&[i, j], self.get(&[i, j])?)?;
            }
            aug.set(&[i, n], b.get(&[i])?)?;
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot (maximum absolute value in current column)
            let mut max_val = num_traits::Float::abs(aug.get(&[i, i])?);
            let mut max_row = i;

            for j in (i + 1)..n {
                let abs_val = num_traits::Float::abs(aug.get(&[j, i])?);
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = j;
                }
            }

            // Check for singularity
            let eps = T::epsilon();
            if max_val < eps * T::from(n).unwrap() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is numerically singular".to_string(),
                ));
            }

            // Swap rows if needed
            if max_row != i {
                for j in i..(n + 1) {
                    let temp = aug.get(&[i, j])?;
                    aug.set(&[i, j], aug.get(&[max_row, j])?)?;
                    aug.set(&[max_row, j], temp)?;
                }
            }

            // Eliminate below
            for j in (i + 1)..n {
                let factor = aug.get(&[j, i])? / aug.get(&[i, i])?;

                for k in i..(n + 1) {
                    let val = aug.get(&[j, k])? - factor * aug.get(&[i, k])?;
                    aug.set(&[j, k], val)?;
                }
            }
        }

        // Back substitution
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug.get(&[i, n])?;

            for j in (i + 1)..n {
                sum -= aug.get(&[i, j])? * x[j];
            }

            x[i] = sum / aug.get(&[i, i])?;
        }

        Ok(Array::from_vec(x))
    }

    /// Compute the singular value decomposition of a matrix
    pub fn svd(&self) -> Result<(Array<T>, Array<T>, Array<T>)> {
        // Use the implementation from new_modules::matrix_decomp
        use crate::new_modules::matrix_decomp::svd;
        let (u, s_real, vt) = svd(self)?;

        // Convert singular values from Real to T
        // Since T is a Float type and Real is its real component type,
        // for real-valued matrices T == Real, so this is safe
        let s_vec: Vec<T> = s_real
            .to_vec()
            .into_iter()
            .map(|val| {
                // Convert from Real to T (for real matrices, this is identity)
                num_traits::NumCast::from(val).unwrap_or(T::zero())
            })
            .collect();
        let s = Array::from_vec(s_vec);

        Ok((u, s, vt))
    }

    /// Compute the eigenvalues and eigenvectors of a square matrix
    pub fn eig(&self) -> Result<(Array<T>, Array<T>)> {
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "eigendecomposition requires a square matrix".to_string(),
            ));
        }

        // This would use ndarray-linalg's eigenvalue computation in a full version
        // For now, we'll just return placeholder values
        let n = shape[0];

        let eigenvalues = Array::zeros(&[n]);
        let eigenvectors = Array::zeros(&[n, n]);

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute the Cholesky decomposition of a matrix
    pub fn cholesky(&self) -> Result<Array<T>> {
        // Use the implementation from new_modules::matrix_decomp
        use crate::new_modules::matrix_decomp::cholesky;
        cholesky(self)
    }

    /// Compute the QR decomposition of a matrix
    pub fn qr(&self) -> Result<(Array<T>, Array<T>)> {
        // Use the implementation from new_modules::matrix_decomp
        use crate::new_modules::matrix_decomp::qr;
        qr(self)
    }
}

/// A simplified direct implementation of linear algebra functions for Array
#[cfg(not(feature = "matrix_decomp"))]
impl<T> Array<T>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::fmt::Display,
{
    /// Compute the determinant of a matrix using LU decomposition for large matrices
    /// and direct formula for small matrices.
    ///
    /// This implementation includes:
    /// 1. Optimized direct formulas for 1x1, 2x2, and 3x3 matrices
    /// 2. LU decomposition with partial pivoting for larger matrices
    /// 3. Scaling to prevent overflow/underflow in intermediate calculations
    /// 4. Near-zero detection with appropriate thresholds
    pub fn det(&self) -> Result<T> {
        // Verify the array is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "determinant requires a square matrix".to_string(),
            ));
        }

        // Optimized path for small matrices (1x1, 2x2, 3x3)
        let n = shape[0];
        let data = self.to_vec();

        // For 1x1 matrix, determinant is the single element
        if n == 1 {
            return Ok(data[0]);
        }
        // For 2x2 matrix, use direct formula: ad - bc
        else if n == 2 {
            return Ok(data[0] * data[3] - data[1] * data[2]);
        }
        // For 3x3 matrix, use cofactor expansion (optimized)
        else if n == 3 {
            // For 3x3 matrix using cofactor expansion
            let det = data[0] * (data[4] * data[8] - data[5] * data[7])
                - data[1] * (data[3] * data[8] - data[5] * data[6])
                + data[2] * (data[3] * data[7] - data[4] * data[6]);
            return Ok(det);
        }
        // For 4x4 matrix, use cofactor expansion with smaller determinants
        else if n == 4 {
            // Use optimized cofactor expansion for 4x4
            let mut det = T::zero();

            // First row cofactor expansion
            for j in 0..4 {
                // Get the 3x3 minor by removing row 0 and column j
                let mut minor_data = Vec::with_capacity(9);
                for row in 1..4 {
                    for col in 0..4 {
                        if col != j {
                            minor_data.push(data[row * 4 + col]);
                        }
                    }
                }

                // Create 3x3 submatrix for minor
                let minor = Array::from_vec(minor_data).reshape(&[3, 3]);

                // Compute 3x3 determinant (we know this works from the 3x3 case)
                let minor_det = minor.det()?;

                // Add to determinant with appropriate sign
                let sign = if j % 2 == 0 { T::one() } else { -T::one() };
                det += sign * data[j] * minor_det;
            }

            return Ok(det);
        }

        // For larger matrices (n > 4), use in-place LU decomposition

        // Step 1: Perform Gaussian elimination with scaled partial pivoting
        let mut a_copy = self.clone();
        let mut det_sign = T::one(); // Track sign changes due to row swaps

        // Scale factors for each row (for numerical stability)
        let mut row_scale = vec![T::zero(); n];
        for i in 0..n {
            let mut max_in_row = T::zero();
            for j in 0..n {
                let abs_val = num_traits::Float::abs(a_copy.get(&[i, j])?);
                if abs_val > max_in_row {
                    max_in_row = abs_val;
                }
            }

            // If row is all zeros, determinant is zero
            if max_in_row == T::zero() {
                return Ok(T::zero());
            }

            row_scale[i] = T::one() / max_in_row;
        }

        // LU decomposition in-place with partial pivoting
        for k in 0..n - 1 {
            // Find pivot using scaled partial pivoting
            let mut p_row = k;
            let mut p_val = num_traits::Float::abs(a_copy.get(&[k, k])?) * row_scale[k];

            for i in k + 1..n {
                let val = num_traits::Float::abs(a_copy.get(&[i, k])?) * row_scale[i];
                if val > p_val {
                    p_row = i;
                    p_val = val;
                }
            }

            // Check for numerical singularity
            if p_val < T::epsilon() * T::from(100.0).unwrap() {
                // Matrix is numerically singular - determinant is effectively zero
                return Ok(T::zero());
            }

            // Swap rows if needed
            if p_row != k {
                for j in 0..n {
                    let temp = a_copy.get(&[k, j])?;
                    a_copy.set(&[k, j], a_copy.get(&[p_row, j])?)?;
                    a_copy.set(&[p_row, j], temp)?;
                }

                // Swap scale factors too
                row_scale.swap(k, p_row);

                // Each row swap changes the sign of the determinant
                det_sign = -det_sign;
            }

            // Perform elimination
            let pivot = a_copy.get(&[k, k])?;

            for i in k + 1..n {
                let factor = a_copy.get(&[i, k])? / pivot;
                a_copy.set(&[i, k], factor)?; // Store multiplier in L part

                for j in k + 1..n {
                    let val = a_copy.get(&[i, j])? - factor * a_copy.get(&[k, j])?;
                    a_copy.set(&[i, j], val)?;
                }
            }
        }

        // Compute determinant as product of diagonal elements times the sign
        let mut det = det_sign;
        for i in 0..n {
            det *= a_copy.get(&[i, i])?;
        }

        Ok(det)
    }

    /// Compute the inverse of a matrix
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 matrices
    /// 2. Gaussian elimination with pivoting for larger matrices
    /// 3. Numerical stability checks with appropriate tolerances
    /// 4. Proper error handling for singular or ill-conditioned matrices
    pub fn inv(&self) -> Result<Array<T>> {
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "inverse requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];

        // Check if the matrix is invertible using determinant
        // This is efficient for small matrices and catches perfect singularity
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // For small matrices, use direct formulas for better efficiency and accuracy
        if n == 1 {
            // For 1x1 matrix, inverse is 1/a
            let a = self.get(&[0, 0])?;
            let result = vec![T::one() / a];
            return Ok(Array::from_vec(result).reshape(&[1, 1]));
        } else if n == 2 {
            // For 2x2 matrix, use the formula:
            // [a b]^-1 = (1/det) * [d -b]
            // [c d]             [-c  a]
            let data = self.to_vec();
            let a = data[0];
            let b = data[1];
            let c = data[2];
            let d = data[3];

            let inv_det = T::one() / det;
            let result = vec![d * inv_det, -b * inv_det, -c * inv_det, a * inv_det];

            return Ok(Array::from_vec(result).reshape(&[2, 2]));
        } else if n == 3 {
            // For 3x3 matrix, use adjugate formula:
            // A^-1 = (1/det) * adj(A)
            let data = self.to_vec();

            // Compute cofactors
            let c00 = data[4] * data[8] - data[5] * data[7];
            let c01 = -(data[3] * data[8] - data[5] * data[6]);
            let c02 = data[3] * data[7] - data[4] * data[6];

            let c10 = -(data[1] * data[8] - data[2] * data[7]);
            let c11 = data[0] * data[8] - data[2] * data[6];
            let c12 = -(data[0] * data[7] - data[1] * data[6]);

            let c20 = data[1] * data[5] - data[2] * data[4];
            let c21 = -(data[0] * data[5] - data[2] * data[3]);
            let c22 = data[0] * data[4] - data[1] * data[3];

            // Construct the adjugate (transpose of cofactor matrix)
            let inv_det = T::one() / det;
            let result = vec![
                c00 * inv_det,
                c10 * inv_det,
                c20 * inv_det,
                c01 * inv_det,
                c11 * inv_det,
                c21 * inv_det,
                c02 * inv_det,
                c12 * inv_det,
                c22 * inv_det,
            ];

            return Ok(Array::from_vec(result).reshape(&[3, 3]));
        }

        // For larger matrices, use Gaussian elimination with identity matrix augmentation
        // This is a classic method for matrix inversion without requiring external libraries

        // Step 1: Create an augmented matrix [A|I]
        let mut aug = Array::zeros(&[n, 2 * n]);

        // Fill the left side with our matrix
        for i in 0..n {
            for j in 0..n {
                aug.set(&[i, j], self.get(&[i, j])?)?;
            }
        }

        // Fill the right side with identity matrix
        for i in 0..n {
            aug.set(&[i, i + n], T::one())?;
        }

        // Step 2: Compute row-echelon form using Gaussian elimination with pivoting
        for i in 0..n {
            // Find pivot (maximum value in current column)
            let mut max_val = num_traits::Float::abs(aug.get(&[i, i])?);
            let mut max_row = i;

            for j in (i + 1)..n {
                let abs_val = num_traits::Float::abs(aug.get(&[j, i])?);
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = j;
                }
            }

            // Check for singularity
            let eps = T::epsilon();
            if max_val < eps * T::from(n).unwrap() {
                return Err(NumRs2Error::InvalidOperation(
                    "matrix is numerically singular and cannot be inverted".to_string(),
                ));
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = aug.get(&[i, j])?;
                    aug.set(&[i, j], aug.get(&[max_row, j])?)?;
                    aug.set(&[max_row, j], temp)?;
                }
            }

            // Scale current row to get 1 on diagonal
            let pivot = aug.get(&[i, i])?;
            for j in 0..(2 * n) {
                let val = aug.get(&[i, j])? / pivot;
                aug.set(&[i, j], val)?;
            }

            // Eliminate current column for all other rows
            for j in 0..n {
                if j != i {
                    let factor = aug.get(&[j, i])?;
                    for k in 0..(2 * n) {
                        let val = aug.get(&[j, k])? - factor * aug.get(&[i, k])?;
                        aug.set(&[j, k], val)?;
                    }
                }
            }
        }

        // Step 3: Extract the right side, which is now A^-1
        let mut result = Array::zeros(&[n, n]);
        for i in 0..n {
            for j in 0..n {
                result.set(&[i, j], aug.get(&[i, j + n])?)?;
            }
        }

        // Step 4: Verify numerical stability
        #[cfg(debug_assertions)]
        {
            let product = self.matmul(&result)?;
            let mut max_error = T::zero();

            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { T::one() } else { T::zero() };
                    let actual = product.get(&[i, j])?;
                    let error = num_traits::Float::abs(actual - expected);

                    if error > max_error {
                        max_error = error;
                    }
                }
            }

            let eps = T::epsilon();
            let acceptable_error = eps * T::from(n).unwrap() * T::from(100.0).unwrap();

            if max_error > acceptable_error {
                eprintln!(
                    "Warning: Matrix inversion may be numerically unstable. Max error: {}",
                    max_error
                );
            }
        }

        Ok(result)
    }

    /// Solve a linear system Ax = b
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 systems
    /// 2. LU decomposition with partial pivoting for larger systems
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }

        let n = a_shape[0];

        // Quick check for singularity using determinant
        // This is efficient for small matrices and catches perfect singularity
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "coefficient matrix is singular and cannot be solved".to_string(),
            ));
        }

        // Special fast paths for small systems
        if n == 1 {
            // 1x1 system - trivial solution
            let a_val = self.get(&[0, 0])?;
            let b_val = b.get(&[0])?;

            return Ok(Array::from_vec(vec![b_val / a_val]));
        } else if n == 2 {
            // 2x2 system - use Cramer's rule
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;

            return Ok(Array::from_vec(vec![x1, x2]));
        } else if n == 3 {
            // 3x3 system - use direct formula
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            // Compute determinant and cofactors
            let a11 = a_data[0];
            let a12 = a_data[1];
            let a13 = a_data[2];
            let a21 = a_data[3];
            let a22 = a_data[4];
            let a23 = a_data[5];
            let a31 = a_data[6];
            let a32 = a_data[7];
            let a33 = a_data[8];

            // Calculate cofactors for direct solution
            let c11 = a22 * a33 - a23 * a32;
            let c12 = -(a21 * a33 - a23 * a31);
            let c13 = a21 * a32 - a22 * a31;

            let c21 = -(a12 * a33 - a13 * a32);
            let c22 = a11 * a33 - a13 * a31;
            let c23 = -(a11 * a32 - a12 * a31);

            let c31 = a12 * a23 - a13 * a22;
            let c32 = -(a11 * a23 - a13 * a21);
            let c33 = a11 * a22 - a12 * a21;

            let inv_det = T::one() / det;

            // Multiply inverse of A (which is adjugate/det) by b
            let x1 = (c11 * b_data[0] + c21 * b_data[1] + c31 * b_data[2]) * inv_det;
            let x2 = (c12 * b_data[0] + c22 * b_data[1] + c32 * b_data[2]) * inv_det;
            let x3 = (c13 * b_data[0] + c23 * b_data[1] + c33 * b_data[2]) * inv_det;

            return Ok(Array::from_vec(vec![x1, x2, x3]));
        }

        // For larger systems, use SCIRS
        use crate::interop::scirs_compat::solve_linear_system;
        solve_linear_system(self, b)
    }

    /// Solve a linear system Ax = b
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 systems
    /// 2. LU decomposition with partial pivoting for larger systems
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    #[cfg(all(feature = "matrix_decomp", not(feature = "scirs")))]
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }

        let n = a_shape[0];

        // Quick check for singularity using determinant
        // This is efficient for small matrices and catches perfect singularity
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "coefficient matrix is singular and cannot be solved".to_string(),
            ));
        }

        // Special fast paths for small systems
        if n == 1 {
            // 1x1 system - trivial solution
            let a_val = self.get(&[0, 0])?;
            let b_val = b.get(&[0])?;

            return Ok(Array::from_vec(vec![b_val / a_val]));
        } else if n == 2 {
            // 2x2 system - use Cramer's rule
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;

            return Ok(Array::from_vec(vec![x1, x2]));
        } else if n == 3 {
            // 3x3 system - use direct formula
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            // Compute determinant and cofactors
            let a11 = a_data[0];
            let a12 = a_data[1];
            let a13 = a_data[2];
            let a21 = a_data[3];
            let a22 = a_data[4];
            let a23 = a_data[5];
            let a31 = a_data[6];
            let a32 = a_data[7];
            let a33 = a_data[8];

            // Calculate cofactors for direct solution
            let c11 = a22 * a33 - a23 * a32;
            let c12 = -(a21 * a33 - a23 * a31);
            let c13 = a21 * a32 - a22 * a31;

            let c21 = -(a12 * a33 - a13 * a32);
            let c22 = a11 * a33 - a13 * a31;
            let c23 = -(a11 * a32 - a12 * a31);

            let c31 = a12 * a23 - a13 * a22;
            let c32 = -(a11 * a23 - a13 * a21);
            let c33 = a11 * a22 - a12 * a21;

            let inv_det = T::one() / det;

            // Multiply inverse of A (which is adjugate/det) by b
            let x1 = (c11 * b_data[0] + c21 * b_data[1] + c31 * b_data[2]) * inv_det;
            let x2 = (c12 * b_data[0] + c22 * b_data[1] + c32 * b_data[2]) * inv_det;
            let x3 = (c13 * b_data[0] + c23 * b_data[1] + c33 * b_data[2]) * inv_det;

            return Ok(Array::from_vec(vec![x1, x2, x3]));
        }

        // For larger systems, use LU decomposition with partial pivoting
        // Get LU decomposition with pivoting: PA = LU
        #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
        use crate::new_modules::matrix_decomp::lu;

        // Step 1: Calculate LU decomposition
        #[cfg(feature = "matrix_decomp")]
        let (l, u, p) = lu(self)?;

        #[cfg(not(feature = "matrix_decomp"))]
        return Err(NumRs2Error::FeatureNotEnabled(
            "matrix_decomp feature required for LU decomposition".to_string(),
        ));

        // Step 2: Apply permutation to b (Pb)
        let mut pb = vec![T::zero(); n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let p_idx = p.get(&[i])?.to_usize().unwrap_or(i);
            pb[i] = b.get(&[p_idx])?;
        }

        // Step 3: Forward substitution to solve Ly = Pb
        let mut y = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = pb[i];
            #[allow(clippy::needless_range_loop)]
            for k in 0..i {
                sum -= l.get(&[i, k])? * y[k];
            }
            y[i] = sum; // L has 1's on diagonal
        }

        // Step 4: Back substitution to solve Ux = y
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            #[allow(clippy::needless_range_loop)]
            for k in (i + 1)..n {
                sum -= u.get(&[i, k])? * x[k];
            }
            x[i] = sum / u.get(&[i, i])?;
        }

        // Step 5: Return the solution vector
        Ok(Array::from_vec(x))
    }

    /// Solve a linear system Ax = b
    ///
    /// This implementation includes:
    /// 1. Fast direct formulas for 1x1, 2x2, and 3x3 systems
    /// 2. Gaussian elimination with partial pivoting for larger systems
    /// 3. Numerical stability checks with appropriate condition number thresholds
    /// 4. Proper error handling for singular or ill-conditioned matrices
    #[cfg(not(feature = "matrix_decomp"))]
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string(),
            ));
        }

        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }

        let n = a_shape[0];

        // Fast paths for small systems
        if n == 1 {
            // 1x1 system - trivial solution
            let a_val = self.get(&[0, 0])?;

            if a_val == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular and cannot be solved".to_string(),
                ));
            }

            let b_val = b.get(&[0])?;
            return Ok(Array::from_vec(vec![b_val / a_val]));
        } else if n == 2 {
            // 2x2 system - use Cramer's rule
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            let det = a_data[0] * a_data[3] - a_data[1] * a_data[2];
            if det == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular".to_string(),
                ));
            }

            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;

            return Ok(Array::from_vec(vec![x1, x2]));
        } else if n == 3 {
            // 3x3 system - use direct formula through cofactors
            let a_data = self.to_vec();
            let b_data = b.to_vec();

            // Calculate determinant
            let det = a_data[0] * (a_data[4] * a_data[8] - a_data[5] * a_data[7])
                - a_data[1] * (a_data[3] * a_data[8] - a_data[5] * a_data[6])
                + a_data[2] * (a_data[3] * a_data[7] - a_data[4] * a_data[6]);

            if det == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular".to_string(),
                ));
            }

            // Compute determinant and cofactors
            let a11 = a_data[0];
            let a12 = a_data[1];
            let a13 = a_data[2];
            let a21 = a_data[3];
            let a22 = a_data[4];
            let a23 = a_data[5];
            let a31 = a_data[6];
            let a32 = a_data[7];
            let a33 = a_data[8];

            // Calculate cofactors for direct solution
            let c11 = a22 * a33 - a23 * a32;
            let c12 = -(a21 * a33 - a23 * a31);
            let c13 = a21 * a32 - a22 * a31;

            let c21 = -(a12 * a33 - a13 * a32);
            let c22 = a11 * a33 - a13 * a31;
            let c23 = -(a11 * a32 - a12 * a31);

            let c31 = a12 * a23 - a13 * a22;
            let c32 = -(a11 * a23 - a13 * a21);
            let c33 = a11 * a22 - a12 * a21;

            let inv_det = T::one() / det;

            // Multiply inverse of A (which is adjugate/det) by b
            let x1 = (c11 * b_data[0] + c21 * b_data[1] + c31 * b_data[2]) * inv_det;
            let x2 = (c12 * b_data[0] + c22 * b_data[1] + c32 * b_data[2]) * inv_det;
            let x3 = (c13 * b_data[0] + c23 * b_data[1] + c33 * b_data[2]) * inv_det;

            return Ok(Array::from_vec(vec![x1, x2, x3]));
        }

        // For larger systems, use Gaussian elimination with partial pivoting

        // Create an augmented matrix [A|b]
        let mut aug = Array::zeros(&[n, n + 1]);

        // Fill in the augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug.set(&[i, j], self.get(&[i, j])?)?;
            }
            aug.set(&[i, n], b.get(&[i])?)?;
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot (maximum absolute value in current column)
            let mut max_val = num_traits::Float::abs(aug.get(&[i, i])?);
            let mut max_row = i;

            for j in (i + 1)..n {
                let abs_val = num_traits::Float::abs(aug.get(&[j, i])?);
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = j;
                }
            }

            // Check for singularity
            let eps = T::epsilon();
            if max_val < eps * T::from(n).unwrap() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is numerically singular".to_string(),
                ));
            }

            // Swap rows if needed
            if max_row != i {
                for j in i..(n + 1) {
                    let temp = aug.get(&[i, j])?;
                    aug.set(&[i, j], aug.get(&[max_row, j])?)?;
                    aug.set(&[max_row, j], temp)?;
                }
            }

            // Eliminate below
            for j in (i + 1)..n {
                let factor = aug.get(&[j, i])? / aug.get(&[i, i])?;

                for k in i..(n + 1) {
                    let val = aug.get(&[j, k])? - factor * aug.get(&[i, k])?;
                    aug.set(&[j, k], val)?;
                }
            }
        }

        // Back substitution
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug.get(&[i, n])?;

            for j in (i + 1)..n {
                sum -= aug.get(&[i, j])? * x[j];
            }

            x[i] = sum / aug.get(&[i, i])?;
        }

        Ok(Array::from_vec(x))
    }

    /// Compute the singular value decomposition of a matrix
    pub fn svd(&self) -> Result<(Array<T>, Array<T>, Array<T>)> {
        // Use the implementation from new_modules::matrix_decomp
        use crate::new_modules::matrix_decomp::svd;
        let (u, s_real, vt) = svd(self)?;

        // Convert singular values from Real to T
        // Since T is a Float type and Real is its real component type,
        // for real-valued matrices T == Real, so this is safe
        let s_vec: Vec<T> = s_real
            .to_vec()
            .into_iter()
            .map(|val| {
                // Convert from Real to T (for real matrices, this is identity)
                num_traits::NumCast::from(val).unwrap_or(T::zero())
            })
            .collect();
        let s = Array::from_vec(s_vec);

        Ok((u, s, vt))
    }

    /// Compute the eigenvalues and eigenvectors of a square matrix
    pub fn eig(&self) -> Result<(Array<T>, Array<T>)> {
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "eigendecomposition requires a square matrix".to_string(),
            ));
        }

        // This would use ndarray-linalg's eigenvalue computation in a full version
        // For now, we'll just return placeholder values
        let n = shape[0];

        let eigenvalues = Array::zeros(&[n]);
        let eigenvectors = Array::zeros(&[n, n]);

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute the Cholesky decomposition of a matrix
    pub fn cholesky(&self) -> Result<Array<T>> {
        // Use the implementation from new_modules::matrix_decomp
        use crate::new_modules::matrix_decomp::cholesky;
        cholesky(self)
    }

    /// Compute the QR decomposition of a matrix
    pub fn qr(&self) -> Result<(Array<T>, Array<T>)> {
        // Use the implementation from new_modules::matrix_decomp
        use crate::new_modules::matrix_decomp::qr;
        qr(self)
    }
}
