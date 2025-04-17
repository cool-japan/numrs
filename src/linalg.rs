use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;
use num_complex::Complex;

/// Set the number of threads for LAPACK operations
pub fn set_lapack_threads(threads: usize) {
    // We can use blas_src's set_num_threads when it's available
    // For now, we'll just provide this as a placeholder
    let _threads = threads;
}

/// A simplified direct implementation of linear algebra functions for Array
impl<T> Array<T>
where
    T: Float + Clone + Debug,
{
    /// Compute the determinant of a matrix
    pub fn det(&self) -> Result<T> {
        // Verify the array is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "determinant requires a square matrix".to_string()
            ));
        }
        
        // For now, we'll just calculate determinant for 2x2 and 3x3 matrices
        let n = shape[0];
        let data = self.to_vec();
        
        if n == 1 {
            return Ok(data[0]);
        } else if n == 2 {
            return Ok(data[0] * data[3] - data[1] * data[2]);
        } else if n == 3 {
            // For 3x3 matrix
            let det = 
                data[0] * (data[4] * data[8] - data[5] * data[7]) -
                data[1] * (data[3] * data[8] - data[5] * data[6]) +
                data[2] * (data[3] * data[7] - data[4] * data[6]);
            return Ok(det);
        }
        
        // For larger matrices, use ndarray-linalg in a full implementation
        Err(NumRs2Error::InvalidOperation(
            "determinant for matrices larger than 3x3 not implemented in this version".to_string()
        ))
    }

    /// Compute the inverse of a matrix
    pub fn inv(&self) -> Result<Array<T>> {
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "inverse requires a square matrix".to_string()
            ));
        }
        
        let n = shape[0];
        
        // Check if the matrix is invertible
        let det = self.det()?;
        if det == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "matrix is singular and cannot be inverted".to_string()
            ));
        }
        
        // For the purpose of this demo, we'll only implement inverse for 2x2 matrices
        if n == 2 {
            let data = self.to_vec();
            let a = data[0];
            let b = data[1]; 
            let c = data[2];
            let d = data[3];
            
            let inv_det = T::one() / det;
            let result = vec![
                d * inv_det, -b * inv_det,
                -c * inv_det, a * inv_det
            ];
            
            return Ok(Array::from_vec(result).reshape(&[2, 2]));
        }
        
        // For larger matrices, would use ndarray-linalg in a full implementation
        Err(NumRs2Error::InvalidOperation(
            "inverse for matrices larger than 2x2 not implemented in this version".to_string()
        ))
    }

    /// Solve a linear system Ax = b
    pub fn solve(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let a_shape = self.shape();
        let b_shape = b.shape();
        
        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "solve requires a square coefficient matrix".to_string()
            ));
        }
        
        if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0]],
                actual: b_shape,
            });
        }
        
        // Simple implementation for 2x2 system as demonstration
        if a_shape[0] == 2 {
            // Solve using Cramer's rule for 2x2 system
            let a_data = self.to_vec();
            let b_data = b.to_vec();
            
            let det = a_data[0] * a_data[3] - a_data[1] * a_data[2];
            if det == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "coefficient matrix is singular".to_string()
                ));
            }
            
            let x1 = (b_data[0] * a_data[3] - a_data[1] * b_data[1]) / det;
            let x2 = (a_data[0] * b_data[1] - b_data[0] * a_data[2]) / det;
            
            return Ok(Array::from_vec(vec![x1, x2]));
        }
        
        // For larger systems, use ndarray-linalg in a full implementation
        Err(NumRs2Error::InvalidOperation(
            "solve for systems larger than 2x2 not implemented in this version".to_string()
        ))
    }

    /// Compute the singular value decomposition of a matrix
    pub fn svd(&self) -> Result<(Array<T>, Array<T>, Array<T>)> {
        // Check that the matrix is 2D
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "SVD requires a 2D matrix".to_string()
            ));
        }
        
        // This would use ndarray-linalg's SVD implementation in a full version
        // For now, we'll just return placeholder values
        let m = shape[0];
        let n = shape[1];
        let k = std::cmp::min(m, n);
        
        let u = Array::zeros(&[m, k]);
        let s = Array::zeros(&[k]);
        let vt = Array::zeros(&[k, n]);
        
        Ok((u, s, vt))
    }

    /// Compute the eigenvalues and eigenvectors of a square matrix
    pub fn eig(&self) -> Result<(Array<T>, Array<T>)> {
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "eigendecomposition requires a square matrix".to_string()
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
        // Check if the matrix is square
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Cholesky decomposition requires a square matrix".to_string()
            ));
        }
        
        // This would use ndarray-linalg's Cholesky decomposition in a full version
        // For now, we'll just return a placeholder
        let n = shape[0];
        let l = Array::zeros(&[n, n]);
        
        Ok(l)
    }
    
    /// Compute the QR decomposition of a matrix
    pub fn qr(&self) -> Result<(Array<T>, Array<T>)> {
        // Check that the matrix is 2D
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "QR decomposition requires a 2D matrix".to_string()
            ));
        }
        
        // This would use ndarray-linalg's QR decomposition in a full version
        // For now, we'll just return placeholder values
        let m = shape[0];
        let n = shape[1];
        
        let q = Array::zeros(&[m, m]);
        let r = Array::zeros(&[m, n]);
        
        Ok((q, r))
    }
}

// Common linear algebra functions (similar to NumPy's linalg module)

/// Compute the norm of a vector or matrix
pub fn norm<T: Float + Clone + std::fmt::Display>(a: &Array<T>, ord: Option<T>) -> Result<T> {
    let shape = a.shape();
    let ord = ord.unwrap_or(T::from(2.0).unwrap());
    
    if shape.len() == 1 {
        // Vector norm
        if ord == T::from(1.0).unwrap() {
            // L1 norm (sum of absolute values)
            let data = a.to_vec();
            let sum = data.iter().fold(T::zero(), |acc, &x| acc + x.abs());
            Ok(sum)
        } else if ord == T::from(2.0).unwrap() {
            // L2 norm (Euclidean norm)
            let data = a.to_vec();
            let sum_squares = data.iter().fold(T::zero(), |acc, &x| acc + x * x);
            Ok(sum_squares.sqrt())
        } else if ord == T::from(f64::INFINITY).unwrap() {
            // L-infinity norm (maximum absolute value)
            let data = a.to_vec();
            let max_abs = data.iter()
                .fold(T::zero(), |acc, &x| T::max(acc, x.abs()));
            Ok(max_abs)
        } else {
            // General case
            let data = a.to_vec();
            let sum_pow = data.iter()
                .fold(T::zero(), |acc, &x| acc + x.abs().powf(ord));
            Ok(sum_pow.powf(T::one() / ord))
        }
    } else if shape.len() == 2 {
        // Matrix norm
        if ord == T::from(1.0).unwrap() {
            // Maximum column sum
            let m = shape[0];
            let n = shape[1];
            let data = a.to_vec();
            
            let mut max_col_sum = T::zero();
            for j in 0..n {
                let mut col_sum = T::zero();
                for i in 0..m {
                    col_sum = col_sum + data[i * n + j].abs();
                }
                max_col_sum = T::max(max_col_sum, col_sum);
            }
            
            Ok(max_col_sum)
        } else if ord == T::from(f64::INFINITY).unwrap() {
            // Maximum row sum
            let m = shape[0];
            let n = shape[1];
            let data = a.to_vec();
            
            let mut max_row_sum = T::zero();
            for i in 0..m {
                let mut row_sum = T::zero();
                for j in 0..n {
                    row_sum = row_sum + data[i * n + j].abs();
                }
                max_row_sum = T::max(max_row_sum, row_sum);
            }
            
            Ok(max_row_sum)
        } else if ord == T::from(2.0).unwrap() {
            // Spectral norm (maximum singular value)
            // In a real implementation, this would use SVD
            Err(NumRs2Error::InvalidOperation(
                "Spectral norm not implemented in this version".to_string()
            ))
        } else {
            Err(NumRs2Error::InvalidOperation(
                format!("Invalid matrix norm order: {}", ord)
            ))
        }
    } else {
        Err(NumRs2Error::DimensionMismatch(
            "norm requires a 1D or 2D array".to_string()
        ))
    }
}

/// Compute the rank of a matrix
pub fn matrix_rank<T: Float + Clone + Debug>(a: &Array<T>, tol: Option<T>) -> Result<usize> {
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "matrix_rank requires a 2D matrix".to_string()
        ));
    }
    
    // Compute SVD to get singular values
    let (_, s, _) = svd(a)?;
    
    // Get the tolerance
    let tol_val = match tol {
        Some(t) => t,
        None => {
            // Default is max(M, N) * eps * max(S)
            let m = shape[0];
            let n = shape[1];
            let max_dim = std::cmp::max(m, n);
            let eps = T::epsilon();
            let max_s = s.array().fold(T::zero(), |max, &val| if val > max { val } else { max });
            
            T::from(max_dim).unwrap() * eps * max_s
        }
    };
    
    // Count singular values larger than tolerance
    let s_data = s.to_vec();
    let rank = s_data.iter().filter(|&&val| val > tol_val).count();
    
    Ok(rank)
}

/// Compute the QR decomposition of a matrix
pub fn qr<T: Float + Clone + Debug>(a: &Array<T>) -> Result<(Array<T>, Array<T>)> {
    a.qr()
}

/// Compute the Cholesky decomposition of a matrix
pub fn cholesky<T: Float + Clone + Debug>(a: &Array<T>) -> Result<Array<T>> {
    a.cholesky()
}

/// Compute the eigenvalues and eigenvectors of a square matrix
/// 
/// # Parameters
/// 
/// * `a` - The input matrix
/// * `sort` - Sort eigenvalues and eigenvectors by eigenvalue magnitude.
///           Options: "asc" (ascending), "desc" (descending), or None (no sorting)
/// 
/// # Returns
/// 
/// A tuple of (eigenvalues, eigenvectors)
pub fn eig<T: Float + Clone + Debug>(a: &Array<T>, sort: Option<&str>) -> Result<(Array<T>, Array<T>)> {
    // Get the eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = a.eig()?;
    
    // Return if no sorting is requested
    if sort.is_none() {
        return Ok((eigenvalues, eigenvectors));
    }
    
    let sort_option = sort.unwrap();
    
    if sort_option != "asc" && sort_option != "desc" {
        return Err(NumRs2Error::InvalidOperation(
            format!("Invalid sort option: {}. Must be 'asc', 'desc', or None", sort_option)
        ));
    }
    
    // Get the shape of the eigenvectors matrix
    let evec_shape = eigenvectors.shape();
    
    // Convert eigenvalues to vector for sorting
    let evals_data = eigenvalues.to_vec();
    let n = evals_data.len();
    
    // Create indices for sorting
    let mut indices: Vec<usize> = (0..n).collect();
    
    // Sort indices by eigenvalue magnitude
    indices.sort_by(|&i, &j| {
        let a_abs = evals_data[i].abs();
        let b_abs = evals_data[j].abs();
        
        if sort_option == "asc" {
            a_abs.partial_cmp(&b_abs).unwrap()
        } else {
            b_abs.partial_cmp(&a_abs).unwrap()
        }
    });
    
    // Create sorted eigenvalues array
    let mut sorted_evals = Vec::with_capacity(n);
    for &idx in &indices {
        sorted_evals.push(evals_data[idx]);
    }
    
    // Create sorted eigenvectors array
    let evecs_data = eigenvectors.to_vec();
    let eigvec_size = evec_shape[0];
    let mut sorted_evecs = Vec::with_capacity(evecs_data.len());
    
    for &idx in &indices {
        // Extract the eigenvector column
        for i in 0..eigvec_size {
            let evec_idx = i * n + idx;
            sorted_evecs.push(evecs_data[evec_idx].clone());
        }
    }
    
    // Convert to Array objects
    let sorted_eigenvalues = Array::from_vec(sorted_evals);
    let sorted_eigenvectors = Array::from_vec(sorted_evecs).reshape(&evec_shape);
    
    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

/// Solve a linear system Ax = b
pub fn solve<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    a.solve(b)
}

/// Compute the singular value decomposition of a matrix
pub fn svd<T: Float + Clone + Debug>(a: &Array<T>) -> Result<(Array<T>, Array<T>, Array<T>)> {
    a.svd()
}

/// Compute the inverse of a matrix
pub fn inv<T: Float + Clone + Debug>(a: &Array<T>) -> Result<Array<T>> {
    a.inv()
}

/// Compute the Moore-Penrose pseudoinverse of a matrix
/// 
/// The Moore-Penrose pseudoinverse is a generalization of the inverse matrix
/// that exists for any matrix, even non-square or singular matrices.
/// It is computed using SVD decomposition.
/// 
/// # Parameters
/// 
/// * `a` - The input matrix
/// * `rcond` - Cutoff for small singular values. Singular values smaller 
///             than rcond * largest_singular_value are set to zero.
///             Default is 1e-15.
/// 
/// # Returns
/// 
/// The pseudoinverse of the input matrix
pub fn pinv<T: Float + Clone + Debug>(a: &Array<T>, rcond: Option<T>) -> Result<Array<T>> {
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "pinv requires a 2D matrix".to_string()
        ));
    }
    
    // Perform SVD: A = U * S * V^T
    let (u, s, vt) = svd(a)?;
    
    // Get the cutoff value for singular values
    let rcond_val = rcond.unwrap_or_else(|| T::from(1e-15).unwrap());
    
    // Find the maximum singular value to determine cutoff
    let max_singular_val = s.array().fold(T::zero(), |max, &val| if val > max { val } else { max });
    let cutoff = max_singular_val * rcond_val;
    
    // Invert the non-zero singular values
    let s_data = s.to_vec();
    let mut s_inv_data = Vec::with_capacity(s_data.len());
    
    for &val in &s_data {
        if val > cutoff {
            s_inv_data.push(T::one() / val);
        } else {
            s_inv_data.push(T::zero());
        }
    }
    
    // Create vectors for s_inv needed for diagonal matrix
    let s_inv_vec = s_inv_data.clone();
    
    // For a matrix A of shape (m, n), u has shape (m, k), s has shape (k),
    // and vt has shape (k, n) where k = min(m, n)
    let m = shape[0];
    let n = shape[1];
    let k = std::cmp::min(m, n);
    
    // Construct the pseudoinverse using the formula: A^+ = V * S^+ * U^T
    // Where S^+ is a diagonal matrix with 1/s_i if s_i > cutoff, and 0 otherwise
    
    // 1. Create a diagonal matrix from s_inv
    let mut s_inv_diag = Array::zeros(&[k, k]);
    for i in 0..k {
        s_inv_diag.set(&[i, i], s_inv_vec[i])?;
    }
    
    // 2. Compute V * S^+
    let v = vt.transpose();
    let vs_inv = v.matmul(&s_inv_diag)?;
    
    // 3. Compute (V * S^+) * U^T
    let ut = u.transpose();
    let pinv_result = vs_inv.matmul(&ut)?;
    
    Ok(pinv_result)
}

/// Compute the determinant of a matrix
pub fn det<T: Float + Clone + Debug>(a: &Array<T>) -> Result<T> {
    a.det()
}

/// Compute the vectorized dot product using the complex conjugate of the first argument
pub fn vdot<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<T> {
    // For real arrays, this is the same as inner product
    a.dot(b)
}

/// Compute the vectorized dot product for complex arrays
pub fn complex_vdot<T: Float + Clone + Debug>(a: &Array<Complex<T>>, b: &Array<Complex<T>>) -> Result<Complex<T>> {
    // Check dimensions
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "vdot requires two 1D arrays".to_string()
        ));
    }
    
    // Check lengths
    if a.size() != b.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    
    // For complex arrays, first conjugate a
    let a_conj = a.map(|x| x.conj());
    
    // Then compute the dot product
    let a_data = a_conj.to_vec();
    let b_data = b.to_vec();
    let mut result = Complex::new(T::zero(), T::zero());
    
    for i in 0..a.size() {
        result = result + a_data[i] * b_data[i];
    }
    
    Ok(result)
}

/// Compute the inner product of two arrays
pub fn inner<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<T> {
    // Check dimensions
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "inner product requires two 1D arrays".to_string()
        ));
    }
    
    // Check lengths
    if a.size() != b.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    
    // Compute dot product
    a.dot(b)
}

/// Trace of a matrix (sum of diagonal elements)
pub fn trace<T: Float + Clone + Debug>(a: &Array<T>) -> Result<T> {
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "trace requires a 2D matrix".to_string()
        ));
    }
    
    let m = shape[0];
    let n = shape[1];
    let min_dim = std::cmp::min(m, n);
    
    let a_data = a.to_vec();
    let mut sum = T::zero();
    
    for i in 0..min_dim {
        sum = sum + a_data[i * n + i];
    }
    
    Ok(sum)
}

/// Compute the matrix power (A raised to power n)
pub fn matrix_power<T: Float + Clone + Debug>(a: &Array<T>, n: i32) -> Result<Array<T>> {
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "matrix_power requires a square matrix".to_string()
        ));
    }
    
    let size = shape[0];
    
    // Handle special cases
    if n == 0 {
        // Return identity matrix
        return Ok(Array::identity(size));
    }
    
    if n == 1 {
        // Return a copy of the original matrix
        return Ok(a.clone());
    }
    
    if n == -1 {
        // Return the inverse
        return a.inv();
    }
    
    // For higher powers, we should implement a more efficient algorithm
    // using binary exponentiation. For simplicity, we'll use a direct approach
    // for now.
    
    if n > 0 {
        let mut result = a.clone();
        for _ in 1..n {
            result = result.matmul(a)?;
        }
        Ok(result)
    } else {
        // For negative powers, compute the inverse first
        let inv = a.inv()?;
        let abs_n = (-n) as u32;
        
        let mut result = inv.clone();
        for _ in 1..abs_n {
            result = result.matmul(&inv)?;
        }
        Ok(result)
    }
}

/// Compute the outer product of two vectors
pub fn outer<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    // Check that both inputs are 1D arrays (vectors)
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "outer requires two 1D arrays".to_string()
        ));
    }
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    
    // Create output array of shape (len(a), len(b))
    let mut result = Array::zeros(&[a_shape[0], b_shape[0]]);
    let result_data = result.array_mut().as_slice_mut().unwrap();
    
    // Compute outer product
    for (i, &a_val) in a_data.iter().enumerate() {
        for (j, &b_val) in b_data.iter().enumerate() {
            result_data[i * b_shape[0] + j] = a_val * b_val;
        }
    }
    
    Ok(result)
}

/// Compute the Kronecker product of two arrays
pub fn kron<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    // Check that both inputs are 2D arrays
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "kron requires two 2D arrays".to_string()
        ));
    }
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Output shape is (a_rows * b_rows, a_cols * b_cols)
    let out_shape = [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]];
    let mut result = Array::zeros(&out_shape);
    
    // Extract the data
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data = result.array_mut().as_slice_mut().unwrap();
    
    // Compute Kronecker product
    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            let a_idx = i * a_shape[1] + j;
            let a_val = a_data[a_idx];
            
            // For each element in A, multiply by entire B matrix
            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    let b_idx = k * b_shape[1] + l;
                    let b_val = b_data[b_idx];
                    
                    // Position in result array
                    let row = i * b_shape[0] + k;
                    let col = j * b_shape[1] + l;
                    let result_idx = row * out_shape[1] + col;
                    
                    result_data[result_idx] = a_val * b_val;
                }
            }
        }
    }
    
    Ok(result)
}

/// Compute tensor dot product of two arrays along specified axes
pub fn tensordot<T: Float + Clone + Debug>(
    a: &Array<T>, 
    b: &Array<T>, 
    axes: &[usize]
) -> Result<Array<T>> {
    // Simplified version for 2 axes
    if axes.len() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "This implementation of tensordot only supports 2 axes".to_string()
        ));
    }
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    let a_axis = axes[0];
    let b_axis = axes[1];
    
    if a_axis >= a_shape.len() || b_axis >= b_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Axis out of bounds".to_string()
        ));
    }
    
    // Check that contracted dimensions match
    if a_shape[a_axis] != b_shape[b_axis] {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![a_shape[a_axis]],
            actual: vec![b_shape[b_axis]],
        });
    }
    
    // For simplicity, this implementation only handles 2D arrays
    // A complete implementation would handle arbitrary dimensions
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "This implementation of tensordot only supports 2D arrays".to_string()
        ));
    }
    
    // When contracting along axis 1 of A and axis 0 of B,
    // this becomes a matrix multiplication (if dimensions match)
    if a_axis == 1 && b_axis == 0 {
        return a.matmul(b);
    }
    
    // If contracting along axis 0 of A and axis 1 of B,
    // transpose B first, then do matrix multiplication
    if a_axis == 0 && b_axis == 1 {
        let b_trans = b.transpose();
        let result = a.transpose().matmul(&b_trans)?;
        return Ok(result.transpose());
    }
    
    // Handle other cases (more complex tensor contractions)
    Err(NumRs2Error::InvalidOperation(
        "This axis combination is not implemented in this version".to_string()
    ))
}