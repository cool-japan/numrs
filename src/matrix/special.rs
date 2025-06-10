use crate::array::Array;
use crate::error::Result;
use crate::matrix::Matrix;
use std::convert::TryFrom;

/// Create a Hilbert matrix of size n
///
/// A Hilbert matrix has elements A\[i,j\] = 1/(i+j+1)
///
/// # Arguments
///
/// * `n` - Size of the matrix (n x n)
///
/// # Returns
///
/// A square Matrix of size n x n
pub fn hilbert(n: usize) -> Matrix<f64> {
    let mut data = Vec::with_capacity(n * n);
    
    for i in 0..n {
        for j in 0..n {
            data.push(1.0 / ((i + j + 1) as f64));
        }
    }
    
    let array = Array::from_vec(data).reshape(&[n, n]);
    Matrix::new(array).unwrap()
}

/// Create a Vandermonde matrix from a vector
///
/// A Vandermonde matrix has elements A\[i,j\] = x\[i\]^j
///
/// # Arguments
///
/// * `x` - Vector of values
/// * `n` - Optional number of columns, defaults to length of x
///
/// # Returns
///
/// A Matrix of size x.len() x n where n defaults to x.len()
pub fn vandermonde(x: Vec<f64>, n: Option<usize>) -> Matrix<f64> {
    let rows = x.len();
    let cols = n.unwrap_or(rows);
    
    let mut data = Vec::with_capacity(rows * cols);
    
    #[allow(clippy::needless_range_loop)]
    for i in 0..rows {
        for j in 0..cols {
            data.push(x[i].powi(j as i32));
        }
    }
    
    let array = Array::from_vec(data).reshape(&[rows, cols]);
    Matrix::new(array).unwrap()
}

/// Create a Toeplitz matrix from first row and first column
///
/// A Toeplitz matrix has constant values along all diagonals
///
/// # Arguments
///
/// * `c` - First column of the matrix
/// * `r` - Optional first row. If None, uses the conjugate of first column.
///
/// # Returns
///
/// A Matrix where the first row is r and first column is c
pub fn toeplitz(c: Vec<f64>, r: Option<Vec<f64>>) -> Result<Matrix<f64>> {
    let rows = c.len();
    
    // Determine the first row
    let first_row = match r {
        Some(row) => row,
        None => vec![c[0]], // Just use first element of c as a scalar
    };
    
    let cols = first_row.len();
    let mut data = Vec::with_capacity(rows * cols);
    
    for i in 0..rows {
        for j in 0..cols {
            if i == 0 {
                // First row
                data.push(first_row[j]);
            } else if j == 0 {
                // First column
                data.push(c[i]);
            } else {
                // Diagonal elements have the same value
                // If i-j ≥ 0, use c[i-j], otherwise use r[j-i]
                if i >= j {
                    data.push(c[i - j]);
                } else {
                    data.push(first_row[j - i]);
                }
            }
        }
    }
    
    let array = Array::from_vec(data).reshape(&[rows, cols]);
    Matrix::new(array)
}

/// Create a circulant matrix from the first row
///
/// A circulant matrix has rows formed by cyclic permutations of the first row
///
/// # Arguments
///
/// * `first_row` - First row of the circulant matrix
///
/// # Returns
///
/// A square Matrix where each row is a cyclic shift of the first row
pub fn circulant(first_row: Vec<f64>) -> Matrix<f64> {
    let n = first_row.len();
    let mut data = Vec::with_capacity(n * n);
    
    for i in 0..n {
        for j in 0..n {
            // For each row, we shift the elements of the first row
            // by the row index, wrapping around the end
            let idx = (j + i) % n;
            data.push(first_row[idx]);
        }
    }
    
    let array = Array::from_vec(data).reshape(&[n, n]);
    Matrix::new(array).unwrap()
}

/// Create a Hankel matrix from the first column and last row
///
/// A Hankel matrix has constant values along all anti-diagonals
///
/// # Arguments
///
/// * `c` - First column of the matrix
/// * `r` - Optional last row. If None, uses zeros.
///
/// # Returns
///
/// A Matrix where the first column is c and the last row is r
pub fn hankel(c: Vec<f64>, r: Option<Vec<f64>>) -> Result<Matrix<f64>> {
    let rows = c.len();
    
    // Determine the last row
    let last_row = match r {
        Some(row) => row,
        None => vec![0.0; rows], // Use zeros by default
    };
    
    let cols = last_row.len();
    let mut data = Vec::with_capacity(rows * cols);
    
    for i in 0..rows {
        for j in 0..cols {
            // The value at position (i,j) depends on the sum i+j
            let sum = i + j;
            
            if sum < rows {
                // Use values from the first column
                data.push(c[sum]);
            } else {
                // Use values from the last row
                data.push(last_row[sum - rows + 1]);
            }
        }
    }
    
    let array = Array::from_vec(data).reshape(&[rows, cols]);
    Matrix::new(array)
}

/// Create a Leslie matrix, which is used in population modeling
///
/// # Arguments
///
/// * `f` - Fertility rates vector
/// * `s` - Survival rates vector (must be 1 less than the length of f)
///
/// # Returns
///
/// A square Matrix of size n x n
///
/// # Errors
///
/// Returns an error if the fertility and survival vectors have incompatible sizes
pub fn leslie(f: Vec<f64>, s: Vec<f64>) -> Result<Matrix<f64>> {
    if f.len() != s.len() + 1 {
        return Err(crate::error::NumRs2Error::DimensionMismatch(
            format!("Leslie matrix requires f.len() to be s.len() + 1, got f.len() = {} and s.len() = {}",
                    f.len(), s.len())
        ));
    }
    
    let n = f.len();
    let mut data = Vec::with_capacity(n * n);
    
    for i in 0..n {
        for j in 0..n {
            if i == 0 {
                // First row contains fertility rates
                data.push(f[j]);
            } else if j == i - 1 {
                // Subdiagonal contains survival rates
                data.push(s[j]);
            } else {
                // All other elements are 0
                data.push(0.0);
            }
        }
    }
    
    let array = Array::from_vec(data).reshape(&[n, n]);
    Matrix::new(array)
}

/// Create a companion matrix from a polynomial's coefficients
///
/// The companion matrix has the property that its eigenvalues are the roots of the polynomial
///
/// # Arguments
///
/// * `coefficients` - Coefficients of the polynomial, starting from the highest degree term
///
/// # Returns
///
/// A square Matrix of size (n-1) x (n-1) where n is the number of coefficients
///
/// # Errors
///
/// Returns an error if the coefficient vector has length less than 2
pub fn companion(coefficients: Vec<f64>) -> Result<Matrix<f64>> {
    if coefficients.len() < 2 {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Companion matrix requires at least 2 coefficients".to_string()
        ));
    }
    
    // Normalize by the leading coefficient
    let a0 = coefficients[0];
    let normalized: Vec<f64> = coefficients.iter().map(|&x| x / a0).collect();
    
    let n = normalized.len() - 1; // The size of the companion matrix is (n-1) x (n-1)
    let mut data = Vec::with_capacity(n * n);
    
    for i in 0..n {
        for j in 0..n {
            if i == j + 1 {
                // Subdiagonal elements are 1
                data.push(1.0);
            } else if j == n - 1 {
                // Last column contains negated coefficients (excluding the leading coefficient)
                data.push(-normalized[i + 1]);
            } else {
                // All other elements are 0
                data.push(0.0);
            }
        }
    }
    
    let array = Array::from_vec(data).reshape(&[n, n]);
    Matrix::new(array)
}

/// Create a Hadamard matrix of order n
///
/// A Hadamard matrix is a square matrix with elements +1 or -1 such that its rows are mutually orthogonal
///
/// # Arguments
///
/// * `n` - Order of the matrix (must be 1, 2, or a multiple of 4)
///
/// # Returns
///
/// A Hadamard Matrix of order n
///
/// # Errors
///
/// Returns an error if n is not a valid order for a Hadamard matrix
pub fn hadamard(n: usize) -> Result<Matrix<f64>> {
    // Check if the order is valid
    if n != 1 && n != 2 && n % 4 != 0 {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            format!("Hadamard matrices are only defined for orders 1, 2, or multiples of 4, got {}", n)
        ));
    }
    
    // Base cases
    if n == 1 {
        return Ok(Matrix::try_from(Array::from_vec(vec![1.0])).unwrap());
    }
    
    if n == 2 {
        let h2 = vec![1.0, 1.0, 1.0, -1.0];
        return Ok(Matrix::try_from(Array::from_vec(h2).reshape(&[2, 2])).unwrap());
    }
    
    // Recursive construction using Sylvester's method
    let mut h = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
    let mut order = 2;
    
    while order < n {
        // Double the size
        let mut h_new = vec![vec![0.0; order * 2]; order * 2];
        
        // Fill the four quadrants
        for i in 0..order {
            for j in 0..order {
                h_new[i][j] = h[i][j];                     // Top-left
                h_new[i][j + order] = h[i][j];             // Top-right
                h_new[i + order][j] = h[i][j];             // Bottom-left
                h_new[i + order][j + order] = -h[i][j];    // Bottom-right
            }
        }
        
        h = h_new;
        order *= 2;
    }
    
    // Convert the nested vector to a flat vector
    let mut data = Vec::with_capacity(n * n);
    for row in h.iter() {
        data.extend_from_slice(row);
    }
    
    let array = Array::from_vec(data).reshape(&[n, n]);
    Matrix::new(array)
}