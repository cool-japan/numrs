use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};
use num_traits::{Zero, One};
use std::convert::TryFrom;

/// Matrix is a 2-dimensional array with matrix-specific behavior and methods
/// 
/// This is equivalent to NumPy's `matrix` class, which differs from the standard ndarray
/// in that multiplication is matrix multiplication rather than element-wise.
#[derive(Clone)]
pub struct Matrix<T> {
    data: Array<T>,
}

impl<T> Matrix<T> 
where T: Clone + Zero + One + PartialEq + Default + PartialOrd
{
    /// Create a new matrix from a 2D array
    /// 
    /// # Arguments
    /// 
    /// * `array` - A 2-dimensional array
    /// 
    /// # Returns
    /// 
    /// A new Matrix instance
    /// 
    /// # Errors
    /// 
    /// Returns an error if the input array is not 2-dimensional
    pub fn new(array: Array<T>) -> Result<Self> {
        // Ensure array is 2D
        if array.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Matrix must be 2-dimensional, got {}-dimensional array", array.ndim())
            ));
        }
        
        Ok(Self { data: array })
    }
    
    /// Create a new matrix from a vector, interpreting it as a column vector
    /// 
    /// # Arguments
    /// 
    /// * `vec` - A vector of values
    /// 
    /// # Returns
    /// 
    /// A new Matrix instance with shape (n, 1) where n is the length of the vector
    pub fn from_vec(vec: Vec<T>) -> Self {
        let n = vec.len();
        let array = Array::from_vec(vec).reshape(&[n, 1]);
        Self { data: array }
    }
    
    /// Create a new matrix from a nested vector
    /// 
    /// # Arguments
    /// 
    /// * `nested_vec` - A nested vector representing a matrix
    /// 
    /// # Returns
    /// 
    /// A new Matrix instance
    /// 
    /// # Errors
    /// 
    /// Returns an error if the nested vector has inconsistent row lengths
    pub fn from_nested_vec(nested_vec: Vec<Vec<T>>) -> Result<Self> {
        // Check if all rows have the same length
        if nested_vec.is_empty() {
            return Err(NumRs2Error::InvalidOperation("Cannot create matrix from empty vector".to_string()));
        }
        
        let first_row_len = nested_vec[0].len();
        if !nested_vec.iter().all(|row| row.len() == first_row_len) {
            return Err(NumRs2Error::InvalidOperation("Inconsistent row lengths in nested vector".to_string()));
        }
        
        // Flatten the nested vector
        let mut flat_vec = Vec::with_capacity(nested_vec.len() * first_row_len);
        let rows = nested_vec.len();
        for row in nested_vec {
            flat_vec.extend(row);
        }
        
        // Create an array and reshape it
        let array = Array::from_vec(flat_vec).reshape(&[rows, first_row_len]);
        Ok(Self { data: array })
    }
    
    /// Create a new matrix filled with zeros
    /// 
    /// # Arguments
    /// 
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// 
    /// # Returns
    /// 
    /// A new Matrix instance filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self 
    where T: Default + Clone
    {
        let array = Array::zeros(&[rows, cols]);
        Self { data: array }
    }
    
    /// Create a new matrix filled with ones
    /// 
    /// # Arguments
    /// 
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// 
    /// # Returns
    /// 
    /// A new Matrix instance filled with ones
    pub fn ones(rows: usize, cols: usize) -> Self 
    where T: From<u8> + Clone
    {
        let array = Array::ones(&[rows, cols]);
        Self { data: array }
    }
    
    /// Create a new identity matrix
    /// 
    /// # Arguments
    /// 
    /// * `n` - Size of the matrix (n x n)
    /// 
    /// # Returns
    /// 
    /// A new square Matrix instance with ones on the diagonal and zeros elsewhere
    pub fn eye(n: usize) -> Self 
    where T: From<u8> + Clone + Default
    {
        let mut array = Array::zeros(&[n, n]);
        
        // Set diagonal elements to 1
        for i in 0..n {
            let value = T::from(1u8);
            array.set(&[i, i], value.clone()).unwrap();
        }
        
        Self { data: array }
    }
    
    /// Get the number of rows in the matrix
    pub fn nrows(&self) -> usize {
        self.data.shape()[0]
    }
    
    /// Get the number of columns in the matrix
    pub fn ncols(&self) -> usize {
        self.data.shape()[1]
    }
    
    /// Get the shape of the matrix as (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1])
    }
    
    /// Get the total number of elements in the matrix
    pub fn size(&self) -> usize {
        self.data.size()
    }
    
    /// Get the underlying array
    pub fn array(&self) -> &Array<T> {
        &self.data
    }
    
    /// Convert the matrix to a regular array
    pub fn to_array(&self) -> Array<T> {
        self.data.clone()
    }
    
    /// Convert the matrix to a vector of vectors
    pub fn to_nested_vec(&self) -> Vec<Vec<T>> {
        let (rows, cols) = self.shape();
        let mut result = Vec::with_capacity(rows);
        
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self.data.get(&[i, j]).unwrap().clone());
            }
            result.push(row);
        }
        
        result
    }
    
    /// Get a value at the specified indices
    pub fn get(&self, i: usize, j: usize) -> Result<T> {
        if i >= self.nrows() || j >= self.ncols() {
            return Err(NumRs2Error::IndexOutOfBounds(
                format!("Index ({}, {}) out of bounds for matrix with shape ({}, {})",
                        i, j, self.nrows(), self.ncols())
            ));
        }
        
        // Get value from underlying array
        Ok(self.data.get(&[i, j])?.clone())
    }
    
    /// Set a value at the specified indices
    pub fn set(&mut self, i: usize, j: usize, value: T) -> Result<()> {
        if i >= self.nrows() || j >= self.ncols() {
            return Err(NumRs2Error::IndexOutOfBounds(
                format!("Index ({}, {}) out of bounds for matrix with shape ({}, {})",
                        i, j, self.nrows(), self.ncols())
            ));
        }
        
        // Set value in underlying array
        self.data.set(&[i, j], value)
    }
    
    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let transposed_array = self.data.transpose();
        Self { data: transposed_array }
    }
    
    /// Get a row of the matrix as a new matrix
    pub fn row(&self, i: usize) -> Result<Self> {
        if i >= self.nrows() {
            return Err(NumRs2Error::IndexOutOfBounds(
                format!("Row index {} out of bounds for matrix with {} rows", i, self.nrows())
            ));
        }
        
        let cols = self.ncols();
        let mut row_data = Vec::with_capacity(cols);
        
        for j in 0..cols {
            row_data.push(self.data.get(&[i, j])?.clone());
        }
        
        // Create a row matrix (1 x cols)
        let row_array = Array::from_vec(row_data).reshape(&[1, cols]);
        Ok(Self { data: row_array })
    }
    
    /// Get a column of the matrix as a new matrix
    pub fn column(&self, j: usize) -> Result<Self> {
        if j >= self.ncols() {
            return Err(NumRs2Error::IndexOutOfBounds(
                format!("Column index {} out of bounds for matrix with {} columns", j, self.ncols())
            ));
        }
        
        let rows = self.nrows();
        let mut col_data = Vec::with_capacity(rows);
        
        for i in 0..rows {
            col_data.push(self.data.get(&[i, j])?.clone());
        }
        
        // Create a column matrix (rows x 1)
        let col_array = Array::from_vec(col_data).reshape(&[rows, 1]);
        Ok(Self { data: col_array })
    }
    
    /// Get the diagonal of the matrix as a new matrix
    pub fn diagonal(&self) -> Self {
        let (rows, cols) = self.shape();
        let diag_len = rows.min(cols);
        let mut diag_data = Vec::with_capacity(diag_len);
        
        for i in 0..diag_len {
            diag_data.push(self.data.get(&[i, i]).unwrap().clone());
        }
        
        // Create a column matrix with the diagonal elements
        let diag_array = Array::from_vec(diag_data).reshape(&[diag_len, 1]);
        Self { data: diag_array }
    }
    
    /// Check if the matrix is square
    pub fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }
    
    /// Check if the matrix is symmetric (A = A^T)
    pub fn is_symmetric(&self) -> bool 
    where T: PartialEq
    {
        if !self.is_square() {
            return false;
        }
        
        let n = self.nrows();
        
        for i in 0..n {
            for j in (i+1)..n {
                let a_ij = self.data.get(&[i, j]).unwrap();
                let a_ji = self.data.get(&[j, i]).unwrap();
                
                if a_ij != a_ji {
                    return false;
                }
            }
        }
        
        true
    }
}

// Matrix multiplication implementation
impl<T> Matrix<T>
where T: Clone + Default + Add<Output = T> + Mul<Output = T> + Zero + One + PartialEq + PartialOrd
{
    /// Perform matrix multiplication
    pub fn dot(&self, other: &Matrix<T>) -> Result<Matrix<T>> {
        if self.ncols() != other.nrows() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![self.nrows(), other.ncols()],
                actual: vec![self.nrows(), self.ncols()],
            });
        }
        
        let m = self.nrows();
        let p = self.ncols(); // = other.nrows()
        let n = other.ncols();
        
        // Initialize result matrix
        let mut result = Matrix::zeros(m, n);
        
        // Compute matrix product
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                
                for k in 0..p {
                    let a_ik = self.data.get(&[i, k]).unwrap().clone();
                    let b_kj = other.data.get(&[k, j]).unwrap().clone();
                    
                    // sum += a_ik * b_kj
                    sum = sum + (a_ik * b_kj);
                }
                
                result.set(i, j, sum)?;
            }
        }
        
        Ok(result)
    }
}

// Implement conversion from Array to Matrix
impl<T> TryFrom<Array<T>> for Matrix<T> 
where T: Clone + Zero + One + PartialEq + Default + PartialOrd
{
    type Error = NumRs2Error;
    
    fn try_from(array: Array<T>) -> Result<Self> {
        Matrix::new(array)
    }
}

// Implement arithmetic operations
impl<T> Add for &Matrix<T> 
where T: Clone + Add<Output = T> + Zero + One + PartialEq + Default + PartialOrd {
    type Output = Matrix<T>;
    
    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        let result_array = self.data.add_broadcast(&other.data).unwrap();
        Matrix { data: result_array }
    }
}

impl<T> Sub for &Matrix<T>
where T: Clone + Sub<Output = T> + Zero + One + PartialEq + Default + PartialOrd {
    type Output = Matrix<T>;
    
    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        let result_array = self.data.subtract_broadcast(&other.data).unwrap();
        Matrix { data: result_array }
    }
}

// For matrices, multiplication is matrix multiplication, not element-wise
impl<T: Clone + Default + Add<Output = T> + Mul<Output = T> + Zero + One + PartialEq + PartialOrd> Mul for &Matrix<T> {
    type Output = Matrix<T>;
    
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        // Call the dot method for matrix multiplication
        self.dot(other).unwrap()
    }
}

impl<T> Div<T> for &Matrix<T> 
where T: Clone + Div<Output = T> + Zero + One + PartialEq + Default + PartialOrd {
    type Output = Matrix<T>;
    
    fn div(self, scalar: T) -> Matrix<T> {
        let result_array = self.data.map(|x| x.clone() / scalar.clone());
        Matrix { data: result_array }
    }
}

// Display implementation
impl<T> fmt::Display for Matrix<T> 
where T: Clone + fmt::Display + Zero + One + PartialEq + Default + PartialOrd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (rows, cols) = self.shape();
        
        writeln!(f, "Matrix({}, {})", rows, cols)?;
        
        for i in 0..rows {
            write!(f, "[")?;
            
            for j in 0..cols {
                let value = match self.data.get(&[i, j]) {
                    Ok(v) => v.clone(),
                    Err(_) => panic!("Failed to access element at ({}, {})", i, j),
                };
                
                if j > 0 {
                    write!(f, ", ")?;
                }
                
                write!(f, "{}", value)?;
            }
            
            writeln!(f, "]")?;
        }
        
        Ok(())
    }
}