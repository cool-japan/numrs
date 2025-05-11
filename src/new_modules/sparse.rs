use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Zero, One};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div};

/// Sparse array implementation for NumRS
/// Coordinate (COO) format sparse array
#[derive(Clone, Debug)]
pub struct SparseArray<T> {
    /// Indices and values of non-zero elements
    pub data: HashMap<Vec<usize>, T>,
    /// Shape of the array
    pub shape: Vec<usize>,
}

impl<T> SparseArray<T>
where
    T: Clone + PartialEq + Zero,
{
    /// Create a new empty sparse array with given shape
    pub fn new(shape: &[usize]) -> Self {
        SparseArray {
            data: HashMap::new(),
            shape: shape.to_vec(),
        }
    }
    
    /// Create a new sparse array from a dense array
    pub fn from_array(array: &Array<T>) -> Self {
        let shape = array.shape();
        let dense_data = array.to_vec();
        
        let mut data = HashMap::new();
        let mut idx = vec![0; shape.len()];
        let mut size = 1;
        
        for i in (0..shape.len()).rev() {
            size *= shape[i];
        }
        
        for (i, value) in dense_data.iter().enumerate().take(size) {
            // Calculate the multi-dimensional index
            let mut temp = i;
            for j in (0..shape.len()).rev() {
                idx[j] = temp % shape[j];
                temp /= shape[j];
            }

            let value = value.clone();
            if value != T::zero() {
                data.insert(idx.clone(), value);
            }
        }
        
        SparseArray {
            data,
            shape,
        }
    }
    
    /// Convert sparse array to dense array
    pub fn to_array(&self) -> Array<T> {
        // Calculate total size
        let size: usize = self.shape.iter().product();
        
        // Create dense array filled with zeros
        let mut dense_data = vec![T::zero(); size];
        
        // Fill non-zero elements
        for (indices, value) in &self.data {
            let mut idx = 0;
            let mut stride = 1;
            
            // Convert multi-dimensional index to flat index
            for i in (0..indices.len()).rev() {
                idx += indices[i] * stride;
                if i > 0 {
                    stride *= self.shape[i];
                }
            }
            
            dense_data[idx] = value.clone();
        }
        
        Array::from_vec(dense_data).reshape(&self.shape)
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
    
    /// Get the density (ratio of non-zero elements to total elements)
    pub fn density(&self) -> f64 {
        let total_size: usize = self.shape.iter().product();
        if total_size == 0 {
            return 0.0;
        }
        
        self.nnz() as f64 / total_size as f64
    }
    
    /// Get the shape of the sparse array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Check if an index is valid for this array's shape
    fn check_index(&self, indices: &[usize]) -> Result<()> {
        if indices.len() != self.shape.len() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Index has wrong number of dimensions: expected {}, got {}", 
                       self.shape.len(), indices.len())
            ));
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Index {} is out of bounds for dimension {} with size {}", 
                           idx, i, self.shape[i])
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get an element at the specified indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        self.check_index(indices)?;
        
        Ok(self.data.get(indices).cloned().unwrap_or_else(T::zero))
    }
    
    /// Set an element at the specified indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        self.check_index(indices)?;
        
        if value == T::zero() {
            self.data.remove(indices);
        } else {
            self.data.insert(indices.to_vec(), value);
        }
        
        Ok(())
    }
}

/// Arithmetic operations for sparse arrays
impl<T> SparseArray<T>
where
    T: Clone + PartialEq + Zero + Add<Output = T>,
{
    /// Add two sparse arrays
    pub fn add(&self, other: &SparseArray<T>) -> Result<SparseArray<T>> {
        if self.shape != other.shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let mut result = SparseArray::new(&self.shape);
        
        // Add entries from self
        for (indices, value) in &self.data {
            result.data.insert(indices.clone(), value.clone());
        }
        
        // Add entries from other
        for (indices, value) in &other.data {
            if let Some(existing) = result.data.get_mut(indices) {
                // Add to existing value
                *existing = existing.clone() + value.clone();
                
                // Remove if result is zero
                if *existing == T::zero() {
                    result.data.remove(indices);
                }
            } else {
                // Insert new value
                result.data.insert(indices.clone(), value.clone());
            }
        }
        
        Ok(result)
    }
}

impl<T> SparseArray<T>
where
    T: Clone + PartialEq + Zero + Sub<Output = T>,
{
    /// Subtract another sparse array
    pub fn subtract(&self, other: &SparseArray<T>) -> Result<SparseArray<T>> {
        if self.shape != other.shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let mut result = SparseArray::new(&self.shape);
        
        // Add entries from self
        for (indices, value) in &self.data {
            result.data.insert(indices.clone(), value.clone());
        }
        
        // Subtract entries from other
        for (indices, value) in &other.data {
            if let Some(existing) = result.data.get_mut(indices) {
                // Subtract from existing value
                *existing = existing.clone() - value.clone();
                
                // Remove if result is zero
                if *existing == T::zero() {
                    result.data.remove(indices);
                }
            } else {
                // Insert negated value
                let neg_value = T::zero() - value.clone();
                if neg_value != T::zero() {
                    result.data.insert(indices.clone(), neg_value);
                }
            }
        }
        
        Ok(result)
    }
}

impl<T> SparseArray<T>
where
    T: Clone + PartialEq + Zero + Mul<Output = T>,
{
    /// Elementwise multiplication with another sparse array
    pub fn multiply(&self, other: &SparseArray<T>) -> Result<SparseArray<T>> {
        if self.shape != other.shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let mut result = SparseArray::new(&self.shape);
        
        // For elementwise multiplication, only need to consider indices that are in both arrays
        for (indices, value) in &self.data {
            if let Some(other_value) = other.data.get(indices) {
                let product = value.clone() * other_value.clone();
                if product != T::zero() {
                    result.data.insert(indices.clone(), product);
                }
            }
        }
        
        Ok(result)
    }
    
    /// Multiply by a scalar
    pub fn multiply_scalar(&self, scalar: T) -> SparseArray<T> {
        if scalar == T::zero() {
            return SparseArray::new(&self.shape);
        }
        
        let mut result = SparseArray::new(&self.shape);
        
        for (indices, value) in &self.data {
            let product = value.clone() * scalar.clone();
            if product != T::zero() {
                result.data.insert(indices.clone(), product);
            }
        }
        
        result
    }
}

impl<T> SparseArray<T>
where
    T: Clone + PartialEq + Zero + Div<Output = T>,
{
    /// Elementwise division by another sparse array
    pub fn divide(&self, other: &SparseArray<T>) -> Result<SparseArray<T>> {
        if self.shape != other.shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let mut result = SparseArray::new(&self.shape);
        
        // For each non-zero element in self
        for (indices, value) in &self.data {
            // Get the corresponding value in other
            let other_value = other.data.get(indices).cloned().unwrap_or_else(T::zero);
            
            if other_value == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "Division by zero in sparse array".to_string()
                ));
            }
            
            let quotient = value.clone() / other_value;
            if quotient != T::zero() {
                result.data.insert(indices.clone(), quotient);
            }
        }
        
        Ok(result)
    }
    
    /// Divide by a scalar
    pub fn divide_scalar(&self, scalar: T) -> Result<SparseArray<T>> {
        if scalar == T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "Division by zero scalar".to_string()
            ));
        }
        
        let mut result = SparseArray::new(&self.shape);
        
        for (indices, value) in &self.data {
            let quotient = value.clone() / scalar.clone();
            if quotient != T::zero() {
                result.data.insert(indices.clone(), quotient);
            }
        }
        
        Ok(result)
    }
}

/// Types of sparse matrices for specialized use cases
#[derive(Clone, Debug)]
pub enum SparseMatrixFormat {
    /// Coordinate format (most general)
    COO,
    /// Compressed Sparse Row format (efficient for row operations)
    CSR,
    /// Compressed Sparse Column format (efficient for column operations)
    CSC,
    /// Diagonal format (efficient for diagonal-heavy matrices)
    DIA,
}

/// Specialized sparse matrix for 2D arrays
#[derive(Clone, Debug)]
pub struct SparseMatrix<T> {
    /// The underlying sparse array
    pub array: SparseArray<T>,
    /// The internal format of the sparse matrix
    pub format: SparseMatrixFormat,
    /// For CSR/CSC formats: indices/pointers
    pub indices: Option<Vec<usize>>,
    pub indptr: Option<Vec<usize>>,
    /// For DIA format: offsets
    pub diag_offsets: Option<Vec<isize>>,
}

impl<T> SparseMatrix<T>
where
    T: Clone + PartialEq + Zero + Debug,
{
    /// Create a new sparse matrix in COO format
    pub fn new(shape: &[usize]) -> Result<Self> {
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "SparseMatrix requires a 2D shape".to_string()
            ));
        }
        
        Ok(SparseMatrix {
            array: SparseArray::new(shape),
            format: SparseMatrixFormat::COO,
            indices: None,
            indptr: None,
            diag_offsets: None,
        })
    }
    
    /// Create a new sparse matrix from a dense array
    pub fn from_array(array: &Array<T>) -> Result<Self> {
        let shape = array.shape();
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "SparseMatrix requires a 2D array".to_string()
            ));
        }
        
        Ok(SparseMatrix {
            array: SparseArray::from_array(array),
            format: SparseMatrixFormat::COO,
            indices: None,
            indptr: None,
            diag_offsets: None,
        })
    }
    
    /// Create a new identity sparse matrix
    pub fn eye(n: usize) -> Result<Self>
    where
        T: One,
    {
        let mut matrix = SparseMatrix::new(&[n, n])?;
        
        for i in 0..n {
            matrix.array.set(&[i, i], T::one())?;
        }
        
        Ok(matrix)
    }
    
    /// Create a new diagonal sparse matrix from a vector
    pub fn diag(diagonal: &[T]) -> Result<Self> {
        let n = diagonal.len();
        let mut matrix = SparseMatrix::new(&[n, n])?;
        
        for (i, value) in diagonal.iter().enumerate().take(n) {
            if *value != T::zero() {
                matrix.array.set(&[i, i], value.clone())?;
            }
        }
        
        Ok(matrix)
    }
    
    /// Convert to CSR format for efficient row operations
    pub fn to_csr(&mut self) -> Result<()> {
        if let SparseMatrixFormat::CSR = self.format {
            // Already in CSR format
            return Ok(());
        }
        
        let n_rows = self.array.shape[0];
        let n_cols = self.array.shape[1];
        let nnz = self.array.nnz();
        
        // Create data structures for CSR format
        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        
        // Initialize row pointer
        indptr.push(0);
        
        // Sort entries by row, then column
        let mut entries: Vec<((usize, usize), T)> = self.array.data.iter()
            .map(|(idx, val)| ((idx[0], idx[1]), val.clone()))
            .collect();
        
        entries.sort_by_key(|((row, col), _)| (*row, *col));
        
        // Build CSR representation
        let mut current_row = 0;
        for ((row, col), val) in entries {
            // Add empty rows if needed
            while current_row < row {
                current_row += 1;
                indptr.push(data.len());
            }
            
            // Add this element
            data.push(val);
            indices.push(col);
        }
        
        // Finish indptr
        while indptr.len() <= n_rows {
            indptr.push(data.len());
        }
        
        // Create new array with the data
        let mut new_array = SparseArray::new(&[n_rows, n_cols]);
        for i in 0..n_rows {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            
            for j in row_start..row_end {
                let col = indices[j];
                let val = data[j].clone();
                new_array.set(&[i, col], val)?;
            }
        }
        
        // Update the matrix
        self.array = new_array;
        self.format = SparseMatrixFormat::CSR;
        self.indices = Some(indices);
        self.indptr = Some(indptr);
        self.diag_offsets = None;
        
        Ok(())
    }
    
    /// Convert to CSC format for efficient column operations
    pub fn to_csc(&mut self) -> Result<()> {
        if let SparseMatrixFormat::CSC = self.format {
            // Already in CSC format
            return Ok(());
        }
        
        let n_rows = self.array.shape[0];
        let n_cols = self.array.shape[1];
        let nnz = self.array.nnz();
        
        // Create data structures for CSC format
        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(n_cols + 1);
        
        // Initialize column pointer
        indptr.push(0);
        
        // Sort entries by column, then row
        let mut entries: Vec<((usize, usize), T)> = self.array.data.iter()
            .map(|(idx, val)| ((idx[0], idx[1]), val.clone()))
            .collect();
        
        entries.sort_by_key(|((row, col), _)| (*col, *row));
        
        // Build CSC representation
        let mut current_col = 0;
        for ((row, col), val) in entries {
            // Add empty columns if needed
            while current_col < col {
                current_col += 1;
                indptr.push(data.len());
            }
            
            // Add this element
            data.push(val);
            indices.push(row);
        }
        
        // Finish indptr
        while indptr.len() <= n_cols {
            indptr.push(data.len());
        }
        
        // Create new array with the data
        let mut new_array = SparseArray::new(&[n_rows, n_cols]);
        for j in 0..n_cols {
            let col_start = indptr[j];
            let col_end = indptr[j + 1];
            
            for i in col_start..col_end {
                let row = indices[i];
                let val = data[i].clone();
                new_array.set(&[row, j], val)?;
            }
        }
        
        // Update the matrix
        self.array = new_array;
        self.format = SparseMatrixFormat::CSC;
        self.indices = Some(indices);
        self.indptr = Some(indptr);
        self.diag_offsets = None;
        
        Ok(())
    }
    
    /// Convert to diagonal format for diagonal-heavy matrices
    pub fn to_dia(&mut self) -> Result<()> {
        if let SparseMatrixFormat::DIA = self.format {
            // Already in DIA format
            return Ok(());
        }
        
        let n_rows = self.array.shape[0];
        let n_cols = self.array.shape[1];
        
        // Find all diagonals with non-zero elements
        let mut diag_indices = std::collections::HashSet::new();
        
        for indices in self.array.data.keys() {
            let row = indices[0];
            let col = indices[1];
            let diag_idx = col as isize - row as isize;
            diag_indices.insert(diag_idx);
        }
        
        // Convert to sorted vector of diagonal offsets
        let mut diag_offsets: Vec<isize> = diag_indices.into_iter().collect();
        diag_offsets.sort();
        
        // Create new array with the data organized by diagonals
        let mut new_array = SparseArray::new(&[n_rows, n_cols]);
        
        for indices in self.array.data.keys() {
            let row = indices[0];
            let col = indices[1];
            let value = self.array.data.get(indices).cloned().unwrap_or_else(T::zero);
            
            new_array.set(&[row, col], value)?;
        }
        
        // Update the matrix
        self.array = new_array;
        self.format = SparseMatrixFormat::DIA;
        self.indices = None;
        self.indptr = None;
        self.diag_offsets = Some(diag_offsets);
        
        Ok(())
    }
    
    /// Convert to dense array
    pub fn to_array(&self) -> Array<T> {
        self.array.to_array()
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.array.nnz()
    }
    
    /// Get the density (ratio of non-zero elements to total elements)
    pub fn density(&self) -> f64 {
        self.array.density()
    }
    
    /// Get the shape of the sparse matrix
    pub fn shape(&self) -> &[usize] {
        &self.array.shape
    }
    
    /// Get an element at the specified indices
    pub fn get(&self, row: usize, col: usize) -> Result<T> {
        self.array.get(&[row, col])
    }
    
    /// Set an element at the specified indices
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        self.array.set(&[row, col], value)
    }
}

// Implement matrix operations for sparse matrices
impl<T> SparseMatrix<T>
where
    T: Clone + PartialEq + Zero + Add<Output = T> + Mul<Output = T> + Debug,
{
    /// Matrix multiplication for sparse matrices
    pub fn matmul(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>> {
        let self_shape = self.array.shape();
        let other_shape = other.array.shape();
        
        if self_shape[1] != other_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![self_shape[0], other_shape[1]],
                actual: vec![self_shape[0], self_shape[1]],
            });
        }
        
        // Convert sparse matrices to appropriate formats for efficient multiplication
        let mut self_matrix = self.clone();
        let mut other_matrix = other.clone();
        
        // For matrix multiplication, CSR for left operand and CSC for right operand are efficient
        self_matrix.to_csr()?;
        other_matrix.to_csc()?;
        
        let n_rows = self_shape[0];
        let n_cols = other_shape[1];
        let _k = self_shape[1];  // = other_shape[0]
        
        let mut result = SparseMatrix::new(&[n_rows, n_cols])?;
        
        // Get CSR and CSC data
        let self_indptr = self_matrix.indptr.as_ref().unwrap();
        let self_indices = self_matrix.indices.as_ref().unwrap();
        
        let other_indptr = other_matrix.indptr.as_ref().unwrap();
        let other_indices = other_matrix.indices.as_ref().unwrap();
        
        // Perform sparse matrix multiplication
        for i in 0..n_rows {
            let row_start = self_indptr[i];
            let row_end = self_indptr[i + 1];
            
            for j in 0..n_cols {
                let col_start = other_indptr[j];
                let col_end = other_indptr[j + 1];
                
                let mut sum = T::zero();
                let mut added = false;
                
                // Find common indices in row i of A and column j of B
                let mut row_idx = row_start;
                let mut col_idx = col_start;
                
                while row_idx < row_end && col_idx < col_end {
                    let row_k = self_indices[row_idx];
                    let col_k = other_indices[col_idx];
                    
                    match row_k.cmp(&col_k) {
                        std::cmp::Ordering::Equal => {
                            // Common index found, multiply and add
                            let a_val = self_matrix.array.get(&[i, row_k])?;
                            let b_val = other_matrix.array.get(&[col_k, j])?;

                            sum = sum + a_val * b_val;
                            added = true;

                            row_idx += 1;
                            col_idx += 1;
                        },
                        std::cmp::Ordering::Less => {
                            row_idx += 1;
                        },
                        std::cmp::Ordering::Greater => {
                            col_idx += 1;
                        }
                    }
                }
                
                // Only add non-zero elements to the result
                if added && sum != T::zero() {
                    result.array.set(&[i, j], sum)?;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Transpose the sparse matrix
    pub fn transpose(&self) -> Result<SparseMatrix<T>> {
        let n_rows = self.array.shape[0];
        let n_cols = self.array.shape[1];
        
        let mut result = SparseMatrix::new(&[n_cols, n_rows])?;
        
        for (indices, value) in &self.array.data {
            result.array.set(&[indices[1], indices[0]], value.clone())?;
        }
        
        Ok(result)
    }
}

// Add tests to verify the implementation
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sparse_array_creation() {
        // Create a sparse array
        let mut sparse = SparseArray::new(&[3, 3]);
        
        // Set some values
        sparse.set(&[0, 0], 1.0).unwrap();
        sparse.set(&[1, 1], 2.0).unwrap();
        sparse.set(&[2, 2], 3.0).unwrap();
        
        // Check non-zero count
        assert_eq!(sparse.nnz(), 3);
        
        // Check density
        assert_relative_eq!(sparse.density(), 3.0 / 9.0);
        
        // Check retrieval
        assert_relative_eq!(sparse.get(&[0, 0]).unwrap(), 1.0);
        assert_relative_eq!(sparse.get(&[1, 1]).unwrap(), 2.0);
        assert_relative_eq!(sparse.get(&[2, 2]).unwrap(), 3.0);
        assert_relative_eq!(sparse.get(&[0, 1]).unwrap(), 0.0);
    }
    
    #[test]
    fn test_sparse_array_from_dense() {
        // Create a dense array
        let dense = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
            .reshape(&[3, 3]);
        
        // Convert to sparse
        let sparse = SparseArray::from_array(&dense);
        
        // Check non-zero count
        assert_eq!(sparse.nnz(), 3);
        
        // Check retrieval
        assert_relative_eq!(sparse.get(&[0, 0]).unwrap(), 1.0);
        assert_relative_eq!(sparse.get(&[1, 1]).unwrap(), 2.0);
        assert_relative_eq!(sparse.get(&[2, 2]).unwrap(), 3.0);
    }
    
    #[test]
    fn test_sparse_array_to_dense() {
        // Create a sparse array
        let mut sparse = SparseArray::new(&[3, 3]);
        
        // Set some values
        sparse.set(&[0, 0], 1.0).unwrap();
        sparse.set(&[1, 1], 2.0).unwrap();
        sparse.set(&[2, 2], 3.0).unwrap();
        
        // Convert to dense
        let dense = sparse.to_array();
        
        // Check values
        let dense_data = dense.to_vec();
        
        assert_relative_eq!(dense_data[0], 1.0);
        assert_relative_eq!(dense_data[4], 2.0);
        assert_relative_eq!(dense_data[8], 3.0);
        
        for i in [1, 2, 3, 5, 6, 7] {
            assert_relative_eq!(dense_data[i], 0.0);
        }
    }
    
    #[test]
    fn test_sparse_array_arithmetic() {
        // Create two sparse arrays
        let mut a = SparseArray::new(&[3, 3]);
        let mut b = SparseArray::new(&[3, 3]);
        
        // Set values in a
        a.set(&[0, 0], 1.0).unwrap();
        a.set(&[1, 1], 2.0).unwrap();
        a.set(&[2, 2], 3.0).unwrap();
        
        // Set values in b
        b.set(&[0, 0], 2.0).unwrap();
        b.set(&[1, 1], 1.0).unwrap();
        b.set(&[0, 2], 4.0).unwrap();
        
        // Test addition
        let sum = a.add(&b).unwrap();
        assert_relative_eq!(sum.get(&[0, 0]).unwrap(), 3.0);
        assert_relative_eq!(sum.get(&[1, 1]).unwrap(), 3.0);
        assert_relative_eq!(sum.get(&[2, 2]).unwrap(), 3.0);
        assert_relative_eq!(sum.get(&[0, 2]).unwrap(), 4.0);
        
        // Test subtraction
        let diff = a.subtract(&b).unwrap();
        assert_relative_eq!(diff.get(&[0, 0]).unwrap(), -1.0);
        assert_relative_eq!(diff.get(&[1, 1]).unwrap(), 1.0);
        assert_relative_eq!(diff.get(&[2, 2]).unwrap(), 3.0);
        assert_relative_eq!(diff.get(&[0, 2]).unwrap(), -4.0);
        
        // Test multiplication
        let prod = a.multiply(&b).unwrap();
        assert_relative_eq!(prod.get(&[0, 0]).unwrap(), 2.0);
        assert_relative_eq!(prod.get(&[1, 1]).unwrap(), 2.0);
        assert_relative_eq!(prod.get(&[2, 2]).unwrap(), 0.0);
        assert_relative_eq!(prod.get(&[0, 2]).unwrap(), 0.0);
        
        // Test scalar multiplication
        let scaled = a.multiply_scalar(2.0);
        assert_relative_eq!(scaled.get(&[0, 0]).unwrap(), 2.0);
        assert_relative_eq!(scaled.get(&[1, 1]).unwrap(), 4.0);
        assert_relative_eq!(scaled.get(&[2, 2]).unwrap(), 6.0);
    }
    
    #[test]
    fn test_sparse_matrix_creation() {
        // Create a sparse matrix
        let mut sparse = SparseMatrix::new(&[3, 3]).unwrap();
        
        // Set some values
        sparse.set(0, 0, 1.0).unwrap();
        sparse.set(1, 1, 2.0).unwrap();
        sparse.set(2, 2, 3.0).unwrap();
        
        // Check non-zero count
        assert_eq!(sparse.nnz(), 3);
        
        // Check retrieval
        assert_relative_eq!(sparse.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(sparse.get(1, 1).unwrap(), 2.0);
        assert_relative_eq!(sparse.get(2, 2).unwrap(), 3.0);
        assert_relative_eq!(sparse.get(0, 1).unwrap(), 0.0);
    }
    
    #[test]
    fn test_sparse_matrix_special_constructors() {
        // Create an identity matrix with explicit type
        let eye: SparseMatrix<f64> = SparseMatrix::eye(3).unwrap();
        
        // Check diagonal elements
        assert_relative_eq!(eye.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(eye.get(1, 1).unwrap(), 1.0);
        assert_relative_eq!(eye.get(2, 2).unwrap(), 1.0);
        
        // Check off-diagonal elements
        assert_relative_eq!(eye.get(0, 1).unwrap(), 0.0);
        assert_relative_eq!(eye.get(1, 2).unwrap(), 0.0);
        
        // Create a diagonal matrix
        let diag = SparseMatrix::diag(&[1.0, 2.0, 3.0]).unwrap();
        
        // Check diagonal elements
        assert_relative_eq!(diag.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(diag.get(1, 1).unwrap(), 2.0);
        assert_relative_eq!(diag.get(2, 2).unwrap(), 3.0);
        
        // Check off-diagonal elements
        assert_relative_eq!(diag.get(0, 1).unwrap(), 0.0);
        assert_relative_eq!(diag.get(1, 2).unwrap(), 0.0);
    }
    
    #[test]
    fn test_sparse_matrix_format_conversion() {
        // Create a sparse matrix
        let mut sparse = SparseMatrix::new(&[3, 3]).unwrap();
        
        // Set some values
        sparse.set(0, 0, 1.0).unwrap();
        sparse.set(0, 2, 2.0).unwrap();
        sparse.set(1, 1, 3.0).unwrap();
        sparse.set(2, 0, 4.0).unwrap();
        
        // Convert to CSR format
        sparse.to_csr().unwrap();
        
        // Check that format is updated
        if let SparseMatrixFormat::CSR = sparse.format {
            // Format is correct
        } else {
            panic!("Format should be CSR");
        }
        
        // Check that data is still accessible
        assert_relative_eq!(sparse.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(sparse.get(0, 2).unwrap(), 2.0);
        assert_relative_eq!(sparse.get(1, 1).unwrap(), 3.0);
        assert_relative_eq!(sparse.get(2, 0).unwrap(), 4.0);
        
        // Convert to CSC format
        sparse.to_csc().unwrap();
        
        // Check that format is updated
        if let SparseMatrixFormat::CSC = sparse.format {
            // Format is correct
        } else {
            panic!("Format should be CSC");
        }
        
        // Check that data is still accessible
        assert_relative_eq!(sparse.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(sparse.get(0, 2).unwrap(), 2.0);
        assert_relative_eq!(sparse.get(1, 1).unwrap(), 3.0);
        assert_relative_eq!(sparse.get(2, 0).unwrap(), 4.0);
    }
    
    #[test]
    fn test_sparse_matrix_operations() {
        // Create two sparse matrices
        let mut a = SparseMatrix::new(&[3, 3]).unwrap();
        let mut b = SparseMatrix::new(&[3, 2]).unwrap();
        
        // Set values in a
        a.set(0, 0, 1.0).unwrap();
        a.set(0, 1, 2.0).unwrap();
        a.set(1, 0, 3.0).unwrap();
        a.set(1, 1, 4.0).unwrap();
        a.set(2, 0, 5.0).unwrap();
        a.set(2, 1, 6.0).unwrap();
        
        // Set values in b
        b.set(0, 0, 7.0).unwrap();
        b.set(0, 1, 8.0).unwrap();
        b.set(1, 0, 9.0).unwrap();
        b.set(1, 1, 10.0).unwrap();
        
        // Test matrix multiplication
        let c = a.matmul(&b).unwrap();
        
        // Expected result: c = a * b
        assert_relative_eq!(c.get(0, 0).unwrap(), 1.0 * 7.0 + 2.0 * 9.0);
        assert_relative_eq!(c.get(0, 1).unwrap(), 1.0 * 8.0 + 2.0 * 10.0);
        assert_relative_eq!(c.get(1, 0).unwrap(), 3.0 * 7.0 + 4.0 * 9.0);
        assert_relative_eq!(c.get(1, 1).unwrap(), 3.0 * 8.0 + 4.0 * 10.0);
        assert_relative_eq!(c.get(2, 0).unwrap(), 5.0 * 7.0 + 6.0 * 9.0);
        assert_relative_eq!(c.get(2, 1).unwrap(), 5.0 * 8.0 + 6.0 * 10.0);
        
        // Test transpose
        let at = a.transpose().unwrap();
        
        assert_relative_eq!(at.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(at.get(0, 1).unwrap(), 3.0);
        assert_relative_eq!(at.get(0, 2).unwrap(), 5.0);
        assert_relative_eq!(at.get(1, 0).unwrap(), 2.0);
        assert_relative_eq!(at.get(1, 1).unwrap(), 4.0);
        assert_relative_eq!(at.get(1, 2).unwrap(), 6.0);
    }
}