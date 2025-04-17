//! Compatibility with the nalgebra crate
//!
//! This module provides conversions between NumRS Array and nalgebra matrix
//! structures, allowing for interoperability between the two libraries.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use nalgebra::{DMatrix, DVector, Matrix, Scalar, Dim};
use num_traits::NumCast;
use std::fmt::Debug;

/// Convert from nalgebra DMatrix to NumRS Array
///
/// # Arguments
///
/// * `matrix` - nalgebra DMatrix to convert
///
/// # Returns
///
/// A NumRS Array containing the same data as the input nalgebra matrix
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use nalgebra::DMatrix;
/// use numrs2::interop::nalgebra_compat::from_dmatrix;
///
/// // Create a 2x3 nalgebra matrix
/// let na_mat = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// // Convert to NumRS Array
/// let num_arr = from_dmatrix(&na_mat).unwrap();
///
/// assert_eq!(num_arr.shape(), vec![2, 3]);
/// assert_eq!(num_arr.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn from_dmatrix<T>(matrix: &DMatrix<T>) -> Result<Array<T>> 
where 
    T: Clone + Debug + NumCast + Scalar
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    
    // Convert to column-major to row-major ordering
    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            data.push(matrix[(i, j)].clone());
        }
    }
    
    // Create NumRS Array
    let arr = Array::from_vec(data);
    Ok(arr.reshape(&[nrows, ncols]))
}

/// Convert from NumRS Array to nalgebra DMatrix
///
/// # Arguments
///
/// * `arr` - NumRS Array to convert
///
/// # Returns
///
/// A nalgebra DMatrix containing the same data as the input NumRS Array
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use nalgebra::DMatrix;
/// use numrs2::interop::nalgebra_compat::to_dmatrix;
///
/// // Create a 2D NumRS Array
/// let num_arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
///
/// // Convert to nalgebra DMatrix
/// let na_mat = to_dmatrix(&num_arr).unwrap();
///
/// assert_eq!(na_mat.nrows(), 2);
/// assert_eq!(na_mat.ncols(), 3);
/// assert_eq!(na_mat[(0, 0)], 1.0);
/// assert_eq!(na_mat[(1, 2)], 6.0);
/// ```
pub fn to_dmatrix<T>(arr: &Array<T>) -> Result<DMatrix<T>> 
where 
    T: Clone + Debug + Scalar
{
    // Check dimensions
    if arr.ndim() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Cannot convert Array with shape {:?} to DMatrix, expected 2D array", arr.shape())
        ));
    }
    
    let nrows = arr.shape()[0];
    let ncols = arr.shape()[1];
    
    // Convert data from row-major to column-major ordering
    let data = arr.to_vec();
    
    // Create nalgebra DMatrix
    let matrix = DMatrix::from_row_slice(nrows, ncols, &data);
    
    Ok(matrix)
}

/// Convert from nalgebra DVector to NumRS Array
///
/// # Arguments
///
/// * `vector` - nalgebra DVector to convert
///
/// # Returns
///
/// A NumRS Array containing the same data as the input nalgebra vector
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use nalgebra::DVector;
/// use numrs2::interop::nalgebra_compat::from_dvector;
///
/// // Create a nalgebra vector
/// let na_vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// // Convert to NumRS Array
/// let num_arr = from_dvector(&na_vec).unwrap();
///
/// assert_eq!(num_arr.shape(), vec![4]);
/// assert_eq!(num_arr.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub fn from_dvector<T>(vector: &DVector<T>) -> Result<Array<T>> 
where 
    T: Clone + Debug + NumCast + Scalar
{
    let data: Vec<T> = vector.iter().cloned().collect();
    Ok(Array::from_vec(data))
}

/// Convert from NumRS Array to nalgebra DVector
///
/// # Arguments
///
/// * `arr` - NumRS Array to convert
///
/// # Returns
///
/// A nalgebra DVector containing the same data as the input NumRS Array
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use nalgebra::DVector;
/// use numrs2::interop::nalgebra_compat::to_dvector;
///
/// // Create a 1D NumRS Array
/// let num_arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// // Convert to nalgebra DVector
/// let na_vec = to_dvector(&num_arr).unwrap();
///
/// assert_eq!(na_vec.len(), 4);
/// assert_eq!(na_vec[0], 1.0);
/// assert_eq!(na_vec[3], 4.0);
/// ```
pub fn to_dvector<T>(arr: &Array<T>) -> Result<DVector<T>> 
where 
    T: Clone + Debug + Scalar
{
    // Check dimensions - should be 1D or column vector (nx1)
    let is_vector = arr.ndim() == 1 || 
                   (arr.ndim() == 2 && (arr.shape()[0] == 1 || arr.shape()[1] == 1));
    
    if !is_vector {
        return Err(NumRs2Error::DimensionMismatch(
            format!("Cannot convert Array with shape {:?} to DVector, expected 1D array or column/row vector", arr.shape())
        ));
    }
    
    // Get the data
    let data = arr.to_vec();
    
    // Create nalgebra DVector
    let vector = DVector::from_vec(data);
    
    Ok(vector)
}

/// Convert from nalgebra Matrix to NumRS Array
///
/// # Arguments
///
/// * `matrix` - nalgebra Matrix to convert
///
/// # Returns
///
/// A NumRS Array containing the same data as the input nalgebra matrix
pub fn from_matrix<T, R, C, S>(matrix: &Matrix<T, R, C, S>) -> Result<Array<T>> 
where 
    T: Clone + Debug + NumCast + Scalar,
    R: Dim,
    C: Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    
    // Convert to column-major to row-major ordering
    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            data.push(matrix[(i, j)].clone());
        }
    }
    
    // Create NumRS Array
    let arr = Array::from_vec(data);
    Ok(arr.reshape(&[nrows, ncols]))
}

/// Convert nalgebra Dynamic-size matrix to NumRS Array
///
/// This is a convenience function that handles both DMatrix and DVector types
/// using a generic approach.
///
/// # Arguments
///
/// * `matrix` - nalgebra matrix with Dynamic dimensions
///
/// # Returns
///
/// A NumRS Array containing the same data
pub fn from_dynamic<T, R, C, S>(matrix: &Matrix<T, R, C, S>) -> Result<Array<T>> 
where 
    T: Clone + Debug + NumCast + Scalar,
    R: Dim,
    C: Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    from_matrix(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    
    #[test]
    fn test_from_dmatrix() {
        // Create a nalgebra matrix
        let na_mat = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // Convert to NumRS Array
        let num_arr = from_dmatrix(&na_mat).unwrap();
        
        assert_eq!(num_arr.shape(), vec![2, 3]);
        assert_eq!(num_arr.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    
    #[test]
    fn test_to_dmatrix() {
        // Create a NumRS Array
        let num_arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        
        // Convert to nalgebra matrix
        let na_mat = to_dmatrix(&num_arr).unwrap();
        
        assert_eq!(na_mat.nrows(), 2);
        assert_eq!(na_mat.ncols(), 3);
        assert_eq!(na_mat[(0, 0)], 1.0);
        assert_eq!(na_mat[(1, 2)], 6.0);
    }
    
    #[test]
    fn test_from_dvector() {
        // Create a nalgebra vector
        let na_vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        // Convert to NumRS Array
        let num_arr = from_dvector(&na_vec).unwrap();
        
        assert_eq!(num_arr.shape(), vec![4]);
        assert_eq!(num_arr.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_to_dvector() {
        // Create a NumRS Array
        let num_arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        // Convert to nalgebra vector
        let na_vec = to_dvector(&num_arr).unwrap();
        
        assert_eq!(na_vec.len(), 4);
        assert_eq!(na_vec[0], 1.0);
        assert_eq!(na_vec[3], 4.0);
    }
    
    #[test]
    fn test_fixed_size_matrix() {
        // Create a fixed-size Matrix3
        let mat3 = Matrix3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        );
        
        // Convert to NumRS Array
        let arr = from_matrix(&mat3).unwrap();
        
        assert_eq!(arr.shape(), vec![3, 3]);
        assert_eq!(arr.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(arr.get(&[2, 2]).unwrap(), 9.0);
    }
    
    #[test]
    fn test_fixed_size_vector() {
        // Create a fixed-size Vector3
        let vec3 = Vector3::new(1.0, 2.0, 3.0);
        
        // Convert to NumRS Array
        let arr = from_matrix(&vec3).unwrap();
        
        assert_eq!(arr.shape(), vec![3, 1]);
        assert_eq!(arr.to_vec(), vec![1.0, 2.0, 3.0]);
    }
}