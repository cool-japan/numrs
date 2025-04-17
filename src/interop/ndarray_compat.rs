//! Compatibility with the ndarray crate
//!
//! This module provides conversions between NumRS Array and ndarray Array
//! structures, allowing for interoperability between the two libraries.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use ndarray::{Array as NdArray, ArrayD, IxDyn, Dimension};
use num_traits::NumCast;
use std::fmt::Debug;

/// Convert from ndarray::Array to NumRS Array
///
/// # Arguments
///
/// * `ndarr` - ndarray Array to convert
///
/// # Returns
///
/// A NumRS Array containing the same data as the input ndarray Array
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use ndarray::{Array as NdArray, IxDyn};
/// use numrs2::interop::ndarray_compat::from_ndarray;
///
/// // Create a 2D ndarray
/// let nd_arr = NdArray::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
///
/// // Convert to NumRS Array
/// let num_arr = from_ndarray(&nd_arr).unwrap();
///
/// assert_eq!(num_arr.shape(), vec![2, 3]);
/// assert_eq!(num_arr.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn from_ndarray<T, D>(ndarr: &NdArray<T, D>) -> Result<Array<T>> 
where 
    T: Clone + Debug + NumCast,
    D: Dimension,
{
    // Convert shape
    let shape: Vec<usize> = ndarr.shape().to_vec();
    
    // Convert data
    let data: Vec<T> = ndarr.iter().cloned().collect();
    
    // Create NumRS Array
    let arr = Array::from_vec(data);
    Ok(arr.reshape(&shape))
}

/// Convert from NumRS Array to ndarray::Array
///
/// # Arguments
///
/// * `arr` - NumRS Array to convert
///
/// # Returns
///
/// An ndarray Array containing the same data as the input NumRS Array
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use ndarray::{Array as NdArray, IxDyn};
/// use numrs2::interop::ndarray_compat::to_ndarray;
///
/// // Create a 2D NumRS Array
/// let num_arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
///
/// // Convert to ndarray
/// let nd_arr = to_ndarray(&num_arr).unwrap();
///
/// assert_eq!(nd_arr.shape(), &[2, 3]);
/// assert_eq!(nd_arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn to_ndarray<T>(arr: &Array<T>) -> Result<ArrayD<T>> 
where 
    T: Clone + Debug
{
    // Convert shape
    let shape: Vec<usize> = arr.shape();
    
    // Convert data
    let data = arr.to_vec();
    
    // Create ndarray Array
    NdArray::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| NumRs2Error::ConversionError(format!("Failed to convert to ndarray: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_from_ndarray_2d() {
        // Create a 2D ndarray
        let nd_arr = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
        
        // Convert to NumRS Array
        let num_arr = from_ndarray(&nd_arr).unwrap();
        
        assert_eq!(num_arr.shape(), vec![2, 3]);
        assert_eq!(num_arr.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }
    
    #[test]
    fn test_to_ndarray_2d() {
        // Create a 2D NumRS Array
        let num_arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        
        // Convert to ndarray
        let nd_arr = to_ndarray(&num_arr).unwrap();
        
        assert_eq!(nd_arr.shape(), &[2, 3]);
        assert_eq!(nd_arr.as_slice().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }
    
    #[test]
    fn test_round_trip_conversion() {
        // Create a NumRS Array
        let original = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        
        // Convert to ndarray and back
        let nd_arr = to_ndarray(&original).unwrap();
        let round_trip = from_ndarray(&nd_arr).unwrap();
        
        assert_eq!(original.shape(), round_trip.shape());
        assert_eq!(original.to_vec(), round_trip.to_vec());
    }
}