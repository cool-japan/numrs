use crate::array::Array;
use crate::error::Result;

/// Broadcast any number of arrays to a common shape
///
/// # Parameters
///
/// * `arrays` - A slice of arrays to broadcast
///
/// # Returns
///
/// A vector of arrays all broadcast to the same shape
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
/// let b = Array::from_vec(vec![10, 20, 30]).reshape(&[3, 1]);
/// let broadcasts = broadcast_arrays(&[&a, &b]).unwrap();
/// assert_eq!(broadcasts[0].shape(), vec![3, 3]);
/// assert_eq!(broadcasts[1].shape(), vec![3, 3]);
/// ```
pub fn broadcast_arrays<T: Clone>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    // Calculate the broadcast shape
    let mut broadcast_shape = arrays[0].shape();

    for arr in arrays.iter().skip(1) {
        broadcast_shape = Array::<T>::broadcast_shape(&broadcast_shape, &arr.shape())?;
    }

    // Broadcast each array to the common shape
    let mut result = Vec::with_capacity(arrays.len());

    for &arr in arrays {
        let broadcasted = arr.broadcast_to(&broadcast_shape)?;
        result.push(broadcasted);
    }

    Ok(result)
}

/// Broadcast an array to the given shape without copying the data
///
/// # Parameters
///
/// * `array` - The array to broadcast
/// * `shape` - The target shape
///
/// # Returns
///
/// A new array with the specified shape sharing the underlying data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 1x3 array (already 2D)
/// let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
///
/// // Broadcast to 2x3
/// let b = broadcast_to(&a, &[2, 3]).unwrap();
/// assert_eq!(b.shape(), vec![2, 3]);
/// ```
pub fn broadcast_to<T: Clone>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>> {
    array.broadcast_to(shape)
}