use crate::array::Array;
use crate::error::Result;

/// Ensure the input is at least 1-D.
///
/// # Parameters
///
/// * `arys` - One or more arrays to convert
///
/// # Returns
///
/// A tuple of arrays that are at least 1-D
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = atleast_1d(&[&a]).expect("operation should succeed");
/// assert_eq!(b[0].shape(), vec![3]);
/// ```
pub fn atleast_1d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    let mut result = Vec::with_capacity(arys.len());

    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 1-D
            let scalar_value = array
                .get(&[])
                .expect("scalar array (ndim=0) should have a value at empty index");
            result.push(Array::from_vec_shape(vec![scalar_value], &[1])?);
        } else {
            // Already at least 1-D, add a view of the array
            result.push(array.clone());
        }
    }

    Ok(result)
}

/// Ensure the input is at least 2-D.
///
/// # Parameters
///
/// * `arys` - One or more arrays to convert
///
/// # Returns
///
/// A tuple of arrays that are at least 2-D
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = atleast_2d(&[&a]).expect("operation should succeed");
/// assert_eq!(b[0].shape(), vec![1, 3]);
/// ```
pub fn atleast_2d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    let mut result = Vec::with_capacity(arys.len());

    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 2-D
            let scalar_value = array
                .get(&[])
                .expect("scalar array (ndim=0) should have a value at empty index");
            result.push(Array::from_vec_shape(vec![scalar_value], &[1, 1])?);
        } else if array.ndim() == 1 {
            // 1-D, reshape to 2-D
            let data = array.to_vec();
            let new_shape = vec![1, data.len()];
            result.push(Array::from_vec_shape(data, &new_shape)?);
        } else {
            // Already at least 2-D, add a view of the array
            result.push(array.clone());
        }
    }

    Ok(result)
}

/// Ensure the input is at least 3-D.
///
/// # Parameters
///
/// * `arys` - One or more arrays to convert
///
/// # Returns
///
/// A tuple of arrays that are at least 3-D
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = atleast_3d(&[&a]).expect("operation should succeed");
/// assert_eq!(b[0].shape(), vec![1, 3, 1]);
/// ```
pub fn atleast_3d<T: Clone + num_traits::Zero>(arys: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    let mut result = Vec::with_capacity(arys.len());

    for &array in arys {
        if array.ndim() == 0 {
            // Scalar, reshape to 3-D
            let scalar_value = array
                .get(&[])
                .expect("scalar array (ndim=0) should have a value at empty index");
            result.push(Array::from_vec_shape(vec![scalar_value], &[1, 1, 1])?);
        } else if array.ndim() == 1 {
            // 1-D, reshape to 3-D
            let data = array.to_vec();
            let new_shape = vec![1, data.len(), 1];
            result.push(Array::from_vec_shape(data, &new_shape)?);
        } else if array.ndim() == 2 {
            // 2-D, reshape to 3-D
            let data = array.to_vec();
            let shape = array.shape();
            let new_shape = vec![shape[0], shape[1], 1];
            result.push(Array::from_vec_shape(data, &new_shape)?);
        } else {
            // Already at least 3-D, add a view of the array
            result.push(array.clone());
        }
    }

    Ok(result)
}
