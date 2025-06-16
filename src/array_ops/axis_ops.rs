use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

/// Roll the specified axis to a new position
///
/// # Parameters
///
/// * `array` - The input array
/// * `axis` - The axis to roll
/// * `start` - The new position (destination)
///
/// # Returns
///
/// A new array with the axes rearranged
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3D array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
///     .reshape(&[2, 2, 3]);
///
/// // Roll axis 2 to position 0
/// let b = rollaxis(&a, 2, 0).unwrap();
/// assert_eq!(b.shape(), vec![3, 2, 2]);
/// ```
pub fn rollaxis<T: Clone + Zero>(array: &Array<T>, axis: usize, start: usize) -> Result<Array<T>> {
    let shape = array.shape();
    let ndim = shape.len();

    if axis >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis, ndim
        )));
    }

    if start > ndim {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Start position {} exceeds array dimensions {}",
            start, ndim
        )));
    }

    if axis == start || (axis == ndim - 1 && start == ndim) {
        // No change needed
        return Ok(array.clone());
    }

    // Create a new axis order
    let mut axes: Vec<usize> = (0..ndim).collect();

    // Remove the rolled axis
    let rolled_axis = axes.remove(axis);

    // Insert it at the new position
    axes.insert(if start <= axis { start } else { start - 1 }, rolled_axis);

    // Create a new array with the axes in the new order
    // Using a simplified transposes approach for now
    // This is a simplified implementation that's not efficient for large arrays
    // For a proper implementation, we would use ndarray's permute_axes functionality
    let source_shape = array.shape().to_vec();
    let mut target_shape = Vec::with_capacity(ndim);

    for &ax in &axes {
        target_shape.push(source_shape[ax]);
    }

    let mut result_data = vec![T::zero(); array.size()];

    // Iterate through all elements of the array
    let source_size = array.size();
    let source_array = array.array();

    for i in 0..source_size {
        // Convert flat index to multi-dimensional indices
        let mut source_indices = vec![0; ndim];
        let mut remainder = i;
        for j in (0..ndim).rev() {
            source_indices[j] = remainder % source_shape[j];
            remainder /= source_shape[j];
        }

        // Map indices to the new order
        let mut target_indices = vec![0; ndim];
        for (j, &ax) in axes.iter().enumerate() {
            target_indices[j] = source_indices[ax];
        }

        // Calculate flat index in the target array
        let mut target_flat_index = 0;
        let mut multiplier = 1;
        for j in (0..ndim).rev() {
            target_flat_index += target_indices[j] * multiplier;
            multiplier *= target_shape[j];
        }

        // Copy the value
        result_data[target_flat_index] = source_array.as_slice().unwrap()[i].clone();
    }

    // Create the result array
    let result = Array::from_vec(result_data).reshape(&target_shape);
    Ok(result)
}

/// Interchange two axes of an array.
///
/// # Parameters
///
/// * `array` - The array to transform
/// * `axis1` - The first axis to swap
/// * `axis2` - The second axis to swap
///
/// # Returns
///
/// A view of the array with the axes swapped
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let b = swapaxes(&a, 0, 1).unwrap();
/// assert_eq!(b.shape(), vec![3, 2]);
/// ```
pub fn swapaxes<T: Clone>(array: &Array<T>, axis1: usize, axis2: usize) -> Result<Array<T>> {
    let ndim = array.ndim();

    // Check if the axes are valid
    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axes {} and {} are out of bounds for array of dimension {}",
            axis1, axis2, ndim
        )));
    }

    // If axes are the same, return a view of the original array
    if axis1 == axis2 {
        return Ok(array.clone());
    }

    // Create the new shape and permutation array
    let mut permutation = Vec::with_capacity(ndim);
    for i in 0..ndim {
        if i == axis1 {
            permutation.push(axis2);
        } else if i == axis2 {
            permutation.push(axis1);
        } else {
            permutation.push(i);
        }
    }

    // Transpose according to the permutation
    let mut result = array.clone();

    // Permute the axes
    for i in 0..ndim {
        if permutation[i] != i {
            // Find where the i-th axis should go
            let j = permutation[i];

            // Swap axes i and j in the result
            result = result.transpose_axis(i, j);

            // Update the permutation to reflect the swap
            permutation.swap(i, j);
        }
    }

    Ok(result)
}

/// Move the axes of an array to new positions.
///
/// # Parameters
///
/// * `array` - The array to transform
/// * `source` - The original positions of the axes to move
/// * `destination` - The destination positions of the axes to move
///
/// # Returns
///
/// A view of the array with axes moved
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8]).reshape(&[2, 2, 2]);
/// let b = moveaxis(&a, &[0], &[2]).unwrap();
/// assert_eq!(b.shape(), vec![2, 2, 2]);
/// ```
pub fn moveaxis<T: Clone>(
    array: &Array<T>,
    source: &[usize],
    destination: &[usize],
) -> Result<Array<T>> {
    let ndim = array.ndim();

    // Check if the source and destination arrays have the same length
    if source.len() != destination.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Source and destination arrays must have the same length, got {} and {}",
            source.len(),
            destination.len()
        )));
    }

    // Check if the axes are valid
    for &axis in source.iter().chain(destination.iter()) {
        if axis >= ndim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} is out of bounds for array of dimension {}",
                axis, ndim
            )));
        }
    }

    // Create an array to track the new positions of the axes
    let mut perm = Vec::with_capacity(ndim);
    for i in 0..ndim {
        perm.push(i);
    }

    // Move the axes to their destination positions
    for (&src, &dst) in source.iter().zip(destination.iter()) {
        // Remove the source axis
        let src_axis = perm.remove(src);

        // Insert it at the destination position
        if dst < perm.len() {
            perm.insert(dst, src_axis);
        } else {
            perm.push(src_axis);
        }
    }

    // Permute the axes according to perm
    let mut result = array.clone();

    // Apply the permutation
    for i in 0..ndim {
        if perm[i] != i {
            // Find where the i-th axis should go
            let j = perm.iter().position(|&p| p == i).unwrap();

            // Swap axes i and j in the result
            result = result.transpose_axis(i, j);

            // Update the permutation to reflect the swap
            perm.swap(i, j);
        }
    }

    Ok(result)
}