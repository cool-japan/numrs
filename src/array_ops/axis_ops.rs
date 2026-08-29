use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use scirs2_core::ndarray::IxDyn;

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
/// let b = rollaxis(&a, 2, 0).expect("operation should succeed");
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

    // Iterate through all elements of the array. Force standard
    // (C-contiguous) layout first so `as_slice()` below is guaranteed to
    // succeed even if `array` is itself a permuted/strided view (e.g. the
    // result of a prior `moveaxis`/`swapaxes` call).
    let source_size = array.size();
    let standard = array.to_c_layout();
    let source_slice = standard.array().as_slice().ok_or_else(|| {
        NumRs2Error::InvalidOperation(
            "Failed to obtain a contiguous slice of the input array".into(),
        )
    })?;

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
        result_data[target_flat_index] = source_slice[i].clone();
    }

    // Create the result array
    let result = Array::from_vec_shape(result_data, &target_shape)?;
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
/// let b = swapaxes(&a, 0, 1).expect("operation should succeed");
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

    // Swapping two axes is a single transposition, so one call to
    // `transpose_axis` realizes it directly. The result is forced into
    // standard (C-contiguous) layout so that `to_vec()`/`as_slice()` observe
    // the data in the new logical (row-major) order rather than the raw
    // memory order of the underlying permuted view.
    Ok(array.transpose_axis(axis1, axis2).to_c_layout())
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
/// A new array with the given axes moved to their destination positions.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // NumPy: np.moveaxis(np.arange(24).reshape(2, 3, 4), [0], [2]).shape == (3, 4, 2)
/// let a = Array::from_vec((0..24).collect::<Vec<i32>>()).reshape(&[2, 3, 4]);
/// let b = moveaxis(&a, &[0], &[2]).expect("operation should succeed");
/// assert_eq!(b.shape(), vec![3, 4, 2]);
/// assert_eq!(&b.to_vec()[0..4], &[0, 12, 1, 13]);
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

    // Build the gather permutation using NumPy's `moveaxis` algorithm: start
    // with the axes that are *not* being moved (in their relative order),
    // then insert each moved axis at its destination position, processing
    // the (destination, source) pairs in ascending destination order.
    // `order[i]` names the source axis that ends up at output position `i`,
    // i.e. `result.shape()[i] == array.shape()[order[i]]`.
    let mut order: Vec<usize> = (0..ndim).filter(|axis| !source.contains(axis)).collect();

    let mut pairs: Vec<(usize, usize)> = destination
        .iter()
        .copied()
        .zip(source.iter().copied())
        .collect();
    pairs.sort_by_key(|&(dest, _)| dest);

    for (dest, src) in pairs {
        let pos = dest.min(order.len());
        order.insert(pos, src);
    }

    // Transpose directly by `order`, then force standard (C-contiguous)
    // layout so `to_vec()`/`as_slice()` observe the data in the array's new
    // logical (row-major) order rather than the permuted view's raw memory
    // order.
    let permuted = array.array().clone().permuted_axes(IxDyn(&order));
    Ok(Array::from_ndarray(permuted).to_c_layout())
}
