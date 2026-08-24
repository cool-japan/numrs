//! Differentiation and integration functions
//!
//! This module provides numerical differentiation and integration operations:
//! - `diff_extended` - Discrete difference with prepend/append support
//! - `diff` - Calculate discrete difference along an axis
//! - `ediff1d` - Differences between consecutive elements of a flattened array
//! - `trapz` - Trapezoidal integration along an axis

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, Zero};
use scirs2_core::ndarray::Array1;
use scirs2_core::simd_ops::SimdUnifiedOps;

// Import helpers from elementwise module
use super::elementwise::{from_nd_array, to_nd_array_f32, to_nd_array_f64};

/// Calculate the discrete difference along the given axis with prepend/append support
///
/// # Parameters
///
/// * `array` - Input array
/// * `n` - The number of times values are differenced
/// * `axis` - The axis along which the difference is taken. If None, uses the last axis.
/// * `prepend` - Optional values to prepend to the array along the axis before computing differences
/// * `append` - Optional values to append to the array along the axis before computing differences
///
/// # Returns
///
/// The n-th differences with optional prepend/append values included
pub fn diff_extended<T>(
    array: &Array<T>,
    n: usize,
    axis: Option<usize>,
    prepend: Option<&Array<T>>,
    append: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: Clone + Zero + std::ops::Sub<Output = T> + std::ops::Add<Output = T> + 'static,
{
    // If prepend or append is provided, concatenate them first
    let working_array = if prepend.is_some() || append.is_some() {
        let axis_val = axis.unwrap_or(array.ndim().saturating_sub(1));
        let mut arrays_to_concat: Vec<&Array<T>> = Vec::new();

        // We need to own the values to pass references
        if let Some(p) = prepend {
            arrays_to_concat.push(p);
        }
        arrays_to_concat.push(array);
        if let Some(a) = append {
            arrays_to_concat.push(a);
        }

        // Concatenate along the axis
        crate::array_ops::concatenate(&arrays_to_concat, axis_val)?
    } else {
        array.clone()
    };

    // Now compute the diff on the working array
    diff(&working_array, n, axis)
}

/// Calculate the discrete difference along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `n` - The number of times values are differenced. Default is 1.
/// * `axis` - The axis along which the difference is taken. Default is the last axis.
///
/// # Returns
///
/// The n-th differences. The shape of the output is the same as input except along axis where the dimension is smaller by n.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 0.0]);
/// let d = diff(&a, 1, None).expect("diff should succeed");
/// assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0, -7.0]);
/// ```
pub fn diff<T>(array: &Array<T>, n: usize, axis: Option<usize>) -> Result<Array<T>>
where
    T: Clone + Zero + std::ops::Sub<Output = T> + 'static,
{
    if n == 0 {
        return Ok(array.clone());
    }

    let axis = axis.unwrap_or(array.ndim().saturating_sub(1));

    if axis >= array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    let axis_size = array.shape()[axis];
    if axis_size <= n {
        // Create empty array with appropriate shape
        let mut new_shape = array.shape();
        new_shape[axis] = 0;
        return Ok(Array::zeros(&new_shape));
    }

    // Use SIMD for 1D f64/f32 arrays with n=1 and sufficient size
    if array.ndim() == 1 && n == 1 && array.len() >= 64 {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let nd =
                to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(array) });
            let result = f64::simd_diff(&nd.view());
            return Ok(unsafe {
                std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                    result,
                    &[array.len() - 1],
                ))
            });
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let nd =
                to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(array) });
            let result = f32::simd_diff(&nd.view());
            return Ok(unsafe {
                std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                    result,
                    &[array.len() - 1],
                ))
            });
        }
    }

    let mut result = array.clone();

    for _ in 0..n {
        let axis_size = result.shape()[axis];
        if axis_size <= 1 {
            let mut new_shape = result.shape();
            new_shape[axis] = 0;
            return Ok(Array::zeros(&new_shape));
        }

        // Create new shape with axis dimension reduced by 1
        let mut new_shape = result.shape();
        new_shape[axis] -= 1;

        let mut new_data = Vec::with_capacity(new_shape.iter().product());

        // Calculate strides
        let mut strides = vec![1; result.ndim()];
        for i in (0..result.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * result.shape()[i + 1];
        }

        // Iterate through all indices in the new array
        let total_size: usize = new_shape.iter().product();
        for i in 0..total_size {
            // Convert flat index to multi-dimensional indices
            let mut indices = vec![0; new_shape.len()];
            let mut temp = i;
            for j in 0..new_shape.len() {
                indices[j] = temp / strides[j];
                temp %= strides[j];
            }

            // Get values at current position and next position along axis
            let mut indices_next = indices.clone();
            indices_next[axis] += 1;

            let val1 = result.get(&indices)?;
            let val2 = result.get(&indices_next)?;

            new_data.push(val2 - val1);
        }

        result = Array::from_vec_shape(new_data, &new_shape)?;
    }

    Ok(result)
}

/// The differences between consecutive elements of an array
///
/// # Parameters
///
/// * `array` - Input array
/// * `to_end` - Optional values to append at the end of the returned differences
/// * `to_begin` - Optional values to prepend at the beginning of the returned differences
///
/// # Returns
///
/// The differences. Loosely, this is ``a.flatten()[1:] - a.flatten()[:-1]``.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 4, 7, 0]);
/// let d = ediff1d(&a, None, None).expect("ediff1d should succeed");
/// assert_eq!(d.to_vec(), vec![1, 2, 3, -7]);
/// ```
pub fn ediff1d<T>(
    array: &Array<T>,
    to_end: Option<&Array<T>>,
    to_begin: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    let flat = array.to_vec();

    if flat.len() <= 1 {
        // Create result with just to_begin and to_end
        let mut result = Vec::new();
        if let Some(begin) = to_begin {
            result.extend(begin.to_vec());
        }
        if let Some(end) = to_end {
            result.extend(end.to_vec());
        }
        return Ok(Array::from_vec(result));
    }

    let mut result = Vec::new();

    // Add to_begin values
    if let Some(begin) = to_begin {
        result.extend(begin.to_vec());
    }

    // Calculate differences
    for i in 1..flat.len() {
        result.push(flat[i].clone() - flat[i - 1].clone());
    }

    // Add to_end values
    if let Some(end) = to_end {
        result.extend(end.to_vec());
    }

    Ok(Array::from_vec(result))
}

/// Integrate along the given axis using the composite trapezoidal rule
///
/// # Parameters
///
/// * `y` - Input array to integrate
/// * `x` - Optional array of sample points corresponding to the y values
/// * `dx` - Spacing between sample points when x is None. Default is 1.
/// * `axis` - The axis along which to integrate. Default is the last axis.
///
/// # Returns
///
/// Definite integral as approximated by trapezoidal rule
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let integral = trapz(&y, None, 1.0, None).expect("trapz should succeed");
/// assert_eq!(integral.to_vec()[0], 4.0); // (1+2)/2 + (2+3)/2 = 1.5 + 2.5 = 4.0
/// ```
pub fn trapz<T>(y: &Array<T>, x: Option<&Array<T>>, dx: T, axis: Option<usize>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let axis = axis.unwrap_or(y.ndim().saturating_sub(1));

    if axis >= y.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            y.ndim()
        )));
    }

    let axis_size = y.shape()[axis];
    if axis_size < 2 {
        // Create result array with axis dimension removed
        let mut new_shape = y.shape();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        return Ok(Array::zeros(&new_shape));
    }

    // If x is provided, check compatibility
    if let Some(x_arr) = x {
        if x_arr.ndim() != 1 || x_arr.size() != axis_size {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![axis_size],
                actual: x_arr.shape(),
            });
        }
    }

    // Create output shape (remove the integration axis)
    let mut out_shape = y.shape();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_size: usize = out_shape.iter().product();
    let mut result_data = vec![T::zero(); out_size];

    // Calculate strides for input array
    let mut in_strides = vec![1; y.ndim()];
    for i in (0..y.ndim() - 1).rev() {
        in_strides[i] = in_strides[i + 1] * y.shape()[i + 1];
    }

    // Calculate strides for output array
    let mut out_strides = vec![1; out_shape.len()];
    for i in (0..out_shape.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Iterate through output positions
    for out_idx in 0..out_size {
        // Convert flat output index to multi-dimensional indices
        let mut out_indices = vec![0; out_shape.len()];
        let mut temp = out_idx;
        for i in 0..out_shape.len() {
            out_indices[i] = temp / out_strides[i];
            temp %= out_strides[i];
        }

        // Build input indices by inserting axis dimension
        let mut sum = T::zero();

        for i in 0..axis_size - 1 {
            // Build indices for current and next position
            let mut indices_curr = Vec::with_capacity(y.ndim());
            let mut indices_next = Vec::with_capacity(y.ndim());

            let mut out_idx_ptr = 0;
            for j in 0..y.ndim() {
                if j == axis {
                    indices_curr.push(i);
                    indices_next.push(i + 1);
                } else {
                    indices_curr.push(out_indices[out_idx_ptr]);
                    indices_next.push(out_indices[out_idx_ptr]);
                    out_idx_ptr += 1;
                }
            }

            let y_curr = y.get(&indices_curr)?;
            let y_next = y.get(&indices_next)?;

            let width = if let Some(x_arr) = x {
                let x_vec = x_arr.to_vec();
                x_vec[i + 1] - x_vec[i]
            } else {
                dx
            };

            sum = sum
                + (y_curr + y_next) * width / T::from(2.0).expect("2.0 should be representable");
        }

        result_data[out_idx] = sum;
    }

    Array::from_vec_shape(result_data, &out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 0.0]);
        let d = diff(&a, 1, None).expect("diff should succeed");
        assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0, -7.0]);
    }

    #[test]
    fn test_diff_n2() {
        let a = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 0.0]);
        let d = diff(&a, 2, None).expect("diff should succeed");
        // First diff: [1.0, 2.0, 3.0, -7.0]
        // Second diff: [1.0, 1.0, -10.0]
        assert_eq!(d.to_vec(), vec![1.0, 1.0, -10.0]);
    }

    #[test]
    fn test_ediff1d() {
        let a = Array::from_vec(vec![1, 2, 4, 7, 0]);
        let d = ediff1d(&a, None, None).expect("ediff1d should succeed");
        assert_eq!(d.to_vec(), vec![1, 2, 3, -7]);
    }

    #[test]
    fn test_ediff1d_with_prepend_append() {
        let a = Array::from_vec(vec![1, 2, 4]);
        let begin = Array::from_vec(vec![0]);
        let end = Array::from_vec(vec![99]);
        let d = ediff1d(&a, Some(&end), Some(&begin)).expect("ediff1d should succeed");
        // begin: [0], diff: [1, 2], end: [99]
        assert_eq!(d.to_vec(), vec![0, 1, 2, 99]);
    }

    #[test]
    fn test_trapz_basic() {
        let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let integral = trapz(&y, None, 1.0, None).expect("trapz should succeed");
        // (1+2)/2 + (2+3)/2 = 1.5 + 2.5 = 4.0
        assert!((integral.to_vec()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapz_with_dx() {
        let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let integral = trapz(&y, None, 0.5, None).expect("trapz should succeed");
        // (1+2)/2 * 0.5 + (2+3)/2 * 0.5 = 0.75 + 1.25 = 2.0
        assert!((integral.to_vec()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapz_with_x() {
        let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let x = Array::from_vec(vec![0.0, 1.0, 3.0]); // Non-uniform spacing
        let integral = trapz(&y, Some(&x), 1.0, None).expect("trapz should succeed");
        // (1+2)/2 * 1 + (2+3)/2 * 2 = 1.5 + 5.0 = 6.5
        assert!((integral.to_vec()[0] - 6.5).abs() < 1e-10);
    }
}
