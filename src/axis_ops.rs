use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul};

/// Trait for axis-based operations on arrays
pub trait AxisOps<T> {
    /// Sum along specified axis
    fn sum_axis(&self, axis: Option<usize>) -> Result<Array<T>>;

    /// Mean along specified axis
    fn mean_axis(&self, axis: Option<usize>) -> Result<Array<T>>;

    /// Minimum along specified axis
    fn min_axis(&self, axis: Option<usize>) -> Result<Array<T>>;

    /// Maximum along specified axis
    fn max_axis(&self, axis: Option<usize>) -> Result<Array<T>>;

    /// Product along specified axis
    fn prod_axis(&self, axis: Option<usize>) -> Result<Array<T>>
    where
        T: Mul<Output = T> + One;

    /// Cumulative sum along specified axis
    fn cumsum_axis(&self, axis: usize) -> Result<Array<T>>;

    /// Cumulative product along specified axis
    fn cumprod_axis(&self, axis: usize) -> Result<Array<T>>
    where
        T: Mul<Output = T> + One;

    /// Argmin along specified axis
    fn argmin_axis(&self, axis: usize) -> Result<Array<usize>>;

    /// Argmax along specified axis
    fn argmax_axis(&self, axis: usize) -> Result<Array<usize>>;

    /// Variance along specified axis
    fn var_axis(&self, axis: Option<usize>) -> Result<Array<T>>
    where
        T: Float;

    /// Standard deviation along specified axis
    fn std_axis(&self, axis: Option<usize>) -> Result<Array<T>>
    where
        T: Float;
}

impl<T> AxisOps<T> for Array<T>
where
    T: Clone + PartialOrd + Zero + Add<Output = T> + Div<Output = T> + NumCast + std::fmt::Debug,
{
    /// Sum along specified axis
    fn sum_axis(&self, axis: Option<usize>) -> Result<Array<T>> {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                // Create a new array with the same shape as the input but with the axis removed
                let mut output_shape = self.shape();
                output_shape.remove(ax);
                let mut result = Array::zeros(&output_shape);

                // For each element in the output, sum along the axis
                let axis_len = self.shape()[ax];

                // Iterate over the output array
                for i in 0..result.size() {
                    // Calculate multi-dimensional index for output
                    let mut out_idx = Vec::with_capacity(result.ndim());
                    let mut tmp = i;

                    for dim in output_shape.iter().rev() {
                        out_idx.insert(0, tmp % dim);
                        tmp /= dim;
                    }

                    // Insert the axis dimension and iterate over it
                    let mut sum = T::zero();

                    for j in 0..axis_len {
                        let mut in_idx = out_idx.clone();
                        in_idx.insert(ax, j);

                        // Get the value from the input array
                        let val = self.get(&in_idx)?;
                        sum = sum + val;
                    }

                    // Set the result
                    result.set(&out_idx, sum)?;
                }

                Ok(result)
            }
            None => {
                // Sum all elements
                let sum = self.array().fold(T::zero(), |acc, x| acc + x.clone());
                Ok(Array::from_vec(vec![sum]))
            }
        }
    }

    /// Mean along specified axis
    fn mean_axis(&self, axis: Option<usize>) -> Result<Array<T>> {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                let axis_size = self.shape()[ax];
                if axis_size == 0 {
                    return Err(NumRs2Error::InvalidOperation(
                        "Cannot compute mean of empty array".to_string(),
                    ));
                }

                let sum_result = self.sum_axis(ax)?;
                let divisor = T::from(axis_size).ok_or_else(|| {
                    NumRs2Error::ConversionError(
                        "Failed to convert axis size to array type".to_string(),
                    )
                })?;

                let result = sum_result.map(|x| x / divisor.clone());
                Ok(result)
            }
            None => {
                // Mean of all elements
                let total_size = self.size();
                if total_size == 0 {
                    return Err(NumRs2Error::InvalidOperation(
                        "Cannot compute mean of empty array".to_string(),
                    ));
                }

                let sum = self.array().fold(T::zero(), |acc, x| acc + x.clone());
                let divisor = T::from(total_size).ok_or_else(|| {
                    NumRs2Error::ConversionError(
                        "Failed to convert array size to array type".to_string(),
                    )
                })?;

                Ok(Array::from_vec(vec![sum / divisor]))
            }
        }
    }

    /// Minimum along specified axis
    fn min_axis(&self, axis: Option<usize>) -> Result<Array<T>> {
        if self.size() == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot compute minimum of empty array".to_string(),
            ));
        }

        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                // Create a new array with the same shape as the input but with the axis removed
                let mut output_shape = self.shape();
                output_shape.remove(ax);
                let mut result = Array::zeros(&output_shape);

                // For each element in the output, find the minimum along the axis
                let axis_len = self.shape()[ax];

                // Iterate over the output array
                for i in 0..result.size() {
                    // Calculate multi-dimensional index for output
                    let mut out_idx = Vec::with_capacity(result.ndim());
                    let mut tmp = i;

                    for dim in output_shape.iter().rev() {
                        out_idx.insert(0, tmp % dim);
                        tmp /= dim;
                    }

                    // Insert the axis dimension and iterate over it
                    let mut in_idx = out_idx.clone();
                    in_idx.insert(ax, 0);

                    // Initialize with first value along this axis
                    let mut min_val = self.get(&in_idx)?;

                    // Iterate along the axis to find the minimum
                    for j in 1..axis_len {
                        // Update the index along the axis
                        in_idx[ax] = j;

                        // Get the value from the input array
                        let val = self.get(&in_idx)?;

                        if val < min_val {
                            min_val = val;
                        }
                    }

                    // Set the result
                    result.set(&out_idx, min_val)?;
                }

                Ok(result)
            }
            None => {
                // Find minimum of all elements
                let first =
                    self.array().first().cloned().expect(
                        "min_axis called on empty array: this should have been caught earlier",
                    );

                let min = self.array().fold(first, |acc, x| {
                    let val = x.clone();
                    if val < acc {
                        val
                    } else {
                        acc
                    }
                });

                Ok(Array::from_vec(vec![min]))
            }
        }
    }

    /// Maximum along specified axis
    fn max_axis(&self, axis: Option<usize>) -> Result<Array<T>> {
        if self.size() == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot compute maximum of empty array".to_string(),
            ));
        }

        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                // Create a new array with the same shape as the input but with the axis removed
                let mut output_shape = self.shape();
                output_shape.remove(ax);
                let mut result = Array::zeros(&output_shape);

                // For each element in the output, find the maximum along the axis
                let axis_len = self.shape()[ax];

                // Iterate over the output array
                for i in 0..result.size() {
                    // Calculate multi-dimensional index for output
                    let mut out_idx = Vec::with_capacity(result.ndim());
                    let mut tmp = i;

                    for dim in output_shape.iter().rev() {
                        out_idx.insert(0, tmp % dim);
                        tmp /= dim;
                    }

                    // Insert the axis dimension and iterate over it
                    let mut in_idx = out_idx.clone();
                    in_idx.insert(ax, 0);

                    // Initialize with first value along this axis
                    let mut max_val = self.get(&in_idx)?;

                    // Iterate along the axis to find the maximum
                    for j in 1..axis_len {
                        // Update the index along the axis
                        in_idx[ax] = j;

                        // Get the value from the input array
                        let val = self.get(&in_idx)?;

                        if val > max_val {
                            max_val = val;
                        }
                    }

                    // Set the result
                    result.set(&out_idx, max_val)?;
                }

                Ok(result)
            }
            None => {
                // Find maximum of all elements
                let first =
                    self.array().first().cloned().expect(
                        "max_axis called on empty array: this should have been caught earlier",
                    );

                let max = self.array().fold(first, |acc, x| {
                    let val = x.clone();
                    if val > acc {
                        val
                    } else {
                        acc
                    }
                });

                Ok(Array::from_vec(vec![max]))
            }
        }
    }

    /// Product along specified axis
    fn prod_axis(&self, axis: Option<usize>) -> Result<Array<T>>
    where
        T: Mul<Output = T> + One,
    {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                // Create a new array with the same shape as the input but with the axis removed
                let mut output_shape = self.shape();
                output_shape.remove(ax);
                let mut result = Array::zeros(&output_shape);

                // For each element in the output, compute product along the axis
                let axis_len = self.shape()[ax];

                // Iterate over the output array
                for i in 0..result.size() {
                    // Calculate multi-dimensional index for output
                    let mut out_idx = Vec::with_capacity(result.ndim());
                    let mut tmp = i;

                    for dim in output_shape.iter().rev() {
                        out_idx.insert(0, tmp % dim);
                        tmp /= dim;
                    }

                    // Insert the axis dimension and iterate over it
                    let mut prod = T::one();

                    for j in 0..axis_len {
                        let mut in_idx = out_idx.clone();
                        in_idx.insert(ax, j);

                        // Get the value from the input array
                        let val = self.get(&in_idx)?;
                        prod = prod * val;
                    }

                    // Set the result
                    result.set(&out_idx, prod)?;
                }

                Ok(result)
            }
            None => {
                // Product of all elements
                let prod = self.array().fold(T::one(), |acc, x| acc * x.clone());
                Ok(Array::from_vec(vec![prod]))
            }
        }
    }

    /// Cumulative sum along specified axis
    fn cumsum_axis(&self, axis: usize) -> Result<Array<T>> {
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        // Get the shape
        let shape = self.shape();
        let axis_len = shape[axis];

        // Calculate the stride for the axis (number of elements between
        // consecutive positions along `axis`, in logical row-major order;
        // this is 1 when `axis` is the last dimension since an empty
        // product is 1).
        let stride = shape[axis + 1..].iter().product::<usize>();

        // Calculate the number of independent sequences to process
        let n_sequences = shape[..axis].iter().product::<usize>();

        // Snapshot the elements in *logical* row-major order via `.iter()`,
        // which respects strides regardless of memory layout. This both
        // avoids requiring a contiguous backing slice (the array may be a
        // non-contiguous view, e.g. after `transpose_axis`) and avoids
        // repeatedly re-reading the whole array on every inner-loop step.
        let mut data: Vec<T> = self.array().iter().cloned().collect();

        for seq in 0..n_sequences {
            let base_idx = seq * stride * axis_len;

            // Each of the `stride` positions within a sequence block is an
            // independent run to accumulate along `axis`.
            for elem in 0..stride {
                let mut sum = T::zero();
                for i in 0..axis_len {
                    let idx = base_idx + i * stride + elem;
                    sum = sum + data[idx].clone();
                    data[idx] = sum.clone();
                }
            }
        }

        Ok(Array::from_vec(data).reshape(&shape))
    }

    /// Cumulative product along specified axis
    fn cumprod_axis(&self, axis: usize) -> Result<Array<T>>
    where
        T: Mul<Output = T> + One,
    {
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        // Get the shape
        let shape = self.shape();
        let axis_len = shape[axis];

        // Calculate the stride for the axis (see `cumsum_axis` for details).
        let stride = shape[axis + 1..].iter().product::<usize>();

        // Calculate the number of independent sequences to process
        let n_sequences = shape[..axis].iter().product::<usize>();

        // Snapshot the elements in logical row-major order (see
        // `cumsum_axis` for why `.iter()` is used instead of a contiguous
        // slice or `to_vec()`).
        let mut data: Vec<T> = self.array().iter().cloned().collect();

        for seq in 0..n_sequences {
            let base_idx = seq * stride * axis_len;

            // Each of the `stride` positions within a sequence block is an
            // independent run to accumulate along `axis`.
            for elem in 0..stride {
                let mut prod = T::one();
                for i in 0..axis_len {
                    let idx = base_idx + i * stride + elem;
                    prod = prod * data[idx].clone();
                    data[idx] = prod.clone();
                }
            }
        }

        Ok(Array::from_vec(data).reshape(&shape))
    }

    /// Argmin along specified axis
    fn argmin_axis(&self, axis: usize) -> Result<Array<usize>> {
        if self.size() == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot compute argmin of empty array".to_string(),
            ));
        }

        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        // Create output shape - remove the specified axis
        let mut output_shape = self.shape();
        let axis_len = output_shape.remove(axis);

        // Calculate the stride for the axis (also equal to the number of
        // output elements produced per sequence, since `output_shape` is
        // `self.shape()` with `axis` removed: the dimensions after `axis`
        // become both the axis's stride and the per-sequence output count).
        let stride = if axis < self.ndim() - 1 {
            self.shape()[axis + 1..].iter().product::<usize>()
        } else {
            1
        };
        let elements_per_sequence = stride;

        // Calculate the number of independent sequences to process
        let n_sequences = self.shape()[..axis].iter().product::<usize>();

        // Create result array. `output_shape.iter().product()` is already
        // `n_sequences * elements_per_sequence` (the full output size, since
        // `output_shape` = `self.shape()` with `axis` removed = the
        // concatenation of the before-`axis` and after-`axis` dimensions);
        // multiplying by `n_sequences` again here would over-allocate and
        // make the final `.reshape(&output_shape)` fail whenever
        // `n_sequences > 1` (i.e. whenever `axis` is not the first
        // dimension).
        let mut result_data = vec![0; output_shape.iter().product::<usize>()];

        // Snapshot the elements in logical row-major order via `.iter()`,
        // which respects strides regardless of memory layout -- this avoids
        // requiring a contiguous backing slice (the array may be a
        // non-contiguous view, e.g. after `transpose_axis`).
        let slice: Vec<T> = self.array().iter().cloned().collect();

        // For each sequence, compute the argmin
        for seq in 0..n_sequences {
            // Calculate the base index for this sequence
            let base_idx = seq * stride * axis_len;

            // For each element in the output, find the argmin
            for elem in 0..elements_per_sequence {
                // Initialize with first element
                let mut min_val = slice[base_idx + elem].clone();
                let mut min_idx = 0;

                // Find the minimum value and its index
                for i in 1..axis_len {
                    let idx = base_idx + i * stride + elem;
                    let val = slice[idx].clone();

                    if val < min_val {
                        min_val = val;
                        min_idx = i;
                    }
                }

                // Store the argmin
                let result_idx = seq * elements_per_sequence + elem;
                result_data[result_idx] = min_idx;
            }
        }

        Ok(Array::from_vec(result_data).reshape(&output_shape))
    }

    /// Argmax along specified axis
    fn argmax_axis(&self, axis: usize) -> Result<Array<usize>> {
        if self.size() == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot compute argmax of empty array".to_string(),
            ));
        }

        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        // Create output shape - remove the specified axis
        let mut output_shape = self.shape();
        let axis_len = output_shape.remove(axis);

        // Calculate the stride for the axis (also equal to the number of
        // output elements produced per sequence, since `output_shape` is
        // `self.shape()` with `axis` removed: the dimensions after `axis`
        // become both the axis's stride and the per-sequence output count).
        let stride = if axis < self.ndim() - 1 {
            self.shape()[axis + 1..].iter().product::<usize>()
        } else {
            1
        };
        let elements_per_sequence = stride;

        // Calculate the number of independent sequences to process
        let n_sequences = self.shape()[..axis].iter().product::<usize>();

        // Create result array. `output_shape.iter().product()` is already
        // `n_sequences * elements_per_sequence` (the full output size, since
        // `output_shape` = `self.shape()` with `axis` removed = the
        // concatenation of the before-`axis` and after-`axis` dimensions);
        // multiplying by `n_sequences` again here would over-allocate and
        // make the final `.reshape(&output_shape)` fail whenever
        // `n_sequences > 1` (i.e. whenever `axis` is not the first
        // dimension).
        let mut result_data = vec![0; output_shape.iter().product::<usize>()];

        // Snapshot the elements in logical row-major order via `.iter()`,
        // which respects strides regardless of memory layout -- this avoids
        // requiring a contiguous backing slice (the array may be a
        // non-contiguous view, e.g. after `transpose_axis`).
        let slice: Vec<T> = self.array().iter().cloned().collect();

        // For each sequence, compute the argmax
        for seq in 0..n_sequences {
            // Calculate the base index for this sequence
            let base_idx = seq * stride * axis_len;

            // For each element in the output, find the argmax
            for elem in 0..elements_per_sequence {
                // Initialize with first element
                let mut max_val = slice[base_idx + elem].clone();
                let mut max_idx = 0;

                // Find the maximum value and its index
                for i in 1..axis_len {
                    let idx = base_idx + i * stride + elem;
                    let val = slice[idx].clone();

                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }

                // Store the argmax
                let result_idx = seq * elements_per_sequence + elem;
                result_data[result_idx] = max_idx;
            }
        }

        Ok(Array::from_vec(result_data).reshape(&output_shape))
    }

    /// Variance along specified axis
    fn var_axis(&self, axis: Option<usize>) -> Result<Array<T>>
    where
        T: Clone + Float,
    {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        ax,
                        self.ndim()
                    )));
                }

                // Calculate mean along the axis
                let mean = self.mean_axis(Some(ax))?;

                // Calculate squared differences from the mean
                // For each element, calculate the squared difference from the mean
                // For a proper implementation, we would directly index into the arrays
                // using multidimensional indices. For simplicity, we'll use a less efficient approach.

                let self_data = self.to_vec();
                let mean_data = mean.to_vec();
                let mut squared_diffs_data: Vec<T> = Vec::with_capacity(self.size());

                // Calculate the shape of the mean array
                let mut mean_shape = self.shape();
                mean_shape.remove(ax);

                // For each element in the array, find the corresponding mean
                for (i, _) in self_data.iter().enumerate() {
                    // Calculate multi-dimensional index
                    let mut idx = Vec::with_capacity(self.ndim());
                    let mut tmp = i;

                    for dim in self.shape().iter().rev() {
                        idx.insert(0, tmp % dim);
                        tmp /= dim;
                    }

                    // Remove the axis dimension from the index to get the mean index
                    let mut mean_idx = idx.clone();
                    mean_idx.remove(ax);

                    // Calculate linearized mean index
                    let mut mean_i = 0;
                    let mut stride = 1;

                    for (j, &idx_j) in mean_idx.iter().enumerate().rev() {
                        mean_i += idx_j * stride;
                        if j > 0 {
                            stride *= mean_shape[j];
                        }
                    }

                    // Calculate squared difference
                    let diff = self_data[i] - mean_data[mean_i];
                    squared_diffs_data.push(diff * diff);
                }

                let squared_diffs = Array::from_vec(squared_diffs_data).reshape(&self.shape());

                // Calculate mean of squared differences
                squared_diffs.mean_axis(Some(ax))
            }
            None => {
                // Variance of all elements
                let mean = self.mean_axis(None)?;
                let mean_val = mean
                    .array()
                    .first()
                    .cloned()
                    .expect("var_axis: mean calculation should return at least one element");

                // Calculate sum of squared differences
                let squared_diff_sum = self.array().fold(T::zero(), |acc, x| {
                    let val = *x;
                    let diff = val - mean_val;
                    acc + diff * diff
                });

                // Divide by number of elements
                let divisor = T::from(self.size()).ok_or_else(|| {
                    NumRs2Error::ConversionError(
                        "Failed to convert array size to array type".to_string(),
                    )
                })?;

                Ok(Array::from_vec(vec![squared_diff_sum / divisor]))
            }
        }
    }

    /// Standard deviation along specified axis
    fn std_axis(&self, axis: Option<usize>) -> Result<Array<T>>
    where
        T: Clone + Float,
    {
        // Calculate variance
        let var = self.var_axis(axis)?;

        // Take the square root
        Ok(var.map(|x| x.sqrt()))
    }
}

/// Apply a function to 1-D slices along the given axis.
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Axis along which to apply the function
/// * `func` - Function that operates on 1-D arrays and returns a single value
///
/// # Returns
///
/// A new array with the result of applying func along the specified axis
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 2D array
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
///
/// // Apply sum function along axis 0
/// let result = apply_along_axis(&arr, 0, |slice| {
///     slice.to_vec().iter().sum::<f64>()
/// }).expect("apply_along_axis should succeed");
///
/// assert_eq!(result.shape(), vec![3]);
/// assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
/// ```
pub fn apply_along_axis<T, U, F>(array: &Array<T>, axis: usize, func: F) -> Result<Array<U>>
where
    T: Clone + Debug + Zero,
    U: Clone + Debug,
    F: Fn(&Array<T>) -> U,
{
    if axis >= array.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    // Create output shape (remove the specified axis)
    let mut output_shape = array.shape();
    let axis_len = output_shape.remove(axis);

    // Create a buffer to store results
    let output_size = output_shape.iter().product::<usize>();
    let mut results = Vec::with_capacity(output_size);

    // For each slice along the axis, apply the function
    for i in 0..output_size {
        // Calculate multi-dimensional index for output
        let mut out_idx = Vec::with_capacity(output_shape.len());
        let mut tmp = i;

        for dim in output_shape.iter().rev() {
            out_idx.insert(0, tmp % dim);
            tmp /= dim;
        }

        // Extract the 1-D slice along the axis
        let mut slice_data = Vec::with_capacity(axis_len);

        for j in 0..axis_len {
            // Insert the axis dimension into the index
            let mut in_idx = out_idx.clone();
            in_idx.insert(axis, j);

            // Get the value from the input array
            let val = array.get(&in_idx)?;
            slice_data.push(val);
        }

        // Apply the function to the slice
        let slice_array = Array::from_vec(slice_data);
        let result = func(&slice_array);

        // Store the result
        results.push(result);
    }

    // Create the output array
    if output_shape.is_empty() {
        // If the output is a scalar, create a 1-element array
        Ok(Array::from_vec(results))
    } else {
        // Reshape to the output shape
        Ok(Array::from_vec(results).reshape(&output_shape))
    }
}

/// Apply a function repeatedly over multiple axes.
///
/// # Arguments
///
/// * `array` - Input array
/// * `axes` - Sequence of axes over which to apply the function
/// * `func` - Function that operates on arrays and returns arrays with decreased dimension
///
/// # Returns
///
/// A new array with the result of applying func over the specified axes
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3D array
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
///     .reshape(&[2, 2, 2]);
///
/// // Apply sum function over axes 0 and 1
/// let result = apply_over_axes(&arr, &[0, 1], |a, ax| {
///     a.sum_axis(ax)
/// }).expect("apply_over_axes should succeed");
///
/// assert_eq!(result.shape(), vec![1, 1, 2]);
/// assert_eq!(result.to_vec(), vec![16.0, 20.0]);
/// ```
pub fn apply_over_axes<T, F>(array: &Array<T>, axes: &[usize], func: F) -> Result<Array<T>>
where
    T: Clone + Debug,
    F: Fn(&Array<T>, usize) -> Result<Array<T>>,
{
    // Validate axes are in bounds
    for &axis in axes {
        if axis >= array.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                array.ndim()
            )));
        }
    }

    // Start with the input array
    let mut result = array.clone();

    // Apply the function over each axis
    for (i, &axis) in axes.iter().enumerate() {
        // Apply the function to the current result
        result = func(&result, axis)?;

        // Adjust indices for axes after the current one
        // Since the function should be reducing dimensions, we need to account for that
        let shape = result.shape();
        let expected_shape = {
            let mut s = array.shape();
            for (j, &ax) in axes.iter().enumerate() {
                if j <= i {
                    s[ax] = 1;
                }
            }
            s
        };

        // Reshape the result to preserve dimensions
        // This ensures that subsequent axis operations work correctly
        if shape != expected_shape {
            result = result.reshape(&expected_shape);
        }
    }

    Ok(result)
}

/// Create a vectorized function that broadcasts across arrays
///
/// # Arguments
///
/// * `func` - Function that operates on scalar elements
///
/// # Returns
///
/// A new function that operates on arrays by applying func element-wise
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a function that squares a number
/// let square = |x: f64| x * x;
///
/// // Vectorize the function
/// let vec_square = vectorize(square);
///
/// // Apply to an array
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let result = vec_square(&arr);
///
/// assert_eq!(result.to_vec(), vec![1.0, 4.0, 9.0, 16.0]);
/// ```
pub fn vectorize<T, U, F>(func: F) -> impl Fn(&Array<T>) -> Array<U>
where
    T: Clone + Debug,
    U: Clone + Debug,
    F: Fn(T) -> U + Clone,
{
    move |array: &Array<T>| -> Array<U> {
        let data = array.to_vec();
        let func_clone = func.clone();
        let results: Vec<U> = data.into_iter().map(func_clone).collect();
        Array::from_vec(results).reshape(&array.shape())
    }
}
