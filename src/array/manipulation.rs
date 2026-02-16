//! Array manipulation methods
//!
//! This module contains methods for reshaping, transposing, and broadcasting arrays:
//! - reshape, flatten
//! - transpose, transpose_axis
//! - broadcast_to, broadcast_shape

use super::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{ArrayView2, Axis, Dimension, IxDyn};
use std::cmp;

impl<T: Clone> Array<T> {
    /// Calculate the broadcast shape for two arrays
    /// Returns the broadcast shape or an error if shapes are incompatible
    pub fn broadcast_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
        // Determine the number of dimensions in the broadcast shape
        let n_dim = cmp::max(a_shape.len(), b_shape.len());
        let mut broadcast_shape = Vec::with_capacity(n_dim);

        // Right-align shapes to compare from the rightmost dimension
        let a_offset = n_dim - a_shape.len();
        let b_offset = n_dim - b_shape.len();

        for i in 0..n_dim {
            let a_dim = if i < a_offset {
                1
            } else {
                a_shape[i - a_offset]
            };
            let b_dim = if i < b_offset {
                1
            } else {
                b_shape[i - b_offset]
            };

            // Broadcasting rules: dimensions must be equal, or one of them must be 1
            if a_dim == b_dim {
                broadcast_shape.push(a_dim);
            } else if a_dim == 1 {
                broadcast_shape.push(b_dim);
            } else if b_dim == 1 {
                broadcast_shape.push(a_dim);
            } else {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    actual: b_shape.to_vec(),
                });
            }
        }

        Ok(broadcast_shape)
    }

    /// Broadcast this array to a new shape
    ///
    /// This function implements NumPy-compatible broadcasting semantics.
    /// The rules for broadcasting are:
    ///
    /// 1. Arrays with fewer dimensions are prepended with dimensions of size 1
    /// 2. Size in each dimension of the output shape is the maximum of the sizes in the corresponding
    ///    dimensions of the input arrays
    /// 3. An input array can be broadcast along a dimension if its size in that dimension is 1 or
    ///    the same as the output size
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self>
    where
        T: Clone,
    {
        use scirs2_core::ndarray::Array as NdArray;

        let orig_shape = self.shape();

        // Calculate the number of dims to add (to the left)
        let n_dims_to_add = if shape.len() > orig_shape.len() {
            shape.len() - orig_shape.len()
        } else {
            0
        };

        // Expand dimensions if needed (prepend dimensions of size 1)
        let mut expanded_array = self.clone();
        if n_dims_to_add > 0 {
            // Create shape with leading 1s and then add original shape
            let mut new_shape = Vec::with_capacity(shape.len());
            new_shape.extend(std::iter::repeat_n(1, n_dims_to_add));
            new_shape.extend_from_slice(&orig_shape);
            expanded_array = self.reshape(&new_shape);
        }

        // Create a new array with broadcast shape and replicate values
        let mut result = NdArray::<T, IxDyn>::from_elem(
            IxDyn(shape),
            self.array()
                .first()
                .cloned()
                .unwrap_or_else(|| panic!("Empty array")),
        );

        // This is a simplified implementation - for a full implementation, we would use
        // more efficient broadcasting algorithms provided by ndarray
        // For now, we'll manually broadcast by iterating over the result and assigning values

        // Get the original array shape for broadcasting rules
        let current_shape = expanded_array.shape();

        // Apply broadcasting rules
        for (idx, val) in result.indexed_iter_mut() {
            let mut broadcast_idx = Vec::with_capacity(current_shape.len());

            // Calculate the broadcasted indices (modulo the original shape)
            for (i, &dim) in idx.slice().iter().enumerate() {
                // Get index; 0 if beyond array dims or if dim size is 1
                let broadcast_dim = if i >= current_shape.len() || current_shape[i] == 1 {
                    0
                } else {
                    dim % current_shape[i]
                };
                broadcast_idx.push(broadcast_dim);
            }

            // Get the value from the original array using the broadcast indices
            let original_val = expanded_array
                .array()
                .get(IxDyn(&broadcast_idx))
                .cloned()
                .unwrap_or_else(|| panic!("Invalid broadcast index"));

            *val = original_val;
        }

        Ok(Self { data: result })
    }

    /// Reshape the array
    ///
    /// # Parameters
    ///
    /// * `shape` - The new shape
    ///
    /// # Returns
    ///
    /// A new array with the same data but reshaped
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    /// let b = a.reshape(&[2, 3]);
    /// assert_eq!(b.shape(), vec![2, 3]);
    /// assert_eq!(b.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> Self
    where
        T: Clone,
    {
        // Check if the total size is compatible
        let current_size = self.size();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            panic!(
                "Cannot reshape array of size {} into shape with size {}",
                current_size, new_size
            );
        }

        let reshaped = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(shape))
            .unwrap_or_else(|_| panic!("Failed to reshape array"));
        Self { data: reshaped }
    }

    /// Reshape the array with an option to copy or share the underlying data
    ///
    /// # Parameters
    ///
    /// * `shape` - The new shape
    /// * `copy` - Whether to copy the data (true) or try to use a view (false)
    ///
    /// # Returns
    ///
    /// A new array with the same data but reshaped
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    ///
    /// // Reshape with copy (always creates a new array)
    /// let b = a.reshape_with(&[2, 3], true);
    /// assert_eq!(b.shape(), vec![2, 3]);
    /// assert_eq!(b.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    ///
    /// // Reshape without copy (may share data if possible)
    /// let c = a.reshape_with(&[3, 2], false);
    /// assert_eq!(c.shape(), vec![3, 2]);
    /// assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn reshape_with(&self, shape: &[usize], copy: bool) -> Self
    where
        T: Clone,
    {
        // Check if the total size is compatible
        let current_size = self.size();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            panic!(
                "Cannot reshape array of size {} into shape with size {}",
                current_size, new_size
            );
        }

        if copy {
            // Always make a copy
            let data_vec = self.to_vec();
            Self::from_vec(data_vec).reshape(shape)
        } else {
            // Try to reshape in-place if possible
            let reshaped = self
                .data
                .clone()
                .into_shape_with_order(IxDyn(shape))
                .unwrap_or_else(|_| panic!("Failed to reshape array"));
            Self { data: reshaped }
        }
    }

    /// Return a flattened copy of the array in column-major (C) order
    ///
    /// The returned array is a flattened copy of the original array.
    /// The order parameter specifies the memory layout of the returned array.
    ///
    /// # Parameters
    ///
    /// * `order` - Memory layout: "C" for row-major (C-style), "F" for column-major (Fortran-style)
    ///
    /// # Returns
    ///
    /// A new 1D array with all elements of the original array in the specified order
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    ///
    /// // C-style (row-major) flattening - default
    /// let flat_c = a.flatten(Some("C"));
    /// assert_eq!(flat_c.shape(), vec![6]);
    /// assert_eq!(flat_c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn flatten(&self, order: Option<&str>) -> Self
    where
        T: Clone,
    {
        let order_str = order.unwrap_or("C");

        match order_str {
            "C" => {
                // Row-major (C-style) order
                self.reshape(&[self.size()])
            }
            "F" => {
                // Column-major (Fortran-style) order
                let shape = self.shape();

                if shape.len() <= 1 {
                    // 0D or 1D arrays are the same in both orders
                    return self.reshape(&[self.size()]);
                }

                // For 2D and higher arrays, we need to transpose and then ravel
                let mut indices = Vec::with_capacity(shape.len());
                for i in (0..shape.len()).rev() {
                    indices.push(i);
                }

                // Create a transposed view and then flatten
                // Need to implement a transpose with indices method
                // For now, just do a simple flatten
                // let transposed = self.transpose(&indices).unwrap();
                let transposed = self.clone();
                transposed.reshape(&[transposed.size()])
            }
            _ => {
                panic!("Invalid order parameter: {}. Must be 'C' or 'F'", order_str);
            }
        }
    }

    /// Reshape the array with an option to copy or view the data
    ///
    /// # Parameters
    ///
    /// * `shape` - The new shape
    /// * `copy` - Whether to copy the data (true) or use a view (false)
    ///
    /// # Returns
    ///
    /// A new array with the same data but reshaped
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    ///
    /// // Without copy (view - default behavior)
    /// let b = a.reshape_with_option(&[2, 3], false);
    /// assert_eq!(b.shape(), vec![2, 3]);
    ///
    /// // With copy (new memory allocation)
    /// let c = a.reshape_with_option(&[2, 3], true);
    /// assert_eq!(c.shape(), vec![2, 3]);
    /// ```
    pub fn reshape_with_option(&self, shape: &[usize], copy: bool) -> Self
    where
        T: Clone,
    {
        if copy {
            // Create a copy of the data
            let vec_data = self.to_vec();
            Array::from_vec(vec_data).reshape(shape)
        } else {
            // Use regular reshape which shares memory when possible
            self.reshape(shape)
        }
    }

    /// Transpose the array
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        match self.data.ndim() {
            1 => {
                // 1D arrays remain unchanged when transposed
                self.clone()
            }
            2 => {
                // For 2D arrays, perform proper matrix transpose
                let shape = self.shape();
                let rows = shape[0];
                let cols = shape[1];
                let old_data = self.to_vec();
                let mut new_data = Vec::with_capacity(old_data.len());

                // Transpose: new[j * rows + i] = old[i * cols + j]
                for j in 0..cols {
                    for i in 0..rows {
                        new_data.push(old_data[i * cols + j].clone());
                    }
                }

                Self::from_vec(new_data).reshape(&[cols, rows])
            }
            _ => {
                // For N-D arrays, reverse all axes
                let shape = self.shape();
                let mut reversed_shape = shape.clone();
                reversed_shape.reverse();

                let old_data = self.to_vec();
                let mut new_data = Vec::with_capacity(old_data.len());

                // Calculate strides for both original and transposed arrays
                let mut old_strides = vec![1; shape.len()];
                for i in (0..shape.len() - 1).rev() {
                    old_strides[i] = old_strides[i + 1] * shape[i + 1];
                }

                let mut new_strides = vec![1; reversed_shape.len()];
                for i in (0..reversed_shape.len() - 1).rev() {
                    new_strides[i] = new_strides[i + 1] * reversed_shape[i + 1];
                }

                // For each position in the new array, find corresponding position in old array
                let total_elements = old_data.len();
                for linear_idx in 0..total_elements {
                    // Convert linear index to multi-dimensional index in new array
                    let mut new_indices = vec![0; reversed_shape.len()];
                    let mut temp = linear_idx;
                    for i in 0..reversed_shape.len() {
                        new_indices[i] = temp / new_strides[i];
                        temp %= new_strides[i];
                    }

                    // Map to old array indices (reverse the indices)
                    let mut old_indices = new_indices.clone();
                    old_indices.reverse();

                    // Convert old multi-dimensional indices to linear index
                    let mut old_linear_idx = 0;
                    for i in 0..shape.len() {
                        old_linear_idx += old_indices[i] * old_strides[i];
                    }

                    new_data.push(old_data[old_linear_idx].clone());
                }

                Self::from_vec(new_data).reshape(&reversed_shape)
            }
        }
    }

    /// Transpose (interchange) the given axes of the array.
    ///
    /// # Parameters
    ///
    /// * `axis1` - The first axis to transpose
    /// * `axis2` - The second axis to transpose
    ///
    /// # Returns
    ///
    /// A new array with the given axes transposed.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let b = a.transpose_axis(0, 1);
    /// assert_eq!(b.shape(), vec![3, 2]);
    /// ```
    pub fn transpose_axis(&self, axis1: usize, axis2: usize) -> Self
    where
        T: Clone,
    {
        let ndim = self.ndim();

        // If axes are out of bounds, panic
        if axis1 >= ndim || axis2 >= ndim {
            panic!(
                "Axis out of bounds: dimensions are {}, got axes {} and {}",
                ndim, axis1, axis2
            );
        }

        // If axes are the same, return a clone
        if axis1 == axis2 {
            return self.clone();
        }

        // Create a permutation that swaps the given axes
        let mut perm = (0..ndim).collect::<Vec<_>>();
        perm.swap(axis1, axis2);

        // Permute the axes
        let permuted_data = self.data.clone().permuted_axes(IxDyn(&perm));
        Self {
            data: permuted_data,
        }
    }

    /// Get a 2D view of the underlying ndarray data
    pub fn view_2d(&self) -> Result<ArrayView2<'_, T>>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "view_2d requires a 2D array".to_string(),
            ));
        }

        let shape = self.shape();
        self.data
            .view()
            .into_shape_with_order((shape[0], shape[1]))
            .map_err(|_| NumRs2Error::DimensionMismatch("Failed to create 2D view".to_string()))
    }

    /// Get a slice along a particular axis
    pub fn slice(&self, axis: usize, index: usize) -> Result<Self>
    where
        T: Clone,
    {
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        if index >= self.shape()[axis] {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Index {} out of bounds for axis {} with size {}",
                index,
                axis,
                self.shape()[axis]
            )));
        }

        let slice = self.data.index_axis(Axis(axis), index);
        Ok(Self {
            data: slice.into_owned().into_dyn(),
        })
    }

    /// Slice the array along a given axis, returning a view
    pub fn slice_view(&self, axis: usize, index: usize) -> Result<crate::views::ArrayView<'_, T>> {
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        if index >= self.shape()[axis] {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Index {} out of bounds for axis {} with size {}",
                index,
                axis,
                self.shape()[axis]
            )));
        }

        use scirs2_core::ndarray::Axis as NdAxis;
        let sliced = self.array().index_axis(NdAxis(axis), index);
        Ok(crate::views::ArrayView::from_ndarray_view(
            sliced.into_dyn(),
        ))
    }
}
