//! Array methods for indexing operations
//!
//! This module provides array methods for:
//! - `set_mask()` - Set values using a boolean mask
//! - `diag()` - Get diagonal view
//! - `diagonal()` - Extract diagonal with offset
//! - `take()` - Take elements along an axis
//! - `choose()` - Choose elements from choices
//! - `compress()` - Select elements using a boolean mask

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::indexing::IndexSpec;

impl<T: Clone + num_traits::Zero> Array<T> {
    /// Set values in the array using a boolean mask
    pub fn set_mask(&mut self, mask: &Array<bool>, values: &Array<T>) -> Result<()>
    where
        T: Clone,
    {
        // Check that the mask shape is compatible with the array shape
        if self.shape() != mask.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: mask.shape(),
            });
        }

        // Count the number of True values in the mask
        let true_count = mask.array().iter().filter(|&&x| x).count();

        // Check that the values array has the right number of elements
        if values.size() != true_count {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![true_count],
                actual: vec![values.size()],
            });
        }

        // Set the values
        let mut value_idx = 0;
        let values_vec = values.to_vec();

        for (idx, &masked) in mask.array().iter().enumerate() {
            if masked {
                // Get the corresponding multi-dimensional index
                let mut multi_idx = Vec::with_capacity(self.ndim());
                let mut remaining = idx;

                for dim_size in self.shape().iter().rev() {
                    multi_idx.insert(0, remaining % dim_size);
                    remaining /= dim_size;
                }

                // Set the value
                if let Some(elem) = self.array_mut().get_mut(multi_idx.as_slice()) {
                    *elem = values_vec[value_idx].clone();
                }

                value_idx += 1;
            }
        }

        Ok(())
    }

    /// Get a diagonal view of a 2D array
    pub fn diag(&self) -> Result<Self>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "diag requires a 2D array, got {}D",
                self.ndim()
            )));
        }

        let shape = self.shape();
        let min_dim = shape[0].min(shape[1]);

        let mut result = Vec::with_capacity(min_dim);

        for i in 0..min_dim {
            result.push(self.get(&[i, i])?);
        }

        Ok(Self::from_vec(result))
    }

    /// Extract the diagonal from an array
    ///
    /// # Parameters
    ///
    /// * `offset` - Offset of the diagonal from the main diagonal (0 = main diagonal, +ve = above, -ve = below)
    /// * `axis1` - First axis of the 2D subarray to extract diagonal from (default 0)
    /// * `axis2` - Second axis of the 2D subarray to extract diagonal from (default 1)
    ///
    /// # Returns
    ///
    /// Array containing the diagonal elements
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a 3x3 array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    ///
    /// // Extract the main diagonal
    /// let diag = a.diagonal(0, None, None).expect("diagonal should succeed");
    /// assert_eq!(diag.to_vec(), vec![1, 5, 9]);
    ///
    /// // Extract the diagonal above the main diagonal
    /// let diag_above = a.diagonal(1, None, None).expect("diagonal above should succeed");
    /// assert_eq!(diag_above.to_vec(), vec![2, 6]);
    ///
    /// // Extract the diagonal below the main diagonal
    /// let diag_below = a.diagonal(-1, None, None).expect("diagonal below should succeed");
    /// assert_eq!(diag_below.to_vec(), vec![4, 8]);
    /// ```
    pub fn diagonal(
        &self,
        offset: isize,
        axis1: Option<usize>,
        axis2: Option<usize>,
    ) -> Result<Self>
    where
        T: Clone,
    {
        // Default axes
        let ax1 = axis1.unwrap_or(0);
        let ax2 = axis2.unwrap_or(1);

        let ndim = self.ndim();

        if ax1 >= ndim || ax2 >= ndim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axes ({}, {}) out of bounds for array of dimension {}",
                ax1, ax2, ndim
            )));
        }

        if ax1 == ax2 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Axes ({}, {}) cannot be identical",
                ax1, ax2
            )));
        }

        // Get the dimensions of the 2D subarray defined by the axes
        let shape = self.shape();
        let dim1 = shape[ax1];
        let dim2 = shape[ax2];

        // Calculate the min and max valid indices based on offset
        let max_diag_len = if offset >= 0 {
            ((dim2 - offset as usize).min(dim1)) as isize
        } else {
            ((dim1 - (-offset) as usize).min(dim2)) as isize
        };

        if max_diag_len <= 0 {
            // Empty diagonal
            return Ok(Self::from_vec(vec![]));
        }

        // Create the result array
        let mut result = Vec::with_capacity(max_diag_len as usize);

        // Extract the diagonal elements
        for i in 0..max_diag_len {
            let mut idx = vec![0; ndim];

            if offset >= 0 {
                idx[ax1] = i as usize;
                idx[ax2] = (i + offset) as usize;
            } else {
                idx[ax1] = (i - offset) as usize;
                idx[ax2] = i as usize;
            }

            result.push(self.get(&idx)?);
        }

        Ok(Self::from_vec(result))
    }

    /// Take elements from an array along an axis
    ///
    /// # Parameters
    ///
    /// * `indices` - The indices of the values to extract
    /// * `axis` - The axis over which to select values (default is None which flattens the array)
    ///
    /// # Returns
    ///
    /// An array of values at the given indices along the given axis
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a simple array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    ///
    /// // Take elements at indices 0, 2, 4
    /// let result = a.take(&Array::from_vec(vec![0usize, 2, 4]), None).expect("take should succeed");
    /// assert_eq!(result.to_vec(), vec![1, 3, 5]);
    ///
    /// // With a 2D array
    /// let b = a.reshape(&[5, 2]);
    ///
    /// // Take rows at indices 0, 2, 4
    /// let rows = b.take(&Array::from_vec(vec![0usize, 2, 4]), Some(0)).expect("take rows should succeed");
    /// assert_eq!(rows.shape(), vec![3, 2]);
    ///
    /// // Take columns at index 1
    /// let cols = b.take(&Array::from_vec(vec![1usize]), Some(1)).expect("take cols should succeed");
    /// assert_eq!(cols.shape(), vec![5, 1]);
    /// ```
    pub fn take(&self, indices: &Array<usize>, axis: Option<usize>) -> Result<Self>
    where
        T: Clone + ToString,
    {
        let indices_slice = indices.array().as_slice().ok_or_else(|| {
            NumRs2Error::InvalidOperation("indices array should be contiguous".to_string())
        })?;

        // Validate indices
        let check_size = match axis {
            None => self.size(),
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Axis {} is out of bounds for array with {} dimensions",
                        ax,
                        self.ndim()
                    )));
                }
                self.shape()[ax]
            }
        };

        for &idx in indices_slice {
            if idx >= check_size {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} is out of bounds for axis with size {}",
                    idx, check_size
                )));
            }
        }

        match axis {
            None => {
                // Flatten the array and take elements directly
                let flat_data = self.to_vec();
                let mut result = Vec::with_capacity(indices.size());

                for &idx in indices_slice {
                    result.push(flat_data[idx].clone());
                }

                Ok(Self::from_vec(result))
            }
            Some(ax) => {
                let shape = self.shape();
                let axis_dim = shape[ax];

                // Create the output shape
                let mut out_shape = shape.clone();
                out_shape[ax] = indices.size();

                let mut result_data = Vec::with_capacity(self.size() / axis_dim * indices.size());

                for &idx in indices_slice {
                    // Create index specs to select along the axis
                    let mut index_specs = vec![IndexSpec::All; self.ndim()];
                    index_specs[ax] = IndexSpec::Index(idx);

                    // Get the slice at this index
                    let slice = self.index(&index_specs)?;

                    // Append the slice's data to the result
                    result_data.extend(slice.to_vec());
                }

                // Reshape to the correct output shape
                Ok(Self::from_vec_shape(result_data, &out_shape)?)
            }
        }
    }

    /// Choose elements from a list of choices based on an index array
    ///
    /// Parses `self`'s elements as signed indices -- supporting NumPy's
    /// negative-index `wrap`/`clip` semantics, which a `usize`-typed index
    /// array cannot represent -- and normalizes them locally, then
    /// delegates the actual gather (and the broadcasting between the now
    /// non-negative index array and `choices`) to the canonical
    /// [`crate::array_ops::conditional::choose`]; see that function's docs
    /// for full NumPy-compatible semantics. This also removes a latent
    /// panic this method previously had: it read `choices[idx][i]` via a
    /// raw slice index for every `i` in `0..self.size()` with no check
    /// that any choice array was actually that large, so a `self` bigger
    /// than the (mutually-consistent-shaped) `choices` indexed out of
    /// bounds instead of erroring; the canonical's `.get()`-based,
    /// broadcast-checked access can't do that.
    ///
    /// # Parameters
    ///
    /// * `choices` - List of arrays to choose from
    /// * `mode` - Specifies how indices outside bounds are handled ('raise', 'wrap', 'clip')
    ///
    /// # Returns
    ///
    /// An array of elements chosen from the choices at the specified indices
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create an index array
    /// let a = Array::from_vec(vec![0, 1, 2, 1, 0]);
    ///
    /// // Create choice arrays
    /// let c1 = Array::from_vec(vec![10, 20, 30, 40, 50]);
    /// let c2 = Array::from_vec(vec![100, 200, 300, 400, 500]);
    /// let c3 = Array::from_vec(vec![1000, 2000, 3000, 4000, 5000]);
    ///
    /// // Select elements based on the indices
    /// let result = a.choose(&[&c1, &c2, &c3], None).expect("choose should succeed");
    /// assert_eq!(result.to_vec(), vec![10, 200, 3000, 400, 50]);
    /// ```
    pub fn choose(&self, choices: &[&Self], mode: Option<&str>) -> Result<Self>
    where
        T: Clone + ToString,
    {
        if choices.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "No choices provided".to_string(),
            ));
        }

        let n_choices = choices.len();
        let handle_mode = mode.unwrap_or("raise");
        if !matches!(handle_mode, "raise" | "wrap" | "clip") {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Invalid mode: {}. Must be one of 'raise', 'wrap', or 'clip'",
                handle_mode
            )));
        }

        let self_vec = self.to_vec();
        let mut idx_data = Vec::with_capacity(self_vec.len());
        for val in &self_vec {
            let mut idx = val.to_string().parse::<isize>().map_err(|_| {
                NumRs2Error::InvalidOperation("Index values must be integers".to_string())
            })?;

            match handle_mode {
                "raise" => {
                    if idx < 0 || idx >= n_choices as isize {
                        return Err(NumRs2Error::IndexOutOfBounds(format!(
                            "Index {} is out of bounds for choices with size {}",
                            idx, n_choices
                        )));
                    }
                }
                "wrap" => {
                    idx = ((idx % n_choices as isize) + n_choices as isize) % n_choices as isize;
                }
                "clip" => {
                    idx = idx.clamp(0, n_choices as isize - 1);
                }
                _ => unreachable!("mode validated above"),
            }

            idx_data.push(idx as usize);
        }

        // Every element of `idx_data` is now a valid, non-negative,
        // in-bounds `usize` index (validated/normalized above per `mode`),
        // so the canonical only needs its `"raise"` path from here.
        let idx_array = Array::from_vec_shape(idx_data, &self.shape())?;
        crate::array_ops::conditional::choose(&idx_array, choices, "raise")
    }

    /// Select elements from an array using a boolean mask
    ///
    /// Adapts `condition` (generic over any stringify-able `U`, validated to
    /// contain only `"true"`/`"false"` values) into an `Array<bool>` and
    /// delegates to the canonical
    /// [`crate::array_ops::advanced_indexing::compress`]; see that
    /// function's docs for full NumPy-compatible semantics, including its
    /// more permissive handling of a `condition` shorter or longer than the
    /// selected axis/array length (this method previously required an exact
    /// length match, which is stricter than NumPy's `np.compress`).
    ///
    /// # Parameters
    ///
    /// * `condition` - Boolean-valued mask array of the same shape as self
    /// * `axis` - Axis along which to select elements (None for flattened array)
    ///
    /// # Returns
    ///
    /// Array of selected elements
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a simple array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    ///
    /// // Create a condition (select even numbers)
    /// let condition = a.map(|x| x % 2 == 0);
    ///
    /// // Compress the array using the condition
    /// let result = a.compress(&condition, None).expect("compress should succeed");
    /// assert_eq!(result.to_vec(), vec![2, 4, 6, 8, 10]);
    ///
    /// // With a 2D array
    /// let b = a.reshape(&[5, 2]);
    /// let cond_axis = Array::from_vec(vec![true, false, true, false, true]);
    ///
    /// // Select rows with the condition
    /// let compressed = b.compress(&cond_axis, Some(0)).expect("compress along axis should succeed");
    /// assert_eq!(compressed.shape(), vec![3, 2]);
    /// ```
    pub fn compress<U>(&self, condition: &Array<U>, axis: Option<usize>) -> Result<Self>
    where
        U: Clone + ToString,
    {
        // Check if condition contains boolean values, and convert to
        // `Array<bool>` for the canonical implementation. `to_vec()`
        // (rather than requiring `condition` be contiguous) matches the
        // canonical's own tolerance for any memory layout.
        let cond_vec = condition.to_vec();
        let mut bool_vec = Vec::with_capacity(cond_vec.len());
        for val in &cond_vec {
            let val_str = val.to_string();
            bool_vec.push(match val_str.as_str() {
                "true" => true,
                "false" => false,
                _ => {
                    return Err(NumRs2Error::InvalidOperation(
                        "Condition must contain boolean values".to_string(),
                    ))
                }
            });
        }
        // `condition`'s own shape (not just its flat size) only matters
        // in that the canonical requires it 1-D; reshaping a stringified
        // multi-dimensional condition down to 1-D here would silently
        // paper over a caller bug the canonical is meant to reject.
        let bool_condition = Array::<bool>::from_vec_shape(bool_vec, &condition.shape())?;

        crate::array_ops::advanced_indexing::compress(self, &bool_condition, axis)
    }
}
