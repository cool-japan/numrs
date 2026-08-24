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
    ///
    /// Returns `Err(NumRs2Error::ShapeMismatch)` if `self`'s shape cannot be
    /// broadcast to `shape` under the rules above (this never panics).
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self>
    where
        T: Clone,
    {
        // Delegate to ndarray's own (well-tested) broadcasting implementation
        // instead of a hand-rolled index computation: `ArrayBase::broadcast`
        // implements exactly the NumPy `broadcast_to` rules documented above
        // -- including right-aligning and prepending size-1 dimensions -- and
        // returns `None` for incompatible shapes instead of silently tiling
        // or reading out of bounds.
        match self.data.broadcast(IxDyn(shape)) {
            Some(view) => Ok(Self::from_nd(view.to_owned())),
            None => Err(NumRs2Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.shape(),
            }),
        }
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
    ///
    /// # Panics
    ///
    /// Panics if `shape`'s total element count does not match `self.size()`.
    /// Use [`Array::try_reshape`] for a non-panicking version that returns a
    /// [`crate::error::NumRs2Error`].
    pub fn reshape(&self, shape: &[usize]) -> Self
    where
        T: Clone,
    {
        self.try_reshape(shape).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Non-panicking version of [`Array::reshape`].
    ///
    /// Returns `Err` if `shape`'s total element count does not match
    /// `self.size()` instead of panicking. Never panics: if the in-place
    /// reshape ndarray offers is not possible for this array's memory
    /// layout (e.g. a non-contiguous view produced by
    /// [`Array::transpose_axis`]), this falls back to a logical-order copy
    /// rather than failing.
    pub fn try_reshape(&self, shape: &[usize]) -> Result<Self>
    where
        T: Clone,
    {
        // Check if the total size is compatible
        let current_size = self.size();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.shape(),
            });
        }

        // Deep-clones the ndarray (not the Arc): `into_shape_with_order`
        // consumes its receiver, and `try_reshape` borrows `self`, so a fresh
        // buffer is required here exactly as before the Arc-backed storage
        // landed. Callers that can give up ownership should use
        // `into_reshape`, which hands the existing buffer over instead.
        match self.array().clone().into_shape_with_order(IxDyn(shape)) {
            Ok(reshaped) => Ok(Self::from_nd(reshaped)),
            Err(_) => {
                // `into_shape_with_order` only succeeds without copying when
                // the array's memory layout is compatible with the new
                // shape; a non-contiguous array (e.g. after
                // `transpose_axis`) lands here instead. Fall back to a copy
                // built by iterating in *logical* order (`.iter()`, which
                // respects strides) rather than `to_vec()`, which returns
                // the raw backing buffer in *physical* memory order and
                // would silently scramble the data for such arrays.
                let logical_order: Vec<T> = self.array().iter().cloned().collect();
                let fresh = Self::from_vec(logical_order);
                // `fresh` is a local, sole owner of its buffer, so `into_nd`
                // hands the allocation over without copying.
                match fresh.into_nd().into_shape_with_order(IxDyn(shape)) {
                    Ok(reshaped) => Ok(Self::from_nd(reshaped)),
                    Err(e) => Err(NumRs2Error::InvalidOperation(format!(
                        "Failed to reshape array: {e}"
                    ))),
                }
            }
        }
    }

    /// Consuming reshape: reuses `self`'s own backing allocation when
    /// possible, instead of cloning it first.
    ///
    /// Unlike [`Array::try_reshape`] (which takes `&self`, and so must
    /// clone before it can attempt an in-place reshape), `into_reshape`
    /// consumes `self`, so its fast path can hand `self`'s own allocation
    /// straight to `ndarray`'s `into_shape_with_order` with no copy at
    /// all -- but only once this function has confirmed, itself, that the
    /// call will succeed: `ndarray::ArrayBase::into_shape_with_order`
    /// *consumes and drops* its receiver on failure, returning only a
    /// `ShapeError` with no way to recover the original array, so calling
    /// it speculatively (the way `try_reshape` calls it on a disposable
    /// clone) is not an option here. Instead this checks
    /// [`Array::is_c_contiguous`] first (a `&self` borrow, not a move):
    /// when `self` is standard-layout and the element counts already
    /// match (checked below), `into_shape_with_order`'s row-major branch
    /// is guaranteed to return `Ok`, so it is only ever called once that
    /// is already known.
    ///
    /// # Errors
    ///
    /// Returns `Err(NumRs2Error::ShapeMismatch)` if `shape`'s total
    /// element count does not equal `self.size()`.
    ///
    /// Never panics. For a non-contiguous `self` (e.g. a permuted-axes
    /// view produced by [`Array::transpose_axis`]) this falls back to a
    /// logical-order copy, the same fallback (and the same final result)
    /// [`Array::try_reshape`] uses.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    /// let b = a.into_reshape(&[2, 3]).expect("shapes agree");
    /// assert_eq!(b.shape(), vec![2, 3]);
    /// assert_eq!(b.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    ///
    /// let bad = Array::from_vec(vec![1, 2, 3]);
    /// assert!(bad.into_reshape(&[2, 2]).is_err());
    /// ```
    pub fn into_reshape(self, shape: &[usize]) -> Result<Self>
    where
        T: Clone,
    {
        let current_size = self.size();
        let new_size: usize = shape.iter().product();
        if current_size != new_size {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.shape(),
            });
        }

        if self.is_c_contiguous() {
            // Proven to succeed (see doc comment above): never falls into
            // the data-destroying `Err` arm in practice, but we still
            // handle it via `Result` -- consistent with `try_reshape` --
            // rather than `.expect(...)`, since a defensive match costs
            // nothing here.
            return match self.into_nd().into_shape_with_order(IxDyn(shape)) {
                Ok(reshaped) => Ok(Self::from_nd(reshaped)),
                Err(e) => Err(NumRs2Error::InvalidOperation(format!(
                    "Failed to reshape array: {e}"
                ))),
            };
        }

        // Non-contiguous layout: do NOT attempt `into_shape_with_order`
        // here -- on failure it would consume and drop `self.data` with
        // no way to recover it. Fall back to a logical-order copy built
        // from a *borrow* of `self` (not a move), matching
        // `try_reshape`'s fallback and producing the identical result.
        let logical_order: Vec<T> = self.array().iter().cloned().collect();
        Self::from_vec(logical_order).try_reshape(shape)
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
    ///
    /// # Panics
    ///
    /// Panics if `shape`'s total element count does not match `self.size()`.
    /// Use [`Array::try_reshape_with`] for a non-panicking version that
    /// returns a [`crate::error::NumRs2Error`].
    pub fn reshape_with(&self, shape: &[usize], copy: bool) -> Self
    where
        T: Clone,
    {
        self.try_reshape_with(shape, copy)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Non-panicking version of [`Array::reshape_with`].
    ///
    /// Returns `Err` if `shape`'s total element count does not match
    /// `self.size()` instead of panicking.
    pub fn try_reshape_with(&self, shape: &[usize], copy: bool) -> Result<Self>
    where
        T: Clone,
    {
        // Check if the total size is compatible
        let current_size = self.size();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.shape(),
            });
        }

        if copy {
            // Always make a copy. Iterate in logical order (respecting
            // strides) rather than using `to_vec()`, which returns the raw
            // backing buffer in physical memory order and would silently
            // scramble data for a non-contiguous array.
            let logical_order: Vec<T> = self.array().iter().cloned().collect();
            Self::from_vec(logical_order).try_reshape(shape)
        } else {
            // Try to reshape in-place if possible; `try_reshape` already
            // falls back to a logical-order copy for layouts that cannot be
            // reshaped in place, so this never panics.
            self.try_reshape(shape)
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
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Some` value other than `"C"` or `"F"`. Use
    /// [`Array::try_flatten`] for a non-panicking version that returns a
    /// [`crate::error::NumRs2Error`].
    pub fn flatten(&self, order: Option<&str>) -> Self
    where
        T: Clone,
    {
        self.try_flatten(order).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Non-panicking version of [`Array::flatten`].
    ///
    /// Returns `Err` if `order` is a `Some` value other than `"C"` or `"F"`
    /// instead of panicking.
    pub fn try_flatten(&self, order: Option<&str>) -> Result<Self>
    where
        T: Clone,
    {
        let order_str = order.unwrap_or("C");

        match order_str {
            "C" => {
                // Row-major (C-style) order
                self.try_reshape(&[self.size()])
            }
            "F" => {
                // Column-major (Fortran-style) order
                let shape = self.shape();

                if shape.len() <= 1 {
                    // 0D or 1D arrays are the same in both orders
                    return self.try_reshape(&[self.size()]);
                }

                // For 2D and higher arrays, we need to transpose and then ravel
                let mut indices = Vec::with_capacity(shape.len());
                for i in (0..shape.len()).rev() {
                    indices.push(i);
                }

                // Create a transposed view and then flatten
                // Need to implement a transpose with indices method
                // For now, just do a simple flatten
                // let transposed = self.transpose(&indices).expect("valid transpose indices");
                let transposed = self.clone();
                transposed.try_reshape(&[transposed.size()])
            }
            _ => Err(NumRs2Error::InvalidInput(format!(
                "Invalid order parameter: {}. Must be 'C' or 'F'",
                order_str
            ))),
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
            Array::from_vec_shape(vec_data, shape).unwrap_or_else(|e| panic!("{e}"))
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

                Self::from_vec_shape(new_data, &[cols, rows]).unwrap_or_else(|e| panic!("{e}"))
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

                Self::from_vec_shape(new_data, &reversed_shape).unwrap_or_else(|e| panic!("{e}"))
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
    ///
    /// # Panics
    ///
    /// Panics if `axis1` or `axis2` is out of bounds for `self`'s number of
    /// dimensions. Use [`Array::try_transpose_axis`] for a non-panicking
    /// version that returns a [`crate::error::NumRs2Error`].
    pub fn transpose_axis(&self, axis1: usize, axis2: usize) -> Self
    where
        T: Clone,
    {
        self.try_transpose_axis(axis1, axis2)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Non-panicking version of [`Array::transpose_axis`].
    ///
    /// Returns `Err` if `axis1` or `axis2` is out of bounds for `self`'s
    /// number of dimensions instead of panicking.
    pub fn try_transpose_axis(&self, axis1: usize, axis2: usize) -> Result<Self>
    where
        T: Clone,
    {
        let ndim = self.ndim();

        if axis1 >= ndim || axis2 >= ndim {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Axis out of bounds: dimensions are {}, got axes {} and {}",
                ndim, axis1, axis2
            )));
        }

        // If axes are the same, return a clone
        if axis1 == axis2 {
            return Ok(self.clone());
        }

        // Create a permutation that swaps the given axes
        let mut perm = (0..ndim).collect::<Vec<_>>();
        perm.swap(axis1, axis2);

        // Permute the axes
        // Deep-clones the ndarray (not the Arc): `permuted_axes` consumes its
        // receiver and `swap_axes` only borrows `self`.
        let permuted_data = self.array().clone().permuted_axes(IxDyn(&perm));
        Ok(Self::from_nd(permuted_data))
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
        Ok(Self::from_nd(slice.into_owned().into_dyn()))
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // into_reshape
    // -----------------------------------------------------------------

    #[test]
    fn into_reshape_contiguous_fast_path_matches_try_reshape() {
        let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let expected = a.try_reshape(&[2, 3]).expect("shapes agree");
        let got = a.into_reshape(&[2, 3]).expect("shapes agree");
        assert_eq!(got.shape(), expected.shape());
        assert_eq!(got.to_vec(), expected.to_vec());
        assert_eq!(got.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn into_reshape_non_contiguous_falls_back_and_matches_try_reshape() {
        let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let t = a.transpose_axis(0, 1); // shape [3, 2], non-contiguous
        assert!(!t.is_c_contiguous());

        let expected = t.try_reshape(&[6]).expect("shapes agree");
        let got = t.clone().into_reshape(&[6]).expect("shapes agree");
        assert_eq!(got.to_vec(), expected.to_vec());
    }

    #[test]
    fn into_reshape_errs_on_size_mismatch_instead_of_panicking() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let result = a.into_reshape(&[2, 2]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NumRs2Error::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn into_reshape_scalar_and_empty() {
        let empty: Array<f64> = Array::from_vec(vec![]);
        let reshaped = empty.into_reshape(&[0, 5]).expect("0*5 == 0");
        assert_eq!(reshaped.shape(), vec![0, 5]);
    }

    // -----------------------------------------------------------------
    // broadcast_to regressions
    //
    // `broadcast_to` itself is unchanged in this lane (it already
    // delegates to `ndarray`'s own broadcasting and already returns
    // `Result` without panicking, as of the prior commit rewriting it --
    // see git history). What follows is the NumPy-standard coverage this
    // lane is responsible for adding, plus a bench-evidence test.
    //
    // Deliberately NOT restored: a "tiling" fallback that would silently
    // reinterpret an incompatible shape like `[2] -> [4]` as repetition
    // (`[1,2] -> [1,2,1,2]`) via `dim % current_shape[i]`. NumPy's own
    // `np.broadcast_to` raises `ValueError` for exactly that case -- it is
    // not a documented tiling extension, it was a bug, and
    // `tests/test_try_variants.rs::broadcast_to_errs_on_incompatible_shape_instead_of_silently_tiling`
    // already pins the fixed (error) behavior as a regression test. The
    // crate's actual tiling/repetition entry point is the separate,
    // correctly-named `array_ops::tiling::tile` function.
    // -----------------------------------------------------------------

    #[test]
    fn broadcast_to_row_vector_to_taller_2d() {
        // NumPy reference: np.broadcast_to(np.array([[1,2,3]]), (4,3))
        // -> 4 copies of the row [1,2,3].
        let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
        let result = a
            .broadcast_to(&[4, 3])
            .expect("broadcasting [1,3] to [4,3] is valid NumPy broadcasting");
        assert_eq!(result.shape(), vec![4, 3]);
        assert_eq!(result.to_vec(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn broadcast_to_1d_to_nonsquare_2d() {
        // NumPy reference: np.broadcast_to(np.array([1,2,3]), (5,3))
        // -> 5 copies of the row [1,2,3]. (Existing coverage in
        // tests/test_try_variants.rs only exercises the SQUARE [3]->[3,3]
        // case; this pins the non-square case too.)
        let a = Array::from_vec(vec![1, 2, 3]);
        let result = a
            .broadcast_to(&[5, 3])
            .expect("broadcasting [3] to [5,3] is valid NumPy broadcasting");
        assert_eq!(result.shape(), vec![5, 3]);
        assert_eq!(result.size(), 15);
        for row in 0..5 {
            assert_eq!(&result.to_vec()[row * 3..row * 3 + 3], &[1, 2, 3]);
        }
    }

    #[test]
    fn broadcast_to_column_vector_to_2d() {
        // NumPy reference: np.broadcast_to(np.array([[1],[2],[3]]), (3,4))
        // -> each row filled with a single repeated value.
        let a = Array::from_vec(vec![1, 2, 3]).reshape(&[3, 1]);
        let result = a
            .broadcast_to(&[3, 4])
            .expect("broadcasting [3,1] to [3,4] is valid NumPy broadcasting");
        assert_eq!(result.shape(), vec![3, 4]);
        assert_eq!(result.to_vec(), vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn broadcast_to_scalar_to_any_shape() {
        // NumPy reference: np.broadcast_to(np.array([7]), (2, 3, 2))
        let a = Array::from_vec(vec![7]);
        let result = a
            .broadcast_to(&[2, 3, 2])
            .expect("broadcasting [1] to [2,3,2] is valid NumPy broadcasting");
        assert_eq!(result.shape(), vec![2, 3, 2]);
        assert!(result.to_vec().iter().all(|&x| x == 7));
    }

    #[test]
    fn broadcast_to_errs_on_a_second_incompatible_shape() {
        // NumPy reference: np.broadcast_to(np.arange(6).reshape(2,3), (3,3))
        // raises ValueError: dim 0 is 2, target is 3 -- neither equal nor 1.
        // (Distinct incompatible-shape case from the one already pinned in
        // tests/test_try_variants.rs, which covers 1D [2]->[4].)
        let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let result = a.broadcast_to(&[3, 3]);
        assert!(
            result.is_err(),
            "broadcasting [2,3] to [3,3] is not valid NumPy broadcasting and must error, got {:?}",
            result
        );
    }

    /// Pre-`fc464bf` reimplementation of `broadcast_to`'s body, kept only
    /// to produce a before/after performance comparison against the
    /// current `ndarray`-delegated implementation. Not used anywhere
    /// outside this test -- see the git history of `broadcast_to` for the
    /// real prior implementation this mirrors.
    fn old_modulo_broadcast_to_f64(src: &Array<f64>, shape: &[usize]) -> Array<f64> {
        use scirs2_core::ndarray::Array as NdArray;

        let orig_shape = src.shape();
        let n_dims_to_add = shape.len().saturating_sub(orig_shape.len());
        let mut expanded = src.clone();
        if n_dims_to_add > 0 {
            let mut new_shape = Vec::with_capacity(shape.len());
            new_shape.extend(std::iter::repeat_n(1, n_dims_to_add));
            new_shape.extend_from_slice(&orig_shape);
            expanded = src.reshape(&new_shape);
        }

        let mut result = NdArray::<f64, IxDyn>::from_elem(
            IxDyn(shape),
            expanded.array().first().cloned().unwrap_or(0.0),
        );
        let current_shape = expanded.shape();
        for (idx, val) in result.indexed_iter_mut() {
            let mut broadcast_idx = Vec::with_capacity(current_shape.len());
            for (i, &dim) in idx.slice().iter().enumerate() {
                let broadcast_dim = if i >= current_shape.len() || current_shape[i] == 1 {
                    0
                } else {
                    dim % current_shape[i]
                };
                broadcast_idx.push(broadcast_dim);
            }
            *val = expanded
                .array()
                .get(IxDyn(&broadcast_idx))
                .cloned()
                .unwrap_or(0.0);
        }
        Array::from_ndarray(result)
    }

    #[test]
    fn broadcast_to_1x256_to_256x256_perf_evidence() {
        let src = Array::from_vec((0..256).map(|i| i as f64).collect()).reshape(&[1, 256]);
        // 20 iterations is plenty to see a ~55x gap this large; the old
        // (deleted) implementation is slow enough that 200 iterations
        // made this the single slowest test in the gate (~740ms).
        let iters = 20;

        // Sanity first: both implementations must agree on content before
        // comparing their speed.
        let new_result = src.broadcast_to(&[256, 256]).expect("broadcasts");
        let old_result = old_modulo_broadcast_to_f64(&src, &[256, 256]);
        assert_eq!(new_result.to_vec(), old_result.to_vec());

        let t_new = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(src.broadcast_to(&[256, 256]).expect("broadcasts"));
        }
        let new_elapsed = t_new.elapsed();

        let t_old = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(old_modulo_broadcast_to_f64(&src, &[256, 256]));
        }
        let old_elapsed = t_old.elapsed();

        eprintln!(
            "broadcast_to [1,256]->[256,256], {iters} iters: \
             current(ndarray broadcast+to_owned)={:?} ({:.1} ns/iter); \
             old(modulo, Vec-per-element)={:?} ({:.1} ns/iter)",
            new_elapsed,
            new_elapsed.as_nanos() as f64 / iters as f64,
            old_elapsed,
            old_elapsed.as_nanos() as f64 / iters as f64,
        );
    }
}
