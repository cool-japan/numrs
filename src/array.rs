use crate::error::{NumRs2Error, Result};
use ndarray::{Array as NdArray, ArrayView, ArrayView2, Axis, Dimension, IxDyn};
use num_traits::{One, Zero};
use rayon::prelude::*;
use std::cmp;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// A multi-dimensional array type that wraps ndarray
#[derive(Clone)]
pub struct Array<T> {
    data: NdArray<T, IxDyn>,
}

impl<T: Clone> Array<T> {
    /// Create a new array from an ndarray
    pub fn from_ndarray(array: NdArray<T, IxDyn>) -> Self {
        Self { data: array }
    }

    /// Create a new array with the same shape as another array, filled with zeros
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let zeros: Array<i32> = Array::zeros_like(&a);
    /// assert_eq!(zeros.shape(), vec![2, 2]);
    /// assert_eq!(zeros.to_vec(), vec![0, 0, 0, 0]);
    /// ```
    pub fn zeros_like<U>(other: &Array<U>) -> Self
    where
        T: Zero + Clone,
        U: Clone,
    {
        Self::zeros(&other.shape())
    }

    /// Create a new array with the specified shape, data type, and order, filled with zeros
    ///
    /// # Parameters
    ///
    /// * `other` - The array whose shape to copy
    /// * `shape` - Optional shape for the new array. If None, the shape is the same as `other`
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    ///
    /// // Same shape as a
    /// let zeros = Array::<f64>::zeros_like_with(&a, None);
    /// assert_eq!(zeros.shape(), vec![2, 2]);
    /// assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
    ///
    /// // Different shape
    /// let zeros_3d = Array::<i32>::zeros_like_with(&a, Some(&[2, 2, 2]));
    /// assert_eq!(zeros_3d.shape(), vec![2, 2, 2]);
    /// assert_eq!(zeros_3d.size(), 8);
    /// ```
    pub fn zeros_like_with<U>(other: &Array<U>, shape: Option<&[usize]>) -> Self
    where
        T: Zero + Clone,
        U: Clone,
    {
        match shape {
            Some(s) => Self::zeros(s),
            None => Self::zeros(&other.shape()),
        }
    }

    /// Create a new array with the same shape as another array, filled with ones
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let ones: Array<i32> = Array::ones_like(&a);
    /// assert_eq!(ones.shape(), vec![2, 2]);
    /// assert_eq!(ones.to_vec(), vec![1, 1, 1, 1]);
    /// ```
    pub fn ones_like<U>(other: &Array<U>) -> Self
    where
        T: One + Clone,
        U: Clone,
    {
        Self::ones(&other.shape())
    }

    /// Create a new array with the specified shape, data type, and order, filled with ones
    ///
    /// # Parameters
    ///
    /// * `other` - The array whose shape to copy
    /// * `shape` - Optional shape for the new array. If None, the shape is the same as `other`
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    ///
    /// // Same shape as a
    /// let ones = Array::<f64>::ones_like_with(&a, None);
    /// assert_eq!(ones.shape(), vec![2, 2]);
    /// assert_eq!(ones.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// // Different shape
    /// let ones_3d = Array::<i32>::ones_like_with(&a, Some(&[2, 2, 2]));
    /// assert_eq!(ones_3d.shape(), vec![2, 2, 2]);
    /// assert_eq!(ones_3d.size(), 8);
    /// ```
    pub fn ones_like_with<U>(other: &Array<U>, shape: Option<&[usize]>) -> Self
    where
        T: One + Clone,
        U: Clone,
    {
        match shape {
            Some(s) => Self::ones(s),
            None => Self::ones(&other.shape()),
        }
    }

    /// Create a new array with the same shape as another array with uninitialized values
    /// Note: This is similar to NumPy's empty_like but with safe Rust semantics
    /// The array will be initialized with a default value instead of random memory
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let empty = Array::<i32>::empty_like(&a);
    /// assert_eq!(empty.shape(), vec![2, 2]);
    /// // Values are default-initialized
    /// assert_eq!(empty.to_vec(), vec![0, 0, 0, 0]);
    /// ```
    pub fn empty_like<U>(other: &Array<U>) -> Self
    where
        T: Default + Clone,
        U: Clone,
    {
        let shape = other.shape();
        let size: usize = shape.iter().product();
        let vec = vec![T::default(); size];
        Self::from_vec(vec).reshape(&shape)
    }

    /// Create a new array with the specified shape, data type, and order, uninitialized
    /// Note: This is similar to NumPy's empty_like_with but with safe Rust semantics
    ///
    /// # Parameters
    ///
    /// * `other` - The array whose shape to copy
    /// * `shape` - Optional shape for the new array. If None, the shape is the same as `other`
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    ///
    /// // Same shape as a
    /// let empty = Array::<f64>::empty_like_with(&a, None);
    /// assert_eq!(empty.shape(), vec![2, 2]);
    ///
    /// // Different shape
    /// let empty_3d = Array::<i32>::empty_like_with(&a, Some(&[2, 2, 2]));
    /// assert_eq!(empty_3d.shape(), vec![2, 2, 2]);
    /// assert_eq!(empty_3d.size(), 8);
    /// ```
    pub fn empty_like_with<U>(other: &Array<U>, shape: Option<&[usize]>) -> Self
    where
        T: Default + Clone,
        U: Clone,
    {
        let shape_to_use = match shape {
            Some(s) => s,
            None => &other.shape(),
        };

        let size: usize = shape_to_use.iter().product();
        let vec = vec![T::default(); size];
        Self::from_vec(vec).reshape(shape_to_use)
    }

    /// Get reference to the underlying ndarray
    pub fn array(&self) -> &NdArray<T, IxDyn> {
        &self.data
    }

    /// Returns the byte strides of the array
    ///
    /// Byte strides represent the number of bytes to move along each dimension
    /// when navigating the array in memory.
    ///
    /// # Returns
    ///
    /// A vector containing the byte strides for each dimension of the array
    pub fn byte_strides(&self) -> Vec<usize> {
        // Get the memory strides in terms of elements
        let elem_strides = self.data.strides();

        // Convert to byte strides by multiplying by the size of T
        let elem_size = std::mem::size_of::<T>();
        elem_strides
            .iter()
            .map(|&s| s as usize * elem_size)
            .collect()
    }

    /// Get a mutable reference to the underlying ndarray
    pub fn array_mut(&mut self) -> &mut NdArray<T, IxDyn> {
        &mut self.data
    }

    /// Set a value at the specified indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        // Check if indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx,
                    i,
                    self.shape()[i]
                )));
            }
        }

        // Set the value
        if let Some(elem) = self.array_mut().get_mut(indices) {
            *elem = value;
            Ok(())
        } else {
            Err(NumRs2Error::IndexOutOfBounds(format!(
                "Failed to set element at indices {:?}",
                indices
            )))
        }
    }

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
    /// Create a new array from a vector and reshape it
    pub fn from_vec(vec: Vec<T>) -> Self {
        let data = NdArray::from_shape_vec(IxDyn(&[vec.len()]), vec)
            .unwrap_or_else(|e| {
                // This should never happen with a properly sized vector
                // Log the error and create an empty array as last resort
                eprintln!("Critical: Array creation failed: {}. This indicates a serious bug.", e);
                // Create a minimal array that won't cause undefined behavior
                NdArray::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap()
            });
        Self { data }
    }

    /// Create a new array with a specific shape, filled with zeros
    pub fn zeros(shape: &[usize]) -> Self
    where
        T: Zero + Clone,
    {
        let data = NdArray::zeros(IxDyn(shape));
        Self { data }
    }

    /// Create a triangular matrix with ones below the given diagonal and zeros elsewhere
    ///
    /// # Parameters
    ///
    /// * `n` - Number of rows
    /// * `m` - Number of columns (defaults to `n` if None)
    /// * `k` - The diagonal below which to fill with ones (0 for main, positive for above, negative for below)
    /// * `value` - The value to fill with (defaults to 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Lower triangular matrix
    /// let a = Array::<i32>::tri(3, None, 0, None);
    /// assert_eq!(a.shape(), vec![3, 3]);
    /// assert_eq!(a.to_vec(), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
    ///
    /// // Upper triangular matrix (k=1)
    /// let b = Array::<i32>::tri(3, None, 1, None);
    /// assert_eq!(b.shape(), vec![3, 3]);
    /// assert_eq!(b.to_vec(), vec![1, 1, 0, 1, 1, 1, 1, 1, 1]);
    /// ```
    pub fn tri(n: usize, m: Option<usize>, k: isize, value: Option<T>) -> Self
    where
        T: Zero + One + Clone,
    {
        let m = m.unwrap_or(n);
        let value = value.unwrap_or_else(T::one);
        let zero = T::zero();

        let mut result = Self::zeros(&[n, m]);

        for i in 0..n {
            for j in 0..m {
                // NumPy's tri returns 1s on or below the diagonal (i-j <= k)
                if (j as isize) <= (i as isize) + k {
                    result.set(&[i, j], value.clone()).unwrap_or_else(|_| {
                        panic!("Internal error: failed to set element at [{}, {}] in tri function", i, j)
                    });
                } else {
                    result.set(&[i, j], zero.clone()).unwrap_or_else(|_| {
                        panic!("Internal error: failed to set element at [{}, {}] in tri function", i, j)
                    });
                }
            }
        }

        result
    }

    /// Create a lower triangular matrix or extract the lower triangle from an existing matrix
    ///
    /// # Parameters
    ///
    /// * `k` - The diagonal below which to extract/fill (0 for main, positive for above, negative for below)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a 3x3 matrix
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    ///
    /// // Get the lower triangle including the main diagonal
    /// let lower = a.tril(0);
    /// assert_eq!(lower.shape(), vec![3, 3]);
    /// assert_eq!(lower.to_vec(), vec![1, 0, 0, 4, 5, 0, 7, 8, 9]);
    ///
    /// // Get the lower triangle excluding the main diagonal
    /// let strictly_lower = a.tril(-1);
    /// assert_eq!(strictly_lower.shape(), vec![3, 3]);
    /// assert_eq!(strictly_lower.to_vec(), vec![0, 0, 0, 4, 0, 0, 7, 8, 0]);
    /// ```
    pub fn tril(&self, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        if self.ndim() != 2 {
            panic!("tril requires a 2D array");
        }

        let shape = self.shape();
        let n = shape[0];
        let m = shape[1];
        let zero = T::zero();

        let mut result = self.clone();

        for i in 0..n {
            for j in 0..m {
                // Zero out elements above the k-th diagonal
                // In NumPy, the condition is j > i + k
                if (j as isize) > (i as isize) + k {
                    result.set(&[i, j], zero.clone()).unwrap_or_else(|_| {
                        panic!("Internal error: failed to set element at [{}, {}] in tril function", i, j)
                    });
                }
            }
        }

        result
    }

    /// Create an upper triangular matrix or extract the upper triangle from an existing matrix
    ///
    /// # Parameters
    ///
    /// * `k` - The diagonal above which to extract/fill (0 for main, positive for above, negative for below)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a 3x3 matrix
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    ///
    /// // Get the upper triangle including the main diagonal
    /// let upper = a.triu(0);
    /// assert_eq!(upper.shape(), vec![3, 3]);
    /// assert_eq!(upper.to_vec(), vec![1, 2, 3, 0, 5, 6, 0, 0, 9]);
    ///
    /// // Get the upper triangle excluding the main diagonal
    /// let strictly_upper = a.triu(1);
    /// assert_eq!(strictly_upper.shape(), vec![3, 3]);
    /// assert_eq!(strictly_upper.to_vec(), vec![0, 2, 3, 0, 0, 6, 0, 0, 0]);
    /// ```
    pub fn triu(&self, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        if self.ndim() != 2 {
            panic!("triu requires a 2D array");
        }

        let shape = self.shape();
        let n = shape[0];
        let m = shape[1];
        let zero = T::zero();

        let mut result = self.clone();

        for i in 0..n {
            for j in 0..m {
                // Zero out elements below the k-th diagonal
                // In NumPy, the condition is j < i + k
                if (j as isize) < (i as isize) + k {
                    result.set(&[i, j], zero.clone()).unwrap_or_else(|_| {
                        panic!("Internal error: failed to set element at [{}, {}] in triu function", i, j)
                    });
                }
            }
        }

        result
    }

    /// Create a new array with a specific shape, filled with ones
    pub fn ones(shape: &[usize]) -> Self
    where
        T: One + Clone,
    {
        let data = NdArray::ones(IxDyn(shape));
        Self { data }
    }

    /// Create a new array with a specific shape, filled with a specific value
    pub fn full(shape: &[usize], value: T) -> Self
    where
        T: Clone,
    {
        let size: usize = shape.iter().product();
        let vec = vec![value; size];
        Self::from_vec(vec).reshape(shape)
    }

    /// Create a 2D identity matrix of the specified size
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let eye = Array::<i32>::identity(3);
    /// assert_eq!(eye.shape(), vec![3, 3]);
    /// assert_eq!(eye.to_vec(), vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
    /// ```
    pub fn identity(n: usize) -> Self
    where
        T: Zero + One + Clone,
    {
        Self::eye(n, n, 0)
    }

    /// Create a 2D identity matrix of the specified size (compatibility function)
    pub fn eye_square(n: usize) -> Self
    where
        T: Zero + One + Clone,
    {
        Self::eye(n, n, 0)
    }

    /// Create a 2D array with ones on the diagonal and zeros elsewhere
    ///
    /// Parameters:
    /// - `n_rows`: Number of rows
    /// - `n_cols`: Number of columns
    /// - `k`: Index of the diagonal (0 for main diagonal, positive for above, negative for below)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // 3x3 identity matrix
    /// let eye = Array::<i32>::eye(3, 3, 0);
    /// assert_eq!(eye.shape(), vec![3, 3]);
    /// assert_eq!(eye.to_vec(), vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
    ///
    /// // 3x3 matrix with diagonal above the main
    /// let eye_above = Array::<i32>::eye(3, 3, 1);
    /// assert_eq!(eye_above.shape(), vec![3, 3]);
    /// assert_eq!(eye_above.to_vec(), vec![0, 1, 0, 0, 0, 1, 0, 0, 0]);
    ///
    /// // 3x3 matrix with diagonal below the main
    /// let eye_below = Array::<i32>::eye(3, 3, -1);
    /// assert_eq!(eye_below.shape(), vec![3, 3]);
    /// assert_eq!(eye_below.to_vec(), vec![0, 0, 0, 1, 0, 0, 0, 1, 0]);
    ///
    /// // Rectangular matrix
    /// let rect_eye = Array::<f64>::eye(2, 4, 0);
    /// assert_eq!(rect_eye.shape(), vec![2, 4]);
    /// assert_eq!(rect_eye.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    /// ```
    pub fn eye(n_rows: usize, n_cols: usize, k: isize) -> Self
    where
        T: Zero + One + Clone,
    {
        let mut result = Self::zeros(&[n_rows, n_cols]);

        // Optimized diagonal setting with bounds checking
        let diagonal_start = if k >= 0 { 0 } else { (-k) as usize };
        let diagonal_col_start = if k >= 0 { k as usize } else { 0 };

        let max_diagonal_length = n_rows
            .saturating_sub(diagonal_start)
            .min(n_cols.saturating_sub(diagonal_col_start));

        // Set ones on the specified diagonal efficiently
        for i in 0..max_diagonal_length {
            let row = diagonal_start + i;
            let col = diagonal_col_start + i;
            if row < n_rows && col < n_cols {
                result.set(&[row, col], T::one()).unwrap_or_else(|_| {
                    panic!("Internal error: failed to set element at [{}, {}] in eye function", row, col)
                });
            }
        }

        result
    }

    /// Create a 2D array with the given values as a diagonal
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3]);
    /// let diag: Array<i32> = Array::create_diagonal_matrix(&a, 0);
    /// assert_eq!(diag.shape(), vec![3, 3]);
    /// assert_eq!(diag.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
    ///
    /// // Diagonal above the main
    /// let diag_above: Array<i32> = Array::create_diagonal_matrix(&a, 1);
    /// assert_eq!(diag_above.shape(), vec![4, 4]);
    /// assert_eq!(diag_above.to_vec(), vec![0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0]);
    /// ```
    pub fn create_diagonal_matrix_helper(v: &Array<T>, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        if v.ndim() != 1 {
            // In a real implementation, we should return a Result, but for simplicity,
            // we'll panic with a clear message
            panic!("diag requires a 1D array");
        }

        let diag_len = v.size();
        let size = diag_len + k.unsigned_abs();

        let mut result = Self::zeros(&[size, size]);

        // Set values along the specified diagonal
        for i in 0..diag_len {
            if k >= 0 {
                let j = i + k as usize;
                if j < size {
                    result
                        .set(&[i, j], v.array().get([i]).unwrap().clone())
                        .unwrap_or_else(|_| {
                            panic!("Internal error: failed to set element at [{}, {}] in diag function", i, j)
                        });
                }
            } else {
                let i_offset = (-k) as usize;
                if i + i_offset < size {
                    result
                        .set(&[i + i_offset, i], v.array().get([i]).unwrap().clone())
                        .unwrap_or_else(|_| {
                            panic!("Internal error: failed to set element at [{}, {}] in diag function", i + i_offset, i)
                        });
                }
            }
        }

        result
    }

    /// Extract a diagonal from a 2D array or create a diagonal matrix from a 1D array
    ///
    /// # Parameters
    ///
    /// * `v` - Input array. If v is 2D, return its kth diagonal. If v is 1D, return a 2D array with v as its kth diagonal.
    /// * `k` - Diagonal offset. k=0 refers to the main diagonal, k>0 to the kth diagonal above the main, and k<0 to the kth diagonal below the main.
    ///
    /// # Returns
    ///
    /// The diagonal array or a matrix with the diagonal
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a diagonal matrix from a 1D array
    /// let a = Array::from_vec(vec![1, 2, 3]);
    /// let diag: Array<i32> = Array::create_diagonal_matrix(&a, 0);
    /// assert_eq!(diag.shape(), vec![3, 3]);
    /// assert_eq!(diag.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
    ///
    /// // Extract the main diagonal from a 2D array
    /// let b = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    /// let main_diag: Array<i32> = Array::create_diagonal_matrix(&b, 0);
    /// assert_eq!(main_diag.shape(), vec![3]);
    /// assert_eq!(main_diag.to_vec(), vec![1, 5, 9]);
    ///
    /// // Extract diagonal above the main
    /// let above_diag: Array<i32> = Array::create_diagonal_matrix(&b, 1);
    /// assert_eq!(above_diag.shape(), vec![2]);
    /// assert_eq!(above_diag.to_vec(), vec![2, 6]);
    /// ```
    // This needs a different name to avoid conflict with the instance method in indexing.rs
    pub fn create_diagonal_matrix(v: &Array<T>, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        if v.ndim() == 1 {
            // Create a diagonal matrix
            Self::create_diagonal_matrix_helper(v, k)
        } else if v.ndim() == 2 {
            // Extract the diagonal
            let shape = v.shape();
            let n_rows = shape[0];
            let n_cols = shape[1];

            let mut diag_elements = Vec::new();

            // Calculate the length of the diagonal we're extracting
            let diag_len = if k >= 0 {
                std::cmp::min(n_rows, n_cols.saturating_sub(k as usize))
            } else {
                std::cmp::min(n_cols, n_rows.saturating_sub((-k) as usize))
            };

            // Extract the diagonal elements
            for i in 0..diag_len {
                if k >= 0 {
                    let j = i + k as usize;
                    if j < n_cols {
                        diag_elements.push(v.array().get([i, j]).unwrap().clone());
                    }
                } else {
                    let i_offset = (-k) as usize;
                    if i + i_offset < n_rows {
                        diag_elements.push(v.array().get([i + i_offset, i]).unwrap().clone());
                    }
                }
            }

            // Return as a 1D array
            Self::from_vec(diag_elements)
        } else {
            panic!("diag requires a 1D or 2D array");
        }
    }

    /// Extract the diagonal of an array or construct a diagonal array from a 1D array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Extract diagonal from a 2D array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    /// let diag: Array<i32> = Array::diagflat(&a, 0);
    /// assert_eq!(diag.shape(), vec![9, 9]);
    ///
    /// // Create diagonal array from a 1D array
    /// let b = Array::from_vec(vec![1, 2, 3]);
    /// let diag_b: Array<i32> = Array::diagflat(&b, 0);
    /// assert_eq!(diag_b.shape(), vec![3, 3]);
    /// assert_eq!(diag_b.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
    /// ```
    pub fn diagflat(v: &Array<T>, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        // If already 1D, create a diagonal matrix
        if v.ndim() == 1 {
            return Self::create_diagonal_matrix(v, k);
        }

        // Otherwise, flatten the array and create a diagonal matrix
        let flat = v.reshape(&[v.size()]);
        Self::create_diagonal_matrix(&flat, k)
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

    // Removing the duplicate ravel implementation
    // This is a duplicate of the one defined above

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

    /// Return a flattened copy of the array
    ///
    /// # Returns
    ///
    /// A new 1D array with a copy of the data
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let b = a.flatten(None);
    /// assert_eq!(b.shape(), vec![6]);
    /// assert_eq!(b.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    // Implementation moved to the first flatten method above to avoid duplication
    // Implementation of ravel moved to the first ravel method above to avoid duplication
    /// Return the shape of the array
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Return the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Return the total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Return the data as a flat vector
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        let (raw_vec, _) = self.data.clone().into_raw_vec_and_offset();
        raw_vec
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

    /// Get a view of the underlying ndarray data (low-level)
    pub fn ndarray_view(&self) -> ArrayView<T, IxDyn> {
        self.data.view()
    }

    /// Get a mutable reference to self for method chaining
    /// Note: This is a placeholder for what would be a proper mutable view in a complete implementation
    pub fn ndarray_view_mut(&mut self) -> &mut Self
    where
        T: Clone,
    {
        // In a real implementation, we would return an actual mutable view
        // For now, we'll just return a mutable reference to self
        self
    }

    /// Perform element-wise multiplication by a scalar
    ///
    /// # Parameters
    ///
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    ///
    /// A new array with each element multiplied by the scalar
    pub fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Clone + Mul<Output = T>,
    {
        self.map(|x| x * scalar.clone())
    }

    /// Perform element-wise division by a scalar
    ///
    /// # Parameters
    ///
    /// * `scalar` - The scalar value to divide by
    ///
    /// # Returns
    ///
    /// A new array with each element divided by the scalar
    pub fn scalar_div(&self, scalar: T) -> Self
    where
        T: Clone + Div<Output = T>,
    {
        self.map(|x| x / scalar.clone())
    }

    /// Calculate the sum of all elements in the array
    ///
    /// # Returns
    ///
    /// The sum of all elements
    pub fn sum_all(&self) -> T
    where
        T: Clone + Add<Output = T> + Zero,
    {
        let data = self.to_vec();
        data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate the sum along the specified axis
    ///
    /// # Parameters
    ///
    /// * `axis` - The axis along which to sum
    ///
    /// # Returns
    ///
    /// A new array with the specified axis removed
    pub fn sum_axis(&self, axis: usize) -> Result<Self>
    where
        T: Clone + Add<Output = T> + Zero,
    {
        let axis_val = axis;
        if axis_val >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis_val,
                self.ndim()
            )));
        }

        let shape = self.shape();
        let axis_size = shape[axis_val];

        // Calculate the shape of the result
        let mut result_shape = shape.clone();
        result_shape.remove(axis_val);

        // Initialize the result array
        let mut result = Self::zeros(&result_shape);

        // Get the raw data
        let data = self.to_vec();

        // Helper function to calculate indices
        let mut indices = vec![0; shape.len()];
        let mut result_indices = vec![0; result_shape.len()];

        // Calculate the total number of elements in the result
        let result_size = result.size();

        // For each position in the result array
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            for (j, idx) in indices.iter_mut().enumerate() {
                if j == axis_val {
                    *idx = 0; // Start at 0 for the axis we're summing
                } else {
                    *idx = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Sum along the specified axis
            let mut sum = T::zero();
            for k in 0..axis_size {
                indices[axis_val] = k;

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                sum = sum + data[flat_idx].clone();
            }

            // Set the result value
            result.set(&result_indices, sum)?;
        }

        Ok(result)
    }

    /// Get a 2D view of the underlying ndarray data
    pub fn view_2d(&self) -> Result<ArrayView2<T>>
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

    /// Apply a function to each element of the array in parallel
    pub fn par_map<F, U>(&self, f: F) -> Array<U>
    where
        T: Send + Sync + Clone,
        U: Send + Clone,
        F: Fn(T) -> U + Send + Sync,
    {
        let vec_data = self.to_vec();
        let result: Vec<U> = vec_data.par_iter().map(|x| f(x.clone())).collect();

        Array::from_vec(result).reshape(&self.shape())
    }

    /// Apply a function to each element of the array
    pub fn map<F, U>(&self, f: F) -> Array<U>
    where
        U: Clone,
        F: Fn(T) -> U,
        T: Clone,
    {
        let vec_data = self.to_vec();
        let result: Vec<U> = vec_data.iter().map(|x| f(x.clone())).collect();

        Array::from_vec(result).reshape(&self.shape())
    }

    /// Apply a function to corresponding elements of two arrays with broadcasting
    pub fn zip_with<F, U, V>(&self, other: &Array<U>, f: F) -> Result<Array<V>>
    where
        T: Clone,
        U: Clone,
        V: Clone,
        F: Fn(T, U) -> V,
    {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // If shapes are equal, apply function directly without broadcasting
        if a_shape == b_shape {
            let self_data = self.to_vec();
            let other_data = other.to_vec();

            let result: Vec<V> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(a, b)| f(a.clone(), b.clone()))
                .collect();

            return Ok(Array::from_vec(result).reshape(&self.shape()));
        }

        // Calculate broadcast shape
        let broadcast_shape = Self::broadcast_shape(&a_shape, &b_shape)?;

        // Broadcast both arrays to the new shape
        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        // Now apply the function to the broadcasted arrays (which have the same shape)
        let self_data = self_broadcast.to_vec();
        let other_data = other_broadcast.to_vec();

        let result: Vec<V> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| f(a.clone(), b.clone()))
            .collect();

        Ok(Array::from_vec(result).reshape(&broadcast_shape))
    }

    /// Broadcast binary operation between two arrays of potentially different shapes
    pub fn broadcast_op<F, U, V>(&self, other: &Array<U>, op: F) -> Result<Array<V>>
    where
        T: Clone,
        U: Clone,
        V: Clone,
        F: Fn(&Array<T>, &Array<U>) -> Array<V>,
    {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // If shapes are equal, apply operation directly
        if a_shape == b_shape {
            return Ok(op(self, other));
        }

        // Calculate broadcast shape
        let broadcast_shape = Self::broadcast_shape(&a_shape, &b_shape)?;

        // Broadcast both arrays to the new shape
        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        // Apply the operation on the broadcasted arrays
        Ok(op(&self_broadcast, &other_broadcast))
    }
}

// Add sum and product methods
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Zero + Mul<Output = T> + num_traits::One,
{
    /// Calculate the sum of all elements in the array
    pub fn sum(&self) -> T {
        let data = self.to_vec();
        data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate the product of all elements in the array
    pub fn product(&self) -> T {
        let data = self.to_vec();
        data.iter().fold(T::one(), |acc, x| acc * x.clone())
    }
}

// Matrix multiplication
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    /// Perform matrix multiplication using BLAS if available
    ///
    /// Enhanced version with support for broadcasting and stacked matrices.
    /// If arrays have more than 2 dimensions, they are treated as stacks of matrices
    /// and broadcasting rules are applied to stack dimensions.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Handle the basic 2D case directly
        if a_shape.len() == 2 && b_shape.len() == 2 {
            return self.matmul_2d(other);
        }

        // For higher dimensions, we need to handle broadcasting
        // Ensure both arrays have at least 2 dimensions
        let a = if a_shape.len() == 1 {
            self.reshape(&[1, a_shape[0]])
        } else {
            self.clone()
        };

        let b = if b_shape.len() == 1 {
            other.reshape(&[b_shape[0], 1])
        } else {
            other.clone()
        };

        let a_shape = a.shape();
        let b_shape = b.shape();

        // Extract core dimensions (last 2 of each array)
        let a_core_shape = &a_shape[a_shape.len() - 2..];
        let b_core_shape = &b_shape[b_shape.len() - 2..];

        // Check if core dimensions are compatible for matrix multiplication
        if a_core_shape[1] != b_core_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_core_shape[0], b_core_shape[1]],
                actual: vec![a_core_shape[0], a_core_shape[1]],
            });
        }

        // Calculate batch dimensions (all but the last 2 of each array)
        let a_batch_shape = &a_shape[..a_shape.len() - 2];
        let b_batch_shape = &b_shape[..b_shape.len() - 2];

        // Calculate broadcast batch shape
        let broadcast_batch_shape = if a_batch_shape.is_empty() && b_batch_shape.is_empty() {
            vec![]
        } else if a_batch_shape.is_empty() {
            b_batch_shape.to_vec()
        } else if b_batch_shape.is_empty() {
            a_batch_shape.to_vec()
        } else {
            // Use broadcasting rules to get common batch shape
            Self::broadcast_shape(a_batch_shape, b_batch_shape)?
        };

        // Reshape arrays to broadcast batch dimensions
        let a_broadcast_shape = [&broadcast_batch_shape, a_core_shape].concat();
        let b_broadcast_shape = [&broadcast_batch_shape, b_core_shape].concat();

        let a_broadcast = if a_shape == a_broadcast_shape {
            a
        } else {
            a.broadcast_to(&a_broadcast_shape)?
        };

        let b_broadcast = if b_shape == b_broadcast_shape {
            b
        } else {
            b.broadcast_to(&b_broadcast_shape)?
        };

        // Calculate output shape
        let output_core_shape = vec![a_core_shape[0], b_core_shape[1]];
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.extend_from_slice(&output_core_shape);

        // Perform batch matrix multiplication
        let mut result = Self::zeros(&output_shape);

        // Calculate total batch size
        let batch_size: usize = broadcast_batch_shape.iter().product();

        // For each batch, perform matrix multiplication
        for batch_idx in 0..batch_size {
            // Calculate indices for this batch
            let mut batch_indices = Vec::with_capacity(broadcast_batch_shape.len());
            let mut temp = batch_idx;

            for &dim in broadcast_batch_shape.iter().rev() {
                batch_indices.insert(0, temp % dim);
                temp /= dim;
            }

            // Extract matrices for this batch
            let mut a_indices = batch_indices.clone();
            a_indices.push(0); // Placeholder for row index
            a_indices.push(0); // Placeholder for column index

            let mut b_indices = batch_indices.clone();
            b_indices.push(0); // Placeholder for row index
            b_indices.push(0); // Placeholder for column index

            // Perform matrix multiplication for this batch
            let m = a_core_shape[0];
            let n = b_core_shape[1];
            let k = a_core_shape[1];

            for i in 0..m {
                let a_idx_pos = a_indices.len() - 2;
                a_indices[a_idx_pos] = i;

                for j in 0..n {
                    let b_idx_pos = b_indices.len() - 1;
                    b_indices[b_idx_pos] = j;

                    let mut sum = T::zero();

                    for l in 0..k {
                        let a_col_pos = a_indices.len() - 1;
                        a_indices[a_col_pos] = l;
                        let b_row_pos = b_indices.len() - 2;
                        b_indices[b_row_pos] = l;

                        let a_val = a_broadcast.array().get(IxDyn(&a_indices)).unwrap();
                        let b_val = b_broadcast.array().get(IxDyn(&b_indices)).unwrap();

                        sum = sum + a_val.clone() * b_val.clone();
                    }

                    // Calculate output indices
                    let mut output_indices = batch_indices.clone();
                    output_indices.push(i);
                    output_indices.push(j);

                    result.set(&output_indices, sum)?;
                }
            }
        }

        Ok(result)
    }

    /// Basic 2D matrix multiplication (no broadcasting)
    fn matmul_2d(&self, other: &Self) -> Result<Self> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Check dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "matmul_2d requires 2D arrays".to_string(),
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0], b_shape[1]],
                actual: vec![a_shape[0], a_shape[1]],
            });
        }

        // In a complete implementation, we would use BLAS for this
        // For now, we'll implement a simple matrix multiplication algorithm
        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        let result = Self::zeros(&[m, n]);
        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let mut c_data = result.to_vec();

        // Simple matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + a_data[i * k + l].clone() * b_data[l * n + j].clone();
                }
                c_data[i * n + j] = sum;
            }
        }

        Ok(Self::from_vec(c_data).reshape(&[m, n]))
    }

    /// Compute the dot product of two vectors
    pub fn dot(&self, other: &Self) -> Result<T> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Check dimensions
        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "dot product requires 1D arrays".to_string(),
            ));
        }

        if a_shape[0] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a_shape,
                actual: b_shape,
            });
        }

        // Compute dot product
        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let mut result = T::zero();

        for i in 0..a_shape[0] {
            result = result + a_data[i].clone() * b_data[i].clone();
        }

        Ok(result)
    }
}

// We're not using operator overloads to avoid complications with borrowing
// Instead, we'll use the method-based approach like add_broadcast, etc.

/// Method to directly access the slice operation to get a view
impl<T: Clone> Array<T> {
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

        use ndarray::Axis as NdAxis;
        let sliced = self.array().index_axis(NdAxis(axis), index);
        Ok(crate::views::ArrayView::from_ndarray_view(
            sliced.into_dyn(),
        ))
    }
}

// Non-Result returning versions for convenience (assumes same shape)
impl<T: Clone + Add<Output = T>> Array<T> {
    /// Add arrays without broadcasting (for convenience)
    pub fn add(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data + &other.data;
        Array { data: result }
    }

    /// Add arrays with broadcasting
    pub fn add_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data + &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone + Sub<Output = T>> Array<T> {
    /// Subtract arrays without broadcasting (for convenience)
    pub fn subtract(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data - &other.data;
        Array { data: result }
    }

    /// Subtract arrays with broadcasting
    pub fn subtract_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data - &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone + Mul<Output = T>> Array<T> {
    /// Multiply arrays without broadcasting (for convenience)
    pub fn multiply(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data * &other.data;
        Array { data: result }
    }

    /// Multiply arrays with broadcasting
    pub fn multiply_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data * &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone + Div<Output = T>> Array<T> {
    /// Divide arrays without broadcasting (for convenience)
    pub fn divide(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data / &other.data;
        Array { data: result }
    }

    /// Divide arrays with broadcasting
    pub fn divide_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data / &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone> Array<T> {
    /// Return the total number of elements (alias for size)
    pub fn len(&self) -> usize {
        self.size()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if the array is C-contiguous (row-major)
    pub fn is_c_contiguous(&self) -> bool {
        self.data.is_standard_layout()
    }

    /// Check if the array is Fortran-contiguous (column-major)  
    pub fn is_f_contiguous(&self) -> bool {
        // ndarray doesn't have a direct is_fortran_layout, but we can check
        // if the array has the expected strides for Fortran layout
        let shape = self.data.shape();
        let strides = self.data.strides();

        if shape.is_empty() {
            return true;
        }

        // For Fortran layout, stride should increase with dimension
        let mut expected_stride = 1;
        for i in 0..shape.len() {
            if strides[i] != expected_stride as isize {
                return false;
            }
            expected_stride *= shape[i];
        }
        true
    }

    /// Check if the array is contiguous (either C or Fortran)
    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous() || self.is_f_contiguous()
    }

    /// Convert array to C layout (row-major)
    pub fn to_c_layout(&self) -> Self {
        if self.is_c_contiguous() {
            self.clone()
        } else {
            // Convert to standard layout
            let standard = self.data.as_standard_layout();
            Self {
                data: standard.into_owned(),
            }
        }
    }

    /// Convert array to Fortran layout (column-major)
    pub fn to_f_layout(&self) -> Self {
        if self.is_f_contiguous() {
            self.clone()
        } else {
            // For Fortran layout, we need to transpose all dimensions
            // This is a simplified implementation
            let transposed = self.data.clone().reversed_axes();
            Self { data: transposed }
        }
    }
}

// Implement scalar operations
impl<T: Clone + Add<Output = T>> Array<T> {
    /// Add a scalar to the array (element-wise)
    pub fn add_scalar(&self, scalar: T) -> Self {
        self.map(|x| x + scalar.clone())
    }
}

impl<T: Clone + Sub<Output = T>> Array<T> {
    /// Subtract a scalar from the array (element-wise)
    pub fn subtract_scalar(&self, scalar: T) -> Self {
        self.map(|x| x - scalar.clone())
    }
}

impl<T: Clone + Mul<Output = T>> Array<T> {
    /// Multiply the array by a scalar (element-wise)
    pub fn multiply_scalar(&self, scalar: T) -> Self {
        self.map(|x| x * scalar.clone())
    }
}

impl<T: Clone + Div<Output = T>> Array<T> {
    /// Divide the array by a scalar (element-wise)
    pub fn divide_scalar(&self, scalar: T) -> Self {
        self.map(|x| x / scalar.clone())
    }
}

// Display implementation for Array
impl<T: fmt::Display> fmt::Display for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

// Debug implementation for Array
impl<T: fmt::Debug + Clone> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("data", &self.data)
            .finish()
    }
}
