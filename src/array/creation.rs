//! Array creation methods
//!
//! This module contains constructors and factory methods for creating arrays:
//! - zeros, ones, full, empty
//! - identity, eye
//! - tri, tril, triu
//! - diagonal matrices
//! - from_vec

use super::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{One, Zero};
use scirs2_core::ndarray::{Array as NdArray, IxDyn};

impl<T: Clone> Array<T> {
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

    /// Create a new array from a vector and reshape it
    pub fn from_vec(vec: Vec<T>) -> Self {
        let data = NdArray::from_shape_vec(IxDyn(&[vec.len()]), vec).unwrap_or_else(|e| {
            // This should never happen with a properly sized vector
            // Log the error and create an empty array as last resort
            eprintln!(
                "Critical: Array creation failed: {}. This indicates a serious bug.",
                e
            );
            // Create a minimal array that won't cause undefined behavior
            NdArray::from_shape_vec(IxDyn(&[0]), Vec::new())
                .expect("empty array creation should succeed")
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
                // INVARIANT: `result` was just created as `zeros(&[n, m])`, and
                // `i`/`j` are bounded by `0..n`/`0..m`, so `[i, j]` is always a
                // valid index into `result` and `set` cannot return `Err` here.
                // NumPy's tri returns 1s on or below the diagonal (i-j <= k)
                if (j as isize) <= (i as isize) + k {
                    result.set(&[i, j], value.clone()).unwrap_or_else(|_| {
                        panic!(
                            "Internal error: failed to set element at [{}, {}] in tri function",
                            i, j
                        )
                    });
                } else {
                    result.set(&[i, j], zero.clone()).unwrap_or_else(|_| {
                        panic!(
                            "Internal error: failed to set element at [{}, {}] in tri function",
                            i, j
                        )
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
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a 2D array. Use [`Array::try_tril`] for a
    /// non-panicking version that returns a [`crate::error::NumRs2Error`].
    pub fn tril(&self, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        self.try_tril(k)
            .unwrap_or_else(|e| panic!("tril requires a 2D array: {e}"))
    }

    /// Non-panicking version of [`Array::tril`].
    ///
    /// Returns `Err` if `self` is not a 2D array instead of panicking.
    pub fn try_tril(&self, k: isize) -> Result<Self>
    where
        T: Zero + Clone,
    {
        if self.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "tril requires a 2D array".to_string(),
            ));
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
                    result.set(&[i, j], zero.clone())?;
                }
            }
        }

        Ok(result)
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
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a 2D array. Use [`Array::try_triu`] for a
    /// non-panicking version that returns a [`crate::error::NumRs2Error`].
    pub fn triu(&self, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        self.try_triu(k)
            .unwrap_or_else(|e| panic!("triu requires a 2D array: {e}"))
    }

    /// Non-panicking version of [`Array::triu`].
    ///
    /// Returns `Err` if `self` is not a 2D array instead of panicking.
    pub fn try_triu(&self, k: isize) -> Result<Self>
    where
        T: Zero + Clone,
    {
        if self.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "triu requires a 2D array".to_string(),
            ));
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
                    result.set(&[i, j], zero.clone())?;
                }
            }
        }

        Ok(result)
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

    /// Create a new array with a specific shape, with uninitialized values
    ///
    /// Note: This is similar to NumPy's empty but with safe Rust semantics.
    /// The array will be initialized with default values instead of random memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let empty: Array<i32> = Array::empty(&[2, 3]);
    /// assert_eq!(empty.shape(), vec![2, 3]);
    /// assert_eq!(empty.size(), 6);
    /// // Values are default-initialized (zeros for numeric types)
    /// assert_eq!(empty.to_vec(), vec![0, 0, 0, 0, 0, 0]);
    ///
    /// let empty_f64: Array<f64> = Array::empty(&[2, 2]);
    /// assert_eq!(empty_f64.to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
    /// ```
    pub fn empty(shape: &[usize]) -> Self
    where
        T: Default + Clone,
    {
        let size: usize = shape.iter().product();
        let vec = vec![T::default(); size];
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
        //
        // INVARIANT: `max_diagonal_length` is `min(n_rows - diagonal_start,
        // n_cols - diagonal_col_start)` (saturating), so for every
        // `i < max_diagonal_length`, `row = diagonal_start + i < n_rows` and
        // `col = diagonal_col_start + i < n_cols`. `result` has shape
        // `[n_rows, n_cols]`, so `set` cannot return `Err` here; the
        // `row < n_rows && col < n_cols` check below is a defensive
        // restatement of that same invariant, not evidence it can fail.
        for i in 0..max_diagonal_length {
            let row = diagonal_start + i;
            let col = diagonal_col_start + i;
            if row < n_rows && col < n_cols {
                result.set(&[row, col], T::one()).unwrap_or_else(|_| {
                    panic!(
                        "Internal error: failed to set element at [{}, {}] in eye function",
                        row, col
                    )
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
    ///
    /// # Panics
    ///
    /// Panics if `v` is not a 1D array. Use
    /// [`Array::try_create_diagonal_matrix_helper`] for a non-panicking
    /// version that returns a [`crate::error::NumRs2Error`].
    pub fn create_diagonal_matrix_helper(v: &Array<T>, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        Self::try_create_diagonal_matrix_helper(v, k)
            .unwrap_or_else(|e| panic!("diag requires a 1D array: {e}"))
    }

    /// Non-panicking version of [`Array::create_diagonal_matrix_helper`].
    ///
    /// Returns `Err` if `v` is not a 1D array instead of panicking.
    pub fn try_create_diagonal_matrix_helper(v: &Array<T>, k: isize) -> Result<Self>
    where
        T: Zero + Clone,
    {
        if v.ndim() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "diag requires a 1D array".to_string(),
            ));
        }

        let diag_len = v.size();
        let size = diag_len + k.unsigned_abs();

        let mut result = Self::zeros(&[size, size]);

        // Set values along the specified diagonal
        for i in 0..diag_len {
            if k >= 0 {
                let j = i + k as usize;
                if j < size {
                    let value = v.array().get([i]).ok_or_else(|| {
                        NumRs2Error::IndexOutOfBounds(format!(
                            "diag: failed to read source element at [{}]",
                            i
                        ))
                    })?;
                    result.set(&[i, j], value.clone())?;
                }
            } else {
                let i_offset = (-k) as usize;
                if i + i_offset < size {
                    let value = v.array().get([i]).ok_or_else(|| {
                        NumRs2Error::IndexOutOfBounds(format!(
                            "diag: failed to read source element at [{}]",
                            i
                        ))
                    })?;
                    result.set(&[i + i_offset, i], value.clone())?;
                }
            }
        }

        Ok(result)
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
    ///
    /// # Panics
    ///
    /// Panics if `v` is neither a 1D nor a 2D array. Use
    /// [`Array::try_create_diagonal_matrix`] for a non-panicking version
    /// that returns a [`crate::error::NumRs2Error`].
    // This needs a different name to avoid conflict with the instance method in indexing.rs
    pub fn create_diagonal_matrix(v: &Array<T>, k: isize) -> Self
    where
        T: Zero + Clone,
    {
        Self::try_create_diagonal_matrix(v, k)
            .unwrap_or_else(|e| panic!("diag requires a 1D or 2D array: {e}"))
    }

    /// Non-panicking version of [`Array::create_diagonal_matrix`].
    ///
    /// Returns `Err` if `v` is neither a 1D nor a 2D array instead of
    /// panicking.
    pub fn try_create_diagonal_matrix(v: &Array<T>, k: isize) -> Result<Self>
    where
        T: Zero + Clone,
    {
        if v.ndim() == 1 {
            // Create a diagonal matrix
            Self::try_create_diagonal_matrix_helper(v, k)
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
                        let value = v.array().get([i, j]).ok_or_else(|| {
                            NumRs2Error::IndexOutOfBounds(format!(
                                "diag: failed to read source element at [{}, {}]",
                                i, j
                            ))
                        })?;
                        diag_elements.push(value.clone());
                    }
                } else {
                    let i_offset = (-k) as usize;
                    if i + i_offset < n_rows {
                        let value = v.array().get([i + i_offset, i]).ok_or_else(|| {
                            NumRs2Error::IndexOutOfBounds(format!(
                                "diag: failed to read source element at [{}, {}]",
                                i + i_offset,
                                i
                            ))
                        })?;
                        diag_elements.push(value.clone());
                    }
                }
            }

            // Return as a 1D array
            Ok(Self::from_vec(diag_elements))
        } else {
            Err(NumRs2Error::DimensionMismatch(
                "diag requires a 1D or 2D array".to_string(),
            ))
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
}
