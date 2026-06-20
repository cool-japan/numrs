//! Core Array struct definition and basic properties
//!
//! This module contains the fundamental `Array<T>` struct definition,
//! along with basic methods for querying array properties like shape,
//! size, strides, and memory layout.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array as NdArray, Array1, ArrayView, ArrayView1, CowArray, Ix1, IxDyn};
use std::fmt;

/// Type alias for least squares return type
/// Returns (solution, residuals, rank, singular_values)
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub type LstsqResult<T> = Result<(
    super::Array<T>,
    super::Array<T>, // Residuals are same type as matrix elements
    usize,
    super::Array<T>, // Singular values are same type as matrix elements
)>;

/// Flags that describe the memory layout and properties of an array
#[derive(Debug, Clone)]
pub struct ArrayFlags {
    /// Array data is stored in C-contiguous order
    pub c_contiguous: bool,
    /// Array data is stored in Fortran-contiguous order
    pub f_contiguous: bool,
    /// Array data is writeable
    pub writeable: bool,
    /// Array data is aligned
    pub aligned: bool,
    /// Array owns its data
    pub owndata: bool,
}

/// A multi-dimensional array type that wraps ndarray
#[derive(Clone)]
pub struct Array<T> {
    pub(crate) data: NdArray<T, IxDyn>,
}

impl<T: Clone> Array<T> {
    /// Create a new array from an ndarray
    pub fn from_ndarray(array: NdArray<T, IxDyn>) -> Self {
        Self { data: array }
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

    /// Return the total bytes consumed by the elements of the array
    pub fn nbytes(&self) -> usize {
        self.size() * std::mem::size_of::<T>()
    }

    /// Return the length of one array element in bytes
    pub fn itemsize(&self) -> usize {
        std::mem::size_of::<T>()
    }

    /// Return True if array owns its data
    pub fn owns_data(&self) -> bool {
        // ndarray arrays always own their data when created through our interface
        true
    }

    /// Return the memory layout of the array
    pub fn flags(&self) -> ArrayFlags {
        ArrayFlags {
            c_contiguous: self.data.is_standard_layout(),
            f_contiguous: false, // ndarray uses C-order by default
            writeable: true,     // Our arrays are always writable
            aligned: true,       // Rust guarantees proper alignment
            owndata: true,       // We own the data
        }
    }

    /// Return the strides of the array
    pub fn strides(&self) -> Vec<isize> {
        self.data.strides().to_vec()
    }

    /// Return the base array if this is a view, otherwise None
    pub fn base(&self) -> Option<&Array<T>> {
        // Since we always own data, there's no base array
        None
    }

    /// Return the data as a flat vector
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        let (raw_vec, _) = self.data.clone().into_raw_vec_and_offset();
        raw_vec
    }

    /// Borrow the underlying data as a contiguous slice without copying.
    ///
    /// Returns `Some(&[T])` when the array is stored in standard
    /// (C-contiguous) layout, and `None` otherwise. This is the zero-copy
    /// fast path: prefer it over [`Array::to_vec`] whenever the data only
    /// needs to be read, falling back to `to_vec()` for the non-contiguous
    /// case.
    pub fn as_slice(&self) -> Option<&[T]> {
        self.data.as_slice()
    }

    /// Mutably borrow the underlying data as a contiguous slice without
    /// copying.
    ///
    /// Returns `Some(&mut [T])` when the array is stored in standard
    /// (C-contiguous) layout, and `None` otherwise.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.data.as_slice_mut()
    }

    /// Borrow the flattened data as a 1-D `CowArray`, avoiding a copy when the
    /// array is contiguous.
    ///
    /// This backs the SIMD-accelerated paths (in `simd.rs`, `ufuncs.rs` and
    /// `linalg.rs`) which need an [`ArrayView1`] to hand to `scirs2-core`'s
    /// `SimdUnifiedOps`. For the common contiguous case it borrows the
    /// existing buffer with zero allocation; only non-contiguous layouts fall
    /// back to materializing an owned copy via [`Array::to_vec`].
    pub(crate) fn as_cow_1d(&self) -> CowArray<'_, T, Ix1>
    where
        T: Clone,
    {
        match self.as_slice() {
            Some(slice) => CowArray::from(ArrayView1::from(slice)),
            None => CowArray::from(Array1::from_vec(self.to_vec())),
        }
    }

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

    /// Get an element by flat index (for expression template optimization)
    ///
    /// # Parameters
    ///
    /// * `index` - Flat index into the array
    ///
    /// # Returns
    ///
    /// The element at the specified flat index
    ///
    /// # Performance
    ///
    /// This is an O(1) operation, avoiding the O(n) cost of calling to_vec()
    /// for every element access in expression templates.
    pub fn get_flat(&self, index: usize) -> Result<T>
    where
        T: Clone,
    {
        if index >= self.size() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Flat index {} out of bounds for array of size {}",
                index,
                self.size()
            )));
        }

        // Convert flat index to multi-dimensional indices
        let mut indices = Vec::with_capacity(self.ndim());
        let mut remainder = index;
        let shape = self.shape();

        for i in (0..self.ndim()).rev() {
            indices.push(remainder % shape[i]);
            remainder /= shape[i];
        }
        indices.reverse();

        // Access using multi-dimensional indices - O(1) operation
        self.data.get(&indices[..]).cloned().ok_or_else(|| {
            NumRs2Error::IndexOutOfBounds(format!("Failed to get element at flat index {}", index))
        })
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

    /// Get a view of the underlying ndarray data (low-level)
    pub fn ndarray_view(&self) -> ArrayView<'_, T, IxDyn> {
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
