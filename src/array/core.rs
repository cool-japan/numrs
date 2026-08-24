//! Core Array struct definition and basic properties
//!
//! This module contains the fundamental `Array<T>` struct definition,
//! along with basic methods for querying array properties like shape,
//! size, strides, and memory layout.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array as NdArray, Array1, ArrayView, ArrayView1, CowArray, Ix1, IxDyn};
use std::fmt;
use std::sync::Arc;

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

/// A multi-dimensional array type that wraps ndarray.
///
/// # Storage: `Arc`-backed copy-on-write
///
/// The owned buffer lives behind an [`Arc`], which makes [`Clone`] an O(1)
/// reference-count bump instead of an O(n) deep copy. The buffer is *not*
/// shared observably: every mutating path in the crate goes through
/// [`Array::nd_mut`], the single place that calls `Arc::make_mut`, so the
/// first mutation of a shared buffer transparently unshares it (one deep
/// copy) before writing. Callers therefore see exactly the value semantics
/// they saw when the field was a plain owned `ndarray::Array` -- mutating one
/// clone never disturbs another -- while paying for the copy only when a
/// write actually happens.
///
/// [`Array::is_unique`] exposes whether this handle is currently the sole
/// owner of its buffer, which is what the copy-on-write tests assert on.
///
/// # Auto-trait note
///
/// `Arc<X>` is `Send`/`Sync` only when `X: Send + Sync`, so `Array<T>` is now
/// `Send`/`Sync` when `T: Send + Sync` (previously `Array<T>: Send` needed
/// only `T: Send`). This is invisible for every numeric/`Copy` element type;
/// it can only be observed with an exotic `Send + !Sync` element such as
/// `Cell<i32>`.
#[derive(Clone)]
pub struct Array<T> {
    pub(crate) data: Arc<NdArray<T, IxDyn>>,
}

impl<T: Clone> Array<T> {
    /// Create a new array from an ndarray
    pub fn from_ndarray(array: NdArray<T, IxDyn>) -> Self {
        Self::from_nd(array)
    }

    /// Wrap an owned `ndarray` buffer into a fresh, uniquely-owned `Array`.
    ///
    /// This is the crate-internal construction shim: every site that used to
    /// write the struct literal `Array { data: nd }` calls this instead, so
    /// there is exactly one place where the `Arc` is created and no site can
    /// accidentally store an `Arc::clone` where a fresh buffer was meant.
    pub(crate) fn from_nd(nd: NdArray<T, IxDyn>) -> Self {
        Self { data: Arc::new(nd) }
    }

    /// Shared (read-only) access to the underlying `ndarray` buffer.
    ///
    /// Identical to [`Array::array`]; it exists as the crate-internal spelling
    /// so that read sites can be explicit about going through the `Arc` rather
    /// than relying on auto-deref.
    pub(crate) fn nd(&self) -> &NdArray<T, IxDyn> {
        &self.data
    }

    /// Exclusive access to the underlying `ndarray` buffer, unsharing it first.
    ///
    /// **This is the crate's only copy-on-write unshare point** and holds the
    /// only `Arc::make_mut` call in `src/` (enforced by the policy grep in
    /// `scripts/ci-local.sh`). When this handle is the sole owner the call is
    /// a pointer comparison and returns the existing buffer; when the buffer
    /// is shared with another `Array`, it is deep-copied once and this handle
    /// takes sole ownership of the copy, leaving every other handle looking at
    /// the untouched original.
    pub(crate) fn nd_mut(&mut self) -> &mut NdArray<T, IxDyn> {
        Arc::make_mut(&mut self.data)
    }

    /// Consume this handle and yield the owned `ndarray` buffer.
    ///
    /// **This is the crate's only `Arc::try_unwrap` call.** It is O(1) when
    /// this handle is the sole owner (the common case: a temporary being
    /// reshaped or reinterpreted), and performs exactly one deep copy when the
    /// buffer is still shared with another `Array`. The copy is correct
    /// copy-on-write behaviour, not a leak or a missed optimization: the other
    /// handles must keep seeing the original buffer.
    pub(crate) fn into_nd(self) -> NdArray<T, IxDyn> {
        Arc::try_unwrap(self.data).unwrap_or_else(|arc| (*arc).clone())
    }

    /// Return `true` if this handle is the sole owner of its buffer.
    ///
    /// Under the copy-on-write storage a [`Clone`] is an O(1) reference-count
    /// bump, so two `Array`s can transiently share one buffer. This reports
    /// whether that is the case right now; it is purely informational, since
    /// any mutation unshares first and value semantics hold either way.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array::Array;
    ///
    /// let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
    /// assert!(a.is_unique());
    ///
    /// let b = a.clone();
    /// assert!(!a.is_unique());
    /// assert!(!b.is_unique());
    ///
    /// drop(b);
    /// assert!(a.is_unique());
    /// ```
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Get reference to the underlying ndarray
    pub fn array(&self) -> &NdArray<T, IxDyn> {
        self.nd()
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
    ///
    /// Unshares the buffer first (see [`Array::nd_mut`]): if this array
    /// currently shares storage with a clone, the buffer is deep-copied once
    /// so the write cannot be observed through the other handle.
    pub fn array_mut(&mut self) -> &mut NdArray<T, IxDyn> {
        self.nd_mut()
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

        // Check if indices are within bounds.
        //
        // Reads `self.data.shape()` -- a zero-allocation `&[usize]` borrow
        // of the underlying `ndarray`'s dimensions -- rather than calling
        // `self.shape()` per index, which heap-allocates a fresh `Vec` via
        // `to_vec()` on *every* call. `set()` is on the crate's hottest
        // per-element path (see `Array::nd_mut`'s doc comment), so that
        // allocation was a bigger tax than the `Arc::make_mut` unshare
        // check it sits next to. Values are identical either way.
        let shape = self.data.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, i, shape[i]
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

    /// Return the data as a flat vector, in logical (row-major/C) order.
    ///
    /// For a standard-layout (C-contiguous) array this is a fast path that
    /// hands back the underlying memory buffer directly. For a
    /// non-standard-layout array (e.g. produced by [`Array::to_f_layout`],
    /// `reversed_axes`, or a permuted-axes view) the raw memory buffer is
    /// *not* in row-major order for the array's current shape, so it is
    /// walked element-by-element via strides instead. Using the raw buffer
    /// unconditionally would silently return elements in the wrong order
    /// for any such array.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if self.data.is_standard_layout() {
            // Deep-clones the ndarray (not the Arc) so the raw buffer can be
            // taken by value, exactly as before the Arc-backed storage landed.
            let (raw_vec, _) = self.array().clone().into_raw_vec_and_offset();
            raw_vec
        } else {
            self.data.iter().cloned().collect()
        }
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
    ///
    /// Unshares the buffer first (see [`Array::nd_mut`]), so writes through
    /// the returned slice are never visible through a clone of this array.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.nd_mut().as_slice_mut()
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
    ///
    /// When the array is already C-contiguous this is an O(1) `Arc` bump:
    /// the returned array shares this one's buffer until either side is
    /// mutated, at which point copy-on-write unshares it.
    pub fn to_c_layout(&self) -> Self {
        if self.is_c_contiguous() {
            self.clone()
        } else {
            // Convert to standard layout
            let standard = self.data.as_standard_layout();
            Self::from_nd(standard.into_owned())
        }
    }

    /// Convert array to Fortran layout (column-major)
    ///
    /// When the array is already F-contiguous this is an O(1) `Arc` bump
    /// (see [`Array::to_c_layout`]).
    pub fn to_f_layout(&self) -> Self {
        if self.is_f_contiguous() {
            self.clone()
        } else {
            // For Fortran layout, we need to transpose all dimensions
            // This is a simplified implementation
            let transposed = self.array().clone().reversed_axes();
            Self::from_nd(transposed)
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

/// Copy-on-write invariants of the `Arc`-backed storage.
///
/// These are the white-box counterparts to `tests/test_cow_semantics.rs`:
/// they assert on [`Array::is_unique`], i.e. on *whether* the buffer is
/// currently shared, which the black-box tests cannot see. Together they
/// pin both halves of the contract -- sharing really happens (so `clone` is
/// genuinely O(1)) and sharing is never observable (so value semantics are
/// unchanged).
#[cfg(test)]
mod cow_tests {
    use super::Array;

    fn sample() -> Array<f64> {
        Array::from_vec(vec![1.0, 2.0, 3.0, 4.0])
    }

    #[test]
    fn a_fresh_array_is_uniquely_owned() {
        let a = sample();
        assert!(a.is_unique(), "a freshly built array owns its buffer alone");
    }

    #[test]
    fn cloning_shares_the_buffer_with_both_handles() {
        let a = sample();
        let b = a.clone();

        assert!(!a.is_unique(), "the source now shares its buffer");
        assert!(!b.is_unique(), "the clone shares the source's buffer");
        assert!(
            std::ptr::eq(a.array(), b.array()),
            "clone must be an Arc bump, not a fresh allocation"
        );
    }

    #[test]
    fn set_unshares_the_writer_and_leaves_the_other_handle_untouched() {
        let mut a = sample();
        let b = a.clone();
        assert!(!a.is_unique() && !b.is_unique());

        a.set(&[0], 99.0).expect("index 0 is in bounds");

        assert!(a.is_unique(), "the writer unshared its buffer on the write");
        assert!(b.is_unique(), "the other handle is now the sole owner");
        assert!(
            !std::ptr::eq(a.array(), b.array()),
            "the two handles must no longer point at one buffer"
        );
        assert_eq!(a.to_vec(), vec![99.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            b.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0],
            "the untouched handle must still see the ORIGINAL values"
        );
    }

    #[test]
    fn array_mut_unshares_the_writer_and_leaves_the_other_handle_untouched() {
        let mut a = sample();
        let b = a.clone();

        a.array_mut()[[2]] = -7.0;

        assert!(a.is_unique());
        assert!(b.is_unique());
        assert_eq!(a.to_vec(), vec![1.0, 2.0, -7.0, 4.0]);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn array_mut_on_an_unshared_array_keeps_the_same_buffer() {
        let mut a = sample();
        let before = a.array().as_ptr();

        a.array_mut()[[0]] = 5.0;

        assert!(a.is_unique());
        assert_eq!(
            a.array().as_ptr(),
            before,
            "a uniquely-owned array must be mutated in place, never copied"
        );
    }

    #[test]
    fn dropping_one_clone_restores_uniqueness_to_the_survivor() {
        let a = sample();
        {
            let b = a.clone();
            assert!(!a.is_unique());
            assert!(!b.is_unique());
        }
        assert!(a.is_unique(), "the survivor is the sole owner again");
    }

    #[test]
    fn a_three_way_share_unshares_only_the_writer() {
        let mut a = sample();
        let b = a.clone();
        let c = a.clone();

        a.set(&[1], 42.0).expect("index 1 is in bounds");

        assert!(a.is_unique(), "the writer took its own copy");
        assert!(!b.is_unique(), "b and c still share the original buffer");
        assert!(!c.is_unique());
        assert_eq!(a.to_vec(), vec![1.0, 42.0, 3.0, 4.0]);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn into_nd_moves_the_buffer_when_unique_and_copies_when_shared() {
        // Sole owner: the allocation is handed over, not copied.
        let a = sample();
        let addr = a.array().as_ptr();
        let moved = a.into_nd();
        assert_eq!(
            moved.as_ptr(),
            addr,
            "into_nd must not copy a unique buffer"
        );

        // Shared: the surviving handle keeps the original, the extracted
        // buffer is a copy -- correct copy-on-write, not a bug.
        let b = sample();
        let c = b.clone();
        let extracted = b.into_nd();
        assert!(c.is_unique(), "c is the sole owner once b is consumed");
        assert_ne!(
            extracted.as_ptr(),
            c.array().as_ptr(),
            "a shared buffer must be copied out, leaving c's untouched"
        );
        assert_eq!(extracted.iter().cloned().collect::<Vec<_>>(), c.to_vec());
    }

    #[test]
    fn to_c_layout_shares_when_the_layout_already_matches() {
        let a = sample();
        let same = a.to_c_layout();
        assert!(
            std::ptr::eq(a.array(), same.array()),
            "an already-C-contiguous array must be returned as an O(1) share"
        );

        // ...and the share is still copy-on-write, not aliasing.
        let mut same = same;
        same.set(&[0], 0.0).expect("index 0 is in bounds");
        assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
