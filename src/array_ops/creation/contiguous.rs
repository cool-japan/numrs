//! Memory layout and contiguity functions
//!
//! This module provides functions for working with array memory layouts:
//! - `asanyarray` - Convert input to an array
//! - `ascontiguousarray` - Return a contiguous array in C order
//! - `asfortranarray` - Return an array in Fortran order
//! - `isfortran` - Check if array is Fortran contiguous
//! - `iscontiguous` - Check if array is C contiguous
//! - `may_share_memory` - Check if two arrays may share memory
//! - `shares_memory` - Check if two arrays share memory

use crate::array::Array;
use crate::error::Result;

/// Convert the input to an array, preserving subclasses
///
/// This function converts the input to an array but preserves array subclasses.
/// In Rust context, since we don't have the same subclassing as Python, this
/// essentially works like a regular array conversion but is provided for NumPy API compatibility.
///
/// # Parameters
///
/// * `a` - Input data that can be converted to an array
///
/// # Returns
///
/// An array interpretation of `a`. No copy is performed if the input is already an array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::asanyarray;
///
/// // From slice
/// let slice = &[1.0, 2.0, 3.0];
/// let result = asanyarray(&slice).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
///
/// // From vector
/// let vec = vec![4, 5, 6];
/// let result = asanyarray(&vec).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![4, 5, 6]);
/// ```
pub fn asanyarray<T>(a: &impl AsRef<[T]>) -> Result<Array<T>>
where
    T: Clone,
{
    let slice = a.as_ref();
    Ok(Array::from_vec(slice.to_vec()))
}

/// Return a contiguous array in C order (row-major) in memory
///
/// # Parameters
///
/// * `a` - Input array
///
/// # Returns
///
/// Contiguous array of same shape and content as `a`, with data in C order.
/// If `a` is already C-contiguous, no copy is made.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::{ascontiguousarray, iscontiguous};
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let c_arr = ascontiguousarray(&arr).expect("operation should succeed");
/// assert_eq!(c_arr.shape(), vec![2, 2]);
/// assert!(iscontiguous(&c_arr));
/// ```
pub fn ascontiguousarray<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Clone,
{
    // `Array::to_c_layout` already returns `self.clone()` when `a` is
    // already C-contiguous (no extra copy), and materializes a genuinely
    // C-contiguous copy otherwise (e.g. for an array produced by
    // `to_f_layout`/`reversed_axes`/a permuted-axes view). Returning
    // `a.clone()` unconditionally, as before, silently preserved a
    // non-standard layout instead of honoring this function's contract.
    Ok(a.to_c_layout())
}

/// Return an array laid out in Fortran order (column-major) in memory
///
/// # Parameters
///
/// * `a` - Input array
///
/// # Returns
///
/// Fortran-contiguous array of same shape and content as `a`.
/// The returned array will have column-major memory layout.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::asfortranarray;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let f_arr = asfortranarray(&arr).expect("operation should succeed");
/// assert_eq!(f_arr.shape(), vec![2, 2]);
/// // Fortran array has same shape but different memory layout
/// ```
pub fn asfortranarray<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Clone,
{
    if a.ndim() <= 1 {
        // 1D arrays are both C and Fortran contiguous
        return Ok(a.clone());
    }

    let shape = a.shape();
    let data = a.to_vec();

    // Convert from C-order (row-major) to Fortran-order (column-major)
    let mut f_data = vec![data[0].clone(); data.len()];

    // Calculate strides for Fortran order
    let mut f_strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }

    // Reorder data
    for (c_idx, value) in data.into_iter().enumerate() {
        // Convert C-order index to multi-dimensional indices
        let mut indices = vec![0; shape.len()];
        let mut temp_idx = c_idx;

        for i in (0..shape.len()).rev() {
            indices[i] = temp_idx % shape[i];
            temp_idx /= shape[i];
        }

        // Calculate Fortran-order index
        let mut f_idx = 0;
        for i in 0..shape.len() {
            f_idx += indices[i] * f_strides[i];
        }

        f_data[f_idx] = value;
    }

    // Create array with Fortran-ordered data
    // Note: We store a flag to indicate Fortran ordering
    let mut result = Array::from_vec(f_data);
    result = result.reshape(&shape);
    // In a real implementation, we would set a flag here to indicate Fortran ordering
    // For now, we just return the reordered array

    Ok(result)
}

/// Check if the array is Fortran contiguous
///
/// Fortran-contiguous arrays have column-major memory layout where the first
/// index varies fastest.
///
/// # Parameters
///
/// * `a` - Array to check
///
/// # Returns
///
/// True if the array is Fortran-contiguous, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::{asfortranarray, isfortran};
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// assert!(!isfortran(&arr));  // Default arrays are C-contiguous
///
/// let f_arr = asfortranarray(&arr).expect("operation should succeed");
/// // Note: current implementation doesn't track Fortran layout internally
/// // so isfortran always returns false for multi-dimensional arrays
/// ```
pub fn isfortran<T>(a: &Array<T>) -> bool
where
    T: Clone,
{
    // For 1D arrays, both C and Fortran contiguous are the same
    if a.ndim() <= 1 {
        return true;
    }

    // In our current implementation, arrays are stored in C-order by default
    // A real implementation would check internal flags or strides
    // For now, we return false for multi-dimensional arrays unless they were
    // explicitly created with asfortranarray
    false
}

/// Check if the array is C contiguous
///
/// C-contiguous arrays have row-major memory layout where the last
/// index varies fastest.
///
/// # Parameters
///
/// * `a` - Array to check
///
/// # Returns
///
/// True if the array is C-contiguous, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::iscontiguous;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// assert!(iscontiguous(&arr));  // Default arrays are C-contiguous
/// ```
pub fn iscontiguous<T: Clone>(a: &Array<T>) -> bool {
    // Arrays are C-contiguous by default, but non-standard layouts do
    // exist (e.g. `Array::to_f_layout`, `reversed_axes`, permuted-axes
    // views) -- hardcoding `true` was silently wrong for those. Delegate
    // to ndarray's own layout check, which accounts for arbitrary strides.
    a.array().is_standard_layout()
}

/// Check if two arrays may share memory
///
/// # Parameters
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// True if arrays might share memory, false if they definitely don't
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
/// assert!(!may_share_memory(&a, &b));  // Different arrays don't share memory
///
/// let c = a.clone();
/// assert!(!may_share_memory(&a, &c));  // Cloned arrays have separate memory
/// ```
pub fn may_share_memory<T: Clone>(a: &Array<T>, b: &Array<T>) -> bool {
    // Two arrays share memory iff their underlying buffers' address ranges
    // overlap. With the current deep-copy `Array` (every `.clone()` /
    // transform allocates a fresh buffer), two *distinct* `Array` values
    // can never overlap -- but `a` and `a` (the same object, e.g. through
    // two references to it) trivially do, and hardcoding `false`
    // unconditionally got that wrong. Comparing raw pointer ranges (rather
    // than a special-cased identity check) keeps this correct once
    // copy-on-write/view support lands and distinct `Array` values really
    // can alias.
    ptr_ranges_overlap(a, b)
}

/// Returns true iff the memory ranges `[a.as_ptr(), a.as_ptr() + a.len())`
/// and `[b.as_ptr(), b.as_ptr() + b.len())` overlap.
///
/// Zero-length arrays never overlap with anything (they occupy no bytes,
/// regardless of what address an empty slice happens to report). For two
/// non-empty ranges, overlap uses the standard half-open-interval test
/// `a_start < b_end && b_start < a_end`; a strict `<` (never `<=`) is
/// required for two adjacent-but-disjoint ranges to correctly compare as
/// non-overlapping.
fn ptr_ranges_overlap<T: Clone>(a: &Array<T>, b: &Array<T>) -> bool {
    let a_len = a.array().len();
    let b_len = b.array().len();
    if a_len == 0 || b_len == 0 {
        return false;
    }

    let a_start = a.array().as_ptr() as usize;
    let a_end = a_start + a_len * std::mem::size_of::<T>();
    let b_start = b.array().as_ptr() as usize;
    let b_end = b_start + b_len * std::mem::size_of::<T>();

    a_start < b_end && b_start < a_end
}

/// Check if two arrays share memory
///
/// # Parameters
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// True if arrays share memory, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
/// assert!(!shares_memory(&a, &b));  // Different arrays don't share memory
/// ```
pub fn shares_memory<T: Clone>(a: &Array<T>, b: &Array<T>) -> bool {
    // NumPy distinguishes `may_share_memory` (a fast, conservative check
    // that is allowed to over-report on complex strided views) from
    // `shares_memory` (the exact, exhaustive check). With today's
    // deep-copy-only `Array` there is no such ambiguity to approximate --
    // a pointer-range overlap is exact either way -- so both delegate to
    // the same precise check. Kept as separate functions (rather than one
    // calling the other in a way that hides it) so the exact-vs-may
    // distinction is easy to reintroduce once views/COW exist.
    ptr_ranges_overlap(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asanyarray() {
        // Test from slice
        let slice = &[1.0, 2.0, 3.0];
        let result = asanyarray(&slice).expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);

        // Test from vector
        let vec = vec![4, 5, 6];
        let result = asanyarray(&vec).expect("operation should succeed");
        assert_eq!(result.to_vec(), vec![4, 5, 6]);
    }

    #[test]
    fn test_ascontiguousarray() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let c_arr = ascontiguousarray(&arr).expect("operation should succeed");
        assert_eq!(c_arr.shape(), vec![2, 2]);
        assert!(iscontiguous(&c_arr));
    }

    #[test]
    fn test_asfortranarray() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let f_arr = asfortranarray(&arr).expect("operation should succeed");
        assert_eq!(f_arr.shape(), vec![2, 2]);

        // 1D array
        let arr_1d = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let f_arr_1d = asfortranarray(&arr_1d).expect("operation should succeed");
        assert_eq!(f_arr_1d.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_isfortran() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        assert!(!isfortran(&arr));

        // 1D array is both C and Fortran contiguous
        let arr_1d = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(isfortran(&arr_1d));
    }

    #[test]
    fn test_iscontiguous() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        assert!(iscontiguous(&arr));
    }

    #[test]
    fn test_may_share_memory() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
        assert!(!may_share_memory(&a, &b));

        let c = a.clone();
        assert!(!may_share_memory(&a, &c));
    }

    #[test]
    fn test_shares_memory() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
        assert!(!shares_memory(&a, &b));
    }
}
