use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels;
use num_traits::{Float, Zero};
use std::borrow::Cow;
use std::fmt::Debug;

/// Broadcast `a` to `shape`, without cloning/copying when `a` is already
/// that shape -- the common case, since most comparisons are between
/// equal-shaped arrays. [`Cow::Borrowed`] aliases `a` directly; only a
/// genuine shape mismatch pays for [`Array::broadcast_to`]'s copy.
///
/// Used by the free comparison functions below (`greater`, `equal`,
/// `logical_and`, ...), which -- unlike the [`Array`] methods in
/// `comparisons_broadcast.rs` -- do their own broadcast-shape handling
/// instead of going through [`Array::broadcast_op`] (which already has an
/// equal-shape fast path of its own).
pub(crate) fn maybe_broadcast<'a, T: Clone>(
    a: &'a Array<T>,
    shape: &[usize],
) -> Result<Cow<'a, Array<T>>> {
    if a.shape() == shape {
        Ok(Cow::Borrowed(a))
    } else {
        Ok(Cow::Owned(a.broadcast_to(shape)?))
    }
}

/// Comparison utilities for NumRS Arrays
/// Determine if two arrays are element-wise equal within a tolerance
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - The relative tolerance parameter (default: 1e-7)
/// * `atol` - The absolute tolerance parameter (default: 0)
///
/// # Returns
///
/// `true` if the arrays are equal within the given tolerance; `false` otherwise
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![1.0000001, 2.0000002, 3.0000003]);
///
/// // Default tolerances (rtol=1e-7, atol=0)
/// assert!(allclose(&a, &b));
///
/// // Custom tolerances (should still be close enough)
/// assert!(allclose_with_tol(&a, &b, 1e-6, 0.0));
/// ```
pub fn allclose<T>(a: &Array<T>, b: &Array<T>) -> bool
where
    T: Clone + Float + Debug,
{
    allclose_with_tol(
        a,
        b,
        T::from(1e-7).expect("Failed to convert 1e-7 to type T"),
        T::zero(),
    )
}

/// Determine if two arrays are element-wise equal within specified tolerances
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - The relative tolerance parameter
/// * `atol` - The absolute tolerance parameter
///
/// # Returns
///
/// `true` if the arrays are equal within the given tolerance; `false` otherwise
pub fn allclose_with_tol<T>(a: &Array<T>, b: &Array<T>, rtol: T, atol: T) -> bool
where
    T: Clone + Float + Debug,
{
    // Check if shapes are the same (no broadcasting here -- unchanged from
    // before: a shape mismatch is simply "not close", not an error).
    if a.shape() != b.shape() {
        return false;
    }

    // Zero-copy (when contiguous) slice access instead of two `to_vec()`
    // clones; short-circuits on the first mismatch via `Iterator::all`
    // instead of materializing a `Vec<bool>` that was never needed.
    let a_op = kernels::borrow::operand(a);
    let b_op = kernels::borrow::operand(b);
    a_op.iter()
        .zip(b_op.iter())
        .all(|(&a_val, &b_val)| isclose(a_val, b_val, rtol, atol))
}

/// Determine if two floating point values are equal within a tolerance
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `rtol` - The relative tolerance parameter
/// * `atol` - The absolute tolerance parameter
///
/// # Returns
///
/// `true` if the values are equal within the given tolerance; `false` otherwise
pub fn isclose<T>(a: T, b: T, rtol: T, atol: T) -> bool
where
    T: Clone + Float + Debug,
{
    // Check for exact equality first (handles infinity cases)
    if a == b {
        return true;
    }

    // Check if both values are NaN (NaN != NaN)
    if a.is_nan() && b.is_nan() {
        return true;
    }

    // Calculate the tolerance
    let tol = atol + rtol * b.abs();

    // Check if values are close
    (a - b).abs() <= tol
}

/// Determine if two arrays have the same shape and elements
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// `true` if the arrays have the same shape and elements; `false` otherwise
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![1, 2, 3]);
/// let c = Array::from_vec(vec![1, 2, 4]);
///
/// assert!(array_equal(&a, &b, None));
/// assert!(!array_equal(&a, &c, None));
/// ```
/// Check if two arrays are equal (element-wise)
///
/// # Parameters
///
/// * `a` - First array to compare
/// * `b` - Second array to compare
/// * `equal_nan` - If True, treat NaN elements as equal to each other (floating point arrays only)
///
/// # Returns
///
/// * `true` if arrays are equal, `false` otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create two arrays
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![1, 2, 3]);
/// let c = Array::from_vec(vec![1, 2, 4]);
///
/// // Compare arrays
/// assert!(array_equal(&a, &b, None));
/// assert!(!array_equal(&a, &c, None));
///
/// // Compare with NaN handling (floating point only)
/// let d = Array::from_vec(vec![1.0, 2.0, f64::NAN]);
/// let e = Array::from_vec(vec![1.0, 2.0, f64::NAN]);
/// assert!(!array_equal(&d, &e, None)); // Default behavior: NaNs are not equal
/// assert!(array_equal(&d, &e, Some(true))); // With equal_nan=true, NaNs are equal
/// ```
pub fn array_equal<T>(a: &Array<T>, b: &Array<T>, equal_nan: Option<bool>) -> bool
where
    T: Clone + PartialEq + Debug + 'static,
{
    let equal_nan = equal_nan.unwrap_or(false);

    // Check if shapes are the same
    if a.shape() != b.shape() {
        return false;
    }

    // For floating point types, handle NaN equality if requested
    if equal_nan {
        if let Some(result) = array_equal_with_nan_handling(a, b) {
            return result;
        }
    }

    // Regular element-wise comparison (handled by PartialEq)
    a.to_vec() == b.to_vec()
}

/// Helper method specifically for arrays with floating point types to handle NaN equality
fn array_equal_with_nan_handling<T>(a: &Array<T>, b: &Array<T>) -> Option<bool>
where
    T: Clone + PartialEq + Debug + 'static,
{
    // Handle f32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let a_f32 = unsafe { &*(a as *const Array<T> as *const Array<f32>) };
        let b_f32 = unsafe { &*(b as *const Array<T> as *const Array<f32>) };

        let a_vec = a_f32.to_vec();
        let b_vec = b_f32.to_vec();

        if a_vec.len() != b_vec.len() {
            return Some(false);
        }

        for i in 0..a_vec.len() {
            if a_vec[i] != b_vec[i] {
                // If both values are NaN, consider them equal
                if a_vec[i].is_nan() && b_vec[i].is_nan() {
                    continue;
                }
                return Some(false);
            }
        }

        return Some(true);
    }

    // Handle f64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let a_f64 = unsafe { &*(a as *const Array<T> as *const Array<f64>) };
        let b_f64 = unsafe { &*(b as *const Array<T> as *const Array<f64>) };

        let a_vec = a_f64.to_vec();
        let b_vec = b_f64.to_vec();

        if a_vec.len() != b_vec.len() {
            return Some(false);
        }

        for i in 0..a_vec.len() {
            if a_vec[i] != b_vec[i] {
                // If both values are NaN, consider them equal
                if a_vec[i].is_nan() && b_vec[i].is_nan() {
                    continue;
                }
                return Some(false);
            }
        }

        return Some(true);
    }

    // Not a floating point type
    None
}

/// Comprehensive array comparison that supports broadcasting and custom options
///
/// # Parameters
///
/// * `a` - First array to compare
/// * `b` - Second array to compare
/// * `options` - Comparison options
///
/// # Returns
///
/// * `true` if arrays satisfy the comparison criteria, `false` otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::comparisons::{array_compare, ArrayCompareOptions};
///
/// // Create some arrays
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![1, 2, 3]);
/// let c = Array::from_vec(vec![1, 2, 4]);
///
/// // Basic equality comparison
/// let opts = ArrayCompareOptions::default();
/// assert!(array_compare(&a, &b, &opts));
/// assert!(!array_compare(&a, &c, &opts));
///
/// // Ignore a specific index
/// let mut opts = ArrayCompareOptions::default();
/// opts.ignore_indices = Some(vec![2]);
/// assert!(array_compare(&a, &c, &opts)); // Ignores the difference at index 2
///
/// // Compare with broadcasting (1D to 2D)
/// let d = Array::from_vec(vec![1, 2, 3]).reshape(&[3, 1]);
/// let e = Array::from_vec(vec![1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(&[3, 3]);
/// let mut opts = ArrayCompareOptions::default();
/// opts.allow_broadcasting = true;
/// assert!(array_compare(&d, &e, &opts)); // d is broadcast across columns
/// ```
pub fn array_compare<T>(a: &Array<T>, b: &Array<T>, options: &ArrayCompareOptions) -> bool
where
    T: Clone + PartialEq + Debug + 'static,
{
    // If shapes are equal, we can do direct comparison
    if a.shape() == b.shape() {
        return array_compare_equal_shapes(a, b, options);
    }

    // If broadcasting is allowed, try broadcasting before comparison
    if options.allow_broadcasting {
        if let Ok(broadcast_arrays) = crate::stride_tricks::broadcast_arrays(&[a, b]) {
            return array_compare_equal_shapes(&broadcast_arrays[0], &broadcast_arrays[1], options);
        }
    }

    // Shapes are different and broadcasting failed or is not allowed
    false
}

/// Helper function for comparing arrays of the same shape
fn array_compare_equal_shapes<T>(a: &Array<T>, b: &Array<T>, options: &ArrayCompareOptions) -> bool
where
    T: Clone + PartialEq + Debug + 'static,
{
    debug_assert_eq!(a.shape(), b.shape(), "Arrays must have the same shape");

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    // Prepare a mask of indices to ignore (if any)
    let mut ignore_mask = vec![false; a_vec.len()];
    if let Some(indices) = &options.ignore_indices {
        for &idx in indices {
            if idx < ignore_mask.len() {
                ignore_mask[idx] = true;
            }
        }
    }

    // For floating point types, handle NaN equality if requested
    if options.equal_nan {
        if let Some(result) = array_compare_with_nan_handling(a, b, &ignore_mask) {
            return result;
        }
    }

    // Regular comparison with ignore mask
    for i in 0..a_vec.len() {
        if ignore_mask[i] {
            continue; // Skip indices that should be ignored
        }

        if a_vec[i] != b_vec[i] {
            return false;
        }
    }

    true
}

/// Helper method for floating point comparisons with NaN handling
fn array_compare_with_nan_handling<T>(
    a: &Array<T>,
    b: &Array<T>,
    ignore_mask: &[bool],
) -> Option<bool>
where
    T: Clone + PartialEq + Debug + 'static,
{
    // Handle f32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let a_f32 = unsafe { &*(a as *const Array<T> as *const Array<f32>) };
        let b_f32 = unsafe { &*(b as *const Array<T> as *const Array<f32>) };

        let a_vec = a_f32.to_vec();
        let b_vec = b_f32.to_vec();

        for i in 0..a_vec.len() {
            if ignore_mask[i] {
                continue;
            }

            if a_vec[i] != b_vec[i] {
                // If both values are NaN, consider them equal
                if a_vec[i].is_nan() && b_vec[i].is_nan() {
                    continue;
                }
                return Some(false);
            }
        }

        return Some(true);
    }

    // Handle f64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let a_f64 = unsafe { &*(a as *const Array<T> as *const Array<f64>) };
        let b_f64 = unsafe { &*(b as *const Array<T> as *const Array<f64>) };

        let a_vec = a_f64.to_vec();
        let b_vec = b_f64.to_vec();

        for i in 0..a_vec.len() {
            if ignore_mask[i] {
                continue;
            }

            if a_vec[i] != b_vec[i] {
                // If both values are NaN, consider them equal
                if a_vec[i].is_nan() && b_vec[i].is_nan() {
                    continue;
                }
                return Some(false);
            }
        }

        return Some(true);
    }

    // Not a floating point type
    None
}

/// Options for controlling array comparisons
#[derive(Debug, Clone, Default)]
pub struct ArrayCompareOptions {
    /// Treat NaN values as equal
    pub equal_nan: bool,

    /// Allow broadcasting of arrays to compatible shapes
    pub allow_broadcasting: bool,

    /// Specific indices to ignore during comparison (flattened indices)
    pub ignore_indices: Option<Vec<usize>>,

    /// For numerical types, tolerance for considering values equal
    pub rtol: Option<f64>,

    /// For numerical types, absolute tolerance for considering values equal
    pub atol: Option<f64>,
}

/// Determine if all elements in an array evaluate to True
///
/// # Arguments
///
/// * `a` - Input array
///
/// # Returns
///
/// `true` if all elements evaluate to True; `false` otherwise
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![true, true, true]);
/// let b = Array::from_vec(vec![true, false, true]);
///
/// assert!(all(&a));
/// assert!(!all(&b));
/// ```
pub fn all<T>(a: &Array<T>) -> bool
where
    T: Clone + PartialEq + Debug,
    bool: From<T>,
{
    // Direct iteration, no `to_vec()` buffer -- short-circuits on the
    // first `false` instead of materializing a full copy first.
    a.array().iter().all(|val| bool::from(val.clone()))
}

/// Determine if any element in an array evaluates to True
///
/// # Arguments
///
/// * `a` - Input array
///
/// # Returns
///
/// `true` if any element evaluates to True; `false` otherwise
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![false, false, false]);
/// let b = Array::from_vec(vec![false, true, false]);
///
/// assert!(!any(&a));
/// assert!(any(&b));
/// ```
pub fn any<T>(a: &Array<T>) -> bool
where
    T: Clone + PartialEq + Debug,
    bool: From<T>,
{
    // Direct iteration, no `to_vec()` buffer -- short-circuits on the
    // first `true` instead of materializing a full copy first.
    a.array().iter().any(|val| bool::from(val.clone()))
}

/// Create a boolean array with element-wise comparison (a > b)
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A boolean array with elements set to `true` where a > b
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![1, 2, 3]);
///     let b = Array::from_vec(vec![0, 2, 4]);
///
///     let result = greater(&a, &b)?;
///     assert_eq!(result.to_vec(), vec![true, false, false]);
///     Ok(())
/// }
/// ```
pub fn greater<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialOrd + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    // Broadcast only when actually needed (`maybe_broadcast` borrows
    // instead of cloning in the equal-shape case), then compare via a
    // single zero-copy zip pass instead of two `to_vec()` copies.
    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x > y);

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Create a boolean array with element-wise comparison (a >= b)
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A boolean array with elements set to `true` where a >= b
pub fn greater_equal<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialOrd + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x >= y);

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Create a boolean array with element-wise comparison (a < b)
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A boolean array with elements set to `true` where a < b
pub fn less<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialOrd + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x < y);

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Create a boolean array with element-wise comparison (a <= b)
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A boolean array with elements set to `true` where a <= b
pub fn less_equal<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialOrd + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x <= y);

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Create a boolean array with element-wise comparison (a == b)
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A boolean array with elements set to `true` where a == b
pub fn equal<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialEq + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x == y);

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Create a boolean array with element-wise comparison (a != b)
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// A boolean array with elements set to `true` where a != b
pub fn not_equal<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialEq + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x != y);

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Check if two arrays are approximately equal with the given tolerances
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - The relative tolerance parameter
/// * `atol` - The absolute tolerance parameter
///
/// # Returns
///
/// A boolean array with elements set to `true` where elements are approximately equal
pub fn isclose_array<T>(a: &Array<T>, b: &Array<T>, rtol: T, atol: T) -> Result<Array<bool>>
where
    T: Clone + Float + Debug,
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape()).map_err(|_| {
        NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        }
    })?;

    let a_broadcast = maybe_broadcast(a, &broadcast_shape)?;
    let b_broadcast = maybe_broadcast(b, &broadcast_shape)?;

    let a_op = kernels::borrow::operand(&a_broadcast);
    let b_op = kernels::borrow::operand(&b_broadcast);
    let result =
        kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| isclose(x, y, rtol, atol));

    Array::from_vec_shape(result, &broadcast_shape)
}

/// Element-wise logical AND of two arrays
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Boolean array with the same shape as the inputs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![true, true, false, false]);
///     let b = Array::from_vec(vec![true, false, true, false]);
///     let result = logical_and(&a, &b)?;
///     assert_eq!(result.to_vec(), vec![true, false, false, false]);
///     Ok(())
/// }
/// ```
pub fn logical_and(x1: &Array<bool>, x2: &Array<bool>) -> Result<Array<bool>> {
    // Broadcast to a common shape -- `maybe_broadcast` borrows instead of
    // calling `broadcast_to` (an unconditional `to_owned()` copy) when the
    // shapes already match, and the zip below reads through zero-copy
    // slices instead of two more `to_vec()` copies.
    let broadcast_shape = Array::<bool>::broadcast_shape(&x1.shape(), &x2.shape())?;
    let x1_broadcast = maybe_broadcast(x1, &broadcast_shape)?;
    let x2_broadcast = maybe_broadcast(x2, &broadcast_shape)?;

    let x1_op = kernels::borrow::operand(&x1_broadcast);
    let x2_op = kernels::borrow::operand(&x2_broadcast);
    let result_data = kernels::elementwise::binary_serial(&x1_op, &x2_op, |a, b| a && b);

    Array::from_vec_shape(result_data, &broadcast_shape)
}

/// Element-wise logical OR of two arrays
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Boolean array with the same shape as the inputs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![true, true, false, false]);
///     let b = Array::from_vec(vec![true, false, true, false]);
///     let result = logical_or(&a, &b)?;
///     assert_eq!(result.to_vec(), vec![true, true, true, false]);
///     Ok(())
/// }
/// ```
pub fn logical_or(x1: &Array<bool>, x2: &Array<bool>) -> Result<Array<bool>> {
    let broadcast_shape = Array::<bool>::broadcast_shape(&x1.shape(), &x2.shape())?;
    let x1_broadcast = maybe_broadcast(x1, &broadcast_shape)?;
    let x2_broadcast = maybe_broadcast(x2, &broadcast_shape)?;

    let x1_op = kernels::borrow::operand(&x1_broadcast);
    let x2_op = kernels::borrow::operand(&x2_broadcast);
    let result_data = kernels::elementwise::binary_serial(&x1_op, &x2_op, |a, b| a || b);

    Array::from_vec_shape(result_data, &broadcast_shape)
}

/// Element-wise logical NOT of an array
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Boolean array with the same shape as the input
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![true, false, true, false]);
///     let result = logical_not(&a)?;
///     assert_eq!(result.to_vec(), vec![false, true, false, true]);
///     Ok(())
/// }
/// ```
pub fn logical_not(x: &Array<bool>) -> Result<Array<bool>> {
    // `map` already takes the zero-copy contiguous-slice fast path
    // internally (see `Array::map`); no broadcast is needed for a unary op.
    Ok(x.map(|a| !a))
}

/// Element-wise logical XOR of two arrays
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Boolean array with the same shape as the inputs
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![true, true, false, false]);
///     let b = Array::from_vec(vec![true, false, true, false]);
///     let result = logical_xor(&a, &b)?;
///     assert_eq!(result.to_vec(), vec![false, true, true, false]);
///     Ok(())
/// }
/// ```
pub fn logical_xor(x1: &Array<bool>, x2: &Array<bool>) -> Result<Array<bool>> {
    let broadcast_shape = Array::<bool>::broadcast_shape(&x1.shape(), &x2.shape())?;
    let x1_broadcast = maybe_broadcast(x1, &broadcast_shape)?;
    let x2_broadcast = maybe_broadcast(x2, &broadcast_shape)?;

    let x1_op = kernels::borrow::operand(&x1_broadcast);
    let x2_op = kernels::borrow::operand(&x2_broadcast);
    let result_data = kernels::elementwise::binary_serial(&x1_op, &x2_op, |a, b| a ^ b);

    Array::from_vec_shape(result_data, &broadcast_shape)
}

/// Count the number of non-zero values in the array
///
/// # Parameters
///
/// * `a` - Input array
/// * `axis` - If None, count over the flattened array. If Some(axis), count along the specified axis.
///
/// # Returns
///
/// Number of non-zero values as a scalar or array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::comparisons::count_nonzero;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![0, 1, 0, 3, 0, 5]);
///     assert_eq!(count_nonzero(&a, None)?.to_vec()[0], 3);
///
///     let b = Array::from_vec(vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0]).reshape(&[2, 3]);
///     // Count over all elements
///     assert_eq!(count_nonzero(&b, None)?.to_vec()[0], 3);
///
///     // Count along axis 0 (columns)
///     let c = count_nonzero(&b, Some(0))?;
///     assert_eq!(c.to_vec(), vec![1, 1, 1]);
///
///     // Count along axis 1 (rows)
///     let d = count_nonzero(&b, Some(1))?;
///     assert_eq!(d.to_vec(), vec![1, 2]);
///     Ok(())
/// }
/// ```
pub fn count_nonzero<T>(a: &Array<T>, axis: Option<usize>) -> Result<Array<usize>>
where
    T: Clone + Zero + PartialEq,
{
    if let Some(ax) = axis {
        if ax >= a.ndim() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax,
                a.ndim()
            )));
        }

        // Count along specific axis
        let shape = a.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_size = shape[ax];
        let stride_before: usize = shape[..ax].iter().product();
        let stride_after: usize = shape[ax + 1..].iter().product();
        let total_size = stride_before * stride_after;

        let mut counts = vec![0usize; total_size];
        let data = a.to_vec();

        for i in 0..stride_before {
            for j in 0..axis_size {
                for k in 0..stride_after {
                    let idx = i * axis_size * stride_after + j * stride_after + k;
                    let out_idx = i * stride_after + k;
                    if data[idx] != T::zero() {
                        counts[out_idx] += 1;
                    }
                }
            }
        }

        Ok(Array::from_vec_shape(counts, &new_shape)?)
    } else {
        // Count over flattened array
        let count = a.to_vec().into_iter().filter(|x| *x != T::zero()).count();
        Ok(Array::from_vec(vec![count]))
    }
}

/// Return indices that are non-zero in the flattened version of the array
///
/// This is equivalent to calling nonzero(a.ravel()) and returning only the first element
/// of the tuple.
///
/// # Parameters
///
/// * `a` - Input array
///
/// # Returns
///
/// Array of indices of non-zero elements in the flattened array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::comparisons::flatnonzero;
/// use numrs2::error::Result;
///
/// fn main() -> Result<()> {
///     let a = Array::from_vec(vec![0, 1, 0, 3, 0, 5]);
///     let indices = flatnonzero(&a)?;
///     assert_eq!(indices.to_vec(), vec![1, 3, 5]);
///
///     let b = Array::from_vec(vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0]).reshape(&[2, 3]);
///     let indices = flatnonzero(&b)?;
///     assert_eq!(indices.to_vec(), vec![1, 3, 5]);
///     Ok(())
/// }
/// ```
pub fn flatnonzero<T>(a: &Array<T>) -> Result<Array<usize>>
where
    T: Clone + Zero + PartialEq,
{
    let data = a.to_vec();
    let indices: Vec<usize> = data
        .into_iter()
        .enumerate()
        .filter_map(|(idx, val)| if val != T::zero() { Some(idx) } else { None })
        .collect();

    Ok(Array::from_vec(indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allclose() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0000001, 2.0000002, 3.0000003]);
        let c = Array::from_vec(vec![1.001, 2.002, 3.003]);

        // Default tolerances (rtol=1e-7, atol=0)
        assert!(allclose(&a, &b));
        assert!(!allclose(&a, &c));

        // Custom tolerances
        assert!(allclose_with_tol(&a, &c, 1e-2, 0.0));
    }

    #[test]
    fn test_array_equal() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 3]);
        let c = Array::from_vec(vec![1, 2, 4]);

        assert!(array_equal(&a, &b, None));
        assert!(!array_equal(&a, &c, None));

        // Different shapes
        let d = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        assert!(!array_equal(&a, &d, None));
    }

    #[test]
    fn test_isclose() {
        assert!(isclose(1.0, 1.0000001, 1e-7, 0.0));
        assert!(!isclose(1.0, 1.001, 1e-7, 0.0));

        // Test NaN handling
        assert!(isclose(f64::NAN, f64::NAN, 1e-7, 0.0));

        // Test infinity handling
        assert!(isclose(f64::INFINITY, f64::INFINITY, 1e-7, 0.0));
        assert!(!isclose(f64::INFINITY, 1.0, 1e-7, 0.0));
    }

    #[test]
    fn test_all_any() {
        let all_true = Array::from_vec(vec![true, true, true]);
        let mixed = Array::from_vec(vec![true, false, true]);
        let all_false = Array::from_vec(vec![false, false, false]);

        assert!(all(&all_true));
        assert!(!all(&mixed));
        assert!(!all(&all_false));

        assert!(any(&all_true));
        assert!(any(&mixed));
        assert!(!any(&all_false));
    }

    #[test]
    fn test_comparison_ops() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![0, 2, 4]);

        // Test greater
        let result = greater(&a, &b).expect("greater comparison should succeed");
        assert_eq!(result.to_vec(), vec![true, false, false]);

        // Test greater_equal
        let result = greater_equal(&a, &b).expect("greater_equal comparison should succeed");
        assert_eq!(result.to_vec(), vec![true, true, false]);

        // Test less
        let result = less(&a, &b).expect("less comparison should succeed");
        assert_eq!(result.to_vec(), vec![false, false, true]);

        // Test less_equal
        let result = less_equal(&a, &b).expect("less_equal comparison should succeed");
        assert_eq!(result.to_vec(), vec![false, true, true]);

        // Test equal
        let result = equal(&a, &b).expect("equal comparison should succeed");
        assert_eq!(result.to_vec(), vec![false, true, false]);

        // Test not_equal
        let result = not_equal(&a, &b).expect("not_equal comparison should succeed");
        assert_eq!(result.to_vec(), vec![true, false, true]);
    }

    #[test]
    fn test_broadcasting() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1]).reshape(&[1]);

        // Test broadcasting
        let result = equal(&a, &b).expect("broadcast equal should succeed");
        assert_eq!(result.to_vec(), vec![true, false, false]);

        // Test with 2D arrays
        let c = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let d = Array::from_vec(vec![1, 2]).reshape(&[1, 2]);

        let result = equal(&c, &d).expect("2D broadcast equal should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![true, true, false, false]);
    }

    #[test]
    fn test_isclose_array() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0000001, 2.0000002, 3.0000003]);

        // Default tolerances
        let result = isclose_array(&a, &b, 1e-7, 0.0).expect("isclose_array should succeed");
        assert_eq!(result.to_vec(), vec![true, true, true]);

        // Stricter tolerances
        let result = isclose_array(&a, &b, 1e-10, 0.0)
            .expect("isclose_array with strict tol should succeed");
        assert_eq!(result.to_vec(), vec![false, false, false]);
    }

    // ---- broadcast cases vs NumPy ground truth ----
    //
    // Each expected result below is the literal output of the equivalent
    // `np.<fn>` call on the same two arrays (verified against NumPy's
    // documented broadcasting rule: dimensions are compared right-aligned,
    // and a size-1 dimension stretches to match the other operand's).

    #[test]
    fn test_greater_broadcast_matches_numpy() {
        // np.greater([[1],[2],[3]], [0,2,4]) ->
        // [[ True, False, False],
        //  [ True, False, False],
        //  [ True,  True, False]]
        let a = Array::from_vec(vec![1, 2, 3]).reshape(&[3, 1]);
        let b = Array::from_vec(vec![0, 2, 4]).reshape(&[1, 3]);
        let result = greater(&a, &b).expect("broadcast greater should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(
            result.to_vec(),
            vec![true, false, false, true, false, false, true, true, false]
        );
    }

    #[test]
    fn test_less_equal_broadcast_matches_numpy() {
        // np.less_equal([[1],[2],[3]], [0,2,4]) ->
        // [[False,  True,  True],
        //  [False, False,  True],
        //  [False, False,  True]]
        let a = Array::from_vec(vec![1, 2, 3]).reshape(&[3, 1]);
        let b = Array::from_vec(vec![0, 2, 4]).reshape(&[1, 3]);
        let result = less_equal(&a, &b).expect("broadcast less_equal should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(
            result.to_vec(),
            vec![false, true, true, false, true, true, false, false, true]
        );
    }

    #[test]
    fn test_not_equal_broadcast_matches_numpy() {
        // np.not_equal([[1],[2]], [1,2]) ->
        // [[False,  True],
        //  [ True, False]]
        let a = Array::from_vec(vec![1, 2]).reshape(&[2, 1]);
        let b = Array::from_vec(vec![1, 2]).reshape(&[1, 2]);
        let result = not_equal(&a, &b).expect("broadcast not_equal should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![false, true, true, false]);
    }

    #[test]
    fn test_logical_and_broadcast_matches_numpy() {
        // np.logical_and([[True],[False]], [True, False]) ->
        // [[ True, False],
        //  [False, False]]
        let a = Array::from_vec(vec![true, false]).reshape(&[2, 1]);
        let b = Array::from_vec(vec![true, false]).reshape(&[1, 2]);
        let result = logical_and(&a, &b).expect("logical_and broadcast should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![true, false, false, false]);
    }

    #[test]
    fn test_logical_or_broadcast_matches_numpy() {
        // np.logical_or([[True],[False]], [True, False]) ->
        // [[True,  True],
        //  [True, False]]
        let a = Array::from_vec(vec![true, false]).reshape(&[2, 1]);
        let b = Array::from_vec(vec![true, false]).reshape(&[1, 2]);
        let result = logical_or(&a, &b).expect("logical_or broadcast should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![true, true, true, false]);
    }

    #[test]
    fn test_logical_xor_broadcast_matches_numpy() {
        // np.logical_xor([[True],[False]], [True, False]) ->
        // [[False,  True],
        //  [ True, False]]
        let a = Array::from_vec(vec![true, false]).reshape(&[2, 1]);
        let b = Array::from_vec(vec![true, false]).reshape(&[1, 2]);
        let result = logical_xor(&a, &b).expect("logical_xor broadcast should succeed");
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![false, true, true, false]);
    }

    #[test]
    fn test_isclose_array_broadcast_matches_numpy() {
        // np.isclose([[1.],[2.],[3.]], [1., 2.], rtol=1e-7, atol=0.0) ->
        // [[ True, False],
        //  [False,  True],
        //  [False, False]]
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]);
        let b = Array::from_vec(vec![1.0, 2.0]).reshape(&[1, 2]);
        let result =
            isclose_array(&a, &b, 1e-7, 0.0).expect("isclose_array broadcast should succeed");
        assert_eq!(result.shape(), vec![3, 2]);
        assert_eq!(
            result.to_vec(),
            vec![true, false, false, true, false, false]
        );
    }

    #[test]
    fn test_allclose_with_tol_shape_mismatch_is_false_not_broadcast() {
        // Preserved, not-explicitly-asked-to-change semantics: unlike the
        // broadcasting comparison functions above, `allclose`/
        // `allclose_with_tol` never broadcast -- a shape mismatch is
        // simply "not close", never an error and never a broadcast
        // attempt (NumPy's `np.allclose` *does* broadcast; this crate's
        // deliberately does not, both before and after this migration).
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]);
        assert!(!allclose_with_tol(&a, &b, 1e-7, 0.0));
    }

    #[test]
    fn test_all_any_short_circuit_still_correct() {
        let all_true = Array::from_vec(vec![true, true, true, true]);
        let one_false = Array::from_vec(vec![true, true, false, true]);
        let all_false = Array::from_vec(vec![false, false, false, false]);

        assert!(all(&all_true));
        assert!(!all(&one_false));
        assert!(!all(&all_false));

        assert!(!any(&all_false));
        assert!(any(&one_false));
        assert!(any(&all_true));
    }

    #[test]
    fn test_maybe_broadcast_borrows_on_equal_shape() {
        use std::borrow::Cow;
        let a = Array::from_vec(vec![1, 2, 3]);
        let borrowed = maybe_broadcast(&a, &a.shape()).expect("equal shape always succeeds");
        assert!(matches!(borrowed, Cow::Borrowed(_)));

        let b = Array::from_vec(vec![1]);
        let owned = maybe_broadcast(&b, &[3]).expect("[1] broadcasts to [3]");
        assert!(matches!(owned, Cow::Owned(_)));
        assert_eq!(owned.to_vec(), vec![1, 1, 1]);
    }
}
