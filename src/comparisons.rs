use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

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
    T: Clone + Float + Debug
{
    allclose_with_tol(a, b, T::from(1e-7).unwrap(), T::zero())
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
    T: Clone + Float + Debug
{
    // Check if shapes are the same
    if a.shape() != b.shape() {
        return false;
    }
    
    // Convert arrays to vectors
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    
    // Check each element
    for (a_val, b_val) in a_data.iter().zip(b_data.iter()) {
        if !isclose(*a_val, *b_val, rtol, atol) {
            return false;
        }
    }
    
    true
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
    T: Clone + Float + Debug
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
/// assert!(array_equal(&a, &b));
/// assert!(!array_equal(&a, &c));
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
    T: Clone + PartialEq + Debug + 'static
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
    T: Clone + PartialEq + Debug + 'static
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
/// let d = Array::from_vec(vec![1, 2, 3]).reshape(&[3, 1]).unwrap();
/// let e = Array::from_vec(vec![1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(&[3, 3]).unwrap();
/// let mut opts = ArrayCompareOptions::default();
/// opts.allow_broadcasting = true;
/// assert!(array_compare(&d, &e, &opts)); // d is broadcast across columns
/// ```
pub fn array_compare<T>(a: &Array<T>, b: &Array<T>, options: &ArrayCompareOptions) -> bool
where
    T: Clone + PartialEq + Debug + 'static
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
    T: Clone + PartialEq + Debug + 'static
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
fn array_compare_with_nan_handling<T>(a: &Array<T>, b: &Array<T>, ignore_mask: &[bool]) -> Option<bool>
where
    T: Clone + PartialEq + Debug + 'static
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
#[derive(Debug, Clone)]
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

impl Default for ArrayCompareOptions {
    fn default() -> Self {
        Self {
            equal_nan: false,
            allow_broadcasting: false,
            ignore_indices: None,
            rtol: None,
            atol: None,
        }
    }
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
    bool: From<T>
{
    // Check all elements
    a.to_vec().iter().all(|val| bool::from(val.clone()))
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
    bool: From<T>
{
    // Check any element
    a.to_vec().iter().any(|val| bool::from(val.clone()))
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
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![0, 2, 4]);
///
/// let result = greater(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![true, false, false]);
/// ```
pub fn greater<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialOrd + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| a_val > b_val)
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
    T: Clone + PartialOrd + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| a_val >= b_val)
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
    T: Clone + PartialOrd + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| a_val < b_val)
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
    T: Clone + PartialOrd + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| a_val <= b_val)
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
    T: Clone + PartialEq + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| a_val == b_val)
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
    T: Clone + PartialEq + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| a_val != b_val)
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
    T: Clone + Float + Debug
{
    // Check if shapes are compatible for broadcasting
    let broadcast_shape = Array::<T>::broadcast_shape(&a.shape(), &b.shape())
        .map_err(|_| NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        })?;
    
    // Broadcast arrays if needed
    let a_broadcast = if a.shape() != broadcast_shape {
        a.broadcast_to(&broadcast_shape)?
    } else {
        a.clone()
    };
    
    let b_broadcast = if b.shape() != broadcast_shape {
        b.broadcast_to(&broadcast_shape)?
    } else {
        b.clone()
    };
    
    // Convert arrays to vectors
    let a_data = a_broadcast.to_vec();
    let b_data = b_broadcast.to_vec();
    
    // Compare elements
    let result: Vec<bool> = a_data.iter().zip(b_data.iter())
        .map(|(a_val, b_val)| isclose(*a_val, *b_val, rtol, atol))
        .collect();
    
    // Create result array with the broadcast shape
    Ok(Array::from_vec(result).reshape(&broadcast_shape))
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
        assert!(isclose(std::f64::NAN, std::f64::NAN, 1e-7, 0.0));
        
        // Test infinity handling
        assert!(isclose(std::f64::INFINITY, std::f64::INFINITY, 1e-7, 0.0));
        assert!(!isclose(std::f64::INFINITY, 1.0, 1e-7, 0.0));
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
        let result = greater(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![true, false, false]);
        
        // Test greater_equal
        let result = greater_equal(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![true, true, false]);
        
        // Test less
        let result = less(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![false, false, true]);
        
        // Test less_equal
        let result = less_equal(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![false, true, true]);
        
        // Test equal
        let result = equal(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![false, true, false]);
        
        // Test not_equal
        let result = not_equal(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![true, false, true]);
    }
    
    #[test]
    fn test_broadcasting() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1]).reshape(&[1]);
        
        // Test broadcasting
        let result = equal(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![true, false, false]);
        
        // Test with 2D arrays
        let c = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let d = Array::from_vec(vec![1, 2]).reshape(&[1, 2]);
        
        let result = equal(&c, &d).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.to_vec(), vec![true, true, false, false]);
    }
    
    #[test]
    fn test_isclose_array() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0000001, 2.0000002, 3.0000003]);
        
        // Default tolerances
        let result = isclose_array(&a, &b, 1e-7, 0.0).unwrap();
        assert_eq!(result.to_vec(), vec![true, true, true]);
        
        // Stricter tolerances
        let result = isclose_array(&a, &b, 1e-10, 0.0).unwrap();
        assert_eq!(result.to_vec(), vec![false, false, false]);
    }
}