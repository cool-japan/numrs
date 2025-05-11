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
pub fn array_equal<T>(a: &Array<T>, b: &Array<T>) -> bool
where
    T: Clone + PartialEq + Debug
{
    // Check if shapes are the same
    if a.shape() != b.shape() {
        return false;
    }
    
    // Convert arrays to vectors and compare elements
    a.to_vec() == b.to_vec()
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
        
        assert!(array_equal(&a, &b));
        assert!(!array_equal(&a, &c));
        
        // Different shapes
        let d = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        assert!(!array_equal(&a, &d));
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