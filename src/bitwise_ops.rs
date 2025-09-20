//! Bitwise operations for Arrays
//!
//! This module provides bitwise operations for integer arrays, including
//! AND, OR, XOR, NOT, and bit shifting operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};

/// Compute the bit-wise AND of two arrays element-wise
///
/// # Arguments
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
/// * `Result<Array<T>>` - Array with bit-wise AND of corresponding elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::bitwise_and;
///
/// let a = Array::from_vec(vec![13, 17, 21]);  // 01101, 10001, 10101
/// let b = Array::from_vec(vec![9, 7, 15]);    // 01001, 00111, 01111
/// let result = bitwise_and(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![9, 1, 5]); // 01001, 00001, 00101
/// ```
pub fn bitwise_and<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + BitAnd<Output = T>,
{
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| a & b)
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Compute the bit-wise OR of two arrays element-wise
///
/// # Arguments
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
/// * `Result<Array<T>>` - Array with bit-wise OR of corresponding elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::bitwise_or;
///
/// let a = Array::from_vec(vec![13, 17, 21]);  // 01101, 10001, 10101
/// let b = Array::from_vec(vec![9, 7, 15]);    // 01001, 00111, 01111
/// let result = bitwise_or(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![13, 23, 31]); // 01101, 10111, 11111
/// ```
pub fn bitwise_or<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + BitOr<Output = T>,
{
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| a | b)
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Compute the bit-wise XOR (exclusive OR) of two arrays element-wise
///
/// # Arguments
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
/// * `Result<Array<T>>` - Array with bit-wise XOR of corresponding elements
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::bitwise_xor;
///
/// let a = Array::from_vec(vec![13, 17, 21]);  // 01101, 10001, 10101
/// let b = Array::from_vec(vec![9, 7, 15]);    // 01001, 00111, 01111
/// let result = bitwise_xor(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![4, 22, 26]); // 00100, 10110, 11010
/// ```
pub fn bitwise_xor<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + BitXor<Output = T>,
{
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| a ^ b)
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Compute bit-wise NOT (inversion) element-wise
///
/// This function is an alias for `invert`.
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * `Array<T>` - Array with bit-wise NOT of each element
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::bitwise_not;
///
/// let a = Array::from_vec(vec![13u8, 17u8, 21u8]);
/// let result = bitwise_not(&a);
/// assert_eq!(result.to_vec(), vec![242u8, 238u8, 234u8]); // ~13, ~17, ~21
/// ```
pub fn bitwise_not<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Not<Output = T>,
{
    x.map(|val| !val)
}

/// Compute bit-wise inversion element-wise
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * `Array<T>` - Array with bit-wise inversion of each element
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::invert;
///
/// let a = Array::from_vec(vec![13u8, 17u8, 21u8]);
/// let result = invert(&a);
/// assert_eq!(result.to_vec(), vec![242u8, 238u8, 234u8]); // ~13, ~17, ~21
/// ```
pub fn invert<T>(x: &Array<T>) -> Array<T>
where
    T: Clone + Not<Output = T>,
{
    bitwise_not(x)
}

/// Shift the bits of an integer to the left
///
/// # Arguments
/// * `x1` - Input array
/// * `x2` - Number of bits to shift (can be an array or scalar)
///
/// # Returns
/// * `Result<Array<T>>` - Array with bits shifted left
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::left_shift;
///
/// let a = Array::from_vec(vec![5, 10, 15]);
/// let shift = Array::from_vec(vec![1, 2, 3]);
/// let result = left_shift(&a, &shift).unwrap();
/// assert_eq!(result.to_vec(), vec![10, 40, 120]); // 5<<1, 10<<2, 15<<3
/// ```
pub fn left_shift<T, U>(x1: &Array<T>, x2: &Array<U>) -> Result<Array<T>>
where
    T: Clone + Shl<U, Output = T>,
    U: Clone,
{
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| a << b)
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Shift the bits of an integer to the right
///
/// # Arguments
/// * `x1` - Input array
/// * `x2` - Number of bits to shift (can be an array or scalar)
///
/// # Returns
/// * `Result<Array<T>>` - Array with bits shifted right
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::right_shift;
///
/// let a = Array::from_vec(vec![40, 80, 120]);
/// let shift = Array::from_vec(vec![1, 2, 3]);
/// let result = right_shift(&a, &shift).unwrap();
/// assert_eq!(result.to_vec(), vec![20, 20, 15]); // 40>>1, 80>>2, 120>>3
/// ```
pub fn right_shift<T, U>(x1: &Array<T>, x2: &Array<U>) -> Result<Array<T>>
where
    T: Clone + Shr<U, Output = T>,
    U: Clone,
{
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| a >> b)
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Shift the bits of an integer to the left by a scalar amount
///
/// # Arguments
/// * `x` - Input array
/// * `shift` - Number of bits to shift
///
/// # Returns
/// * `Array<T>` - Array with bits shifted left
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::left_shift_scalar;
///
/// let a = Array::from_vec(vec![5, 10, 15]);
/// let result = left_shift_scalar(&a, 2);
/// assert_eq!(result.to_vec(), vec![20, 40, 60]); // 5<<2, 10<<2, 15<<2
/// ```
pub fn left_shift_scalar<T, U>(x: &Array<T>, shift: U) -> Array<T>
where
    T: Clone + Shl<U, Output = T>,
    U: Clone,
{
    x.map(|val| val << shift.clone())
}

/// Shift the bits of an integer to the right by a scalar amount
///
/// # Arguments
/// * `x` - Input array
/// * `shift` - Number of bits to shift
///
/// # Returns
/// * `Array<T>` - Array with bits shifted right
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::bitwise_ops::right_shift_scalar;
///
/// let a = Array::from_vec(vec![40, 80, 120]);
/// let result = right_shift_scalar(&a, 2);
/// assert_eq!(result.to_vec(), vec![10, 20, 30]); // 40>>2, 80>>2, 120>>2
/// ```
pub fn right_shift_scalar<T, U>(x: &Array<T>, shift: U) -> Array<T>
where
    T: Clone + Shr<U, Output = T>,
    U: Clone,
{
    x.map(|val| val >> shift.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_and() {
        let a = Array::from_vec(vec![13, 17, 21]);
        let b = Array::from_vec(vec![9, 7, 15]);
        let result = bitwise_and(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![9, 1, 5]);
    }

    #[test]
    fn test_bitwise_or() {
        let a = Array::from_vec(vec![13, 17, 21]);
        let b = Array::from_vec(vec![9, 7, 15]);
        let result = bitwise_or(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![13, 23, 31]);
    }

    #[test]
    fn test_bitwise_xor() {
        let a = Array::from_vec(vec![13, 17, 21]);
        let b = Array::from_vec(vec![9, 7, 15]);
        let result = bitwise_xor(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![4, 22, 26]);
    }

    #[test]
    fn test_bitwise_not() {
        let a = Array::from_vec(vec![13u8, 17u8, 21u8]);
        let result = bitwise_not(&a);
        assert_eq!(result.to_vec(), vec![242u8, 238u8, 234u8]);
    }

    #[test]
    fn test_left_shift() {
        let a = Array::from_vec(vec![5, 10, 15]);
        let shift = Array::from_vec(vec![1, 2, 3]);
        let result = left_shift(&a, &shift).unwrap();
        assert_eq!(result.to_vec(), vec![10, 40, 120]);
    }

    #[test]
    fn test_right_shift() {
        let a = Array::from_vec(vec![40, 80, 120]);
        let shift = Array::from_vec(vec![1, 2, 3]);
        let result = right_shift(&a, &shift).unwrap();
        assert_eq!(result.to_vec(), vec![20, 20, 15]);
    }

    #[test]
    fn test_shift_scalar() {
        let a = Array::from_vec(vec![5, 10, 15]);
        let left_result = left_shift_scalar(&a, 2);
        assert_eq!(left_result.to_vec(), vec![20, 40, 60]);

        let right_result = right_shift_scalar(&left_result, 2);
        assert_eq!(right_result.to_vec(), vec![5, 10, 15]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2]);

        assert!(bitwise_and(&a, &b).is_err());
        assert!(bitwise_or(&a, &b).is_err());
        assert!(bitwise_xor(&a, &b).is_err());
    }
}
