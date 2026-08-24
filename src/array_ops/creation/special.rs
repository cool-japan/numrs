//! Special array creation functions
//!
//! This module provides functions to create specialized arrays:
//! - `tri` - Create a triangular array
//! - `diagflat` - Create a 2D array with diagonal from flattened input
//! - `vander` - Generate a Vandermonde matrix

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{One, Zero};
use std::ops::Mul;

/// Create a triangular array with given diagonal and type
///
/// An array with ones at and below the given diagonal and zeros elsewhere.
///
/// # Parameters
///
/// * `n` - Number of rows
/// * `m` - Number of columns (if None, defaults to n)
/// * `k` - The sub-diagonal at and below which the array is filled.
///   k = 0 is the main diagonal, while k < 0 is below it,
///   and k > 0 is above. The default is 0.
/// * `dtype` - Data type of the array (for type inference)
///
/// # Returns
///
/// Array with shape (n, m) and requested type
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a 3x3 lower triangular array
/// let result: Array<i32> = tri(3, None, None).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
///
/// // Create a 3x4 array with diagonal offset
/// let result: Array<f64> = tri(3, Some(4), Some(1)).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![1.0, 1.0, 0.0, 0.0,
///                                  1.0, 1.0, 1.0, 0.0,
///                                  1.0, 1.0, 1.0, 1.0]);
///
/// // Create with negative diagonal offset
/// let result: Array<i32> = tri(3, None, Some(-1)).expect("operation should succeed");
/// assert_eq!(result.to_vec(), vec![0, 0, 0, 1, 0, 0, 1, 1, 0]);
/// ```
pub fn tri<T>(n: usize, m: Option<usize>, k: Option<isize>) -> Result<Array<T>>
where
    T: Clone + num_traits::Zero + num_traits::One,
{
    let m = m.unwrap_or(n);
    let k = k.unwrap_or(0);

    let mut data = Vec::with_capacity(n * m);

    for i in 0..n {
        for j in 0..m {
            if j as isize <= i as isize + k {
                data.push(T::one());
            } else {
                data.push(T::zero());
            }
        }
    }

    Array::from_vec_shape(data, &[n, m])
}

/// Create a 2D array with the flattened input as a diagonal
///
/// # Parameters
///
/// * `v` - Input array to be flattened
/// * `k` - Diagonal offset (0 is main diagonal, positive for upper, negative for lower)
///
/// # Returns
///
/// 2D array with the flattened input placed on the k-th diagonal
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::diagflat;
///
/// // 1D input
/// let v = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = diagflat(&v, 0).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![3, 3]);
/// // [[1, 0, 0],
/// //  [0, 2, 0],
/// //  [0, 0, 3]]
///
/// // With offset
/// let result = diagflat(&v, 1).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![4, 4]);
/// // [[0, 1, 0, 0],
/// //  [0, 0, 2, 0],
/// //  [0, 0, 0, 3],
/// //  [0, 0, 0, 0]]
///
/// // 2D input (gets flattened)
/// let v2d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let result = diagflat(&v2d, 0).expect("operation should succeed");
/// assert_eq!(result.shape(), vec![4, 4]);
/// ```
pub fn diagflat<T>(v: &Array<T>, k: i32) -> Result<Array<T>>
where
    T: Clone + Zero,
{
    // Flatten the input array
    let flat_data = v.to_vec();
    let n = flat_data.len();

    // Calculate output matrix size
    let size = (n as i32 + k.abs()) as usize;

    // Create output matrix filled with zeros
    let mut result = vec![T::zero(); size * size];

    // Fill diagonal
    for i in 0..n {
        let row = if k >= 0 { i } else { i + (-k) as usize };
        let col = if k >= 0 { i + k as usize } else { i };

        if row < size && col < size {
            result[row * size + col] = flat_data[i].clone();
        }
    }

    Array::from_vec_shape(result, &[size, size])
}

/// Generate a Vandermonde matrix
///
/// The columns of the output matrix are powers of the input vector.
/// The i-th column is the input vector raised element-wise to the power of N-i-1.
///
/// # Parameters
///
/// * `x` - 1D input array
/// * `n` - Number of columns in output. If None, defaults to len(x)
/// * `increasing` - If true, columns are in increasing powers (default: false)
///
/// # Returns
///
/// Vandermonde matrix. If increasing is false (default), the first column is x^(N-1),
/// the second x^(N-2), and so on. If increasing is true, the columns are x^0, x^1, ..., x^(N-1).
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Default: decreasing powers
/// let v = vander(&x, None, false).expect("operation should succeed");
/// assert_eq!(v.shape(), vec![3, 3]);
/// // [[1, 1, 1],
/// //  [4, 2, 1],
/// //  [9, 3, 1]]
///
/// // Increasing powers
/// let v = vander(&x, None, true).expect("operation should succeed");
/// assert_eq!(v.shape(), vec![3, 3]);
/// // [[1, 1, 1],
/// //  [1, 2, 4],
/// //  [1, 3, 9]]
///
/// // Custom number of columns
/// let v = vander(&x, Some(2), false).expect("operation should succeed");
/// assert_eq!(v.shape(), vec![3, 2]);
/// // [[1, 1],
/// //  [2, 1],
/// //  [3, 1]]
/// ```
pub fn vander<T>(x: &Array<T>, n: Option<usize>, increasing: bool) -> Result<Array<T>>
where
    T: Clone + Zero + One + Mul<Output = T>,
{
    if x.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "vander requires a 1D array".to_string(),
        ));
    }

    let m = x.len();
    let n_cols = n.unwrap_or(m);

    if n_cols == 0 {
        return Ok(Array::zeros(&[m, 0]));
    }

    let x_data = x.to_vec();
    let mut result = vec![T::one(); m * n_cols];

    // Fill the matrix
    for i in 0..m {
        let mut power = T::one();

        if increasing {
            // Powers: 0, 1, 2, ..., n-1
            for j in 0..n_cols {
                result[i * n_cols + j] = power.clone();
                if j < n_cols - 1 {
                    power = power * x_data[i].clone();
                }
            }
        } else {
            // Powers: n-1, n-2, ..., 1, 0
            // First compute x^(n-1)
            for _ in 1..n_cols {
                power = power * x_data[i].clone();
            }

            result[i * n_cols] = power.clone();

            // Then divide by x for each subsequent column
            for j in 1..n_cols {
                if j == n_cols - 1 {
                    result[i * n_cols + j] = T::one();
                } else {
                    // We need division here, but it's not in the trait bounds
                    // For now, we'll recompute the power
                    let mut pow = T::one();
                    for _ in 0..(n_cols - j - 1) {
                        pow = pow * x_data[i].clone();
                    }
                    result[i * n_cols + j] = pow;
                }
            }
        }
    }

    Array::from_vec_shape(result, &[m, n_cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri() {
        // Test basic triangular array
        let result: Array<i32> = tri(3, None, None).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(result.to_vec(), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);

        // Test with different dimensions
        let result: Array<f64> = tri(3, Some(4), Some(1)).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3, 4]);
        assert_eq!(
            result.to_vec(),
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );

        // Test with negative offset
        let result: Array<i32> = tri(3, None, Some(-1)).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(result.to_vec(), vec![0, 0, 0, 1, 0, 0, 1, 1, 0]);
    }

    #[test]
    fn test_diagflat() {
        // Test 1D input
        let v = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let result = diagflat(&v, 0).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(
            result.to_vec(),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );

        // Test with positive offset
        let result = diagflat(&v, 1).expect("operation should succeed");
        assert_eq!(result.shape(), vec![4, 4]);
        // Should have diagonal one above main

        // Test with negative offset
        let result = diagflat(&v, -1).expect("operation should succeed");
        assert_eq!(result.shape(), vec![4, 4]);
        // Should have diagonal one below main
    }

    #[test]
    fn test_vander() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);

        // Test increasing powers
        let v = vander(&x, None, true).expect("operation should succeed");
        assert_eq!(v.shape(), vec![3, 3]);
        assert_eq!(
            v.to_vec(),
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0]
        );

        // Test decreasing powers
        let v = vander(&x, None, false).expect("operation should succeed");
        assert_eq!(v.shape(), vec![3, 3]);
        assert_eq!(
            v.to_vec(),
            vec![1.0, 1.0, 1.0, 4.0, 2.0, 1.0, 9.0, 3.0, 1.0]
        );

        // Test custom number of columns
        let v = vander(&x, Some(2), false).expect("operation should succeed");
        assert_eq!(v.shape(), vec![3, 2]);
        assert_eq!(v.to_vec(), vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);

        // Test error on non-1D input
        let x2d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        assert!(vander(&x2d, None, true).is_err());
    }
}
