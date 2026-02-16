//! Memory-optimized array operations
//!
//! This module provides memory-efficient variants of common array operations
//! that minimize allocations through:
//! - Buffer reuse (in-place operations)
//! - Direct iterator usage (avoiding to_vec())
//! - Stack allocation for small arrays
//! - View-based operations

use super::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::ndarray::{Array1, Axis};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::ops::{Add, Div, Mul, Sub};

/// Threshold for using parallel processing
const PARALLEL_THRESHOLD: usize = 10000;

impl<T: Clone> Array<T> {
    /// Calculate sum without allocating a Vec (uses iterator directly)
    ///
    /// This is more memory-efficient than the standard sum() as it avoids
    /// the to_vec() call.
    pub fn sum_optimized(&self) -> T
    where
        T: Add<Output = T> + Zero + Clone + 'static,
    {
        // Use SIMD for f64 arrays with sufficient size
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let result = unsafe {
                let view = self.data.view();
                // Cast to f64 view
                let ptr = &view as *const _ as *const scirs2_core::ndarray::ArrayView<f64, _>;
                f64::simd_sum(&*ptr)
            };
            return unsafe { std::mem::transmute_copy(&result) };
        }

        // Direct iteration without to_vec() allocation
        self.data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate product without allocating a Vec
    pub fn product_optimized(&self) -> T
    where
        T: Mul<Output = T> + One + Clone,
    {
        self.data.iter().fold(T::one(), |acc, x| acc * x.clone())
    }

    /// In-place map operation that reuses the current array's memory
    ///
    /// This modifies the array in place instead of allocating a new one.
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        for elem in self.data.iter_mut() {
            *elem = f(elem);
        }
    }

    /// Map operation that writes to a pre-allocated output buffer
    ///
    /// This avoids allocation by reusing the provided output array.
    /// Returns an error if the output shape doesn't match.
    pub fn map_to<F>(&self, f: F, output: &mut Array<T>) -> Result<()>
    where
        F: Fn(&T) -> T,
    {
        if self.shape() != output.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: output.shape(),
            });
        }

        for (src, dst) in self.data.iter().zip(output.data.iter_mut()) {
            *dst = f(src);
        }

        Ok(())
    }

    /// Sum along axis without multiple allocations
    ///
    /// This uses ndarray's built-in axis sum which is more efficient
    /// than our manual implementation.
    pub fn sum_axis_optimized(&self, axis: usize) -> Result<Self>
    where
        T: Add<Output = T> + Zero + Clone,
    {
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        // Use ndarray's built-in sum_axis which is optimized
        let result = self.data.sum_axis(Axis(axis));
        Ok(Array::from_ndarray(result))
    }
}

// Optimized statistical operations
impl<T> Array<T>
where
    T: Float + Clone + Zero + NumCast + Send + Sync + 'static,
{
    /// Memory-optimized mean calculation
    ///
    /// Avoids to_vec() by iterating directly over the array.
    pub fn mean_optimized(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        let len = self.len();
        if len >= PARALLEL_THRESHOLD {
            // Use parallel reduction for large arrays
            let sum = self
                .data
                .view()
                .into_par_iter()
                .map(|&x| x)
                .reduce(|| T::zero(), |acc, x| acc + x);
            sum / T::from(len).expect("length should be representable")
        } else {
            let sum: T = self.data.iter().fold(T::zero(), |acc, &x| acc + x);
            sum / T::from(len).expect("length should be representable")
        }
    }

    /// Memory-optimized variance calculation
    ///
    /// Uses a single-pass algorithm with direct iteration.
    pub fn variance_optimized(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        // Use SIMD for f64 with sufficient size
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let view = self.data.view();
            let result = unsafe {
                let ptr = &view as *const _ as *const scirs2_core::ndarray::ArrayView<f64, _>;
                f64::simd_variance(&*ptr)
            };
            return unsafe { std::mem::transmute_copy(&result) };
        }

        let len = self.len();
        let mean = self.mean_optimized();

        if len >= PARALLEL_THRESHOLD {
            let sum_sq_diff = self
                .data
                .view()
                .into_par_iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .reduce(|| T::zero(), |acc, x| acc + x);
            sum_sq_diff / T::from(len).expect("length should be representable")
        } else {
            let sum_sq_diff: T = self
                .data
                .iter()
                .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean));
            sum_sq_diff / T::from(len).expect("length should be representable")
        }
    }

    /// Memory-optimized standard deviation
    pub fn std_optimized(&self) -> T {
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let view = self.data.view();
            let result = unsafe {
                let ptr = &view as *const _ as *const scirs2_core::ndarray::ArrayView<f64, _>;
                f64::simd_std(&*ptr)
            };
            return unsafe { std::mem::transmute_copy(&result) };
        }
        self.variance_optimized().sqrt()
    }

    /// Memory-optimized min calculation
    pub fn min_optimized(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // Use SIMD for f64 with sufficient size
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let view = self.data.view();
            let result = unsafe {
                let ptr = &view as *const _ as *const scirs2_core::ndarray::ArrayView<f64, _>;
                f64::simd_min_element(&*ptr)
            };
            return Some(unsafe { std::mem::transmute_copy(&result) });
        }

        if self.len() >= PARALLEL_THRESHOLD {
            let first = *self.data.iter().next().expect("non-empty array");
            Some(
                self.data
                    .view()
                    .into_par_iter()
                    .copied()
                    .reduce(|| first, |a, b| if a < b { a } else { b }),
            )
        } else {
            self.data.iter().copied().fold(None, |acc, x| match acc {
                None => Some(x),
                Some(min_val) => Some(if x < min_val { x } else { min_val }),
            })
        }
    }

    /// Memory-optimized max calculation
    pub fn max_optimized(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // Use SIMD for f64 with sufficient size
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let view = self.data.view();
            let result = unsafe {
                let ptr = &view as *const _ as *const scirs2_core::ndarray::ArrayView<f64, _>;
                f64::simd_max_element(&*ptr)
            };
            return Some(unsafe { std::mem::transmute_copy(&result) });
        }

        if self.len() >= PARALLEL_THRESHOLD {
            let first = *self.data.iter().next().expect("non-empty array");
            Some(
                self.data
                    .view()
                    .into_par_iter()
                    .copied()
                    .reduce(|| first, |a, b| if a > b { a } else { b }),
            )
        } else {
            self.data.iter().copied().fold(None, |acc, x| match acc {
                None => Some(x),
                Some(max_val) => Some(if x > max_val { x } else { max_val }),
            })
        }
    }
}

// Optimized matrix operations
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    /// Memory-optimized 2D matrix multiplication with pre-allocated output
    ///
    /// This version writes to a pre-allocated output buffer instead of
    /// creating a new array, reducing memory allocations.
    pub fn matmul_to(&self, other: &Self, output: &mut Self) -> Result<()> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Validate dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "matmul requires 2D arrays".to_string(),
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0], b_shape[1]],
                actual: vec![a_shape[0], a_shape[1]],
            });
        }

        let expected_shape = vec![a_shape[0], b_shape[1]];
        if output.shape() != expected_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: expected_shape,
                actual: output.shape(),
            });
        }

        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        // Use direct element access without to_vec()
        const BLOCK_SIZE: usize = 64;

        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                for j_block in (0..n).step_by(BLOCK_SIZE) {
                    let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                    let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                    let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);

                    for i in i_block..i_end {
                        for k_l in k_block..k_end {
                            let a_ik = self.data.get([i, k_l]).expect("valid index").clone();
                            for j in j_block..j_end {
                                let b_kj = other.data.get([k_l, j]).expect("valid index").clone();
                                let c_ij = output.data.get_mut([i, j]).expect("valid output index");
                                *c_ij = c_ij.clone() + a_ik.clone() * b_kj;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized dot product without to_vec()
    pub fn dot_optimized(&self, other: &Self) -> Result<T> {
        let a_shape = self.shape();
        let b_shape = other.shape();

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

        // Direct iteration without allocation
        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (a, b)| acc + a.clone() * b.clone());

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sum_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);

        let sum = arr.sum_optimized();
        assert_abs_diff_eq!(sum, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);

        let mean = arr.mean_optimized();
        assert_abs_diff_eq!(mean, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);

        let var = arr.variance_optimized();
        assert_abs_diff_eq!(var, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_map_inplace() {
        let data = vec![1.0, 2.0, 3.0];
        let mut arr = Array::from_vec(data);

        arr.map_inplace(|x| x * 2.0);

        let result = arr.to_vec();
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matmul_to() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
        let mut c = Array::zeros(&[2, 2]);

        a.matmul_to(&b, &mut c).expect("matmul_to should succeed");

        // Expected result: [[19, 22], [43, 50]]
        let result = c.to_vec();
        assert_abs_diff_eq!(result[0], 19.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 43.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10);
    }
}
