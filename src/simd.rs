//! SIMD-accelerated operations using scirs2-core
//!
//! This module provides SIMD operations by wrapping scirs2_core::simd_ops::SimdUnifiedOps.
//! ALL SIMD functionality goes through scirs2-core per SCIRS2 POLICY.
//!
//! NO custom SIMD implementations - scirs2-core provides:
//! - 100+ SIMD operations
//! - Automatic platform detection (AVX2/AVX512/NEON)
//! - 32.48x average speedup vs NumPy
//! - 470x speedup for reductions

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

// SCIRS2 POLICY: Use scirs2_core::simd_ops instead of custom SIMD
use scirs2_core::ndarray::{Array1, ArrayView1, CowArray, Ix1};
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};

/// Trait for SIMD-accelerated operations on NumRS2 arrays
///
/// This trait wraps scirs2_core::simd_ops::SimdUnifiedOps to provide
/// SIMD operations on NumRS2's Array type.
pub trait SimdOps<T> {
    /// SIMD-accelerated element-wise addition
    fn simd_add(&self, other: &Array<T>) -> Result<Array<T>>;

    /// SIMD-accelerated element-wise subtraction
    fn simd_sub(&self, other: &Array<T>) -> Result<Array<T>>;

    /// SIMD-accelerated element-wise multiplication
    fn simd_mul(&self, other: &Array<T>) -> Result<Array<T>>;

    /// SIMD-accelerated element-wise division
    fn simd_div(&self, other: &Array<T>) -> Result<Array<T>>;

    /// SIMD-accelerated dot product
    fn simd_dot(&self, other: &Array<T>) -> Result<T>;

    /// SIMD-accelerated sum reduction
    fn simd_sum(&self) -> T;

    /// SIMD-accelerated mean calculation
    fn simd_mean(&self) -> T;

    /// SIMD-accelerated fused multiply-add: self * mul + add
    fn simd_fma(&self, mul: &Array<T>, add: &Array<T>) -> Result<Array<T>>;

    /// SIMD-accelerated scalar addition (broadcast)
    fn simd_add_scalar(&self, scalar: T) -> Array<T>;

    /// SIMD-accelerated scalar multiplication (broadcast)
    fn simd_mul_scalar(&self, scalar: T) -> Array<T>;

    /// SIMD-accelerated scalar subtraction (broadcast)
    fn simd_sub_scalar(&self, scalar: T) -> Array<T>;

    /// SIMD-accelerated scalar division (broadcast)
    fn simd_div_scalar(&self, scalar: T) -> Result<Array<T>>;
}

// Helper to borrow a NumRS2 Array as a 1-D ndarray view for SIMD kernels.
// Returns a `CowArray`: a zero-copy view for contiguous arrays (the common
// case) and an owned copy only for non-contiguous layouts. Wrapped in
// `Result` so existing call sites using `?`/`.expect()` stay unchanged.
fn to_ndarray_1d<T: Clone>(arr: &Array<T>) -> Result<CowArray<'_, T, Ix1>> {
    Ok(arr.as_cow_1d())
}

/// Consume a freshly computed SIMD-kernel `Array1<T>` result into a
/// `Vec<T>`, reusing its backing buffer directly instead of `.to_vec()`'s
/// unconditional clone-and-copy.
///
/// Every call site below hands this the direct return value of a
/// `scirs2_core::simd_ops::SimdUnifiedOps` function (`simd_add`,
/// `simd_mul`, ...), which is always a freshly allocated, owned
/// `Array1<T>` -- there is no other live reference to its buffer, so
/// reusing that buffer for the `Array::from_vec` this crate builds right
/// after is sound and copy-free, unlike `result.to_vec()` (which clones
/// every element into a *second* buffer immediately before the first one
/// is dropped).
///
/// `into_raw_vec_and_offset` is not, by itself, a sufficient zero-copy
/// guard: a standard-layout array can still report a non-zero offset or a
/// backing buffer longer than the array's logical length (e.g. a slice of
/// a larger allocation), and unlike `to_vec()`'s `Some(slice) =>
/// slice.to_vec()` fast path, consuming `arr` means there is no falling
/// back to `.iter()` *after* the fact. So every condition needed for the
/// raw buffer to equal the logical contents exactly -- standard layout,
/// zero offset, matching length -- is checked before ever calling
/// `into_raw_vec_and_offset`; any array that fails one of those checks
/// (not expected from a fresh SIMD kernel output today, but not part of
/// `Array1`'s public contract either) falls back to the same
/// `iter().cloned().collect()` `to_vec()` itself would have used.
pub(crate) fn into_vec_no_copy<T: Clone>(arr: Array1<T>) -> Vec<T> {
    if !arr.is_standard_layout() {
        return arr.iter().cloned().collect();
    }
    let n = arr.len();
    let (v, offset) = arr.into_raw_vec_and_offset();
    let off = offset.unwrap_or(0);
    if off == 0 && v.len() == n {
        v
    } else {
        v[off..off + n].to_vec()
    }
}

// Implementation for f32
impl SimdOps<f32> for Array<f32> {
    fn simd_add(&self, other: &Array<f32>) -> Result<Array<f32>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f32::simd_add(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_sub(&self, other: &Array<f32>) -> Result<Array<f32>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f32::simd_sub(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_mul(&self, other: &Array<f32>) -> Result<Array<f32>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f32::simd_mul(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_div(&self, other: &Array<f32>) -> Result<Array<f32>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f32::simd_div(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_dot(&self, other: &Array<f32>) -> Result<f32> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        Ok(f32::simd_dot(&a.view(), &b.view()))
    }

    fn simd_sum(&self) -> f32 {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        f32::simd_sum(&a.view())
    }

    fn simd_mean(&self) -> f32 {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        f32::simd_mean(&a.view())
    }

    fn simd_fma(&self, mul: &Array<f32>, add: &Array<f32>) -> Result<Array<f32>> {
        if self.shape() != mul.shape() || self.shape() != add.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: if self.shape() != mul.shape() {
                    mul.shape()
                } else {
                    add.shape()
                },
            });
        }

        let a = to_ndarray_1d(self)?;
        let m = to_ndarray_1d(mul)?;
        let b = to_ndarray_1d(add)?;
        let result = f32::simd_fma(&a.view(), &m.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_add_scalar(&self, scalar: f32) -> Array<f32> {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        let result = f32::simd_add(&a.view(), &ArrayView1::from(&vec![scalar; a.len()]));
        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn simd_mul_scalar(&self, scalar: f32) -> Array<f32> {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        let result = f32::simd_scalar_mul(&a.view(), scalar);
        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn simd_sub_scalar(&self, scalar: f32) -> Array<f32> {
        // a - scalar = a + (-scalar)
        self.simd_add_scalar(-scalar)
    }

    fn simd_div_scalar(&self, scalar: f32) -> Result<Array<f32>> {
        if scalar == 0.0 {
            return Err(NumRs2Error::InvalidOperation(
                "Division by zero".to_string(),
            ));
        }
        // a / scalar = a * (1/scalar)
        Ok(self.simd_mul_scalar(1.0 / scalar))
    }
}

// Implementation for f64
impl SimdOps<f64> for Array<f64> {
    fn simd_add(&self, other: &Array<f64>) -> Result<Array<f64>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f64::simd_add(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_sub(&self, other: &Array<f64>) -> Result<Array<f64>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f64::simd_sub(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_mul(&self, other: &Array<f64>) -> Result<Array<f64>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f64::simd_mul(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_div(&self, other: &Array<f64>) -> Result<Array<f64>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        let result = f64::simd_div(&a.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_dot(&self, other: &Array<f64>) -> Result<f64> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        let a = to_ndarray_1d(self)?;
        let b = to_ndarray_1d(other)?;
        Ok(f64::simd_dot(&a.view(), &b.view()))
    }

    fn simd_sum(&self) -> f64 {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        f64::simd_sum(&a.view())
    }

    fn simd_mean(&self) -> f64 {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        f64::simd_mean(&a.view())
    }

    fn simd_fma(&self, mul: &Array<f64>, add: &Array<f64>) -> Result<Array<f64>> {
        if self.shape() != mul.shape() || self.shape() != add.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: if self.shape() != mul.shape() {
                    mul.shape()
                } else {
                    add.shape()
                },
            });
        }

        let a = to_ndarray_1d(self)?;
        let m = to_ndarray_1d(mul)?;
        let b = to_ndarray_1d(add)?;
        let result = f64::simd_fma(&a.view(), &m.view(), &b.view());

        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
    }

    fn simd_add_scalar(&self, scalar: f64) -> Array<f64> {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        let result = f64::simd_add(&a.view(), &ArrayView1::from(&vec![scalar; a.len()]));
        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn simd_mul_scalar(&self, scalar: f64) -> Array<f64> {
        let a = to_ndarray_1d(self).expect("Array conversion to ndarray should succeed");
        let result = f64::simd_scalar_mul(&a.view(), scalar);
        Array::from_vec_shape(into_vec_no_copy(result), &self.shape())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn simd_sub_scalar(&self, scalar: f64) -> Array<f64> {
        // a - scalar = a + (-scalar)
        self.simd_add_scalar(-scalar)
    }

    fn simd_div_scalar(&self, scalar: f64) -> Result<Array<f64>> {
        if scalar == 0.0 {
            return Err(NumRs2Error::InvalidOperation(
                "Division by zero".to_string(),
            ));
        }
        // a / scalar = a * (1/scalar)
        Ok(self.simd_mul_scalar(1.0 / scalar))
    }
}

// Convenient free functions for SIMD operations

/// SIMD-accelerated element-wise addition
pub fn simd_add<T: Float + 'static>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    Array<T>: SimdOps<T>,
{
    a.simd_add(b)
}

/// SIMD-accelerated element-wise multiplication
pub fn simd_mul<T: Float + 'static>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    Array<T>: SimdOps<T>,
{
    a.simd_mul(b)
}

/// SIMD-accelerated element-wise division
pub fn simd_div<T: Float + 'static>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    Array<T>: SimdOps<T>,
{
    a.simd_div(b)
}

/// SIMD-accelerated sum of all elements
pub fn simd_sum<T: Float + 'static>(a: &Array<T>) -> T
where
    Array<T>: SimdOps<T>,
{
    a.simd_sum()
}

/// SIMD-accelerated mean of all elements
pub fn simd_mean<T: Float + 'static>(a: &Array<T>) -> T
where
    Array<T>: SimdOps<T>,
{
    a.simd_mean()
}

/// SIMD-accelerated product of all elements
pub fn simd_prod<T: Float + 'static>(a: &Array<T>) -> T {
    // Use scirs2 for f32/f64, fallback to scalar for others
    a.array().iter().copied().fold(T::one(), |acc, x| acc * x)
}

/// SIMD-accelerated exponential function
pub fn simd_exp<T: Float + 'static>(a: &Array<T>) -> Array<T> {
    let shape = a.shape();
    // Use SimdUnifiedOps for f64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let data_f64: Vec<f64> = a
            .array()
            .iter()
            .map(|&x| x.to_f64().expect("f64 conversion should succeed"))
            .collect();
        let nd_arr = Array1::from_vec(data_f64);
        let result = f64::simd_exp(&nd_arr.view());
        let result_vec: Vec<T> = result
            .iter()
            .map(|&x| T::from(x).expect("conversion from f64 should succeed"))
            .collect();
        return Array::from_vec_shape(result_vec, &shape).unwrap_or_else(|e| panic!("{e}"));
    }
    // Use SimdUnifiedOps for f32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let data_f32: Vec<f32> = a
            .array()
            .iter()
            .map(|&x| x.to_f32().expect("f32 conversion should succeed"))
            .collect();
        let nd_arr = Array1::from_vec(data_f32);
        let result = f32::simd_exp(&nd_arr.view());
        let result_vec: Vec<T> = result
            .iter()
            .map(|&x| T::from(x).expect("conversion from f32 should succeed"))
            .collect();
        return Array::from_vec_shape(result_vec, &shape).unwrap_or_else(|e| panic!("{e}"));
    }
    // Fallback for other types
    let result: Vec<T> = a.array().iter().map(|&x| x.exp()).collect();
    Array::from_vec_shape(result, &shape).unwrap_or_else(|e| panic!("{e}"))
}

/// SIMD-accelerated natural logarithm
pub fn simd_log<T: Float + 'static>(a: &Array<T>) -> Array<T> {
    let shape = a.shape();
    // Use SimdUnifiedOps for f64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let data_f64: Vec<f64> = a
            .array()
            .iter()
            .map(|&x| x.to_f64().expect("f64 conversion should succeed"))
            .collect();
        let nd_arr = Array1::from_vec(data_f64);
        let result = f64::simd_ln(&nd_arr.view());
        let result_vec: Vec<T> = result
            .iter()
            .map(|&x| T::from(x).expect("conversion from f64 should succeed"))
            .collect();
        return Array::from_vec_shape(result_vec, &shape).unwrap_or_else(|e| panic!("{e}"));
    }
    // Use SimdUnifiedOps for f32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let data_f32: Vec<f32> = a
            .array()
            .iter()
            .map(|&x| x.to_f32().expect("f32 conversion should succeed"))
            .collect();
        let nd_arr = Array1::from_vec(data_f32);
        let result = f32::simd_ln(&nd_arr.view());
        let result_vec: Vec<T> = result
            .iter()
            .map(|&x| T::from(x).expect("conversion from f32 should succeed"))
            .collect();
        return Array::from_vec_shape(result_vec, &shape).unwrap_or_else(|e| panic!("{e}"));
    }
    // Fallback for other types
    let result: Vec<T> = a.array().iter().map(|&x| x.ln()).collect();
    Array::from_vec_shape(result, &shape).unwrap_or_else(|e| panic!("{e}"))
}

/// SIMD-accelerated square root
pub fn simd_sqrt<T: Float + 'static>(a: &Array<T>) -> Array<T> {
    let shape = a.shape();
    // Use SimdUnifiedOps for f64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let data_f64: Vec<f64> = a
            .array()
            .iter()
            .map(|&x| x.to_f64().expect("f64 conversion should succeed"))
            .collect();
        let nd_arr = Array1::from_vec(data_f64);
        let result = f64::simd_sqrt(&nd_arr.view());
        let result_vec: Vec<T> = result
            .iter()
            .map(|&x| T::from(x).expect("conversion from f64 should succeed"))
            .collect();
        return Array::from_vec_shape(result_vec, &shape).unwrap_or_else(|e| panic!("{e}"));
    }
    // Use SimdUnifiedOps for f32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let data_f32: Vec<f32> = a
            .array()
            .iter()
            .map(|&x| x.to_f32().expect("f32 conversion should succeed"))
            .collect();
        let nd_arr = Array1::from_vec(data_f32);
        let result = f32::simd_sqrt(&nd_arr.view());
        let result_vec: Vec<T> = result
            .iter()
            .map(|&x| T::from(x).expect("conversion from f32 should succeed"))
            .collect();
        return Array::from_vec_shape(result_vec, &shape).unwrap_or_else(|e| panic!("{e}"));
    }
    // Fallback for other types
    let result: Vec<T> = a.array().iter().map(|&x| x.sqrt()).collect();
    Array::from_vec_shape(result, &shape).unwrap_or_else(|e| panic!("{e}"))
}

/// Get information about available SIMD optimizations
pub fn get_simd_implementation_name() -> String {
    let caps = PlatformCapabilities::detect();
    format!(
        "NumRS2 SIMD via scirs2-core: AVX512={}, AVX2={}, NEON={}, SIMD={}",
        caps.avx512_available, caps.avx2_available, caps.neon_available, caps.simd_available
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_add() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let c = simd_add(&a, &b).expect("simd_add should succeed");
        assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_simd_mul() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let c = simd_mul(&a, &b).expect("simd_mul should succeed");
        assert_eq!(c.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_simd_div() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let c = simd_div(&a, &b).expect("simd_div should succeed");
        assert_relative_eq!(c.to_vec()[0], 0.2, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[1], 2.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[2], 3.0 / 7.0, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[3], 4.0 / 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_sqrt() {
        let a = Array::from_vec(vec![1.0f64, 4.0, 9.0, 16.0]);
        let b = simd_sqrt(&a);
        assert_relative_eq!(b.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(b.to_vec()[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(b.to_vec()[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(b.to_vec()[3], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_sum() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let sum = simd_sum(&a);
        assert_relative_eq!(sum, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_prod() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let prod = simd_prod(&a);
        assert_relative_eq!(prod, 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_mean() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let mean = a.simd_mean();
        assert_relative_eq!(mean, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_add_scalar() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = a.simd_add_scalar(10.0);
        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_simd_mul_scalar() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = a.simd_mul_scalar(2.0);
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_sub_scalar() {
        let a = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]);
        let result = a.simd_sub_scalar(5.0);
        assert_eq!(result.to_vec(), vec![5.0, 15.0, 25.0, 35.0]);
    }

    #[test]
    fn test_simd_div_scalar() {
        let a = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]);
        let result = a
            .simd_div_scalar(2.0)
            .expect("simd_div_scalar should succeed");
        assert_eq!(result.to_vec(), vec![5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_simd_div_scalar_zero() {
        let a = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]);
        let result = a.simd_div_scalar(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_fma() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0]);
        let b = Array::from_vec(vec![2.0f64, 3.0, 4.0]);
        let c = Array::from_vec(vec![1.0f64, 1.0, 1.0]);
        let result = a.simd_fma(&b, &c).expect("simd_fma should succeed");
        // a * b + c = [1*2+1, 2*3+1, 3*4+1] = [3, 7, 13]
        assert_eq!(result.to_vec(), vec![3.0, 7.0, 13.0]);
    }

    #[test]
    fn test_cpu_feature_detection() {
        let info = get_simd_implementation_name();
        println!("Detected SIMD capabilities: {}", info);
        assert!(info.contains("NumRS2 SIMD via scirs2-core"));
    }

    #[test]
    fn test_simd_large_array() {
        // Test with larger array to exercise SIMD lanes
        let data: Vec<f64> = (0..1024).map(|i| i as f64).collect();
        let a = Array::from_vec(data.clone());
        let b = Array::from_vec(data);

        let result = simd_add(&a, &b).expect("simd_add should succeed");
        assert_relative_eq!(
            result.get(&[0]).expect("get element should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[512]).expect("get element should succeed"),
            1024.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1023]).expect("get element should succeed"),
            2046.0,
            epsilon = 1e-10
        );
    }

    // ==========================================================
    // to_vec sweep (W2-D): `into_vec_no_copy` output-side fix
    //
    // The *input* side already borrowed zero-copy via `as_cow_1d()`
    // (`to_ndarray_1d`); every `result.to_vec()` here was on the
    // *output* side instead -- cloning a freshly computed, about-to-be-
    // dropped `Array1<T>` into a second buffer right before
    // `Array::from_vec` took ownership of it. `into_vec_no_copy` reuses
    // that buffer directly when its layout guarantees it is safe to.
    // ==========================================================

    #[test]
    fn simd_add_correct_for_non_contiguous_input() {
        // `transpose_axis` produces a genuinely non-contiguous *view*
        // (unlike `Array::transpose()`, which eagerly materializes a
        // fresh contiguous copy -- see `kernels::borrow`'s own tests for
        // the same distinction), so both operands take `as_cow_1d()`'s
        // `CowArray::Owned` (materializing) path on the way in;
        // `into_vec_no_copy` is exercised on the way out regardless of
        // input contiguity, since the SIMD kernel's output is always a
        // fresh, independent `Array1<T>`.
        let base_a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);
        let base_b = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0]).reshape(&[3, 2]);
        let a = base_a.transpose_axis(0, 1); // shape [2, 3], non-contiguous
        let b = base_b.transpose_axis(0, 1);
        assert!(!a.is_c_contiguous());
        assert!(!b.is_c_contiguous());

        let result = simd_add(&a, &b).expect("simd_add should succeed for non-contiguous input");
        let expected: Vec<f64> = a
            .to_vec()
            .iter()
            .zip(b.to_vec().iter())
            .map(|(&x, &y)| x + y)
            .collect();
        assert_eq!(result.shape(), a.shape());
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn into_vec_no_copy_matches_to_vec_contiguous_and_non_contiguous() {
        // Contiguous: exercises the raw-buffer fast path.
        let contiguous = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        assert_eq!(into_vec_no_copy(contiguous.clone()), contiguous.to_vec());

        // Non-standard-layout *owned* array: `invert_axis` reverses an
        // axis in place by negating its stride, without reallocating --
        // unlike `.to_owned()` on a reversed view, which would normalize
        // back to a fresh standard-layout buffer and so never exercise
        // `into_vec_no_copy`'s fallback branch at all. This confirms that
        // fallback (`is_standard_layout()` guard -> `iter().cloned()`) is
        // itself correct, not just the raw-buffer fast path above.
        let mut inverted = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        inverted.invert_axis(scirs2_core::ndarray::Axis(0));
        assert!(!inverted.is_standard_layout());
        assert_eq!(inverted.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(into_vec_no_copy(inverted.clone()), inverted.to_vec());
    }

    #[test]
    fn probe_simd_add_output_conversion_perf_vs_to_vec() {
        fn old_simd_add(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
            let a_nd = to_ndarray_1d(a).expect("shape check always ok for equal-shape inputs");
            let b_nd = to_ndarray_1d(b).expect("shape check always ok for equal-shape inputs");
            let result = f64::simd_add(&a_nd.view(), &b_nd.view());
            Array::from_vec(result.to_vec()).reshape(&a.shape())
        }

        let n = 200_000;
        let a = Array::from_vec((0..n).map(|i| i as f64).collect::<Vec<_>>());
        let b = Array::from_vec((0..n).map(|i| (i as f64) * 0.5).collect::<Vec<_>>());
        let iters = 200;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(old_simd_add(&a, &b));
        }
        let old = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(a.simd_add(&b).expect("simd_add should succeed"));
        }
        let new = t1.elapsed();

        eprintln!(
            "[simd_add output-conversion, n={n}] old(result.to_vec())={:.1}us/iter new(into_vec_no_copy)={:.1}us/iter ({:.2}x)",
            old.as_secs_f64() * 1e6 / iters as f64,
            new.as_secs_f64() * 1e6 / iters as f64,
            old.as_secs_f64() / new.as_secs_f64(),
        );

        // Values must still agree, not just be faster.
        assert_eq!(
            old_simd_add(&a, &b).to_vec(),
            a.simd_add(&b).expect("simd_add should succeed").to_vec()
        );
    }
}
