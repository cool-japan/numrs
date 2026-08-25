//! Arithmetic operations using NEON SIMD
//!
//! This module provides optimized array and scalar arithmetic operations
//! for ARM NEON including add, subtract, multiply, divide, FMA, and negation.

use crate::array::Array;

use super::core::{NeonEnhancedOps, NEON_F64_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// NEON f64 Array Arithmetic Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON vectorized array addition for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_add_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vsum = vaddq_f64(va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vsum);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i] + data_b[i];
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized array subtraction for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sub_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vdiff = vsubq_f64(va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vdiff);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i] - data_b[i];
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized array multiplication for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_mul_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vprod = vmulq_f64(va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vprod);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i] * data_b[i];
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized array division for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_div_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vquot = vdivq_f64(va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vquot);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i] / data_b[i];
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    // =========================================================================
    // Scalar Operations
    // =========================================================================

    /// NEON vectorized scalar addition for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_add_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let data = a.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let vscalar = vdupq_n_f64(scalar);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data.as_ptr().add(i));
                let vsum = vaddq_f64(va, vscalar);
                vst1q_f64(result.as_mut_ptr().add(i), vsum);
            }
        }

        for i in simd_len..len {
            result[i] = data[i] + scalar;
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized scalar multiplication for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_mul_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let data = a.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let vscalar = vdupq_n_f64(scalar);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data.as_ptr().add(i));
                let vprod = vmulq_f64(va, vscalar);
                vst1q_f64(result.as_mut_ptr().add(i), vprod);
            }
        }

        for i in simd_len..len {
            result[i] = data[i] * scalar;
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized subtract scalar for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sub_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let data = a.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let vscalar = vdupq_n_f64(scalar);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data.as_ptr().add(i));
                let vdiff = vsubq_f64(va, vscalar);
                vst1q_f64(result.as_mut_ptr().add(i), vdiff);
            }
        }

        for i in simd_len..len {
            result[i] = data[i] - scalar;
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized divide scalar for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_div_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let data = a.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let vscalar = vdupq_n_f64(scalar);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data.as_ptr().add(i));
                let vquot = vdivq_f64(va, vscalar);
                vst1q_f64(result.as_mut_ptr().add(i), vquot);
            }
        }

        for i in simd_len..len {
            result[i] = data[i] / scalar;
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    // =========================================================================
    // FMA and Negation
    // =========================================================================

    /// NEON vectorized fused multiply-add for f64: a * b + c
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_fma_f64(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let data_c = c.to_vec();
        let len = data_a.len().min(data_b.len()).min(data_c.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vc = vld1q_f64(data_c.as_ptr().add(i));
                // FMA: a * b + c
                let vfma = vfmaq_f64(vc, va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vfma);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i].mul_add(data_b[i], data_c[i]);
        }

        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized negation for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_negative_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let vneg = vnegq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), vneg);
            }
        }

        for i in simd_len..len {
            result[i] = -data[i];
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized reciprocal for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_reciprocal_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let one = vdupq_n_f64(1.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let vrecip = vdivq_f64(one, v);
                vst1q_f64(result.as_mut_ptr().add(i), vrecip);
            }
        }

        for i in simd_len..len {
            result[i] = 1.0 / data[i];
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized square for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_square_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let sq_v = vmulq_f64(v, v);
                vst1q_f64(result.as_mut_ptr().add(i), sq_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i] * data[i];
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }
}

// =============================================================================
// Non-aarch64 Fallback Implementations
// =============================================================================

#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn vectorized_add_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        a.add(b)
    }

    pub fn vectorized_sub_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        a.subtract(b)
    }

    pub fn vectorized_mul_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        a.multiply(b)
    }

    pub fn vectorized_div_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        a.divide(b)
    }

    pub fn vectorized_add_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x + scalar)
    }

    pub fn vectorized_mul_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x * scalar)
    }

    pub fn vectorized_sub_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x - scalar)
    }

    pub fn vectorized_div_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x / scalar)
    }

    pub fn vectorized_fma_f64(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let data_c = c.to_vec();
        let len = data_a.len().min(data_b.len()).min(data_c.len());
        let result: Vec<f64> = (0..len)
            .map(|i| data_a[i].mul_add(data_b[i], data_c[i]))
            .collect();
        Array::from_vec_shape(result, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn vectorized_negative_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| -x)
    }

    pub fn vectorized_reciprocal_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| 1.0 / x)
    }

    pub fn vectorized_square_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x * x)
    }
}
