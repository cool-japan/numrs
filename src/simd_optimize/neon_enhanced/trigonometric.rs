//! Trigonometric operations using NEON SIMD
//!
//! This module provides optimized sin, cos, tan, and related functions
//! for ARM NEON.

use crate::array::Array;

use super::core::{NeonEnhancedOps, NEON_F32_LANES, NEON_F64_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// NEON f32 Trigonometric Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON trigonometric functions
    #[cfg(target_arch = "aarch64")]
    pub fn neon_sin_cos_f32(input: &Array<f32>) -> (Array<f32>, Array<f32>) {
        let data = input.to_vec();
        let mut sin_result = vec![0.0f32; data.len()];
        let mut cos_result = vec![0.0f32; data.len()];

        unsafe {
            Self::vectorized_sin_cos_neon_f32(&data, &mut sin_result, &mut cos_result);
        }

        (
            Array::from_vec_shape(sin_result, &input.shape()).unwrap_or_else(|e| panic!("{e}")),
            Array::from_vec_shape(cos_result, &input.shape()).unwrap_or_else(|e| panic!("{e}")),
        )
    }

    /// NEON simultaneous sin/cos computation
    #[cfg(target_arch = "aarch64")]
    unsafe fn vectorized_sin_cos_neon_f32(
        input: &[f32],
        sin_output: &mut [f32],
        cos_output: &mut [f32],
    ) {
        let len = input.len();
        let simd_len = len & !(NEON_F32_LANES - 1);

        let _pi = vdupq_n_f32(std::f32::consts::PI);
        let _two_pi = vdupq_n_f32(2.0 * std::f32::consts::PI);
        let _pi_2 = vdupq_n_f32(std::f32::consts::PI / 2.0);
        let one = vdupq_n_f32(1.0);
        let _zero = vdupq_n_f32(0.0);

        // Taylor series coefficients
        let sin_c3 = vdupq_n_f32(-1.0 / 6.0);
        let sin_c5 = vdupq_n_f32(1.0 / 120.0);
        let sin_c7 = vdupq_n_f32(-1.0 / 5040.0);

        let cos_c2 = vdupq_n_f32(-1.0 / 2.0);
        let cos_c4 = vdupq_n_f32(1.0 / 24.0);
        let cos_c6 = vdupq_n_f32(-1.0 / 720.0);

        for i in (0..simd_len).step_by(NEON_F32_LANES) {
            let mut x = vld1q_f32(input.as_ptr().add(i));

            // Range reduction (simplified)
            let mut temp_x = [0.0f32; NEON_F32_LANES];
            vst1q_f32(temp_x.as_mut_ptr(), x);

            for j in 0..NEON_F32_LANES {
                temp_x[j] %= 2.0 * std::f32::consts::PI;
                if temp_x[j] > std::f32::consts::PI {
                    temp_x[j] -= 2.0 * std::f32::consts::PI;
                }
            }

            x = vld1q_f32(temp_x.as_ptr());

            // Compute powers of x
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            let x4 = vmulq_f32(x3, x);
            let x5 = vmulq_f32(x4, x);
            let x6 = vmulq_f32(x5, x);
            let x7 = vmulq_f32(x6, x);

            // Taylor series for sin(x)
            let sin_poly = vfmaq_f32(vfmaq_f32(vfmaq_f32(x, sin_c3, x3), sin_c5, x5), sin_c7, x7);

            // Taylor series for cos(x)
            let cos_poly = vfmaq_f32(
                vfmaq_f32(vfmaq_f32(one, cos_c2, x2), cos_c4, x4),
                cos_c6,
                x6,
            );

            vst1q_f32(sin_output.as_mut_ptr().add(i), sin_poly);
            vst1q_f32(cos_output.as_mut_ptr().add(i), cos_poly);
        }

        // Handle remaining elements
        for i in simd_len..len {
            sin_output[i] = input[i].sin();
            cos_output[i] = input[i].cos();
        }
    }
}

// =============================================================================
// NEON f64 Trigonometric Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON vectorized sine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sin_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let _two_pi = vdupq_n_f64(2.0 * std::f64::consts::PI);
            let _pi = vdupq_n_f64(std::f64::consts::PI);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let mut x = vld1q_f64(data.as_ptr().add(i));

                // Range reduction to [-pi, pi]
                let mut temp = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp.as_mut_ptr(), x);
                for j in 0..NEON_F64_LANES {
                    temp[j] = temp[j].rem_euclid(2.0 * std::f64::consts::PI);
                    if temp[j] > std::f64::consts::PI {
                        temp[j] -= 2.0 * std::f64::consts::PI;
                    }
                }
                x = vld1q_f64(temp.as_ptr());

                // Taylor series: sin(x) ~ x - x^3/3! + x^5/5! - x^7/7!
                let x2 = vmulq_f64(x, x);
                let x3 = vmulq_f64(x2, x);
                let x5 = vmulq_f64(x3, x2);
                let x7 = vmulq_f64(x5, x2);

                let c3 = vdupq_n_f64(-1.0 / 6.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c7 = vdupq_n_f64(-1.0 / 5040.0);

                let res = vfmaq_f64(vfmaq_f64(vfmaq_f64(x, c3, x3), c5, x5), c7, x7);
                vst1q_f64(result.as_mut_ptr().add(i), res);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].sin();
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized cosine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_cos_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let one = vdupq_n_f64(1.0);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let mut x = vld1q_f64(data.as_ptr().add(i));

                // Range reduction
                let mut temp = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp.as_mut_ptr(), x);
                for j in 0..NEON_F64_LANES {
                    temp[j] = temp[j].rem_euclid(2.0 * std::f64::consts::PI);
                    if temp[j] > std::f64::consts::PI {
                        temp[j] -= 2.0 * std::f64::consts::PI;
                    }
                }
                x = vld1q_f64(temp.as_ptr());

                // Taylor series: cos(x) ~ 1 - x^2/2! + x^4/4! - x^6/6!
                let x2 = vmulq_f64(x, x);
                let x4 = vmulq_f64(x2, x2);
                let x6 = vmulq_f64(x4, x2);

                let c2 = vdupq_n_f64(-0.5);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c6 = vdupq_n_f64(-1.0 / 720.0);

                let res = vfmaq_f64(vfmaq_f64(vfmaq_f64(one, c2, x2), c4, x4), c6, x6);
                vst1q_f64(result.as_mut_ptr().add(i), res);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].cos();
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized tangent for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_tan_f64(input: &Array<f64>) -> Array<f64> {
        // tan(x) = sin(x) / cos(x)
        let sin_result = Self::vectorized_sin_f64(input);
        let cos_result = Self::vectorized_cos_f64(input);

        let sin_data = sin_result.to_vec();
        let cos_data = cos_result.to_vec();
        let mut result = vec![0.0f64; sin_data.len()];
        let len = sin_data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let s = vld1q_f64(sin_data.as_ptr().add(i));
                let c = vld1q_f64(cos_data.as_ptr().add(i));
                let t = vdivq_f64(s, c);
                vst1q_f64(result.as_mut_ptr().add(i), t);
            }
        }

        for i in simd_len..len {
            result[i] = sin_data[i] / cos_data[i];
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized inverse sine (arcsin) for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_asin_f64(input: &Array<f64>) -> Array<f64> {
        // Use scalar fallback with SIMD for simple operations
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.asin()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized inverse cosine (arccos) for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_acos_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.acos()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized inverse tangent (arctan) for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_atan_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.atan()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized atan2 for f64: atan(y/x) with proper quadrant
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_atan2_f64(y: &Array<f64>, x: &Array<f64>) -> Array<f64> {
        let data_y = y.to_vec();
        let data_x = x.to_vec();
        let len = data_y.len().min(data_x.len());
        let result: Vec<f64> = (0..len).map(|i| data_y[i].atan2(data_x[i])).collect();
        Array::from_vec_shape(result, &y.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized hyperbolic sine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sinh_f64(input: &Array<f64>) -> Array<f64> {
        // sinh(x) = (exp(x) - exp(-x)) / 2
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let half = vdupq_n_f64(0.5);
            let _neg_one = vdupq_n_f64(-1.0);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let x = vld1q_f64(data.as_ptr().add(i));

                // Compute exp(x) and exp(-x) using scalar for accuracy
                let mut temp_x = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp_x.as_mut_ptr(), x);

                let mut exp_x = [0.0f64; NEON_F64_LANES];
                let mut exp_neg_x = [0.0f64; NEON_F64_LANES];
                for j in 0..NEON_F64_LANES {
                    exp_x[j] = temp_x[j].exp();
                    exp_neg_x[j] = (-temp_x[j]).exp();
                }

                let vexp_x = vld1q_f64(exp_x.as_ptr());
                let vexp_neg_x = vld1q_f64(exp_neg_x.as_ptr());
                let diff = vsubq_f64(vexp_x, vexp_neg_x);
                let res = vmulq_f64(diff, half);
                vst1q_f64(result.as_mut_ptr().add(i), res);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].sinh();
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized hyperbolic cosine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_cosh_f64(input: &Array<f64>) -> Array<f64> {
        // cosh(x) = (exp(x) + exp(-x)) / 2
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let half = vdupq_n_f64(0.5);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let x = vld1q_f64(data.as_ptr().add(i));

                let mut temp_x = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp_x.as_mut_ptr(), x);

                let mut exp_x = [0.0f64; NEON_F64_LANES];
                let mut exp_neg_x = [0.0f64; NEON_F64_LANES];
                for j in 0..NEON_F64_LANES {
                    exp_x[j] = temp_x[j].exp();
                    exp_neg_x[j] = (-temp_x[j]).exp();
                }

                let vexp_x = vld1q_f64(exp_x.as_ptr());
                let vexp_neg_x = vld1q_f64(exp_neg_x.as_ptr());
                let sum = vaddq_f64(vexp_x, vexp_neg_x);
                let res = vmulq_f64(sum, half);
                vst1q_f64(result.as_mut_ptr().add(i), res);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].cosh();
        }

        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized hyperbolic tangent for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_tanh_f64(input: &Array<f64>) -> Array<f64> {
        // tanh(x) = sinh(x) / cosh(x)
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.tanh()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized inverse hyperbolic sine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_asinh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.asinh()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized inverse hyperbolic cosine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_acosh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.acosh()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// NEON vectorized inverse hyperbolic tangent for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_atanh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.atanh()).collect();
        Array::from_vec_shape(result, &input.shape()).unwrap_or_else(|e| panic!("{e}"))
    }
}

// =============================================================================
// Non-aarch64 Fallback Implementations
// =============================================================================

#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn neon_sin_cos_f32(input: &Array<f32>) -> (Array<f32>, Array<f32>) {
        (input.map(|x| x.sin()), input.map(|x| x.cos()))
    }

    pub fn vectorized_sin_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.sin())
    }

    pub fn vectorized_cos_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.cos())
    }

    pub fn vectorized_tan_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.tan())
    }

    pub fn vectorized_asin_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.asin())
    }

    pub fn vectorized_acos_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.acos())
    }

    pub fn vectorized_atan_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.atan())
    }

    pub fn vectorized_atan2_f64(y: &Array<f64>, x: &Array<f64>) -> Array<f64> {
        let data_y = y.to_vec();
        let data_x = x.to_vec();
        let len = data_y.len().min(data_x.len());
        let result: Vec<f64> = (0..len).map(|i| data_y[i].atan2(data_x[i])).collect();
        Array::from_vec_shape(result, &y.shape())?
    }

    pub fn vectorized_sinh_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.sinh())
    }

    pub fn vectorized_cosh_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.cosh())
    }

    pub fn vectorized_tanh_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.tanh())
    }

    pub fn vectorized_asinh_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.asinh())
    }

    pub fn vectorized_acosh_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.acosh())
    }

    pub fn vectorized_atanh_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.atanh())
    }
}
