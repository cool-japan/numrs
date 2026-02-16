//! Comparison and sign operations using NEON SIMD
//!
//! This module provides optimized abs, sign, maximum, minimum, clamp,
//! and copysign functions for ARM NEON.

use crate::array::Array;

use super::core::{NeonEnhancedOps, NEON_F64_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// NEON f64 Comparison Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON vectorized absolute value for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_abs_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let abs_v = vabsq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), abs_v);
            }
        }

        // Scalar fallback for remaining elements
        for i in simd_len..len {
            result[i] = data[i].abs();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized sign for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sign_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let zero = vdupq_n_f64(0.0);
            let _one = vdupq_n_f64(1.0);
            let _neg_one = vdupq_n_f64(-1.0);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));

                // Compare v > 0 and v < 0
                let gt_zero = vcgtq_f64(v, zero);
                let lt_zero = vcltq_f64(v, zero);

                // Select: if v > 0 -> 1, if v < 0 -> -1, else 0
                let pos_mask = vreinterpretq_f64_u64(gt_zero);
                let neg_mask = vreinterpretq_f64_u64(lt_zero);

                let mut temp = [0.0f64; NEON_F64_LANES];
                let mut _temp_pos = [0.0f64; NEON_F64_LANES];
                let mut _temp_neg = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp.as_mut_ptr(), v);
                vst1q_f64(_temp_pos.as_mut_ptr(), pos_mask);
                vst1q_f64(_temp_neg.as_mut_ptr(), neg_mask);

                for j in 0..NEON_F64_LANES {
                    if temp[j] > 0.0 {
                        result[i + j] = 1.0;
                    } else if temp[j] < 0.0 {
                        result[i + j] = -1.0;
                    } else {
                        result[i + j] = 0.0;
                    }
                }
            }
        }

        for i in simd_len..len {
            result[i] = if data[i] > 0.0 {
                1.0
            } else if data[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized element-wise maximum for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_maximum_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vmax = vmaxq_f64(va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vmax);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i].max(data_b[i]);
        }

        Array::from_vec(result).reshape(&a.shape())
    }

    /// NEON vectorized element-wise minimum for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_minimum_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                let vmin = vminq_f64(va, vb);
                vst1q_f64(result.as_mut_ptr().add(i), vmin);
            }
        }

        for i in simd_len..len {
            result[i] = data_a[i].min(data_b[i]);
        }

        Array::from_vec(result).reshape(&a.shape())
    }

    /// NEON vectorized clamp for f64: clamp values to [min_val, max_val]
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_clamp_f64(input: &Array<f64>, min_val: f64, max_val: f64) -> Array<f64> {
        let data = input.to_vec();
        let len = data.len();
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let vmin = vdupq_n_f64(min_val);
            let vmax = vdupq_n_f64(max_val);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                // Clamp: max(min_val, min(max_val, v))
                let v_clamped = vmaxq_f64(vmin, vminq_f64(vmax, v));
                vst1q_f64(result.as_mut_ptr().add(i), v_clamped);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].clamp(min_val, max_val);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized copysign for f64: magnitude from a, sign from b
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_copysign_f64(magnitude: &Array<f64>, sign: &Array<f64>) -> Array<f64> {
        let data_mag = magnitude.to_vec();
        let data_sign = sign.to_vec();
        let len = data_mag.len().min(data_sign.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let vm = vld1q_f64(data_mag.as_ptr().add(i));
                let vs = vld1q_f64(data_sign.as_ptr().add(i));

                // Get absolute value of magnitude
                let abs_m = vabsq_f64(vm);

                // Apply sign from sign array
                // For each lane, copysign
                let lane0 = vgetq_lane_f64(abs_m, 0).copysign(vgetq_lane_f64(vs, 0));
                let lane1 = vgetq_lane_f64(abs_m, 1).copysign(vgetq_lane_f64(vs, 1));

                result[i] = lane0;
                result[i + 1] = lane1;
            }
        }

        for i in simd_len..len {
            result[i] = data_mag[i].abs().copysign(data_sign[i]);
        }

        Array::from_vec(result).reshape(&magnitude.shape())
    }
}

// =============================================================================
// Non-aarch64 Fallback Implementations
// =============================================================================

#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn vectorized_abs_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.abs())
    }

    pub fn vectorized_sign_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    pub fn vectorized_maximum_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let result: Vec<f64> = (0..len).map(|i| data_a[i].max(data_b[i])).collect();
        Array::from_vec(result).reshape(&a.shape())
    }

    pub fn vectorized_minimum_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let result: Vec<f64> = (0..len).map(|i| data_a[i].min(data_b[i])).collect();
        Array::from_vec(result).reshape(&a.shape())
    }

    pub fn vectorized_clamp_f64(input: &Array<f64>, min_val: f64, max_val: f64) -> Array<f64> {
        input.map(|x| x.clamp(min_val, max_val))
    }

    pub fn vectorized_copysign_f64(magnitude: &Array<f64>, sign: &Array<f64>) -> Array<f64> {
        let data_mag = magnitude.to_vec();
        let data_sign = sign.to_vec();
        let len = data_mag.len().min(data_sign.len());
        let result: Vec<f64> = (0..len)
            .map(|i| data_mag[i].abs().copysign(data_sign[i]))
            .collect();
        Array::from_vec(result).reshape(&magnitude.shape())
    }
}
