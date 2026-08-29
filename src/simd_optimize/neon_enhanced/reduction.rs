//! Reduction operations using NEON SIMD
//!
//! This module provides optimized sum, prod, max, min, mean, variance, std,
//! norm, and dot product functions for ARM NEON.

use crate::array::Array;
use crate::error::Result;
// See `matmul.rs`: only the aarch64 SIMD implementation raises this.
#[cfg(target_arch = "aarch64")]
use crate::error::NumRs2Error;

use super::core::NeonEnhancedOps;
#[cfg(target_arch = "aarch64")]
use super::core::{NEON_F32_LANES, NEON_F64_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// NEON f32 Reduction Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON optimized sum reduction
    #[cfg(target_arch = "aarch64")]
    pub fn neon_sum_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        unsafe { Self::reduction_sum_neon_f32(&data) }
    }

    /// NEON sum reduction implementation
    #[cfg(target_arch = "aarch64")]
    unsafe fn reduction_sum_neon_f32(input: &[f32]) -> f32 {
        let len = input.len();
        let simd_len = len & !(NEON_F32_LANES * 4 - 1);

        // Use multiple accumulators
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for i in (0..simd_len).step_by(NEON_F32_LANES * 4) {
            let v0 = vld1q_f32(input.as_ptr().add(i));
            let v1 = vld1q_f32(input.as_ptr().add(i + NEON_F32_LANES));
            let v2 = vld1q_f32(input.as_ptr().add(i + NEON_F32_LANES * 2));
            let v3 = vld1q_f32(input.as_ptr().add(i + NEON_F32_LANES * 3));

            acc0 = vaddq_f32(acc0, v0);
            acc1 = vaddq_f32(acc1, v1);
            acc2 = vaddq_f32(acc2, v2);
            acc3 = vaddq_f32(acc3, v3);
        }

        // Combine accumulators
        let combined01 = vaddq_f32(acc0, acc1);
        let combined23 = vaddq_f32(acc2, acc3);
        let total = vaddq_f32(combined01, combined23);

        // Horizontal sum
        let sum2 = vpadd_f32(vget_low_f32(total), vget_high_f32(total));
        let sum1 = vpadd_f32(sum2, sum2);
        let mut result = vget_lane_f32(sum1, 0);

        // Handle remaining elements
        for &item in &input[simd_len..] {
            result += item;
        }

        result
    }

    /// NEON dot product optimization
    #[cfg(target_arch = "aarch64")]
    pub fn neon_dot_f32(a: &Array<f32>, b: &Array<f32>) -> Result<f32> {
        if a.shape() != b.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a.shape(),
                actual: b.shape(),
            });
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();

        unsafe { Ok(Self::dot_product_neon_f32(&a_data, &b_data)) }
    }

    /// NEON dot product implementation
    #[cfg(target_arch = "aarch64")]
    unsafe fn dot_product_neon_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let simd_len = len & !(NEON_F32_LANES - 1);

        let mut acc = vdupq_n_f32(0.0);

        for i in (0..simd_len).step_by(NEON_F32_LANES) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
        }

        // Horizontal sum
        let sum2 = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum1 = vpadd_f32(sum2, sum2);
        let mut result = vget_lane_f32(sum1, 0);

        // Handle remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }

        result
    }
}

// =============================================================================
// NEON f64 Reduction Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON vectorized sum reduction for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sum_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum = unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vacc = vaddq_f64(vacc, v);
            }
            // Horizontal add: sum both lanes
            vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1)
        };

        for i in simd_len..len {
            sum += data[i];
        }

        sum
    }

    /// NEON vectorized product reduction for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_prod_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut prod = unsafe {
            let mut vacc = vdupq_n_f64(1.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vacc = vmulq_f64(vacc, v);
            }
            // Horizontal multiply: multiply both lanes
            vgetq_lane_f64(vacc, 0) * vgetq_lane_f64(vacc, 1)
        };

        for i in simd_len..len {
            prod *= data[i];
        }

        prod
    }

    /// NEON vectorized max reduction for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_max_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return f64::NEG_INFINITY;
        }

        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut max_val = unsafe {
            let mut vmax = vdupq_n_f64(f64::NEG_INFINITY);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vmax = vmaxq_f64(vmax, v);
            }
            // Horizontal max
            let lane0 = vgetq_lane_f64(vmax, 0);
            let lane1 = vgetq_lane_f64(vmax, 1);
            lane0.max(lane1)
        };

        for i in simd_len..len {
            max_val = max_val.max(data[i]);
        }

        max_val
    }

    /// NEON vectorized min reduction for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_min_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return f64::INFINITY;
        }

        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut min_val = unsafe {
            let mut vmin = vdupq_n_f64(f64::INFINITY);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vmin = vminq_f64(vmin, v);
            }
            // Horizontal min
            let lane0 = vgetq_lane_f64(vmin, 0);
            let lane1 = vgetq_lane_f64(vmin, 1);
            lane0.min(lane1)
        };

        for i in simd_len..len {
            min_val = min_val.min(data[i]);
        }

        min_val
    }

    /// NEON vectorized mean for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_mean_f64(input: &Array<f64>) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }
        Self::vectorized_sum_f64(input) / (len as f64)
    }

    /// NEON vectorized dot product for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_dot_f64(a: &Array<f64>, b: &Array<f64>) -> f64 {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum = unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                // FMA: acc + a * b
                vacc = vfmaq_f64(vacc, va, vb);
            }
            // Horizontal add
            vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1)
        };

        for i in simd_len..len {
            sum += data_a[i] * data_b[i];
        }

        sum
    }

    /// NEON vectorized L2 norm for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_norm_l2_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum_sq = unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                // FMA: acc + v * v
                vacc = vfmaq_f64(vacc, v, v);
            }
            // Horizontal add
            vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1)
        };

        for i in simd_len..len {
            sum_sq += data[i] * data[i];
        }

        sum_sq.sqrt()
    }

    /// NEON vectorized L1 norm for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_norm_l1_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum_abs = unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let vabs = vabsq_f64(v);
                vacc = vaddq_f64(vacc, vabs);
            }
            // Horizontal add
            vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1)
        };

        for i in simd_len..len {
            sum_abs += data[i].abs();
        }

        sum_abs
    }

    /// NEON vectorized variance for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_variance_f64(input: &Array<f64>) -> f64 {
        let mean = Self::vectorized_mean_f64(input);
        let data = input.to_vec();
        let len = data.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum_sq_diff = unsafe {
            let vmean = vdupq_n_f64(mean);
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let diff = vsubq_f64(v, vmean);
                // FMA: acc + diff * diff
                vacc = vfmaq_f64(vacc, diff, diff);
            }
            vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1)
        };

        for i in simd_len..len {
            let diff = data[i] - mean;
            sum_sq_diff += diff * diff;
        }

        sum_sq_diff / (len as f64)
    }

    /// NEON vectorized standard deviation for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_std_f64(input: &Array<f64>) -> f64 {
        Self::vectorized_variance_f64(input).sqrt()
    }
}

// =============================================================================
// Non-aarch64 Fallback Implementations
// =============================================================================

#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn neon_sum_f32(input: &Array<f32>) -> f32 {
        input.sum()
    }

    pub fn neon_dot_f32(a: &Array<f32>, b: &Array<f32>) -> Result<f32> {
        a.dot(b)
    }

    pub fn vectorized_sum_f64(input: &Array<f64>) -> f64 {
        input.sum()
    }

    pub fn vectorized_prod_f64(input: &Array<f64>) -> f64 {
        input.product()
    }

    pub fn vectorized_max_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn vectorized_min_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn vectorized_mean_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        if len == 0 {
            return 0.0;
        }
        data.iter().sum::<f64>() / len as f64
    }

    pub fn vectorized_dot_f64(a: &Array<f64>, b: &Array<f64>) -> f64 {
        a.dot(b).unwrap_or(0.0)
    }

    pub fn vectorized_norm_l2_f64(input: &Array<f64>) -> f64 {
        input.to_vec().iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn vectorized_norm_l1_f64(input: &Array<f64>) -> f64 {
        input.to_vec().iter().map(|x| x.abs()).sum()
    }

    pub fn vectorized_variance_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        if len == 0 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / len as f64;
        let sum_sq_diff: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        sum_sq_diff / (len as f64)
    }

    pub fn vectorized_std_f64(input: &Array<f64>) -> f64 {
        Self::vectorized_variance_f64(input).sqrt()
    }
}
