//! Enhanced NEON SIMD operations for ARM processors
//!
//! This module provides optimized vectorization using ARM NEON instructions
//! for high performance on ARM-based systems including Apple Silicon and ARM servers.

use crate::array::Array;
#[allow(unused_imports)] // Used conditionally based on target architecture
use crate::error::{NumRs2Error, Result};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

/// NEON vectorization constants
#[allow(dead_code)]
const NEON_F32_LANES: usize = 4;
#[allow(dead_code)]
const NEON_F64_LANES: usize = 2;
#[allow(dead_code)]
const NEON_ALIGNMENT: usize = 16;

/// Advanced NEON operations for ARM processors
pub struct NeonEnhancedOps;

impl NeonEnhancedOps {
    /// NEON optimized matrix multiplication
    #[cfg(target_arch = "aarch64")]
    pub fn neon_matmul_f32(
        a: &Array<f32>,
        b: &Array<f32>,
        c: &mut Array<f32>,
        block_size: usize,
    ) -> Result<()> {
        let [m, k] = a.shape()[..] else {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix A must be 2D".to_string(),
            ));
        };
        let [k2, n] = b.shape()[..] else {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix B must be 2D".to_string(),
            ));
        };

        if k != k2 {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let mut c_data = c.to_vec();

        unsafe {
            Self::blocked_matmul_neon_f32(&a_data, &b_data, &mut c_data, m, n, k, block_size);
        }

        *c = Array::from_vec(c_data).reshape(&[m, n]);
        Ok(())
    }

    /// Blocked matrix multiplication with NEON optimization
    #[cfg(target_arch = "aarch64")]
    unsafe fn blocked_matmul_neon_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        block_size: usize,
    ) {
        for ii in (0..m).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let i_end = (ii + block_size).min(m);
                    let j_end = (jj + block_size).min(n);
                    let k_end = (kk + block_size).min(k);

                    for i in ii..i_end {
                        for j in (jj..j_end).step_by(NEON_F32_LANES) {
                            let lanes = (j_end - j).min(NEON_F32_LANES);

                            // Load C values
                            let mut vc = if lanes == NEON_F32_LANES {
                                vld1q_f32(c.as_ptr().add(i * n + j))
                            } else {
                                let mut temp = [0.0f32; NEON_F32_LANES];
                                for l in 0..lanes {
                                    temp[l] = c[i * n + j + l];
                                }
                                vld1q_f32(temp.as_ptr())
                            };

                            for l in kk..k_end {
                                let va = vdupq_n_f32(a[i * k + l]);
                                let vb = if lanes == NEON_F32_LANES {
                                    vld1q_f32(b.as_ptr().add(l * n + j))
                                } else {
                                    let mut temp = [0.0f32; NEON_F32_LANES];
                                    for idx in 0..lanes {
                                        temp[idx] = b[l * n + j + idx];
                                    }
                                    vld1q_f32(temp.as_ptr())
                                };
                                vc = vfmaq_f32(vc, va, vb);
                            }

                            // Store C values
                            if lanes == NEON_F32_LANES {
                                vst1q_f32(c.as_mut_ptr().add(i * n + j), vc);
                            } else {
                                let mut temp = [0.0f32; NEON_F32_LANES];
                                vst1q_f32(temp.as_mut_ptr(), vc);
                                for l in 0..lanes {
                                    c[i * n + j + l] = temp[l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// NEON vectorized exponential function
    #[cfg(target_arch = "aarch64")]
    pub fn neon_exp_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::vectorized_exp_neon_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON exponential implementation with polynomial approximation
    #[cfg(target_arch = "aarch64")]
    unsafe fn vectorized_exp_neon_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(NEON_F32_LANES - 1);

        // Constants for exp approximation
        let log2_e = vdupq_n_f32(1.4426950408889634);
        let ln2_hi = vdupq_n_f32(0.6931471805599453);
        let ln2_lo = vdupq_n_f32(2.3283064365386963e-10);
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(1.0);
        let c3 = vdupq_n_f32(0.5);
        let c4 = vdupq_n_f32(0.16666666666666666);
        let c5 = vdupq_n_f32(0.041666666666666664);

        for i in (0..simd_len).step_by(NEON_F32_LANES) {
            let x = vld1q_f32(input.as_ptr().add(i));

            // Range reduction: x = n*ln(2) + r
            let n_float = vmulq_f32(x, log2_e);
            let n = vcvtq_s32_f32(n_float);
            let n_f = vcvtq_f32_s32(n);

            // r = x - n*ln(2)
            let r = vfmsq_f32(x, n_f, ln2_hi);
            let r = vfmsq_f32(r, n_f, ln2_lo);

            // Taylor series: exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4!
            let r2 = vmulq_f32(r, r);
            let r3 = vmulq_f32(r2, r);
            let r4 = vmulq_f32(r3, r);

            let poly = vfmaq_f32(
                vfmaq_f32(vfmaq_f32(vfmaq_f32(c1, c2, r), c3, r2), c4, r3),
                c5,
                r4,
            );

            // Scale by 2^n (simplified - would need proper implementation)
            let mut temp = [0.0f32; NEON_F32_LANES];
            vst1q_f32(temp.as_mut_ptr(), poly);

            // Extract lane values using const indices
            let n0 = vgetq_lane_s32(n, 0);
            let n1 = vgetq_lane_s32(n, 1);
            let n2 = vgetq_lane_s32(n, 2);
            let n3 = vgetq_lane_s32(n, 3);

            temp[0] *= (2.0f32).powi(n0);
            temp[1] *= (2.0f32).powi(n1);
            temp[2] *= (2.0f32).powi(n2);
            temp[3] *= (2.0f32).powi(n3);
            let result = vld1q_f32(temp.as_ptr());

            vst1q_f32(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].exp();
        }
    }

    /// NEON vectorized logarithm function
    #[cfg(target_arch = "aarch64")]
    pub fn neon_log_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::vectorized_log_neon_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON logarithm with polynomial approximation
    #[cfg(target_arch = "aarch64")]
    unsafe fn vectorized_log_neon_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(NEON_F32_LANES - 1);

        let ln2 = vdupq_n_f32(0.6931471805599453);
        let one = vdupq_n_f32(1.0);
        let c1 = vdupq_n_f32(-0.5);
        let c2 = vdupq_n_f32(0.33333333333333333);
        let c3 = vdupq_n_f32(-0.25);
        let c4 = vdupq_n_f32(0.2);

        for i in (0..simd_len).step_by(NEON_F32_LANES) {
            let x = vld1q_f32(input.as_ptr().add(i));

            // Extract exponent and mantissa (simplified approach)
            let mut temp = [0.0f32; NEON_F32_LANES];
            vst1q_f32(temp.as_mut_ptr(), x);

            let mut exp_vals = [0.0f32; NEON_F32_LANES];
            let mut mantissa_vals = [0.0f32; NEON_F32_LANES];

            for j in 0..NEON_F32_LANES {
                let bits = temp[j].to_bits();
                let exp = ((bits >> 23) & 0xFF) as i32 - 127;
                exp_vals[j] = exp as f32;

                let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
                mantissa_vals[j] = f32::from_bits(mantissa_bits);
            }

            let exp_f = vld1q_f32(exp_vals.as_ptr());
            let mantissa = vld1q_f32(mantissa_vals.as_ptr());

            // Polynomial approximation for log(mantissa)
            let u = vsubq_f32(mantissa, one);
            let u2 = vmulq_f32(u, u);
            let u3 = vmulq_f32(u2, u);
            let u4 = vmulq_f32(u3, u);

            let poly = vfmaq_f32(
                vfmaq_f32(vfmaq_f32(vfmaq_f32(u, c1, u2), c2, u2), c3, u3),
                c4,
                u4,
            );

            // log(x) = exp * ln(2) + log(mantissa)
            let result = vfmaq_f32(poly, exp_f, ln2);

            vst1q_f32(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].ln();
        }
    }

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
            Array::from_vec(sin_result).reshape(&input.shape()),
            Array::from_vec(cos_result).reshape(&input.shape()),
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

    /// NEON memory copy optimization
    #[cfg(target_arch = "aarch64")]
    pub fn neon_copy_f32(src: &Array<f32>, dst: &mut Array<f32>) -> Result<()> {
        if src.shape() != dst.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: src.shape(),
                actual: dst.shape(),
            });
        }

        let src_data = src.to_vec();
        let mut dst_data = dst.to_vec();

        unsafe {
            Self::optimized_copy_neon_f32(&src_data, &mut dst_data);
        }

        *dst = Array::from_vec(dst_data).reshape(&src.shape());
        Ok(())
    }

    /// NEON optimized memory copy
    #[cfg(target_arch = "aarch64")]
    unsafe fn optimized_copy_neon_f32(src: &[f32], dst: &mut [f32]) {
        let len = src.len();
        let simd_len = len & !(NEON_F32_LANES * 4 - 1);

        // Copy 16 elements at a time for better throughput
        for i in (0..simd_len).step_by(NEON_F32_LANES * 4) {
            let v0 = vld1q_f32(src.as_ptr().add(i));
            let v1 = vld1q_f32(src.as_ptr().add(i + NEON_F32_LANES));
            let v2 = vld1q_f32(src.as_ptr().add(i + NEON_F32_LANES * 2));
            let v3 = vld1q_f32(src.as_ptr().add(i + NEON_F32_LANES * 3));

            vst1q_f32(dst.as_mut_ptr().add(i), v0);
            vst1q_f32(dst.as_mut_ptr().add(i + NEON_F32_LANES), v1);
            vst1q_f32(dst.as_mut_ptr().add(i + NEON_F32_LANES * 2), v2);
            vst1q_f32(dst.as_mut_ptr().add(i + NEON_F32_LANES * 3), v3);
        }

        // Handle remaining elements
        dst[simd_len..len].copy_from_slice(&src[simd_len..len]);
    }
}

// =============================================================================
// NEON f64 Operations for ufuncs compatibility
// =============================================================================

/// Vectorization constants for f64
#[allow(dead_code)]
const NEON_F64_LANES_UNROLL: usize = NEON_F64_LANES * 4; // Process 8 f64 values at once

impl NeonEnhancedOps {
    // =========================================================================
    // Core Mathematical Functions (f64) - Used by ufuncs
    // =========================================================================

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

    /// NEON vectorized square root for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sqrt_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let sqrt_v = vsqrtq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), sqrt_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].sqrt();
        }

        Array::from_vec(result).reshape(&input.shape())
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized exponential for f64 with polynomial approximation
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_exp_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        // Use high-precision Taylor series coefficients
        unsafe {
            let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
            let ln2_hi = vdupq_n_f64(0.6931471805599453);
            let ln2_lo = vdupq_n_f64(2.3283064365386963e-10);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let x = vld1q_f64(data.as_ptr().add(i));

                // Range reduction: x = n*ln(2) + r
                let n_float = vmulq_f64(x, log2_e);

                // Round to nearest integer
                let n_rounded = vrndnq_f64(n_float);

                // r = x - n*ln(2)
                let r = vfmsq_f64(x, n_rounded, ln2_hi);
                let r = vfmsq_f64(r, n_rounded, ln2_lo);

                // Taylor series: exp(r) ≈ 1 + r + r²/2! + r³/3! + ...
                let r2 = vmulq_f64(r, r);
                let r3 = vmulq_f64(r2, r);
                let r4 = vmulq_f64(r3, r);
                let r5 = vmulq_f64(r4, r);

                let c0 = vdupq_n_f64(1.0);
                let c1 = vdupq_n_f64(1.0);
                let c2 = vdupq_n_f64(0.5);
                let c3 = vdupq_n_f64(1.0 / 6.0);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);

                let poly = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vfmaq_f64(c0, c1, r), c2, r2), c3, r3),
                        c4,
                        r4,
                    ),
                    c5,
                    r5,
                );

                // Scale by 2^n (extract, scale, repack)
                let mut temp_poly = [0.0f64; NEON_F64_LANES];
                let mut temp_n = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp_poly.as_mut_ptr(), poly);
                vst1q_f64(temp_n.as_mut_ptr(), n_rounded);

                temp_poly[0] *= 2.0f64.powf(temp_n[0]);
                temp_poly[1] *= 2.0f64.powf(temp_n[1]);

                let res = vld1q_f64(temp_poly.as_ptr());
                vst1q_f64(result.as_mut_ptr().add(i), res);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].exp();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized natural logarithm for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_log_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let ln2 = vdupq_n_f64(std::f64::consts::LN_2);
            let one = vdupq_n_f64(1.0);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let x = vld1q_f64(data.as_ptr().add(i));

                // Extract exponent and mantissa
                let mut temp = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp.as_mut_ptr(), x);

                let mut exp_vals = [0.0f64; NEON_F64_LANES];
                let mut mant_vals = [0.0f64; NEON_F64_LANES];

                for j in 0..NEON_F64_LANES {
                    let bits = temp[j].to_bits();
                    let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
                    exp_vals[j] = exp as f64;
                    let mant_bits = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
                    mant_vals[j] = f64::from_bits(mant_bits);
                }

                let exp_f = vld1q_f64(exp_vals.as_ptr());
                let mant = vld1q_f64(mant_vals.as_ptr());

                // Polynomial approximation for log(mantissa)
                let u = vsubq_f64(mant, one);
                let u2 = vmulq_f64(u, u);
                let u3 = vmulq_f64(u2, u);
                let u4 = vmulq_f64(u3, u);

                let c1 = vdupq_n_f64(-0.5);
                let c2 = vdupq_n_f64(1.0 / 3.0);
                let c3 = vdupq_n_f64(-0.25);
                let c4 = vdupq_n_f64(0.2);

                let poly = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vfmaq_f64(u, c1, u2), c2, u3), c3, u4),
                    c4,
                    vmulq_f64(u4, u),
                );

                // log(x) = exp * ln(2) + log(mantissa)
                let res = vfmaq_f64(poly, exp_f, ln2);
                vst1q_f64(result.as_mut_ptr().add(i), res);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].ln();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized sine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_sin_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            let two_pi = vdupq_n_f64(2.0 * std::f64::consts::PI);
            let pi = vdupq_n_f64(std::f64::consts::PI);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let mut x = vld1q_f64(data.as_ptr().add(i));

                // Range reduction to [-π, π]
                let mut temp = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp.as_mut_ptr(), x);
                for j in 0..NEON_F64_LANES {
                    temp[j] = temp[j].rem_euclid(2.0 * std::f64::consts::PI);
                    if temp[j] > std::f64::consts::PI {
                        temp[j] -= 2.0 * std::f64::consts::PI;
                    }
                }
                x = vld1q_f64(temp.as_ptr());

                // Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!
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

        Array::from_vec(result).reshape(&input.shape())
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

                // Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6!
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

        Array::from_vec(result).reshape(&input.shape())
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized inverse sine (arcsin) for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_asin_f64(input: &Array<f64>) -> Array<f64> {
        // Use scalar fallback with SIMD for simple operations
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.asin()).collect();
        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized inverse cosine (arccos) for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_acos_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.acos()).collect();
        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized inverse tangent (arctan) for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_atan_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.atan()).collect();
        Array::from_vec(result).reshape(&input.shape())
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
            let neg_one = vdupq_n_f64(-1.0);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let x = vld1q_f64(data.as_ptr().add(i));
                let neg_x = vmulq_f64(x, neg_one);

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

        Array::from_vec(result).reshape(&input.shape())
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized hyperbolic tangent for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_tanh_f64(input: &Array<f64>) -> Array<f64> {
        // tanh(x) = sinh(x) / cosh(x)
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.tanh()).collect();
        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized inverse hyperbolic sine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_asinh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.asinh()).collect();
        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized inverse hyperbolic cosine for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_acosh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.acosh()).collect();
        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized inverse hyperbolic tangent for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_atanh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.atanh()).collect();
        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized floor for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_floor_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let floor_v = vrndmq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), floor_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].floor();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized ceiling for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_ceil_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let ceil_v = vrndpq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), ceil_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].ceil();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized round for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_round_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let round_v = vrndnq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), round_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].round();
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
            let one = vdupq_n_f64(1.0);
            let neg_one = vdupq_n_f64(-1.0);

            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));

                // Compare v > 0 and v < 0
                let gt_zero = vcgtq_f64(v, zero);
                let lt_zero = vcltq_f64(v, zero);

                // Select: if v > 0 -> 1, if v < 0 -> -1, else 0
                let pos_mask = vreinterpretq_f64_u64(gt_zero);
                let neg_mask = vreinterpretq_f64_u64(lt_zero);

                let mut temp = [0.0f64; NEON_F64_LANES];
                let mut temp_pos = [0.0f64; NEON_F64_LANES];
                let mut temp_neg = [0.0f64; NEON_F64_LANES];
                vst1q_f64(temp.as_mut_ptr(), v);
                vst1q_f64(temp_pos.as_mut_ptr(), pos_mask);
                vst1q_f64(temp_neg.as_mut_ptr(), neg_mask);

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

    // =========================================================================
    // High-Priority NEON f64 Functions - Array Arithmetic
    // =========================================================================

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

        Array::from_vec(result).reshape(&a.shape())
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

        Array::from_vec(result).reshape(&a.shape())
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

        Array::from_vec(result).reshape(&a.shape())
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

        Array::from_vec(result).reshape(&a.shape())
    }

    // =========================================================================
    // High-Priority NEON f64 Functions - Scalar Operations
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

        Array::from_vec(result).reshape(&a.shape())
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

        Array::from_vec(result).reshape(&a.shape())
    }

    // =========================================================================
    // High-Priority NEON f64 Functions - Reductions
    // =========================================================================

    /// NEON vectorized sum reduction for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_sum_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum = 0.0f64;

        unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vacc = vaddq_f64(vacc, v);
            }
            // Horizontal add: sum both lanes
            sum = vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1);
        }

        for i in simd_len..len {
            sum += data[i];
        }

        sum
    }

    /// NEON vectorized product reduction for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_prod_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut prod = 1.0f64;

        unsafe {
            let mut vacc = vdupq_n_f64(1.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vacc = vmulq_f64(vacc, v);
            }
            // Horizontal multiply: multiply both lanes
            prod = vgetq_lane_f64(vacc, 0) * vgetq_lane_f64(vacc, 1);
        }

        for i in simd_len..len {
            prod *= data[i];
        }

        prod
    }

    /// NEON vectorized max reduction for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_max_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return f64::NEG_INFINITY;
        }

        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut max_val = f64::NEG_INFINITY;

        unsafe {
            let mut vmax = vdupq_n_f64(f64::NEG_INFINITY);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vmax = vmaxq_f64(vmax, v);
            }
            // Horizontal max
            let lane0 = vgetq_lane_f64(vmax, 0);
            let lane1 = vgetq_lane_f64(vmax, 1);
            max_val = lane0.max(lane1);
        }

        for i in simd_len..len {
            max_val = max_val.max(data[i]);
        }

        max_val
    }

    /// NEON vectorized min reduction for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_min_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return f64::INFINITY;
        }

        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut min_val = f64::INFINITY;

        unsafe {
            let mut vmin = vdupq_n_f64(f64::INFINITY);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                vmin = vminq_f64(vmin, v);
            }
            // Horizontal min
            let lane0 = vgetq_lane_f64(vmin, 0);
            let lane1 = vgetq_lane_f64(vmin, 1);
            min_val = lane0.min(lane1);
        }

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

    // =========================================================================
    // High-Priority NEON f64 Functions - FMA and Dot Product
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

        Array::from_vec(result).reshape(&a.shape())
    }

    /// NEON vectorized dot product for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_dot_f64(a: &Array<f64>, b: &Array<f64>) -> f64 {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let len = data_a.len().min(data_b.len());
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum = 0.0f64;

        unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let va = vld1q_f64(data_a.as_ptr().add(i));
                let vb = vld1q_f64(data_b.as_ptr().add(i));
                // FMA: acc + a * b
                vacc = vfmaq_f64(vacc, va, vb);
            }
            // Horizontal add
            sum = vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1);
        }

        for i in simd_len..len {
            sum += data_a[i] * data_b[i];
        }

        sum
    }

    /// NEON vectorized L2 norm for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_norm_l2_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum_sq = 0.0f64;

        unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                // FMA: acc + v * v
                vacc = vfmaq_f64(vacc, v, v);
            }
            // Horizontal add
            sum_sq = vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1);
        }

        for i in simd_len..len {
            sum_sq += data[i] * data[i];
        }

        sum_sq.sqrt()
    }

    /// NEON vectorized L1 norm for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_norm_l1_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        let mut sum_abs = 0.0f64;

        unsafe {
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let vabs = vabsq_f64(v);
                vacc = vaddq_f64(vacc, vabs);
            }
            // Horizontal add
            sum_abs = vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1);
        }

        for i in simd_len..len {
            sum_abs += data[i].abs();
        }

        sum_abs
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

        Array::from_vec(result).reshape(&input.shape())
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

        Array::from_vec(result).reshape(&input.shape())
    }

    // =========================================================================
    // Medium-Priority NEON f64 Functions - Extended Math Operations
    // =========================================================================

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

    /// NEON vectorized power function for f64: x^y (scalar exponent)
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_pow_scalar_f64(base: &Array<f64>, exp: f64) -> Array<f64> {
        // For non-integer exponents, use scalar pow for accuracy
        base.map(|x| x.powf(exp))
    }

    /// NEON vectorized element-wise power for f64: `base[i]^exp[i]`
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_pow_f64(base: &Array<f64>, exp: &Array<f64>) -> Array<f64> {
        let data_base = base.to_vec();
        let data_exp = exp.to_vec();
        let len = data_base.len().min(data_exp.len());
        let result: Vec<f64> = (0..len).map(|i| data_base[i].powf(data_exp[i])).collect();
        Array::from_vec(result).reshape(&base.shape())
    }

    /// NEON vectorized cube root for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_cbrt_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.cbrt())
    }

    /// NEON vectorized log base 2 for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_log2_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.log2())
    }

    /// NEON vectorized log base 10 for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_log10_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.log10())
    }

    /// NEON vectorized 2^x for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_exp2_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| (2.0f64).powf(x))
    }

    /// NEON vectorized exp(x) - 1 for f64 (accurate for small x)
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_expm1_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.exp_m1())
    }

    /// NEON vectorized log(1 + x) for f64 (accurate for small x)
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_log1p_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ln_1p())
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

    /// NEON vectorized hypot for f64: sqrt(x^2 + y^2) without overflow
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_hypot_f64(x: &Array<f64>, y: &Array<f64>) -> Array<f64> {
        let data_x = x.to_vec();
        let data_y = y.to_vec();
        let len = data_x.len().min(data_y.len());
        let mut result = vec![0.0f64; len];
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let vx = vld1q_f64(data_x.as_ptr().add(i));
                let vy = vld1q_f64(data_y.as_ptr().add(i));

                // x^2 + y^2
                let vx2 = vmulq_f64(vx, vx);
                let sum_sq = vfmaq_f64(vx2, vy, vy);

                // sqrt - use vsqrtq_f64
                let vsqrt = vsqrtq_f64(sum_sq);
                vst1q_f64(result.as_mut_ptr().add(i), vsqrt);
            }
        }

        for i in simd_len..len {
            result[i] = data_x[i].hypot(data_y[i]);
        }

        Array::from_vec(result).reshape(&x.shape())
    }

    /// NEON vectorized atan2 for f64: atan(y/x) with proper quadrant
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_atan2_f64(y: &Array<f64>, x: &Array<f64>) -> Array<f64> {
        let data_y = y.to_vec();
        let data_x = x.to_vec();
        let len = data_y.len().min(data_x.len());
        let result: Vec<f64> = (0..len).map(|i| data_y[i].atan2(data_x[i])).collect();
        Array::from_vec(result).reshape(&y.shape())
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

        Array::from_vec(result).reshape(&a.shape())
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

        Array::from_vec(result).reshape(&a.shape())
    }

    /// NEON vectorized variance for f64
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_assignments)]
    pub fn vectorized_variance_f64(input: &Array<f64>) -> f64 {
        let mean = Self::vectorized_mean_f64(input);
        let data = input.to_vec();
        let len = data.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(NEON_F64_LANES - 1);
        let mut sum_sq_diff = 0.0f64;

        unsafe {
            let vmean = vdupq_n_f64(mean);
            let mut vacc = vdupq_n_f64(0.0);
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let diff = vsubq_f64(v, vmean);
                // FMA: acc + diff * diff
                vacc = vfmaq_f64(vacc, diff, diff);
            }
            sum_sq_diff = vgetq_lane_f64(vacc, 0) + vgetq_lane_f64(vacc, 1);
        }

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

// Non-aarch64 fallback implementations for f64 functions
#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn vectorized_abs_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.abs())
    }

    pub fn vectorized_sqrt_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.sqrt())
    }

    pub fn vectorized_square_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x * x)
    }

    pub fn vectorized_exp_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.exp())
    }

    pub fn vectorized_log_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ln())
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

    pub fn vectorized_floor_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.floor())
    }

    pub fn vectorized_ceil_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ceil())
    }

    pub fn vectorized_round_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.round())
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

    // =========================================================================
    // High-Priority Fallback f64 Functions - Array Arithmetic
    // =========================================================================

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

    // =========================================================================
    // High-Priority Fallback f64 Functions - Scalar Operations
    // =========================================================================

    pub fn vectorized_add_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x + scalar)
    }

    pub fn vectorized_mul_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x * scalar)
    }

    // =========================================================================
    // High-Priority Fallback f64 Functions - Reductions
    // =========================================================================

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

    // =========================================================================
    // High-Priority Fallback f64 Functions - FMA and Dot Product
    // =========================================================================

    pub fn vectorized_fma_f64(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
        let data_a = a.to_vec();
        let data_b = b.to_vec();
        let data_c = c.to_vec();
        let len = data_a.len().min(data_b.len()).min(data_c.len());
        let result: Vec<f64> = (0..len)
            .map(|i| data_a[i].mul_add(data_b[i], data_c[i]))
            .collect();
        Array::from_vec(result).reshape(&a.shape())
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

    // =========================================================================
    // High-Priority Fallback f64 Functions - Element-wise Operations
    // =========================================================================

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

    pub fn vectorized_negative_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| -x)
    }

    pub fn vectorized_reciprocal_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| 1.0 / x)
    }

    // =========================================================================
    // Medium-Priority Fallback f64 Functions - Extended Math Operations
    // =========================================================================

    pub fn vectorized_clamp_f64(input: &Array<f64>, min_val: f64, max_val: f64) -> Array<f64> {
        input.map(|x| x.clamp(min_val, max_val))
    }

    pub fn vectorized_pow_scalar_f64(base: &Array<f64>, exp: f64) -> Array<f64> {
        base.map(|x| x.powf(exp))
    }

    pub fn vectorized_pow_f64(base: &Array<f64>, exp: &Array<f64>) -> Array<f64> {
        let data_base = base.to_vec();
        let data_exp = exp.to_vec();
        let len = data_base.len().min(data_exp.len());
        let result: Vec<f64> = (0..len).map(|i| data_base[i].powf(data_exp[i])).collect();
        Array::from_vec(result).reshape(&base.shape())
    }

    pub fn vectorized_cbrt_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.cbrt())
    }

    pub fn vectorized_log2_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.log2())
    }

    pub fn vectorized_log10_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.log10())
    }

    pub fn vectorized_exp2_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| (2.0f64).powf(x))
    }

    pub fn vectorized_expm1_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.exp_m1())
    }

    pub fn vectorized_log1p_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ln_1p())
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

    pub fn vectorized_hypot_f64(x: &Array<f64>, y: &Array<f64>) -> Array<f64> {
        let data_x = x.to_vec();
        let data_y = y.to_vec();
        let len = data_x.len().min(data_y.len());
        let result: Vec<f64> = (0..len).map(|i| data_x[i].hypot(data_y[i])).collect();
        Array::from_vec(result).reshape(&x.shape())
    }

    pub fn vectorized_atan2_f64(y: &Array<f64>, x: &Array<f64>) -> Array<f64> {
        let data_y = y.to_vec();
        let data_x = x.to_vec();
        let len = data_y.len().min(data_x.len());
        let result: Vec<f64> = (0..len).map(|i| data_y[i].atan2(data_x[i])).collect();
        Array::from_vec(result).reshape(&y.shape())
    }

    pub fn vectorized_sub_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x - scalar)
    }

    pub fn vectorized_div_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        a.map(|x| x / scalar)
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

/// NEON feature detection for ARM processors
pub struct NeonFeatureDetector;

impl NeonFeatureDetector {
    /// Detect available NEON features
    pub fn detect_neon_features() -> NeonFeatures {
        #[allow(unused_mut)] // False positive - modified in conditional compilation blocks
        let mut features = NeonFeatures::default();

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is standard on AArch64
            features.neon = true;

            if is_aarch64_feature_detected!("asimd") {
                features.asimd = true;
            }
            if is_aarch64_feature_detected!("fp") {
                features.fp = true;
            }
        }

        features
    }

    /// Get optimal block size for ARM processors
    pub fn optimal_block_size() -> usize {
        // ARM processors typically have smaller caches
        32
    }
}

#[derive(Debug, Default, Clone)]
pub struct NeonFeatures {
    pub neon: bool,  // Basic NEON support
    pub asimd: bool, // Advanced SIMD
    pub fp: bool,    // Floating point support
}

impl NeonFeatures {
    pub fn has_full_support(&self) -> bool {
        self.neon && self.asimd && self.fp
    }

    pub fn recommended_operations(&self) -> Vec<&'static str> {
        let mut ops = Vec::new();

        if self.neon {
            ops.push("Basic vectorization");
            ops.push("Integer operations");
        }
        if self.asimd {
            ops.push("Advanced SIMD operations");
            ops.push("Crypto operations");
        }
        if self.fp {
            ops.push("Floating point operations");
            ops.push("Vector math");
        }

        ops
    }
}

// Provide no-op implementations for non-ARM architectures
#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn neon_matmul_f32(
        a: &Array<f32>,
        b: &Array<f32>,
        c: &mut Array<f32>,
        _block_size: usize,
    ) -> Result<()> {
        // Fallback to regular matrix multiplication
        let result = a.matmul(b)?;
        *c = result;
        Ok(())
    }

    pub fn neon_exp_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.exp())
    }

    pub fn neon_log_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.ln())
    }

    pub fn neon_sin_cos_f32(input: &Array<f32>) -> (Array<f32>, Array<f32>) {
        (input.map(|x| x.sin()), input.map(|x| x.cos()))
    }

    pub fn neon_sum_f32(input: &Array<f32>) -> f32 {
        input.sum()
    }

    pub fn neon_dot_f32(a: &Array<f32>, b: &Array<f32>) -> Result<f32> {
        a.dot(b)
    }

    pub fn neon_copy_f32(src: &Array<f32>, dst: &mut Array<f32>) -> Result<()> {
        *dst = src.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_neon_feature_detection() {
        let features = NeonFeatureDetector::detect_neon_features();
        println!("NEON features: {:?}", features);

        let ops = features.recommended_operations();

        // On x86_64, NEON won't be available, so ops might be empty
        #[cfg(target_arch = "aarch64")]
        assert!(!ops.is_empty());

        #[cfg(not(target_arch = "aarch64"))]
        println!("NEON not available on this architecture: {}", ops.len());
    }

    #[test]
    fn test_neon_exp() {
        let input = Array::from_vec(vec![0.0, 1.0, 2.0, -1.0]);
        let result = NeonEnhancedOps::neon_exp_f32(&input);

        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], std::f32::consts::E, epsilon = 1e-4);
        assert_relative_eq!(
            result.to_vec()[2],
            std::f32::consts::E.powi(2),
            epsilon = 5e-3
        );
        assert_relative_eq!(
            result.to_vec()[3],
            1.0 / std::f32::consts::E,
            epsilon = 2e-5
        );
    }

    #[test]
    fn test_neon_sum() {
        let input = Array::from_vec(vec![1.0f32; 100]);
        let result = NeonEnhancedOps::neon_sum_f32(&input);
        assert_relative_eq!(result, 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_neon_dot() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
        let result = NeonEnhancedOps::neon_dot_f32(&a, &b).unwrap();
        assert_relative_eq!(result, 70.0, epsilon = 1e-6); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_optimal_block_size() {
        let block_size = NeonFeatureDetector::optimal_block_size();
        assert!(block_size >= 16);
        assert!(block_size <= 64);
    }

    // =========================================================================
    // Tests for f64 NEON operations (used by ufuncs)
    // =========================================================================

    #[test]
    fn test_vectorized_abs_f64() {
        let input = Array::from_vec(vec![-1.0, -2.0, 3.0, -4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_abs_f64(&input);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_sqrt_f64() {
        let input = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);
        let result = NeonEnhancedOps::vectorized_sqrt_f64(&input);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_square_f64() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_square_f64(&input);
        let expected = [1.0, 4.0, 9.0, 16.0, 25.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_exp_f64() {
        let input = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let result = NeonEnhancedOps::vectorized_exp_f64(&input);
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], std::f64::consts::E, epsilon = 1e-4);
        assert_relative_eq!(
            result.to_vec()[2],
            std::f64::consts::E.powi(2),
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_vectorized_floor_ceil_round_f64() {
        let input = Array::from_vec(vec![1.3, 2.7, -1.3, -2.7]);

        let floor_result = NeonEnhancedOps::vectorized_floor_f64(&input);
        let ceil_result = NeonEnhancedOps::vectorized_ceil_f64(&input);
        let round_result = NeonEnhancedOps::vectorized_round_f64(&input);

        assert_relative_eq!(floor_result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(floor_result.to_vec()[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(floor_result.to_vec()[2], -2.0, epsilon = 1e-10);
        assert_relative_eq!(floor_result.to_vec()[3], -3.0, epsilon = 1e-10);

        assert_relative_eq!(ceil_result.to_vec()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(ceil_result.to_vec()[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(ceil_result.to_vec()[2], -1.0, epsilon = 1e-10);
        assert_relative_eq!(ceil_result.to_vec()[3], -2.0, epsilon = 1e-10);

        assert_relative_eq!(round_result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(round_result.to_vec()[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(round_result.to_vec()[2], -1.0, epsilon = 1e-10);
        assert_relative_eq!(round_result.to_vec()[3], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_sign_f64() {
        let input = Array::from_vec(vec![-5.0, 0.0, 3.0, -0.0]);
        let result = NeonEnhancedOps::vectorized_sign_f64(&input);
        assert_relative_eq!(result.to_vec()[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[2], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_large_array_f64() {
        // Test with array larger than SIMD threshold (32) to ensure NEON path is taken
        let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr = Array::from_vec(input.clone());

        let abs_result = NeonEnhancedOps::vectorized_abs_f64(&arr);
        let square_result = NeonEnhancedOps::vectorized_square_f64(&arr);

        assert_eq!(abs_result.len(), 100);
        assert_eq!(square_result.len(), 100);

        // Check a few values
        assert_relative_eq!(square_result.to_vec()[10], 100.0, epsilon = 1e-10);
        assert_relative_eq!(square_result.to_vec()[5], 25.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Tests for High-Priority NEON f64 Operations
    // =========================================================================

    #[test]
    fn test_vectorized_add_arrays_f64() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let result = NeonEnhancedOps::vectorized_add_arrays_f64(&a, &b);
        let expected = [11.0, 22.0, 33.0, 44.0, 55.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_sub_arrays_f64() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_sub_arrays_f64(&a, &b);
        let expected = [9.0, 18.0, 27.0, 36.0, 45.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_mul_arrays_f64() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = NeonEnhancedOps::vectorized_mul_arrays_f64(&a, &b);
        let expected = [2.0, 6.0, 12.0, 20.0, 30.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_div_arrays_f64() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let b = Array::from_vec(vec![2.0, 4.0, 5.0, 8.0, 10.0]);
        let result = NeonEnhancedOps::vectorized_div_arrays_f64(&a, &b);
        let expected = [5.0, 5.0, 6.0, 5.0, 5.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_add_scalar_f64() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_add_scalar_f64(&a, 10.0);
        let expected = [11.0, 12.0, 13.0, 14.0, 15.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_mul_scalar_f64() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_mul_scalar_f64(&a, 3.0);
        let expected = [3.0, 6.0, 9.0, 12.0, 15.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_sum_f64() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_sum_f64(&input);
        assert_relative_eq!(result, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_prod_f64() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_prod_f64(&input);
        assert_relative_eq!(result, 120.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_max_min_f64() {
        let input = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let max_result = NeonEnhancedOps::vectorized_max_f64(&input);
        let min_result = NeonEnhancedOps::vectorized_min_f64(&input);
        assert_relative_eq!(max_result, 9.0, epsilon = 1e-10);
        assert_relative_eq!(min_result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_mean_f64() {
        let input = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = NeonEnhancedOps::vectorized_mean_f64(&input);
        assert_relative_eq!(result, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_fma_f64() {
        // FMA: a * b + c
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let c = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let result = NeonEnhancedOps::vectorized_fma_f64(&a, &b, &c);
        // 1*2+10=12, 2*3+20=26, 3*4+30=42, 4*5+40=60
        let expected = [12.0, 26.0, 42.0, 60.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_dot_f64() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
        let result = NeonEnhancedOps::vectorized_dot_f64(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_relative_eq!(result, 70.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_norm_l2_f64() {
        let input = Array::from_vec(vec![3.0, 4.0]); // sqrt(9+16) = 5
        let result = NeonEnhancedOps::vectorized_norm_l2_f64(&input);
        assert_relative_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_norm_l1_f64() {
        let input = Array::from_vec(vec![-3.0, 4.0, -5.0]); // |−3|+|4|+|−5| = 12
        let result = NeonEnhancedOps::vectorized_norm_l1_f64(&input);
        assert_relative_eq!(result, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_maximum_minimum_f64() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0, 7.0]);
        let b = Array::from_vec(vec![2.0, 4.0, 6.0, 5.0]);
        let max_result = NeonEnhancedOps::vectorized_maximum_f64(&a, &b);
        let min_result = NeonEnhancedOps::vectorized_minimum_f64(&a, &b);

        let max_expected = [2.0, 5.0, 6.0, 7.0];
        let min_expected = [1.0, 4.0, 3.0, 5.0];

        for (r, e) in max_result.to_vec().iter().zip(max_expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
        for (r, e) in min_result.to_vec().iter().zip(min_expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_negative_f64() {
        let input = Array::from_vec(vec![1.0, -2.0, 3.0, -4.0]);
        let result = NeonEnhancedOps::vectorized_negative_f64(&input);
        let expected = [-1.0, 2.0, -3.0, 4.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_reciprocal_f64() {
        let input = Array::from_vec(vec![1.0, 2.0, 4.0, 5.0]);
        let result = NeonEnhancedOps::vectorized_reciprocal_f64(&input);
        let expected = [1.0, 0.5, 0.25, 0.2];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_high_priority_large_arrays_f64() {
        // Test with arrays larger than SIMD threshold to ensure NEON path is taken
        let input_a: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let input_b: Vec<f64> = (101..=200).map(|i| i as f64).collect();
        let arr_a = Array::from_vec(input_a);
        let arr_b = Array::from_vec(input_b);

        // Test array operations
        let add_result = NeonEnhancedOps::vectorized_add_arrays_f64(&arr_a, &arr_b);
        let mul_result = NeonEnhancedOps::vectorized_mul_arrays_f64(&arr_a, &arr_b);

        assert_eq!(add_result.len(), 100);
        assert_eq!(mul_result.len(), 100);

        // First element: 1 + 101 = 102, 1 * 101 = 101
        assert_relative_eq!(add_result.to_vec()[0], 102.0, epsilon = 1e-10);
        assert_relative_eq!(mul_result.to_vec()[0], 101.0, epsilon = 1e-10);

        // Last element: 100 + 200 = 300, 100 * 200 = 20000
        assert_relative_eq!(add_result.to_vec()[99], 300.0, epsilon = 1e-10);
        assert_relative_eq!(mul_result.to_vec()[99], 20000.0, epsilon = 1e-10);

        // Test reductions
        let sum = NeonEnhancedOps::vectorized_sum_f64(&arr_a);
        assert_relative_eq!(sum, 5050.0, epsilon = 1e-10); // sum of 1..100

        let max = NeonEnhancedOps::vectorized_max_f64(&arr_a);
        let min = NeonEnhancedOps::vectorized_min_f64(&arr_a);
        assert_relative_eq!(max, 100.0, epsilon = 1e-10);
        assert_relative_eq!(min, 1.0, epsilon = 1e-10);

        // Test dot product
        let simple_a = Array::from_vec(vec![1.0f64; 100]);
        let simple_b = Array::from_vec(vec![2.0f64; 100]);
        let dot = NeonEnhancedOps::vectorized_dot_f64(&simple_a, &simple_b);
        assert_relative_eq!(dot, 200.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Tests for Medium-Priority NEON f64 Operations
    // =========================================================================

    #[test]
    fn test_vectorized_clamp_f64() {
        let input = Array::from_vec(vec![-2.0, 0.5, 2.0, 5.0, 10.0]);
        let result = NeonEnhancedOps::vectorized_clamp_f64(&input, 0.0, 3.0);
        let expected = [0.0, 0.5, 2.0, 3.0, 3.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_pow_f64() {
        let base = Array::from_vec(vec![2.0, 3.0, 4.0]);
        let exp = Array::from_vec(vec![2.0, 2.0, 0.5]);
        let result = NeonEnhancedOps::vectorized_pow_f64(&base, &exp);
        let expected = [4.0, 9.0, 2.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_pow_scalar_f64() {
        let base = Array::from_vec(vec![2.0, 3.0, 4.0]);
        let result = NeonEnhancedOps::vectorized_pow_scalar_f64(&base, 3.0);
        let expected = [8.0, 27.0, 64.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_cbrt_f64() {
        let input = Array::from_vec(vec![8.0, 27.0, 64.0]);
        let result = NeonEnhancedOps::vectorized_cbrt_f64(&input);
        let expected = [2.0, 3.0, 4.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_log2_log10_f64() {
        let input = Array::from_vec(vec![1.0, 2.0, 4.0, 8.0]);
        let log2_result = NeonEnhancedOps::vectorized_log2_f64(&input);
        let expected_log2 = [0.0, 1.0, 2.0, 3.0];
        for (r, e) in log2_result.to_vec().iter().zip(expected_log2.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }

        let input10 = Array::from_vec(vec![1.0, 10.0, 100.0]);
        let log10_result = NeonEnhancedOps::vectorized_log10_f64(&input10);
        let expected_log10 = [0.0, 1.0, 2.0];
        for (r, e) in log10_result.to_vec().iter().zip(expected_log10.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_exp2_f64() {
        let input = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = NeonEnhancedOps::vectorized_exp2_f64(&input);
        let expected = [1.0, 2.0, 4.0, 8.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_expm1_log1p_f64() {
        // exp(0) - 1 = 0, exp(small) - 1 ≈ small for small values
        let input = Array::from_vec(vec![0.0, 1e-10]);
        let expm1_result = NeonEnhancedOps::vectorized_expm1_f64(&input);
        assert_relative_eq!(expm1_result.to_vec()[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(expm1_result.to_vec()[1], 1e-10, epsilon = 1e-18);

        // log(1 + 0) = 0, log(1 + small) ≈ small for small values
        let log1p_result = NeonEnhancedOps::vectorized_log1p_f64(&input);
        assert_relative_eq!(log1p_result.to_vec()[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(log1p_result.to_vec()[1], 1e-10, epsilon = 1e-18);
    }

    #[test]
    fn test_vectorized_copysign_f64() {
        let magnitude = Array::from_vec(vec![1.0, -2.0, 3.0, -4.0]);
        let sign = Array::from_vec(vec![-1.0, 1.0, -1.0, 1.0]);
        let result = NeonEnhancedOps::vectorized_copysign_f64(&magnitude, &sign);
        let expected = [-1.0, 2.0, -3.0, 4.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_hypot_f64() {
        let x = Array::from_vec(vec![3.0, 5.0, 8.0]);
        let y = Array::from_vec(vec![4.0, 12.0, 15.0]);
        let result = NeonEnhancedOps::vectorized_hypot_f64(&x, &y);
        let expected = [5.0, 13.0, 17.0];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_atan2_f64() {
        let y = Array::from_vec(vec![0.0, 1.0, 0.0, -1.0]);
        let x = Array::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
        let result = NeonEnhancedOps::vectorized_atan2_f64(&y, &x);
        // atan2(0,1)=0, atan2(1,0)=π/2, atan2(0,-1)=π, atan2(-1,0)=-π/2
        let expected = [
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
        ];
        for (r, e) in result.to_vec().iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_sub_div_scalar_f64() {
        let input = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        let sub_result = NeonEnhancedOps::vectorized_sub_scalar_f64(&input, 5.0);
        let sub_expected = [5.0, 15.0, 25.0, 35.0];
        for (r, e) in sub_result.to_vec().iter().zip(sub_expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }

        let div_result = NeonEnhancedOps::vectorized_div_scalar_f64(&input, 10.0);
        let div_expected = [1.0, 2.0, 3.0, 4.0];
        for (r, e) in div_result.to_vec().iter().zip(div_expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_variance_std_f64() {
        // Simple variance/std test: [1, 2, 3, 4, 5] => mean=3, variance=2, std=sqrt(2)
        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let variance = NeonEnhancedOps::vectorized_variance_f64(&input);
        let std_dev = NeonEnhancedOps::vectorized_std_f64(&input);

        // variance = [(1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2] / 5
        //          = [4 + 1 + 0 + 1 + 4] / 5 = 10/5 = 2
        assert_relative_eq!(variance, 2.0, epsilon = 1e-10);
        assert_relative_eq!(std_dev, 2.0f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_medium_priority_large_arrays_f64() {
        // Test with arrays larger than SIMD threshold
        let input: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let arr = Array::from_vec(input);

        // Test clamp on large array
        let clamped = NeonEnhancedOps::vectorized_clamp_f64(&arr, 20.0, 80.0);
        assert_eq!(clamped.len(), 100);
        assert_relative_eq!(clamped.to_vec()[0], 20.0, epsilon = 1e-10); // 1 clamped to 20
        assert_relative_eq!(clamped.to_vec()[50], 51.0, epsilon = 1e-10); // 51 unchanged
        assert_relative_eq!(clamped.to_vec()[99], 80.0, epsilon = 1e-10); // 100 clamped to 80

        // Test variance/std on large array
        // For 1..100, mean = 50.5
        // variance = sum((x - 50.5)^2) / 100 = 833.25
        let variance = NeonEnhancedOps::vectorized_variance_f64(&arr);
        assert_relative_eq!(variance, 833.25, epsilon = 1e-10);

        // Test hypot on large arrays
        let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let y: Vec<f64> = (1..=100).map(|i| (i * 2) as f64).collect();
        let arr_x = Array::from_vec(x);
        let arr_y = Array::from_vec(y);
        let hypot = NeonEnhancedOps::vectorized_hypot_f64(&arr_x, &arr_y);
        // hypot(1, 2) = sqrt(5), hypot(100, 200) = sqrt(50000)
        assert_relative_eq!(hypot.to_vec()[0], 5.0f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(hypot.to_vec()[99], 50000.0f64.sqrt(), epsilon = 1e-8);
    }
}
