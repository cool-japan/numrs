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
                temp_x[j] = temp_x[j] % (2.0 * std::f32::consts::PI);
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
        for i in simd_len..len {
            dst[i] = src[i];
        }
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
}
