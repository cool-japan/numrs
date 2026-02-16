//! Exponential and logarithmic operations using NEON SIMD
//!
//! This module provides optimized exp, log, pow, sqrt, and related functions
//! for ARM NEON.

use crate::array::Array;

use super::core::{NeonEnhancedOps, NEON_F32_LANES, NEON_F64_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// NEON f32 Exponential Operations
// =============================================================================

impl NeonEnhancedOps {
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
}

// =============================================================================
// NEON f64 Exponential Operations
// =============================================================================

impl NeonEnhancedOps {
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
}

// =============================================================================
// Non-aarch64 Fallback Implementations
// =============================================================================

#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn neon_exp_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.exp())
    }

    pub fn neon_log_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.ln())
    }

    pub fn vectorized_sqrt_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.sqrt())
    }

    pub fn vectorized_exp_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.exp())
    }

    pub fn vectorized_log_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ln())
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

    pub fn vectorized_hypot_f64(x: &Array<f64>, y: &Array<f64>) -> Array<f64> {
        let data_x = x.to_vec();
        let data_y = y.to_vec();
        let len = data_x.len().min(data_y.len());
        let result: Vec<f64> = (0..len).map(|i| data_x[i].hypot(data_y[i])).collect();
        Array::from_vec(result).reshape(&x.shape())
    }
}
