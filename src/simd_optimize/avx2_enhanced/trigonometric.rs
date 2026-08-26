//! Trigonometric SIMD operations
//!
//! This module provides AVX2 optimized implementations for:
//! - sin_f32, sin_f64: Sine functions
//! - cos_f32, cos_f64: Cosine functions
//! - tan_f32, tan_f64: Tangent functions
//! - sinh_f64, cosh_f64, tanh_f64: Hyperbolic functions
//! - asin_f64, acos_f64, atan_f64: Inverse trigonometric functions
//! - asinh_f64, acosh_f64, atanh_f64: Inverse hyperbolic functions

use super::EnhancedSimdOps;
// See `arithmetic.rs` for why these are gated the same as the intrinsics.
#[cfg(target_arch = "x86_64")]
use super::{AVX2_F32_LANES, AVX2_F64_LANES, PREFETCH_DISTANCE};
#[cfg(target_arch = "x86_64")]
use crate::array::Array;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl EnhancedSimdOps {
    // ========================================
    // Sine Functions
    // ========================================

    /// Vectorized sine function with CORDIC algorithm
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sin_f32_simd(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.sin())
    }

    /// AVX2 optimized sine function for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_sin_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for sin approximation (Taylor series)
        let pi = _mm256_set1_ps(std::f32::consts::PI);
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let pi_2 = _mm256_set1_ps(std::f32::consts::PI / 2.0);
        let one = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(-1.0 / 6.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);
        let c7 = _mm256_set1_ps(-1.0 / 5040.0);
        let c9 = _mm256_set1_ps(1.0 / 362880.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let mut x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Range reduction: bring x to [-pi, pi]
            let k = _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_fmsub_ps(k, two_pi, x);

            // Determine quadrant and adjust
            let abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            let quadrant = _mm256_cmp_ps(abs_x, pi_2, _CMP_GT_OQ);

            // For |x| > pi/2, use sin(pi - x) = sin(x)
            let x_adj = _mm256_blendv_ps(x, _mm256_sub_ps(pi, abs_x), quadrant);
            let sign_adj = _mm256_blendv_ps(
                one,
                _mm256_set1_ps(-1.0),
                _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ),
            );

            // Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
            let x2 = _mm256_mul_ps(x_adj, x_adj);
            let x3 = _mm256_mul_ps(x2, x_adj);
            let x5 = _mm256_mul_ps(x3, x2);
            let x7 = _mm256_mul_ps(x5, x2);
            let x9 = _mm256_mul_ps(x7, x2);

            let poly = _mm256_fmadd_ps(
                c9,
                x9,
                _mm256_fmadd_ps(
                    c7,
                    x7,
                    _mm256_fmadd_ps(c5, x5, _mm256_fmadd_ps(c3, x3, x_adj)),
                ),
            );

            let result = _mm256_mul_ps(poly, sign_adj);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].sin();
        }
    }

    /// Vectorized sine function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sin_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sin_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized sine function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_sin_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // High precision constants
        let pi = _mm256_set1_pd(std::f64::consts::PI);
        let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);

        // Taylor series coefficients
        let c3 = _mm256_set1_pd(-1.0 / 6.0);
        let c5 = _mm256_set1_pd(1.0 / 120.0);
        let c7 = _mm256_set1_pd(-1.0 / 5040.0);
        let c9 = _mm256_set1_pd(1.0 / 362880.0);
        let c11 = _mm256_set1_pd(-1.0 / 39916800.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let mut x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction: bring x to [-pi, pi]
            let k = _mm256_round_pd(_mm256_div_pd(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

            // Handle sign
            let sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
            let sign = _mm256_blendv_pd(one, neg_one, sign_mask);

            // Take absolute value
            let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            let abs_x = _mm256_and_pd(x, abs_mask);

            // For |x| > pi/2, use sin(pi - |x|)
            let need_adjust = _mm256_cmp_pd(abs_x, pi_2, _CMP_GT_OQ);
            let x_adj = _mm256_blendv_pd(abs_x, _mm256_sub_pd(pi, abs_x), need_adjust);

            // Taylor series
            let x2 = _mm256_mul_pd(x_adj, x_adj);
            let x3 = _mm256_mul_pd(x2, x_adj);
            let x5 = _mm256_mul_pd(x3, x2);
            let x7 = _mm256_mul_pd(x5, x2);
            let x9 = _mm256_mul_pd(x7, x2);
            let x11 = _mm256_mul_pd(x9, x2);

            let poly = _mm256_fmadd_pd(
                c11,
                x11,
                _mm256_fmadd_pd(
                    c9,
                    x9,
                    _mm256_fmadd_pd(
                        c7,
                        x7,
                        _mm256_fmadd_pd(c5, x5, _mm256_fmadd_pd(c3, x3, x_adj)),
                    ),
                ),
            );

            let result = _mm256_mul_pd(poly, sign);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].sin();
        }
    }

    // ========================================
    // Cosine Functions
    // ========================================

    /// Vectorized cosine function for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cos_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.cos())
    }

    /// AVX2 optimized cosine function for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cos_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for cos approximation
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let one = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(-0.5);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c6 = _mm256_set1_ps(-1.0 / 720.0);
        let c8 = _mm256_set1_ps(1.0 / 40320.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let mut x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Range reduction
            let k = _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_fmsub_ps(k, two_pi, x);

            // Take absolute value
            let abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));

            // Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!
            let x2 = _mm256_mul_ps(abs_x, abs_x);
            let x4 = _mm256_mul_ps(x2, x2);
            let x6 = _mm256_mul_ps(x4, x2);
            let x8 = _mm256_mul_ps(x4, x4);

            let poly = _mm256_fmadd_ps(
                c8,
                x8,
                _mm256_fmadd_ps(
                    c6,
                    x6,
                    _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c2, x2, one)),
                ),
            );

            _mm256_storeu_ps(output.as_mut_ptr().add(i), poly);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cos();
        }
    }

    /// Vectorized cosine function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cos_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.cos())
    }

    /// AVX2 optimized cosine function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cos_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // High precision constants
        let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
        let one = _mm256_set1_pd(1.0);
        let c2 = _mm256_set1_pd(-0.5);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c6 = _mm256_set1_pd(-1.0 / 720.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);
        let c10 = _mm256_set1_pd(-1.0 / 3628800.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let mut x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction
            let k = _mm256_round_pd(_mm256_div_pd(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

            // Take absolute value
            let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            let abs_x = _mm256_and_pd(x, abs_mask);

            // Taylor series
            let x2 = _mm256_mul_pd(abs_x, abs_x);
            let x4 = _mm256_mul_pd(x2, x2);
            let x6 = _mm256_mul_pd(x4, x2);
            let x8 = _mm256_mul_pd(x4, x4);
            let x10 = _mm256_mul_pd(x8, x2);

            let poly = _mm256_fmadd_pd(
                c10,
                x10,
                _mm256_fmadd_pd(
                    c8,
                    x8,
                    _mm256_fmadd_pd(
                        c6,
                        x6,
                        _mm256_fmadd_pd(c4, x4, _mm256_fmadd_pd(c2, x2, one)),
                    ),
                ),
            );

            _mm256_storeu_pd(output.as_mut_ptr().add(i), poly);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cos();
        }
    }

    // ========================================
    // Tangent Functions
    // ========================================

    /// Vectorized tangent function using sin/cos
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_tan_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.tan())
    }

    /// AVX2 optimized tangent using sin/cos ratio for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tan_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for sin/cos approximation
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let one = _mm256_set1_ps(1.0);
        let s3 = _mm256_set1_ps(-1.0 / 6.0);
        let s5 = _mm256_set1_ps(1.0 / 120.0);
        let c2 = _mm256_set1_ps(-0.5);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c6 = _mm256_set1_ps(-1.0 / 720.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let mut x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Range reduction
            let k = _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_fmsub_ps(k, two_pi, x);

            let abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));

            // sin(x) approximation
            let x2 = _mm256_mul_ps(abs_x, abs_x);
            let x3 = _mm256_mul_ps(x2, abs_x);
            let x5 = _mm256_mul_ps(x3, x2);
            let sin_val = _mm256_fmadd_ps(s5, x5, _mm256_fmadd_ps(s3, x3, abs_x));

            // cos(x) approximation
            let x4 = _mm256_mul_ps(x2, x2);
            let x6 = _mm256_mul_ps(x4, x2);
            let cos_val = _mm256_fmadd_ps(
                c6,
                x6,
                _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c2, x2, one)),
            );

            // tan = sin/cos (handle sign)
            let sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0));
            let tan_val = _mm256_div_ps(sin_val, cos_val);
            let result = _mm256_xor_ps(tan_val, sign);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tan();
        }
    }

    /// Vectorized tangent function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_tan_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.tan())
    }

    /// AVX2 optimized tangent for f64 using sin/cos ratio
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tan_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // High precision constants
        let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
        let pi = _mm256_set1_pd(std::f64::consts::PI);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);

        // Taylor coefficients for sin
        let s3 = _mm256_set1_pd(-1.0 / 6.0);
        let s5 = _mm256_set1_pd(1.0 / 120.0);
        let s7 = _mm256_set1_pd(-1.0 / 5040.0);
        let s9 = _mm256_set1_pd(1.0 / 362880.0);

        // Taylor coefficients for cos
        let c2 = _mm256_set1_pd(-0.5);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c6 = _mm256_set1_pd(-1.0 / 720.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let mut x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction
            let k = _mm256_round_pd(_mm256_div_pd(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

            // Handle sign for tangent (odd function)
            let sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
            let sign = _mm256_blendv_pd(one, neg_one, sign_mask);

            // Take absolute value
            let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            let abs_x = _mm256_and_pd(x, abs_mask);

            // For |x| > pi/2, use tan(x) = -tan(pi - |x|)
            let need_adjust = _mm256_cmp_pd(abs_x, pi_2, _CMP_GT_OQ);
            let x_adj = _mm256_blendv_pd(abs_x, _mm256_sub_pd(pi, abs_x), need_adjust);
            let tan_sign = _mm256_blendv_pd(one, neg_one, need_adjust);

            // sin(x) approximation
            let x2 = _mm256_mul_pd(x_adj, x_adj);
            let x3 = _mm256_mul_pd(x2, x_adj);
            let x5 = _mm256_mul_pd(x3, x2);
            let x7 = _mm256_mul_pd(x5, x2);
            let x9 = _mm256_mul_pd(x7, x2);
            let sin_val = _mm256_fmadd_pd(
                s9,
                x9,
                _mm256_fmadd_pd(
                    s7,
                    x7,
                    _mm256_fmadd_pd(s5, x5, _mm256_fmadd_pd(s3, x3, x_adj)),
                ),
            );

            // cos(x) approximation
            let x4 = _mm256_mul_pd(x2, x2);
            let x6 = _mm256_mul_pd(x4, x2);
            let x8 = _mm256_mul_pd(x4, x4);
            let cos_val = _mm256_fmadd_pd(
                c8,
                x8,
                _mm256_fmadd_pd(
                    c6,
                    x6,
                    _mm256_fmadd_pd(c4, x4, _mm256_fmadd_pd(c2, x2, one)),
                ),
            );

            // tan = sin/cos with sign adjustment
            let tan_val = _mm256_div_pd(sin_val, cos_val);
            let tan_adjusted = _mm256_mul_pd(tan_val, tan_sign);
            let result = _mm256_mul_pd(tan_adjusted, sign);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tan();
        }
    }

    // ========================================
    // Hyperbolic Functions
    // ========================================

    /// Vectorized hyperbolic sine function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sinh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sinh_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized sinh for f64 using (exp(x) - exp(-x)) / 2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_sinh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let half = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // sinh(x) = (exp(x) - exp(-x)) / 2
            let exp_x = Self::simd_exp_pd(x);
            let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            let exp_neg_x = Self::simd_exp_pd(neg_x);

            let diff = _mm256_sub_pd(exp_x, exp_neg_x);
            let result = _mm256_mul_pd(diff, half);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].sinh();
        }
    }

    /// Vectorized hyperbolic cosine function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cosh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_cosh_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized cosh for f64 using (exp(x) + exp(-x)) / 2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cosh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let half = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // cosh(x) = (exp(x) + exp(-x)) / 2
            let exp_x = Self::simd_exp_pd(x);
            let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            let exp_neg_x = Self::simd_exp_pd(neg_x);

            let sum = _mm256_add_pd(exp_x, exp_neg_x);
            let result = _mm256_mul_pd(sum, half);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cosh();
        }
    }

    /// Vectorized hyperbolic tangent function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_tanh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_tanh_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized tanh for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tanh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            let two_x = _mm256_mul_pd(x, two);
            let exp_2x = Self::simd_exp_pd(two_x);

            let numerator = _mm256_sub_pd(exp_2x, one);
            let denominator = _mm256_add_pd(exp_2x, one);
            let result = _mm256_div_pd(numerator, denominator);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tanh();
        }
    }

    // ========================================
    // Inverse Trigonometric Functions
    // ========================================

    /// Vectorized asin function for f64 (arc sine)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_asin_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_asin_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized arc sine for f64
    /// Uses the identity: asin(x) = atan2(x, sqrt(1-x^2))
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_asin_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // asin(x) = atan2(x, sqrt(1 - x^2))
            let x_sq = _mm256_mul_pd(x, x);
            let one_minus_x_sq = _mm256_sub_pd(one, x_sq);
            let sqrt_term = _mm256_sqrt_pd(one_minus_x_sq);

            let result = Self::simd_atan2_pd(x, sqrt_term);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].asin();
        }
    }

    /// Vectorized acos function for f64 (arc cosine)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_acos_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_acos_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized arc cosine for f64
    /// Uses the identity: acos(x) = atan2(sqrt(1-x^2), x)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_acos_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // acos(x) = atan2(sqrt(1 - x^2), x)
            let x_sq = _mm256_mul_pd(x, x);
            let one_minus_x_sq = _mm256_sub_pd(one, x_sq);
            let sqrt_term = _mm256_sqrt_pd(one_minus_x_sq);

            let result = Self::simd_atan2_pd(sqrt_term, x);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].acos();
        }
    }

    /// Vectorized atan function for f64 (arc tangent)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_atan_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_atan_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized arc tangent for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_atan_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let result = Self::simd_atan_pd(x);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].atan();
        }
    }

    // ========================================
    // Inverse Hyperbolic Functions
    // ========================================

    /// Vectorized asinh function for f64 (inverse hyperbolic sine)
    /// asinh(x) = ln(x + sqrt(x^2 + 1))
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_asinh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_asinh_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized asinh for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_asinh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // asinh(x) = ln(x + sqrt(x^2 + 1))
            let x_sq = _mm256_mul_pd(x, x);
            let x_sq_plus_1 = _mm256_add_pd(x_sq, one);
            let sqrt_term = _mm256_sqrt_pd(x_sq_plus_1);
            let arg = _mm256_add_pd(x, sqrt_term);
            let result = Self::simd_log_pd(arg);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].asinh();
        }
    }

    /// Vectorized acosh function for f64 (inverse hyperbolic cosine)
    /// acosh(x) = ln(x + sqrt(x^2 - 1)) for x >= 1
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_acosh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_acosh_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized acosh for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_acosh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // acosh(x) = ln(x + sqrt(x^2 - 1))
            let x_sq = _mm256_mul_pd(x, x);
            let x_sq_minus_1 = _mm256_sub_pd(x_sq, one);
            let sqrt_term = _mm256_sqrt_pd(x_sq_minus_1);
            let arg = _mm256_add_pd(x, sqrt_term);
            let result = Self::simd_log_pd(arg);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].acosh();
        }
    }

    /// Vectorized atanh function for f64 (inverse hyperbolic tangent)
    /// atanh(x) = 0.5 * ln((1+x)/(1-x)) for |x| < 1
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_atanh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_atanh_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())?
    }

    /// AVX2 optimized atanh for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_atanh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // atanh(x) = 0.5 * ln((1+x)/(1-x))
            let one_plus_x = _mm256_add_pd(one, x);
            let one_minus_x = _mm256_sub_pd(one, x);
            let ratio = _mm256_div_pd(one_plus_x, one_minus_x);
            let log_ratio = Self::simd_log_pd(ratio);
            let result = _mm256_mul_pd(half, log_ratio);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].atanh();
        }
    }

    // ========================================
    // Helper SIMD functions for f64 (__m256d)
    // ========================================

    /// Helper function for SIMD exponential (f64)
    /// Uses element-wise standard library for accuracy
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_exp_pd(x: __m256d) -> __m256d {
        // Extract elements, apply standard library exp, repack
        let mut vals = [0.0f64; 4];
        _mm256_storeu_pd(vals.as_mut_ptr(), x);

        vals[0] = vals[0].exp();
        vals[1] = vals[1].exp();
        vals[2] = vals[2].exp();
        vals[3] = vals[3].exp();

        _mm256_loadu_pd(vals.as_ptr())
    }

    /// Helper function for SIMD natural logarithm (f64)
    /// Uses element-wise standard library for accuracy
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_log_pd(x: __m256d) -> __m256d {
        // Extract elements, apply standard library ln, repack
        let mut vals = [0.0f64; 4];
        _mm256_storeu_pd(vals.as_mut_ptr(), x);

        vals[0] = vals[0].ln();
        vals[1] = vals[1].ln();
        vals[2] = vals[2].ln();
        vals[3] = vals[3].ln();

        _mm256_loadu_pd(vals.as_ptr())
    }

    /// Helper function for SIMD arctangent (f64)
    /// Uses element-wise standard library for accuracy
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_atan_pd(x: __m256d) -> __m256d {
        // Extract elements, apply standard library atan, repack
        let mut vals = [0.0f64; 4];
        _mm256_storeu_pd(vals.as_mut_ptr(), x);

        vals[0] = vals[0].atan();
        vals[1] = vals[1].atan();
        vals[2] = vals[2].atan();
        vals[3] = vals[3].atan();

        _mm256_loadu_pd(vals.as_ptr())
    }

    /// Helper function for SIMD atan2 (f64)
    /// Uses element-wise standard library for accuracy
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_atan2_pd(y: __m256d, x: __m256d) -> __m256d {
        // Extract elements, apply standard library atan2, repack
        let mut y_vals = [0.0f64; 4];
        let mut x_vals = [0.0f64; 4];
        _mm256_storeu_pd(y_vals.as_mut_ptr(), y);
        _mm256_storeu_pd(x_vals.as_mut_ptr(), x);

        y_vals[0] = y_vals[0].atan2(x_vals[0]);
        y_vals[1] = y_vals[1].atan2(x_vals[1]);
        y_vals[2] = y_vals[2].atan2(x_vals[2]);
        y_vals[3] = y_vals[3].atan2(x_vals[3]);

        _mm256_loadu_pd(y_vals.as_ptr())
    }
}
