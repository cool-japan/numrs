//! Exponential and logarithmic SIMD operations
//!
//! This module provides AVX2 optimized implementations for:
//! - exp_f32, exp_f64: Exponential function
//! - log_f32, log_f64: Natural logarithm
//! - log10_f32, log10_f64: Base-10 logarithm
//! - log2_f32, log2_f64: Base-2 logarithm
//! - pow_f32, pow_f64: Power function x^n
//! - sqrt_f32, sqrt_f64: Square root
//! - log1p_f64: log(1+x) for small x
//! - expm1_f64: exp(x)-1 for small x
//! - cbrt_f64: Cube root

use super::EnhancedSimdOps;
// See `arithmetic.rs` for why these are gated the same as the intrinsics.
#[cfg(target_arch = "x86_64")]
use super::{AVX2_F32_LANES, AVX2_F64_LANES, PREFETCH_DISTANCE};
#[cfg(target_arch = "x86_64")]
use crate::array::Array;
#[cfg(target_arch = "x86_64")]
use crate::error::Result;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl EnhancedSimdOps {
    // ========================================
    // Exponential Functions
    // ========================================

    /// Vectorized exponential function with Taylor series approximation
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_exp_f32(input: &Array<f32>) -> Result<Array<f32>> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_exp_f32(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized exponential function
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_exp_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for exp approximation
        let log2_e = _mm256_set1_ps(1.4426950408889634);
        let ln2_hi = _mm256_set1_ps(0.6931471805599453);
        let ln2_lo = _mm256_set1_ps(2.3283064365386963e-10);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(0.5);
        let c4 = _mm256_set1_ps(0.16666666666666666);
        let c5 = _mm256_set1_ps(0.041666666666666664);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Range reduction: x = n*ln(2) + r
            let n_float = _mm256_mul_ps(x, log2_e);
            let n = _mm256_cvtps_epi32(n_float);
            let n_f = _mm256_cvtepi32_ps(n);

            // r = x - n*ln(2)
            let r = _mm256_fmsub_ps(n_f, ln2_hi, x);
            let r = _mm256_fmsub_ps(n_f, ln2_lo, r);

            // Taylor series: exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4!
            let r2 = _mm256_mul_ps(r, r);
            let r3 = _mm256_mul_ps(r2, r);
            let r4 = _mm256_mul_ps(r3, r);

            let poly = _mm256_fmadd_ps(
                c5,
                r4,
                _mm256_fmadd_ps(c4, r3, _mm256_fmadd_ps(c3, r2, _mm256_fmadd_ps(c2, r, c1))),
            );

            // Scale by 2^n
            let result = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_add_epi32(n, _mm256_set1_epi32(127)),
                23,
            ));
            let final_result = _mm256_mul_ps(poly, result);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), final_result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].exp();
        }
    }

    /// Vectorized exponential function for f64 with Taylor series approximation
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_exp_f64(input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_exp_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized exponential function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_exp_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Constants for exp approximation (high precision for f64)
        let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
        let ln2_hi = _mm256_set1_pd(0.693147180559945309417232121458176568);
        let ln2_lo = _mm256_set1_pd(1.94210120611385413671396746603066e-16);

        // Taylor series coefficients: 1/n! for n = 0..7
        let c0 = _mm256_set1_pd(1.0);
        let c1 = _mm256_set1_pd(1.0);
        let c2 = _mm256_set1_pd(0.5);
        let c3 = _mm256_set1_pd(0.16666666666666666);
        let c4 = _mm256_set1_pd(0.041666666666666664);
        let c5 = _mm256_set1_pd(0.008333333333333333);
        let c6 = _mm256_set1_pd(0.001388888888888889);
        let c7 = _mm256_set1_pd(0.0001984126984126984);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction: x = n*ln(2) + r
            let n_float = _mm256_mul_pd(x, log2_e);
            let n = _mm256_cvtpd_epi32(n_float);
            let n_wide = _mm256_cvtepi32_epi64(n);
            let n_f = _mm256_cvtepi32_pd(n);

            // r = x - n*ln(2)
            let r = _mm256_sub_pd(x, _mm256_mul_pd(n_f, ln2_hi));
            let r = _mm256_sub_pd(r, _mm256_mul_pd(n_f, ln2_lo));

            // Taylor series: exp(r) ≈ 1 + r + r²/2! + ... + r⁷/7!
            let r2 = _mm256_mul_pd(r, r);
            let r3 = _mm256_mul_pd(r2, r);
            let r4 = _mm256_mul_pd(r2, r2);

            let poly_high =
                _mm256_fmadd_pd(c7, r3, _mm256_fmadd_pd(c6, r2, _mm256_fmadd_pd(c5, r, c4)));
            let poly_low =
                _mm256_fmadd_pd(c3, r3, _mm256_fmadd_pd(c2, r2, _mm256_fmadd_pd(c1, r, c0)));
            let poly = _mm256_fmadd_pd(poly_high, r4, poly_low);

            // Scale by 2^n
            let bias = _mm256_set1_epi64x(1023);
            let n_biased = _mm256_add_epi64(n_wide, bias);
            let exp_scale = _mm256_slli_epi64(n_biased, 52);
            let scale = _mm256_castsi256_pd(exp_scale);

            let final_result = _mm256_mul_pd(poly, scale);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), final_result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].exp();
        }
    }

    // ========================================
    // Logarithmic Functions
    // ========================================

    /// Vectorized logarithm function with Newton-Raphson refinement
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.ln())
    }

    /// AVX2 optimized logarithm function
    // Implemented but not wired to the public wrapper, which currently uses a
    // scalar fallback. The whole `simd_optimize` module is deprecated in favor
    // of `scirs2_core::simd_ops::SimdUnifiedOps`, so wiring this up is not
    // planned. Retained for reference rather than deleted.
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for log approximation
        let ln2 = _mm256_set1_ps(0.6931471805599453);
        let c1 = _mm256_set1_ps(-0.5);
        let c2 = _mm256_set1_ps(0.33333333333333333);
        let c3 = _mm256_set1_ps(-0.25);
        let c4 = _mm256_set1_ps(0.2);
        let one = _mm256_set1_ps(1.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Extract exponent
            let x_int = _mm256_castps_si256(x);
            let exp = _mm256_sub_epi32(_mm256_srli_epi32(x_int, 23), _mm256_set1_epi32(127));
            let exp_f = _mm256_cvtepi32_ps(exp);

            // Extract mantissa
            let mantissa = _mm256_castsi256_ps(_mm256_or_si256(
                _mm256_and_si256(x_int, _mm256_set1_epi32(0x007FFFFF)),
                _mm256_set1_epi32(0x3F800000),
            ));

            // Polynomial approximation for log(1+u)
            let u = _mm256_sub_ps(mantissa, one);
            let u2 = _mm256_mul_ps(u, u);
            let u3 = _mm256_mul_ps(u2, u);
            let u4 = _mm256_mul_ps(u3, u);

            let poly = _mm256_fmadd_ps(
                c4,
                u4,
                _mm256_fmadd_ps(c3, u3, _mm256_fmadd_ps(c2, u2, _mm256_fmadd_ps(c1, u2, u))),
            );

            // log(x) = exp * ln(2) + log(mantissa)
            let result = _mm256_fmadd_ps(exp_f, ln2, poly);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].ln();
        }
    }

    /// Vectorized logarithm function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ln())
    }

    /// AVX2 optimized logarithm function for f64
    // Unwired AVX2 kernel -- see the note on `avx2_log_f32` above.
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Constants for log approximation
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let c1 = _mm256_set1_pd(-0.5);
        let c2 = _mm256_set1_pd(1.0 / 3.0);
        let c3 = _mm256_set1_pd(-0.25);
        let c4 = _mm256_set1_pd(0.2);
        let c5 = _mm256_set1_pd(-1.0 / 6.0);
        let c6 = _mm256_set1_pd(1.0 / 7.0);
        let one = _mm256_set1_pd(1.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Extract exponent
            let x_int = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_int, 52);
            let bias = _mm256_set1_epi64x(1023);
            let exp_unbiased = _mm256_sub_epi64(exp_bits, bias);

            // Convert i64 to f64
            let mut exp_arr = [0i64; 4];
            _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_unbiased);
            let exp_f = _mm256_set_pd(
                exp_arr[3] as f64,
                exp_arr[2] as f64,
                exp_arr[1] as f64,
                exp_arr[0] as f64,
            );

            // Extract mantissa
            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000);
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            // Polynomial approximation
            let u = _mm256_sub_pd(mantissa, one);
            let u2 = _mm256_mul_pd(u, u);
            let u3 = _mm256_mul_pd(u2, u);
            let u4 = _mm256_mul_pd(u2, u2);
            let u5 = _mm256_mul_pd(u4, u);
            let u6 = _mm256_mul_pd(u3, u3);
            let u7 = _mm256_mul_pd(u4, u3);

            let poly = _mm256_fmadd_pd(
                c6,
                u7,
                _mm256_fmadd_pd(
                    c5,
                    u6,
                    _mm256_fmadd_pd(
                        c4,
                        u5,
                        _mm256_fmadd_pd(
                            c3,
                            u4,
                            _mm256_fmadd_pd(c2, u3, _mm256_fmadd_pd(c1, u2, u)),
                        ),
                    ),
                ),
            );

            // log(x) = exp * ln(2) + log(mantissa)
            let result = _mm256_fmadd_pd(exp_f, ln2, poly);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].ln();
        }
    }

    /// Vectorized log10 function (log base 10)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log10_f32(input: &Array<f32>) -> Array<f32> {
        input.map(|x| x.log10())
    }

    /// AVX2 optimized log10 using log(x) * log10(e)
    // Unwired AVX2 kernel -- see the note on `avx2_log_f32` above.
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log10_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let log10_e = _mm256_set1_ps(std::f32::consts::LOG10_E);
        let one = _mm256_set1_ps(1.0);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let c1 = _mm256_set1_ps(2.0);
        let c3 = _mm256_set1_ps(2.0 / 3.0);
        let c5 = _mm256_set1_ps(2.0 / 5.0);
        let c7 = _mm256_set1_ps(2.0 / 7.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Extract exponent and mantissa
            let exp_mask = _mm256_set1_epi32(0x7F800000_u32 as i32);
            let mant_mask = _mm256_set1_epi32(0x007FFFFF_u32 as i32);

            let x_bits = _mm256_castps_si256(x);
            let exp_bits = _mm256_and_si256(x_bits, exp_mask);
            let exp = _mm256_sub_epi32(_mm256_srli_epi32(exp_bits, 23), _mm256_set1_epi32(127));
            let exp_f = _mm256_cvtepi32_ps(exp);

            // Normalize mantissa to [1, 2)
            let mant_bits = _mm256_or_si256(
                _mm256_and_si256(x_bits, mant_mask),
                _mm256_set1_epi32(0x3F800000_u32 as i32),
            );
            let m = _mm256_castsi256_ps(mant_bits);

            // Compute ln(m) using Taylor series
            let y = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let y2 = _mm256_mul_ps(y, y);

            let term3 = _mm256_mul_ps(_mm256_mul_ps(y, y2), c3);
            let term5 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(y, y2), y2), c5);
            let term7 = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(y, y2), y2), y2),
                c7,
            );

            let ln_m = _mm256_mul_ps(
                c1,
                _mm256_add_ps(y, _mm256_add_ps(term3, _mm256_add_ps(term5, term7))),
            );

            // ln(x) = ln(m) + exp * ln(2)
            let ln_x = _mm256_fmadd_ps(exp_f, ln2, ln_m);

            // log10(x) = ln(x) * log10(e)
            let result = _mm256_mul_ps(ln_x, log10_e);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].log10();
        }
    }

    /// Vectorized log10 function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log10_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.log10())
    }

    /// AVX2 optimized log10 for f64
    // Unwired AVX2 kernel -- see the note on `avx2_log_f32` above.
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log10_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let log10_e = _mm256_set1_pd(std::f64::consts::LOG10_E);
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let one = _mm256_set1_pd(1.0);
        let c1 = _mm256_set1_pd(-0.5);
        let c2 = _mm256_set1_pd(1.0 / 3.0);
        let c3 = _mm256_set1_pd(-0.25);
        let c4 = _mm256_set1_pd(0.2);
        let c5 = _mm256_set1_pd(-1.0 / 6.0);
        let c6 = _mm256_set1_pd(1.0 / 7.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Extract exponent
            let x_int = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_int, 52);
            let bias = _mm256_set1_epi64x(1023);
            let exp_unbiased = _mm256_sub_epi64(exp_bits, bias);

            let mut exp_arr = [0i64; 4];
            _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_unbiased);
            let exp_f = _mm256_set_pd(
                exp_arr[3] as f64,
                exp_arr[2] as f64,
                exp_arr[1] as f64,
                exp_arr[0] as f64,
            );

            // Extract mantissa
            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000);
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            // Polynomial approximation
            let u = _mm256_sub_pd(mantissa, one);
            let u2 = _mm256_mul_pd(u, u);
            let u3 = _mm256_mul_pd(u2, u);
            let u4 = _mm256_mul_pd(u2, u2);
            let u5 = _mm256_mul_pd(u4, u);
            let u6 = _mm256_mul_pd(u3, u3);
            let u7 = _mm256_mul_pd(u4, u3);

            let poly = _mm256_fmadd_pd(
                c6,
                u7,
                _mm256_fmadd_pd(
                    c5,
                    u6,
                    _mm256_fmadd_pd(
                        c4,
                        u5,
                        _mm256_fmadd_pd(
                            c3,
                            u4,
                            _mm256_fmadd_pd(c2, u3, _mm256_fmadd_pd(c1, u2, u)),
                        ),
                    ),
                ),
            );

            // log(x) = exp * ln(2) + log(mantissa)
            let ln_x = _mm256_fmadd_pd(exp_f, ln2, poly);

            // log10(x) = ln(x) * log10(e)
            let result = _mm256_mul_pd(ln_x, log10_e);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].log10();
        }
    }

    /// Vectorized log2 function (log base 2)
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log2_f32(input: &Array<f32>) -> Result<Array<f32>> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_log2_f32(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized log2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log2_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let one = _mm256_set1_ps(1.0);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let c1 = _mm256_set1_ps(2.0);
        let c3 = _mm256_set1_ps(2.0 / 3.0);
        let c5 = _mm256_set1_ps(2.0 / 5.0);
        let c7 = _mm256_set1_ps(2.0 / 7.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            let exp_mask = _mm256_set1_epi32(0x7F800000_u32 as i32);
            let mant_mask = _mm256_set1_epi32(0x007FFFFF_u32 as i32);

            let x_bits = _mm256_castps_si256(x);
            let exp_bits = _mm256_and_si256(x_bits, exp_mask);
            let exp = _mm256_sub_epi32(_mm256_srli_epi32(exp_bits, 23), _mm256_set1_epi32(127));
            let exp_f = _mm256_cvtepi32_ps(exp);

            let mant_bits = _mm256_or_si256(
                _mm256_and_si256(x_bits, mant_mask),
                _mm256_set1_epi32(0x3F800000_u32 as i32),
            );
            let m = _mm256_castsi256_ps(mant_bits);

            let y = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let y2 = _mm256_mul_ps(y, y);

            let term3 = _mm256_mul_ps(_mm256_mul_ps(y, y2), c3);
            let term5 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(y, y2), y2), c5);
            let term7 = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(y, y2), y2), y2),
                c7,
            );

            let ln_m = _mm256_mul_ps(
                c1,
                _mm256_add_ps(y, _mm256_add_ps(term3, _mm256_add_ps(term5, term7))),
            );

            let ln_x = _mm256_fmadd_ps(exp_f, ln2, ln_m);

            let result = _mm256_mul_ps(ln_x, log2_e);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].log2();
        }
    }

    /// Vectorized log2 function for f64
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log2_f64(input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_log2_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized log2 for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log2_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let one = _mm256_set1_pd(1.0);
        let c1 = _mm256_set1_pd(-0.5);
        let c2 = _mm256_set1_pd(1.0 / 3.0);
        let c3 = _mm256_set1_pd(-0.25);
        let c4 = _mm256_set1_pd(0.2);
        let c5 = _mm256_set1_pd(-1.0 / 6.0);
        let c6 = _mm256_set1_pd(1.0 / 7.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            let x_int = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_int, 52);
            let bias = _mm256_set1_epi64x(1023);
            let exp_unbiased = _mm256_sub_epi64(exp_bits, bias);

            let mut exp_arr = [0i64; 4];
            _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_unbiased);
            let exp_f = _mm256_set_pd(
                exp_arr[3] as f64,
                exp_arr[2] as f64,
                exp_arr[1] as f64,
                exp_arr[0] as f64,
            );

            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000);
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            let u = _mm256_sub_pd(mantissa, one);
            let u2 = _mm256_mul_pd(u, u);
            let u3 = _mm256_mul_pd(u2, u);
            let u4 = _mm256_mul_pd(u2, u2);
            let u5 = _mm256_mul_pd(u4, u);
            let u6 = _mm256_mul_pd(u3, u3);
            let u7 = _mm256_mul_pd(u4, u3);

            let poly = _mm256_fmadd_pd(
                c6,
                u7,
                _mm256_fmadd_pd(
                    c5,
                    u6,
                    _mm256_fmadd_pd(
                        c4,
                        u5,
                        _mm256_fmadd_pd(
                            c3,
                            u4,
                            _mm256_fmadd_pd(c2, u3, _mm256_fmadd_pd(c1, u2, u)),
                        ),
                    ),
                ),
            );

            let ln_x = _mm256_fmadd_pd(exp_f, ln2, poly);

            let result = _mm256_mul_pd(ln_x, log2_e);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].log2();
        }
    }

    // ========================================
    // Power Functions
    // ========================================

    /// Vectorized power function for f32: pow(x, n) = x^n
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_pow_f32(input: &Array<f32>, n: f32) -> Array<f32> {
        input.map(|x| x.powf(n))
    }

    /// AVX2 optimized power function for f32
    // Unwired AVX2 kernel -- see the note on `avx2_log_f32` above.
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_pow_f32(input: &[f32], n: f32, output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for log approximation
        let ln2 = _mm256_set1_ps(0.6931471805599453);
        let log_c1 = _mm256_set1_ps(-0.5);
        let log_c2 = _mm256_set1_ps(0.33333333333333333);
        let log_c3 = _mm256_set1_ps(-0.25);
        let log_c4 = _mm256_set1_ps(0.2);
        let one_ps = _mm256_set1_ps(1.0);

        // Constants for exp approximation
        let log2_e = _mm256_set1_ps(1.4426950408889634);
        let ln2_hi = _mm256_set1_ps(0.6931471805599453);
        let ln2_lo = _mm256_set1_ps(2.3283064365386963e-10);
        let exp_c1 = _mm256_set1_ps(1.0);
        let exp_c2 = _mm256_set1_ps(1.0);
        let exp_c3 = _mm256_set1_ps(0.5);
        let exp_c4 = _mm256_set1_ps(0.16666666666666666);
        let exp_c5 = _mm256_set1_ps(0.041666666666666664);

        let n_vec = _mm256_set1_ps(n);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Step 1: Compute log(x)
            let x_int = _mm256_castps_si256(x);
            let exp = _mm256_sub_epi32(_mm256_srli_epi32(x_int, 23), _mm256_set1_epi32(127));
            let exp_f = _mm256_cvtepi32_ps(exp);

            let mantissa = _mm256_castsi256_ps(_mm256_or_si256(
                _mm256_and_si256(x_int, _mm256_set1_epi32(0x007FFFFF)),
                _mm256_set1_epi32(0x3F800000),
            ));

            let u = _mm256_sub_ps(mantissa, one_ps);
            let u2 = _mm256_mul_ps(u, u);
            let u3 = _mm256_mul_ps(u2, u);
            let u4 = _mm256_mul_ps(u3, u);

            let log_poly = _mm256_fmadd_ps(
                log_c4,
                u4,
                _mm256_fmadd_ps(
                    log_c3,
                    u3,
                    _mm256_fmadd_ps(log_c2, u2, _mm256_fmadd_ps(log_c1, u2, u)),
                ),
            );

            let log_x = _mm256_fmadd_ps(exp_f, ln2, log_poly);

            // Step 2: Compute n * log(x)
            let n_log_x = _mm256_mul_ps(n_vec, log_x);

            // Step 3: Compute exp(n * log(x))
            let k_float = _mm256_mul_ps(n_log_x, log2_e);
            let k = _mm256_cvtps_epi32(k_float);
            let k_f = _mm256_cvtepi32_ps(k);

            let r = _mm256_fmsub_ps(k_f, ln2_hi, n_log_x);
            let r = _mm256_fmsub_ps(k_f, ln2_lo, r);

            let r2 = _mm256_mul_ps(r, r);
            let r3 = _mm256_mul_ps(r2, r);
            let r4 = _mm256_mul_ps(r3, r);

            let exp_poly = _mm256_fmadd_ps(
                exp_c5,
                r4,
                _mm256_fmadd_ps(
                    exp_c4,
                    r3,
                    _mm256_fmadd_ps(exp_c3, r2, _mm256_fmadd_ps(exp_c2, r, exp_c1)),
                ),
            );

            let scale = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_add_epi32(k, _mm256_set1_epi32(127)),
                23,
            ));
            let pow_result = _mm256_mul_ps(exp_poly, scale);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), pow_result);
        }

        for i in simd_len..len {
            output[i] = input[i].powf(n);
        }
    }

    /// Vectorized power function for f64: pow(x, n) = x^n
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_pow_f64(input: &Array<f64>, n: f64) -> Array<f64> {
        input.map(|x| x.powf(n))
    }

    /// AVX2 optimized power function for f64
    // Unwired AVX2 kernel -- see the note on `avx2_log_f32` above.
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_pow_f64(input: &[f64], n: f64, output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let log_c1 = _mm256_set1_pd(-0.5);
        let log_c2 = _mm256_set1_pd(1.0 / 3.0);
        let log_c3 = _mm256_set1_pd(-0.25);
        let log_c4 = _mm256_set1_pd(0.2);
        let log_c5 = _mm256_set1_pd(-1.0 / 6.0);
        let one_pd = _mm256_set1_pd(1.0);

        let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
        let ln2_hi = _mm256_set1_pd(0.693147180559945309417232121458176568);
        let ln2_lo = _mm256_set1_pd(1.94210120611385413671396746603066e-16);
        let exp_c0 = _mm256_set1_pd(1.0);
        let exp_c1 = _mm256_set1_pd(1.0);
        let exp_c2 = _mm256_set1_pd(0.5);
        let exp_c3 = _mm256_set1_pd(0.16666666666666666);
        let exp_c4 = _mm256_set1_pd(0.041666666666666664);
        let exp_c5 = _mm256_set1_pd(0.008333333333333333);
        let exp_c6 = _mm256_set1_pd(0.001388888888888889);

        let n_vec = _mm256_set1_pd(n);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Step 1: Compute log(x)
            let x_int = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_int, 52);
            let bias = _mm256_set1_epi64x(1023);
            let exp_unbiased = _mm256_sub_epi64(exp_bits, bias);

            let mut exp_arr = [0i64; 4];
            _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_unbiased);
            let exp_f = _mm256_set_pd(
                exp_arr[3] as f64,
                exp_arr[2] as f64,
                exp_arr[1] as f64,
                exp_arr[0] as f64,
            );

            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000);
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            let u = _mm256_sub_pd(mantissa, one_pd);
            let u2 = _mm256_mul_pd(u, u);
            let u3 = _mm256_mul_pd(u2, u);
            let u4 = _mm256_mul_pd(u3, u);
            let u5 = _mm256_mul_pd(u4, u);

            let log_poly = _mm256_fmadd_pd(
                log_c5,
                u5,
                _mm256_fmadd_pd(
                    log_c4,
                    u4,
                    _mm256_fmadd_pd(
                        log_c3,
                        u3,
                        _mm256_fmadd_pd(log_c2, u2, _mm256_fmadd_pd(log_c1, u2, u)),
                    ),
                ),
            );

            let log_x = _mm256_fmadd_pd(exp_f, ln2, log_poly);

            // Step 2: Compute n * log(x)
            let n_log_x = _mm256_mul_pd(n_vec, log_x);

            // Step 3: Compute exp(n * log(x))
            let k_float = _mm256_mul_pd(n_log_x, log2_e);
            let k = _mm256_cvtpd_epi32(k_float);
            let k_wide = _mm256_cvtepi32_epi64(k);
            let k_f = _mm256_cvtepi32_pd(k);

            let r = _mm256_sub_pd(n_log_x, _mm256_mul_pd(k_f, ln2_hi));
            let r = _mm256_sub_pd(r, _mm256_mul_pd(k_f, ln2_lo));

            let r2 = _mm256_mul_pd(r, r);
            let r3 = _mm256_mul_pd(r2, r);
            let r4 = _mm256_mul_pd(r2, r2);

            let exp_poly_high = _mm256_fmadd_pd(exp_c6, r2, _mm256_fmadd_pd(exp_c5, r, exp_c4));
            let exp_poly_low = _mm256_fmadd_pd(
                exp_c3,
                r3,
                _mm256_fmadd_pd(exp_c2, r2, _mm256_fmadd_pd(exp_c1, r, exp_c0)),
            );
            let exp_poly = _mm256_fmadd_pd(exp_poly_high, r4, exp_poly_low);

            let bias_64 = _mm256_set1_epi64x(1023);
            let k_biased = _mm256_add_epi64(k_wide, bias_64);
            let exp_scale = _mm256_slli_epi64(k_biased, 52);
            let scale = _mm256_castsi256_pd(exp_scale);

            let pow_result = _mm256_mul_pd(exp_poly, scale);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), pow_result);
        }

        for i in simd_len..len {
            output[i] = input[i].powf(n);
        }
    }

    // ========================================
    // Square Root Functions
    // ========================================

    /// Vectorized square root function for f32
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sqrt_f32(input: &Array<f32>) -> Result<Array<f32>> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_sqrt_f32(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized square root for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sqrt_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let result = _mm256_sqrt_ps(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].sqrt();
        }
    }

    /// Vectorized square root function for f64
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sqrt_f64(input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sqrt_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized square root for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sqrt_f64(input: &[f64], output: &mut [f64]) {
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
            let result = _mm256_sqrt_pd(x);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].sqrt();
        }
    }

    // ========================================
    // Special Exponential/Log Functions
    // ========================================

    /// Vectorized log1p for f64 (more accurate for small x)
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log1p_f64(input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_log1p_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized log1p for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log1p_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Taylor series coefficients for ln(1+x)
        let c2 = _mm256_set1_pd(-0.5);
        let c3 = _mm256_set1_pd(1.0 / 3.0);
        let c4 = _mm256_set1_pd(-0.25);
        let c5 = _mm256_set1_pd(0.2);
        let c6 = _mm256_set1_pd(-1.0 / 6.0);
        let c7 = _mm256_set1_pd(1.0 / 7.0);
        let c8 = _mm256_set1_pd(-0.125);
        let threshold = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // For small |x|, use Taylor series
            let abs_x = _mm256_and_pd(
                x,
                _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)),
            );
            let use_taylor = _mm256_cmp_pd(abs_x, threshold, _CMP_LT_OQ);

            // Taylor series: x - x²/2 + x³/3 - x⁴/4 + ...
            let x2 = _mm256_mul_pd(x, x);
            let x3 = _mm256_mul_pd(x2, x);
            let x4 = _mm256_mul_pd(x2, x2);
            let x5 = _mm256_mul_pd(x4, x);
            let x6 = _mm256_mul_pd(x3, x3);
            let x7 = _mm256_mul_pd(x4, x3);
            let x8 = _mm256_mul_pd(x4, x4);

            let taylor = _mm256_fmadd_pd(
                c8,
                x8,
                _mm256_fmadd_pd(
                    c7,
                    x7,
                    _mm256_fmadd_pd(
                        c6,
                        x6,
                        _mm256_fmadd_pd(
                            c5,
                            x5,
                            _mm256_fmadd_pd(
                                c4,
                                x4,
                                _mm256_fmadd_pd(c3, x3, _mm256_fmadd_pd(c2, x2, x)),
                            ),
                        ),
                    ),
                ),
            );

            // For larger |x|, use regular log
            let one = _mm256_set1_pd(1.0);
            let one_plus_x = _mm256_add_pd(one, x);

            // Extract from scalar
            let mut arr = [0.0f64; 4];
            _mm256_storeu_pd(arr.as_mut_ptr(), one_plus_x);
            let regular = _mm256_set_pd(arr[3].ln(), arr[2].ln(), arr[1].ln(), arr[0].ln());

            let result = _mm256_blendv_pd(regular, taylor, use_taylor);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].ln_1p();
        }
    }

    /// Vectorized expm1 for f64 (more accurate for small x)
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_expm1_f64(input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_expm1_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized expm1 for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_expm1_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Taylor series coefficients for exp(x) - 1
        let c2 = _mm256_set1_pd(0.5);
        let c3 = _mm256_set1_pd(1.0 / 6.0);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c5 = _mm256_set1_pd(1.0 / 120.0);
        let c6 = _mm256_set1_pd(1.0 / 720.0);
        let c7 = _mm256_set1_pd(1.0 / 5040.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);
        let threshold = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            let abs_x = _mm256_and_pd(
                x,
                _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)),
            );
            let use_taylor = _mm256_cmp_pd(abs_x, threshold, _CMP_LT_OQ);

            // Taylor series: x + x²/2! + x³/3! + ...
            let x2 = _mm256_mul_pd(x, x);
            let x3 = _mm256_mul_pd(x2, x);
            let x4 = _mm256_mul_pd(x2, x2);
            let x5 = _mm256_mul_pd(x4, x);
            let x6 = _mm256_mul_pd(x3, x3);
            let x7 = _mm256_mul_pd(x4, x3);
            let x8 = _mm256_mul_pd(x4, x4);

            let taylor = _mm256_fmadd_pd(
                c8,
                x8,
                _mm256_fmadd_pd(
                    c7,
                    x7,
                    _mm256_fmadd_pd(
                        c6,
                        x6,
                        _mm256_fmadd_pd(
                            c5,
                            x5,
                            _mm256_fmadd_pd(
                                c4,
                                x4,
                                _mm256_fmadd_pd(c3, x3, _mm256_fmadd_pd(c2, x2, x)),
                            ),
                        ),
                    ),
                ),
            );

            // For larger |x|, use regular exp - 1
            let mut arr = [0.0f64; 4];
            _mm256_storeu_pd(arr.as_mut_ptr(), x);
            let regular = _mm256_set_pd(
                arr[3].exp() - 1.0,
                arr[2].exp() - 1.0,
                arr[1].exp() - 1.0,
                arr[0].exp() - 1.0,
            );

            let result = _mm256_blendv_pd(regular, taylor, use_taylor);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].exp_m1();
        }
    }

    /// Vectorized cube root for f64
    ///
    /// # Errors
    ///
    /// Returns `NumRs2Error::ShapeMismatch` if the result cannot be reshaped
    /// to the input shape.
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cbrt_f64(input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_cbrt_f64(&data, &mut result);
        }

        Array::from_vec_shape(result, &input.shape())
    }

    /// AVX2 optimized cube root for f64
    /// Uses x^(1/3) = exp(log(x)/3) for positive values
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cbrt_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let one_third = _mm256_set1_pd(1.0 / 3.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Get sign
            let sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);

            // Take absolute value
            let abs_x = _mm256_and_pd(
                x,
                _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)),
            );

            // Compute cbrt using scalar for simplicity
            let mut arr = [0.0f64; 4];
            _mm256_storeu_pd(arr.as_mut_ptr(), abs_x);
            let cbrt_vals = _mm256_set_pd(
                arr[3].powf(1.0 / 3.0),
                arr[2].powf(1.0 / 3.0),
                arr[1].powf(1.0 / 3.0),
                arr[0].powf(1.0 / 3.0),
            );

            // Restore sign
            let neg_cbrt = _mm256_sub_pd(_mm256_setzero_pd(), cbrt_vals);
            let result = _mm256_blendv_pd(cbrt_vals, neg_cbrt, sign_mask);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Suppress unused variable warning
        let _ = one_third;

        for i in simd_len..len {
            output[i] = input[i].cbrt();
        }
    }
}
