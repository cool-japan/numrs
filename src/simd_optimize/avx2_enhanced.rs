//! Enhanced AVX2 SIMD operations with advanced vectorization
//!
//! This module provides highly optimized SIMD implementations for advanced
//! mathematical operations, cache-aware algorithms, and specialized functions.

use crate::array::Array;
#[allow(unused_imports)] // Used in some configurations
use crate::error::{NumRs2Error, Result};
use crate::simd::SimdOps;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Enhanced vectorization constants
#[allow(dead_code)]
const AVX2_F32_LANES: usize = 8;
#[allow(dead_code)]
const AVX2_F64_LANES: usize = 4;
#[allow(dead_code)]
const CACHE_LINE_SIZE: usize = 64;
#[allow(dead_code)]
const L1_CACHE_SIZE: usize = 32 * 1024;
#[allow(dead_code)]
const PREFETCH_DISTANCE: usize = 512;

/// Advanced vectorized operations with cache optimization
pub struct EnhancedSimdOps;

impl EnhancedSimdOps {
    /// Cache-aware matrix multiplication with SIMD optimization
    #[cfg(target_arch = "x86_64")]
    pub fn cache_aware_matmul_f32(
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
            Self::blocked_matmul_avx2_f32(&a_data, &b_data, &mut c_data, m, n, k, block_size);
        }

        *c = Array::from_vec(c_data).reshape(&[m, n]);
        Ok(())
    }

    /// Blocked matrix multiplication with AVX2 optimization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn blocked_matmul_avx2_f32(
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
                        for j in (jj..j_end).step_by(AVX2_F32_LANES) {
                            let lanes = (j_end - j).min(AVX2_F32_LANES);

                            // Load C values
                            let mut vc = if lanes == AVX2_F32_LANES {
                                _mm256_loadu_ps(c.as_ptr().add(i * n + j))
                            } else {
                                let mut temp = [0.0f32; AVX2_F32_LANES];
                                for l in 0..lanes {
                                    temp[l] = c[i * n + j + l];
                                }
                                _mm256_loadu_ps(temp.as_ptr())
                            };

                            for l in kk..k_end {
                                let va = _mm256_set1_ps(a[i * k + l]);
                                let vb = if lanes == AVX2_F32_LANES {
                                    _mm256_loadu_ps(b.as_ptr().add(l * n + j))
                                } else {
                                    let mut temp = [0.0f32; AVX2_F32_LANES];
                                    for idx in 0..lanes {
                                        temp[idx] = b[l * n + j + idx];
                                    }
                                    _mm256_loadu_ps(temp.as_ptr())
                                };
                                vc = _mm256_fmadd_ps(va, vb, vc);
                            }

                            // Store C values
                            if lanes == AVX2_F32_LANES {
                                _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), vc);
                            } else {
                                let mut temp = [0.0f32; AVX2_F32_LANES];
                                _mm256_storeu_ps(temp.as_mut_ptr(), vc);
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

    /// Vectorized exponential function with Taylor series approximation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_exp_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_exp_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
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
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_exp_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_exp_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized exponential function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_exp_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Constants for exp approximation (high precision for f64)
        let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E); // 1.4426950408889634073599246810019
        let ln2_hi = _mm256_set1_pd(0.693147180559945309417232121458176568); // High part of ln(2)
        let ln2_lo = _mm256_set1_pd(1.94210120611385413671396746603066e-16); // Low part of ln(2)

        // Taylor series coefficients: 1/n! for n = 0..7
        let c0 = _mm256_set1_pd(1.0);
        let c1 = _mm256_set1_pd(1.0);
        let c2 = _mm256_set1_pd(0.5);
        let c3 = _mm256_set1_pd(0.16666666666666666); // 1/6
        let c4 = _mm256_set1_pd(0.041666666666666664); // 1/24
        let c5 = _mm256_set1_pd(0.008333333333333333); // 1/120
        let c6 = _mm256_set1_pd(0.001388888888888889); // 1/720
        let c7 = _mm256_set1_pd(0.0001984126984126984); // 1/5040

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line (adjusted for f64 size)
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction: x = n*ln(2) + r, where r is small
            let n_float = _mm256_mul_pd(x, log2_e);
            let n = _mm256_cvtpd_epi32(n_float);
            let n_wide = _mm256_cvtepi32_epi64(n);
            let n_f = _mm256_cvtepi32_pd(n);

            // r = x - n*ln(2) using extended precision
            let r = _mm256_sub_pd(x, _mm256_mul_pd(n_f, ln2_hi));
            let r = _mm256_sub_pd(r, _mm256_mul_pd(n_f, ln2_lo));

            // Taylor series: exp(r) ≈ 1 + r + r²/2! + r³/3! + ... + r⁷/7!
            // Using Horner's method for numerical stability
            let r2 = _mm256_mul_pd(r, r);
            let r3 = _mm256_mul_pd(r2, r);
            let r4 = _mm256_mul_pd(r2, r2);

            // poly = c0 + c1*r + c2*r² + c3*r³ + c4*r⁴ + c5*r⁵ + c6*r⁶ + c7*r⁷
            let poly_high =
                _mm256_fmadd_pd(c7, r3, _mm256_fmadd_pd(c6, r2, _mm256_fmadd_pd(c5, r, c4)));
            let poly_low =
                _mm256_fmadd_pd(c3, r3, _mm256_fmadd_pd(c2, r2, _mm256_fmadd_pd(c1, r, c0)));
            let poly = _mm256_fmadd_pd(poly_high, r4, poly_low);

            // Scale by 2^n: multiply by 2^n by manipulating the exponent bits
            // For f64: exponent bias is 1023, mantissa is 52 bits
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

    /// Vectorized logarithm function with Newton-Raphson refinement
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_log_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized logarithm function
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

            // Use polynomial approximation for log(1+x) where x = mantissa - 1
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

    /// Vectorized logarithm function for f64 with high-precision polynomial approximation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_log_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized logarithm function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Constants for log approximation (high precision for f64)
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        // Taylor series coefficients for ln(1+u): u - u²/2 + u³/3 - u⁴/4 + u⁵/5 - u⁶/6 + u⁷/7
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

            // Extract exponent (f64: bias=1023, mantissa=52 bits)
            let x_int = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_int, 52);
            let bias = _mm256_set1_epi64x(1023);
            let exp_unbiased = _mm256_sub_epi64(exp_bits, bias);

            // Convert i64 to f64 (manual conversion since _mm256_cvtepi64_pd doesn't exist in AVX2)
            let mut exp_arr = [0i64; 4];
            _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_unbiased);
            let exp_f = _mm256_set_pd(
                exp_arr[3] as f64,
                exp_arr[2] as f64,
                exp_arr[1] as f64,
                exp_arr[0] as f64,
            );

            // Extract mantissa and set exponent to 0 (so value is in [1, 2))
            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000); // 1.0 in f64 representation
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            // Polynomial approximation for ln(1+u) where u = mantissa - 1
            let u = _mm256_sub_pd(mantissa, one);
            let u2 = _mm256_mul_pd(u, u);
            let u3 = _mm256_mul_pd(u2, u);
            let u4 = _mm256_mul_pd(u2, u2);
            let u5 = _mm256_mul_pd(u4, u);
            let u6 = _mm256_mul_pd(u3, u3);
            let u7 = _mm256_mul_pd(u4, u3);

            // poly = u + c1*u² + c2*u³ + c3*u⁴ + c4*u⁵ + c5*u⁶ + c6*u⁷
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
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_log10_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized log10 using log(x) * log10(e)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log10_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // log10(x) = ln(x) * log10(e) = ln(x) * 0.4342944819...
        let log10_e = _mm256_set1_ps(std::f32::consts::LOG10_E);

        // Constants for ln approximation (same as vectorized_log_f32)
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

            // Taylor series: 2*(y + y^3/3 + y^5/5 + y^7/7)
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

    /// Vectorized log2 function (log base 2)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log2_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_log2_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized log2 using log(x) * log2(e)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log2_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // log2(x) = ln(x) * log2(e) = ln(x) * 1.4426950408...
        let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);

        // Constants for ln approximation
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

            // log2(x) = ln(x) * log2(e)
            let result = _mm256_mul_ps(ln_x, log2_e);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].log2();
        }
    }

    /// Vectorized log10 function for f64 (log base 10)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log10_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_log10_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized log10 for f64 using log(x) * log10(e)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log10_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // log10(x) = ln(x) * log10(e)
        let log10_e = _mm256_set1_pd(std::f64::consts::LOG10_E);

        // Constants for ln approximation (high precision for f64)
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let one = _mm256_set1_pd(1.0);
        let c1 = _mm256_set1_pd(-0.5);
        let c2 = _mm256_set1_pd(1.0 / 3.0);
        let c3 = _mm256_set1_pd(-0.25);
        let c4 = _mm256_set1_pd(0.2);
        let c5 = _mm256_set1_pd(-1.0 / 6.0);
        let c6 = _mm256_set1_pd(1.0 / 7.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Extract exponent (f64: bias=1023, mantissa=52 bits)
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

            // Extract mantissa and set exponent to 0
            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000);
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            // Polynomial approximation for ln(1+u) where u = mantissa - 1
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

    /// Vectorized log2 function for f64 (log base 2)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log2_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_log2_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized log2 for f64 using log(x) * log2(e)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log2_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // log2(x) = ln(x) * log2(e)
        let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);

        // Constants for ln approximation
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let one = _mm256_set1_pd(1.0);
        let c1 = _mm256_set1_pd(-0.5);
        let c2 = _mm256_set1_pd(1.0 / 3.0);
        let c3 = _mm256_set1_pd(-0.25);
        let c4 = _mm256_set1_pd(0.2);
        let c5 = _mm256_set1_pd(-1.0 / 6.0);
        let c6 = _mm256_set1_pd(1.0 / 7.0);

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

            // Polynomial approximation for ln(1+u)
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

            // log2(x) = ln(x) * log2(e)
            let result = _mm256_mul_pd(ln_x, log2_e);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].log2();
        }
    }

    /// Vectorized power function for f32: pow(x, n) = x^n
    /// Uses exp(n * log(x)) identity for SIMD computation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_pow_f32(input: &Array<f32>, n: f32) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_pow_f32(&data, n, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized power function for f32
    /// Computes x^n using exp(n * log(x)) with fused SIMD operations
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
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // === STEP 1: Compute log(x) ===
            // Extract exponent
            let x_int = _mm256_castps_si256(x);
            let exp = _mm256_sub_epi32(_mm256_srli_epi32(x_int, 23), _mm256_set1_epi32(127));
            let exp_f = _mm256_cvtepi32_ps(exp);

            // Extract mantissa
            let mantissa = _mm256_castsi256_ps(_mm256_or_si256(
                _mm256_and_si256(x_int, _mm256_set1_epi32(0x007FFFFF)),
                _mm256_set1_epi32(0x3F800000),
            ));

            // Polynomial approximation for log(1+u) where u = mantissa - 1
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

            // log(x) = exp * ln(2) + log(mantissa)
            let log_x = _mm256_fmadd_ps(exp_f, ln2, log_poly);

            // === STEP 2: Compute n * log(x) ===
            let n_log_x = _mm256_mul_ps(n_vec, log_x);

            // === STEP 3: Compute exp(n * log(x)) ===
            // Range reduction: y = k*ln(2) + r
            let k_float = _mm256_mul_ps(n_log_x, log2_e);
            let k = _mm256_cvtps_epi32(k_float);
            let k_f = _mm256_cvtepi32_ps(k);

            // r = n_log_x - k*ln(2)
            let r = _mm256_fmsub_ps(k_f, ln2_hi, n_log_x);
            let r = _mm256_fmsub_ps(k_f, ln2_lo, r);

            // Taylor series: exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4!
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

            // Scale by 2^k
            let scale = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_add_epi32(k, _mm256_set1_epi32(127)),
                23,
            ));
            let pow_result = _mm256_mul_ps(exp_poly, scale);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), pow_result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].powf(n);
        }
    }

    /// Vectorized power function for f64: pow(x, n) = x^n
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_pow_f64(input: &Array<f64>, n: f64) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_pow_f64(&data, n, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized power function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_pow_f64(input: &[f64], n: f64, output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Constants for log approximation (high precision for f64)
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let log_c1 = _mm256_set1_pd(-0.5);
        let log_c2 = _mm256_set1_pd(1.0 / 3.0);
        let log_c3 = _mm256_set1_pd(-0.25);
        let log_c4 = _mm256_set1_pd(0.2);
        let log_c5 = _mm256_set1_pd(-1.0 / 6.0);
        let one_pd = _mm256_set1_pd(1.0);

        // Constants for exp approximation
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
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // === STEP 1: Compute log(x) for f64 ===
            // Extract exponent (f64: bias=1023, mantissa=52 bits)
            let x_int = _mm256_castpd_si256(x);
            let exp_bits = _mm256_srli_epi64(x_int, 52);
            let bias = _mm256_set1_epi64x(1023);
            let exp_unbiased = _mm256_sub_epi64(exp_bits, bias);
            // Convert i64 to f64 (manual conversion since _mm256_cvtepi64_pd doesn't exist in AVX2)
            let mut exp_arr = [0i64; 4];
            _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_unbiased);
            let exp_f = _mm256_set_pd(
                exp_arr[3] as f64,
                exp_arr[2] as f64,
                exp_arr[1] as f64,
                exp_arr[0] as f64,
            );

            // Extract mantissa and set exponent to 0 (so value is in [1, 2))
            let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);
            let exp_one = _mm256_set1_epi64x(0x3FF0000000000000); // 1.0 in f64 representation
            let mantissa = _mm256_castsi256_pd(_mm256_or_si256(
                _mm256_and_si256(x_int, mantissa_mask),
                exp_one,
            ));

            // Polynomial approximation for log(1+u) where u = mantissa - 1
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

            // log(x) = exp * ln(2) + log(mantissa)
            let log_x = _mm256_fmadd_pd(exp_f, ln2, log_poly);

            // === STEP 2: Compute n * log(x) ===
            let n_log_x = _mm256_mul_pd(n_vec, log_x);

            // === STEP 3: Compute exp(n * log(x)) ===
            // Range reduction: y = k*ln(2) + r
            let k_float = _mm256_mul_pd(n_log_x, log2_e);
            let k = _mm256_cvtpd_epi32(k_float);
            let k_wide = _mm256_cvtepi32_epi64(k);
            let k_f = _mm256_cvtepi32_pd(k);

            // r = n_log_x - k*ln(2) using extended precision
            let r = _mm256_sub_pd(n_log_x, _mm256_mul_pd(k_f, ln2_hi));
            let r = _mm256_sub_pd(r, _mm256_mul_pd(k_f, ln2_lo));

            // Taylor series: exp(r) ≈ 1 + r + r²/2! + ... + r⁶/6!
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

            // Scale by 2^k: manipulate exponent bits
            let bias_64 = _mm256_set1_epi64x(1023);
            let k_biased = _mm256_add_epi64(k_wide, bias_64);
            let exp_scale = _mm256_slli_epi64(k_biased, 52);
            let scale = _mm256_castsi256_pd(exp_scale);

            let pow_result = _mm256_mul_pd(exp_poly, scale);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), pow_result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].powf(n);
        }
    }

    /// Vectorized sine function with CORDIC algorithm
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sin_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_sin_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized sine function
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

            // Range reduction: bring x to [-π, π]
            let k = _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_fmsub_ps(k, two_pi, x);

            // Determine quadrant and adjust
            let abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            let quadrant = _mm256_cmp_ps(abs_x, pi_2, _CMP_GT_OQ);

            // For |x| > π/2, use sin(π - x) = sin(x)
            let x_adj = _mm256_blendv_ps(x, _mm256_sub_ps(pi, abs_x), quadrant);
            let sign_adj = _mm256_blendv_ps(
                one,
                _mm256_set1_ps(-1.0),
                _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ),
            );

            // Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9!
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

    /// Vectorized cosine function with Taylor series approximation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cos_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_cos_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized cosine function
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cos_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for cos approximation (Taylor series)
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let pi = _mm256_set1_ps(std::f32::consts::PI);
        let pi_2 = _mm256_set1_ps(std::f32::consts::PI / 2.0);
        let one = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(-0.5);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c6 = _mm256_set1_ps(-1.0 / 720.0);
        let c8 = _mm256_set1_ps(1.0 / 40320.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let mut x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Range reduction: bring x to [-π, π]
            let k = _mm256_round_ps(_mm256_div_ps(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_fmsub_ps(k, two_pi, x);

            // Take absolute value and handle sign
            let abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));

            // Check if |x| > π/2, if so use cos(x) = -cos(π - |x|)
            let need_sign_flip = _mm256_cmp_ps(abs_x, pi_2, _CMP_GT_OQ);
            let x_adj = _mm256_blendv_ps(abs_x, _mm256_sub_ps(pi, abs_x), need_sign_flip);
            let sign = _mm256_blendv_ps(one, _mm256_set1_ps(-1.0), need_sign_flip);

            // Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8!
            let x2 = _mm256_mul_ps(x_adj, x_adj);
            let x4 = _mm256_mul_ps(x2, x2);
            let x6 = _mm256_mul_ps(x4, x2);
            let x8 = _mm256_mul_ps(x6, x2);

            let poly = _mm256_fmadd_ps(
                c8,
                x8,
                _mm256_fmadd_ps(
                    c6,
                    x6,
                    _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c2, x2, one)),
                ),
            );

            let result = _mm256_mul_ps(poly, sign);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cos();
        }
    }

    /// Vectorized sine function for f64 with high-precision Taylor series
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sin_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sin_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized sine function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_sin_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // High precision constants for sin approximation (Taylor series)
        let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
        let pi = _mm256_set1_pd(std::f64::consts::PI);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);

        // Taylor coefficients: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - x¹¹/11!
        let c3 = _mm256_set1_pd(-1.0 / 6.0);
        let c5 = _mm256_set1_pd(1.0 / 120.0);
        let c7 = _mm256_set1_pd(-1.0 / 5040.0);
        let c9 = _mm256_set1_pd(1.0 / 362880.0);
        let c11 = _mm256_set1_pd(-1.0 / 39916800.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let mut x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction: bring x to [-π, π]
            let k = _mm256_round_pd(_mm256_div_pd(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

            // Handle sign
            let sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
            let sign = _mm256_blendv_pd(one, neg_one, sign_mask);

            // Take absolute value
            let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            let abs_x = _mm256_and_pd(x, abs_mask);

            // For |x| > π/2, use sin(x) = sin(π - |x|)
            let need_adjust = _mm256_cmp_pd(abs_x, pi_2, _CMP_GT_OQ);
            let x_adj = _mm256_blendv_pd(abs_x, _mm256_sub_pd(pi, abs_x), need_adjust);

            // Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - x¹¹/11!
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

    /// Vectorized cosine function for f64 with high-precision Taylor series
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cos_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_cos_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized cosine function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cos_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // High precision constants for cos approximation (Taylor series)
        let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
        let pi = _mm256_set1_pd(std::f64::consts::PI);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);

        // Taylor coefficients: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8! - x¹⁰/10!
        let c2 = _mm256_set1_pd(-0.5);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c6 = _mm256_set1_pd(-1.0 / 720.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);
        let c10 = _mm256_set1_pd(-1.0 / 3628800.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let mut x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction: bring x to [-π, π]
            let k = _mm256_round_pd(_mm256_div_pd(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

            // Take absolute value (cos is even function)
            let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            let abs_x = _mm256_and_pd(x, abs_mask);

            // For |x| > π/2, use cos(x) = -cos(π - |x|)
            let need_sign_flip = _mm256_cmp_pd(abs_x, pi_2, _CMP_GT_OQ);
            let x_adj = _mm256_blendv_pd(abs_x, _mm256_sub_pd(pi, abs_x), need_sign_flip);
            let sign = _mm256_blendv_pd(one, neg_one, need_sign_flip);

            // Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8! - x¹⁰/10!
            let x2 = _mm256_mul_pd(x_adj, x_adj);
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

            let result = _mm256_mul_pd(poly, sign);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cos();
        }
    }

    /// Vectorized square root function using AVX2 intrinsics
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sqrt_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_sqrt_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized square root
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sqrt_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            // Prefetch next cache line
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

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].sqrt();
        }
    }

    /// Vectorized square root function for f64 using AVX2 intrinsics
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sqrt_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sqrt_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized square root for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sqrt_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
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

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].sqrt();
        }
    }

    /// Vectorized tangent function using sin/cos
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_tan_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_tan_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized tangent using sin/cos ratio
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tan_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Constants for sin/cos approximation
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let pi_2 = _mm256_set1_ps(std::f32::consts::PI / 2.0);
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

            // Clamp to avoid division issues near pi/2
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
            let sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0)); // Extract sign bit
            let tan_val = _mm256_div_ps(sin_val, cos_val);
            let result = _mm256_xor_ps(tan_val, sign);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tan();
        }
    }

    /// Vectorized tangent function for f64 using sin/cos
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_tan_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_tan_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized tangent for f64 using sin/cos ratio
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tan_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // High precision constants for sin/cos approximation
        let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
        let pi = _mm256_set1_pd(std::f64::consts::PI);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);

        // Taylor coefficients for sin: x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9!
        let s3 = _mm256_set1_pd(-1.0 / 6.0);
        let s5 = _mm256_set1_pd(1.0 / 120.0);
        let s7 = _mm256_set1_pd(-1.0 / 5040.0);
        let s9 = _mm256_set1_pd(1.0 / 362880.0);

        // Taylor coefficients for cos: 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8!
        let c2 = _mm256_set1_pd(-0.5);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        let c6 = _mm256_set1_pd(-1.0 / 720.0);
        let c8 = _mm256_set1_pd(1.0 / 40320.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let mut x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Range reduction: bring x to [-π, π]
            let k = _mm256_round_pd(_mm256_div_pd(x, two_pi), _MM_FROUND_TO_NEAREST_INT);
            x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

            // Handle sign for tangent (odd function)
            let sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
            let sign = _mm256_blendv_pd(one, neg_one, sign_mask);

            // Take absolute value
            let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            let abs_x = _mm256_and_pd(x, abs_mask);

            // For |x| > π/2, use tan(x) = -tan(π - |x|) for proper quadrant handling
            let need_adjust = _mm256_cmp_pd(abs_x, pi_2, _CMP_GT_OQ);
            let x_adj = _mm256_blendv_pd(abs_x, _mm256_sub_pd(pi, abs_x), need_adjust);
            let tan_sign = _mm256_blendv_pd(one, neg_one, need_adjust);

            // sin(x) approximation using Taylor series
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

            // cos(x) approximation using Taylor series
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

            // tan = sin/cos with proper sign handling
            let tan_val = _mm256_div_pd(sin_val, cos_val);
            let result = _mm256_mul_pd(_mm256_mul_pd(tan_val, tan_sign), sign);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tan();
        }
    }

    /// Vectorized hyperbolic sine function
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sinh_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_sinh_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized hyperbolic sine: sinh(x) = (e^x - e^(-x)) / 2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_sinh_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let half = _mm256_set1_ps(0.5);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Compute e^x and e^(-x) using our optimized exp
            // For sinh, we use the identity: sinh(x) = (e^x - e^(-x)) / 2
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

            // Simple exp approximation for SIMD (Taylor series)
            let exp_x = Self::simd_exp_ps(x);
            let exp_neg_x = Self::simd_exp_ps(neg_x);

            let result = _mm256_mul_ps(_mm256_sub_ps(exp_x, exp_neg_x), half);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].sinh();
        }
    }

    /// Vectorized hyperbolic cosine function
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cosh_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_cosh_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized hyperbolic cosine: cosh(x) = (e^x + e^(-x)) / 2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cosh_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let half = _mm256_set1_ps(0.5);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

            let exp_x = Self::simd_exp_ps(x);
            let exp_neg_x = Self::simd_exp_ps(neg_x);

            let result = _mm256_mul_ps(_mm256_add_ps(exp_x, exp_neg_x), half);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cosh();
        }
    }

    /// Vectorized hyperbolic tangent function
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_tanh_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::avx2_tanh_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized hyperbolic tangent: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tanh_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let two = _mm256_set1_ps(2.0);
        let one = _mm256_set1_ps(1.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
            let two_x = _mm256_mul_ps(x, two);
            let exp_2x = Self::simd_exp_ps(two_x);

            let numerator = _mm256_sub_ps(exp_2x, one);
            let denominator = _mm256_add_ps(exp_2x, one);
            let result = _mm256_div_ps(numerator, denominator);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].tanh();
        }
    }

    /// Vectorized hyperbolic sine function for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sinh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sinh_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized hyperbolic sine for f64: sinh(x) = (e^x - e^(-x)) / 2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_sinh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let half = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);

            // sinh(x) = (e^x - e^(-x)) / 2
            let exp_x = Self::simd_exp_pd(x);
            let exp_neg_x = Self::simd_exp_pd(neg_x);

            let result = _mm256_mul_pd(_mm256_sub_pd(exp_x, exp_neg_x), half);
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized hyperbolic cosine for f64: cosh(x) = (e^x + e^(-x)) / 2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cosh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let half = _mm256_set1_pd(0.5);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // Prefetch next cache line
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);

            // cosh(x) = (e^x + e^(-x)) / 2
            let exp_x = Self::simd_exp_pd(x);
            let exp_neg_x = Self::simd_exp_pd(neg_x);

            let result = _mm256_mul_pd(_mm256_add_pd(exp_x, exp_neg_x), half);
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized hyperbolic tangent for f64: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_tanh_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let two = _mm256_set1_pd(2.0);
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

            // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
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

    // =========================================
    // INVERSE TRIGONOMETRIC FUNCTIONS (f64)
    // =========================================

    /// Vectorized asin function for f64 (arc sine)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_asin_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_asin_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized arc sine for f64
    /// Uses the identity: asin(x) = atan2(x, sqrt(1-x²))
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

            // asin(x) = atan2(x, sqrt(1 - x²))
            let x_sq = _mm256_mul_pd(x, x);
            let one_minus_x_sq = _mm256_sub_pd(one, x_sq);
            let sqrt_term = _mm256_sqrt_pd(one_minus_x_sq);

            // Use SIMD atan2 approximation
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized arc cosine for f64
    /// Uses the identity: acos(x) = atan2(sqrt(1-x²), x)
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

            // acos(x) = atan2(sqrt(1 - x²), x)
            let x_sq = _mm256_mul_pd(x, x);
            let one_minus_x_sq = _mm256_sub_pd(one, x_sq);
            let sqrt_term = _mm256_sqrt_pd(one_minus_x_sq);

            // Use SIMD atan2 approximation
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

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized arc tangent for f64
    /// Uses polynomial approximation with range reduction
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

    /// SIMD atan approximation for f64 using polynomial
    /// Range reduction: |x| <= 1 using atan(x) = π/2 - atan(1/x) for |x| > 1
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn simd_atan_pd(x: __m256d) -> __m256d {
        // Constants
        let one = _mm256_set1_pd(1.0);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let sign_mask = _mm256_set1_pd(-0.0);

        // Get sign and absolute value
        let sign = _mm256_and_pd(x, sign_mask);
        let abs_x = _mm256_andnot_pd(sign_mask, x);

        // Range reduction: if |x| > 1, use atan(x) = sign(x) * (π/2 - atan(1/|x|))
        let gt_one = _mm256_cmp_pd(abs_x, one, _CMP_GT_OQ);
        let x_inv = _mm256_div_pd(one, abs_x);
        let x_reduced = _mm256_blendv_pd(abs_x, x_inv, gt_one);

        // Polynomial approximation for atan(x), |x| <= 1
        // Using minimax polynomial coefficients for best accuracy
        let x2 = _mm256_mul_pd(x_reduced, x_reduced);

        // Coefficients for atan polynomial (Horner's method)
        // atan(x) ≈ x - x³/3 + x⁵/5 - x⁷/7 + ...
        let c1 = _mm256_set1_pd(-0.333333333333331);
        let c2 = _mm256_set1_pd(0.199999999996591);
        let c3 = _mm256_set1_pd(-0.142857142725034);
        let c4 = _mm256_set1_pd(0.111111104054623);
        let c5 = _mm256_set1_pd(-0.090908995008245);
        let c6 = _mm256_set1_pd(0.076922533029620);
        let c7 = _mm256_set1_pd(-0.066657806901329);

        // Evaluate polynomial using Horner's method
        let mut p = c7;
        p = _mm256_fmadd_pd(p, x2, c6);
        p = _mm256_fmadd_pd(p, x2, c5);
        p = _mm256_fmadd_pd(p, x2, c4);
        p = _mm256_fmadd_pd(p, x2, c3);
        p = _mm256_fmadd_pd(p, x2, c2);
        p = _mm256_fmadd_pd(p, x2, c1);

        // atan(x) = x + x³ * p(x²)
        let x3 = _mm256_mul_pd(x_reduced, _mm256_mul_pd(x_reduced, x_reduced));
        let atan_reduced = _mm256_fmadd_pd(x3, p, x_reduced);

        // Undo range reduction: if |x| > 1, result = π/2 - atan(1/|x|)
        let atan_adjusted = _mm256_sub_pd(pi_2, atan_reduced);
        let atan_result = _mm256_blendv_pd(atan_reduced, atan_adjusted, gt_one);

        // Restore sign
        _mm256_or_pd(atan_result, sign)
    }

    /// SIMD atan2(y, x) approximation for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn simd_atan2_pd(y: __m256d, x: __m256d) -> __m256d {
        // Constants
        let zero = _mm256_setzero_pd();
        let pi = _mm256_set1_pd(std::f64::consts::PI);
        let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
        let sign_mask = _mm256_set1_pd(-0.0);

        // Compute atan(y/x)
        let ratio = _mm256_div_pd(y, x);
        let atan_ratio = Self::simd_atan_pd(ratio);

        // Get signs
        let x_negative = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
        let y_negative = _mm256_cmp_pd(y, zero, _CMP_LT_OQ);
        let x_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
        let y_zero = _mm256_cmp_pd(y, zero, _CMP_EQ_OQ);

        // Adjust for quadrant
        // If x < 0 and y >= 0: result = atan(y/x) + π
        // If x < 0 and y < 0: result = atan(y/x) - π
        // If x == 0 and y > 0: result = π/2
        // If x == 0 and y < 0: result = -π/2
        // If x == 0 and y == 0: result = 0

        let add_pi = _mm256_blendv_pd(pi, _mm256_sub_pd(zero, pi), y_negative);
        let adjusted = _mm256_add_pd(atan_ratio, add_pi);
        let result = _mm256_blendv_pd(atan_ratio, adjusted, x_negative);

        // Handle x == 0 cases
        let y_sign = _mm256_and_pd(y, sign_mask);
        let pi_2_signed = _mm256_or_pd(pi_2, y_sign);
        let x_zero_result = _mm256_blendv_pd(pi_2_signed, zero, y_zero);
        _mm256_blendv_pd(result, x_zero_result, x_zero)
    }

    // =========================================
    // INVERSE HYPERBOLIC FUNCTIONS (f64)
    // =========================================

    /// Vectorized asinh function for f64 (inverse hyperbolic sine)
    /// asinh(x) = ln(x + sqrt(x² + 1))
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_asinh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_asinh_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized inverse hyperbolic sine for f64
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

            // asinh(x) = ln(x + sqrt(x² + 1))
            let x_sq = _mm256_mul_pd(x, x);
            let x_sq_plus_1 = _mm256_add_pd(x_sq, one);
            let sqrt_term = _mm256_sqrt_pd(x_sq_plus_1);
            let sum = _mm256_add_pd(x, sqrt_term);

            // Use SIMD log approximation
            let result = Self::simd_log_pd(sum);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].asinh();
        }
    }

    /// Vectorized acosh function for f64 (inverse hyperbolic cosine)
    /// acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_acosh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_acosh_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized inverse hyperbolic cosine for f64
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

            // acosh(x) = ln(x + sqrt(x² - 1))
            let x_sq = _mm256_mul_pd(x, x);
            let x_sq_minus_1 = _mm256_sub_pd(x_sq, one);
            let sqrt_term = _mm256_sqrt_pd(x_sq_minus_1);
            let sum = _mm256_add_pd(x, sqrt_term);

            // Use SIMD log approximation
            let result = Self::simd_log_pd(sum);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].acosh();
        }
    }

    /// Vectorized atanh function for f64 (inverse hyperbolic tangent)
    /// atanh(x) = 0.5 * ln((1 + x) / (1 - x)) for |x| < 1
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_atanh_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_atanh_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized inverse hyperbolic tangent for f64
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

            // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
            let one_plus_x = _mm256_add_pd(one, x);
            let one_minus_x = _mm256_sub_pd(one, x);
            let ratio = _mm256_div_pd(one_plus_x, one_minus_x);

            // Use SIMD log approximation
            let log_ratio = Self::simd_log_pd(ratio);
            let result = _mm256_mul_pd(half, log_ratio);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].atanh();
        }
    }

    /// Helper SIMD log function for f64 (used by inverse hyperbolic functions)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn simd_log_pd(x: __m256d) -> __m256d {
        // Constants
        let one = _mm256_set1_pd(1.0);
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);

        // Extract exponent and mantissa
        // Get exponent bits (IEEE 754)
        let exp_mask = _mm256_set1_pd(f64::from_bits(0x7FF0000000000000u64));
        let mantissa_mask = _mm256_set1_pd(f64::from_bits(0x000FFFFFFFFFFFFFu64));
        let exp_bias = _mm256_set1_pd(1023.0);

        // Extract exponent
        let exp_bits = _mm256_and_pd(x, exp_mask);
        let exp_i64 = _mm256_castpd_si256(exp_bits);
        let exp_shifted = _mm256_srli_epi64(exp_i64, 52);

        // Convert exponent to f64 (workaround for AVX2's lack of _mm256_cvtepi64_pd)
        let mut exp_arr = [0i64; 4];
        _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_shifted);
        let exp_f = _mm256_set_pd(
            exp_arr[3] as f64,
            exp_arr[2] as f64,
            exp_arr[1] as f64,
            exp_arr[0] as f64,
        );
        let exp_unbiased = _mm256_sub_pd(exp_f, exp_bias);

        // Normalize mantissa to [1, 2)
        let mantissa_bits = _mm256_and_pd(x, mantissa_mask);
        let one_bits = _mm256_castsi256_pd(_mm256_set1_epi64x(0x3FF0000000000000i64));
        let mantissa = _mm256_or_pd(mantissa_bits, one_bits);

        // Reduce mantissa to [sqrt(2)/2, sqrt(2)]
        let sqrt2_inv = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let sqrt2 = _mm256_set1_pd(std::f64::consts::SQRT_2);
        let needs_adjust = _mm256_cmp_pd(mantissa, sqrt2, _CMP_GT_OQ);
        let m_adjusted =
            _mm256_blendv_pd(mantissa, _mm256_mul_pd(mantissa, sqrt2_inv), needs_adjust);
        let e_adjusted = _mm256_blendv_pd(
            exp_unbiased,
            _mm256_add_pd(exp_unbiased, _mm256_set1_pd(0.5)),
            needs_adjust,
        );

        // Polynomial approximation for log(1 + y) where y = m_adjusted - 1
        let y = _mm256_sub_pd(m_adjusted, one);

        // Coefficients for ln(1+y) approximation
        let c1 = _mm256_set1_pd(1.0);
        let c2 = _mm256_set1_pd(-0.5);
        let c3 = _mm256_set1_pd(0.333333333333333);
        let c4 = _mm256_set1_pd(-0.25);
        let c5 = _mm256_set1_pd(0.2);
        let c6 = _mm256_set1_pd(-0.166666666666667);

        // Evaluate polynomial using Horner's method
        let y2 = _mm256_mul_pd(y, y);
        let mut p = c6;
        p = _mm256_fmadd_pd(p, y, c5);
        p = _mm256_fmadd_pd(p, y, c4);
        p = _mm256_fmadd_pd(p, y, c3);
        p = _mm256_fmadd_pd(p, y, c2);
        p = _mm256_fmadd_pd(p, y, c1);
        let log_m = _mm256_mul_pd(y, p);

        // Final result: exp * ln(2) + log(m)
        _mm256_fmadd_pd(e_adjusted, ln2, log_m)
    }

    // =========================================
    // HIGH-PRECISION NUMERICAL FUNCTIONS (f64)
    // =========================================

    /// Vectorized log1p function for f64 (ln(1+x))
    /// Provides better precision than ln(1+x) for small x
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_log1p_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_log1p_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized log1p for f64
    /// Uses special handling for small values to maintain precision
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_log1p_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);
        let threshold = _mm256_set1_pd(0.5); // Use special algorithm for |x| < 0.5

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

            // For small |x| < 0.5, use Taylor series: log1p(x) ≈ x - x²/2 + x³/3 - x⁴/4 + ...
            let use_taylor = _mm256_cmp_pd(abs_x, threshold, _CMP_LT_OQ);

            // Taylor series coefficients
            let c2 = _mm256_set1_pd(-0.5);
            let c3 = _mm256_set1_pd(0.333333333333333);
            let c4 = _mm256_set1_pd(-0.25);
            let c5 = _mm256_set1_pd(0.2);
            let c6 = _mm256_set1_pd(-0.166666666666667);
            let c7 = _mm256_set1_pd(0.142857142857143);
            let c8 = _mm256_set1_pd(-0.125);

            let x2 = _mm256_mul_pd(x, x);
            let x3 = _mm256_mul_pd(x2, x);
            let x4 = _mm256_mul_pd(x2, x2);
            let x5 = _mm256_mul_pd(x4, x);
            let x6 = _mm256_mul_pd(x4, x2);
            let x7 = _mm256_mul_pd(x4, x3);
            let x8 = _mm256_mul_pd(x4, x4);

            // Compute Taylor series
            let mut taylor = x;
            taylor = _mm256_fmadd_pd(c2, x2, taylor);
            taylor = _mm256_fmadd_pd(c3, x3, taylor);
            taylor = _mm256_fmadd_pd(c4, x4, taylor);
            taylor = _mm256_fmadd_pd(c5, x5, taylor);
            taylor = _mm256_fmadd_pd(c6, x6, taylor);
            taylor = _mm256_fmadd_pd(c7, x7, taylor);
            taylor = _mm256_fmadd_pd(c8, x8, taylor);

            // For larger |x|, use log(1+x) directly
            let one_plus_x = _mm256_add_pd(one, x);
            let log_result = Self::simd_log_pd(one_plus_x);

            // Blend results based on threshold
            let result = _mm256_blendv_pd(log_result, taylor, use_taylor);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements using standard library for precision
        for i in simd_len..len {
            output[i] = input[i].ln_1p();
        }
    }

    /// Vectorized expm1 function for f64 (e^x - 1)
    /// Provides better precision than exp(x) - 1 for small x
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_expm1_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_expm1_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized expm1 for f64
    /// Uses special handling for small values to maintain precision
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_expm1_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);
        let threshold = _mm256_set1_pd(0.5); // Use Taylor series for |x| < 0.5

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

            // For small |x| < 0.5, use Taylor series: expm1(x) ≈ x + x²/2! + x³/3! + ...
            let use_taylor = _mm256_cmp_pd(abs_x, threshold, _CMP_LT_OQ);

            // Taylor series: e^x - 1 = x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + ...
            let c2 = _mm256_set1_pd(0.5); // 1/2!
            let c3 = _mm256_set1_pd(0.166666666666666667); // 1/3!
            let c4 = _mm256_set1_pd(0.0416666666666666667); // 1/4!
            let c5 = _mm256_set1_pd(0.00833333333333333333); // 1/5!
            let c6 = _mm256_set1_pd(0.00138888888888888889); // 1/6!
            let c7 = _mm256_set1_pd(0.000198412698412698413); // 1/7!
            let c8 = _mm256_set1_pd(2.48015873015873016e-05); // 1/8!

            let x2 = _mm256_mul_pd(x, x);
            let x3 = _mm256_mul_pd(x2, x);
            let x4 = _mm256_mul_pd(x2, x2);
            let x5 = _mm256_mul_pd(x4, x);
            let x6 = _mm256_mul_pd(x4, x2);
            let x7 = _mm256_mul_pd(x4, x3);
            let x8 = _mm256_mul_pd(x4, x4);

            // Compute Taylor series
            let mut taylor = x;
            taylor = _mm256_fmadd_pd(c2, x2, taylor);
            taylor = _mm256_fmadd_pd(c3, x3, taylor);
            taylor = _mm256_fmadd_pd(c4, x4, taylor);
            taylor = _mm256_fmadd_pd(c5, x5, taylor);
            taylor = _mm256_fmadd_pd(c6, x6, taylor);
            taylor = _mm256_fmadd_pd(c7, x7, taylor);
            taylor = _mm256_fmadd_pd(c8, x8, taylor);

            // For larger |x|, use exp(x) - 1 directly
            let exp_x = Self::simd_exp_pd(x);
            let exp_minus_one = _mm256_sub_pd(exp_x, one);

            // Blend results based on threshold
            let result = _mm256_blendv_pd(exp_minus_one, taylor, use_taylor);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements using standard library for precision
        for i in simd_len..len {
            output[i] = input[i].exp_m1();
        }
    }

    /// Vectorized cbrt function for f64 (cube root)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cbrt_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_cbrt_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized cube root for f64
    /// Uses Newton-Raphson iteration: x_{n+1} = (2*x_n + a/x_n²) / 3
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_cbrt_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let one_third = _mm256_set1_pd(1.0 / 3.0);
        let sign_mask = _mm256_set1_pd(-0.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Extract sign and work with absolute value
            let sign = _mm256_and_pd(x, sign_mask);
            let abs_x = _mm256_andnot_pd(sign_mask, x);

            // Initial approximation using bit manipulation
            // cbrt(x) ≈ x^(1/3), we use log/exp approximation for initial guess
            // x^(1/3) = exp(log(x)/3)
            let log_x = Self::simd_log_pd(abs_x);
            let log_x_div_3 = _mm256_mul_pd(log_x, one_third);
            let mut y = Self::simd_exp_pd(log_x_div_3);

            // Newton-Raphson iterations for refinement
            // x_{n+1} = (2*x_n + a/x_n²) / 3
            for _ in 0..3 {
                let y_sq = _mm256_mul_pd(y, y);
                let x_div_y_sq = _mm256_div_pd(abs_x, y_sq);
                let two_y = _mm256_mul_pd(two, y);
                let sum = _mm256_add_pd(two_y, x_div_y_sq);
                y = _mm256_mul_pd(sum, one_third);
            }

            // Handle zero case
            let zero = _mm256_setzero_pd();
            let is_zero = _mm256_cmp_pd(abs_x, zero, _CMP_EQ_OQ);
            y = _mm256_blendv_pd(y, zero, is_zero);

            // Restore sign
            let result = _mm256_or_pd(y, sign);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].cbrt();
        }
    }

    // =========================================
    // ROUNDING FUNCTIONS (f64) - Hardware SIMD
    // =========================================

    /// Vectorized floor function for f64
    /// Uses hardware AVX instruction for maximum performance
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_floor_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_floor_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized floor for f64 using hardware instruction
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_floor_f64(input: &[f64], output: &mut [f64]) {
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
            let result = _mm256_floor_pd(x);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].floor();
        }
    }

    /// Vectorized ceil function for f64
    /// Uses hardware AVX instruction for maximum performance
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_ceil_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_ceil_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized ceil for f64 using hardware instruction
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_ceil_f64(input: &[f64], output: &mut [f64]) {
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
            let result = _mm256_ceil_pd(x);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].ceil();
        }
    }

    /// Vectorized round function for f64 (round to nearest even)
    /// Uses hardware AVX instruction for maximum performance
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_round_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_round_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized round for f64 using hardware instruction
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_round_f64(input: &[f64], output: &mut [f64]) {
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
            // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0x08
            let result = _mm256_round_pd(x, 0x08);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].round();
        }
    }

    /// Vectorized trunc function for f64 (round toward zero)
    /// Uses hardware AVX instruction for maximum performance
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_trunc_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_trunc_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized trunc for f64 using hardware instruction
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_trunc_f64(input: &[f64], output: &mut [f64]) {
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
            // _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC = 0x0B
            let result = _mm256_round_pd(x, 0x0B);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].trunc();
        }
    }

    // =========================================
    // CONVERSION FUNCTIONS (f64) - Hardware SIMD
    // =========================================

    /// Vectorized degrees function for f64 (radians to degrees)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_degrees_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_degrees_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized radians to degrees conversion
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_degrees_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // 180 / π
        let rad_to_deg = _mm256_set1_pd(180.0 / std::f64::consts::PI);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let result = _mm256_mul_pd(x, rad_to_deg);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        let factor = 180.0 / std::f64::consts::PI;
        for i in simd_len..len {
            output[i] = input[i] * factor;
        }
    }

    /// Vectorized radians function for f64 (degrees to radians)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_radians_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_radians_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized degrees to radians conversion
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_radians_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // π / 180
        let deg_to_rad = _mm256_set1_pd(std::f64::consts::PI / 180.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let result = _mm256_mul_pd(x, deg_to_rad);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        let factor = std::f64::consts::PI / 180.0;
        for i in simd_len..len {
            output[i] = input[i] * factor;
        }
    }

    // =========================================
    // UTILITY FUNCTIONS (f64) - Hardware SIMD
    // =========================================

    /// Vectorized abs function for f64
    /// Uses bit manipulation to clear sign bit - extremely fast
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_abs_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_abs_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized absolute value for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_abs_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Mask to clear sign bit (all bits except sign bit)
        let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let result = _mm256_and_pd(x, abs_mask);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].abs();
        }
    }

    /// Vectorized sign function for f64
    /// Returns -1.0, 0.0, or 1.0 based on sign
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sign_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_sign_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized sign function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sign_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));

            // Create masks for positive and negative values
            let pos_mask = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);
            let neg_mask = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);

            // Start with zero, blend in 1 for positive, -1 for negative
            let result = _mm256_blendv_pd(_mm256_blendv_pd(zero, neg_one, neg_mask), one, pos_mask);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            let x = input[i];
            output[i] = if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            };
        }
    }

    /// Vectorized clip function for f64
    /// Clips values to [min, max] range using hardware min/max
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_clip_f64(input: &Array<f64>, min_val: f64, max_val: f64) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_clip_f64(&data, &mut result, min_val, max_val);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized clip function for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_clip_f64(input: &[f64], output: &mut [f64], min_val: f64, max_val: f64) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let min_vec = _mm256_set1_pd(min_val);
        let max_vec = _mm256_set1_pd(max_val);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            // clip: max(min_val, min(x, max_val))
            let clipped_upper = _mm256_min_pd(x, max_vec);
            let result = _mm256_max_pd(clipped_upper, min_vec);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].max(min_val).min(max_val);
        }
    }

    /// Vectorized hypot function for f64 (hypotenuse: sqrt(x² + y²))
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_hypot_f64(x: &Array<f64>, y: &Array<f64>) -> Array<f64> {
        let x_data = x.to_vec();
        let y_data = y.to_vec();
        let len = x_data.len().min(y_data.len());
        let mut result = vec![0.0f64; len];

        unsafe {
            Self::avx2_hypot_f64(&x_data[..len], &y_data[..len], &mut result);
        }

        Array::from_vec(result).reshape(&x.shape())
    }

    /// AVX2 optimized hypot for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_hypot_f64(x: &[f64], y: &[f64], output: &mut [f64]) {
        let len = x.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    x.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    y.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_pd(y.as_ptr().add(i));

            // sqrt(x² + y²) using FMA
            let x_sq = _mm256_mul_pd(x_vec, x_vec);
            let sum_sq = _mm256_fmadd_pd(y_vec, y_vec, x_sq);
            let result = _mm256_sqrt_pd(sum_sq);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
        }
    }

    // ========================================
    // SIMD Reduction Operations for f64
    // ========================================

    /// Vectorized sum reduction for f64
    /// Uses parallel accumulation with horizontal sum at the end
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sum_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        unsafe { Self::avx2_sum_f64(&data) }
    }

    /// AVX2 optimized sum reduction for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sum_f64(input: &[f64]) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Use 4 accumulators to hide latency
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        // Unrolled loop with 4 accumulators
        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let v0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let v2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let v3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            sum0 = _mm256_add_pd(sum0, v0);
            sum1 = _mm256_add_pd(sum1, v1);
            sum2 = _mm256_add_pd(sum2, v2);
            sum3 = _mm256_add_pd(sum3, v3);
        }

        // Process remaining SIMD chunks
        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            sum0 = _mm256_add_pd(sum0, v);
        }

        // Combine accumulators
        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);

        // Horizontal sum of 4 doubles
        let sum_low = _mm256_extractf128_pd(sum0, 0);
        let sum_high = _mm256_extractf128_pd(sum0, 1);
        let sum_128 = _mm_add_pd(sum_low, sum_high);
        let sum_final = _mm_hadd_pd(sum_128, sum_128);

        let mut result = _mm_cvtsd_f64(sum_final);

        // Handle tail elements
        for i in simd_len..len {
            result += input[i];
        }

        result
    }

    /// Vectorized product reduction for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_prod_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        unsafe { Self::avx2_prod_f64(&data) }
    }

    /// AVX2 optimized product reduction for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_prod_f64(input: &[f64]) -> f64 {
        let len = input.len();
        if len == 0 {
            return 1.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Use 4 accumulators
        let mut prod0 = _mm256_set1_pd(1.0);
        let mut prod1 = _mm256_set1_pd(1.0);
        let mut prod2 = _mm256_set1_pd(1.0);
        let mut prod3 = _mm256_set1_pd(1.0);

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        // Unrolled loop
        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let v0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let v2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let v3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            prod0 = _mm256_mul_pd(prod0, v0);
            prod1 = _mm256_mul_pd(prod1, v1);
            prod2 = _mm256_mul_pd(prod2, v2);
            prod3 = _mm256_mul_pd(prod3, v3);
        }

        // Process remaining SIMD chunks
        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            prod0 = _mm256_mul_pd(prod0, v);
        }

        // Combine accumulators
        prod0 = _mm256_mul_pd(prod0, prod1);
        prod2 = _mm256_mul_pd(prod2, prod3);
        prod0 = _mm256_mul_pd(prod0, prod2);

        // Horizontal product of 4 doubles
        let prod_low = _mm256_extractf128_pd(prod0, 0);
        let prod_high = _mm256_extractf128_pd(prod0, 1);
        let prod_128 = _mm_mul_pd(prod_low, prod_high);
        let prod_shuffle = _mm_shuffle_pd(prod_128, prod_128, 1);
        let prod_final = _mm_mul_pd(prod_128, prod_shuffle);

        let mut result = _mm_cvtsd_f64(prod_final);

        // Handle tail elements
        for i in simd_len..len {
            result *= input[i];
        }

        result
    }

    /// Vectorized min reduction for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_min_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return f64::INFINITY;
        }
        unsafe { Self::avx2_min_f64(&data) }
    }

    /// AVX2 optimized min reduction for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_min_f64(input: &[f64]) -> f64 {
        let len = input.len();
        if len == 0 {
            return f64::INFINITY;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Initialize with first element
        let mut min0 = _mm256_set1_pd(input[0]);
        let mut min1 = min0;
        let mut min2 = min0;
        let mut min3 = min0;

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        // Unrolled loop
        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let v0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let v2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let v3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            min0 = _mm256_min_pd(min0, v0);
            min1 = _mm256_min_pd(min1, v1);
            min2 = _mm256_min_pd(min2, v2);
            min3 = _mm256_min_pd(min3, v3);
        }

        // Process remaining SIMD chunks
        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            min0 = _mm256_min_pd(min0, v);
        }

        // Combine accumulators
        min0 = _mm256_min_pd(min0, min1);
        min2 = _mm256_min_pd(min2, min3);
        min0 = _mm256_min_pd(min0, min2);

        // Horizontal min of 4 doubles
        let min_low = _mm256_extractf128_pd(min0, 0);
        let min_high = _mm256_extractf128_pd(min0, 1);
        let min_128 = _mm_min_pd(min_low, min_high);
        let min_shuffle = _mm_shuffle_pd(min_128, min_128, 1);
        let min_final = _mm_min_pd(min_128, min_shuffle);

        let mut result = _mm_cvtsd_f64(min_final);

        // Handle tail elements
        for i in simd_len..len {
            if input[i] < result {
                result = input[i];
            }
        }

        result
    }

    /// Vectorized max reduction for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_max_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return f64::NEG_INFINITY;
        }
        unsafe { Self::avx2_max_f64(&data) }
    }

    /// AVX2 optimized max reduction for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_max_f64(input: &[f64]) -> f64 {
        let len = input.len();
        if len == 0 {
            return f64::NEG_INFINITY;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Initialize with first element
        let mut max0 = _mm256_set1_pd(input[0]);
        let mut max1 = max0;
        let mut max2 = max0;
        let mut max3 = max0;

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        // Unrolled loop
        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let v0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let v2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let v3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            max0 = _mm256_max_pd(max0, v0);
            max1 = _mm256_max_pd(max1, v1);
            max2 = _mm256_max_pd(max2, v2);
            max3 = _mm256_max_pd(max3, v3);
        }

        // Process remaining SIMD chunks
        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            max0 = _mm256_max_pd(max0, v);
        }

        // Combine accumulators
        max0 = _mm256_max_pd(max0, max1);
        max2 = _mm256_max_pd(max2, max3);
        max0 = _mm256_max_pd(max0, max2);

        // Horizontal max of 4 doubles
        let max_low = _mm256_extractf128_pd(max0, 0);
        let max_high = _mm256_extractf128_pd(max0, 1);
        let max_128 = _mm_max_pd(max_low, max_high);
        let max_shuffle = _mm_shuffle_pd(max_128, max_128, 1);
        let max_final = _mm_max_pd(max_128, max_shuffle);

        let mut result = _mm_cvtsd_f64(max_final);

        // Handle tail elements
        for i in simd_len..len {
            if input[i] > result {
                result = input[i];
            }
        }

        result
    }

    // ========================================
    // Additional SIMD Element-wise Operations
    // ========================================

    /// Vectorized square function for f64 (x * x)
    /// Simple but highly efficient vectorized operation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_square_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        unsafe {
            Self::avx2_square_f64(&data, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized square for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_square_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Unroll loop for better pipelining
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let x2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let x3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_mul_pd(x0, x0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_mul_pd(x1, x1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_mul_pd(x2, x2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_mul_pd(x3, x3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_mul_pd(x, x));
        }

        for i in simd_len..len {
            output[i] = input[i] * input[i];
        }
    }

    /// Vectorized reciprocal function for f64 (1.0 / x)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_reciprocal_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        unsafe {
            Self::avx2_reciprocal_f64(&data, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized reciprocal for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_reciprocal_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let one = _mm256_set1_pd(1.0);

        // Unroll loop
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let x2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let x3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_div_pd(one, x0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_div_pd(one, x1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_div_pd(one, x2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_div_pd(one, x3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_div_pd(one, x));
        }

        for i in simd_len..len {
            output[i] = 1.0 / input[i];
        }
    }

    /// Vectorized negative function for f64 (-x)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_negative_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        unsafe {
            Self::avx2_negative_f64(&data, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized negative for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_negative_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        // XOR with sign bit to negate
        let sign_mask = _mm256_set1_pd(-0.0);

        // Unroll loop
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let x2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let x3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_xor_pd(x0, sign_mask));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_xor_pd(x1, sign_mask),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_xor_pd(x2, sign_mask),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_xor_pd(x3, sign_mask),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_xor_pd(x, sign_mask));
        }

        for i in simd_len..len {
            output[i] = -input[i];
        }
    }

    /// Vectorized copysign function for f64 - copies sign of y to magnitude of x
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_copysign_f64(x: &Array<f64>, y: &Array<f64>) -> Array<f64> {
        let x_data = x.to_vec();
        let y_data = y.to_vec();
        let len = x_data.len().min(y_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_copysign_f64(&x_data[..len], &y_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&x.shape())
    }

    /// AVX2 optimized copysign for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_copysign_f64(x: &[f64], y: &[f64], output: &mut [f64]) {
        let len = x.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        // Mask to extract sign bit
        let sign_mask = _mm256_set1_pd(-0.0);
        // Mask to extract magnitude
        let mag_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_pd(y.as_ptr().add(i));

            // Extract magnitude of x and sign of y
            let magnitude = _mm256_and_pd(x_vec, mag_mask);
            let sign = _mm256_and_pd(y_vec, sign_mask);

            // Combine them
            let result = _mm256_or_pd(magnitude, sign);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = x[i].abs().copysign(y[i]);
        }
    }

    // ========================================
    // SIMD Binary Array Operations for f64
    // ========================================

    /// Vectorized element-wise addition of two f64 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_add_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_add_arrays_f64(&a_data[..len], &b_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized element-wise addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_arrays_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_add_pd(a0, b0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_add_pd(a1, b1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_add_pd(a2, b2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_add_pd(a3, b3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_add_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = a[i] + b[i];
        }
    }

    /// Vectorized element-wise subtraction of two f64 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sub_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_sub_arrays_f64(&a_data[..len], &b_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized element-wise subtraction
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sub_arrays_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_sub_pd(a0, b0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_sub_pd(a1, b1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_sub_pd(a2, b2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_sub_pd(a3, b3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_sub_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = a[i] - b[i];
        }
    }

    /// Vectorized element-wise multiplication of two f64 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_mul_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_mul_arrays_f64(&a_data[..len], &b_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized element-wise multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_mul_arrays_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_mul_pd(a0, b0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_mul_pd(a1, b1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_mul_pd(a2, b2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_mul_pd(a3, b3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_mul_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = a[i] * b[i];
        }
    }

    /// Vectorized element-wise division of two f64 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_div_arrays_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_div_arrays_f64(&a_data[..len], &b_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized element-wise division
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_div_arrays_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_div_pd(a0, b0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_div_pd(a1, b1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_div_pd(a2, b2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_div_pd(a3, b3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_div_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = a[i] / b[i];
        }
    }

    /// Vectorized FMA (fused multiply-add): a * b + c
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_fma_f64(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let c_data = c.to_vec();
        let len = a_data.len().min(b_data.len()).min(c_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_fma_f64(&a_data[..len], &b_data[..len], &c_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_fma_f64(a: &[f64], b: &[f64], c: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            let cv = _mm256_loadu_pd(c.as_ptr().add(i));
            // a * b + c
            let result = _mm256_fmadd_pd(av, bv, cv);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = a[i] * b[i] + c[i];
        }
    }

    /// Vectorized scalar-array addition
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_add_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let data = a.to_vec();
        let mut result = vec![0.0f64; data.len()];
        unsafe {
            Self::avx2_add_scalar_f64(&data, scalar, &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized scalar-array addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_scalar_f64(a: &[f64], scalar: f64, output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let scalar_vec = _mm256_set1_pd(scalar);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_add_pd(a0, scalar_vec));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_add_pd(a1, scalar_vec),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_add_pd(a2, scalar_vec),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_add_pd(a3, scalar_vec),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_add_pd(av, scalar_vec));
        }

        for i in simd_len..len {
            output[i] = a[i] + scalar;
        }
    }

    /// Vectorized scalar-array multiplication
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_mul_scalar_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let data = a.to_vec();
        let mut result = vec![0.0f64; data.len()];
        unsafe {
            Self::avx2_mul_scalar_f64(&data, scalar, &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized scalar-array multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_mul_scalar_f64(a: &[f64], scalar: f64, output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let scalar_vec = _mm256_set1_pd(scalar);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_mul_pd(a0, scalar_vec));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_mul_pd(a1, scalar_vec),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_mul_pd(a2, scalar_vec),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_mul_pd(a3, scalar_vec),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_mul_pd(av, scalar_vec));
        }

        for i in simd_len..len {
            output[i] = a[i] * scalar;
        }
    }

    // ========================================
    // SIMD Vector Operations
    // ========================================

    /// Vectorized dot product for f64 arrays
    /// Uses FMA for better accuracy and performance
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_dot_f64(a: &Array<f64>, b: &Array<f64>) -> f64 {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        unsafe { Self::avx2_dot_f64(&a_data[..len], &b_data[..len]) }
    }

    /// AVX2 optimized dot product for f64 using FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_dot_f64(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Use 4 accumulators for better pipelining
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        // Main unrolled loop with FMA
        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            // FMA: sum += a * b
            sum0 = _mm256_fmadd_pd(a0, b0, sum0);
            sum1 = _mm256_fmadd_pd(a1, b1, sum1);
            sum2 = _mm256_fmadd_pd(a2, b2, sum2);
            sum3 = _mm256_fmadd_pd(a3, b3, sum3);
        }

        // Process remaining SIMD chunks
        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_pd(av, bv, sum0);
        }

        // Combine accumulators
        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);

        // Horizontal sum
        let sum_low = _mm256_extractf128_pd(sum0, 0);
        let sum_high = _mm256_extractf128_pd(sum0, 1);
        let sum_128 = _mm_add_pd(sum_low, sum_high);
        let sum_final = _mm_hadd_pd(sum_128, sum_128);

        let mut result = _mm_cvtsd_f64(sum_final);

        // Handle tail elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }

        result
    }

    /// Vectorized L2 norm (Euclidean norm) for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_l2_f64(a: &Array<f64>) -> f64 {
        let data = a.to_vec();
        unsafe { Self::avx2_norm_l2_f64(&data) }
    }

    /// AVX2 optimized L2 norm using FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_norm_l2_f64(a: &[f64]) -> f64 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        // Use 4 accumulators
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            // FMA: sum += a * a
            sum0 = _mm256_fmadd_pd(a0, a0, sum0);
            sum1 = _mm256_fmadd_pd(a1, a1, sum1);
            sum2 = _mm256_fmadd_pd(a2, a2, sum2);
            sum3 = _mm256_fmadd_pd(a3, a3, sum3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            sum0 = _mm256_fmadd_pd(av, av, sum0);
        }

        // Combine accumulators
        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);

        // Horizontal sum
        let sum_low = _mm256_extractf128_pd(sum0, 0);
        let sum_high = _mm256_extractf128_pd(sum0, 1);
        let sum_128 = _mm_add_pd(sum_low, sum_high);
        let sum_final = _mm_hadd_pd(sum_128, sum_128);

        let mut sum_sq = _mm_cvtsd_f64(sum_final);

        // Handle tail
        for i in simd_len..len {
            sum_sq += a[i] * a[i];
        }

        sum_sq.sqrt()
    }

    /// Vectorized L1 norm (Manhattan norm) for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_l1_f64(a: &Array<f64>) -> f64 {
        let data = a.to_vec();
        unsafe { Self::avx2_norm_l1_f64(&data) }
    }

    /// AVX2 optimized L1 norm
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_norm_l1_f64(a: &[f64]) -> f64 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);
        let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));

        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            sum0 = _mm256_add_pd(sum0, _mm256_and_pd(a0, abs_mask));
            sum1 = _mm256_add_pd(sum1, _mm256_and_pd(a1, abs_mask));
            sum2 = _mm256_add_pd(sum2, _mm256_and_pd(a2, abs_mask));
            sum3 = _mm256_add_pd(sum3, _mm256_and_pd(a3, abs_mask));
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            sum0 = _mm256_add_pd(sum0, _mm256_and_pd(av, abs_mask));
        }

        // Combine
        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);

        // Horizontal sum
        let sum_low = _mm256_extractf128_pd(sum0, 0);
        let sum_high = _mm256_extractf128_pd(sum0, 1);
        let sum_128 = _mm_add_pd(sum_low, sum_high);
        let sum_final = _mm_hadd_pd(sum_128, sum_128);

        let mut result = _mm_cvtsd_f64(sum_final);

        for i in simd_len..len {
            result += a[i].abs();
        }

        result
    }

    // ========================================
    // SIMD Comparison Operations for f64
    // ========================================

    /// Vectorized element-wise maximum of two arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_maximum_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_maximum_f64(&a_data[..len], &b_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized element-wise maximum
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_maximum_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_max_pd(a0, b0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_max_pd(a1, b1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_max_pd(a2, b2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_max_pd(a3, b3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_max_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = if a[i] > b[i] { a[i] } else { b[i] };
        }
    }

    /// Vectorized element-wise minimum of two arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_minimum_f64(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_minimum_f64(&a_data[..len], &b_data[..len], &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized element-wise minimum
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_minimum_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
        let len = a.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));
            let a2 = _mm256_loadu_pd(a.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let a3 = _mm256_loadu_pd(a.as_ptr().add(i + 3 * AVX2_F64_LANES));

            let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
            let b1 = _mm256_loadu_pd(b.as_ptr().add(i + AVX2_F64_LANES));
            let b2 = _mm256_loadu_pd(b.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let b3 = _mm256_loadu_pd(b.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_min_pd(a0, b0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_min_pd(a1, b1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_min_pd(a2, b2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_min_pd(a3, b3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_min_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = if a[i] < b[i] { a[i] } else { b[i] };
        }
    }

    // ========================================
    // SIMD Statistical Operations for f64
    // ========================================

    /// Vectorized variance computation for f64
    /// Uses two-pass algorithm: first compute mean, then sum of squared differences
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_variance_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return 0.0;
        }
        unsafe { Self::avx2_variance_f64(&data) }
    }

    /// AVX2 optimized variance using FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_variance_f64(input: &[f64]) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        // First pass: compute mean using SIMD sum
        let sum = Self::avx2_sum_f64(input);
        let mean = sum / len as f64;

        // Second pass: compute sum of squared differences
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let mean_vec = _mm256_set1_pd(mean);

        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let x2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let x3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            // Compute (x - mean)
            let d0 = _mm256_sub_pd(x0, mean_vec);
            let d1 = _mm256_sub_pd(x1, mean_vec);
            let d2 = _mm256_sub_pd(x2, mean_vec);
            let d3 = _mm256_sub_pd(x3, mean_vec);

            // Accumulate (x - mean)² using FMA
            sum0 = _mm256_fmadd_pd(d0, d0, sum0);
            sum1 = _mm256_fmadd_pd(d1, d1, sum1);
            sum2 = _mm256_fmadd_pd(d2, d2, sum2);
            sum3 = _mm256_fmadd_pd(d3, d3, sum3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let d = _mm256_sub_pd(x, mean_vec);
            sum0 = _mm256_fmadd_pd(d, d, sum0);
        }

        // Combine accumulators
        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);

        // Horizontal sum
        let sum_low = _mm256_extractf128_pd(sum0, 0);
        let sum_high = _mm256_extractf128_pd(sum0, 1);
        let sum_128 = _mm_add_pd(sum_low, sum_high);
        let sum_final = _mm_hadd_pd(sum_128, sum_128);

        let mut sum_sq_diff = _mm_cvtsd_f64(sum_final);

        // Handle tail
        for i in simd_len..len {
            let d = input[i] - mean;
            sum_sq_diff += d * d;
        }

        sum_sq_diff / len as f64
    }

    /// Vectorized standard deviation for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_std_f64(input: &Array<f64>) -> f64 {
        Self::vectorized_variance_f64(input).sqrt()
    }

    /// Vectorized mean computation for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_mean_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return 0.0;
        }
        unsafe { Self::avx2_sum_f64(&data) / data.len() as f64 }
    }

    /// Helper function for SIMD exp approximation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn simd_exp_ps(x: __m256) -> __m256 {
        // Constants for exp approximation
        let log2_e = _mm256_set1_ps(1.4426950408889634);
        let ln2_hi = _mm256_set1_ps(0.6931471805599453);
        let ln2_lo = _mm256_set1_ps(2.3283064365386963e-10);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(0.5);
        let c4 = _mm256_set1_ps(0.16666666666666666);
        let c5 = _mm256_set1_ps(0.041666666666666664);

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
        _mm256_mul_ps(poly, result)
    }

    /// Helper function for SIMD exp approximation for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn simd_exp_pd(x: __m256d) -> __m256d {
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

        // Range reduction: x = n*ln(2) + r
        let n_float = _mm256_mul_pd(x, log2_e);
        let n = _mm256_cvtpd_epi32(n_float);
        let n_wide = _mm256_cvtepi32_epi64(n);
        let n_f = _mm256_cvtepi32_pd(n);

        // r = x - n*ln(2) using extended precision
        let r = _mm256_sub_pd(x, _mm256_mul_pd(n_f, ln2_hi));
        let r = _mm256_sub_pd(r, _mm256_mul_pd(n_f, ln2_lo));

        // Taylor series: exp(r) ≈ 1 + r + r²/2! + ... + r⁶/6!
        let r2 = _mm256_mul_pd(r, r);
        let r3 = _mm256_mul_pd(r2, r);
        let r4 = _mm256_mul_pd(r2, r2);

        let poly_high = _mm256_fmadd_pd(c6, r2, _mm256_fmadd_pd(c5, r, c4));
        let poly_low = _mm256_fmadd_pd(c3, r3, _mm256_fmadd_pd(c2, r2, _mm256_fmadd_pd(c1, r, c0)));
        let poly = _mm256_fmadd_pd(poly_high, r4, poly_low);

        // Scale by 2^n: manipulate exponent bits
        let bias = _mm256_set1_epi64x(1023);
        let n_biased = _mm256_add_epi64(n_wide, bias);
        let exp_scale = _mm256_slli_epi64(n_biased, 52);
        let scale = _mm256_castsi256_pd(exp_scale);

        _mm256_mul_pd(poly, scale)
    }

    /// Advanced reduction operations with Kahan summation for numerical stability
    #[cfg(target_arch = "x86_64")]
    pub fn kahan_sum_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        unsafe { Self::avx2_kahan_sum_f32(&data) }
    }

    /// AVX2 Kahan summation for improved numerical stability
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_kahan_sum_f32(input: &[f32]) -> f32 {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum = _mm256_setzero_ps();
        let mut c = _mm256_setzero_ps(); // Compensation for lost low-order bits

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let y = _mm256_sub_ps(x, c); // Compensated input
            let t = _mm256_add_ps(sum, y); // New sum
            c = _mm256_sub_ps(_mm256_sub_ps(t, sum), y); // Update compensation
            sum = t;
        }

        // Horizontal sum of SIMD register
        let hi128 = _mm256_extractf128_ps(sum, 1);
        let lo128 = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0x1B);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        let final_sum = _mm_add_ps(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        // Handle remaining elements with scalar Kahan summation
        let mut scalar_c = 0.0f32;
        for &item in &input[simd_len..] {
            let y = item - scalar_c;
            let t = result + y;
            scalar_c = (t - result) - y;
            result = t;
        }

        result
    }

    /// Vectorized complex number operations
    #[cfg(target_arch = "x86_64")]
    pub fn complex_multiply_f32(
        a_real: &Array<f32>,
        a_imag: &Array<f32>,
        b_real: &Array<f32>,
        b_imag: &Array<f32>,
    ) -> Result<(Array<f32>, Array<f32>)> {
        if a_real.shape() != a_imag.shape()
            || b_real.shape() != b_imag.shape()
            || a_real.shape() != b_real.shape()
        {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a_real.shape(),
                actual: b_real.shape(),
            });
        }

        let len = a_real.len();
        let a_r = a_real.to_vec();
        let a_i = a_imag.to_vec();
        let b_r = b_real.to_vec();
        let b_i = b_imag.to_vec();

        let mut c_r = vec![0.0f32; len];
        let mut c_i = vec![0.0f32; len];

        unsafe {
            Self::avx2_complex_mul_f32(&a_r, &a_i, &b_r, &b_i, &mut c_r, &mut c_i);
        }

        Ok((
            Array::from_vec(c_r).reshape(&a_real.shape()),
            Array::from_vec(c_i).reshape(&a_real.shape()),
        ))
    }

    /// AVX2 complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_complex_mul_f32(
        a_real: &[f32],
        a_imag: &[f32],
        b_real: &[f32],
        b_imag: &[f32],
        c_real: &mut [f32],
        c_imag: &mut [f32],
    ) {
        let len = a_real.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let ar = _mm256_loadu_ps(a_real.as_ptr().add(i));
            let ai = _mm256_loadu_ps(a_imag.as_ptr().add(i));
            let br = _mm256_loadu_ps(b_real.as_ptr().add(i));
            let bi = _mm256_loadu_ps(b_imag.as_ptr().add(i));

            // Real part: ar*br - ai*bi
            let cr = _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi));
            // Imaginary part: ar*bi + ai*br
            let ci = _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));

            _mm256_storeu_ps(c_real.as_mut_ptr().add(i), cr);
            _mm256_storeu_ps(c_imag.as_mut_ptr().add(i), ci);
        }

        // Handle remaining elements
        for i in simd_len..len {
            c_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
            c_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
        }
    }

    /// SIMD-optimized dot product for f32 vectors
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_dot_f32(x: &Array<f32>, y: &Array<f32>) -> Result<f32> {
        if x.shape() != y.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x.shape(),
                actual: y.shape(),
            });
        }

        let x_data = x.to_vec();
        let y_data = y.to_vec();

        if x_data.len() != y_data.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Vectors must have the same length".to_string(),
            ));
        }

        unsafe { Ok(Self::avx2_dot_f32(&x_data, &y_data)) }
    }

    /// AVX2 optimized dot product implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_dot_f32(x: &[f32], y: &[f32]) -> f32 {
        let len = x.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum_vec = _mm256_setzero_ps();

        // Process 8 elements at a time with AVX2
        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            // Prefetch next cache line for better memory bandwidth
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    x.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    y.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_ps(y.as_ptr().add(i));

            // Use FMA for x * y + sum for better performance and accuracy
            sum_vec = _mm256_fmadd_ps(x_vec, y_vec, sum_vec);
        }

        // Horizontal sum of the 8 lanes
        let hi128 = _mm256_extractf128_ps(sum_vec, 1);
        let lo128 = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0x1B);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        let final_sum = _mm_add_ps(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        // Handle remaining elements with scalar operations
        for i in simd_len..len {
            result += x[i] * y[i];
        }

        result
    }

    /// SIMD-optimized L2 norm (Euclidean norm) for f32 vectors
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_l2_f32(x: &Array<f32>) -> f32 {
        let x_data = x.to_vec();
        unsafe { Self::avx2_norm_l2_f32(&x_data).sqrt() }
    }

    /// AVX2 optimized sum of squares for L2 norm
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_norm_l2_f32(x: &[f32]) -> f32 {
        let len = x.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum_vec = _mm256_setzero_ps();

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    x.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));

            // Compute x² and accumulate using FMA
            sum_vec = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
        }

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(sum_vec, 1);
        let lo128 = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0x1B);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        let final_sum = _mm_add_ps(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        // Handle remaining elements
        for i in simd_len..len {
            result += x[i] * x[i];
        }

        result
    }

    /// SIMD-optimized L1 norm (Manhattan norm) for f32 vectors
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_l1_f32(x: &Array<f32>) -> f32 {
        let x_data = x.to_vec();
        unsafe { Self::avx2_norm_l1_f32(&x_data) }
    }

    /// AVX2 optimized L1 norm implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_norm_l1_f32(x: &[f32]) -> f32 {
        let len = x.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum_vec = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0); // 0x80000000 for each element

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    x.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));

            // Compute absolute value by clearing the sign bit
            let abs_x = _mm256_andnot_ps(sign_mask, x_vec);
            sum_vec = _mm256_add_ps(sum_vec, abs_x);
        }

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(sum_vec, 1);
        let lo128 = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0x1B);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        let final_sum = _mm_add_ps(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        // Handle remaining elements
        for i in simd_len..len {
            result += x[i].abs();
        }

        result
    }

    /// Vectorized L-infinity norm for f32 (max absolute value)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_linf_f32(x: &Array<f32>) -> f32 {
        let x_data = x.to_vec();
        unsafe { Self::avx2_norm_linf_f32(&x_data) }
    }

    /// AVX2 optimized L-infinity norm for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_norm_linf_f32(x: &[f32]) -> f32 {
        let len = x.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut max_vec = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    x.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
            let abs_x = _mm256_andnot_ps(sign_mask, x_vec);
            max_vec = _mm256_max_ps(max_vec, abs_x);
        }

        // Horizontal max
        let hi128 = _mm256_extractf128_ps(max_vec, 1);
        let lo128 = _mm256_castps256_ps128(max_vec);
        let max128 = _mm_max_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(max128, max128, 0x1B);
        let maxs = _mm_max_ps(max128, shuf);
        let shuf2 = _mm_shuffle_ps(maxs, maxs, 0x01);
        let final_max = _mm_max_ps(maxs, shuf2);
        let mut result = _mm_cvtss_f32(final_max);

        // Handle remaining elements
        for i in simd_len..len {
            result = result.max(x[i].abs());
        }

        result
    }

    /// Vectorized L-infinity norm for f64 (max absolute value)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_linf_f64(x: &Array<f64>) -> f64 {
        let x_data = x.to_vec();
        unsafe { Self::avx2_norm_linf_f64(&x_data) }
    }

    /// AVX2 optimized L-infinity norm for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_norm_linf_f64(x: &[f64]) -> f64 {
        let len = x.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let mut max_vec = _mm256_setzero_pd();
        let sign_mask = _mm256_set1_pd(-0.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    x.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
            let abs_x = _mm256_andnot_pd(sign_mask, x_vec);
            max_vec = _mm256_max_pd(max_vec, abs_x);
        }

        // Horizontal max
        let hi128 = _mm256_extractf128_pd(max_vec, 1);
        let lo128 = _mm256_castpd256_pd128(max_vec);
        let max128 = _mm_max_pd(hi128, lo128);
        let shuf = _mm_shuffle_pd(max128, max128, 1);
        let final_max = _mm_max_pd(max128, shuf);
        let mut result = _mm_cvtsd_f64(final_max);

        // Handle remaining elements
        for i in simd_len..len {
            result = result.max(x[i].abs());
        }

        result
    }

    // ========================================
    // SIMD Reduction Operations for f32
    // ========================================

    /// Vectorized sum reduction for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_sum_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        unsafe { Self::avx2_sum_f32(&data) }
    }

    /// AVX2 optimized sum reduction for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sum_f32(input: &[f32]) -> f32 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F32_LANES - 1);

        // Use 4 accumulators
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let unroll_len = simd_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(i + AVX2_F32_LANES));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(i + 3 * AVX2_F32_LANES));

            sum0 = _mm256_add_ps(sum0, v0);
            sum1 = _mm256_add_ps(sum1, v1);
            sum2 = _mm256_add_ps(sum2, v2);
            sum3 = _mm256_add_ps(sum3, v3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F32_LANES) {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            sum0 = _mm256_add_ps(sum0, v);
        }

        // Combine accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Horizontal sum of 8 floats
        let hi128 = _mm256_extractf128_ps(sum0, 1);
        let lo128 = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
        let final_sum = _mm_add_ps(sums, shuf2);

        let mut result = _mm_cvtss_f32(final_sum);

        for i in simd_len..len {
            result += input[i];
        }

        result
    }

    /// Vectorized min reduction for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_min_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        if data.is_empty() {
            return f32::INFINITY;
        }
        unsafe { Self::avx2_min_f32(&data) }
    }

    /// AVX2 optimized min reduction for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_min_f32(input: &[f32]) -> f32 {
        let len = input.len();
        if len == 0 {
            return f32::INFINITY;
        }

        let simd_len = len & !(AVX2_F32_LANES - 1);
        let mut min0 = _mm256_set1_ps(input[0]);
        let mut min1 = min0;
        let mut min2 = min0;
        let mut min3 = min0;

        let unroll_len = simd_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(i + AVX2_F32_LANES));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(i + 3 * AVX2_F32_LANES));

            min0 = _mm256_min_ps(min0, v0);
            min1 = _mm256_min_ps(min1, v1);
            min2 = _mm256_min_ps(min2, v2);
            min3 = _mm256_min_ps(min3, v3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F32_LANES) {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            min0 = _mm256_min_ps(min0, v);
        }

        // Combine accumulators
        min0 = _mm256_min_ps(min0, min1);
        min2 = _mm256_min_ps(min2, min3);
        min0 = _mm256_min_ps(min0, min2);

        // Horizontal min
        let hi128 = _mm256_extractf128_ps(min0, 1);
        let lo128 = _mm256_castps256_ps128(min0);
        let min128 = _mm_min_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(min128, min128, 0b10_11_00_01);
        let mins = _mm_min_ps(min128, shuf);
        let shuf2 = _mm_shuffle_ps(mins, mins, 0b00_00_00_10);
        let final_min = _mm_min_ps(mins, shuf2);

        let mut result = _mm_cvtss_f32(final_min);

        for i in simd_len..len {
            if input[i] < result {
                result = input[i];
            }
        }

        result
    }

    /// Vectorized max reduction for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_max_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        if data.is_empty() {
            return f32::NEG_INFINITY;
        }
        unsafe { Self::avx2_max_f32(&data) }
    }

    /// AVX2 optimized max reduction for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_max_f32(input: &[f32]) -> f32 {
        let len = input.len();
        if len == 0 {
            return f32::NEG_INFINITY;
        }

        let simd_len = len & !(AVX2_F32_LANES - 1);
        let mut max0 = _mm256_set1_ps(input[0]);
        let mut max1 = max0;
        let mut max2 = max0;
        let mut max3 = max0;

        let unroll_len = simd_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(i + AVX2_F32_LANES));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(i + 3 * AVX2_F32_LANES));

            max0 = _mm256_max_ps(max0, v0);
            max1 = _mm256_max_ps(max1, v1);
            max2 = _mm256_max_ps(max2, v2);
            max3 = _mm256_max_ps(max3, v3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F32_LANES) {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            max0 = _mm256_max_ps(max0, v);
        }

        // Combine accumulators
        max0 = _mm256_max_ps(max0, max1);
        max2 = _mm256_max_ps(max2, max3);
        max0 = _mm256_max_ps(max0, max2);

        // Horizontal max
        let hi128 = _mm256_extractf128_ps(max0, 1);
        let lo128 = _mm256_castps256_ps128(max0);
        let max128 = _mm_max_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(max128, max128, 0b10_11_00_01);
        let maxs = _mm_max_ps(max128, shuf);
        let shuf2 = _mm_shuffle_ps(maxs, maxs, 0b00_00_00_10);
        let final_max = _mm_max_ps(maxs, shuf2);

        let mut result = _mm_cvtss_f32(final_max);

        for i in simd_len..len {
            if input[i] > result {
                result = input[i];
            }
        }

        result
    }

    /// Vectorized abs for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_abs_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];
        unsafe {
            Self::avx2_abs_f32(&data, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized abs for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_abs_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let result = _mm256_and_ps(x, abs_mask);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].abs();
        }
    }

    /// Vectorized clip for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_clip_f32(input: &Array<f32>, min_val: f32, max_val: f32) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];
        unsafe {
            Self::avx2_clip_f32(&data, min_val, max_val, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized clip for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_clip_f32(input: &[f32], min_val: f32, max_val: f32, output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);
        let min_vec = _mm256_set1_ps(min_val);
        let max_vec = _mm256_set1_ps(max_val);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let clipped = _mm256_min_ps(_mm256_max_ps(x, min_vec), max_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), clipped);
        }

        for i in simd_len..len {
            output[i] = input[i].max(min_val).min(max_val);
        }
    }

    // ========================================
    // SIMD Diff/Cumsum Operations
    // ========================================

    /// Vectorized first difference for f64 (diff[i] = arr[i+1] - arr[i])
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_diff_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        if data.len() < 2 {
            return Array::from_vec(vec![]);
        }
        let mut result = vec![0.0f64; data.len() - 1];
        unsafe {
            Self::avx2_diff_f64(&data, &mut result);
        }
        Array::from_vec(result)
    }

    /// AVX2 optimized first difference for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_diff_f64(input: &[f64], output: &mut [f64]) {
        let len = output.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let current = _mm256_loadu_pd(input.as_ptr().add(i));
            let next = _mm256_loadu_pd(input.as_ptr().add(i + 1));
            let diff = _mm256_sub_pd(next, current);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), diff);
        }

        for i in simd_len..len {
            output[i] = input[i + 1] - input[i];
        }
    }

    /// Vectorized cumulative sum for f64
    /// Note: Cumsum is inherently sequential, but we can still optimize the inner loop
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cumsum_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        if data.is_empty() {
            return Array::from_vec(vec![]);
        }
        let mut result = vec![0.0f64; data.len()];
        unsafe {
            Self::avx2_cumsum_f64(&data, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// Optimized cumulative sum for f64
    /// Uses blocking to improve cache performance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_cumsum_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        if len == 0 {
            return;
        }

        // Cumulative sum is inherently sequential, but we can optimize memory access
        // Process in blocks for better cache behavior
        const BLOCK_SIZE: usize = 64;
        let mut running_sum = 0.0f64;

        for block_start in (0..len).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(len);

            // Process block with sequential sum
            for i in block_start..block_end {
                running_sum += input[i];
                output[i] = running_sum;
            }
        }
    }

    /// Vectorized first difference for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_diff_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        if data.len() < 2 {
            return Array::from_vec(vec![]);
        }
        let mut result = vec![0.0f32; data.len() - 1];
        unsafe {
            Self::avx2_diff_f32(&data, &mut result);
        }
        Array::from_vec(result)
    }

    /// AVX2 optimized first difference for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_diff_f32(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let current = _mm256_loadu_ps(input.as_ptr().add(i));
            let next = _mm256_loadu_ps(input.as_ptr().add(i + 1));
            let diff = _mm256_sub_ps(next, current);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), diff);
        }

        for i in simd_len..len {
            output[i] = input[i + 1] - input[i];
        }
    }

    /// Vectorized linspace for f64
    /// Generates n evenly spaced values from start to stop
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_linspace_f64(start: f64, stop: f64, n: usize) -> Array<f64> {
        if n == 0 {
            return Array::from_vec(vec![]);
        }
        if n == 1 {
            return Array::from_vec(vec![start]);
        }
        let mut result = vec![0.0f64; n];
        let step = (stop - start) / (n - 1) as f64;
        unsafe {
            Self::avx2_linspace_f64(start, step, &mut result);
        }
        Array::from_vec(result)
    }

    /// AVX2 optimized linspace for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_linspace_f64(start: f64, step: f64, output: &mut [f64]) {
        let len = output.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let start_vec = _mm256_set1_pd(start);
        let step_vec = _mm256_set1_pd(step);
        let indices_base = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        let step_4 = _mm256_set1_pd(4.0 * step);

        let mut current_indices = indices_base;

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // value = start + index * step
            let values = _mm256_fmadd_pd(current_indices, step_vec, start_vec);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), values);
            current_indices = _mm256_add_pd(current_indices, _mm256_set1_pd(AVX2_F64_LANES as f64));
        }

        for i in simd_len..len {
            output[i] = start + (i as f64) * step;
        }
    }

    /// Vectorized arange for f64
    /// Generates values from start to stop with given step
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_arange_f64(start: f64, stop: f64, step: f64) -> Array<f64> {
        if step == 0.0 || (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Array::from_vec(vec![]);
        }
        let n = ((stop - start) / step).ceil() as usize;
        let mut result = vec![0.0f64; n];
        unsafe {
            Self::avx2_linspace_f64(start, step, &mut result);
        }
        Array::from_vec(result)
    }

    /// Vectorized gradient computation for f64 (central differences)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_gradient_f64(input: &Array<f64>, spacing: f64) -> Array<f64> {
        let data = input.to_vec();
        let len = data.len();
        if len < 2 {
            return Array::from_vec(vec![0.0; len.max(1)]);
        }
        let mut result = vec![0.0f64; len];
        unsafe {
            Self::avx2_gradient_f64(&data, spacing, &mut result);
        }
        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized gradient using central differences
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_gradient_f64(input: &[f64], spacing: f64, output: &mut [f64]) {
        let len = input.len();
        if len < 2 {
            if len == 1 {
                output[0] = 0.0;
            }
            return;
        }

        // Forward difference at the start
        output[0] = (input[1] - input[0]) / spacing;

        // Backward difference at the end
        output[len - 1] = (input[len - 1] - input[len - 2]) / spacing;

        // Central differences for the middle
        if len > 2 {
            let mid_len = len - 2;
            let simd_len = mid_len & !(AVX2_F64_LANES - 1);
            let half_spacing_inv = _mm256_set1_pd(0.5 / spacing);

            for i in (0..simd_len).step_by(AVX2_F64_LANES) {
                let prev = _mm256_loadu_pd(input.as_ptr().add(i));
                let next = _mm256_loadu_pd(input.as_ptr().add(i + 2));
                let diff = _mm256_sub_pd(next, prev);
                let grad = _mm256_mul_pd(diff, half_spacing_inv);
                _mm256_storeu_pd(output.as_mut_ptr().add(i + 1), grad);
            }

            for i in simd_len..mid_len {
                output[i + 1] = (input[i + 2] - input[i]) / (2.0 * spacing);
            }
        }
    }

    /// Memory bandwidth optimized copy with prefetching
    #[cfg(target_arch = "x86_64")]
    pub fn optimized_copy_f32(src: &Array<f32>, dst: &mut Array<f32>) -> Result<()> {
        if src.shape() != dst.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: src.shape(),
                actual: dst.shape(),
            });
        }

        let src_data = src.to_vec();
        let mut dst_data = dst.to_vec();

        unsafe {
            Self::avx2_optimized_copy_f32(&src_data, &mut dst_data);
        }

        *dst = Array::from_vec(dst_data).reshape(&src.shape());
        Ok(())
    }

    /// AVX2 optimized memory copy with prefetching
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_optimized_copy_f32(src: &[f32], dst: &mut [f32]) {
        let len = src.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            // Prefetch next cache lines
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    src.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    dst.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let data = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), data);
        }

        // Handle remaining elements
        dst[simd_len..len].copy_from_slice(&src[simd_len..len]);
    }
}

/// Performance monitoring for SIMD operations
pub struct SimdPerformanceMonitor {
    pub operations_count: u64,
    pub total_elements: u64,
    pub cache_misses: u64,
    pub vectorization_ratio: f64,
}

impl Default for SimdPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            operations_count: 0,
            total_elements: 0,
            cache_misses: 0,
            vectorization_ratio: 0.0,
        }
    }

    pub fn record_operation(&mut self, elements: usize, vectorized_elements: usize) {
        self.operations_count += 1;
        self.total_elements += elements as u64;
        self.vectorization_ratio = vectorized_elements as f64 / elements as f64;
    }

    pub fn elements_per_second(&self, duration_secs: f64) -> f64 {
        self.total_elements as f64 / duration_secs
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// SIMD benchmark utilities
pub struct SimdBenchmark;

impl SimdBenchmark {
    /// Benchmark different SIMD implementations
    pub fn compare_implementations(size: usize, iterations: usize) -> SimdBenchmarkResults {
        use std::time::Instant;

        let data1 = Array::from_vec((0..size).map(|i| i as f32).collect::<Vec<_>>());
        let data2 = Array::from_vec((0..size).map(|i| (i + 1) as f32).collect::<Vec<_>>());

        // Benchmark scalar addition
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = data1.add(&data2);
        }
        let scalar_time = start.elapsed().as_nanos() as f64;

        // Benchmark SIMD addition
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = data1.simd_add(&data2).unwrap();
        }
        let simd_time = start.elapsed().as_nanos() as f64;

        SimdBenchmarkResults {
            scalar_time_ns: scalar_time / iterations as f64,
            simd_time_ns: simd_time / iterations as f64,
            speedup: scalar_time / simd_time,
            elements: size,
            throughput_elements_per_ns: size as f64 / (simd_time / iterations as f64),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimdBenchmarkResults {
    pub scalar_time_ns: f64,
    pub simd_time_ns: f64,
    pub speedup: f64,
    pub elements: usize,
    pub throughput_elements_per_ns: f64,
}

impl SimdBenchmarkResults {
    pub fn print_summary(&self) {
        println!("SIMD Benchmark Results:");
        println!("  Elements: {}", self.elements);
        println!("  Scalar time: {:.2} ns", self.scalar_time_ns);
        println!("  SIMD time: {:.2} ns", self.simd_time_ns);
        println!("  Speedup: {:.2}x", self.speedup);
        println!(
            "  Throughput: {:.2} elements/ns",
            self.throughput_elements_per_ns
        );
    }
}

/// AVX2 vectorized sin function
#[cfg(target_arch = "x86_64")]
pub fn vectorized_sin_f32(input: &Array<f32>) -> Array<f32> {
    let input_data = input.to_vec();
    let mut result = vec![0.0f32; input_data.len()];

    for (i, &x) in input_data.iter().enumerate() {
        result[i] = x.sin();
    }

    Array::from_vec(result).reshape(&input.shape())
}

/// Kahan summation for improved numerical accuracy  
#[cfg(target_arch = "x86_64")]
pub fn kahan_sum_f32(input: &Array<f32>) -> f32 {
    let data = input.to_vec();
    let mut sum = 0.0f32;
    let mut c = 0.0f32; // A running compensation for lost low-order bits

    for &value in &data {
        let y = value - c; // So far, so good: c is zero
        let t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost
        c = (t - sum) - y; // (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
        sum = t; // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_enhanced_exp() {
        let input = Array::from_vec(vec![0.0, 1.0, 2.0, -1.0]);
        let result = EnhancedSimdOps::vectorized_exp_f32(&input);

        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], std::f32::consts::E, epsilon = 1e-6);
        assert_relative_eq!(
            result.to_vec()[2],
            std::f32::consts::E.powi(2),
            epsilon = 1e-5
        );
        assert_relative_eq!(
            result.to_vec()[3],
            1.0 / std::f32::consts::E,
            epsilon = 1e-6
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_enhanced_log() {
        let input = Array::from_vec(vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
        let result = EnhancedSimdOps::vectorized_log_f32(&input);

        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.to_vec()[2], 2.0, epsilon = 1e-5);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", not(feature = "ci-safe")))]
    fn test_enhanced_sin() {
        let input = Array::from_vec(vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
        let result = vectorized_sin_f32(&input);

        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.to_vec()[2], 0.0, epsilon = 1e-5);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", not(feature = "ci-safe")))]
    fn test_kahan_sum() {
        let input = Array::from_vec(vec![1.0f32; 1000]);
        let result = kahan_sum_f32(&input);
        assert_relative_eq!(result, 1000.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", not(feature = "ci-safe")))]
    fn test_complex_multiply() {
        let a_r = Array::from_vec(vec![1.0, 2.0]);
        let a_i = Array::from_vec(vec![3.0, 4.0]);
        let b_r = Array::from_vec(vec![5.0, 6.0]);
        let b_i = Array::from_vec(vec![7.0, 8.0]);

        let (c_r, c_i) = EnhancedSimdOps::complex_multiply_f32(&a_r, &a_i, &b_r, &b_i).unwrap();

        // (1+3i) * (5+7i) = 5 + 7i + 15i - 21 = -16 + 22i
        assert_relative_eq!(c_r.to_vec()[0], -16.0, epsilon = 1e-6);
        assert_relative_eq!(c_i.to_vec()[0], 22.0, epsilon = 1e-6);

        // (2+4i) * (6+8i) = 12 + 16i + 24i - 32 = -20 + 40i
        assert_relative_eq!(c_r.to_vec()[1], -20.0, epsilon = 1e-6);
        assert_relative_eq!(c_i.to_vec()[1], 40.0, epsilon = 1e-6);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = SimdPerformanceMonitor::new();
        monitor.record_operation(1000, 800);

        assert_eq!(monitor.operations_count, 1);
        assert_eq!(monitor.total_elements, 1000);
        assert_relative_eq!(monitor.vectorization_ratio, 0.8, epsilon = 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_dot_product_f32() {
        let x = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let y = Array::from_vec(vec![2.0f32, 3.0, 4.0, 5.0]);

        let result = EnhancedSimdOps::vectorized_dot_f32(&x, &y).unwrap();
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0; // 40.0

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_dot_product_f64() {
        let x = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let y = Array::from_vec(vec![2.0f64, 3.0, 4.0, 5.0]);

        let result = EnhancedSimdOps::vectorized_dot_f64(&x, &y);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0; // 40.0

        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_dot_product_large() {
        // Test with larger vectors that will use SIMD lanes fully
        let size = 1000;
        let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let x = Array::from_vec(x_data.clone());
        let y = Array::from_vec(y_data.clone());

        let simd_result = EnhancedSimdOps::vectorized_dot_f32(&x, &y).unwrap();

        // Compute expected result using scalar operations
        let expected: f32 = x_data.iter().zip(y_data.iter()).map(|(a, b)| a * b).sum();

        assert_relative_eq!(simd_result, expected, epsilon = 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_l2_norm_f32() {
        let x = Array::from_vec(vec![3.0f32, 4.0]);
        let result = EnhancedSimdOps::vectorized_norm_l2_f32(&x);
        let expected = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt(); // 5.0

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_l2_norm_f64() {
        let x = Array::from_vec(vec![3.0f64, 4.0]);
        let result = EnhancedSimdOps::vectorized_norm_l2_f64(&x);
        let expected = (3.0f64 * 3.0 + 4.0 * 4.0).sqrt(); // 5.0

        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_l1_norm_f32() {
        let x = Array::from_vec(vec![-3.0f32, 4.0, -2.0, 1.0]);
        let result = EnhancedSimdOps::vectorized_norm_l1_f32(&x);
        let expected = 3.0f32 + 4.0 + 2.0 + 1.0; // 10.0

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_norm_large_vector() {
        // Test with a large vector to ensure SIMD lanes are used
        let size = 1000;
        let x_data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
        let x = Array::from_vec(x_data.clone());

        let simd_l2 = EnhancedSimdOps::vectorized_norm_l2_f32(&x);
        let simd_l1 = EnhancedSimdOps::vectorized_norm_l1_f32(&x);

        // Compute expected results
        let expected_l2 = x_data.iter().map(|&val| val * val).sum::<f32>().sqrt();
        let expected_l1 = x_data.iter().map(|&val| val.abs()).sum::<f32>();

        assert_relative_eq!(simd_l2, expected_l2, epsilon = 1e-2);
        assert_relative_eq!(simd_l1, expected_l1, epsilon = 1e-5);
    }
}
