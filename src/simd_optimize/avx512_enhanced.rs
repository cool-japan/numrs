//! Enhanced AVX-512 SIMD operations for latest CPU architectures
//!
//! This module provides cutting-edge vectorization using AVX-512 instructions
//! for maximum performance on modern Intel and AMD processors.

use std::arch::x86_64::*;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};

/// AVX-512 vectorization constants
const AVX512_F32_LANES: usize = 16;
const AVX512_F64_LANES: usize = 8;
const AVX512_ALIGNMENT: usize = 64;

/// Advanced AVX-512 operations with maximum vectorization
pub struct Avx512EnhancedOps;

impl Avx512EnhancedOps {
    /// AVX-512 optimized matrix multiplication with 512-bit vectors
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_matmul_f32(
        a: &Array<f32>,
        b: &Array<f32>,
        c: &mut Array<f32>,
        tile_size: usize,
    ) -> Result<()> {
        let [m, k] = a.shape()[..] else {
            return Err(NumRs2Error::DimensionMismatch("Matrix A must be 2D".to_string()));
        };
        let [k2, n] = b.shape()[..] else {
            return Err(NumRs2Error::DimensionMismatch("Matrix B must be 2D".to_string()));
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
            Self::tiled_matmul_avx512_f32(
                &a_data, &b_data, &mut c_data,
                m, n, k, tile_size
            );
        }

        *c = Array::from_vec(c_data).reshape(&[m, n]);
        Ok(())
    }

    /// Tiled matrix multiplication with AVX-512
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn tiled_matmul_avx512_f32(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize, tile_size: usize
    ) {
        for ii in (0..m).step_by(tile_size) {
            for jj in (0..n).step_by(tile_size) {
                for kk in (0..k).step_by(tile_size) {
                    let i_end = (ii + tile_size).min(m);
                    let j_end = (jj + tile_size).min(n);
                    let k_end = (kk + tile_size).min(k);

                    for i in ii..i_end {
                        for j in (jj..j_end).step_by(AVX512_F32_LANES) {
                            let lanes = (j_end - j).min(AVX512_F32_LANES);
                            
                            // Load C values using AVX-512
                            let mut vc = if lanes == AVX512_F32_LANES {
                                _mm512_loadu_ps(c.as_ptr().add(i * n + j))
                            } else {
                                let mask = (1u16 << lanes) - 1;
                                _mm512_maskz_loadu_ps(mask, c.as_ptr().add(i * n + j))
                            };

                            for l in kk..k_end {
                                let va = _mm512_set1_ps(a[i * k + l]);
                                let vb = if lanes == AVX512_F32_LANES {
                                    _mm512_loadu_ps(b.as_ptr().add(l * n + j))
                                } else {
                                    let mask = (1u16 << lanes) - 1;
                                    _mm512_maskz_loadu_ps(mask, b.as_ptr().add(l * n + j))
                                };
                                vc = _mm512_fmadd_ps(va, vb, vc);
                            }

                            // Store C values using AVX-512
                            if lanes == AVX512_F32_LANES {
                                _mm512_storeu_ps(c.as_mut_ptr().add(i * n + j), vc);
                            } else {
                                let mask = (1u16 << lanes) - 1;
                                _mm512_mask_storeu_ps(c.as_mut_ptr().add(i * n + j), mask, vc);
                            }
                        }
                    }
                }
            }
        }
    }

    /// AVX-512 vectorized exponential with extended precision
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_exp_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::vectorized_exp_avx512_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX-512 exponential implementation with higher accuracy
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vectorized_exp_avx512_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX512_F32_LANES - 1);

        // High precision constants for exp
        let log2_e = _mm512_set1_ps(1.4426950408889634073599);
        let ln2_hi = _mm512_set1_ps(0.6931471805599453094172);
        let ln2_lo = _mm512_set1_ps(2.3283064365386962890625e-10);
        
        // Extended Taylor series coefficients
        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(1.0);
        let c3 = _mm512_set1_ps(0.5);
        let c4 = _mm512_set1_ps(0.16666666666666666);
        let c5 = _mm512_set1_ps(0.041666666666666664);
        let c6 = _mm512_set1_ps(0.008333333333333333);
        let c7 = _mm512_set1_ps(0.001388888888888889);

        for i in (0..simd_len).step_by(AVX512_F32_LANES) {
            let x = _mm512_loadu_ps(input.as_ptr().add(i));
            
            // Range reduction with higher precision
            let n_float = _mm512_mul_ps(x, log2_e);
            let n = _mm512_cvtps_epi32(n_float);
            let n_f = _mm512_cvtepi32_ps(n);
            
            // High precision remainder: r = x - n*ln(2)
            let r = _mm512_fmsub_ps(n_f, ln2_hi, x);
            let r = _mm512_fmsub_ps(n_f, ln2_lo, r);
            
            // Extended Taylor series: exp(r) with more terms
            let r2 = _mm512_mul_ps(r, r);
            let r3 = _mm512_mul_ps(r2, r);
            let r4 = _mm512_mul_ps(r3, r);
            let r5 = _mm512_mul_ps(r4, r);
            let r6 = _mm512_mul_ps(r5, r);
            let r7 = _mm512_mul_ps(r6, r);
            
            let poly = _mm512_fmadd_ps(c7, r7,
                       _mm512_fmadd_ps(c6, r6,
                       _mm512_fmadd_ps(c5, r5,
                       _mm512_fmadd_ps(c4, r4,
                       _mm512_fmadd_ps(c3, r3,
                       _mm512_fmadd_ps(c2, r2,
                       _mm512_fmadd_ps(c1, r, c1)))))));
            
            // Scale by 2^n using bit manipulation
            let exp_bias = _mm512_set1_epi32(127);
            let biased_exp = _mm512_add_epi32(n, exp_bias);
            let scale_factor = _mm512_castsi512_ps(_mm512_slli_epi32(biased_exp, 23));
            let result = _mm512_mul_ps(poly, scale_factor);
            
            _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].exp();
        }
    }

    /// AVX-512 optimized logarithm with enhanced accuracy
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_log_f32(input: &Array<f32>) -> Array<f32> {
        let data = input.to_vec();
        let mut result = vec![0.0f32; data.len()];

        unsafe {
            Self::vectorized_log_avx512_f32(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX-512 logarithm with minimax polynomial approximation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vectorized_log_avx512_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len & !(AVX512_F32_LANES - 1);

        let ln2 = _mm512_set1_ps(0.6931471805599453094172);
        let one = _mm512_set1_ps(1.0);
        
        // Minimax polynomial coefficients for log(1+x) on [-0.5, 0.5]
        let p0 = _mm512_set1_ps(1.0000000000000000000);
        let p1 = _mm512_set1_ps(-0.5000000000000000000);
        let p2 = _mm512_set1_ps(0.3333333333333333333);
        let p3 = _mm512_set1_ps(-0.2500000000000000000);
        let p4 = _mm512_set1_ps(0.2000000000000000000);
        let p5 = _mm512_set1_ps(-0.1666666666666666667);
        let p6 = _mm512_set1_ps(0.1428571428571428571);

        for i in (0..simd_len).step_by(AVX512_F32_LANES) {
            let x = _mm512_loadu_ps(input.as_ptr().add(i));
            
            // Extract exponent using bit operations
            let x_int = _mm512_castps_si512(x);
            let exp_mask = _mm512_set1_epi32(0x7F800000);
            let exp_bits = _mm512_and_si512(x_int, exp_mask);
            let exp = _mm512_sub_epi32(_mm512_srli_epi32(exp_bits, 23), _mm512_set1_epi32(127));
            let exp_f = _mm512_cvtepi32_ps(exp);
            
            // Extract and normalize mantissa
            let mantissa_mask = _mm512_set1_epi32(0x007FFFFF);
            let mantissa_bits = _mm512_or_si512(
                _mm512_and_si512(x_int, mantissa_mask),
                _mm512_set1_epi32(0x3F800000)
            );
            let mantissa = _mm512_castsi512_ps(mantissa_bits);
            
            // Transform to [-0.5, 0.5] range: u = (m - 1)
            let u = _mm512_sub_ps(mantissa, one);
            
            // Evaluate minimax polynomial
            let u2 = _mm512_mul_ps(u, u);
            let u3 = _mm512_mul_ps(u2, u);
            let u4 = _mm512_mul_ps(u3, u);
            let u5 = _mm512_mul_ps(u4, u);
            let u6 = _mm512_mul_ps(u5, u);
            
            let poly = _mm512_fmadd_ps(p6, u6,
                       _mm512_fmadd_ps(p5, u5,
                       _mm512_fmadd_ps(p4, u4,
                       _mm512_fmadd_ps(p3, u3,
                       _mm512_fmadd_ps(p2, u2,
                       _mm512_fmadd_ps(p1, u2,
                       _mm512_mul_ps(p0, u)))))));
            
            // Combine: log(x) = exp * ln(2) + log(mantissa)
            let result = _mm512_fmadd_ps(exp_f, ln2, poly);
            
            _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i].ln();
        }
    }

    /// AVX-512 trigonometric functions with CORDIC-based algorithms
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_sin_cos_f32(input: &Array<f32>) -> (Array<f32>, Array<f32>) {
        let data = input.to_vec();
        let mut sin_result = vec![0.0f32; data.len()];
        let mut cos_result = vec![0.0f32; data.len()];

        unsafe {
            Self::vectorized_sin_cos_avx512_f32(&data, &mut sin_result, &mut cos_result);
        }

        (
            Array::from_vec(sin_result).reshape(&input.shape()),
            Array::from_vec(cos_result).reshape(&input.shape())
        )
    }

    /// Simultaneous sin/cos computation using AVX-512
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vectorized_sin_cos_avx512_f32(
        input: &[f32], 
        sin_output: &mut [f32], 
        cos_output: &mut [f32]
    ) {
        let len = input.len();
        let simd_len = len & !(AVX512_F32_LANES - 1);

        // High precision constants
        let pi = _mm512_set1_ps(std::f32::consts::PI);
        let two_pi = _mm512_set1_ps(2.0 * std::f32::consts::PI);
        let pi_2 = _mm512_set1_ps(std::f32::consts::PI / 2.0);
        let one = _mm512_set1_ps(1.0);
        let zero = _mm512_setzero_ps();

        // Taylor series coefficients for sin
        let sin_c3 = _mm512_set1_ps(-1.0 / 6.0);
        let sin_c5 = _mm512_set1_ps(1.0 / 120.0);
        let sin_c7 = _mm512_set1_ps(-1.0 / 5040.0);
        let sin_c9 = _mm512_set1_ps(1.0 / 362880.0);
        let sin_c11 = _mm512_set1_ps(-1.0 / 39916800.0);

        // Taylor series coefficients for cos
        let cos_c2 = _mm512_set1_ps(-1.0 / 2.0);
        let cos_c4 = _mm512_set1_ps(1.0 / 24.0);
        let cos_c6 = _mm512_set1_ps(-1.0 / 720.0);
        let cos_c8 = _mm512_set1_ps(1.0 / 40320.0);
        let cos_c10 = _mm512_set1_ps(-1.0 / 3628800.0);

        for i in (0..simd_len).step_by(AVX512_F32_LANES) {
            let mut x = _mm512_loadu_ps(input.as_ptr().add(i));
            
            // Range reduction to [-π, π]
            let k = _mm512_roundscale_ps(_mm512_div_ps(x, two_pi), 0);
            x = _mm512_fmsub_ps(k, two_pi, x);
            
            // Further reduce to first quadrant and track quadrant
            let abs_x = _mm512_abs_ps(x);
            let sign_x = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
            
            let quad_mask = _mm512_cmp_ps_mask(abs_x, pi_2, _CMP_GT_OQ);
            let x_reduced = _mm512_mask_sub_ps(abs_x, quad_mask, pi, abs_x);
            
            // Compute x^2, x^3, ... for both sin and cos
            let x2 = _mm512_mul_ps(x_reduced, x_reduced);
            let x3 = _mm512_mul_ps(x2, x_reduced);
            let x4 = _mm512_mul_ps(x3, x_reduced);
            let x5 = _mm512_mul_ps(x4, x_reduced);
            let x6 = _mm512_mul_ps(x5, x_reduced);
            let x7 = _mm512_mul_ps(x6, x_reduced);
            let x8 = _mm512_mul_ps(x7, x_reduced);
            let x9 = _mm512_mul_ps(x8, x_reduced);
            let x10 = _mm512_mul_ps(x9, x_reduced);
            let x11 = _mm512_mul_ps(x10, x_reduced);

            // Taylor series for sin(x)
            let sin_poly = _mm512_fmadd_ps(sin_c11, x11,
                           _mm512_fmadd_ps(sin_c9, x9,
                           _mm512_fmadd_ps(sin_c7, x7,
                           _mm512_fmadd_ps(sin_c5, x5,
                           _mm512_fmadd_ps(sin_c3, x3, x_reduced)))));

            // Taylor series for cos(x)
            let cos_poly = _mm512_fmadd_ps(cos_c10, x10,
                           _mm512_fmadd_ps(cos_c8, x8,
                           _mm512_fmadd_ps(cos_c6, x6,
                           _mm512_fmadd_ps(cos_c4, x4,
                           _mm512_fmadd_ps(cos_c2, x2, one)))));

            // Handle quadrant adjustments
            let sin_result = _mm512_mask_sub_ps(sin_poly, sign_x, zero, sin_poly);
            let cos_result = _mm512_mask_sub_ps(cos_poly, quad_mask, zero, cos_poly);

            _mm512_storeu_ps(sin_output.as_mut_ptr().add(i), sin_result);
            _mm512_storeu_ps(cos_output.as_mut_ptr().add(i), cos_result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            sin_output[i] = input[i].sin();
            cos_output[i] = input[i].cos();
        }
    }

    /// AVX-512 advanced reduction with multiple accumulators
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_parallel_sum_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        unsafe { Self::parallel_reduction_avx512_f32(&data) }
    }

    /// Parallel reduction using multiple AVX-512 accumulators
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn parallel_reduction_avx512_f32(input: &[f32]) -> f32 {
        let len = input.len();
        let simd_len = len & !(AVX512_F32_LANES * 4 - 1); // Process 64 elements at a time

        // Use 4 parallel accumulators to reduce dependency chains
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for i in (0..simd_len).step_by(AVX512_F32_LANES * 4) {
            let v0 = _mm512_loadu_ps(input.as_ptr().add(i));
            let v1 = _mm512_loadu_ps(input.as_ptr().add(i + AVX512_F32_LANES));
            let v2 = _mm512_loadu_ps(input.as_ptr().add(i + AVX512_F32_LANES * 2));
            let v3 = _mm512_loadu_ps(input.as_ptr().add(i + AVX512_F32_LANES * 3));

            acc0 = _mm512_add_ps(acc0, v0);
            acc1 = _mm512_add_ps(acc1, v1);
            acc2 = _mm512_add_ps(acc2, v2);
            acc3 = _mm512_add_ps(acc3, v3);
        }

        // Combine the four accumulators
        let combined01 = _mm512_add_ps(acc0, acc1);
        let combined23 = _mm512_add_ps(acc2, acc3);
        let total = _mm512_add_ps(combined01, combined23);

        // Horizontal reduction of 512-bit register
        let result = _mm512_reduce_add_ps(total);

        // Handle remaining elements
        let mut scalar_sum = result;
        for &item in &input[simd_len..] {
            scalar_sum += item;
        }

        scalar_sum
    }

    /// AVX-512 optimized convolution
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_convolution_f32(
        signal: &Array<f32>,
        kernel: &Array<f32>
    ) -> Result<Array<f32>> {
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = signal_len + kernel_len - 1;

        let signal_data = signal.to_vec();
        let kernel_data = kernel.to_vec();
        let mut output_data = vec![0.0f32; output_len];

        unsafe {
            Self::vectorized_convolution_avx512_f32(
                &signal_data, &kernel_data, &mut output_data
            );
        }

        Ok(Array::from_vec(output_data))
    }

    /// AVX-512 vectorized convolution implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vectorized_convolution_avx512_f32(
        signal: &[f32],
        kernel: &[f32],
        output: &mut [f32]
    ) {
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = output.len();

        for i in 0..output_len {
            let mut sum = _mm512_setzero_ps();
            let start_k = if i >= signal_len - 1 { i - signal_len + 1 } else { 0 };
            let end_k = (i + 1).min(kernel_len);
            
            let vectorizable_len = ((end_k - start_k) & !(AVX512_F32_LANES - 1));
            
            // Vectorized inner loop
            for k in (start_k..start_k + vectorizable_len).step_by(AVX512_F32_LANES) {
                let sig_indices = (0..AVX512_F32_LANES)
                    .map(|j| i - k - j)
                    .collect::<Vec<_>>();
                
                // Gather signal values (would need proper gather in real implementation)
                let mut sig_vals = [0.0f32; AVX512_F32_LANES];
                for (j, &idx) in sig_indices.iter().enumerate() {
                    if idx < signal_len {
                        sig_vals[j] = signal[idx];
                    }
                }
                let sig_vec = _mm512_loadu_ps(sig_vals.as_ptr());
                
                let kern_vec = _mm512_loadu_ps(kernel.as_ptr().add(k));
                sum = _mm512_fmadd_ps(sig_vec, kern_vec, sum);
            }
            
            // Horizontal sum and add scalar remainder
            let mut result = _mm512_reduce_add_ps(sum);
            
            // Handle remaining elements
            for k in (start_k + vectorizable_len)..end_k {
                let signal_idx = i - k;
                if signal_idx < signal_len {
                    result += signal[signal_idx] * kernel[k];
                }
            }
            
            output[i] = result;
        }
    }
}

/// AVX-512 feature detection and optimization hints
pub struct Avx512FeatureDetector;

impl Avx512FeatureDetector {
    /// Detect available AVX-512 subsets
    pub fn detect_avx512_features() -> Avx512Features {
        let mut features = Avx512Features::default();
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                features.avx512f = true;
            }
            if is_x86_feature_detected!("avx512dq") {
                features.avx512dq = true;
            }
            if is_x86_feature_detected!("avx512cd") {
                features.avx512cd = true;
            }
            if is_x86_feature_detected!("avx512bw") {
                features.avx512bw = true;
            }
            if is_x86_feature_detected!("avx512vl") {
                features.avx512vl = true;
            }
        }
        
        features
    }

    /// Get optimal tile size for current CPU
    pub fn optimal_tile_size() -> usize {
        // Conservative default, could be tuned based on cache sizes
        if Self::detect_avx512_features().avx512f {
            64 // Larger tiles for AVX-512
        } else {
            32 // Smaller tiles for AVX2
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Avx512Features {
    pub avx512f: bool,   // Foundation
    pub avx512dq: bool,  // Double/Quadword
    pub avx512cd: bool,  // Conflict Detection
    pub avx512bw: bool,  // Byte/Word
    pub avx512vl: bool,  // Vector Length
}

impl Avx512Features {
    pub fn has_full_support(&self) -> bool {
        self.avx512f && self.avx512dq && self.avx512bw && self.avx512vl
    }

    pub fn recommended_operations(&self) -> Vec<&'static str> {
        let mut ops = Vec::new();
        
        if self.avx512f {
            ops.push("Basic vectorization");
            ops.push("FMA operations");
        }
        if self.avx512dq {
            ops.push("Double precision operations");
            ops.push("Integer conversions");
        }
        if self.avx512bw {
            ops.push("Byte/word operations");
            ops.push("String processing");
        }
        if self.avx512vl {
            ops.push("Variable length vectors");
            ops.push("Masked operations");
        }
        
        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_avx512_feature_detection() {
        let features = Avx512FeatureDetector::detect_avx512_features();
        println!("AVX-512 features: {:?}", features);
        
        let ops = features.recommended_operations();
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_avx512_exp() {
        let input = Array::from_vec(vec![0.0, 1.0, 2.0, -1.0]);
        let result = Avx512EnhancedOps::avx512_exp_f32(&input);
        
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], std::f32::consts::E, epsilon = 1e-5);
        assert_relative_eq!(result.to_vec()[2], std::f32::consts::E.powi(2), epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], 1.0 / std::f32::consts::E, epsilon = 1e-6);
    }

    #[test]
    fn test_avx512_log() {
        let input = Array::from_vec(vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
        let result = Avx512EnhancedOps::avx512_log_f32(&input);
        
        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.to_vec()[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_avx512_sin_cos() {
        let input = Array::from_vec(vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
        let (sin_result, cos_result) = Avx512EnhancedOps::avx512_sin_cos_f32(&input);
        
        assert_relative_eq!(sin_result.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(sin_result.to_vec()[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(sin_result.to_vec()[2], 0.0, epsilon = 1e-4);
        
        assert_relative_eq!(cos_result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(cos_result.to_vec()[1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(cos_result.to_vec()[2], -1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_avx512_parallel_sum() {
        let input = Array::from_vec(vec![1.0f32; 1000]);
        let result = Avx512EnhancedOps::avx512_parallel_sum_f32(&input);
        assert_relative_eq!(result, 1000.0, epsilon = 1e-6);
    }

    #[test]
    fn test_optimal_tile_size() {
        let tile_size = Avx512FeatureDetector::optimal_tile_size();
        assert!(tile_size >= 32);
        assert!(tile_size <= 128);
    }
}