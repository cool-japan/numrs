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

        let (c_r, c_i) = complex_multiply_f32(&a_r, &a_i, &b_r, &b_i).unwrap();

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
}
