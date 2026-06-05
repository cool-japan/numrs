//! Special SIMD operations
//!
//! This module provides AVX2 optimized implementations for:
//! - Cache-aware matrix multiplication
//! - Complex number multiplication
//! - Kahan summation
//! - Diff and cumsum operations
//! - Linspace and arange generation
//! - Gradient computation
//! - Memory copy optimization

use super::{EnhancedSimdOps, AVX2_F32_LANES, AVX2_F64_LANES, PREFETCH_DISTANCE};
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl EnhancedSimdOps {
    // ========================================
    // Cache-Aware Matrix Multiplication
    // ========================================

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

                            for kp in kk..k_end {
                                let a_val = _mm256_set1_ps(a[i * k + kp]);

                                // Load B values
                                let vb = if lanes == AVX2_F32_LANES {
                                    _mm256_loadu_ps(b.as_ptr().add(kp * n + j))
                                } else {
                                    let mut temp = [0.0f32; AVX2_F32_LANES];
                                    for l in 0..lanes {
                                        temp[l] = b[kp * n + j + l];
                                    }
                                    _mm256_loadu_ps(temp.as_ptr())
                                };

                                // FMA: vc += a_val * vb
                                vc = _mm256_fmadd_ps(a_val, vb, vc);
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

    // ========================================
    // Complex Number Operations
    // ========================================

    /// Complex number multiplication for f32 arrays
    /// Given (a_r + a_i*i) * (b_r + b_i*i) = (a_r*b_r - a_i*b_i) + (a_r*b_i + a_i*b_r)*i
    #[cfg(target_arch = "x86_64")]
    pub fn complex_multiply_f32(
        a_real: &Array<f32>,
        a_imag: &Array<f32>,
        b_real: &Array<f32>,
        b_imag: &Array<f32>,
    ) -> Result<(Array<f32>, Array<f32>)> {
        let a_r = a_real.to_vec();
        let a_i = a_imag.to_vec();
        let b_r = b_real.to_vec();
        let b_i = b_imag.to_vec();

        let len = a_r.len().min(a_i.len()).min(b_r.len()).min(b_i.len());
        let mut c_r = vec![0.0f32; len];
        let mut c_i = vec![0.0f32; len];

        unsafe {
            Self::avx2_complex_multiply_f32(
                &a_r[..len],
                &a_i[..len],
                &b_r[..len],
                &b_i[..len],
                &mut c_r,
                &mut c_i,
            );
        }

        Ok((
            Array::from_vec(c_r).reshape(&a_real.shape()),
            Array::from_vec(c_i).reshape(&a_real.shape()),
        ))
    }

    /// AVX2 optimized complex multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_complex_multiply_f32(
        a_r: &[f32],
        a_i: &[f32],
        b_r: &[f32],
        b_i: &[f32],
        c_r: &mut [f32],
        c_i: &mut [f32],
    ) {
        let len = a_r.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            // Load inputs
            let ar = _mm256_loadu_ps(a_r.as_ptr().add(i));
            let ai = _mm256_loadu_ps(a_i.as_ptr().add(i));
            let br = _mm256_loadu_ps(b_r.as_ptr().add(i));
            let bi = _mm256_loadu_ps(b_i.as_ptr().add(i));

            // Real part: a_r*b_r - a_i*b_i
            let real = _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi));

            // Imaginary part: a_r*b_i + a_i*b_r
            let imag = _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));

            // Store results
            _mm256_storeu_ps(c_r.as_mut_ptr().add(i), real);
            _mm256_storeu_ps(c_i.as_mut_ptr().add(i), imag);
        }

        // Handle remaining elements
        for i in simd_len..len {
            c_r[i] = a_r[i] * b_r[i] - a_i[i] * b_i[i];
            c_i[i] = a_r[i] * b_i[i] + a_i[i] * b_r[i];
        }
    }

    // ========================================
    // Kahan Summation
    // ========================================

    /// SIMD-accelerated Kahan summation for improved numerical accuracy
    #[cfg(target_arch = "x86_64")]
    pub fn simd_kahan_sum_f32(input: &Array<f32>) -> f32 {
        let data = input.to_vec();
        // Use scalar Kahan summation for accuracy
        let mut sum = 0.0f32;
        let mut c = 0.0f32;

        for &value in &data {
            let y = value - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        sum
    }

    /// AVX2 optimized Kahan summation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_kahan_sum_f32(input: &[f32]) -> f32 {
        let len = input.len();
        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum = _mm256_setzero_ps();
        let mut c = _mm256_setzero_ps(); // Compensation

        for i in (0..simd_len).step_by(AVX2_F32_LANES) {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));

            // y = v - c
            let y = _mm256_sub_ps(v, c);
            // t = sum + y
            let t = _mm256_add_ps(sum, y);
            // c = (t - sum) - y
            c = _mm256_sub_ps(_mm256_sub_ps(t, sum), y);
            // sum = t
            sum = t;
        }

        // Horizontal sum of SIMD vector
        let hi128 = _mm256_extractf128_ps(sum, 1);
        let lo128 = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
        let final_sum = _mm_add_ss(sums, shuf2);

        let mut result = _mm_cvtss_f32(final_sum);
        let mut c_scalar = 0.0f32;

        // Handle remaining elements with scalar Kahan
        for i in simd_len..len {
            let y = input[i] - c_scalar;
            let t = result + y;
            c_scalar = (t - result) - y;
            result = t;
        }

        result
    }

    // ========================================
    // Diff and Cumsum Operations
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

    /// AVX2 optimized diff for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_diff_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len() - 1;
        let simd_len = len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + 1));
            let diff = _mm256_sub_pd(x1, x0);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), diff);
        }

        for i in simd_len..len {
            output[i] = input[i + 1] - input[i];
        }
    }

    /// Vectorized cumulative sum for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_cumsum_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        if data.is_empty() {
            return Array::from_vec(vec![]);
        }
        let mut result = vec![0.0f64; data.len()];

        // Cumsum is inherently sequential, but we can use SIMD for partial sums
        // then combine them. For simplicity, use scalar here.
        let mut sum = 0.0;
        for (i, &v) in data.iter().enumerate() {
            sum += v;
            result[i] = sum;
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    // ========================================
    // Linspace and Arange
    // ========================================

    /// Vectorized linspace generation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_linspace_f64(start: f64, stop: f64, num: usize) -> Array<f64> {
        if num == 0 {
            return Array::from_vec(vec![]);
        }
        if num == 1 {
            return Array::from_vec(vec![start]);
        }

        let mut result = vec![0.0f64; num];
        let step = (stop - start) / (num - 1) as f64;

        unsafe {
            Self::avx2_linspace_f64(start, step, &mut result);
        }

        Array::from_vec(result)
    }

    /// AVX2 optimized linspace
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_linspace_f64(start: f64, step: f64, output: &mut [f64]) {
        let len = output.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);

        let start_vec = _mm256_set1_pd(start);
        let step_vec = _mm256_set1_pd(step);
        let indices_step = _mm256_set1_pd(AVX2_F64_LANES as f64);

        // Initial indices: [0, 1, 2, 3]
        let mut indices = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            // value = start + indices * step
            let values = _mm256_fmadd_pd(indices, step_vec, start_vec);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), values);
            indices = _mm256_add_pd(indices, indices_step);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = start + i as f64 * step;
        }
    }

    /// Vectorized arange generation
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_arange_f64(start: f64, stop: f64, step: f64) -> Array<f64> {
        if step == 0.0 || (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Array::from_vec(vec![]);
        }

        let num = ((stop - start) / step).ceil() as usize;
        let mut result = vec![0.0f64; num];

        unsafe {
            Self::avx2_arange_f64(start, step, &mut result);
        }

        Array::from_vec(result)
    }

    /// AVX2 optimized arange
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_arange_f64(start: f64, step: f64, output: &mut [f64]) {
        Self::avx2_linspace_f64(start, step, output);
    }

    // ========================================
    // Gradient Computation
    // ========================================

    /// Vectorized gradient computation for f64
    /// Uses central differences: grad[i] = (x[i+1] - x[i-1]) / 2
    /// For boundaries: forward/backward differences
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_gradient_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let len = data.len();
        if len == 0 {
            return Array::from_vec(vec![]);
        }
        if len == 1 {
            return Array::from_vec(vec![0.0]);
        }

        let mut result = vec![0.0f64; len];

        // Forward difference at start
        result[0] = data[1] - data[0];

        // Central differences for interior
        unsafe {
            Self::avx2_gradient_f64(&data, &mut result);
        }

        // Backward difference at end
        result[len - 1] = data[len - 1] - data[len - 2];

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized gradient (central differences)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_gradient_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        if len < 3 {
            return;
        }

        let half = _mm256_set1_pd(0.5);
        let interior_len = len - 2;
        let simd_len = interior_len & !(AVX2_F64_LANES - 1);

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let idx = i + 1; // Start from index 1 for central diff
            let prev = _mm256_loadu_pd(input.as_ptr().add(idx - 1));
            let next = _mm256_loadu_pd(input.as_ptr().add(idx + 1));
            let grad = _mm256_mul_pd(_mm256_sub_pd(next, prev), half);
            _mm256_storeu_pd(output.as_mut_ptr().add(idx), grad);
        }

        // Handle remaining interior elements
        for i in (simd_len + 1)..(len - 1) {
            output[i] = (input[i + 1] - input[i - 1]) * 0.5;
        }
    }

    // ========================================
    // Memory Copy Optimization
    // ========================================

    /// Optimized memory copy for f32 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn simd_copy_f32(src: &Array<f32>) -> Array<f32> {
        let src_data = src.to_vec();
        let mut dst = vec![0.0f32; src_data.len()];

        unsafe {
            Self::avx2_copy_f32(&src_data, &mut dst);
        }

        Array::from_vec(dst).reshape(&src.shape())
    }

    /// Optimized memory copy for f64 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn simd_copy_f64(src: &Array<f64>) -> Array<f64> {
        let src_data = src.to_vec();
        let mut dst = vec![0.0f64; src_data.len()];

        unsafe {
            Self::avx2_copy_f64(&src_data, &mut dst);
        }

        Array::from_vec(dst).reshape(&src.shape())
    }

    /// AVX2 optimized copy for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_copy_f32(src: &[f32], dst: &mut [f32]) {
        let len = src.len();
        let simd_len = len & !(4 * AVX2_F32_LANES - 1);

        // Unrolled copy loop
        for i in (0..simd_len).step_by(4 * AVX2_F32_LANES) {
            // Prefetch ahead
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(
                    src.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let v0 = _mm256_loadu_ps(src.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(src.as_ptr().add(i + AVX2_F32_LANES));
            let v2 = _mm256_loadu_ps(src.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let v3 = _mm256_loadu_ps(src.as_ptr().add(i + 3 * AVX2_F32_LANES));

            _mm256_storeu_ps(dst.as_mut_ptr().add(i), v0);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + AVX2_F32_LANES), v1);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 2 * AVX2_F32_LANES), v2);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 3 * AVX2_F32_LANES), v3);
        }

        // Handle remaining elements after SIMD chunks
        dst[simd_len..].copy_from_slice(&src[simd_len..]);
    }

    /// AVX2 optimized copy
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_copy_f64(src: &[f64], dst: &mut [f64]) {
        let len = src.len();
        let simd_len = len & !(4 * AVX2_F64_LANES - 1);

        // Unrolled copy loop
        for i in (0..simd_len).step_by(4 * AVX2_F64_LANES) {
            // Prefetch ahead
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    src.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let v0 = _mm256_loadu_pd(src.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(src.as_ptr().add(i + AVX2_F64_LANES));
            let v2 = _mm256_loadu_pd(src.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let v3 = _mm256_loadu_pd(src.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(dst.as_mut_ptr().add(i), v0);
            _mm256_storeu_pd(dst.as_mut_ptr().add(i + AVX2_F64_LANES), v1);
            _mm256_storeu_pd(dst.as_mut_ptr().add(i + 2 * AVX2_F64_LANES), v2);
            _mm256_storeu_pd(dst.as_mut_ptr().add(i + 3 * AVX2_F64_LANES), v3);
        }

        // Handle remaining elements after SIMD chunks
        dst[simd_len..].copy_from_slice(&src[simd_len..]);
    }
}
