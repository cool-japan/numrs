//! Reduction SIMD operations
//!
//! This module provides AVX2 optimized implementations for:
//! - sum_f32, sum_f64: Sum reduction
//! - product_f64: Product reduction
//! - min_f32, min_f64: Minimum reduction
//! - max_f32, max_f64: Maximum reduction
//! - dot_f32, dot_f64: Dot product
//! - norm_l1_f32, norm_l1_f64: L1 norm (Manhattan)
//! - norm_l2_f32, norm_l2_f64: L2 norm (Euclidean)
//! - norm_inf_f64: L-infinity norm
//! - variance_f64: Variance
//! - std_f64: Standard deviation
//! - mean_f64: Mean

use super::EnhancedSimdOps;
// See `arithmetic.rs` for why these are gated the same as the intrinsics.
#[cfg(target_arch = "x86_64")]
use super::{AVX2_F32_LANES, AVX2_F64_LANES};
#[cfg(target_arch = "x86_64")]
use crate::array::Array;
#[cfg(target_arch = "x86_64")]
use crate::error::Result;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl EnhancedSimdOps {
    // ========================================
    // Sum Reductions
    // ========================================

    /// Vectorized sum reduction for f64
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

        // Horizontal sum
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

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(sum0, 1);
        let lo128 = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
        let final_sum = _mm_add_ss(sums, shuf2);

        let mut result = _mm_cvtss_f32(final_sum);

        for i in simd_len..len {
            result += input[i];
        }

        result
    }

    // ========================================
    // Product Reduction
    // ========================================

    /// Vectorized product reduction for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_product_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        unsafe { Self::avx2_product_f64(&data) }
    }

    /// AVX2 optimized product reduction for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_product_f64(input: &[f64]) -> f64 {
        let len = input.len();
        if len == 0 {
            return 1.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);

        let mut prod0 = _mm256_set1_pd(1.0);
        let mut prod1 = _mm256_set1_pd(1.0);

        let unroll_len = simd_len & !(2 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(2 * AVX2_F64_LANES) {
            let v0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));

            prod0 = _mm256_mul_pd(prod0, v0);
            prod1 = _mm256_mul_pd(prod1, v1);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            prod0 = _mm256_mul_pd(prod0, v);
        }

        // Combine accumulators
        prod0 = _mm256_mul_pd(prod0, prod1);

        // Horizontal product
        let prod_low = _mm256_extractf128_pd(prod0, 0);
        let prod_high = _mm256_extractf128_pd(prod0, 1);
        let prod_128 = _mm_mul_pd(prod_low, prod_high);
        let prod_shuffle = _mm_shuffle_pd(prod_128, prod_128, 1);
        let prod_final = _mm_mul_pd(prod_128, prod_shuffle);

        let mut result = _mm_cvtsd_f64(prod_final);

        for i in simd_len..len {
            result *= input[i];
        }

        result
    }

    // ========================================
    // Min/Max Reductions
    // ========================================

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

        let mut min0 = _mm256_set1_pd(input[0]);
        let mut min1 = min0;
        let mut min2 = min0;
        let mut min3 = min0;

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

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

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            min0 = _mm256_min_pd(min0, v);
        }

        // Combine accumulators
        min0 = _mm256_min_pd(min0, min1);
        min2 = _mm256_min_pd(min2, min3);
        min0 = _mm256_min_pd(min0, min2);

        // Horizontal min
        let min_low = _mm256_extractf128_pd(min0, 0);
        let min_high = _mm256_extractf128_pd(min0, 1);
        let min_128 = _mm_min_pd(min_low, min_high);
        let min_shuffle = _mm_shuffle_pd(min_128, min_128, 1);
        let min_final = _mm_min_pd(min_128, min_shuffle);

        let mut result = _mm_cvtsd_f64(min_final);

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

        let mut max0 = _mm256_set1_pd(input[0]);
        let mut max1 = max0;
        let mut max2 = max0;
        let mut max3 = max0;

        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

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

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let v = _mm256_loadu_pd(input.as_ptr().add(i));
            max0 = _mm256_max_pd(max0, v);
        }

        // Combine accumulators
        max0 = _mm256_max_pd(max0, max1);
        max2 = _mm256_max_pd(max2, max3);
        max0 = _mm256_max_pd(max0, max2);

        // Horizontal max
        let max_low = _mm256_extractf128_pd(max0, 0);
        let max_high = _mm256_extractf128_pd(max0, 1);
        let max_128 = _mm_max_pd(max_low, max_high);
        let max_shuffle = _mm_shuffle_pd(max_128, max_128, 1);
        let max_final = _mm_max_pd(max_128, max_shuffle);

        let mut result = _mm_cvtsd_f64(max_final);

        for i in simd_len..len {
            if input[i] > result {
                result = input[i];
            }
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

    // ========================================
    // Dot Product
    // ========================================

    /// Vectorized dot product for f64 arrays
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

    /// Vectorized dot product for f32 arrays
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_dot_f32(a: &Array<f32>, b: &Array<f32>) -> Result<f32> {
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let len = a_data.len().min(b_data.len());
        Ok(unsafe { Self::avx2_dot_f32(&a_data[..len], &b_data[..len]) })
    }

    /// AVX2 optimized dot product for f32 using FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let unroll_len = simd_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + AVX2_F32_LANES));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 3 * AVX2_F32_LANES));

            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(i + AVX2_F32_LANES));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 3 * AVX2_F32_LANES));

            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F32_LANES) {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(av, bv, sum0);
        }

        // Combine accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(sum0, 1);
        let lo128 = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
        let final_sum = _mm_add_ss(sums, shuf2);

        let mut result = _mm_cvtss_f32(final_sum);

        for i in simd_len..len {
            result += a[i] * b[i];
        }

        result
    }

    // ========================================
    // Norm Functions
    // ========================================

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

        for i in simd_len..len {
            result += a[i].abs();
        }

        result
    }

    /// Vectorized L2 norm for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_l2_f32(a: &Array<f32>) -> f32 {
        let data = a.to_vec();
        unsafe { Self::avx2_norm_l2_f32(&data) }
    }

    /// AVX2 optimized L2 norm for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_norm_l2_f32(a: &[f32]) -> f32 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F32_LANES - 1);

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let unroll_len = simd_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + AVX2_F32_LANES));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 3 * AVX2_F32_LANES));

            sum0 = _mm256_fmadd_ps(a0, a0, sum0);
            sum1 = _mm256_fmadd_ps(a1, a1, sum1);
            sum2 = _mm256_fmadd_ps(a2, a2, sum2);
            sum3 = _mm256_fmadd_ps(a3, a3, sum3);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F32_LANES) {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(av, av, sum0);
        }

        // Combine accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(sum0, 1);
        let lo128 = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
        let final_sum = _mm_add_ss(sums, shuf2);

        let mut sum_sq = _mm_cvtss_f32(final_sum);

        for i in simd_len..len {
            sum_sq += a[i] * a[i];
        }

        sum_sq.sqrt()
    }

    /// Vectorized L1 norm for f32
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_l1_f32(a: &Array<f32>) -> f32 {
        let data = a.to_vec();
        unsafe { Self::avx2_norm_l1_f32(&data) }
    }

    /// AVX2 optimized L1 norm for f32
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_norm_l1_f32(a: &[f32]) -> f32 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F32_LANES - 1);
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let unroll_len = simd_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + AVX2_F32_LANES));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 3 * AVX2_F32_LANES));

            sum0 = _mm256_add_ps(sum0, _mm256_and_ps(a0, abs_mask));
            sum1 = _mm256_add_ps(sum1, _mm256_and_ps(a1, abs_mask));
            sum2 = _mm256_add_ps(sum2, _mm256_and_ps(a2, abs_mask));
            sum3 = _mm256_add_ps(sum3, _mm256_and_ps(a3, abs_mask));
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F32_LANES) {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            sum0 = _mm256_add_ps(sum0, _mm256_and_ps(av, abs_mask));
        }

        // Combine accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Horizontal sum
        let hi128 = _mm256_extractf128_ps(sum0, 1);
        let lo128 = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(hi128, lo128);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
        let final_sum = _mm_add_ss(sums, shuf2);

        let mut result = _mm_cvtss_f32(final_sum);

        for i in simd_len..len {
            result += a[i].abs();
        }

        result
    }

    /// Vectorized L-infinity norm for f64 (maximum absolute value)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_norm_inf_f64(a: &Array<f64>) -> f64 {
        let data = a.to_vec();
        unsafe { Self::avx2_norm_inf_f64(&data) }
    }

    /// AVX2 optimized L-infinity norm
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_norm_inf_f64(a: &[f64]) -> f64 {
        let len = a.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);
        let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));

        let mut max0 = _mm256_setzero_pd();
        let mut max1 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(2 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(2 * AVX2_F64_LANES) {
            let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let a1 = _mm256_loadu_pd(a.as_ptr().add(i + AVX2_F64_LANES));

            max0 = _mm256_max_pd(max0, _mm256_and_pd(a0, abs_mask));
            max1 = _mm256_max_pd(max1, _mm256_and_pd(a1, abs_mask));
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            max0 = _mm256_max_pd(max0, _mm256_and_pd(av, abs_mask));
        }

        // Combine accumulators
        max0 = _mm256_max_pd(max0, max1);

        // Horizontal max
        let max_low = _mm256_extractf128_pd(max0, 0);
        let max_high = _mm256_extractf128_pd(max0, 1);
        let max_128 = _mm_max_pd(max_low, max_high);
        let max_shuffle = _mm_shuffle_pd(max_128, max_128, 1);
        let max_final = _mm_max_pd(max_128, max_shuffle);

        let mut result = _mm_cvtsd_f64(max_final);

        for i in simd_len..len {
            let abs_val = a[i].abs();
            if abs_val > result {
                result = abs_val;
            }
        }

        result
    }

    // ========================================
    // Statistical Functions
    // ========================================

    /// Vectorized variance calculation for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_variance_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        let n = data.len();
        if n == 0 {
            return 0.0;
        }

        let mean = Self::vectorized_sum_f64(input) / n as f64;
        unsafe { Self::avx2_variance_f64(&data, mean) }
    }

    /// AVX2 optimized variance calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_variance_f64(input: &[f64], mean: f64) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = len & !(AVX2_F64_LANES - 1);
        let mean_vec = _mm256_set1_pd(mean);

        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();

        let unroll_len = simd_len & !(2 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(2 * AVX2_F64_LANES) {
            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));

            let diff0 = _mm256_sub_pd(x0, mean_vec);
            let diff1 = _mm256_sub_pd(x1, mean_vec);

            sum0 = _mm256_fmadd_pd(diff0, diff0, sum0);
            sum1 = _mm256_fmadd_pd(diff1, diff1, sum1);
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let diff = _mm256_sub_pd(x, mean_vec);
            sum0 = _mm256_fmadd_pd(diff, diff, sum0);
        }

        // Combine accumulators
        sum0 = _mm256_add_pd(sum0, sum1);

        // Horizontal sum
        let sum_low = _mm256_extractf128_pd(sum0, 0);
        let sum_high = _mm256_extractf128_pd(sum0, 1);
        let sum_128 = _mm_add_pd(sum_low, sum_high);
        let sum_final = _mm_hadd_pd(sum_128, sum_128);

        let mut result = _mm_cvtsd_f64(sum_final);

        for i in simd_len..len {
            let diff = input[i] - mean;
            result += diff * diff;
        }

        result / len as f64
    }

    /// Vectorized standard deviation for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_std_f64(input: &Array<f64>) -> f64 {
        Self::vectorized_variance_f64(input).sqrt()
    }

    /// Vectorized mean for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_mean_f64(input: &Array<f64>) -> f64 {
        let data = input.to_vec();
        if data.is_empty() {
            return 0.0;
        }
        Self::vectorized_sum_f64(input) / data.len() as f64
    }
}
