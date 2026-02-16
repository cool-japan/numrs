//! Comparison and utility SIMD operations
//!
//! This module provides AVX2 optimized implementations for:
//! - abs_f32, abs_f64: Absolute value
//! - sign_f64: Sign function
//! - clip_f32, clip_f64: Clamp values to range
//! - copysign_f64: Copy sign from one value to another
//! - maximum_f64, minimum_f64: Element-wise max/min
//! - floor_f64, ceil_f64, round_f64, trunc_f64: Rounding functions
//! - degrees_f64, radians_f64: Angle conversions
//! - hypot_f64: Hypotenuse calculation

use super::{EnhancedSimdOps, AVX2_F32_LANES, AVX2_F64_LANES, PREFETCH_DISTANCE};
use crate::array::Array;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl EnhancedSimdOps {
    // ========================================
    // Absolute Value Functions
    // ========================================

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

        // Mask to clear sign bit
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

    // ========================================
    // Sign Function
    // ========================================

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

    // ========================================
    // Clip Functions
    // ========================================

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
    // Copysign Function
    // ========================================

    /// Vectorized copysign function for f64
    /// Returns |x| with the sign of y
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
    // Element-wise Min/Max Functions
    // ========================================

    /// Vectorized element-wise maximum of two f64 arrays
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

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_max_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = a[i].max(b[i]);
        }
    }

    /// Vectorized element-wise minimum of two f64 arrays
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

        for i in (0..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_min_pd(av, bv));
        }

        for i in simd_len..len {
            output[i] = a[i].min(b[i]);
        }
    }

    // ========================================
    // Rounding Functions
    // ========================================

    /// Vectorized floor function for f64 (round toward -infinity)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_floor_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_floor_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized floor for f64
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
            // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC = 0x09
            let result = _mm256_round_pd(x, 0x09);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].floor();
        }
    }

    /// Vectorized ceil function for f64 (round toward +infinity)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_ceil_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_ceil_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized ceil for f64
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
            // _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC = 0x0A
            let result = _mm256_round_pd(x, 0x0A);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].ceil();
        }
    }

    /// Vectorized round function for f64 (round to nearest)
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_round_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_round_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized round for f64
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
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_trunc_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];

        unsafe {
            Self::avx2_trunc_f64(&data, &mut result);
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// AVX2 optimized trunc for f64
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

    // ========================================
    // Angle Conversion Functions
    // ========================================

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

        // 180 / pi
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

        // pi / 180
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

    // ========================================
    // Hypotenuse Function
    // ========================================

    /// Vectorized hypot function for f64 (hypotenuse: sqrt(x^2 + y^2))
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

            // sqrt(x^2 + y^2) using FMA
            let x_sq = _mm256_mul_pd(x_vec, x_vec);
            let sum_sq = _mm256_fmadd_pd(y_vec, y_vec, x_sq);
            let result = _mm256_sqrt_pd(sum_sq);

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
        }
    }
}
