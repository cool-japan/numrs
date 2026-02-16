//! Arithmetic SIMD operations
//!
//! This module provides AVX2 optimized implementations for:
//! - add_arrays_f64: Element-wise addition
//! - sub_arrays_f64: Element-wise subtraction
//! - mul_arrays_f64: Element-wise multiplication
//! - div_arrays_f64: Element-wise division
//! - fma_f64: Fused multiply-add
//! - scalar_mul_f64: Scalar multiplication
//! - square_f64: Element-wise square
//! - reciprocal_f64: Element-wise reciprocal (1/x)
//! - negative_f64: Element-wise negation

use super::{EnhancedSimdOps, AVX2_F64_LANES, PREFETCH_DISTANCE};
use crate::array::Array;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl EnhancedSimdOps {
    // ========================================
    // Binary Array Operations
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

    // ========================================
    // FMA (Fused Multiply-Add) Operations
    // ========================================

    /// Vectorized fused multiply-add for f64: result = a * b + c
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

    /// AVX2 optimized fused multiply-add
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_fma_f64(a: &[f64], b: &[f64], c: &[f64], output: &mut [f64]) {
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

            let c0 = _mm256_loadu_pd(c.as_ptr().add(i));
            let c1 = _mm256_loadu_pd(c.as_ptr().add(i + AVX2_F64_LANES));
            let c2 = _mm256_loadu_pd(c.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let c3 = _mm256_loadu_pd(c.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_fmadd_pd(a0, b0, c0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_fmadd_pd(a1, b1, c1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_fmadd_pd(a2, b2, c2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_fmadd_pd(a3, b3, c3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let av = _mm256_loadu_pd(a.as_ptr().add(i));
            let bv = _mm256_loadu_pd(b.as_ptr().add(i));
            let cv = _mm256_loadu_pd(c.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_fmadd_pd(av, bv, cv));
        }

        for i in simd_len..len {
            output[i] = a[i] * b[i] + c[i];
        }
    }

    // ========================================
    // Scalar Operations
    // ========================================

    /// Vectorized scalar multiplication for f64
    #[cfg(target_arch = "x86_64")]
    pub fn vectorized_scalar_mul_f64(a: &Array<f64>, scalar: f64) -> Array<f64> {
        let a_data = a.to_vec();
        let mut result = vec![0.0f64; a_data.len()];
        unsafe {
            Self::avx2_scalar_mul_f64(&a_data, scalar, &mut result);
        }
        Array::from_vec(result).reshape(&a.shape())
    }

    /// AVX2 optimized scalar multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_scalar_mul_f64(a: &[f64], scalar: f64, output: &mut [f64]) {
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
    // Unary Operations
    // ========================================

    /// Vectorized square function for f64 (x * x)
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
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            if i + PREFETCH_DISTANCE / 2 < len {
                _mm_prefetch(
                    input.as_ptr().add(i + PREFETCH_DISTANCE / 2) as *const i8,
                    _MM_HINT_T0,
                );
            }

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

    /// AVX2 optimized negation for f64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_negative_f64(input: &[f64], output: &mut [f64]) {
        let len = input.len();
        let simd_len = len & !(AVX2_F64_LANES - 1);
        let zero = _mm256_setzero_pd();
        let unroll_len = simd_len & !(4 * AVX2_F64_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F64_LANES) {
            let x0 = _mm256_loadu_pd(input.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(input.as_ptr().add(i + AVX2_F64_LANES));
            let x2 = _mm256_loadu_pd(input.as_ptr().add(i + 2 * AVX2_F64_LANES));
            let x3 = _mm256_loadu_pd(input.as_ptr().add(i + 3 * AVX2_F64_LANES));

            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_sub_pd(zero, x0));
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + AVX2_F64_LANES),
                _mm256_sub_pd(zero, x1),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 2 * AVX2_F64_LANES),
                _mm256_sub_pd(zero, x2),
            );
            _mm256_storeu_pd(
                output.as_mut_ptr().add(i + 3 * AVX2_F64_LANES),
                _mm256_sub_pd(zero, x3),
            );
        }

        for i in (unroll_len..simd_len).step_by(AVX2_F64_LANES) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            _mm256_storeu_pd(output.as_mut_ptr().add(i), _mm256_sub_pd(zero, x));
        }

        for i in simd_len..len {
            output[i] = -input[i];
        }
    }
}
