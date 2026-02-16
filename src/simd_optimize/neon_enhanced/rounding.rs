//! Rounding operations using NEON SIMD
//!
//! This module provides optimized floor, ceil, and round functions
//! for ARM NEON.

use crate::array::Array;

use super::core::{NeonEnhancedOps, NEON_F64_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// NEON f64 Rounding Operations
// =============================================================================

impl NeonEnhancedOps {
    /// NEON vectorized floor for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_floor_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let floor_v = vrndmq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), floor_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].floor();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized ceiling for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_ceil_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let ceil_v = vrndpq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), ceil_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].ceil();
        }

        Array::from_vec(result).reshape(&input.shape())
    }

    /// NEON vectorized round for f64
    #[cfg(target_arch = "aarch64")]
    pub fn vectorized_round_f64(input: &Array<f64>) -> Array<f64> {
        let data = input.to_vec();
        let mut result = vec![0.0f64; data.len()];
        let len = data.len();
        let simd_len = len & !(NEON_F64_LANES - 1);

        unsafe {
            for i in (0..simd_len).step_by(NEON_F64_LANES) {
                let v = vld1q_f64(data.as_ptr().add(i));
                let round_v = vrndnq_f64(v);
                vst1q_f64(result.as_mut_ptr().add(i), round_v);
            }
        }

        for i in simd_len..len {
            result[i] = data[i].round();
        }

        Array::from_vec(result).reshape(&input.shape())
    }
}

// =============================================================================
// Non-aarch64 Fallback Implementations
// =============================================================================

#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn vectorized_floor_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.floor())
    }

    pub fn vectorized_ceil_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.ceil())
    }

    pub fn vectorized_round_f64(input: &Array<f64>) -> Array<f64> {
        input.map(|x| x.round())
    }
}
