//! Matrix multiplication operations using NEON SIMD
//!
//! This module provides optimized matrix multiplication for ARM NEON.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

use super::core::{NeonEnhancedOps, NEON_F32_LANES};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

impl NeonEnhancedOps {
    /// NEON optimized matrix multiplication
    #[cfg(target_arch = "aarch64")]
    pub fn neon_matmul_f32(
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
            Self::blocked_matmul_neon_f32(&a_data, &b_data, &mut c_data, m, n, k, block_size);
        }

        *c = Array::from_vec(c_data).reshape(&[m, n]);
        Ok(())
    }

    /// Blocked matrix multiplication with NEON optimization
    #[cfg(target_arch = "aarch64")]
    unsafe fn blocked_matmul_neon_f32(
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
                        for j in (jj..j_end).step_by(NEON_F32_LANES) {
                            let lanes = (j_end - j).min(NEON_F32_LANES);

                            // Load C values
                            let mut vc = if lanes == NEON_F32_LANES {
                                vld1q_f32(c.as_ptr().add(i * n + j))
                            } else {
                                let mut temp = [0.0f32; NEON_F32_LANES];
                                for l in 0..lanes {
                                    temp[l] = c[i * n + j + l];
                                }
                                vld1q_f32(temp.as_ptr())
                            };

                            for l in kk..k_end {
                                let va = vdupq_n_f32(a[i * k + l]);
                                let vb = if lanes == NEON_F32_LANES {
                                    vld1q_f32(b.as_ptr().add(l * n + j))
                                } else {
                                    let mut temp = [0.0f32; NEON_F32_LANES];
                                    for idx in 0..lanes {
                                        temp[idx] = b[l * n + j + idx];
                                    }
                                    vld1q_f32(temp.as_ptr())
                                };
                                vc = vfmaq_f32(vc, va, vb);
                            }

                            // Store C values
                            if lanes == NEON_F32_LANES {
                                vst1q_f32(c.as_mut_ptr().add(i * n + j), vc);
                            } else {
                                let mut temp = [0.0f32; NEON_F32_LANES];
                                vst1q_f32(temp.as_mut_ptr(), vc);
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

    /// NEON memory copy optimization
    #[cfg(target_arch = "aarch64")]
    pub fn neon_copy_f32(src: &Array<f32>, dst: &mut Array<f32>) -> Result<()> {
        if src.shape() != dst.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: src.shape(),
                actual: dst.shape(),
            });
        }

        let src_data = src.to_vec();
        let mut dst_data = dst.to_vec();

        unsafe {
            Self::optimized_copy_neon_f32(&src_data, &mut dst_data);
        }

        *dst = Array::from_vec(dst_data).reshape(&src.shape());
        Ok(())
    }

    /// NEON optimized memory copy
    #[cfg(target_arch = "aarch64")]
    unsafe fn optimized_copy_neon_f32(src: &[f32], dst: &mut [f32]) {
        let len = src.len();
        let simd_len = len & !(NEON_F32_LANES * 4 - 1);

        // Copy 16 elements at a time for better throughput
        for i in (0..simd_len).step_by(NEON_F32_LANES * 4) {
            let v0 = vld1q_f32(src.as_ptr().add(i));
            let v1 = vld1q_f32(src.as_ptr().add(i + NEON_F32_LANES));
            let v2 = vld1q_f32(src.as_ptr().add(i + NEON_F32_LANES * 2));
            let v3 = vld1q_f32(src.as_ptr().add(i + NEON_F32_LANES * 3));

            vst1q_f32(dst.as_mut_ptr().add(i), v0);
            vst1q_f32(dst.as_mut_ptr().add(i + NEON_F32_LANES), v1);
            vst1q_f32(dst.as_mut_ptr().add(i + NEON_F32_LANES * 2), v2);
            vst1q_f32(dst.as_mut_ptr().add(i + NEON_F32_LANES * 3), v3);
        }

        // Handle remaining elements
        dst[simd_len..len].copy_from_slice(&src[simd_len..len]);
    }
}

// Provide no-op implementations for non-ARM architectures
#[cfg(not(target_arch = "aarch64"))]
impl NeonEnhancedOps {
    pub fn neon_matmul_f32(
        a: &Array<f32>,
        b: &Array<f32>,
        c: &mut Array<f32>,
        _block_size: usize,
    ) -> Result<()> {
        // Fallback to regular matrix multiplication
        let result = a.matmul(b)?;
        *c = result;
        Ok(())
    }

    pub fn neon_copy_f32(src: &Array<f32>, dst: &mut Array<f32>) -> Result<()> {
        *dst = src.clone();
        Ok(())
    }
}
