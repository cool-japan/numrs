//! Enhanced AVX-512 SIMD operations for latest CPU architectures
//!
//! This module provides cutting-edge vectorization using stable AVX2 instructions
//! for maximum performance on modern Intel and AMD processors.
//!
//! Note: AVX-512 features are currently unstable in Rust, so this module
//! provides production-ready implementations using stable AVX2 instructions.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

// AVX512 operations - currently using stable AVX2 implementations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 vectorization constants for production stability
#[allow(dead_code)]
const AVX2_F32_LANES: usize = 8;
#[allow(dead_code)]
const AVX2_F64_LANES: usize = 4;
#[allow(dead_code)]
const AVX2_ALIGNMENT: usize = 32;

/// Advanced AVX2 operations with maximum vectorization
pub struct Avx2EnhancedOps;

impl Avx2EnhancedOps {
    /// AVX2 optimized matrix multiplication with 256-bit vectors
    #[cfg(target_arch = "x86_64")]
    pub fn avx2_matmul_f32(
        a: &Array<f32>,
        b: &Array<f32>,
        c: &mut Array<f32>,
        tile_size: usize,
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
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Matrix dimensions mismatch: A is {}x{}, B is {}x{}",
                m, k, k2, n
            )));
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let mut c_data = vec![0.0f32; m * n];

        if is_x86_feature_detected!("avx2") {
            unsafe {
                Self::tiled_matmul_avx2_f32(&a_data, &b_data, &mut c_data, m, n, k, tile_size);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += a_data[i * k + l] * b_data[l * n + j];
                    }
                    c_data[i * n + j] = sum;
                }
            }
        }

        *c = Array::from_vec(c_data).reshape(&[m, n]);
        Ok(())
    }

    /// Tiled matrix multiplication with AVX2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn tiled_matmul_avx2_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        tile_size: usize,
    ) {
        for ii in (0..m).step_by(tile_size) {
            for jj in (0..n).step_by(tile_size) {
                for kk in (0..k).step_by(tile_size) {
                    let i_end = (ii + tile_size).min(m);
                    let j_end = (jj + tile_size).min(n);
                    let k_end = (kk + tile_size).min(k);

                    for i in ii..i_end {
                        for j in (jj..j_end).step_by(AVX2_F32_LANES) {
                            let lanes = (j_end - j).min(AVX2_F32_LANES);

                            // Load C values using AVX2
                            let mut vc = if lanes == AVX2_F32_LANES {
                                _mm256_loadu_ps(c.as_ptr().add(i * n + j))
                            } else {
                                // Handle partial loads for AVX2
                                let mut values = [0.0f32; 8];
                                for idx in 0..lanes {
                                    values[idx] = *c.get_unchecked(i * n + j + idx);
                                }
                                _mm256_loadu_ps(values.as_ptr())
                            };

                            for l in kk..k_end {
                                let va = _mm256_set1_ps(a[i * k + l]);
                                let vb = if lanes == AVX2_F32_LANES {
                                    _mm256_loadu_ps(b.as_ptr().add(l * n + j))
                                } else {
                                    let mut values = [0.0f32; 8];
                                    for idx in 0..lanes {
                                        values[idx] = *b.get_unchecked(l * n + j + idx);
                                    }
                                    _mm256_loadu_ps(values.as_ptr())
                                };

                                vc = _mm256_fmadd_ps(va, vb, vc);
                            }

                            // Store results
                            if lanes == AVX2_F32_LANES {
                                _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), vc);
                            } else {
                                let mut values = [0.0f32; 8];
                                _mm256_storeu_ps(values.as_mut_ptr(), vc);
                                for idx in 0..lanes {
                                    *c.get_unchecked_mut(i * n + j + idx) = values[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// AVX2 optimized element-wise operations
    #[cfg(target_arch = "x86_64")]
    pub fn avx2_add_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
        if a.shape() != b.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Arrays must have the same shape".to_string(),
            ));
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let mut result = vec![0.0f32; a_data.len()];

        if is_x86_feature_detected!("avx2") {
            unsafe {
                Self::vectorized_add_avx2(&a_data, &b_data, &mut result);
            }
        } else {
            for i in 0..a_data.len() {
                result[i] = a_data[i] + b_data[i];
            }
        }

        Ok(Array::from_vec(result).reshape(&a.shape()))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vectorized_add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let vectorizable_len = len & !(AVX2_F32_LANES - 1);

        // Process AVX2_F32_LANES elements at a time
        for i in (0..vectorizable_len).step_by(AVX2_F32_LANES) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vresult = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
        }

        // Handle remaining elements
        for i in vectorizable_len..len {
            result[i] = a[i] + b[i];
        }
    }

    /// AVX2 optimized dot product
    #[cfg(target_arch = "x86_64")]
    pub fn avx2_dot_f32(a: &Array<f32>, b: &Array<f32>) -> Result<f32> {
        if a.shape() != b.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Arrays must have the same shape".to_string(),
            ));
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();

        if is_x86_feature_detected!("avx2") {
            unsafe { Ok(Self::vectorized_dot_avx2(&a_data, &b_data)) }
        } else {
            let mut sum = 0.0f32;
            for i in 0..a_data.len() {
                sum += a_data[i] * b_data[i];
            }
            Ok(sum)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vectorized_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let vectorizable_len = len & !(AVX2_F32_LANES - 1);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Process 4 AVX2 vectors at a time for better ILP
        let unroll_len = vectorizable_len & !(4 * AVX2_F32_LANES - 1);

        for i in (0..unroll_len).step_by(4 * AVX2_F32_LANES) {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + AVX2_F32_LANES));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + AVX2_F32_LANES));
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

            let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 2 * AVX2_F32_LANES));
            let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 2 * AVX2_F32_LANES));
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

            let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 3 * AVX2_F32_LANES));
            let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 3 * AVX2_F32_LANES));
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        }

        // Process remaining vectorizable elements
        for i in (unroll_len..vectorizable_len).step_by(AVX2_F32_LANES) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
        }

        // Combine accumulators
        let combined01 = _mm256_add_ps(acc0, acc1);
        let combined23 = _mm256_add_ps(acc2, acc3);
        let total = _mm256_add_ps(combined01, combined23);

        // Horizontal sum
        let sum_vec = _mm256_hadd_ps(total, total);
        let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        let low = _mm256_castps256_ps128(sum_vec);
        let high = _mm256_extractf128_ps(sum_vec, 1);
        let final_sum = _mm_add_ps(low, high);
        let mut result = _mm_cvtss_f32(final_sum);

        // Handle remaining elements
        for i in vectorizable_len..len {
            result += a[i] * b[i];
        }

        result
    }

    /// AVX2 optimized convolution
    #[cfg(target_arch = "x86_64")]
    pub fn avx2_convolution_f32(signal: &Array<f32>, kernel: &Array<f32>) -> Result<Array<f32>> {
        let signal_data = signal.to_vec();
        let kernel_data = kernel.to_vec();
        let signal_len = signal_data.len();
        let kernel_len = kernel_data.len();

        if kernel_len > signal_len {
            return Err(NumRs2Error::DimensionMismatch(
                "Kernel cannot be larger than signal".to_string(),
            ));
        }

        let output_len = signal_len - kernel_len + 1;
        let mut result = vec![0.0f32; output_len];

        if is_x86_feature_detected!("avx2") {
            unsafe {
                Self::vectorized_convolution_avx2(&signal_data, &kernel_data, &mut result);
            }
        } else {
            // Scalar fallback
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    sum += signal_data[i + j] * kernel_data[j];
                }
                result[i] = sum;
            }
        }

        Ok(Array::from_vec(result).reshape(&[output_len]))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vectorized_convolution_avx2(signal: &[f32], kernel: &[f32], result: &mut [f32]) {
        let _signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = result.len();

        for i in 0..output_len {
            let mut sum = _mm256_setzero_ps();
            let mut scalar_sum = 0.0f32;

            // Vectorized part
            let start_k = 0;
            let end_k = kernel_len;
            let vectorizable_len = (end_k - start_k) & !(AVX2_F32_LANES - 1);

            for k in (start_k..start_k + vectorizable_len).step_by(AVX2_F32_LANES) {
                let sig_vals =
                    std::slice::from_raw_parts(signal.as_ptr().add(i + k), AVX2_F32_LANES);
                let sig_vec = _mm256_loadu_ps(sig_vals.as_ptr());

                let kern_vec = _mm256_loadu_ps(kernel.as_ptr().add(k));
                sum = _mm256_fmadd_ps(sig_vec, kern_vec, sum);
            }

            // Horizontal sum of vector
            let sum_vec = _mm256_hadd_ps(sum, sum);
            let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            let low = _mm256_castps256_ps128(sum_vec);
            let high = _mm256_extractf128_ps(sum_vec, 1);
            let final_sum = _mm_add_ps(low, high);
            let result_val = _mm_cvtss_f32(final_sum);

            // Handle remaining elements
            for k in start_k + vectorizable_len..end_k {
                scalar_sum += signal[i + k] * kernel[k];
            }

            result[i] = result_val + scalar_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::array::Array;
    #[cfg(target_arch = "x86_64")]
    use approx::assert_relative_eq;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matrix_multiplication() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Array::from_vec(a_data).reshape(&[2, 3]);

        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = Array::from_vec(b_data).reshape(&[3, 2]);

        let mut c = Array::zeros(&[2, 2]);

        Avx2EnhancedOps::avx2_matmul_f32(&a, &b, &mut c, 32).unwrap();

        // Expected result: [[58, 64], [139, 154]]
        let c_data = c.to_vec();
        assert_relative_eq!(c_data[0], 58.0, epsilon = 1e-5);
        assert_relative_eq!(c_data[1], 64.0, epsilon = 1e-5);
        assert_relative_eq!(c_data[2], 139.0, epsilon = 1e-5);
        assert_relative_eq!(c_data[3], 154.0, epsilon = 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_add() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a = Array::from_vec(a_data).reshape(&[3, 3]);

        let b_data = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let b = Array::from_vec(b_data).reshape(&[3, 3]);

        let result = Avx2EnhancedOps::avx2_add_f32(&a, &b).unwrap();
        let result_data = result.to_vec();

        for val in result_data {
            assert_relative_eq!(val, 10.0, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_dot_product() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let a = Array::from_vec(a_data).reshape(&[4]);

        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let b = Array::from_vec(b_data).reshape(&[4]);

        let result = Avx2EnhancedOps::avx2_dot_f32(&a, &b).unwrap();

        // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_relative_eq!(result, 70.0, epsilon = 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_convolution() {
        let signal_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal = Array::from_vec(signal_data).reshape(&[5]);

        let kernel_data = vec![0.5, 0.5];
        let kernel = Array::from_vec(kernel_data).reshape(&[2]);

        let result = Avx2EnhancedOps::avx2_convolution_f32(&signal, &kernel).unwrap();
        let result_data = result.to_vec();

        // Expected: [1.5, 2.5, 3.5, 4.5]
        assert_eq!(result_data.len(), 4);
        assert_relative_eq!(result_data[0], 1.5, epsilon = 1e-5);
        assert_relative_eq!(result_data[1], 2.5, epsilon = 1e-5);
        assert_relative_eq!(result_data[2], 3.5, epsilon = 1e-5);
        assert_relative_eq!(result_data[3], 4.5, epsilon = 1e-5);
    }
}
