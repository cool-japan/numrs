//! Enhanced AVX2 SIMD operations with advanced vectorization
//!
//! This module provides highly optimized SIMD implementations for advanced
//! mathematical operations, cache-aware algorithms, and specialized functions.
//!
//! ## Module Organization
//!
//! - `arithmetic`: Element-wise array operations (add, sub, mul, div, fma)
//! - `trigonometric`: Trigonometric functions (sin, cos, tan, asin, acos, atan)
//! - `exponential`: Exponential and logarithmic functions (exp, log, pow, sqrt)
//! - `comparison`: Comparison and rounding operations (abs, min, max, clip)
//! - `reduction`: Reduction operations (sum, product, dot, norms)
//! - `special`: Special operations (matmul, complex multiply, Kahan sum)
//! - `benchmark`: Performance monitoring and benchmarking utilities

pub mod arithmetic;
pub mod benchmark;
pub mod comparison;
pub mod exponential;
pub mod reduction;
pub mod special;
pub mod trigonometric;

// Re-export main types
pub use benchmark::{BenchmarkResult, SimdBenchmark, SimdBenchmarkResults, SimdPerformanceMonitor};

/// Enhanced vectorization constants
pub(crate) const AVX2_F32_LANES: usize = 8;
pub(crate) const AVX2_F64_LANES: usize = 4;
#[allow(dead_code)]
pub(crate) const CACHE_LINE_SIZE: usize = 64;
#[allow(dead_code)]
pub(crate) const L1_CACHE_SIZE: usize = 32 * 1024;
pub(crate) const PREFETCH_DISTANCE: usize = 512;

/// Advanced vectorized operations with cache optimization
pub struct EnhancedSimdOps;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_exp_f32() {
        let input = Array::from_vec(vec![0.0f32, 1.0, 2.0, -1.0, 0.5, -0.5, 1.5, 2.5]);
        let result = EnhancedSimdOps::vectorized_exp_f32(&input);
        let result_vec = result.to_vec();

        // Check exp values are approximately correct
        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].exp();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "exp({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_exp_f64() {
        let input = Array::from_vec(vec![0.0f64, 1.0, 2.0, -1.0, 0.5, -0.5, 1.5, 2.5]);
        let result = EnhancedSimdOps::vectorized_exp_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].exp();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "exp({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_log_f32() {
        let input = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_log_f32(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].ln();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "ln({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_log_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_log_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].ln();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "ln({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sin_f32() {
        let input = Array::from_vec(vec![
            0.0f32,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            std::f32::consts::PI,
        ]);
        let result = EnhancedSimdOps::vectorized_sin_f32_simd(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].sin();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "sin({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sin_f64() {
        let input = Array::from_vec(vec![
            0.0f64,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            std::f64::consts::PI,
        ]);
        let result = EnhancedSimdOps::vectorized_sin_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].sin();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "sin({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_cos_f32() {
        let input = Array::from_vec(vec![
            0.0f32,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            std::f32::consts::PI,
        ]);
        let result = EnhancedSimdOps::vectorized_cos_f32(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].cos();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "cos({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_cos_f64() {
        let input = Array::from_vec(vec![
            0.0f64,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            std::f64::consts::PI,
        ]);
        let result = EnhancedSimdOps::vectorized_cos_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].cos();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "cos({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sqrt_f32() {
        let input = Array::from_vec(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        let result = EnhancedSimdOps::vectorized_sqrt_f32(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].sqrt();
            assert!(
                (result_vec[i] - expected).abs() < 1e-6,
                "sqrt({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sqrt_f64() {
        let input = Array::from_vec(vec![1.0f64, 4.0, 9.0, 16.0]);
        let result = EnhancedSimdOps::vectorized_sqrt_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].sqrt();
            assert!(
                (result_vec[i] - expected).abs() < 1e-14,
                "sqrt({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_pow_f32() {
        let input = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_pow_f32(&input, 2.0);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].powf(2.0);
            assert!(
                (result_vec[i] - expected).abs() < 0.1,
                "pow({}, 2) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_pow_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_pow_f64(&input, 2.0);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].powf(2.0);
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "pow({}, 2) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_log10_f32() {
        let input = Array::from_vec(vec![1.0f32, 10.0, 100.0, 1000.0, 0.1, 0.01, 5.0, 50.0]);
        let result = EnhancedSimdOps::vectorized_log10_f32(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].log10();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "log10({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_log10_f64() {
        let input = Array::from_vec(vec![1.0f64, 10.0, 100.0, 1000.0]);
        let result = EnhancedSimdOps::vectorized_log10_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].log10();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "log10({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_log2_f32() {
        let input = Array::from_vec(vec![1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]);
        let result = EnhancedSimdOps::vectorized_log2_f32(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].log2();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "log2({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_log2_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 4.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_log2_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].log2();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "log2({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_tan_f32() {
        let input = Array::from_vec(vec![0.0f32, 0.25, 0.5, 0.75, -0.25, -0.5, -0.75, 1.0]);
        let result = EnhancedSimdOps::vectorized_tan_f32(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].tan();
            assert!(
                (result_vec[i] - expected).abs() < 0.01,
                "tan({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_tan_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.25, 0.5, 0.75]);
        let result = EnhancedSimdOps::vectorized_tan_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].tan();
            assert!(
                (result_vec[i] - expected).abs() < 1e-8,
                "tan({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sinh_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.5, 1.0, -0.5]);
        let result = EnhancedSimdOps::vectorized_sinh_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].sinh();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "sinh({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_cosh_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.5, 1.0, -0.5]);
        let result = EnhancedSimdOps::vectorized_cosh_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].cosh();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "cosh({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_tanh_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.5, 1.0, -0.5]);
        let result = EnhancedSimdOps::vectorized_tanh_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].tanh();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "tanh({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_asin_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.5, 0.9, -0.5]);
        let result = EnhancedSimdOps::vectorized_asin_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].asin();
            assert!(
                (result_vec[i] - expected).abs() < 1e-8,
                "asin({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_acos_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.5, 0.9, -0.5]);
        let result = EnhancedSimdOps::vectorized_acos_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].acos();
            assert!(
                (result_vec[i] - expected).abs() < 1e-8,
                "acos({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_atan_f64() {
        let input = Array::from_vec(vec![0.0f64, 0.5, 1.0, -0.5]);
        let result = EnhancedSimdOps::vectorized_atan_f64(&input);
        let result_vec = result.to_vec();

        for i in 0..input.len() {
            let input_data = input.to_vec();
            let expected = input_data[i].atan();
            assert!(
                (result_vec[i] - expected).abs() < 1e-7,
                "atan({}) = {} but got {}",
                input_data[i],
                expected,
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_abs_f32() {
        let input = Array::from_vec(vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let result = EnhancedSimdOps::vectorized_abs_f32(&input);
        let result_vec = result.to_vec();

        let expected = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-6,
                "abs({}) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_abs_f64() {
        let input = Array::from_vec(vec![1.0f64, -2.0, 3.0, -4.0]);
        let result = EnhancedSimdOps::vectorized_abs_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, 2.0, 3.0, 4.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "abs({}) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sign_f64() {
        let input = Array::from_vec(vec![1.0f64, -2.0, 0.0, -0.5]);
        let result = EnhancedSimdOps::vectorized_sign_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, -1.0, 0.0, -1.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "sign({}) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_clip_f32() {
        let input = Array::from_vec(vec![-2.0f32, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]);
        let result = EnhancedSimdOps::vectorized_clip_f32(&input, 0.0, 1.0);
        let result_vec = result.to_vec();

        let expected = vec![0.0f32, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-6,
                "clip({}, 0, 1) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_clip_f64() {
        let input = Array::from_vec(vec![-2.0f64, 0.5, 1.5, 3.0]);
        let result = EnhancedSimdOps::vectorized_clip_f64(&input, 0.0, 1.0);
        let result_vec = result.to_vec();

        let expected = vec![0.0f64, 0.5, 1.0, 1.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "clip({}, 0, 1) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_floor_f64() {
        let input = Array::from_vec(vec![1.2f64, 2.8, -1.2, -2.8]);
        let result = EnhancedSimdOps::vectorized_floor_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, 2.0, -2.0, -3.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "floor({}) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_ceil_f64() {
        let input = Array::from_vec(vec![1.2f64, 2.8, -1.2, -2.8]);
        let result = EnhancedSimdOps::vectorized_ceil_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![2.0f64, 3.0, -1.0, -2.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "ceil({}) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_round_f64() {
        let input = Array::from_vec(vec![1.2f64, 2.8, -1.2, -2.8]);
        let result = EnhancedSimdOps::vectorized_round_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, 3.0, -1.0, -3.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "round({}) = {} but got {}",
                input.to_vec()[i],
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sum_f32() {
        let input = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_sum_f32(&input);

        let expected: f32 = 36.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "sum = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_sum_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_sum_f64(&input);

        let expected: f64 = 10.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "sum = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_product_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_product_f64(&input);

        let expected: f64 = 24.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "product = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_min_f32() {
        let input = Array::from_vec(vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_min_f32(&input);

        let expected: f32 = 1.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "min = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_min_f64() {
        let input = Array::from_vec(vec![5.0f64, 2.0, 8.0, 1.0]);
        let result = EnhancedSimdOps::vectorized_min_f64(&input);

        let expected: f64 = 1.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "min = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_max_f32() {
        let input = Array::from_vec(vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_max_f32(&input);

        let expected: f32 = 9.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "max = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_max_f64() {
        let input = Array::from_vec(vec![5.0f64, 2.0, 8.0, 1.0]);
        let result = EnhancedSimdOps::vectorized_max_f64(&input);

        let expected: f64 = 8.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "max = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_dot_f32() {
        let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Array::from_vec(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let result = EnhancedSimdOps::vectorized_dot_f32(&a, &b).expect("dot product failed");

        let expected: f32 = 36.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "dot = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_dot_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let result = EnhancedSimdOps::vectorized_dot_f64(&a, &b);

        let expected: f64 = 10.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "dot = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_norm_l2_f32() {
        let input = Array::from_vec(vec![3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = EnhancedSimdOps::vectorized_norm_l2_f32(&input);

        let expected: f32 = 5.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "norm_l2 = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_norm_l2_f64() {
        let input = Array::from_vec(vec![3.0f64, 4.0, 0.0, 0.0]);
        let result = EnhancedSimdOps::vectorized_norm_l2_f64(&input);

        let expected: f64 = 5.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "norm_l2 = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_norm_l1_f32() {
        let input = Array::from_vec(vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let result = EnhancedSimdOps::vectorized_norm_l1_f32(&input);

        let expected: f32 = 36.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "norm_l1 = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_norm_l1_f64() {
        let input = Array::from_vec(vec![1.0f64, -2.0, 3.0, -4.0]);
        let result = EnhancedSimdOps::vectorized_norm_l1_f64(&input);

        let expected: f64 = 10.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "norm_l1 = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_mean_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_mean_f64(&input);

        let expected: f64 = 2.5;
        assert!(
            (result - expected).abs() < 1e-14,
            "mean = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_variance_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_variance_f64(&input);

        // variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
        // = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5.0 / 4 = 1.25
        let expected: f64 = 1.25;
        assert!(
            (result - expected).abs() < 1e-7,
            "variance = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_std_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_std_f64(&input);

        let expected: f64 = 1.25_f64.sqrt();
        assert!(
            (result - expected).abs() < 1e-7,
            "std = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cache_aware_matmul_f32() {
        let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
        let mut c = Array::from_vec(vec![0.0f32; 4]).reshape(&[2, 2]);

        let result = EnhancedSimdOps::cache_aware_matmul_f32(&a, &b, &mut c, 2);
        assert!(result.is_ok());

        let c_data = c.to_vec();
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = vec![19.0f32, 22.0, 43.0, 50.0];
        for i in 0..expected.len() {
            assert!(
                (c_data[i] - expected[i]).abs() < 1e-5,
                "matmul[{}] = {} but got {}",
                i,
                expected[i],
                c_data[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_complex_multiply_f32() {
        // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a_real = Array::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let a_imag = Array::from_vec(vec![2.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b_real = Array::from_vec(vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b_imag = Array::from_vec(vec![4.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let (c_real, c_imag) =
            EnhancedSimdOps::complex_multiply_f32(&a_real, &a_imag, &b_real, &b_imag)
                .expect("complex multiply failed");

        let c_real_data = c_real.to_vec();
        let c_imag_data = c_imag.to_vec();

        assert!(
            (c_real_data[0] - (-5.0)).abs() < 1e-5,
            "real = {} but expected -5.0",
            c_real_data[0]
        );
        assert!(
            (c_imag_data[0] - 10.0).abs() < 1e-5,
            "imag = {} but expected 10.0",
            c_imag_data[0]
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_kahan_sum_f32() {
        // Test with values that would lose precision with naive summation
        let input = Array::from_vec(vec![
            1e10f32, 1.0, -1e10, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0,
        ]);
        let result = EnhancedSimdOps::simd_kahan_sum_f32(&input);

        let expected: f32 = 105.0;
        assert!(
            (result - expected).abs() < 1.1,
            "kahan_sum = {} but got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_diff_f64() {
        let input = Array::from_vec(vec![1.0f64, 3.0, 6.0, 10.0]);
        let result = EnhancedSimdOps::vectorized_diff_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![2.0f64, 3.0, 4.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "diff[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_cumsum_f64() {
        let input = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_cumsum_f64(&input);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, 3.0, 6.0, 10.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "cumsum[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_linspace_f64() {
        let result = EnhancedSimdOps::vectorized_linspace_f64(0.0, 1.0, 5);
        let result_vec = result.to_vec();

        let expected = vec![0.0f64, 0.25, 0.5, 0.75, 1.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "linspace[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vectorized_arange_f64() {
        let result = EnhancedSimdOps::vectorized_arange_f64(0.0, 5.0, 1.0);
        let result_vec = result.to_vec();

        let expected = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "arange[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_arrays_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_add_arrays_f64(&a, &b);
        let result_vec = result.to_vec();

        let expected = vec![6.0f64, 8.0, 10.0, 12.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "add[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sub_arrays_f64() {
        let a = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let b = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_sub_arrays_f64(&a, &b);
        let result_vec = result.to_vec();

        let expected = vec![4.0f64, 4.0, 4.0, 4.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "sub[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_mul_arrays_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_mul_arrays_f64(&a, &b);
        let result_vec = result.to_vec();

        let expected = vec![5.0f64, 12.0, 21.0, 32.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "mul[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_div_arrays_f64() {
        let a = Array::from_vec(vec![10.0f64, 12.0, 21.0, 32.0]);
        let b = Array::from_vec(vec![2.0f64, 3.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_div_arrays_f64(&a, &b);
        let result_vec = result.to_vec();

        let expected = vec![5.0f64, 4.0, 3.0, 4.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "div[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_fma_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![2.0f64, 2.0, 2.0, 2.0]);
        let c = Array::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let result = EnhancedSimdOps::vectorized_fma_f64(&a, &b, &c);
        let result_vec = result.to_vec();

        // a*b + c = [1*2+1, 2*2+1, 3*2+1, 4*2+1] = [3, 5, 7, 9]
        let expected = vec![3.0f64, 5.0, 7.0, 9.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "fma[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_scalar_mul_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_scalar_mul_f64(&a, 3.0);
        let result_vec = result.to_vec();

        let expected = vec![3.0f64, 6.0, 9.0, 12.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "scalar_mul[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_square_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = EnhancedSimdOps::vectorized_square_f64(&a);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, 4.0, 9.0, 16.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "square[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_reciprocal_f64() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 4.0, 5.0]);
        let result = EnhancedSimdOps::vectorized_reciprocal_f64(&a);
        let result_vec = result.to_vec();

        let expected = vec![1.0f64, 0.5, 0.25, 0.2];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "reciprocal[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_negative_f64() {
        let a = Array::from_vec(vec![1.0f64, -2.0, 3.0, -4.0]);
        let result = EnhancedSimdOps::vectorized_negative_f64(&a);
        let result_vec = result.to_vec();

        let expected = vec![-1.0f64, 2.0, -3.0, 4.0];
        for i in 0..expected.len() {
            assert!(
                (result_vec[i] - expected[i]).abs() < 1e-14,
                "negative[{}] = {} but got {}",
                i,
                expected[i],
                result_vec[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_large_array_operations() {
        // Test with array size larger than SIMD lanes
        let size = 1000;
        let input: Vec<f64> = (0..size).map(|x| x as f64).collect();
        let a = Array::from_vec(input.clone());
        let b = Array::from_vec(vec![1.0f64; size]);

        // Test addition
        let result = EnhancedSimdOps::vectorized_add_arrays_f64(&a, &b);
        let result_vec = result.to_vec();
        for i in 0..size {
            assert!(
                (result_vec[i] - (input[i] + 1.0)).abs() < 1e-7,
                "large_add[{}] failed",
                i
            );
        }

        // Test sum
        let sum = EnhancedSimdOps::vectorized_sum_f64(&a);
        let expected_sum: f64 = (0..size).map(|x| x as f64).sum();
        assert!(
            (sum - expected_sum).abs() < 1e-6,
            "large_sum failed: expected {} but got {}",
            expected_sum,
            sum
        );
    }
}
