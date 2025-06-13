//! SIMD-optimized trait implementations
//!
//! This module provides trait implementations that leverage SIMD optimizations
//! for better performance while maintaining compatibility with the existing trait system.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::traits::{ArrayOps, ArrayReduction, ArrayMath, LinearAlgebra, FloatingPoint};
use super::unified_dispatcher::global_dispatcher;

/// SIMD-optimized extension methods for Array<f32>
/// 
/// These methods provide high-performance SIMD implementations that can be used
/// to accelerate numerical computations on x86_64 and ARM architectures.
pub trait SimdArrayOps {
    /// SIMD-optimized element-wise addition
    fn simd_add(&self, other: &Self) -> Result<Array<f32>>;
    
    /// SIMD-optimized element-wise multiplication
    fn simd_mul(&self, other: &Self) -> Result<Array<f32>>;
    
    /// SIMD-optimized sum reduction
    fn simd_sum(&self) -> f32;
    
    /// SIMD-optimized exponential function
    fn simd_exp(&self) -> Array<f32>;
    
    /// SIMD-optimized logarithm function
    fn simd_log(&self) -> Array<f32>;
    
    /// SIMD-optimized trigonometric functions
    fn simd_sin_cos(&self) -> (Array<f32>, Array<f32>);
    
    /// SIMD-optimized matrix multiplication
    fn simd_matmul(&self, other: &Self) -> Result<Array<f32>>;
    
    /// SIMD-optimized dot product
    fn simd_dot(&self, other: &Self) -> Result<f32>;
    
    /// SIMD-optimized memory copy
    fn simd_copy(&self) -> Result<Array<f32>>;
}

impl SimdArrayOps for Array<f32> {
    fn simd_add(&self, other: &Self) -> Result<Array<f32>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }
        
        // Use SIMD optimization where available
        let self_data = self.to_vec();
        let other_data = other.to_vec();
        let mut result_data = vec![0.0f32; self_data.len()];

        // Check if SIMD optimization is available
        let dispatcher = global_dispatcher();
        match dispatcher.implementation_info().name {
            "AVX2" | "AVX-512" => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    super::avx2_ops::avx2_add_f32(&self_data, &other_data, &mut result_data);
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..self_data.len() {
                        result_data[i] = self_data[i] + other_data[i];
                    }
                }
            }
            "NEON" => {
                #[cfg(target_arch = "aarch64")]
                {
                    // NEON implementation would go here
                    for i in 0..self_data.len() {
                        result_data[i] = self_data[i] + other_data[i];
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    for i in 0..self_data.len() {
                        result_data[i] = self_data[i] + other_data[i];
                    }
                }
            }
            _ => {
                // Scalar fallback
                for i in 0..self_data.len() {
                    result_data[i] = self_data[i] + other_data[i];
                }
            }
        }

        Ok(Array::from_vec(result_data).reshape(&self.shape()))
    }

    fn simd_mul(&self, other: &Self) -> Result<Array<f32>> {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }
        
        let self_data = self.to_vec();
        let other_data = other.to_vec();
        let mut result_data = vec![0.0f32; self_data.len()];

        // Use SIMD optimization for multiplication
        let dispatcher = global_dispatcher();
        match dispatcher.implementation_info().name {
            "AVX2" | "AVX-512" => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    super::avx2_ops::avx2_mul_f32(&self_data, &other_data, &mut result_data);
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..self_data.len() {
                        result_data[i] = self_data[i] * other_data[i];
                    }
                }
            }
            _ => {
                for i in 0..self_data.len() {
                    result_data[i] = self_data[i] * other_data[i];
                }
            }
        }

        Ok(Array::from_vec(result_data).reshape(&self.shape()))
    }

    fn simd_sum(&self) -> f32 {
        global_dispatcher().optimized_sum_f32(self)
    }

    fn simd_exp(&self) -> Array<f32> {
        global_dispatcher().optimized_exp_f32(self)
    }

    fn simd_log(&self) -> Array<f32> {
        global_dispatcher().optimized_log_f32(self)
    }

    fn simd_sin_cos(&self) -> (Array<f32>, Array<f32>) {
        global_dispatcher().optimized_sin_cos_f32(self)
    }

    fn simd_matmul(&self, other: &Self) -> Result<Array<f32>> {
        global_dispatcher().optimized_matmul_f32(self, other)
    }

    fn simd_dot(&self, other: &Self) -> Result<f32> {
        global_dispatcher().optimized_dot_f32(self, other)
    }

    fn simd_copy(&self) -> Result<Array<f32>> {
        global_dispatcher().optimized_copy_f32(self)
    }
}

/// Convenience macro for creating SIMD-optimized arrays
#[macro_export]
macro_rules! simd_array {
    ($($x:expr),* $(,)?) => {
        Array::from_vec(vec![$($x),*])
    };
    ($x:expr; $n:expr) => {
        Array::from_vec(vec![$x; $n])
    };
}

/// Performance hints for SIMD operations
pub struct SimdPerformanceHints;

impl SimdPerformanceHints {
    /// Get recommended array size for optimal SIMD performance
    pub fn optimal_array_size() -> usize {
        let dispatcher = global_dispatcher();
        match dispatcher.implementation_info().vector_width {
            512 => 16 * 4, // AVX-512: 16 f32 elements * 4 for good ILP
            256 => 8 * 4,  // AVX2: 8 f32 elements * 4 for good ILP
            128 => 4 * 4,  // NEON: 4 f32 elements * 4 for good ILP
            _ => 16,       // Conservative default
        }
    }

    /// Check if array size is SIMD-friendly
    pub fn is_simd_friendly(size: usize) -> bool {
        let dispatcher = global_dispatcher();
        let vector_elements = match dispatcher.implementation_info().vector_width {
            512 => 16, // AVX-512 f32
            256 => 8,  // AVX2 f32
            128 => 4,  // NEON f32
            _ => 4,    // Conservative default
        };
        
        size % vector_elements == 0 && size >= vector_elements * 2
    }

    /// Get alignment requirement for optimal SIMD performance
    pub fn alignment_requirement() -> usize {
        let dispatcher = global_dispatcher();
        dispatcher.implementation_info().vector_width / 8 // Convert bits to bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::stats::Statistics;
    use crate::math::ElementWiseMath;

    #[test]
    fn test_simd_array_ops() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        // Test SIMD-optimized operations
        let sum = a.simd_add(&b).unwrap();
        assert_eq!(sum.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);

        let product = a.simd_mul(&b).unwrap();
        assert_eq!(product.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_simd_reductions() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let sum = array.simd_sum();
        assert_relative_eq!(sum, 10.0, epsilon = 1e-6);

        let mean = array.mean();
        assert_relative_eq!(mean, 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_math_functions() {
        let array = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0]);

        let sqrt_result = array.sqrt();
        assert_relative_eq!(sqrt_result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(sqrt_result.to_vec()[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(sqrt_result.to_vec()[2], 3.0, epsilon = 1e-6);
        assert_relative_eq!(sqrt_result.to_vec()[3], 4.0, epsilon = 1e-6);

        let exp_input = Array::from_vec(vec![0.0, 1.0]);
        let exp_result = exp_input.simd_exp();
        assert_relative_eq!(exp_result.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(exp_result.to_vec()[1], std::f32::consts::E, epsilon = 1e-5);
    }

    #[test]
    fn test_performance_hints() {
        let optimal_size = SimdPerformanceHints::optimal_array_size();
        assert!(optimal_size >= 16);

        let is_friendly = SimdPerformanceHints::is_simd_friendly(64);
        println!("Size 64 is SIMD-friendly: {}", is_friendly);

        let alignment = SimdPerformanceHints::alignment_requirement();
        assert!(alignment >= 16);
    }

    #[test]
    fn test_simd_array_macro() {
        let array = simd_array![1.0, 2.0, 3.0, 4.0];
        assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}