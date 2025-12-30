//! AVX-512-optimized implementations for common numerical operations
//!
//! This module provides highly optimized AVX-512 implementations for common
//! numerical operations used in NumRS. These implementations leverage AVX-512 
//! intrinsics for maximum performance on x86_64 CPUs with AVX-512 support.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::mem;
use std::alloc::{alloc, Layout};
use std::ptr;
use std::f32;
use std::f64;

/// Element-wise addition of two f32 arrays using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and b have the same shape
/// - a, b, and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from each array
        // Use unaligned loads for flexibility
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(idx));
        
        // Perform addition
        let result_vec = _mm512_add_ps(a_vec, b_vec);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i] + b[i];
    }
}

/// Element-wise addition of two f64 arrays using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and b have the same shape
/// - a, b, and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_add_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from each array
        // Use unaligned loads for flexibility
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_pd(b.as_ptr().add(idx));
        
        // Perform addition
        let result_vec = _mm512_add_pd(a_vec, b_vec);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i] + b[i];
    }
}

/// Element-wise multiplication of two f32 arrays using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and b have the same shape
/// - a, b, and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from each array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(idx));
        
        // Perform multiplication
        let result_vec = _mm512_mul_ps(a_vec, b_vec);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i] * b[i];
    }
}

/// Element-wise multiplication of two f64 arrays using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and b have the same shape
/// - a, b, and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_mul_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from each array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_pd(b.as_ptr().add(idx));
        
        // Perform multiplication
        let result_vec = _mm512_mul_pd(a_vec, b_vec);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i] * b[i];
    }
}

/// Element-wise division of two f32 arrays using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and b have the same shape
/// - a, b, and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_div_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from each array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(idx));
        
        // Perform division
        let result_vec = _mm512_div_ps(a_vec, b_vec);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i] / b[i];
    }
}

/// Element-wise division of two f64 arrays using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and b have the same shape
/// - a, b, and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_div_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from each array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        let b_vec = _mm512_loadu_pd(b.as_ptr().add(idx));
        
        // Perform division
        let result_vec = _mm512_div_pd(a_vec, b_vec);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i] / b[i];
    }
}

/// Element-wise square root of f32 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_sqrt_f32(a: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        
        // Perform square root
        let result_vec = _mm512_sqrt_ps(a_vec);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].sqrt();
    }
}

/// Element-wise square root of f64 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_sqrt_f64(a: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        
        // Perform square root
        let result_vec = _mm512_sqrt_pd(a_vec);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].sqrt();
    }
}

/// Horizontal sum of f32 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a is properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_sum_f32(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    // Initialize accumulator
    let mut sum_vec = _mm512_setzero_ps();
    
    // Process chunks of 16 elements
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        
        // Add to accumulator
        sum_vec = _mm512_add_ps(sum_vec, a_vec);
    }
    
    // Horizontal sum of accumulated vector
    // AVX-512 provides a built-in horizontal sum operation
    let sum = _mm512_reduce_add_ps(sum_vec);
    
    // Add remaining elements
    let remainder_start = simd_chunks * simd_width;
    let mut result = sum;
    for i in remainder_start..a.len() {
        result += a[i];
    }
    
    result
}

/// Horizontal sum of f64 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a is properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_sum_f64(a: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    // Initialize accumulator
    let mut sum_vec = _mm512_setzero_pd();
    
    // Process chunks of 8 elements
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        
        // Add to accumulator
        sum_vec = _mm512_add_pd(sum_vec, a_vec);
    }
    
    // Horizontal sum of accumulated vector
    // AVX-512 provides a built-in horizontal sum operation
    let sum = _mm512_reduce_add_pd(sum_vec);
    
    // Add remaining elements
    let remainder_start = simd_chunks * simd_width;
    let mut result = sum;
    for i in remainder_start..a.len() {
        result += a[i];
    }
    
    result
}

/// Element-wise exponential of f32 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_exp_f32(a: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    // Constants for computing exp(x)
    // exp(x) can be approximated using the following algorithm:
    // 1. Express x = n + f, where n is an integer and f is a fraction in [-0.5, 0.5]
    // 2. exp(x) = exp(n + f) = exp(n) * exp(f)
    // 3. exp(n) = 2^(n * log2(e)) can be computed directly
    // 4. exp(f) can be approximated with a polynomial
    
    // log2(e) constant
    let log2e = _mm512_set1_ps(1.442695040888963f32);
    
    // Constants for polynomial approximation of exp(f) where f in [-0.5, 0.5]
    let c1 = _mm512_set1_ps(1.0f32);
    let c2 = _mm512_set1_ps(1.0f32);
    let c3 = _mm512_set1_ps(0.5f32);
    let c4 = _mm512_set1_ps(0.1666666666f32);
    let c5 = _mm512_set1_ps(0.0416666666f32);
    
    // Range reduction bounds
    let half = _mm512_set1_ps(0.5f32);
    let neg_half = _mm512_set1_ps(-0.5f32);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        
        // Step 1: x * log2(e)
        let x_log2e = _mm512_mul_ps(a_vec, log2e);
        
        // Step 2: Split into integer and fractional parts
        // Using round to nearest integer
        let n = _mm512_roundscale_ps::<(_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)>(x_log2e);
        
        // Fractional part: f = x * log2(e) - n
        let f = _mm512_sub_ps(x_log2e, n);
        
        // Step 3: Compute 2^n
        // Convert n to int32, add 127*2^23 (to generate 2^n in the exponent field)
        // then convert back to float
        let pow2n = {
            // Convert float to int with truncation
            let n_int = _mm512_cvttps_epi32(n);
            
            // Biased exponent (add 127 << 23 to get 2^n in the exponent field)
            let biased_n = _mm512_add_epi32(n_int, _mm512_set1_epi32(127 << 23));
            
            // Shift to proper position for IEEE 754 float exponent
            let biased_n_shifted = _mm512_slli_epi32(biased_n, 23);
            
            // Reinterpret bits as float
            _mm512_castsi512_ps(biased_n_shifted)
        };
        
        // Step 4: Compute exp(f) using polynomial approximation
        // exp(f) ≈ 1 + f + f^2/2 + f^3/6 + f^4/24
        let f2 = _mm512_mul_ps(f, f);
        let f3 = _mm512_mul_ps(f2, f);
        let f4 = _mm512_mul_ps(f2, f2);
        
        let poly = _mm512_add_ps(
            c1, _mm512_add_ps(
                f, _mm512_add_ps(
                    _mm512_mul_ps(c3, f2), _mm512_add_ps(
                        _mm512_mul_ps(c4, f3),
                        _mm512_mul_ps(c5, f4)
                    )
                )
            )
        );
        
        // Combine 2^n and exp(f)
        let exp_x = _mm512_mul_ps(pow2n, poly);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), exp_x);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].exp();
    }
}

/// Element-wise exponential of f64 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_exp_f64(a: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    // Constants for computing exp(x)
    // log2(e) constant
    let log2e = _mm512_set1_pd(1.442695040888963);
    
    // Constants for polynomial approximation of exp(f) where f in [-0.5, 0.5]
    let c1 = _mm512_set1_pd(1.0);
    let c2 = _mm512_set1_pd(1.0);
    let c3 = _mm512_set1_pd(0.5);
    let c4 = _mm512_set1_pd(0.1666666666666667);
    let c5 = _mm512_set1_pd(0.0416666666666667);
    let c6 = _mm512_set1_pd(0.0083333333333333);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        
        // Step 1: x * log2(e)
        let x_log2e = _mm512_mul_pd(a_vec, log2e);
        
        // Step 2: Split into integer and fractional parts
        // Using round to nearest integer
        let n = _mm512_roundscale_pd::<(_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)>(x_log2e);
        
        // Fractional part: f = x * log2(e) - n
        let f = _mm512_sub_pd(x_log2e, n);
        
        // Step 3: Compute 2^n
        // Convert n to int64, add 1023*2^52 (to generate 2^n in the exponent field)
        // then convert back to double
        let pow2n = {
            // Convert double to int with truncation
            let n_int = _mm512_cvttpd_epi32(n);
            
            // Extend to 64-bit integers
            let n_int64 = _mm512_cvtepi32_epi64(_mm256_castsi256_si128(n_int));
            
            // Biased exponent (add 1023 << 52 to get 2^n in the exponent field)
            let biased_n = _mm512_add_epi64(n_int64, _mm512_set1_epi64(1023 << 52));
            
            // Shift to proper position for IEEE 754 double exponent
            let biased_n_shifted = _mm512_slli_epi64(biased_n, 52);
            
            // Reinterpret bits as double
            _mm512_castsi512_pd(biased_n_shifted)
        };
        
        // Step 4: Compute exp(f) using polynomial approximation
        // exp(f) ≈ 1 + f + f^2/2 + f^3/6 + f^4/24 + f^5/120
        let f2 = _mm512_mul_pd(f, f);
        let f3 = _mm512_mul_pd(f2, f);
        let f4 = _mm512_mul_pd(f2, f2);
        let f5 = _mm512_mul_pd(f4, f);
        
        let poly = _mm512_add_pd(
            c1, _mm512_add_pd(
                f, _mm512_add_pd(
                    _mm512_mul_pd(c3, f2), _mm512_add_pd(
                        _mm512_mul_pd(c4, f3), _mm512_add_pd(
                            _mm512_mul_pd(c5, f4),
                            _mm512_mul_pd(c6, f5)
                        )
                    )
                )
            )
        );
        
        // Combine 2^n and exp(f)
        let exp_x = _mm512_mul_pd(pow2n, poly);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), exp_x);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].exp();
    }
}

/// Wrapper for AVX-512-optimized element-wise addition for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_add_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
    // Check shapes match
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    
    // Flatten arrays to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_add_f32(&a_data, &b_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_add_f32(&a_data, &b_data, &mut result_data);
            }
        } else {
            // Fallback to AVX2 or scalar addition
            if features.avx2 {
                unsafe {
                    crate::simd_optimize::avx2_ops::avx2_add_f32(&a_data, &b_data, &mut result_data);
                }
            } else {
                // Fallback to scalar addition
                for i in 0..a_data.len() {
                    result_data[i] = a_data[i] + b_data[i];
                }
            }
        }
    }
    
    // Reshape result back to original shape
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// Wrapper for AVX-512-optimized element-wise addition for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_add_f64(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    // Check shapes match
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    
    // Flatten arrays to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_add_f64(&a_data, &b_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_add_f64(&a_data, &b_data, &mut result_data);
            }
        } else {
            // Fallback to AVX2 or scalar addition
            if features.avx2 {
                unsafe {
                    crate::simd_optimize::avx2_ops::avx2_add_f64(&a_data, &b_data, &mut result_data);
                }
            } else {
                // Fallback to scalar addition
                for i in 0..a_data.len() {
                    result_data[i] = a_data[i] + b_data[i];
                }
            }
        }
    }
    
    // Reshape result back to original shape
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// Wrapper for AVX-512-optimized element-wise multiplication for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_mul_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
    // Check shapes match
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    
    // Flatten arrays to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_mul_f32(&a_data, &b_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_mul_f32(&a_data, &b_data, &mut result_data);
            }
        } else {
            // Fallback to AVX2 or scalar multiplication
            if features.avx2 {
                unsafe {
                    crate::simd_optimize::avx2_ops::avx2_mul_f32(&a_data, &b_data, &mut result_data);
                }
            } else {
                // Fallback to scalar multiplication
                for i in 0..a_data.len() {
                    result_data[i] = a_data[i] * b_data[i];
                }
            }
        }
    }
    
    // Reshape result back to original shape
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// Wrapper for AVX-512-optimized element-wise multiplication for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_mul_f64(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    // Check shapes match
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    
    // Flatten arrays to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_mul_f64(&a_data, &b_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_mul_f64(&a_data, &b_data, &mut result_data);
            }
        } else {
            // Fallback to AVX2 or scalar multiplication
            if features.avx2 {
                unsafe {
                    crate::simd_optimize::avx2_ops::avx2_mul_f64(&a_data, &b_data, &mut result_data);
                }
            } else {
                // Fallback to scalar multiplication
                for i in 0..a_data.len() {
                    result_data[i] = a_data[i] * b_data[i];
                }
            }
        }
    }
    
    // Reshape result back to original shape
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// Wrapper for AVX-512-optimized element-wise square root for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_sqrt_f32(a: &Array<f32>) -> Array<f32> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_sqrt_f32(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_sqrt_f32(&a_data, &mut result_data);
            }
        } else {
            // Fallback to AVX2 or scalar square root
            if features.avx2 {
                unsafe {
                    crate::simd_optimize::avx2_ops::avx2_sqrt_f32(&a_data, &mut result_data);
                }
            } else {
                // Fallback to scalar square root
                for i in 0..a_data.len() {
                    result_data[i] = a_data[i].sqrt();
                }
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized element-wise square root for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_sqrt_f64(a: &Array<f64>) -> Array<f64> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_sqrt_f64(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_sqrt_f64(&a_data, &mut result_data);
            }
        } else {
            // Fallback to AVX2 or scalar square root
            if features.avx2 {
                unsafe {
                    crate::simd_optimize::avx2_ops::avx2_sqrt_f64(&a_data, &mut result_data);
                }
            } else {
                // Fallback to scalar square root
                for i in 0..a_data.len() {
                    result_data[i] = a_data[i].sqrt();
                }
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized sum for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_sum_f32(a: &Array<f32>) -> f32 {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        return avx512_sum_f32(&a_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                return avx512_sum_f32(&a_data);
            }
        } else {
            // Fallback to AVX2 or scalar sum
            if features.avx2 {
                unsafe {
                    return crate::simd_optimize::avx2_ops::avx2_sum_f32(&a_data);
                }
            } else {
                // Fallback to scalar sum
                return a_data.iter().sum();
            }
        }
    }
    
    // Fallback for non-x86_64 platforms
    a_data.iter().sum()
}

/// Wrapper for AVX-512-optimized sum for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_sum_f64(a: &Array<f64>) -> f64 {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        return avx512_sum_f64(&a_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                return avx512_sum_f64(&a_data);
            }
        } else {
            // Fallback to AVX2 or scalar sum
            if features.avx2 {
                unsafe {
                    return crate::simd_optimize::avx2_ops::avx2_sum_f64(&a_data);
                }
            } else {
                // Fallback to scalar sum
                return a_data.iter().sum();
            }
        }
    }
    
    // Fallback for non-x86_64 platforms
    a_data.iter().sum()
}

/// Wrapper for AVX-512-optimized element-wise exponential for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_exp_f32(a: &Array<f32>) -> Array<f32> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_exp_f32(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_exp_f32(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].exp();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized element-wise exponential for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_exp_f64(a: &Array<f64>) -> Array<f64> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_exp_f64(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_exp_f64(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].exp();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}