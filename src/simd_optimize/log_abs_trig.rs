//! Additional AVX-512-optimized implementations for trigonometric, logarithmic, and abs functions

use crate::array::Array;
use crate::error::Result;
use std::mem;
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

/// Element-wise natural logarithm of f32 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_log_f32(a: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    // Constants for computing log(x)
    // Using polynomial approximation for log(1+f)
    // Where we can express x = 2^n * (1+f)
    
    let one = _mm512_set1_ps(1.0f32);
    let neg_one_half = _mm512_set1_ps(-0.5f32);
    let one_third = _mm512_set1_ps(1.0f32 / 3.0f32);
    let neg_one_fourth = _mm512_set1_ps(-0.25f32);
    
    // Logarithm of 2 for the scaling factor
    let ln2 = _mm512_set1_ps(0.693147180559945f32);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        
        // Step 1: Extract exponent and mantissa
        // x = 2^n * mantissa, where mantissa in [1, 2)
        let x_bits = _mm512_castps_si512(a_vec);
        
        // Extract exponent (biased by 127)
        let exp_bits = _mm512_srli_epi32(x_bits, 23);
        let exp = _mm512_sub_epi32(exp_bits, _mm512_set1_epi32(127));
        let exp_f = _mm512_cvtepi32_ps(exp);
        
        // Extract mantissa and add the implicit 1.0
        let mantissa_mask = _mm512_set1_epi32(0x007FFFFF);
        let mantissa_bits = _mm512_and_si512(x_bits, mantissa_mask);
        let mantissa_bits_with_ones = _mm512_or_si512(mantissa_bits, _mm512_set1_epi32(0x3F800000)); // Add implicit 1.0
        let mantissa = _mm512_castsi512_ps(mantissa_bits_with_ones);
        
        // Calculate f = mantissa - 1.0
        let f = _mm512_sub_ps(mantissa, one);
        
        // Step 2: log(x) = log(2^n * mantissa) = n*log(2) + log(mantissa)
        // and log(mantissa) = log(1+f) ≈ f - f^2/2 + f^3/3 - f^4/4 + ...
        
        // Compute polynomial approximation for log(1+f)
        let f2 = _mm512_mul_ps(f, f);
        let f3 = _mm512_mul_ps(f2, f);
        let f4 = _mm512_mul_ps(f2, f2);
        
        let log_mantissa = _mm512_add_ps(
            f, _mm512_add_ps(
                _mm512_mul_ps(neg_one_half, f2), _mm512_add_ps(
                    _mm512_mul_ps(one_third, f3),
                    _mm512_mul_ps(neg_one_fourth, f4)
                )
            )
        );
        
        // Compute n*log(2) + log(mantissa)
        let n_log2 = _mm512_mul_ps(exp_f, ln2);
        let log_x = _mm512_add_ps(n_log2, log_mantissa);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), log_x);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].ln();
    }
}

/// Element-wise natural logarithm of f64 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_log_f64(a: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    // Constants for computing log(x)
    // Using polynomial approximation for log(1+f)
    // Where we can express x = 2^n * (1+f)
    
    let one = _mm512_set1_pd(1.0);
    let neg_one_half = _mm512_set1_pd(-0.5);
    let one_third = _mm512_set1_pd(1.0 / 3.0);
    let neg_one_fourth = _mm512_set1_pd(-0.25);
    let one_fifth = _mm512_set1_pd(0.2);
    
    // Logarithm of 2 for the scaling factor
    let ln2 = _mm512_set1_pd(0.693147180559945);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        
        // Step 1: Extract exponent and mantissa
        // For doubles, this is more complex due to the bit layout
        
        // Extract exponent (biased by 1023)
        let x_bits = _mm512_castpd_si512(a_vec);
        
        // Mask to extract the exponent (bits 52-62)
        let exp_mask = _mm512_set1_epi64(0x7FF0000000000000);
        let exp_bits = _mm512_and_si512(x_bits, exp_mask);
        let exp_bits_shifted = _mm512_srli_epi64(exp_bits, 52);
        
        // Convert to double, then subtract the bias (1023)
        let exp_bias = _mm512_set1_epi64(1023);
        let exp_unbiased = _mm512_sub_epi64(exp_bits_shifted, exp_bias);
        let exp_f = _mm512_cvtepi64_pd(exp_unbiased);
        
        // Extract mantissa and add the implicit 1.0
        let mantissa_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFF);
        let mantissa_bits = _mm512_and_si512(x_bits, mantissa_mask);
        
        // Set the exponent to 0 + bias (1023), which is equivalent to [1.0, 2.0)
        let mantissa_bits_with_exp = _mm512_or_si512(mantissa_bits, _mm512_set1_epi64(0x3FF0000000000000));
        let mantissa = _mm512_castsi512_pd(mantissa_bits_with_exp);
        
        // Calculate f = mantissa - 1.0
        let f = _mm512_sub_pd(mantissa, one);
        
        // Step 2: log(x) = log(2^n * mantissa) = n*log(2) + log(mantissa)
        // and log(mantissa) = log(1+f) ≈ f - f^2/2 + f^3/3 - f^4/4 + f^5/5 ...
        
        // Compute polynomial approximation for log(1+f)
        let f2 = _mm512_mul_pd(f, f);
        let f3 = _mm512_mul_pd(f2, f);
        let f4 = _mm512_mul_pd(f2, f2);
        let f5 = _mm512_mul_pd(f4, f);
        
        let log_mantissa = _mm512_add_pd(
            f, _mm512_add_pd(
                _mm512_mul_pd(neg_one_half, f2), _mm512_add_pd(
                    _mm512_mul_pd(one_third, f3), _mm512_add_pd(
                        _mm512_mul_pd(neg_one_fourth, f4),
                        _mm512_mul_pd(one_fifth, f5)
                    )
                )
            )
        );
        
        // Compute n*log(2) + log(mantissa)
        let n_log2 = _mm512_mul_pd(exp_f, ln2);
        let log_x = _mm512_add_pd(n_log2, log_mantissa);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), log_x);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].ln();
    }
}

/// Element-wise absolute value of f32 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_abs_f32(a: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    // Mask to clear the sign bit (all bits except the MSB)
    let abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        
        // Clear the sign bit using a bitwise AND operation
        let a_bits = _mm512_castps_si512(a_vec);
        let abs_bits = _mm512_and_si512(a_bits, abs_mask);
        let abs_vec = _mm512_castsi512_ps(abs_bits);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), abs_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].abs();
    }
}

/// Element-wise absolute value of f64 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_abs_f64(a: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    // Mask to clear the sign bit (all bits except the MSB)
    let abs_mask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        
        // Clear the sign bit using a bitwise AND operation
        let a_bits = _mm512_castpd_si512(a_vec);
        let abs_bits = _mm512_and_si512(a_bits, abs_mask);
        let abs_vec = _mm512_castsi512_pd(abs_bits);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), abs_vec);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].abs();
    }
}

/// Wrapper for AVX-512-optimized element-wise logarithm for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_log_f32(a: &Array<f32>) -> Array<f32> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_log_f32(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_log_f32(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].ln();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized element-wise logarithm for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_log_f64(a: &Array<f64>) -> Array<f64> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_log_f64(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_log_f64(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].ln();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized element-wise absolute value for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_abs_f32(a: &Array<f32>) -> Array<f32> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_abs_f32(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_abs_f32(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].abs();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized element-wise absolute value for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_abs_f64(a: &Array<f64>) -> Array<f64> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_abs_f64(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_abs_f64(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].abs();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Element-wise sine calculation of f32 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_sin_f32(a: &[f32], result: &mut [f32]) {
    use std::arch::x86_64::*;
    
    // We process 16 elements at a time with AVX-512
    let simd_width = 16;
    let simd_chunks = a.len() / simd_width;
    
    // Constants for range reduction and polynomial approximation
    let two_over_pi = _mm512_set1_ps(0.6366197723675814f32); // 2/π
    let pi_over_two = _mm512_set1_ps(PI_F32 / 2.0);
    
    // Constants for polynomial approximation of sin(x) for x in [-π/2, π/2]
    let c1 = _mm512_set1_ps(1.0f32);
    let c3 = _mm512_set1_ps(-1.0f32 / 6.0f32);
    let c5 = _mm512_set1_ps(1.0f32 / 120.0f32);
    let c7 = _mm512_set1_ps(-1.0f32 / 5040.0f32);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 16 f32 values (512 bits) from array
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(idx));
        
        // Range reduction: reduce input to [-π/2, π/2]
        // 1. Divide by π/2 to get the number of π/2 intervals
        let k_float = _mm512_mul_ps(a_vec, two_over_pi);
        
        // 2. Round to nearest integer
        let k = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(k_float);
        
        // 3. Subtract k*π/2 from input
        let x = _mm512_fnmadd_ps(k, pi_over_two, a_vec); // a - k*π/2
        
        // Now x is in the range [-π/4, π/4]
        
        // Calculate polynomial approximation of sin(x)
        let x2 = _mm512_mul_ps(x, x);
        let x3 = _mm512_mul_ps(x, x2);
        let x5 = _mm512_mul_ps(x3, x2);
        let x7 = _mm512_mul_ps(x5, x2);
        
        let sin_x = _mm512_add_ps(
            x, _mm512_add_ps(
                _mm512_mul_ps(c3, x3), _mm512_add_ps(
                    _mm512_mul_ps(c5, x5),
                    _mm512_mul_ps(c7, x7)
                )
            )
        );
        
        // Handle sign based on the quadrant (determined by k mod 4)
        // For sin: sin(x + n*π/2) = sin(x) when n mod 4 = 0
        //                           cos(x) when n mod 4 = 1
        //                          -sin(x) when n mod 4 = 2
        //                          -cos(x) when n mod 4 = 3
        
        // Get k mod 4 by using a bitwise AND with 3 (binary 11)
        let k_int = _mm512_cvttps_epi32(k);
        let quadrant = _mm512_and_epi32(k_int, _mm512_set1_epi32(3));
        
        // Create masks for different quadrants
        let mask_q1 = _mm512_cmpeq_epi32_mask(quadrant, _mm512_set1_epi32(1));
        let mask_q2 = _mm512_cmpeq_epi32_mask(quadrant, _mm512_set1_epi32(2));
        let mask_q3 = _mm512_cmpeq_epi32_mask(quadrant, _mm512_set1_epi32(3));
        
        // For Q1 and Q3, we need to compute cos(x) instead of sin(x)
        // cos(x) = 1 - x^2/2 + x^4/24 - x^6/720 + ...
        
        let c2 = _mm512_set1_ps(-0.5f32);
        let c4 = _mm512_set1_ps(1.0f32 / 24.0f32);
        let c6 = _mm512_set1_ps(-1.0f32 / 720.0f32);
        
        let x4 = _mm512_mul_ps(x2, x2);
        let x6 = _mm512_mul_ps(x4, x2);
        
        let cos_x = _mm512_add_ps(
            c1, _mm512_add_ps(
                _mm512_mul_ps(c2, x2), _mm512_add_ps(
                    _mm512_mul_ps(c4, x4),
                    _mm512_mul_ps(c6, x6)
                )
            )
        );
        
        // Select sin_x or cos_x based on quadrant
        let result_vec = _mm512_mask_blend_ps(mask_q1 | mask_q3, sin_x, cos_x);
        
        // Apply sign based on quadrant (negative for Q2 and Q3)
        let neg_sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000)); // IEEE 754 sign bit for f32
        let sign_mask = _mm512_mask_blend_ps(mask_q2 | mask_q3, _mm512_setzero_ps(), neg_sign_mask);
        
        let final_result = _mm512_xor_ps(result_vec, sign_mask);
        
        // Store result
        _mm512_storeu_ps(result.as_mut_ptr().add(idx), final_result);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].sin();
    }
}

/// Element-wise sine calculation of f64 array using AVX-512
///
/// # Safety
///
/// This function uses AVX-512 intrinsics and requires:
/// - The CPU supports AVX-512 instructions
/// - a and result are properly aligned for AVX-512 (64-byte alignment)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn avx512_sin_f64(a: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;
    
    // We process 8 elements at a time with AVX-512
    let simd_width = 8;
    let simd_chunks = a.len() / simd_width;
    
    // Constants for range reduction and polynomial approximation
    let two_over_pi = _mm512_set1_pd(0.6366197723675814); // 2/π
    let pi_over_two = _mm512_set1_pd(PI_F64 / 2.0);
    
    // Constants for polynomial approximation of sin(x) for x in [-π/2, π/2]
    let c1 = _mm512_set1_pd(1.0);
    let c3 = _mm512_set1_pd(-1.0 / 6.0);
    let c5 = _mm512_set1_pd(1.0 / 120.0);
    let c7 = _mm512_set1_pd(-1.0 / 5040.0);
    let c9 = _mm512_set1_pd(1.0 / 362880.0);
    
    for i in 0..simd_chunks {
        let idx = i * simd_width;
        
        // Load 8 f64 values (512 bits) from array
        let a_vec = _mm512_loadu_pd(a.as_ptr().add(idx));
        
        // Range reduction: reduce input to [-π/2, π/2]
        // 1. Divide by π/2 to get the number of π/2 intervals
        let k_float = _mm512_mul_pd(a_vec, two_over_pi);
        
        // 2. Round to nearest integer
        let k = _mm512_roundscale_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(k_float);
        
        // 3. Subtract k*π/2 from input
        let x = _mm512_fnmadd_pd(k, pi_over_two, a_vec); // a - k*π/2
        
        // Now x is in the range [-π/4, π/4]
        
        // Calculate polynomial approximation of sin(x)
        let x2 = _mm512_mul_pd(x, x);
        let x3 = _mm512_mul_pd(x, x2);
        let x5 = _mm512_mul_pd(x3, x2);
        let x7 = _mm512_mul_pd(x5, x2);
        let x9 = _mm512_mul_pd(x7, x2);
        
        let sin_x = _mm512_add_pd(
            x, _mm512_add_pd(
                _mm512_mul_pd(c3, x3), _mm512_add_pd(
                    _mm512_mul_pd(c5, x5), _mm512_add_pd(
                        _mm512_mul_pd(c7, x7),
                        _mm512_mul_pd(c9, x9)
                    )
                )
            )
        );
        
        // Handle sign based on the quadrant (determined by k mod 4)
        // For sin: sin(x + n*π/2) = sin(x) when n mod 4 = 0
        //                           cos(x) when n mod 4 = 1
        //                          -sin(x) when n mod 4 = 2
        //                          -cos(x) when n mod 4 = 3
        
        // Get k mod 4 by using a bitwise AND with 3 (binary 11)
        let k_int = _mm512_cvttpd_epi32(k);
        let quadrant = _mm256_and_si256(_mm256_castsi128_si256(k_int), _mm256_set1_epi32(3));
        
        // Convert back to 64-bit integers
        let quadrant_64 = _mm512_cvtepi32_epi64(_mm256_castsi256_si128(quadrant));
        
        // Create masks for different quadrants
        let mask_q1 = _mm512_cmpeq_epi64_mask(quadrant_64, _mm512_set1_epi64(1));
        let mask_q2 = _mm512_cmpeq_epi64_mask(quadrant_64, _mm512_set1_epi64(2));
        let mask_q3 = _mm512_cmpeq_epi64_mask(quadrant_64, _mm512_set1_epi64(3));
        
        // For Q1 and Q3, we need to compute cos(x) instead of sin(x)
        // cos(x) = 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 ...
        
        let c2 = _mm512_set1_pd(-0.5);
        let c4 = _mm512_set1_pd(1.0 / 24.0);
        let c6 = _mm512_set1_pd(-1.0 / 720.0);
        let c8 = _mm512_set1_pd(1.0 / 40320.0);
        
        let x4 = _mm512_mul_pd(x2, x2);
        let x6 = _mm512_mul_pd(x4, x2);
        let x8 = _mm512_mul_pd(x4, x4);
        
        let cos_x = _mm512_add_pd(
            c1, _mm512_add_pd(
                _mm512_mul_pd(c2, x2), _mm512_add_pd(
                    _mm512_mul_pd(c4, x4), _mm512_add_pd(
                        _mm512_mul_pd(c6, x6),
                        _mm512_mul_pd(c8, x8)
                    )
                )
            )
        );
        
        // Select sin_x or cos_x based on quadrant
        let result_vec = _mm512_mask_blend_pd(mask_q1 | mask_q3, sin_x, cos_x);
        
        // Apply sign based on quadrant (negative for Q2 and Q3)
        let neg_sign_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000)); // IEEE 754 sign bit for f64
        let sign_mask = _mm512_mask_blend_pd(mask_q2 | mask_q3, _mm512_setzero_pd(), neg_sign_mask);
        
        let final_result = _mm512_xor_pd(result_vec, sign_mask);
        
        // Store result
        _mm512_storeu_pd(result.as_mut_ptr().add(idx), final_result);
    }
    
    // Handle remaining elements
    let remainder_start = simd_chunks * simd_width;
    for i in remainder_start..a.len() {
        result[i] = a[i].sin();
    }
}

/// Wrapper for AVX-512-optimized element-wise sine for f32
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_sin_f32(a: &Array<f32>) -> Array<f32> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_sin_f32(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_sin_f32(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].sin();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}

/// Wrapper for AVX-512-optimized element-wise sine for f64
#[cfg(target_arch = "x86_64")]
pub fn avx512_optimized_sin_f64(a: &Array<f64>) -> Array<f64> {
    // Flatten array to 1D for easier SIMD processing
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];
    
    // Check if AVX-512 is available
    #[cfg(target_feature = "avx512f")]
    unsafe {
        // Use AVX-512 implementation
        avx512_sin_f64(&a_data, &mut result_data);
    }
    
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Use CPU detection at runtime
        let features = crate::simd_optimize::detect_cpu_features();
        if features.avx512f {
            unsafe {
                // Use AVX-512 with runtime detection
                avx512_sin_f64(&a_data, &mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for i in 0..a_data.len() {
                result_data[i] = a_data[i].sin();
            }
        }
    }
    
    // Reshape result back to original shape
    Array::from_vec(result_data).reshape(&a.shape())
}