//! AVX2-optimized SIMD operations
//!
//! This module provides optimized implementations of common array
//! operations using AVX2 intrinsics.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized element-wise addition for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Process 8 f32 elements per iteration using AVX2
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] + b[i];
    }
}

/// AVX2-optimized element-wise addition for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_add_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Process 4 f64 elements per iteration using AVX2
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] + b[i];
    }
}

/// AVX2-optimized element-wise multiplication for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Process 8 f32 elements per iteration using AVX2
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] * b[i];
    }
}

/// AVX2-optimized element-wise multiplication for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_mul_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Process 4 f64 elements per iteration using AVX2
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] * b[i];
    }
}

/// AVX2-optimized element-wise division for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_div_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Process 8 f32 elements per iteration using AVX2
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] / b[i];
    }
}

/// AVX2-optimized element-wise division for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_div_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Process 4 f64 elements per iteration using AVX2
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let vresult = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] / b[i];
    }
}

/// AVX2-optimized square root for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_sqrt_f32(a: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Process 8 f32 elements per iteration using AVX2
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vresult = _mm256_sqrt_ps(va);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i].sqrt();
    }
}

/// AVX2-optimized square root for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_sqrt_f64(a: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Process 4 f64 elements per iteration using AVX2
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vresult = _mm256_sqrt_pd(va);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i].sqrt();
    }
}

/// AVX2-optimized sum reduction for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_sum_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Initialize accumulator
    let mut vacc = _mm256_setzero_ps();

    // Process 8 f32 elements per iteration using AVX2
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        vacc = _mm256_add_ps(vacc, va);
    }

    // Horizontal sum of the accumulator
    let hi128 = _mm256_extractf128_ps(vacc, 1);
    let lo128 = _mm256_castps256_ps128(vacc);
    let sum128 = _mm_add_ps(hi128, lo128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x1B);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    let final_sum = _mm_add_ps(sums, shuf2);
    let mut result = _mm_cvtss_f32(final_sum);

    // Process remaining elements
    for item in &a[simd_len..] {
        result += item;
    }

    result
}

/// AVX2-optimized sum reduction for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_sum_f64(a: &[f64]) -> f64 {
    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Initialize accumulator
    let mut vacc = _mm256_setzero_pd();

    // Process 4 f64 elements per iteration using AVX2
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        vacc = _mm256_add_pd(vacc, va);
    }

    // Horizontal sum of the accumulator
    let hi128 = _mm256_extractf128_pd(vacc, 1);
    let lo128 = _mm256_castpd256_pd128(vacc);
    let sum128 = _mm_add_pd(hi128, lo128);
    let shuf = _mm_shuffle_pd(sum128, sum128, 0x01);
    let final_sum = _mm_add_pd(sum128, shuf);
    let mut result = _mm_cvtsd_f64(final_sum);

    // Process remaining elements
    for item in &a[simd_len..] {
        result += item;
    }

    result
}

/// AVX2-optimized fused multiply-add for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_fma_f32(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Process 8 f32 elements per iteration using AVX2 + FMA
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vc = _mm256_loadu_ps(c.as_ptr().add(i));
        let vresult = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] * b[i] + c[i];
    }
}

/// AVX2-optimized fused multiply-add for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_fma_f64(a: &[f64], b: &[f64], c: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), result.len());

    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Process 4 f64 elements per iteration using AVX2 + FMA
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let vc = _mm256_loadu_pd(c.as_ptr().add(i));
        let vresult = _mm256_fmadd_pd(va, vb, vc);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), vresult);
    }

    // Process remaining elements
    for i in simd_len..len {
        result[i] = a[i] * b[i] + c[i];
    }
}

/// AVX2-optimized dot product for f32 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let simd_len = len & !7; // Process 8 elements at a time

    // Initialize accumulator
    let mut vacc = _mm256_setzero_ps();

    // Process 8 f32 elements per iteration using AVX2 + FMA
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }

    // Horizontal sum of the accumulator
    let hi128 = _mm256_extractf128_ps(vacc, 1);
    let lo128 = _mm256_castps256_ps128(vacc);
    let sum128 = _mm_add_ps(hi128, lo128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x1B);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    let final_sum = _mm_add_ps(sums, shuf2);
    let mut result = _mm_cvtss_f32(final_sum);

    // Process remaining elements
    for i in simd_len..len {
        result += a[i] * b[i];
    }

    result
}

/// AVX2-optimized dot product for f64 arrays
///
/// # Safety
///
/// This function requires AVX2 support. The caller must ensure:
/// - The target CPU supports AVX2 instructions
/// - All arrays have the same length
/// - Memory alignment requirements are met for optimal performance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let simd_len = len & !3; // Process 4 elements at a time

    // Initialize accumulator
    let mut vacc = _mm256_setzero_pd();

    // Process 4 f64 elements per iteration using AVX2 + FMA
    for i in (0..simd_len).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        vacc = _mm256_fmadd_pd(va, vb, vacc);
    }

    // Horizontal sum of the accumulator
    let hi128 = _mm256_extractf128_pd(vacc, 1);
    let lo128 = _mm256_castpd256_pd128(vacc);
    let sum128 = _mm_add_pd(hi128, lo128);
    let shuf = _mm_shuffle_pd(sum128, sum128, 0x01);
    let final_sum = _mm_add_pd(sum128, shuf);
    let mut result = _mm_cvtsd_f64(final_sum);

    // Process remaining elements
    for i in simd_len..len {
        result += a[i] * b[i];
    }

    result
}
