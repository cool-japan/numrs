//! CPU feature detection and SIMD optimization for NumRS
//!
//! This module provides functionality for detecting available CPU features
//! and selecting the most efficient SIMD implementation for the current hardware.

pub mod avx2_enhanced;
pub mod avx2_ops;
#[cfg(feature = "unstable")]
pub mod avx512_enhanced;
pub mod feature_detect;
pub mod neon_enhanced;
pub mod simd_select;
pub mod simd_traits;
pub mod unified_dispatcher;

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

// Re-export the main functions for convenience
pub use feature_detect::{detect_cpu_features, CpuFeatures};
pub use simd_select::{select_simd_implementation, SimdImplementation};
pub use simd_traits::SimdPerformanceHints;
pub use unified_dispatcher::{global_dispatcher, optimized, UnifiedSimdDispatcher};

/// CPU feature detection and SIMD implementation selection in one step
///
/// # Returns
///
/// The selected SIMD implementation based on detected CPU features
pub fn detect_and_select() -> SimdImplementation {
    let features = detect_cpu_features();
    select_simd_implementation(&features)
}

/// AVX2-optimized array addition for f32
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_add_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_add_f32(&a_data, &b_data, &mut result_data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86_64
            for ((a, b), r) in a_data.iter().zip(b_data.iter()).zip(result_data.iter_mut()) {
                *r = a + b;
            }
        }
    }

    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// Fallback for non-x86_64 systems
#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_add_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a + b)
        .collect();
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// AVX2-optimized array addition for f64
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_add_f64(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_add_f64(&a_data, &b_data, &mut result_data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for ((a, b), r) in a_data.iter().zip(b_data.iter()).zip(result_data.iter_mut()) {
                *r = a + b;
            }
        }
    }

    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// AVX2-optimized array multiplication for f32
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_mul_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_mul_f32(&a_data, &b_data, &mut result_data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for ((a, b), r) in a_data.iter().zip(b_data.iter()).zip(result_data.iter_mut()) {
                *r = a * b;
            }
        }
    }

    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// AVX2-optimized array multiplication for f64
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_mul_f64(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_mul_f64(&a_data, &b_data, &mut result_data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for ((a, b), r) in a_data.iter().zip(b_data.iter()).zip(result_data.iter_mut()) {
                *r = a * b;
            }
        }
    }

    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// AVX2-optimized square root for f32
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_sqrt_f32(a: &Array<f32>) -> Array<f32> {
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f32; a_data.len()];

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_sqrt_f32(&a_data, &mut result_data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for (a, r) in a_data.iter().zip(result_data.iter_mut()) {
                *r = a.sqrt();
            }
        }
    }

    Array::from_vec(result_data).reshape(&a.shape())
}

/// AVX2-optimized square root for f64
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_sqrt_f64(a: &Array<f64>) -> Array<f64> {
    let a_data = a.to_vec();
    let mut result_data = vec![0.0f64; a_data.len()];

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_sqrt_f64(&a_data, &mut result_data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for (a, r) in a_data.iter().zip(result_data.iter_mut()) {
                *r = a.sqrt();
            }
        }
    }

    Array::from_vec(result_data).reshape(&a.shape())
}

/// AVX2-optimized sum for f32
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_sum_f32(a: &Array<f32>) -> f32 {
    let a_data = a.to_vec();

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            avx2_ops::avx2_sum_f32(&a_data)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            a_data.iter().sum()
        }
    }
}

/// AVX2-optimized sum for f64
#[cfg(target_arch = "x86_64")]
pub fn avx2_optimized_sum_f64(a: &Array<f64>) -> f64 {
    let a_data = a.to_vec();
    unsafe { avx2_ops::avx2_sum_f64(&a_data) }
}

/// Fallback implementations for non-x86_64 systems
#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_add_f64(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data: Vec<f64> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a + b)
        .collect();
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_mul_f32(a: &Array<f32>, b: &Array<f32>) -> Result<Array<f32>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a * b)
        .collect();
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_mul_f64(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data: Vec<f64> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a * b)
        .collect();
    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_sqrt_f32(a: &Array<f32>) -> Array<f32> {
    let a_data = a.to_vec();
    let result_data: Vec<f32> = a_data.iter().map(|x| x.sqrt()).collect();
    Array::from_vec(result_data).reshape(&a.shape())
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_sqrt_f64(a: &Array<f64>) -> Array<f64> {
    let a_data = a.to_vec();
    let result_data: Vec<f64> = a_data.iter().map(|x| x.sqrt()).collect();
    Array::from_vec(result_data).reshape(&a.shape())
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_sum_f32(a: &Array<f32>) -> f32 {
    let a_data = a.to_vec();
    a_data.iter().sum()
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_optimized_sum_f64(a: &Array<f64>) -> f64 {
    let a_data = a.to_vec();
    a_data.iter().sum()
}
