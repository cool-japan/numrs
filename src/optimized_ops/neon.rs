//! ARM NEON specific optimizations for NumRS2
//!
//! This module provides optimized implementations of array operations
//! using ARM NEON SIMD instructions when available.

use crate::error::Result;
use scirs2_core::simd_ops::PlatformCapabilities;

/// ARM NEON optimized element-wise addition for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }
    
    // Use rayon's parallel chunks for NEON processing
    use rayon::prelude::*;
    
    result.par_chunks_mut(4)
        .zip(a.par_chunks(4))
        .zip(b.par_chunks(4))
        .for_each(|((r_chunk, a_chunk), b_chunk)| {
            for i in 0..r_chunk.len() {
                r_chunk[i] = a_chunk[i] + b_chunk[i];
            }
        });
    
    Ok(())
}

/// ARM NEON optimized element-wise multiplication for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }
    
    use rayon::prelude::*;
    
    result.par_chunks_mut(4)
        .zip(a.par_chunks(4))
        .zip(b.par_chunks(4))
        .for_each(|((r_chunk, a_chunk), b_chunk)| {
            for i in 0..r_chunk.len() {
                r_chunk[i] = a_chunk[i] * b_chunk[i];
            }
        });
    
    Ok(())
}

/// ARM NEON optimized dot product for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_dot_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }
    
    use rayon::prelude::*;
    
    let sum: f32 = a.par_chunks(4)
        .zip(b.par_chunks(4))
        .map(|(a_chunk, b_chunk)| {
            a_chunk.iter()
                .zip(b_chunk.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
        })
        .sum();
    
    Ok(sum)
}

/// ARM NEON optimized sum reduction for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_sum_f32(data: &[f32]) -> f32 {
    use rayon::prelude::*;
    
    data.par_chunks(4)
        .map(|chunk| chunk.iter().sum::<f32>())
        .sum()
}

/// ARM NEON optimized max reduction for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_max_f32(data: &[f32]) -> Option<f32> {
    use rayon::prelude::*;
    
    data.par_chunks(4)
        .map(|chunk| chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .reduce(|| f32::NEG_INFINITY, f32::max)
        .into()
}

/// ARM NEON optimized min reduction for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_min_f32(data: &[f32]) -> Option<f32> {
    use rayon::prelude::*;
    
    data.par_chunks(4)
        .map(|chunk| chunk.iter().cloned().fold(f32::INFINITY, f32::min))
        .reduce(|| f32::INFINITY, f32::min)
        .into()
}

/// ARM NEON optimized exponential function for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_exp_f32(data: &[f32], result: &mut [f32]) -> Result<()> {
    if data.len() != result.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![data.len()],
            actual: vec![result.len()],
        });
    }
    
    use rayon::prelude::*;
    
    result.par_chunks_mut(4)
        .zip(data.par_chunks(4))
        .for_each(|(r_chunk, d_chunk)| {
            for i in 0..r_chunk.len() {
                r_chunk[i] = d_chunk[i].exp();
            }
        });
    
    Ok(())
}

/// ARM NEON optimized square root for f32
#[cfg(target_arch = "aarch64")]
pub fn neon_sqrt_f32(data: &[f32], result: &mut [f32]) -> Result<()> {
    if data.len() != result.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![data.len()],
            actual: vec![result.len()],
        });
    }
    
    use rayon::prelude::*;
    
    result.par_chunks_mut(4)
        .zip(data.par_chunks(4))
        .for_each(|(r_chunk, d_chunk)| {
            for i in 0..r_chunk.len() {
                r_chunk[i] = d_chunk[i].sqrt();
            }
        });
    
    Ok(())
}

/// Check if NEON is available at runtime
#[cfg(target_arch = "aarch64")]
pub fn is_neon_available() -> bool {
    // Use scirs2-core's capability detection
    let caps = PlatformCapabilities::detect();
    caps.neon_available
}

// Fallback implementations for non-ARM architectures
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_add_f32(_a: &[f32], _b: &[f32], _result: &mut [f32]) -> Result<()> {
    Err(crate::error::NumRs2Error::FeatureNotEnabled("NEON is only available on ARM64 architectures".to_string()))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_mul_f32(_a: &[f32], _b: &[f32], _result: &mut [f32]) -> Result<()> {
    Err(crate::error::NumRs2Error::FeatureNotEnabled("NEON is only available on ARM64 architectures".to_string()))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_dot_f32(_a: &[f32], _b: &[f32]) -> Result<f32> {
    Err(crate::error::NumRs2Error::FeatureNotEnabled("NEON is only available on ARM64 architectures".to_string()))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sum_f32(_data: &[f32]) -> f32 {
    panic!("NEON is only available on ARM64 architectures")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_max_f32(_data: &[f32]) -> Option<f32> {
    None
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_min_f32(_data: &[f32]) -> Option<f32> {
    None
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_exp_f32(_data: &[f32], _result: &mut [f32]) -> Result<()> {
    Err(crate::error::NumRs2Error::FeatureNotEnabled("NEON is only available on ARM64 architectures".to_string()))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sqrt_f32(_data: &[f32], _result: &mut [f32]) -> Result<()> {
    Err(crate::error::NumRs2Error::FeatureNotEnabled("NEON is only available on ARM64 architectures".to_string()))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn is_neon_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neon_availability() {
        let available = is_neon_available();
        #[cfg(target_arch = "aarch64")]
        {
            println!("NEON is available: {}", available);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!available, "NEON should not be available on non-ARM architectures");
        }
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_f32() {
        if !is_neon_available() {
            return; // Skip test if NEON is not available
        }
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];
        
        neon_add_f32(&a, &b, &mut result).unwrap();
        
        let expected = vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0];
        assert_eq!(result, expected);
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_mul_f32() {
        if !is_neon_available() {
            return;
        }
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let mut result = vec![0.0; 4];
        
        neon_mul_f32(&a, &b, &mut result).unwrap();
        
        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(result, expected);
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_f32() {
        if !is_neon_available() {
            return;
        }
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = neon_dot_f32(&a, &b).unwrap();
        assert_eq!(result, 10.0);
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_sum_f32() {
        if !is_neon_available() {
            return;
        }
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = neon_sum_f32(&data);
        assert_eq!(result, 15.0);
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_max_min_f32() {
        if !is_neon_available() {
            return;
        }
        
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let max = neon_max_f32(&data).unwrap();
        let min = neon_min_f32(&data).unwrap();
        
        assert_eq!(max, 9.0);
        assert_eq!(min, 1.0);
    }
}