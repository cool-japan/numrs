//! SIMD implementation selection based on CPU features
//!
//! This module provides functionality for selecting the most efficient
//! SIMD implementation based on the detected CPU features.

use crate::simd_optimize::feature_detect::CpuFeatures;

/// Represents the available SIMD implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdImplementation {
    /// Generic scalar implementation (no SIMD)
    Scalar,
    /// SSE implementation (x86_64)
    SSE,
    /// AVX implementation (x86_64)
    AVX,
    /// AVX2 implementation (x86_64)
    AVX2,
    /// AVX-512 implementation (x86_64)
    AVX512,
    /// NEON implementation (aarch64)
    NEON,
    /// SVE implementation (aarch64)
    SVE,
}

impl SimdImplementation {
    /// Get the name of the SIMD implementation
    pub fn name(&self) -> &'static str {
        match self {
            SimdImplementation::Scalar => "Scalar",
            SimdImplementation::SSE => "SSE",
            SimdImplementation::AVX => "AVX",
            SimdImplementation::AVX2 => "AVX2",
            SimdImplementation::AVX512 => "AVX512",
            SimdImplementation::NEON => "NEON",
            SimdImplementation::SVE => "SVE",
        }
    }
    
    /// Check if the implementation is AVX2 or better
    pub fn is_avx2_or_better(&self) -> bool {
        matches!(self, SimdImplementation::AVX2 | SimdImplementation::AVX512)
    }
    
    /// Check if the implementation supports FMA operations
    pub fn supports_fma(&self, features: &CpuFeatures) -> bool {
        match self {
            SimdImplementation::AVX2 | SimdImplementation::AVX512 => features.fma,
            _ => false,
        }
    }
    
    /// Check if the implementation is NEON or better
    pub fn is_neon_or_better(&self) -> bool {
        matches!(self, SimdImplementation::NEON | SimdImplementation::SVE)
    }
}

/// Select the most efficient SIMD implementation based on CPU features
///
/// # Arguments
///
/// * `features` - The detected CPU features
///
/// # Returns
///
/// The selected SIMD implementation
pub fn select_simd_implementation(features: &CpuFeatures) -> SimdImplementation {
    // Check for x86_64 features in order of preference
    if features.avx512f {
        return SimdImplementation::AVX512;
    }
    
    if features.avx2 {
        return SimdImplementation::AVX2;
    }
    
    if features.avx {
        return SimdImplementation::AVX;
    }
    
    if features.sse2 {
        return SimdImplementation::SSE;
    }
    
    // Check for aarch64 features
    if features.sve {
        return SimdImplementation::SVE;
    }
    
    if features.neon {
        return SimdImplementation::NEON;
    }
    
    // Fall back to scalar implementation
    SimdImplementation::Scalar
}

/// Apply a specific SIMD strategy based on CPU features
///
/// This is a helper function to facilitate selecting different implementations
/// based on the available CPU features.
///
/// # Arguments
///
/// * `features` - The detected CPU features
/// * `scalar` - The scalar implementation to use if SIMD is not available
/// * `sse` - The SSE implementation to use if SSE is available
/// * `avx` - The AVX implementation to use if AVX is available
/// * `avx2` - The AVX2 implementation to use if AVX2 is available
/// * `avx512` - The AVX-512 implementation to use if AVX-512 is available
/// * `neon` - The NEON implementation to use if NEON is available
/// * `sve` - The SVE implementation to use if SVE is available
///
/// # Returns
///
/// The result of the selected implementation
pub fn apply_simd_strategy<T, S, SSE, AVX, AVX2, AVX512, NEON, SVE>(
    features: &CpuFeatures,
    scalar: S,
    sse: SSE,
    avx: AVX,
    avx2: AVX2,
    avx512: AVX512,
    neon: NEON,
    sve: SVE,
) -> T
where
    S: FnOnce() -> T,
    SSE: FnOnce() -> T,
    AVX: FnOnce() -> T,
    AVX2: FnOnce() -> T,
    AVX512: FnOnce() -> T,
    NEON: FnOnce() -> T,
    SVE: FnOnce() -> T,
{
    let implementation = select_simd_implementation(features);
    
    match implementation {
        SimdImplementation::AVX512 => avx512(),
        SimdImplementation::AVX2 => avx2(),
        SimdImplementation::AVX => avx(),
        SimdImplementation::SSE => sse(),
        SimdImplementation::NEON => neon(),
        SimdImplementation::SVE => sve(),
        SimdImplementation::Scalar => scalar(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_selection() {
        // Test with no features
        let features = CpuFeatures::default();
        assert_eq!(select_simd_implementation(&features), SimdImplementation::Scalar);
        
        // Test with SSE2 only
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        assert_eq!(select_simd_implementation(&features), SimdImplementation::SSE);
        
        // Test with AVX
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        features.avx = true;
        assert_eq!(select_simd_implementation(&features), SimdImplementation::AVX);
        
        // Test with AVX2
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        features.avx = true;
        features.avx2 = true;
        assert_eq!(select_simd_implementation(&features), SimdImplementation::AVX2);
        
        // Test with all x86_64 features
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        features.avx = true;
        features.avx2 = true;
        features.avx512f = true;
        assert_eq!(select_simd_implementation(&features), SimdImplementation::AVX512);
        
        // Test with NEON
        let mut features = CpuFeatures::default();
        features.neon = true;
        assert_eq!(select_simd_implementation(&features), SimdImplementation::NEON);
        
        // Test with SVE
        let mut features = CpuFeatures::default();
        features.neon = true;
        features.sve = true;
        assert_eq!(select_simd_implementation(&features), SimdImplementation::SVE);
    }
    
    #[test]
    fn test_simd_strategy() {
        // Create a test function that returns a string based on the implementation
        let apply_test = |features: &CpuFeatures| {
            apply_simd_strategy(
                features,
                || "scalar",
                || "sse",
                || "avx",
                || "avx2",
                || "avx512",
                || "neon",
                || "sve",
            )
        };
        
        // Test with no features
        let features = CpuFeatures::default();
        assert_eq!(apply_test(&features), "scalar");
        
        // Test with SSE2 only
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        assert_eq!(apply_test(&features), "sse");
        
        // Test with AVX
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        features.avx = true;
        assert_eq!(apply_test(&features), "avx");
        
        // Test with AVX2
        let mut features = CpuFeatures::default();
        features.sse2 = true;
        features.avx = true;
        features.avx2 = true;
        assert_eq!(apply_test(&features), "avx2");
    }
}
