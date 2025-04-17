//! CPU feature detection for SIMD optimization
//!
//! This module provides functionality for detecting the CPU features
//! available on the current hardware, allowing the library to select
//! the most efficient SIMD implementation.

#[cfg(target_arch = "x86_64")]
// x86_64 intrinsics would be used here in a full implementation

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Represents CPU features relevant for SIMD optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// SSE2 support (x86_64)
    pub sse2: bool,
    /// SSE3 support (x86_64)
    pub sse3: bool,
    /// SSSE3 support (x86_64)
    pub ssse3: bool,
    /// SSE4.1 support (x86_64)
    pub sse4_1: bool,
    /// SSE4.2 support (x86_64)
    pub sse4_2: bool,
    /// AVX support (x86_64)
    pub avx: bool,
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// FMA support (x86_64)
    pub fma: bool,
    /// AVX-512F support (x86_64)
    pub avx512f: bool,
    /// NEON support (aarch64)
    pub neon: bool,
    /// SVE support (aarch64)
    pub sve: bool,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        // Default to no features
        Self {
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            fma: false,
            avx512f: false,
            neon: false,
            sve: false,
        }
    }
}

impl CpuFeatures {
    /// Create a new CpuFeatures with all features enabled
    /// 
    /// This is useful for testing and benchmarking.
    pub fn all_enabled() -> Self {
        Self {
            sse2: true,
            sse3: true,
            ssse3: true,
            sse4_1: true,
            sse4_2: true,
            avx: true,
            avx2: true,
            fma: true,
            avx512f: true,
            neon: true,
            sve: true,
        }
    }
}

/// Detect CPU features on x86_64 architecture
#[cfg(target_arch = "x86_64")]
fn detect_x86_features() -> CpuFeatures {
    let mut features = CpuFeatures::default();
    
    // SSE2 is available on all x86_64 CPUs
    features.sse2 = true;
    
    // Use is_x86_feature_detected! to detect CPU features
    if std::is_x86_feature_detected!("sse3") {
        features.sse3 = true;
    }
    
    if std::is_x86_feature_detected!("ssse3") {
        features.ssse3 = true;
    }
    
    if std::is_x86_feature_detected!("sse4.1") {
        features.sse4_1 = true;
    }
    
    if std::is_x86_feature_detected!("sse4.2") {
        features.sse4_2 = true;
    }
    
    if std::is_x86_feature_detected!("avx") {
        features.avx = true;
    }
    
    if std::is_x86_feature_detected!("avx2") {
        features.avx2 = true;
    }
    
    if std::is_x86_feature_detected!("fma") {
        features.fma = true;
    }
    
    if std::is_x86_feature_detected!("avx512f") {
        features.avx512f = true;
    }
    
    features
}

/// Detect CPU features on aarch64 architecture
#[cfg(target_arch = "aarch64")]
fn detect_aarch64_features() -> CpuFeatures {
    let mut features = CpuFeatures::default();
    
    // NEON is available on all aarch64 CPUs
    features.neon = true;
    
    // Check for SVE
    #[cfg(target_feature = "sve")]
    {
        features.sve = true;
    }
    
    features
}

/// Detect CPU features on other architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_other_features() -> CpuFeatures {
    // Return default features for unsupported architectures
    CpuFeatures::default()
}

/// Detect CPU features for the current hardware
///
/// This function detects the CPU features available on the current hardware,
/// which can be used to select the most efficient SIMD implementation.
///
/// # Returns
///
/// A `CpuFeatures` struct containing the detected CPU features
pub fn detect_cpu_features() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    return detect_x86_features();
    
    #[cfg(target_arch = "aarch64")]
    return detect_aarch64_features();
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return detect_other_features();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_feature_detection() {
        let features = detect_cpu_features();
        
        #[cfg(target_arch = "x86_64")]
        {
            // SSE2 should always be available on x86_64
            assert!(features.sse2);
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // NEON should always be available on aarch64
            assert!(features.neon);
        }
    }
}