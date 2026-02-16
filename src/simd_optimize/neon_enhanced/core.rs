//! Core types and constants for NEON SIMD operations
//!
//! This module provides the foundational types and constants for ARM NEON
//! SIMD operations on ARM processors including Apple Silicon and ARM servers.

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

/// NEON vectorization constants
pub const NEON_F32_LANES: usize = 4;
pub const NEON_F64_LANES: usize = 2;
#[allow(dead_code)]
pub const NEON_ALIGNMENT: usize = 16;

/// Vectorization constants for f64 (unrolled)
#[allow(dead_code)]
pub const NEON_F64_LANES_UNROLL: usize = NEON_F64_LANES * 4; // Process 8 f64 values at once

/// Advanced NEON operations for ARM processors
pub struct NeonEnhancedOps;

/// NEON feature detection for ARM processors
pub struct NeonFeatureDetector;

impl NeonFeatureDetector {
    /// Detect available NEON features
    pub fn detect_neon_features() -> NeonFeatures {
        #[allow(unused_mut)] // False positive - modified in conditional compilation blocks
        let mut features = NeonFeatures::default();

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is standard on AArch64
            features.neon = true;

            if is_aarch64_feature_detected!("asimd") {
                features.asimd = true;
            }
            if is_aarch64_feature_detected!("fp") {
                features.fp = true;
            }
        }

        features
    }

    /// Get optimal block size for ARM processors
    pub fn optimal_block_size() -> usize {
        // ARM processors typically have smaller caches
        32
    }
}

#[derive(Debug, Default, Clone)]
pub struct NeonFeatures {
    pub neon: bool,  // Basic NEON support
    pub asimd: bool, // Advanced SIMD
    pub fp: bool,    // Floating point support
}

impl NeonFeatures {
    pub fn has_full_support(&self) -> bool {
        self.neon && self.asimd && self.fp
    }

    pub fn recommended_operations(&self) -> Vec<&'static str> {
        let mut ops = Vec::new();

        if self.neon {
            ops.push("Basic vectorization");
            ops.push("Integer operations");
        }
        if self.asimd {
            ops.push("Advanced SIMD operations");
            ops.push("Crypto operations");
        }
        if self.fp {
            ops.push("Floating point operations");
            ops.push("Vector math");
        }

        ops
    }
}
