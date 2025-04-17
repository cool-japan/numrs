//! CPU feature detection and SIMD optimization for NumRS
//!
//! This module provides functionality for detecting available CPU features
//! and selecting the most efficient SIMD implementation for the current hardware.

pub mod feature_detect;
pub mod simd_select;

// Re-export the main functions for convenience
pub use feature_detect::{detect_cpu_features, CpuFeatures};
pub use simd_select::{select_simd_implementation, SimdImplementation};

/// CPU feature detection and SIMD implementation selection in one step
///
/// # Returns
///
/// The selected SIMD implementation based on detected CPU features
pub fn detect_and_select() -> SimdImplementation {
    let features = detect_cpu_features();
    select_simd_implementation(&features)
}
