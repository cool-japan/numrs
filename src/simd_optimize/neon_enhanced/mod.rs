//! Enhanced NEON SIMD operations for ARM processors
//!
//! This module provides optimized vectorization using ARM NEON instructions
//! for high performance on ARM-based systems including Apple Silicon and ARM servers.
//!
//! # Module Structure
//!
//! - `core` - Core types, constants, and feature detection
//! - `matmul` - Matrix multiplication operations
//! - `arithmetic` - Array and scalar arithmetic operations
//! - `exponential` - Exponential and logarithmic functions
//! - `trigonometric` - Trigonometric functions
//! - `comparison` - Comparison and sign operations
//! - `reduction` - Reduction operations (sum, prod, mean, etc.)
//! - `rounding` - Floor, ceil, and round operations
//!
//! # Example
//!
//! ```ignore
//! use numrs2::simd_optimize::neon_enhanced::{NeonEnhancedOps, NeonFeatureDetector};
//! use numrs2::array::Array;
//!
//! // Check if NEON is available
//! let features = NeonFeatureDetector::detect_neon_features();
//! if features.has_full_support() {
//!     let arr = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
//!     let sum = NeonEnhancedOps::vectorized_sum_f64(&arr);
//! }
//! ```

mod arithmetic;
mod comparison;
mod core;
mod exponential;
mod matmul;
mod reduction;
mod rounding;
mod trigonometric;

#[cfg(test)]
mod tests;

// Re-export the main types
pub use self::core::{
    NeonEnhancedOps, NeonFeatureDetector, NeonFeatures, NEON_ALIGNMENT, NEON_F32_LANES,
    NEON_F64_LANES,
};
