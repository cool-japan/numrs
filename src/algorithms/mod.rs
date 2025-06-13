//! High-performance numerical algorithms
//!
//! This module contains optimized implementations of numerical algorithms
//! with special attention to cache efficiency, memory layout, and SIMD utilization.

pub mod cache_aware;

// Re-export main types and functions
pub use cache_aware::{
    CacheAwareArrayOps, CacheAwareFFT, CacheAwareConvolution, BandwidthOptimizer,
    MemoryOperation, BandwidthEstimate,
};