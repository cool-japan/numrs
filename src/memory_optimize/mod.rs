//! Memory layout optimization for cache efficiency
//!
//! This module provides functionality for optimizing memory layout to improve
//! cache efficiency and overall performance of numerical operations.
//!
//! # Submodules
//!
//! - `alignment` - Data alignment utilities for SIMD and cache optimization
//! - `cache_layout` - Cache-aware data layout strategies (Morton, Hilbert, blocked)
//! - `memory_placement` - Memory placement optimization (NUMA, prefetching)
//! - `access_patterns` - Memory access pattern analysis and optimization

pub mod access_patterns;
pub mod alignment;
pub mod cache_layout;
pub mod memory_placement;

// Re-export the main functions for convenience
pub use alignment::{align_data, AlignmentStrategy};
pub use cache_layout::{optimize_layout, LayoutStrategy};
pub use memory_placement::{optimize_placement, PlacementStrategy};

// Re-export access patterns types
pub use access_patterns::{
    cache_aware_binary_op, cache_aware_copy, cache_aware_transform, detect_layout, AccessPattern,
    AccessStats, Block, BlockedIterator, CacheConfig, CacheLevel, MemoryLayout, OptimizationHints,
    StrideOptimizer, Tile2D, TiledIterator2D,
};

/// Helper function to optimize memory layout and placement in one call
pub fn optimize_memory<T: Copy>(
    data: &mut [T],
    layout: LayoutStrategy,
    placement: PlacementStrategy,
) {
    // Apply layout optimization first
    optimize_layout(data, layout);

    // Then apply placement optimization
    optimize_placement(data, placement);
}
