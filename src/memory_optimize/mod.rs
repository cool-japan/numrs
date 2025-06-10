//! Memory layout optimization for cache efficiency
//!
//! This module provides functionality for optimizing memory layout to improve
//! cache efficiency and overall performance of numerical operations.

pub mod alignment;
pub mod cache_layout;
pub mod memory_placement;

// Re-export the main functions for convenience
pub use alignment::{align_data, AlignmentStrategy};
pub use cache_layout::{optimize_layout, LayoutStrategy};
pub use memory_placement::{optimize_placement, PlacementStrategy};

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
