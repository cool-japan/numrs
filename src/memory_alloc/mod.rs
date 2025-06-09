//! Custom memory allocation for numerical workloads
//!
//! This module provides specialized memory allocators optimized for
//! numerical computing workloads.

pub mod pool;
pub mod arena;
pub mod aligned;
pub mod strategy;
pub mod large_scale;
pub mod out_of_core;

// Re-export the main types and functions for convenience
pub use pool::{PoolAllocator, PoolConfig};
pub use arena::{ArenaAllocator, ArenaConfig};
pub use aligned::{AlignedAllocator, AlignmentConfig};
pub use strategy::{AllocStrategy, MemoryAllocator, get_default_allocator};
pub use large_scale::{
    LargeScaleManager, LargeScaleConfig, MemoryTracker, MemoryStats, SpillStats,
    ChunkIterator, init_global_manager, get_global_manager, should_spill_globally,
    spill_data_globally, load_spilled_data_globally, get_global_memory_stats,
    get_global_spill_stats
};
pub use out_of_core::{
    OutOfCoreArray, OutOfCoreConfig, CacheStrategy, CacheStats
};

/// Initialize the global allocator with the preferred strategy
pub fn init_global_allocator(strategy: AllocStrategy) {
    strategy::set_global_allocator(strategy);
}

/// Helper function to get the current global allocator strategy
pub fn get_global_allocator_strategy() -> AllocStrategy {
    strategy::get_global_allocator_strategy()
}

/// Helper function to reset the global allocator to the default strategy
pub fn reset_global_allocator() {
    strategy::reset_global_allocator();
}
