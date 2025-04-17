//! Custom memory allocation for numerical workloads
//!
//! This module provides specialized memory allocators optimized for
//! numerical computing workloads.

pub mod pool;
pub mod arena;
pub mod aligned;
pub mod strategy;

// Re-export the main types and functions for convenience
pub use pool::{PoolAllocator, PoolConfig};
pub use arena::{ArenaAllocator, ArenaConfig};
pub use aligned::{AlignedAllocator, AlignmentConfig};
pub use strategy::{AllocStrategy, MemoryAllocator, get_default_allocator};

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
