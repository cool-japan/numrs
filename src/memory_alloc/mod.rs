//! Custom memory allocation for numerical workloads
//!
//! This module provides specialized memory allocators optimized for
//! numerical computing workloads.

pub mod aligned;
pub mod arena;
pub mod benchmarking;
pub mod cache_optimization;
pub mod enhanced_traits;
pub mod large_scale;
pub mod monitored_allocator;
pub mod out_of_core;
pub mod performance_tuning;
pub mod pool;
pub mod strategy;

// Re-export the main types and functions for convenience
pub use aligned::{AlignedAllocator, AlignmentConfig};
pub use arena::{ArenaAllocator, ArenaConfig};
pub use benchmarking::{
    AllocatorBenchmark, BenchmarkConfig, BenchmarkResults, benchmark_configs,
};
pub use cache_optimization::{
    CacheOptimizedAllocator, CacheConfig, CacheLevel, CacheMetrics, 
    CacheOptimizationRecommendation, CacheOptimizationType, cache_constants,
};
pub use enhanced_traits::{
    EnhancedAllocatorBridge, IntelligentAllocationStrategy, NumericalArrayAllocator,
};
pub use large_scale::{
    get_global_memory_stats, get_global_spill_stats, init_global_manager,
    load_spilled_data_globally, should_spill_globally, spill_data_globally, with_global_manager,
    with_global_manager_mut, ChunkIterator, LargeScaleConfig, LargeScaleManager, MemoryStats,
    MemoryTracker, SpillStats,
};
pub use monitored_allocator::{
    MonitoredAllocator, MonitoredAllocatorFactory, MonitoringConfig, presets,
};
pub use out_of_core::{CacheStats, CacheStrategy, OutOfCoreArray, OutOfCoreConfig};
pub use performance_tuning::{
    PerformanceTuner, PerformanceMetrics, OptimizationRecommendation, OptimizationType, TuningConfig,
    init_global_tuner, with_global_tuner, with_global_tuner_mut,
};
pub use pool::{PoolAllocator, PoolConfig};
pub use strategy::{get_default_allocator, AllocStrategy, MemoryAllocator};

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
