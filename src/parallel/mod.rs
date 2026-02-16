//! Parallel processing enhancements and workload balancing
//!
//! This module provides advanced parallel processing capabilities including:
//! - **Thread Pool**: Enhanced work-stealing thread pool with adaptive scheduling
//! - **Scheduler**: Priority-based task scheduler with profiling and cost estimation
//! - **Load Balancer**: Dynamic load balancing with NUMA awareness
//! - **Work Stealing**: High-performance work-stealing implementation
//! - **Parallel Algorithms**: Optimized parallel implementations (sort, scan, reduce, map-reduce, pipeline)
//!
//! # Features
//!
//! ## Work-Stealing Thread Pool
//! - Per-thread work-stealing deques for efficient load distribution
//! - Thread affinity and CPU pinning support
//! - Adaptive thread count based on workload
//! - Priority-based task scheduling
//!
//! ## Advanced Scheduling
//! - Adaptive task granularity (automatic splitting of large tasks)
//! - Task profiling and cost estimation
//! - Dynamic load balancing
//! - Task cancellation support
//!
//! ## NUMA Awareness
//! - NUMA-aware allocation and scheduling
//! - Work stealing across NUMA nodes
//! - Memory locality optimization
//!
//! ## Parallel Algorithms
//! - Parallel sort (merge sort, quick sort)
//! - Parallel scan (prefix sum)
//! - Parallel reduction with custom operations
//! - Parallel map-reduce operations
//! - Parallel pipeline processing
//!
//! # Example
//!
//! ```no_run
//! use numrs2::parallel::{ThreadPool, ParallelArrayOps, ParallelConfig};
//!
//! // Create a thread pool
//! let pool = ThreadPool::new().expect("Failed to create thread pool");
//!
//! // Submit tasks
//! for i in 0..10 {
//!     pool.submit(move || {
//!         println!("Task {} executing", i);
//!     }).expect("Failed to submit task");
//! }
//!
//! // Wait for completion
//! pool.wait().expect("Failed to wait");
//!
//! // Use parallel algorithms
//! let config = ParallelConfig::default();
//! let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
//!
//! let data = vec![5, 2, 8, 1, 9];
//! let mut sorted = data.clone();
//! ops.parallel_sort(&mut sorted).expect("Failed to sort");
//! ```

// Core parallel computing modules
pub mod load_balancer;
pub mod parallel_algorithms;
pub mod parallel_allocator;
pub mod scheduler;
pub mod thread_pool;
pub mod work_stealing;

// Re-export load balancer types (NUMA-aware dynamic load balancing)
pub use load_balancer::{
    BalancingStrategy, LoadBalancer, LoadBalancingAdvisor, LoadBalancingAnalysis, WorkloadMetrics,
};

// Re-export parallel algorithm types (map/reduce/filter/sort/scan)
pub use parallel_algorithms::{
    parallel_prefix_sum, parallel_scan, ParallelArrayOps, ParallelConfig, ParallelFFT,
    ParallelMatrixOps, ParallelPipeline, ParallelQuickSort, ScanMode,
};

// Re-export allocator types (parallel memory management)
pub use parallel_allocator::{ParallelAllocator, ParallelAllocatorConfig, ThreadLocalAllocator};

// Re-export scheduler types (adaptive task scheduler)
pub use scheduler::{ParallelScheduler, SchedulerConfig, SchedulerStats, TaskPriority};

// Re-export thread pool types (work-stealing thread pool)
pub use thread_pool::{Priority, ThreadPool, ThreadPoolConfig, ThreadPoolStats};

// Re-export work stealing types (work-stealing deque implementation)
pub use work_stealing::{task, PoolStats, Task, TaskResult, WorkStealingConfig, WorkStealingPool};

use crate::error::{NumRs2Error, Result};
use std::sync::Arc;
use std::time::Duration;

/// Global parallel execution context
pub struct ParallelContext {
    scheduler: Arc<ParallelScheduler>,
    load_balancer: Arc<LoadBalancer>,
    work_stealing_pool: Arc<WorkStealingPool>,
}

impl ParallelContext {
    /// Create a new parallel context with optimal configuration for the system
    pub fn new() -> Result<Self> {
        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let scheduler_config = SchedulerConfig::optimal_for_cores(num_cores);
        let scheduler = Arc::new(ParallelScheduler::new(scheduler_config)?);

        let load_balancer = Arc::new(LoadBalancer::new(BalancingStrategy::Adaptive, num_cores)?);

        let work_stealing_pool = Arc::new(WorkStealingPool::new(num_cores)?);

        Ok(Self {
            scheduler,
            load_balancer,
            work_stealing_pool,
        })
    }

    /// Create a parallel context with custom configuration
    pub fn with_config(
        scheduler_config: SchedulerConfig,
        balancing_strategy: BalancingStrategy,
        num_threads: usize,
    ) -> Result<Self> {
        let scheduler = Arc::new(ParallelScheduler::new(scheduler_config)?);
        let load_balancer = Arc::new(LoadBalancer::new(balancing_strategy, num_threads)?);
        let work_stealing_pool = Arc::new(WorkStealingPool::new(num_threads)?);

        Ok(Self {
            scheduler,
            load_balancer,
            work_stealing_pool,
        })
    }

    /// Get the scheduler
    pub fn scheduler(&self) -> &Arc<ParallelScheduler> {
        &self.scheduler
    }

    /// Get the load balancer
    pub fn load_balancer(&self) -> &Arc<LoadBalancer> {
        &self.load_balancer
    }

    /// Get the work-stealing pool
    pub fn work_stealing_pool(&self) -> &Arc<WorkStealingPool> {
        &self.work_stealing_pool
    }

    /// Shutdown the parallel context gracefully
    pub fn shutdown(&self) -> Result<()> {
        self.work_stealing_pool.shutdown()?;
        self.scheduler.shutdown()?;
        Ok(())
    }

    /// Get current workload statistics
    pub fn workload_stats(&self) -> WorkloadMetrics {
        self.load_balancer.current_metrics()
    }
}

impl Default for ParallelContext {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback: create a minimal parallel context
            // This should rarely happen, but prevents panics
            let num_cores = 1; // Conservative fallback
            let scheduler_config = SchedulerConfig::optimal_for_cores(num_cores);

            // Create components with minimal configuration
            // If any of these fail, we have a serious system issue
            let scheduler = ParallelScheduler::new(scheduler_config)
                .ok()
                .map(Arc::new)
                .unwrap_or_else(|| {
                    // Last resort: create a scheduler with absolute minimal config
                    Arc::new(
                        ParallelScheduler::new(SchedulerConfig {
                            num_threads: 1,
                            max_queue_size: 100,
                            enable_thread_affinity: false,
                            enable_adaptive_scheduling: false,
                            time_slice_ms: 10,
                            work_stealing_threshold: 5,
                            cache_aware_scheduling: false,
                        })
                        .unwrap_or_else(|_| {
                            panic!("Cannot create even minimal ParallelScheduler - system unusable")
                        }),
                    )
                });

            let load_balancer = LoadBalancer::new(BalancingStrategy::Adaptive, num_cores)
                .ok()
                .map(Arc::new)
                .unwrap_or_else(|| panic!("Cannot create LoadBalancer - system unusable"));

            let work_stealing_pool = WorkStealingPool::new(num_cores)
                .ok()
                .map(Arc::new)
                .unwrap_or_else(|| panic!("Cannot create WorkStealingPool - system unusable"));

            Self {
                scheduler,
                load_balancer,
                work_stealing_pool,
            }
        })
    }
}

lazy_static::lazy_static! {
    /// Thread-safe global parallel context instance
    static ref GLOBAL_PARALLEL_CONTEXT: std::sync::Mutex<Option<Arc<ParallelContext>>> =
        std::sync::Mutex::new(None);

    /// Global persistent thread pool for reuse across all parallel operations
    /// This eliminates thread creation/destruction overhead
    static ref GLOBAL_THREAD_POOL: Arc<ThreadPool> = {
        let num_threads = std::env::var("NUMRS2_THREAD_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(2); // Default to 2 threads (sweet spot from benchmarks)

        let config = ThreadPoolConfig {
            num_threads: Some(num_threads),
            enable_thread_pinning: false,
            adaptive_threads: false,
            min_threads: 1,
            max_threads: num_threads,
            queue_capacity: 10000,
            steal_interval: Duration::from_micros(100),
            idle_timeout: Duration::from_millis(100),
        };

        Arc::new(ThreadPool::with_config(config).unwrap_or_else(|_| {
            // Fallback to minimal pool
            ThreadPool::with_config(ThreadPoolConfig {
                num_threads: Some(1),
                ..Default::default()
            }).expect("Failed to create fallback thread pool")
        }))
    };
}

/// Initialize the global parallel context
pub fn initialize_parallel_context() -> Result<()> {
    let context = Arc::new(ParallelContext::new()?);
    let mut global = GLOBAL_PARALLEL_CONTEXT.lock().map_err(|e| {
        NumRs2Error::RuntimeError(format!("Failed to acquire global context lock: {}", e))
    })?;
    *global = Some(context);
    Ok(())
}

/// Get the global parallel context
pub fn global_parallel_context() -> Result<Arc<ParallelContext>> {
    let global = GLOBAL_PARALLEL_CONTEXT.lock().map_err(|e| {
        NumRs2Error::RuntimeError(format!("Failed to acquire global context lock: {}", e))
    })?;
    global.clone().ok_or_else(|| {
        NumRs2Error::RuntimeError("Global parallel context not initialized".to_string())
    })
}

/// Shutdown the global parallel context
pub fn shutdown_parallel_context() -> Result<()> {
    let mut global = GLOBAL_PARALLEL_CONTEXT.lock().map_err(|e| {
        NumRs2Error::RuntimeError(format!("Failed to acquire global context lock: {}", e))
    })?;
    if let Some(context) = global.take() {
        context.shutdown()?;
    }
    Ok(())
}

/// Get the global persistent thread pool
///
/// This pool is initialized once and reused across all parallel operations
/// to eliminate thread creation/destruction overhead.
pub fn global_thread_pool() -> Arc<ThreadPool> {
    Arc::clone(&GLOBAL_THREAD_POOL)
}

/// Get the number of threads in the global thread pool
pub fn global_thread_count() -> usize {
    GLOBAL_THREAD_POOL.num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_context_creation() {
        let context =
            ParallelContext::new().expect("ParallelContext creation should succeed in test");
        assert!(context.scheduler.num_threads() > 0);
        assert!(context.load_balancer.num_workers() > 0);
    }

    #[test]
    fn test_global_context_initialization() {
        initialize_parallel_context().expect("initialize_parallel_context should succeed in test");
        let context =
            global_parallel_context().expect("global_parallel_context should succeed in test");
        assert!(context.scheduler.num_threads() > 0);
        shutdown_parallel_context().expect("shutdown_parallel_context should succeed in test");
    }

    #[test]
    fn test_workload_stats() {
        let context =
            ParallelContext::new().expect("ParallelContext creation should succeed in test");
        let stats = context.workload_stats();
        assert_eq!(stats.active_tasks, 0);
        assert!(stats.total_throughput >= 0.0);
    }
}
