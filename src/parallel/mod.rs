//! Parallel processing enhancements and workload balancing
//!
//! This module provides advanced parallel processing capabilities including
//! work-stealing schedulers, dynamic load balancing, and parallel algorithms
//! optimized for numerical computations.

pub mod load_balancer;
pub mod parallel_algorithms;
pub mod parallel_allocator;
pub mod scheduler;
pub mod work_stealing;

// Re-export main types
pub use load_balancer::{BalancingStrategy, LoadBalancer, WorkloadMetrics};
pub use parallel_algorithms::{ParallelArrayOps, ParallelConfig, ParallelFFT, ParallelMatrixOps};
pub use parallel_allocator::{ParallelAllocator, ParallelAllocatorConfig, ThreadLocalAllocator};
pub use scheduler::{ParallelScheduler, SchedulerConfig, TaskPriority};
pub use work_stealing::{task, Task, TaskResult, WorkStealingPool};

use crate::error::{NumRs2Error, Result};
use std::sync::Arc;

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
        Self::new().expect("Failed to create default parallel context")
    }
}

lazy_static::lazy_static! {
    /// Thread-safe global parallel context instance
    static ref GLOBAL_PARALLEL_CONTEXT: std::sync::Mutex<Option<Arc<ParallelContext>>> =
        std::sync::Mutex::new(None);
}

/// Initialize the global parallel context
pub fn initialize_parallel_context() -> Result<()> {
    let context = Arc::new(ParallelContext::new()?);
    let mut global = GLOBAL_PARALLEL_CONTEXT.lock().unwrap();
    *global = Some(context);
    Ok(())
}

/// Get the global parallel context
pub fn global_parallel_context() -> Result<Arc<ParallelContext>> {
    let global = GLOBAL_PARALLEL_CONTEXT.lock().unwrap();
    global.clone().ok_or_else(|| {
        NumRs2Error::RuntimeError("Global parallel context not initialized".to_string())
    })
}

/// Shutdown the global parallel context
pub fn shutdown_parallel_context() -> Result<()> {
    let mut global = GLOBAL_PARALLEL_CONTEXT.lock().unwrap();
    if let Some(context) = global.take() {
        context.shutdown()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_context_creation() {
        let context = ParallelContext::new().unwrap();
        assert!(context.scheduler.num_threads() > 0);
        assert!(context.load_balancer.num_workers() > 0);
    }

    #[test]
    fn test_global_context_initialization() {
        initialize_parallel_context().unwrap();
        let context = global_parallel_context().unwrap();
        assert!(context.scheduler.num_threads() > 0);
        shutdown_parallel_context().unwrap();
    }

    #[test]
    fn test_workload_stats() {
        let context = ParallelContext::new().unwrap();
        let stats = context.workload_stats();
        assert_eq!(stats.active_tasks, 0);
        assert!(stats.total_throughput >= 0.0);
    }
}
