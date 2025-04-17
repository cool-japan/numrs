//! Parallel processing optimization
//!
//! This module provides functionality for optimizing parallel processing
//! to improve performance for numerical operations.

pub mod threshold;
pub mod scheduling;
pub mod workload;

// Re-export the main functions for convenience
pub use threshold::{ParallelizationThreshold, adaptive_threshold, get_optimal_threshold, set_global_threshold};
pub use scheduling::{SchedulingStrategy, optimize_scheduling, work_stealing_scheduler};
pub use workload::{WorkloadPartitioning, partition_workload};

/// Helper function to optimize parallel computation in one call
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element (in arbitrary units)
/// * `scheduling` - The scheduling strategy to use
///
/// # Returns
///
/// The optimal number of threads to use for the computation
pub fn optimize_parallel_computation(
    array_size: usize,
    element_cost: f64,
    scheduling: SchedulingStrategy,
) -> usize {
    // Get the adaptive threshold based on array size and element cost
    let threshold = adaptive_threshold(array_size, element_cost);
    
    // If the array is smaller than the threshold, use a single thread
    if array_size <= threshold {
        return 1;
    }
    
    // Otherwise, determine the optimal number of threads
    let num_threads = rayon::current_num_threads();
    let optimal_threads = optimize_scheduling(array_size, element_cost, scheduling, num_threads);
    
    optimal_threads
}

/// Configuration for parallel processing
#[derive(Debug, Clone, Copy)]
pub struct ParallelConfig {
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Minimum array size for parallelization
    pub min_parallel_size: usize,
    /// Optimal chunk size for parallel processing
    pub chunk_size: usize,
    /// Maximum number of threads to use
    pub max_threads: Option<usize>,
    /// Scheduling strategy
    pub scheduling_strategy: SchedulingStrategy,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            use_parallel: true,
            min_parallel_size: 1000, // Default threshold
            chunk_size: 250,         // Default chunk size
            max_threads: None,       // Use all available threads by default
            scheduling_strategy: SchedulingStrategy::Adaptive,
        }
    }
}

impl ParallelConfig {
    /// Create a new ParallelConfig with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a new ParallelConfig optimized for the given array size and element cost
    pub fn optimized(array_size: usize, element_cost: f64) -> Self {
        let threshold = adaptive_threshold(array_size, element_cost);
        let chunk_size = (threshold / 4).max(100); // A reasonable chunk size derived from threshold
        
        Self {
            use_parallel: array_size >= threshold,
            min_parallel_size: threshold,
            chunk_size,
            max_threads: None,
            scheduling_strategy: SchedulingStrategy::Adaptive,
        }
    }
    
    /// Set whether to use parallel processing
    pub fn with_parallel(mut self, use_parallel: bool) -> Self {
        self.use_parallel = use_parallel;
        self
    }
    
    /// Set the minimum array size for parallelization
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_parallel_size = min_size;
        self
    }
    
    /// Set the chunk size for parallel processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
    
    /// Set the maximum number of threads to use
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = Some(max_threads);
        self
    }
    
    /// Set the scheduling strategy
    pub fn with_scheduling(mut self, strategy: SchedulingStrategy) -> Self {
        self.scheduling_strategy = strategy;
        self
    }
    
    /// Should this computation be parallelized based on the configuration?
    pub fn should_parallelize(&self, array_size: usize) -> bool {
        self.use_parallel && array_size >= self.min_parallel_size
    }
    
    /// Get the optimal number of threads to use
    pub fn optimal_threads(&self, array_size: usize, element_cost: f64) -> usize {
        if !self.should_parallelize(array_size) {
            return 1;
        }
        
        let available_threads = rayon::current_num_threads();
        let mut optimal = optimize_scheduling(
            array_size, 
            element_cost, 
            self.scheduling_strategy, 
            available_threads
        );
        
        // Limit to max_threads if specified
        if let Some(max) = self.max_threads {
            optimal = optimal.min(max);
        }
        
        optimal
    }
}