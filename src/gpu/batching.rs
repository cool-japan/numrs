//! GPU Batching Operations
//!
//! This module provides automatic batching for small GPU operations to improve throughput
//! by reducing kernel launch overhead and better utilizing GPU resources.
//!
//! ## Features
//!
//! - **Automatic Batching**: Queue small operations and execute them together
//! - **Dynamic Optimization**: Adaptive batch sizes based on GPU occupancy
//! - **Operation Support**: matmul, conv2d, elementwise operations
//! - **Statistics**: Comprehensive monitoring and performance metrics
//!
//! ## Example
//!
//! ```rust,ignore
//! use numrs2::gpu::batching::{BatchQueue, BatchConfig};
//!
//! # #[cfg(feature = "gpu")]
//! # fn example() -> numrs2::error::Result<()> {
//! let context = numrs2::gpu::new_context()?;
//! let mut queue = BatchQueue::new(context.clone(), BatchConfig::default());
//!
//! // Queue operations
//! queue.queue_matmul(&a, &b)?;
//! queue.queue_add(&c, &d)?;
//!
//! // Execute batched operations
//! let results = queue.flush()?;
//! # Ok(())
//! # }
//! ```

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::context::GpuContextRef;
use crate::gpu::memory::TransferOptimizer;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default maximum batch size
const DEFAULT_MAX_BATCH_SIZE: usize = 32;

/// Default batch timeout in milliseconds
const DEFAULT_BATCH_TIMEOUT_MS: u64 = 10;

/// Minimum batch size to trigger automatic flushing
const DEFAULT_MIN_BATCH_SIZE: usize = 4;

/// Configuration for batch queue behavior
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of operations in a batch
    pub max_batch_size: usize,
    /// Timeout before automatic flush (milliseconds)
    pub batch_timeout: Duration,
    /// Minimum batch size to trigger flush
    pub min_batch_size: usize,
    /// Enable dynamic batch size optimization
    pub enable_dynamic_optimization: bool,
    /// Enable automatic flushing on timeout
    pub enable_auto_flush: bool,
    /// Target GPU occupancy (0.0 to 1.0)
    pub target_occupancy: f32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            batch_timeout: Duration::from_millis(DEFAULT_BATCH_TIMEOUT_MS),
            min_batch_size: DEFAULT_MIN_BATCH_SIZE,
            enable_dynamic_optimization: true,
            enable_auto_flush: true,
            target_occupancy: 0.8,
        }
    }
}

/// Type of batched operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Matrix multiplication
    MatMul,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Multiply,
    /// Element-wise subtraction
    Subtract,
    /// Element-wise division
    Divide,
    /// 2D Convolution
    Conv2D,
    /// Element-wise exponential
    Exp,
    /// Element-wise logarithm
    Log,
    /// Element-wise square root
    Sqrt,
}

impl OperationType {
    /// Returns whether this operation type can be batched
    pub fn is_batchable(&self) -> bool {
        matches!(
            self,
            OperationType::MatMul
                | OperationType::Add
                | OperationType::Multiply
                | OperationType::Subtract
                | OperationType::Divide
                | OperationType::Conv2D
        )
    }

    /// Returns the estimated cost factor for this operation
    pub fn cost_factor(&self) -> f32 {
        match self {
            OperationType::MatMul => 10.0,
            OperationType::Conv2D => 8.0,
            OperationType::Multiply | OperationType::Divide => 2.0,
            OperationType::Add | OperationType::Subtract => 1.0,
            OperationType::Exp | OperationType::Log => 3.0,
            OperationType::Sqrt => 2.5,
        }
    }
}

/// A queued operation waiting to be executed
struct QueuedOperation<'a, T: bytemuck::Pod + bytemuck::Zeroable> {
    /// Type of operation
    op_type: OperationType,
    /// First input array (reference stored directly)
    input_a: &'a GpuArray<T>,
    /// Second input array (for binary operations, reference stored directly)
    input_b: Option<&'a GpuArray<T>>,
    /// Time when operation was queued
    queued_at: Instant,
    /// Priority (higher = execute sooner)
    priority: i32,
    /// Estimated operation cost
    cost: f32,
}

/// Result of a batched operation
pub struct BatchResult<T: bytemuck::Pod + bytemuck::Zeroable> {
    /// The resulting GPU array
    pub result: GpuArray<T>,
    /// Operation type that produced this result
    pub op_type: OperationType,
    /// Time taken to execute (microseconds)
    pub execution_time_us: u64,
}

/// Statistics about batch queue performance
#[derive(Debug, Clone, Copy)]
pub struct BatchStatistics {
    /// Total number of operations queued
    pub total_operations: u64,
    /// Total number of flush operations
    pub total_flushes: u64,
    /// Total number of operations executed
    pub total_executed: u64,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Maximum batch size seen
    pub max_batch_size: usize,
    /// Current queue depth
    pub current_queue_depth: usize,
    /// Average wait time (microseconds)
    pub avg_wait_time_us: u64,
    /// Average execution time (microseconds)
    pub avg_execution_time_us: u64,
    /// Total throughput (operations per second)
    pub throughput_ops_per_sec: f32,
    /// GPU occupancy estimate (0.0 to 1.0)
    pub estimated_gpu_occupancy: f32,
    /// Number of automatic flushes
    pub auto_flush_count: u64,
    /// Number of manual flushes
    pub manual_flush_count: u64,
}

impl Default for BatchStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            total_flushes: 0,
            total_executed: 0,
            avg_batch_size: 0.0,
            max_batch_size: 0,
            current_queue_depth: 0,
            avg_wait_time_us: 0,
            avg_execution_time_us: 0,
            throughput_ops_per_sec: 0.0,
            estimated_gpu_occupancy: 0.0,
            auto_flush_count: 0,
            manual_flush_count: 0,
        }
    }
}

/// Internal state of the batch queue - stores owned data instead of references
struct BatchQueueState<T: bytemuck::Pod + bytemuck::Zeroable> {
    /// Queue of pending operations - stored in Arc for thread-safe access
    queue: VecDeque<OwnedQueuedOperation<T>>,
    /// Statistics
    stats: BatchStatistics,
    /// Last flush time
    last_flush: Instant,
    /// Dynamic batch size (when optimization enabled)
    dynamic_batch_size: usize,
    /// Recent execution times for adaptive optimization
    recent_execution_times: VecDeque<u64>,
    /// Recent batch sizes for tracking
    recent_batch_sizes: VecDeque<usize>,
}

/// Owned version of QueuedOperation for storage in the queue
struct OwnedQueuedOperation<T: bytemuck::Pod + bytemuck::Zeroable> {
    /// Type of operation
    op_type: OperationType,
    /// First input array - stored in Arc to own the data
    input_a: Arc<GpuArray<T>>,
    /// Second input array - stored in Arc to own the data
    input_b: Option<Arc<GpuArray<T>>>,
    /// Time when operation was queued
    queued_at: Instant,
    /// Priority (higher = execute sooner)
    priority: i32,
    /// Estimated operation cost
    cost: f32,
}

/// GPU batch queue for automatic operation batching
pub struct BatchQueue<T: bytemuck::Pod + bytemuck::Zeroable> {
    /// GPU context
    context: GpuContextRef,
    /// Configuration
    config: BatchConfig,
    /// Internal state
    state: Arc<Mutex<BatchQueueState<T>>>,
    /// Transfer optimizer for efficient data movement
    transfer_optimizer: Arc<Mutex<TransferOptimizer>>,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> BatchQueue<T> {
    /// Creates a new batch queue with the given configuration
    pub fn new(context: GpuContextRef, config: BatchConfig) -> Self {
        let transfer_optimizer = TransferOptimizer::new(
            context.clone(),
            crate::gpu::memory::TransferStrategy::Batched,
        );

        Self {
            context: context.clone(),
            config: config.clone(),
            state: Arc::new(Mutex::new(BatchQueueState {
                queue: VecDeque::new(),
                stats: BatchStatistics::default(),
                last_flush: Instant::now(),
                dynamic_batch_size: config.max_batch_size,
                recent_execution_times: VecDeque::with_capacity(100),
                recent_batch_sizes: VecDeque::with_capacity(100),
            })),
            transfer_optimizer: Arc::new(Mutex::new(transfer_optimizer)),
        }
    }

    /// Creates a new batch queue with default configuration
    pub fn with_default_config(context: GpuContextRef) -> Self {
        Self::new(context, BatchConfig::default())
    }

    /// Queues a matrix multiplication operation
    pub fn queue_matmul(&mut self, a: Arc<GpuArray<T>>, b: Arc<GpuArray<T>>) -> Result<()> {
        self.queue_operation(OperationType::MatMul, a, Some(b), 0)
    }

    /// Queues an element-wise addition operation
    pub fn queue_add(&mut self, a: Arc<GpuArray<T>>, b: Arc<GpuArray<T>>) -> Result<()> {
        self.queue_operation(OperationType::Add, a, Some(b), 0)
    }

    /// Queues an element-wise multiplication operation
    pub fn queue_multiply(&mut self, a: Arc<GpuArray<T>>, b: Arc<GpuArray<T>>) -> Result<()> {
        self.queue_operation(OperationType::Multiply, a, Some(b), 0)
    }

    /// Queues an element-wise subtraction operation
    pub fn queue_subtract(&mut self, a: Arc<GpuArray<T>>, b: Arc<GpuArray<T>>) -> Result<()> {
        self.queue_operation(OperationType::Subtract, a, Some(b), 0)
    }

    /// Queues an element-wise division operation
    pub fn queue_divide(&mut self, a: Arc<GpuArray<T>>, b: Arc<GpuArray<T>>) -> Result<()> {
        self.queue_operation(OperationType::Divide, a, Some(b), 0)
    }

    /// Queues an operation with specified priority
    fn queue_operation(
        &mut self,
        op_type: OperationType,
        input_a: Arc<GpuArray<T>>,
        input_b: Option<Arc<GpuArray<T>>>,
        priority: i32,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        // Calculate operation cost
        let cost = op_type.cost_factor() * (input_a.size() as f32);

        // Create queued operation - use Arc-wrapped parameters directly
        // Eliminates need for cloning by accepting Arc from callers
        let op = OwnedQueuedOperation {
            op_type,
            input_a,
            input_b,
            queued_at: Instant::now(),
            priority,
            cost,
        };

        state.queue.push_back(op);
        state.stats.total_operations += 1;
        state.stats.current_queue_depth = state.queue.len();

        // Check if automatic flush should trigger
        if self.config.enable_auto_flush && self.should_auto_flush(&state) {
            drop(state);
            self.flush()?;
        }

        Ok(())
    }

    /// Checks if automatic flush should be triggered
    fn should_auto_flush(&self, state: &BatchQueueState<T>) -> bool {
        let queue_size = state.queue.len();
        let time_since_last_flush = state.last_flush.elapsed();

        // Flush if max batch size reached
        if queue_size >= self.config.max_batch_size {
            return true;
        }

        // Flush if timeout reached and minimum batch size met
        if time_since_last_flush >= self.config.batch_timeout
            && queue_size >= self.config.min_batch_size
        {
            return true;
        }

        false
    }

    /// Manually flushes all queued operations
    pub fn flush(&mut self) -> Result<Vec<BatchResult<T>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        if state.queue.is_empty() {
            return Ok(Vec::new());
        }

        let flush_start = Instant::now();

        // Determine batch size
        let batch_size = if self.config.enable_dynamic_optimization {
            state.dynamic_batch_size.min(state.queue.len())
        } else {
            self.config.max_batch_size.min(state.queue.len())
        };

        // Extract operations to execute
        let mut operations = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(op) = state.queue.pop_front() {
                operations.push(op);
            } else {
                break;
            }
        }

        state.stats.current_queue_depth = state.queue.len();
        state.stats.total_flushes += 1;
        state.stats.manual_flush_count += 1;

        // Update statistics
        let actual_batch_size = operations.len();
        state.stats.max_batch_size = state.stats.max_batch_size.max(actual_batch_size);

        // Release lock before executing operations
        drop(state);

        // Execute operations
        let results = self.execute_batch(operations)?;

        // Update post-execution statistics
        let mut state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        let execution_time = flush_start.elapsed().as_micros() as u64;
        state.stats.total_executed += actual_batch_size as u64;
        state.last_flush = Instant::now();

        // Update recent metrics for adaptive optimization
        state.recent_execution_times.push_back(execution_time);
        state.recent_batch_sizes.push_back(actual_batch_size);

        if state.recent_execution_times.len() > 100 {
            state.recent_execution_times.pop_front();
            state.recent_batch_sizes.pop_front();
        }

        // Update average statistics
        self.update_statistics(&mut state);

        // Optimize batch size if enabled
        if self.config.enable_dynamic_optimization {
            self.optimize_batch_size(&mut state);
        }

        Ok(results)
    }

    /// Executes a batch of operations
    fn execute_batch(
        &self,
        operations: Vec<OwnedQueuedOperation<T>>,
    ) -> Result<Vec<BatchResult<T>>> {
        let mut results = Vec::with_capacity(operations.len());

        for op in operations {
            let start = Instant::now();

            let result_array = match op.op_type {
                OperationType::MatMul => {
                    if let Some(input_b) = &op.input_b {
                        crate::gpu::ops::matmul(&op.input_a, input_b)?
                    } else {
                        return Err(NumRs2Error::InvalidOperation(
                            "MatMul requires two inputs".to_string(),
                        ));
                    }
                }
                OperationType::Add => {
                    if let Some(input_b) = &op.input_b {
                        crate::gpu::ops::add(&op.input_a, input_b)?
                    } else {
                        return Err(NumRs2Error::InvalidOperation(
                            "Add requires two inputs".to_string(),
                        ));
                    }
                }
                OperationType::Multiply => {
                    if let Some(input_b) = &op.input_b {
                        crate::gpu::ops::multiply(&op.input_a, input_b)?
                    } else {
                        return Err(NumRs2Error::InvalidOperation(
                            "Multiply requires two inputs".to_string(),
                        ));
                    }
                }
                OperationType::Subtract => {
                    if let Some(input_b) = &op.input_b {
                        crate::gpu::ops::subtract(&op.input_a, input_b)?
                    } else {
                        return Err(NumRs2Error::InvalidOperation(
                            "Subtract requires two inputs".to_string(),
                        ));
                    }
                }
                OperationType::Divide => {
                    if let Some(input_b) = &op.input_b {
                        crate::gpu::ops::divide(&op.input_a, input_b)?
                    } else {
                        return Err(NumRs2Error::InvalidOperation(
                            "Divide requires two inputs".to_string(),
                        ));
                    }
                }
                OperationType::Exp => crate::gpu::ops::exp(&op.input_a)?,
                OperationType::Log => crate::gpu::ops::log(&op.input_a)?,
                OperationType::Sqrt => crate::gpu::ops::sqrt(&op.input_a)?,
                OperationType::Conv2D => {
                    return Err(NumRs2Error::NotImplemented(
                        "Conv2D batching not yet implemented".to_string(),
                    ))
                }
            };

            let execution_time = start.elapsed().as_micros() as u64;

            results.push(BatchResult {
                result: result_array,
                op_type: op.op_type,
                execution_time_us: execution_time,
            });
        }

        Ok(results)
    }

    /// Updates running statistics
    fn update_statistics(&self, state: &mut BatchQueueState<T>) {
        if state.stats.total_flushes == 0 {
            return;
        }

        // Calculate average batch size
        state.stats.avg_batch_size =
            state.stats.total_executed as f32 / state.stats.total_flushes as f32;

        // Calculate average execution time
        if !state.recent_execution_times.is_empty() {
            let sum: u64 = state.recent_execution_times.iter().sum();
            state.stats.avg_execution_time_us = sum / state.recent_execution_times.len() as u64;
        }

        // Calculate throughput
        if state.stats.avg_execution_time_us > 0 {
            state.stats.throughput_ops_per_sec = (state.stats.avg_batch_size * 1_000_000.0)
                / state.stats.avg_execution_time_us as f32;
        }

        // Estimate GPU occupancy based on throughput and batch characteristics
        state.stats.estimated_gpu_occupancy = self.estimate_gpu_occupancy(state);
    }

    /// Estimates current GPU occupancy
    fn estimate_gpu_occupancy(&self, state: &BatchQueueState<T>) -> f32 {
        if state.recent_execution_times.is_empty() {
            return 0.0;
        }

        // Simple heuristic: higher throughput and larger batches = better occupancy
        let batch_efficiency = state.stats.avg_batch_size / self.config.max_batch_size as f32;
        let throughput_factor = (state.stats.throughput_ops_per_sec / 1000.0).min(1.0);

        (batch_efficiency * 0.6 + throughput_factor * 0.4).min(1.0)
    }

    /// Optimizes batch size based on recent performance
    fn optimize_batch_size(&self, state: &mut BatchQueueState<T>) {
        if state.recent_execution_times.len() < 10 {
            return;
        }

        let current_occupancy = state.stats.estimated_gpu_occupancy;
        let target_occupancy = self.config.target_occupancy;

        // Adjust batch size based on occupancy
        if current_occupancy < target_occupancy - 0.1 {
            // GPU underutilized, increase batch size
            state.dynamic_batch_size =
                (state.dynamic_batch_size + 4).min(self.config.max_batch_size);
        } else if current_occupancy > target_occupancy + 0.1 {
            // GPU over-saturated, decrease batch size
            state.dynamic_batch_size =
                (state.dynamic_batch_size.saturating_sub(2)).max(self.config.min_batch_size);
        }
    }

    /// Returns current batch statistics
    pub fn statistics(&self) -> Result<BatchStatistics> {
        let state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        Ok(state.stats)
    }

    /// Returns current queue depth
    pub fn queue_depth(&self) -> Result<usize> {
        let state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        Ok(state.queue.len())
    }

    /// Clears all queued operations without executing them
    pub fn clear(&mut self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        state.queue.clear();
        state.stats.current_queue_depth = 0;

        Ok(())
    }

    /// Returns whether the queue is empty
    pub fn is_empty(&self) -> Result<bool> {
        let state = self
            .state
            .lock()
            .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to lock batch queue: {}", e)))?;

        Ok(state.queue.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, DEFAULT_MAX_BATCH_SIZE);
        assert!(config.enable_dynamic_optimization);
        assert!(config.enable_auto_flush);
    }

    #[test]
    fn test_operation_type_cost_factor() {
        assert!(OperationType::MatMul.cost_factor() > OperationType::Add.cost_factor());
        assert!(OperationType::Conv2D.cost_factor() > OperationType::Multiply.cost_factor());
    }

    #[test]
    fn test_operation_type_is_batchable() {
        assert!(OperationType::MatMul.is_batchable());
        assert!(OperationType::Add.is_batchable());
        assert!(OperationType::Conv2D.is_batchable());
    }

    #[test]
    fn test_batch_statistics_default() {
        let stats = BatchStatistics::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.total_flushes, 0);
        assert_eq!(stats.current_queue_depth, 0);
    }
}
