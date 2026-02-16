//! GPU Memory Management
//!
//! This module provides GPU memory pooling, buffer reuse, and optimized memory transfers.
//! It includes garbage collection for unused buffers and strategies to minimize CPU-GPU
//! data transfer overhead.
//!
//! ## Features
//!
//! - **Memory Pooling**: Reuse GPU buffers to reduce allocation overhead
//! - **Garbage Collection**: Automatic cleanup of unused buffers
//! - **Transfer Optimization**: Batch transfers and asynchronous operations
//! - **Memory Tracking**: Monitor GPU memory usage and availability
//!
//! ## Example
//!
//! ```rust,ignore
//! use numrs2::gpu::memory::{GpuMemoryPool, TransferStrategy};
//!
//! # #[cfg(feature = "gpu")]
//! # fn example() -> numrs2::error::Result<()> {
//! let context = numrs2::gpu::new_context()?;
//! let mut pool = GpuMemoryPool::new(context.clone());
//!
//! // Allocate a buffer from the pool
//! let buffer = pool.allocate(1024, wgpu::BufferUsages::STORAGE)?;
//!
//! // Buffer is automatically returned to pool when dropped
//! drop(buffer);
//!
//! // Run garbage collection to free unused buffers
//! pool.collect_garbage(0.8); // Keep 80% of recent buffers
//! # Ok(())
//! # }
//! ```

use crate::error::{MemoryError, NumRs2Error, OperationContext, Result};
use crate::gpu::context::GpuContextRef;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default size threshold for large transfers (16MB)
const LARGE_TRANSFER_THRESHOLD: u64 = 16 * 1024 * 1024;

/// Default maximum pool size per buffer size (100 buffers)
const DEFAULT_MAX_POOL_SIZE: usize = 100;

/// Default buffer expiration time (5 minutes)
const DEFAULT_BUFFER_EXPIRATION: Duration = Duration::from_secs(300);

/// GPU memory pool for buffer reuse
///
/// The memory pool maintains separate pools for different buffer sizes and usage types,
/// allowing efficient reuse without fragmentation. Buffers are automatically returned
/// to the pool when dropped.
///
/// CACHE ALIGNMENT: Aligned to 64-byte cache lines to prevent false sharing when
/// multiple threads access the pool concurrently. The `Arc<Mutex<BufferPools>>` is a
/// hot synchronization point, and proper alignment ensures the mutex and its data
/// occupy separate cache lines from other thread-local data, reducing cache coherency
/// traffic and improving parallel GPU memory allocation performance.
#[repr(align(64))]
pub struct GpuMemoryPool {
    /// Reference to the GPU context
    context: GpuContextRef,
    /// Pooled buffers organized by size and usage
    pools: Arc<Mutex<BufferPools>>,
    /// Configuration for the memory pool
    config: PoolConfig,
}

/// Configuration for the memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers to keep per size/usage combination
    pub max_pool_size: usize,
    /// Time after which unused buffers expire
    pub buffer_expiration: Duration,
    /// Whether to enable automatic garbage collection
    pub auto_gc: bool,
    /// Minimum retention rate during garbage collection (0.0 to 1.0)
    pub gc_retention_rate: f32,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: DEFAULT_MAX_POOL_SIZE,
            buffer_expiration: DEFAULT_BUFFER_EXPIRATION,
            auto_gc: true,
            gc_retention_rate: 0.8,
        }
    }
}

/// Storage for pooled buffers
struct BufferPools {
    /// Map from (size, usage_bits) to queue of available buffers
    pools: HashMap<(u64, u32), VecDeque<PooledBuffer>>,
    /// Total number of buffers in all pools
    total_buffers: usize,
    /// Total bytes allocated across all pools
    total_bytes: u64,
}

/// A buffer stored in the pool with metadata
struct PooledBuffer {
    /// The actual GPU buffer
    buffer: wgpu::Buffer,
    /// Size of the buffer in bytes
    size: u64,
    /// When this buffer was last used
    last_used: Instant,
}

impl GpuMemoryPool {
    /// Creates a new GPU memory pool with default configuration
    pub fn new(context: GpuContextRef) -> Self {
        Self::with_config(context, PoolConfig::default())
    }

    /// Creates a new GPU memory pool with custom configuration
    pub fn with_config(context: GpuContextRef, config: PoolConfig) -> Self {
        Self {
            context,
            pools: Arc::new(Mutex::new(BufferPools {
                pools: HashMap::new(),
                total_buffers: 0,
                total_bytes: 0,
            })),
            config,
        }
    }

    /// Allocates a buffer from the pool or creates a new one
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the buffer in bytes
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    ///
    /// A managed buffer that will be returned to the pool when dropped
    pub fn allocate(&mut self, size: u64, usage: wgpu::BufferUsages) -> Result<ManagedBuffer> {
        let usage_bits = usage.bits();
        let key = (size, usage_bits);

        // Try to get a buffer from the pool
        let buffer = {
            let mut pools = self.pools.lock().map_err(|e| {
                NumRs2Error::from(MemoryError::gpu_memory_error(
                    &format!("Failed to lock buffer pool: {}", e),
                    None,
                ))
            })?;

            if let Some(pool) = pools.pools.get_mut(&key) {
                if let Some(mut pooled) = pool.pop_front() {
                    pooled.last_used = Instant::now();
                    pools.total_buffers = pools.total_buffers.saturating_sub(1);
                    pools.total_bytes = pools.total_bytes.saturating_sub(size);
                    Some(pooled.buffer)
                } else {
                    None
                }
            } else {
                None
            }
        };

        // If no buffer available in pool, create a new one
        let buffer = if let Some(buf) = buffer {
            buf
        } else {
            self.context.create_empty_buffer(size, usage)
        };

        Ok(ManagedBuffer {
            buffer: Some(buffer),
            size,
            usage_bits,
            pool: Arc::clone(&self.pools),
            returned: false,
        })
    }

    /// Runs garbage collection to free unused buffers
    ///
    /// # Arguments
    ///
    /// * `retention_rate` - Fraction of recent buffers to keep (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// The number of buffers freed and total bytes freed
    pub fn collect_garbage(&mut self, retention_rate: f32) -> Result<(usize, u64)> {
        let retention_rate = retention_rate.clamp(0.0, 1.0);
        let cutoff_time = Instant::now() - self.config.buffer_expiration;

        let mut pools = self.pools.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock buffer pool during GC: {}", e),
                None,
            ))
        })?;

        let mut freed_buffers = 0;
        let mut freed_bytes = 0u64;

        // Process each pool
        for pool in pools.pools.values_mut() {
            let original_len = pool.len();

            // Calculate how many buffers to keep
            let keep_count = ((original_len as f32) * retention_rate).ceil() as usize;

            // Collect indices to remove
            let mut kept = 0;
            pool.retain(|pooled| {
                let should_keep = pooled.last_used > cutoff_time && kept < keep_count;
                if should_keep {
                    kept += 1;
                    true
                } else {
                    freed_buffers += 1;
                    freed_bytes += pooled.size;
                    false
                }
            });
        }

        // Update totals
        pools.total_buffers = pools.total_buffers.saturating_sub(freed_buffers);
        pools.total_bytes = pools.total_bytes.saturating_sub(freed_bytes);

        // Remove empty pools
        pools.pools.retain(|_, pool| !pool.is_empty());

        Ok((freed_buffers, freed_bytes))
    }

    /// Returns the current pool statistics
    pub fn statistics(&self) -> Result<PoolStatistics> {
        let pools = self.pools.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock buffer pool: {}", e),
                None,
            ))
        })?;

        Ok(PoolStatistics {
            total_buffers: pools.total_buffers,
            total_bytes: pools.total_bytes,
            pool_count: pools.pools.len(),
        })
    }

    /// Clears all buffers from the pool
    pub fn clear(&mut self) -> Result<()> {
        let mut pools = self.pools.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock buffer pool during clear: {}", e),
                None,
            ))
        })?;

        pools.pools.clear();
        pools.total_buffers = 0;
        pools.total_bytes = 0;

        Ok(())
    }
}

/// Statistics about the memory pool
#[derive(Debug, Clone, Copy)]
pub struct PoolStatistics {
    /// Total number of buffers in the pool
    pub total_buffers: usize,
    /// Total bytes allocated in the pool
    pub total_bytes: u64,
    /// Number of different pool types
    pub pool_count: usize,
}

/// A managed GPU buffer that returns to the pool when dropped
pub struct ManagedBuffer {
    buffer: Option<wgpu::Buffer>,
    size: u64,
    usage_bits: u32,
    pool: Arc<Mutex<BufferPools>>,
    returned: bool,
}

impl ManagedBuffer {
    /// Gets a reference to the underlying buffer
    ///
    /// # Panics
    ///
    /// Panics if the buffer has already been returned to the pool
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer
            .as_ref()
            .expect("Buffer has been returned to pool")
    }

    /// Gets the size of the buffer in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Manually returns the buffer to the pool
    pub fn return_to_pool(&mut self) {
        if !self.returned {
            self.returned = true;

            // Take the buffer out of the Option
            if let Some(buffer) = self.buffer.take() {
                // Try to return buffer to pool (ignore errors on drop)
                if let Ok(mut pools) = self.pool.lock() {
                    let key = (self.size, self.usage_bits);
                    let pool = pools.pools.entry(key).or_insert_with(VecDeque::new);

                    pool.push_back(PooledBuffer {
                        buffer,
                        size: self.size,
                        last_used: Instant::now(),
                    });

                    pools.total_buffers += 1;
                    pools.total_bytes += self.size;
                }
            }
        }
    }
}

impl Drop for ManagedBuffer {
    fn drop(&mut self) {
        self.return_to_pool();
    }
}

/// Strategy for transferring data between CPU and GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Transfer immediately (synchronous)
    Immediate,
    /// Batch multiple transfers together
    Batched,
    /// Use asynchronous transfer
    Async,
}

/// Memory transfer optimizer for CPU-GPU data movement
///
/// CACHE ALIGNMENT: Aligned to 64-byte cache lines for optimal transfer performance.
/// The batch_queue and async_queue are frequently accessed during GPU operations,
/// and cache alignment ensures efficient batching and reduces memory latency when
/// coordinating multiple concurrent transfers.
#[repr(align(64))]
pub struct TransferOptimizer {
    context: GpuContextRef,
    strategy: TransferStrategy,
    batch_queue: Vec<PendingTransfer>,
    async_queue: Arc<Mutex<Vec<AsyncTransfer>>>,
}

struct PendingTransfer {
    source: Vec<u8>,
    destination: wgpu::Buffer,
    offset: u64,
}

/// Asynchronous transfer operation
struct AsyncTransfer {
    id: u64,
    size: u64,
    submitted: std::time::Instant,
    completed: bool,
}

impl TransferOptimizer {
    /// Creates a new transfer optimizer
    pub fn new(context: GpuContextRef, strategy: TransferStrategy) -> Self {
        Self {
            context,
            strategy,
            batch_queue: Vec::new(),
            async_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Determines if a transfer is considered large
    pub fn is_large_transfer(size: u64) -> bool {
        size >= LARGE_TRANSFER_THRESHOLD
    }

    /// Queues a data transfer to GPU
    pub fn queue_transfer<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
        buffer: &wgpu::Buffer,
    ) -> Result<()> {
        self.queue_transfer_with_offset(data, buffer, 0)
    }

    /// Queues a data transfer to GPU with offset
    pub fn queue_transfer_with_offset<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
        buffer: &wgpu::Buffer,
        offset: u64,
    ) -> Result<()> {
        let byte_data = bytemuck::cast_slice(data);

        match self.strategy {
            TransferStrategy::Immediate => {
                self.context.queue().write_buffer(buffer, offset, byte_data);
                Ok(())
            }
            TransferStrategy::Batched => {
                // For batched transfers, store the transfer for later
                self.batch_queue.push(PendingTransfer {
                    source: byte_data.to_vec(),
                    destination: buffer.clone(),
                    offset,
                });
                Ok(())
            }
            TransferStrategy::Async => {
                // For async, queue the transfer and track it
                let transfer_id = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|e| {
                        NumRs2Error::from(MemoryError::gpu_memory_error(
                            &format!("Failed to generate transfer ID: {}", e),
                            None,
                        ))
                    })?
                    .as_nanos() as u64;

                self.context.queue().write_buffer(buffer, offset, byte_data);

                let mut async_queue = self.async_queue.lock().map_err(|e| {
                    NumRs2Error::from(MemoryError::gpu_memory_error(
                        &format!("Failed to lock async queue: {}", e),
                        None,
                    ))
                })?;

                async_queue.push(AsyncTransfer {
                    id: transfer_id,
                    size: byte_data.len() as u64,
                    submitted: Instant::now(),
                    completed: false,
                });

                Ok(())
            }
        }
    }

    /// Flushes all pending batched transfers
    pub fn flush(&mut self) -> Result<()> {
        if !self.batch_queue.is_empty() {
            // Process all batched transfers
            for transfer in self.batch_queue.drain(..) {
                self.context.queue().write_buffer(
                    &transfer.destination,
                    transfer.offset,
                    &transfer.source,
                );
            }
        }
        Ok(())
    }

    /// Gets the number of pending async transfers
    pub fn pending_transfers(&self) -> Result<usize> {
        let async_queue = self.async_queue.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock async queue: {}", e),
                None,
            ))
        })?;

        Ok(async_queue.iter().filter(|t| !t.completed).count())
    }

    /// Waits for all async transfers to complete
    pub fn wait_for_completion(&mut self) -> Result<()> {
        // Poll the device to ensure all transfers are complete
        self.context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                NumRs2Error::from(MemoryError::gpu_memory_error(
                    &format!("Failed to poll device: {:?}", e),
                    None,
                ))
            })?;

        let mut async_queue = self.async_queue.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock async queue: {}", e),
                None,
            ))
        })?;

        // Mark all transfers as completed
        for transfer in async_queue.iter_mut() {
            transfer.completed = true;
        }

        Ok(())
    }

    /// Clears completed async transfers from the queue
    pub fn clear_completed(&mut self) -> Result<usize> {
        let mut async_queue = self.async_queue.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock async queue: {}", e),
                None,
            ))
        })?;

        let before_count = async_queue.len();
        async_queue.retain(|t| !t.completed);

        Ok(before_count - async_queue.len())
    }

    /// Gets the current transfer strategy
    pub fn strategy(&self) -> TransferStrategy {
        self.strategy
    }

    /// Sets a new transfer strategy
    pub fn set_strategy(&mut self, strategy: TransferStrategy) {
        self.strategy = strategy;
    }
}

/// Double buffer for streaming GPU operations
///
/// Maintains two buffers and alternates between them to allow overlapping
/// compute and data transfer operations.
pub struct DoubleBuffer {
    context: GpuContextRef,
    buffers: [wgpu::Buffer; 2],
    current_index: usize,
    size: u64,
    usage: wgpu::BufferUsages,
}

impl DoubleBuffer {
    /// Creates a new double buffer
    pub fn new(context: GpuContextRef, size: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer_a = context.create_empty_buffer(size, usage);
        let buffer_b = context.create_empty_buffer(size, usage);

        Self {
            context,
            buffers: [buffer_a, buffer_b],
            current_index: 0,
            size,
            usage,
        }
    }

    /// Gets the current buffer
    pub fn current(&self) -> &wgpu::Buffer {
        &self.buffers[self.current_index]
    }

    /// Gets the next buffer (for prefetching)
    pub fn next(&self) -> &wgpu::Buffer {
        &self.buffers[1 - self.current_index]
    }

    /// Swaps the buffers
    pub fn swap(&mut self) {
        self.current_index = 1 - self.current_index;
    }

    /// Gets the buffer size
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Writes data to the current buffer
    pub fn write_current<T: bytemuck::Pod>(&self, data: &[T]) {
        self.context
            .queue()
            .write_buffer(self.current(), 0, bytemuck::cast_slice(data));
    }

    /// Writes data to the next buffer (for prefetching)
    pub fn write_next<T: bytemuck::Pod>(&self, data: &[T]) {
        self.context
            .queue()
            .write_buffer(self.next(), 0, bytemuck::cast_slice(data));
    }
}

/// Buffer aliasing manager for memory-efficient operations
///
/// Allows multiple GPU arrays to share the same underlying buffer when appropriate,
/// reducing memory usage and improving cache locality.
pub struct BufferAliasManager {
    context: GpuContextRef,
    aliases: Arc<Mutex<HashMap<u64, Vec<BufferAlias>>>>,
}

struct BufferAlias {
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
    usage: wgpu::BufferUsages,
    ref_count: usize,
}

impl BufferAliasManager {
    /// Creates a new buffer alias manager
    pub fn new(context: GpuContextRef) -> Self {
        Self {
            context,
            aliases: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Creates or reuses a buffer with aliasing
    pub fn get_or_create_buffer(
        &mut self,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer> {
        let key = size;

        let mut aliases = self.aliases.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock alias manager: {}", e),
                None,
            ))
        })?;

        // Try to find an existing alias
        if let Some(alias_list) = aliases.get_mut(&key) {
            for alias in alias_list.iter_mut() {
                if alias.usage == usage {
                    alias.ref_count += 1;
                    return Ok(alias.buffer.clone());
                }
            }
        }

        // Create new buffer
        let buffer = self.context.create_empty_buffer(size, usage);

        // Add to aliases
        let alias = BufferAlias {
            buffer: buffer.clone(),
            offset: 0,
            size,
            usage,
            ref_count: 1,
        };

        aliases.entry(key).or_insert_with(Vec::new).push(alias);

        Ok(buffer)
    }

    /// Releases a buffer alias
    pub fn release_buffer(&mut self, size: u64, usage: wgpu::BufferUsages) -> Result<()> {
        let key = size;

        let mut aliases = self.aliases.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock alias manager: {}", e),
                None,
            ))
        })?;

        if let Some(alias_list) = aliases.get_mut(&key) {
            for alias in alias_list.iter_mut() {
                if alias.usage == usage && alias.ref_count > 0 {
                    alias.ref_count -= 1;
                    break;
                }
            }

            // Remove aliases with zero ref count
            alias_list.retain(|a| a.ref_count > 0);

            if alias_list.is_empty() {
                aliases.remove(&key);
            }
        }

        Ok(())
    }

    /// Gets statistics about buffer aliasing
    pub fn statistics(&self) -> Result<AliasStatistics> {
        let aliases = self.aliases.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock alias manager: {}", e),
                None,
            ))
        })?;

        let total_aliases: usize = aliases.values().map(|v| v.len()).sum();
        let total_refs: usize = aliases
            .values()
            .flat_map(|v| v.iter().map(|a| a.ref_count))
            .sum();

        Ok(AliasStatistics {
            total_aliases,
            total_references: total_refs,
            buffer_sizes: aliases.len(),
        })
    }

    /// Clears all buffer aliases
    pub fn clear(&mut self) -> Result<()> {
        let mut aliases = self.aliases.lock().map_err(|e| {
            NumRs2Error::from(MemoryError::gpu_memory_error(
                &format!("Failed to lock alias manager: {}", e),
                None,
            ))
        })?;

        aliases.clear();
        Ok(())
    }
}

/// Statistics about buffer aliasing
#[derive(Debug, Clone, Copy)]
pub struct AliasStatistics {
    /// Total number of buffer aliases
    pub total_aliases: usize,
    /// Total number of references to aliased buffers
    pub total_references: usize,
    /// Number of different buffer sizes
    pub buffer_sizes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_pool_size, DEFAULT_MAX_POOL_SIZE);
        assert!(config.auto_gc);
        assert_eq!(config.gc_retention_rate, 0.8);
    }

    #[test]
    fn test_is_large_transfer() {
        assert!(!TransferOptimizer::is_large_transfer(1024));
        assert!(!TransferOptimizer::is_large_transfer(
            LARGE_TRANSFER_THRESHOLD - 1
        ));
        assert!(TransferOptimizer::is_large_transfer(
            LARGE_TRANSFER_THRESHOLD
        ));
        assert!(TransferOptimizer::is_large_transfer(
            LARGE_TRANSFER_THRESHOLD + 1
        ));
    }

    #[test]
    fn test_transfer_strategy() {
        assert_ne!(TransferStrategy::Immediate, TransferStrategy::Batched);
        assert_ne!(TransferStrategy::Immediate, TransferStrategy::Async);
        assert_ne!(TransferStrategy::Batched, TransferStrategy::Async);
    }
}
