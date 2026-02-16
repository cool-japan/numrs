//! Tests for GPU memory management

use numrs2::array::Array;
use numrs2::gpu::memory::{GpuMemoryPool, PoolConfig, TransferOptimizer, TransferStrategy};
use numrs2::gpu::new_context;
use std::time::Duration;

#[tokio::test]
async fn test_memory_pool_creation() {
    let context = new_context().expect("Failed to create GPU context");
    let pool = GpuMemoryPool::new(context);

    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 0);
    assert_eq!(stats.total_bytes, 0);
    assert_eq!(stats.pool_count, 0);
}

#[tokio::test]
async fn test_memory_pool_allocation() {
    let context = new_context().expect("Failed to create GPU context");
    let mut pool = GpuMemoryPool::new(context.clone());

    // Allocate a buffer
    let buffer = pool
        .allocate(1024, wgpu::BufferUsages::STORAGE)
        .expect("Failed to allocate buffer");

    assert_eq!(buffer.size(), 1024);

    // Buffer should not be in pool while in use
    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 0);
}

#[tokio::test]
async fn test_memory_pool_reuse() {
    let context = new_context().expect("Failed to create GPU context");
    let mut pool = GpuMemoryPool::new(context.clone());

    // Allocate and drop a buffer
    {
        let _buffer = pool
            .allocate(1024, wgpu::BufferUsages::STORAGE)
            .expect("Failed to allocate buffer");
    }

    // Buffer should be returned to pool
    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 1);
    assert_eq!(stats.total_bytes, 1024);

    // Allocate again - should reuse the buffer
    let buffer2 = pool
        .allocate(1024, wgpu::BufferUsages::STORAGE)
        .expect("Failed to allocate buffer");
    assert_eq!(buffer2.size(), 1024);

    // Pool should now be empty again
    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 0);
}

#[tokio::test]
async fn test_memory_pool_garbage_collection() {
    let context = new_context().expect("Failed to create GPU context");
    let config = PoolConfig {
        max_pool_size: 100,
        buffer_expiration: Duration::from_millis(100),
        auto_gc: true,
        gc_retention_rate: 0.5,
    };
    let mut pool = GpuMemoryPool::with_config(context.clone(), config);

    // Allocate and return several buffers
    for _ in 0..10 {
        let _buffer = pool
            .allocate(1024, wgpu::BufferUsages::STORAGE)
            .expect("Failed to allocate buffer");
    }

    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 10);

    // Run garbage collection with 50% retention
    let (freed_buffers, freed_bytes) = pool
        .collect_garbage(0.5)
        .expect("Failed to collect garbage");

    assert!(freed_buffers > 0);
    assert_eq!(freed_bytes, freed_buffers as u64 * 1024);

    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert!(stats.total_buffers < 10);
}

#[tokio::test]
async fn test_memory_pool_clear() {
    let context = new_context().expect("Failed to create GPU context");
    let mut pool = GpuMemoryPool::new(context.clone());

    // Allocate and return several buffers
    for _ in 0..5 {
        let _buffer = pool
            .allocate(2048, wgpu::BufferUsages::STORAGE)
            .expect("Failed to allocate buffer");
    }

    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 5);

    // Clear the pool
    pool.clear().expect("Failed to clear pool");

    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 0);
    assert_eq!(stats.total_bytes, 0);
}

#[tokio::test]
async fn test_transfer_optimizer_immediate() {
    let context = new_context().expect("Failed to create GPU context");
    let mut optimizer = TransferOptimizer::new(context.clone(), TransferStrategy::Immediate);

    assert_eq!(optimizer.strategy(), TransferStrategy::Immediate);

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buffer = context.create_empty_buffer(
        (data.len() * std::mem::size_of::<f32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    optimizer
        .queue_transfer(&data, &buffer)
        .expect("Failed to queue transfer");

    optimizer.flush().expect("Failed to flush transfers");
}

#[tokio::test]
async fn test_transfer_optimizer_batched() {
    let context = new_context().expect("Failed to create GPU context");
    let mut optimizer = TransferOptimizer::new(context.clone(), TransferStrategy::Batched);

    assert_eq!(optimizer.strategy(), TransferStrategy::Batched);

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buffer = context.create_empty_buffer(
        (data.len() * std::mem::size_of::<f32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    optimizer
        .queue_transfer(&data, &buffer)
        .expect("Failed to queue transfer");

    // Flush should process all queued transfers
    optimizer.flush().expect("Failed to flush transfers");
}

#[tokio::test]
async fn test_transfer_strategy_change() {
    let context = new_context().expect("Failed to create GPU context");
    let mut optimizer = TransferOptimizer::new(context.clone(), TransferStrategy::Immediate);

    assert_eq!(optimizer.strategy(), TransferStrategy::Immediate);

    optimizer.set_strategy(TransferStrategy::Batched);
    assert_eq!(optimizer.strategy(), TransferStrategy::Batched);

    optimizer.set_strategy(TransferStrategy::Async);
    assert_eq!(optimizer.strategy(), TransferStrategy::Async);
}

#[tokio::test]
async fn test_is_large_transfer() {
    assert!(!TransferOptimizer::is_large_transfer(1024));
    assert!(!TransferOptimizer::is_large_transfer(1024 * 1024));
    assert!(!TransferOptimizer::is_large_transfer(10 * 1024 * 1024));
    assert!(TransferOptimizer::is_large_transfer(16 * 1024 * 1024));
    assert!(TransferOptimizer::is_large_transfer(100 * 1024 * 1024));
}

#[tokio::test]
async fn test_pool_statistics() {
    let context = new_context().expect("Failed to create GPU context");
    let mut pool = GpuMemoryPool::new(context.clone());

    // Allocate buffers of different sizes
    {
        let _b1 = pool
            .allocate(1024, wgpu::BufferUsages::STORAGE)
            .expect("Failed to allocate");
        let _b2 = pool
            .allocate(2048, wgpu::BufferUsages::STORAGE)
            .expect("Failed to allocate");
        let _b3 = pool
            .allocate(1024, wgpu::BufferUsages::STORAGE)
            .expect("Failed to allocate");
    }

    let stats = pool.statistics().expect("Failed to get pool statistics");
    assert_eq!(stats.total_buffers, 3);
    assert_eq!(stats.total_bytes, 1024 + 2048 + 1024);
    // Should have 2 different pool types (1024 and 2048)
    assert_eq!(stats.pool_count, 2);
}

#[tokio::test]
async fn test_transfer_with_offset() {
    let context = new_context().expect("Failed to create GPU context");
    let mut optimizer = TransferOptimizer::new(context.clone(), TransferStrategy::Immediate);

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buffer_size = (data.len() * 2 * std::mem::size_of::<f32>()) as u64;
    let buffer = context.create_empty_buffer(
        buffer_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    // Write to first half
    optimizer
        .queue_transfer_with_offset(&data, &buffer, 0)
        .expect("Failed to queue transfer");

    // Write to second half
    optimizer
        .queue_transfer_with_offset(&data, &buffer, (data.len() * std::mem::size_of::<f32>()) as u64)
        .expect("Failed to queue transfer with offset");
}

#[tokio::test]
async fn test_async_transfer_queue() {
    let context = new_context().expect("Failed to create GPU context");
    let mut optimizer = TransferOptimizer::new(context.clone(), TransferStrategy::Async);

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buffer = context.create_empty_buffer(
        (data.len() * std::mem::size_of::<f32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    // Queue multiple async transfers
    for _ in 0..5 {
        optimizer
            .queue_transfer(&data, &buffer)
            .expect("Failed to queue async transfer");
    }

    // Should have pending transfers
    let pending = optimizer
        .pending_transfers()
        .expect("Failed to get pending count");
    assert!(pending > 0);

    // Wait for completion
    optimizer
        .wait_for_completion()
        .expect("Failed to wait for completion");

    // Clear completed transfers
    let cleared = optimizer
        .clear_completed()
        .expect("Failed to clear completed");
    assert_eq!(cleared, 5);
}

#[tokio::test]
async fn test_double_buffer_creation() {
    use numrs2::gpu::memory::DoubleBuffer;

    let context = new_context().expect("Failed to create GPU context");
    let double_buffer = DoubleBuffer::new(context, 1024, wgpu::BufferUsages::STORAGE);

    assert_eq!(double_buffer.size(), 1024);
}

#[tokio::test]
async fn test_double_buffer_swap() {
    use numrs2::gpu::memory::DoubleBuffer;

    let context = new_context().expect("Failed to create GPU context");
    let mut double_buffer = DoubleBuffer::new(context, 1024, wgpu::BufferUsages::STORAGE);

    let first_current = double_buffer.current() as *const _;
    let first_next = double_buffer.next() as *const _;

    // Swap buffers
    double_buffer.swap();

    let second_current = double_buffer.current() as *const _;
    let second_next = double_buffer.next() as *const _;

    // Current should now be what was next
    assert_eq!(first_next, second_current);
    assert_eq!(first_current, second_next);
}

#[tokio::test]
async fn test_double_buffer_write() {
    use numrs2::gpu::memory::DoubleBuffer;

    let context = new_context().expect("Failed to create GPU context");
    let double_buffer = DoubleBuffer::new(
        context,
        16,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    double_buffer.write_current(&data);
    double_buffer.write_next(&data);
}

#[tokio::test]
async fn test_buffer_alias_manager() {
    use numrs2::gpu::memory::BufferAliasManager;

    let context = new_context().expect("Failed to create GPU context");
    let mut manager = BufferAliasManager::new(context);

    // Allocate buffers
    let _buffer1 = manager
        .get_or_create_buffer(1024, wgpu::BufferUsages::STORAGE)
        .expect("Failed to create buffer");

    let stats = manager.statistics().expect("Failed to get statistics");
    assert_eq!(stats.total_aliases, 1);
    assert_eq!(stats.total_references, 1);
}

#[tokio::test]
async fn test_buffer_alias_reuse() {
    use numrs2::gpu::memory::BufferAliasManager;

    let context = new_context().expect("Failed to create GPU context");
    let mut manager = BufferAliasManager::new(context);

    // Create buffer
    let buffer1 = manager
        .get_or_create_buffer(1024, wgpu::BufferUsages::STORAGE)
        .expect("Failed to create buffer");

    // Request same size and usage - should get aliased buffer
    let buffer2 = manager
        .get_or_create_buffer(1024, wgpu::BufferUsages::STORAGE)
        .expect("Failed to get buffer");

    let stats = manager.statistics().expect("Failed to get statistics");
    assert_eq!(stats.total_aliases, 1);
    assert_eq!(stats.total_references, 2);

    // Buffers should be the same (pointer equality check would be ideal, but we can't easily do that)
}

#[tokio::test]
async fn test_buffer_alias_release() {
    use numrs2::gpu::memory::BufferAliasManager;

    let context = new_context().expect("Failed to create GPU context");
    let mut manager = BufferAliasManager::new(context);

    // Create and release buffer
    {
        let _buffer = manager
            .get_or_create_buffer(1024, wgpu::BufferUsages::STORAGE)
            .expect("Failed to create buffer");

        let stats = manager.statistics().expect("Failed to get statistics");
        assert_eq!(stats.total_references, 1);
    }

    // Release the buffer
    manager
        .release_buffer(1024, wgpu::BufferUsages::STORAGE)
        .expect("Failed to release buffer");

    let stats = manager.statistics().expect("Failed to get statistics");
    assert_eq!(stats.total_aliases, 0);
}

#[tokio::test]
async fn test_buffer_alias_clear() {
    use numrs2::gpu::memory::BufferAliasManager;

    let context = new_context().expect("Failed to create GPU context");
    let mut manager = BufferAliasManager::new(context);

    // Create multiple buffers
    for _ in 0..5 {
        let _buffer = manager
            .get_or_create_buffer(1024, wgpu::BufferUsages::STORAGE)
            .expect("Failed to create buffer");
    }

    let stats_before = manager.statistics().expect("Failed to get statistics");
    assert!(stats_before.total_references > 0);

    // Clear all aliases
    manager.clear().expect("Failed to clear aliases");

    let stats_after = manager.statistics().expect("Failed to get statistics");
    assert_eq!(stats_after.total_aliases, 0);
    assert_eq!(stats_after.total_references, 0);
}
