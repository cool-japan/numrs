//! Tests for GPU batching operations
//!
//! `NumRs2Error` is a large, crate-wide shared enum (see the sibling
//! `tests/test_gpu_reductions.rs`, which carries the identical allow for the
//! identical reason); shrinking it is out of scope here, so the tests below
//! that propagate it with `?` are allowed to return a large `Err` variant
//! rather than rewriting them to the `.expect()` style used elsewhere in
//! this directory.
#![allow(clippy::result_large_err)]

use numrs2::array::Array;
use numrs2::gpu::batching::{BatchConfig, BatchQueue, OperationType};
use numrs2::gpu::{new_context, GpuArray};
use std::sync::Arc;

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_creation() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig::default();
    let queue: BatchQueue<f32> = BatchQueue::new(context, config);

    assert!(queue.is_empty()?);
    assert_eq!(queue.queue_depth()?, 0);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_add_operations() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: false, // Disable auto-flush for testing
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue operations
    queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_multiply(a_gpu.clone(), b_gpu.clone())?;

    assert_eq!(queue.queue_depth()?, 2);
    assert!(!queue.is_empty()?);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_flush() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue operations
    queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_multiply(a_gpu.clone(), b_gpu.clone())?;

    // Flush and get results
    let results = queue.flush()?;

    assert_eq!(results.len(), 2);
    assert!(queue.is_empty()?);
    assert_eq!(queue.queue_depth()?, 0);

    // Verify results
    let add_result = &results[0];
    let mul_result = &results[1];

    assert_eq!(add_result.op_type, OperationType::Add);
    assert_eq!(mul_result.op_type, OperationType::Multiply);

    // Convert to CPU and check values
    let add_cpu = add_result.result.to_array()?;
    let mul_cpu = mul_result.result.to_array()?;

    let expected_add = vec![6.0f32, 8.0, 10.0, 12.0];
    let expected_mul = vec![5.0f32, 12.0, 21.0, 32.0];

    assert_eq!(add_cpu.to_vec(), expected_add);
    assert_eq!(mul_cpu.to_vec(), expected_mul);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_statistics() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Get initial statistics
    let stats_before = queue.statistics()?;
    assert_eq!(stats_before.total_operations, 0);
    assert_eq!(stats_before.total_flushes, 0);

    // Queue and flush operations
    queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_multiply(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_subtract(a_gpu.clone(), b_gpu.clone())?;

    let stats_after_queue = queue.statistics()?;
    assert_eq!(stats_after_queue.total_operations, 3);
    assert_eq!(stats_after_queue.current_queue_depth, 3);

    queue.flush()?;

    let stats_after_flush = queue.statistics()?;
    assert_eq!(stats_after_flush.total_flushes, 1);
    assert_eq!(stats_after_flush.total_executed, 3);
    assert_eq!(stats_after_flush.current_queue_depth, 0);
    assert!(stats_after_flush.avg_batch_size > 0.0);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_auto_flush() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: true,
        max_batch_size: 2, // Small batch size for testing
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue operations - should auto-flush after 2 operations
    queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_multiply(a_gpu.clone(), b_gpu.clone())?;

    // Check that auto-flush occurred
    let stats = queue.statistics()?;
    assert!(stats.total_flushes > 0);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_matmul() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test matrices
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue matmul operation
    queue.queue_matmul(a_gpu.clone(), b_gpu.clone())?;

    // Flush and get result
    let results = queue.flush()?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].op_type, OperationType::MatMul);

    // Convert to CPU and verify
    let result_cpu = results[0].result.to_array()?;
    let result_vec = result_cpu.to_vec();

    // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    //         = [[19, 22], [43, 50]]
    let expected = [19.0f32, 22.0, 43.0, 50.0];

    for (i, (&actual, &expected)) in result_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at index {}: {} != {}",
            i,
            actual,
            expected
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_clear() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue operations
    queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_multiply(a_gpu.clone(), b_gpu.clone())?;

    assert_eq!(queue.queue_depth()?, 2);

    // Clear queue
    queue.clear()?;

    assert_eq!(queue.queue_depth()?, 0);
    assert!(queue.is_empty()?);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_mixed_operations() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![4.0f32, 9.0, 16.0, 25.0]).reshape(&[4]);
    let b = Array::from_vec(vec![2.0f32, 3.0, 4.0, 5.0]).reshape(&[4]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue various operations
    queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_subtract(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_multiply(a_gpu.clone(), b_gpu.clone())?;
    queue.queue_divide(a_gpu.clone(), b_gpu.clone())?;

    // Flush and verify
    let results = queue.flush()?;

    assert_eq!(results.len(), 4);

    let add_result = results[0].result.to_array()?.to_vec();
    let sub_result = results[1].result.to_array()?.to_vec();
    let mul_result = results[2].result.to_array()?.to_vec();
    let div_result = results[3].result.to_array()?.to_vec();

    // Verify each result
    assert_eq!(add_result, vec![6.0f32, 12.0, 20.0, 30.0]);
    assert_eq!(sub_result, vec![2.0f32, 6.0, 12.0, 20.0]);
    assert_eq!(mul_result, vec![8.0f32, 27.0, 64.0, 125.0]);
    assert_eq!(div_result, vec![2.0f32, 3.0, 4.0, 5.0]);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_queue_dynamic_optimization() -> numrs2::error::Result<()> {
    let context = new_context()?;
    let config = BatchConfig {
        enable_dynamic_optimization: true,
        enable_auto_flush: false,
        max_batch_size: 16,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32; 100]).reshape(&[100]);
    let b = Array::from_vec(vec![2.0f32; 100]).reshape(&[100]);

    let a_gpu = Arc::new(GpuArray::from_array_with_context(&a, context.clone())?);
    let b_gpu = Arc::new(GpuArray::from_array_with_context(&b, context.clone())?);

    // Queue multiple batches to allow optimization to kick in
    for _ in 0..5 {
        for _ in 0..8 {
            queue.queue_add(a_gpu.clone(), b_gpu.clone())?;
        }
        queue.flush()?;
    }

    let stats = queue.statistics()?;

    // Verify that optimization ran
    assert!(stats.total_flushes >= 5);
    assert!(stats.total_executed >= 40);
    assert!(stats.estimated_gpu_occupancy >= 0.0);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_operation_type_properties() {
    assert!(OperationType::MatMul.is_batchable());
    assert!(OperationType::Add.is_batchable());
    assert!(OperationType::Conv2D.is_batchable());

    assert!(OperationType::MatMul.cost_factor() > OperationType::Add.cost_factor());
    assert!(OperationType::Conv2D.cost_factor() > OperationType::Multiply.cost_factor());
}

#[test]
#[cfg(feature = "gpu")]
fn test_batch_config_custom() {
    let config = BatchConfig {
        max_batch_size: 64,
        batch_timeout: std::time::Duration::from_millis(20),
        min_batch_size: 8,
        enable_dynamic_optimization: false,
        enable_auto_flush: false,
        target_occupancy: 0.9,
    };

    assert_eq!(config.max_batch_size, 64);
    assert_eq!(config.min_batch_size, 8);
    assert!(!config.enable_dynamic_optimization);
    assert!(!config.enable_auto_flush);
}
