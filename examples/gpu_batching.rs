//! GPU Batching Operations Example
//!
//! This example demonstrates the automatic batching capabilities for GPU operations,
//! including queue management, dynamic optimization, and performance monitoring.
//!
//! Run with: `cargo run --example gpu_batching --features gpu`

#![allow(clippy::result_large_err)]

#[cfg(feature = "gpu")]
use numrs2::array::Array;
#[cfg(feature = "gpu")]
use numrs2::gpu::batching::{BatchConfig, BatchQueue};
#[cfg(feature = "gpu")]
use numrs2::gpu::{new_context, GpuArray};
#[cfg(feature = "gpu")]
use scirs2_core::random::rngs::StdRng;
#[cfg(feature = "gpu")]
use scirs2_core::random::SeedableRng;
#[cfg(feature = "gpu")]
use scirs2_core::random::{Distribution, Uniform};
#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
fn main() -> numrs2::error::Result<()> {
    println!("=== NumRS2 GPU Batching Example ===\n");

    // 1. Basic Batching Usage
    println!("1. Basic Batching Usage");
    println!("------------------------");
    basic_batching_demo()?;
    println!();

    // 2. Automatic vs Manual Flushing
    println!("2. Automatic vs Manual Flushing");
    println!("--------------------------------");
    flushing_demo()?;
    println!();

    // 3. Dynamic Batch Size Optimization
    println!("3. Dynamic Batch Size Optimization");
    println!("-----------------------------------");
    dynamic_optimization_demo()?;
    println!();

    // 4. Performance Comparison
    println!("4. Performance Comparison");
    println!("-------------------------");
    performance_comparison()?;
    println!();

    // 5. Statistics Monitoring
    println!("5. Statistics Monitoring");
    println!("------------------------");
    statistics_demo()?;
    println!();

    // 6. Real-World Use Case: ML Inference
    println!("6. Real-World Use Case: ML Inference");
    println!("-------------------------------------");
    ml_inference_demo()?;
    println!();

    println!("=== Example Complete ===");
    Ok(())
}

#[cfg(feature = "gpu")]
fn basic_batching_demo() -> numrs2::error::Result<()> {
    println!("Creating GPU context and batch queue...");
    let context = new_context()?;

    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create test arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).reshape(&[5]);
    let b = Array::from_vec(vec![2.0f32, 3.0, 4.0, 5.0, 6.0]).reshape(&[5]);

    println!("Input arrays:");
    println!("  A: {:?}", a.to_vec());
    println!("  B: {:?}", b.to_vec());

    // Transfer to GPU
    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    // Queue multiple operations
    println!("\nQueuing operations...");
    queue.queue_add(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
    queue.queue_multiply(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
    queue.queue_subtract(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;

    println!("Queue depth: {}", queue.queue_depth()?);

    // Flush all operations
    println!("\nFlushing batch...");
    let results = queue.flush()?;

    println!("Executed {} operations", results.len());

    // Display results
    for (i, result) in results.iter().enumerate() {
        let cpu_result = result.result.to_array()?;
        println!(
            "  Operation {}: {:?} -> {:?}",
            i,
            result.op_type,
            cpu_result.to_vec()
        );
        println!("    Execution time: {} µs", result.execution_time_us);
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn flushing_demo() -> numrs2::error::Result<()> {
    let context = new_context()?;

    // Test manual flushing
    println!("Testing manual flushing...");
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    let a = Array::from_vec(vec![1.0f32; 10]).reshape(&[10]);
    let b = Array::from_vec(vec![2.0f32; 10]).reshape(&[10]);

    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    for _ in 0..5 {
        queue.queue_add(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
    }

    println!("  Queued 5 operations, depth: {}", queue.queue_depth()?);
    queue.flush()?;
    println!("  After manual flush, depth: {}", queue.queue_depth()?);

    // Test automatic flushing
    println!("\nTesting automatic flushing...");
    let config = BatchConfig {
        enable_auto_flush: true,
        max_batch_size: 3,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    println!("  Queuing operations with auto-flush enabled (max_batch_size=3)...");
    for i in 0..5 {
        queue.queue_add(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
        let depth = queue.queue_depth()?;
        println!("    After operation {}: queue depth = {}", i + 1, depth);
    }

    let stats = queue.statistics()?;
    println!("  Total flushes: {}", stats.total_flushes);

    Ok(())
}

#[cfg(feature = "gpu")]
fn dynamic_optimization_demo() -> numrs2::error::Result<()> {
    let context = new_context()?;

    let config = BatchConfig {
        enable_dynamic_optimization: true,
        enable_auto_flush: false,
        max_batch_size: 32,
        target_occupancy: 0.8,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Create larger arrays for meaningful optimization
    let size = 1000;
    let a = Array::from_vec(vec![1.0f32; size]).reshape(&[size]);
    let b = Array::from_vec(vec![2.0f32; size]).reshape(&[size]);

    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    println!("Running multiple batches to trigger optimization...");

    for batch in 0..10 {
        // Queue operations
        for _ in 0..16 {
            queue.queue_add(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
        }

        // Flush and get statistics
        queue.flush()?;

        let stats = queue.statistics()?;
        println!(
            "  Batch {}: avg_size={:.1}, occupancy={:.2}, throughput={:.1} ops/sec",
            batch,
            stats.avg_batch_size,
            stats.estimated_gpu_occupancy,
            stats.throughput_ops_per_sec
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn performance_comparison() -> numrs2::error::Result<()> {
    let context = new_context()?;

    let size = 1000;
    let num_ops = 50;

    // Create test data
    let a = Array::from_vec(vec![1.0f32; size]).reshape(&[size]);
    let b = Array::from_vec(vec![2.0f32; size]).reshape(&[size]);

    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    // Test unbatched execution
    println!("Unbatched execution ({} operations)...", num_ops);
    let start = Instant::now();

    for _ in 0..num_ops {
        let _result = numrs2::gpu::add(&a_gpu, &b_gpu)?;
    }

    let unbatched_time = start.elapsed();
    println!("  Time: {:?}", unbatched_time);

    // Test batched execution
    println!("\nBatched execution ({} operations)...", num_ops);
    let config = BatchConfig {
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    let start = Instant::now();

    for _ in 0..num_ops {
        queue.queue_add(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
    }
    let _results = queue.flush()?;

    let batched_time = start.elapsed();
    println!("  Time: {:?}", batched_time);

    // Calculate speedup
    let speedup = unbatched_time.as_secs_f64() / batched_time.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("✓ Batching improves performance!");
    } else {
        println!("Note: For small operations, batching overhead may dominate");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn statistics_demo() -> numrs2::error::Result<()> {
    let context = new_context()?;

    let config = BatchConfig {
        enable_dynamic_optimization: true,
        enable_auto_flush: false,
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    let a = Array::from_vec(vec![1.0f32; 100]).reshape(&[100]);
    let b = Array::from_vec(vec![2.0f32; 100]).reshape(&[100]);

    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    // Run multiple batches
    for _ in 0..5 {
        for _ in 0..10 {
            queue.queue_add(Arc::new(a_gpu.clone()), Arc::new(b_gpu.clone()))?;
        }
        queue.flush()?;
    }

    // Get comprehensive statistics
    let stats = queue.statistics()?;

    println!("Comprehensive Statistics:");
    println!("  Total Operations: {}", stats.total_operations);
    println!("  Total Flushes: {}", stats.total_flushes);
    println!("  Total Executed: {}", stats.total_executed);
    println!("  Average Batch Size: {:.2}", stats.avg_batch_size);
    println!("  Maximum Batch Size: {}", stats.max_batch_size);
    println!("  Current Queue Depth: {}", stats.current_queue_depth);
    println!(
        "  Average Execution Time: {} µs",
        stats.avg_execution_time_us
    );
    println!("  Throughput: {:.1} ops/sec", stats.throughput_ops_per_sec);
    println!(
        "  Estimated GPU Occupancy: {:.1}%",
        stats.estimated_gpu_occupancy * 100.0
    );
    println!("  Auto Flush Count: {}", stats.auto_flush_count);
    println!("  Manual Flush Count: {}", stats.manual_flush_count);

    Ok(())
}

#[cfg(feature = "gpu")]
fn ml_inference_demo() -> numrs2::error::Result<()> {
    println!("Simulating ML inference with batched operations...");

    let context = new_context()?;

    let config = BatchConfig {
        enable_auto_flush: true,
        max_batch_size: 8,
        batch_timeout: std::time::Duration::from_millis(5),
        ..Default::default()
    };

    let mut queue: BatchQueue<f32> = BatchQueue::new(context.clone(), config);

    // Simulate inference requests coming in
    let batch_size = 4;
    let feature_dim = 128;

    // Create random input data (simulating inference requests)
    let mut rng = StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0.0f32, 1.0).expect("Failed to create Uniform distribution");

    println!("Processing {} inference requests...", 20);

    for request_id in 0..20 {
        // Generate input features
        let input_data: Vec<f32> = (0..feature_dim).map(|_| uniform.sample(&mut rng)).collect();

        let input = Array::from_vec(input_data).reshape(&[feature_dim]);
        let input_gpu = GpuArray::from_array_with_context(&input, context.clone())?;

        // Weights (simulated)
        let weight_data: Vec<f32> = (0..feature_dim).map(|_| uniform.sample(&mut rng)).collect();

        let weights = Array::from_vec(weight_data).reshape(&[feature_dim]);
        let weights_gpu = GpuArray::from_array_with_context(&weights, context.clone())?;

        // Queue inference computation (simulated as element-wise operations)
        queue.queue_multiply(Arc::new(input_gpu.clone()), Arc::new(weights_gpu.clone()))?;

        if (request_id + 1) % 5 == 0 {
            let stats = queue.statistics()?;
            println!(
                "  After {} requests: queue_depth={}, flushes={}",
                request_id + 1,
                stats.current_queue_depth,
                stats.total_flushes
            );
        }
    }

    // Final flush
    let remaining = queue.flush()?;
    println!("Final flush processed {} operations", remaining.len());

    let final_stats = queue.statistics()?;
    println!("\nInference Statistics:");
    println!("  Total requests processed: {}", final_stats.total_executed);
    println!("  Average batch size: {:.2}", final_stats.avg_batch_size);
    println!(
        "  Throughput: {:.1} inferences/sec",
        final_stats.throughput_ops_per_sec
    );

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU support is not enabled. Recompile with --features gpu");
}
