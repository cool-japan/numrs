//! GPU Acceleration Example
//!
//! This example demonstrates how to use GPU acceleration in NumRS2 for various operations
//! including memory management, linear algebra, and performance comparisons.
//!
//! Run with: `cargo run --example gpu_acceleration --features gpu`

#![allow(clippy::result_large_err)]

#[cfg(feature = "gpu")]
use numrs2::array::Array;
#[cfg(feature = "gpu")]
use numrs2::gpu::benchmarks::{BenchmarkConfig, BenchmarkRunner};
#[cfg(feature = "gpu")]
use numrs2::gpu::compute::{KernelBuilder, KernelOp, ShaderCache};
#[cfg(feature = "gpu")]
use numrs2::gpu::linalg;
#[cfg(feature = "gpu")]
use numrs2::gpu::memory::{
    BufferAliasManager, DoubleBuffer, GpuMemoryPool, TransferOptimizer, TransferStrategy,
};
#[cfg(feature = "gpu")]
use numrs2::gpu::{new_context, GpuArray};

#[cfg(feature = "gpu")]
fn main() -> numrs2::error::Result<()> {
    println!("=== NumRS2 GPU Acceleration Example ===\n");

    // 1. GPU Context and Information
    println!("1. Creating GPU Context");
    println!("------------------------");
    let context = new_context()?;
    if let Some(info) = numrs2::gpu::util::get_gpu_info() {
        println!("GPU Information:\n{}", info);
    }
    println!();

    // 2. Basic GPU Operations
    println!("2. Basic GPU Operations");
    println!("-----------------------");
    basic_operations(&context)?;
    println!();

    // 3. Linear Algebra on GPU
    println!("3. GPU Linear Algebra");
    println!("---------------------");
    linear_algebra_demo(&context)?;
    println!();

    // 4. Memory Management
    println!("4. GPU Memory Management");
    println!("------------------------");
    memory_management_demo(&context)?;
    println!();

    // 6. Advanced Compute Shaders
    println!("6. Advanced Compute Shaders");
    println!("----------------------------");
    compute_shader_demo(&context)?;
    println!();

    // 5. Performance Benchmarks
    println!("5. Performance Benchmarks");
    println!("-------------------------");
    performance_benchmarks(&context)?;
    println!();

    println!("=== Example Complete ===");
    Ok(())
}

#[cfg(feature = "gpu")]
fn compute_shader_demo(context: &numrs2::gpu::GpuContextRef) -> numrs2::error::Result<()> {
    println!("Testing shader caching and kernel composition...");

    // Create a shader cache
    let cache = ShaderCache::new(context.clone());

    println!(
        "Shader cache created with {} shaders",
        cache.shader_count()?
    );

    // Build a composite kernel
    let kernel = KernelBuilder::new()
        .add_operation(KernelOp::Add)
        .add_operation(KernelOp::Sqrt)
        .add_operation(KernelOp::Exp)
        .build()?;

    println!("Built composite kernel with 3 operations");
    println!("Kernel preview (first 200 chars):");
    println!("{}", &kernel[..200.min(kernel.len())]);

    // Test double buffering
    let double_buf = DoubleBuffer::new(
        context.clone(),
        1024,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    println!(
        "\nDouble buffer created with size: {} bytes",
        double_buf.size()
    );

    // Test buffer aliasing
    let mut alias_manager = BufferAliasManager::new(context.clone());
    let _buf1 = alias_manager.get_or_create_buffer(2048, wgpu::BufferUsages::STORAGE)?;
    let _buf2 = alias_manager.get_or_create_buffer(2048, wgpu::BufferUsages::STORAGE)?;

    let alias_stats = alias_manager.statistics()?;
    println!("Buffer alias statistics:");
    println!("  Total aliases: {}", alias_stats.total_aliases);
    println!("  Total references: {}", alias_stats.total_references);
    println!("  Buffer sizes tracked: {}", alias_stats.buffer_sizes);

    Ok(())
}

#[cfg(feature = "gpu")]
fn basic_operations(context: &numrs2::gpu::GpuContextRef) -> numrs2::error::Result<()> {
    // Create simple arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).reshape(&[5]);
    let b = Array::from_vec(vec![5.0f32, 4.0, 3.0, 2.0, 1.0]).reshape(&[5]);

    println!("CPU Array A: {:?}", a.to_vec());
    println!("CPU Array B: {:?}", b.to_vec());

    // Transfer to GPU
    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    println!(
        "Transferred to GPU: A ({}), B ({})",
        a_gpu.size(),
        b_gpu.size()
    );

    // Perform GPU operations
    let add_result = numrs2::gpu::add(&a_gpu, &b_gpu)?;
    let mul_result = numrs2::gpu::multiply(&a_gpu, &b_gpu)?;

    // Transfer back to CPU
    let add_cpu = add_result.to_array()?;
    let mul_cpu = mul_result.to_array()?;

    println!("Addition Result: {:?}", add_cpu.to_vec());
    println!("Multiplication Result: {:?}", mul_cpu.to_vec());

    Ok(())
}

#[cfg(feature = "gpu")]
fn linear_algebra_demo(context: &numrs2::gpu::GpuContextRef) -> numrs2::error::Result<()> {
    // Create matrices
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let b = Array::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[3, 2]);

    println!("Matrix A (2x3):");
    for i in 0..2 {
        print!("[");
        for j in 0..3 {
            print!("{:6.2}", a.get(&[i, j])?);
        }
        println!(" ]");
    }

    println!("\nMatrix B (3x2):");
    for i in 0..3 {
        print!("[");
        for j in 0..2 {
            print!("{:6.2}", b.get(&[i, j])?);
        }
        println!(" ]");
    }

    // Transfer to GPU
    let a_gpu = GpuArray::from_array_with_context(&a, context.clone())?;
    let b_gpu = GpuArray::from_array_with_context(&b, context.clone())?;

    // Matrix multiplication
    let c_gpu = linalg::matmul(&a_gpu, &b_gpu)?;
    let c = c_gpu.to_array()?;

    println!("\nResult C = A * B (2x2):");
    for i in 0..2 {
        print!("[");
        for j in 0..2 {
            print!("{:6.2}", c.get(&[i, j])?);
        }
        println!(" ]");
    }

    // Vector operations
    let x = Array::from_vec(vec![1.0f32, 2.0, 3.0]).reshape(&[3]);
    let y = Array::from_vec(vec![4.0f32, 5.0, 6.0]).reshape(&[3]);

    let x_gpu = GpuArray::from_array_with_context(&x, context.clone())?;
    let y_gpu = GpuArray::from_array_with_context(&y, context.clone())?;

    let dot_product = linalg::dot(&x_gpu, &y_gpu)?;
    let norm_x = linalg::norm_l2(&x_gpu)?;

    println!("\nVector x: {:?}", x.to_vec());
    println!("Vector y: {:?}", y.to_vec());
    println!("Dot product x·y: {:.2}", dot_product);
    println!("L2 norm of x: {:.2}", norm_x);

    Ok(())
}

#[cfg(feature = "gpu")]
fn memory_management_demo(context: &numrs2::gpu::GpuContextRef) -> numrs2::error::Result<()> {
    // Create a memory pool
    let mut pool = GpuMemoryPool::new(context.clone());

    println!("Created GPU memory pool");

    // Allocate some buffers
    {
        let _buf1 = pool.allocate(1024, wgpu::BufferUsages::STORAGE)?;
        let _buf2 = pool.allocate(2048, wgpu::BufferUsages::STORAGE)?;
        let _buf3 = pool.allocate(1024, wgpu::BufferUsages::STORAGE)?;
        println!("Allocated 3 buffers (1KB, 2KB, 1KB)");
    }

    // Buffers are returned to pool when dropped
    let stats = pool.statistics()?;
    println!("Pool statistics after return:");
    println!("  Total buffers: {}", stats.total_buffers);
    println!("  Total bytes: {}", stats.total_bytes);
    println!("  Pool types: {}", stats.pool_count);

    // Reuse buffers
    let _buf4 = pool.allocate(1024, wgpu::BufferUsages::STORAGE)?;
    println!("\nAllocated new buffer (should reuse from pool)");
    let stats = pool.statistics()?;
    println!("Pool statistics after reuse:");
    println!("  Total buffers: {}", stats.total_buffers);

    // Demonstrate transfer optimizer
    let mut optimizer = TransferOptimizer::new(context.clone(), TransferStrategy::Batched);
    println!("\nCreated transfer optimizer with batched strategy");
    println!("Transfer strategy: {:?}", optimizer.strategy());

    optimizer.set_strategy(TransferStrategy::Immediate);
    println!("Changed to immediate strategy: {:?}", optimizer.strategy());

    Ok(())
}

#[cfg(feature = "gpu")]
fn performance_benchmarks(context: &numrs2::gpu::GpuContextRef) -> numrs2::error::Result<()> {
    let runner = BenchmarkRunner::new(context.clone());
    let config = BenchmarkConfig {
        warmup_iterations: 2,
        benchmark_iterations: 5,
        include_cpu: true,
        measure_transfers: true,
    };

    println!("Running benchmarks (2 warmup, 5 iterations)...\n");

    // Small matrix multiplication
    println!("Small Matrix Multiplication (64x64):");
    let small_results = runner.benchmark_matmul(64, 64, 64, &config)?;
    print_benchmark_results(&small_results);

    // Medium matrix multiplication
    println!("\nMedium Matrix Multiplication (256x256):");
    let medium_results = runner.benchmark_matmul(256, 256, 256, &config)?;
    print_benchmark_results(&medium_results);

    // Element-wise operations
    println!("\nElement-wise Operations (1M elements):");
    let elem_results = runner.benchmark_elementwise(1_000_000, &config)?;
    print_benchmark_results(&elem_results);

    // Memory transfer benchmark
    println!("\nMemory Transfer Bandwidth (16MB):");
    let (to_gpu_bw, from_gpu_bw) = runner.benchmark_memory_transfer(16 * 1024 * 1024)?;
    println!("  CPU -> GPU: {:.2} GB/s", to_gpu_bw);
    println!("  GPU -> CPU: {:.2} GB/s", from_gpu_bw);

    Ok(())
}

#[cfg(feature = "gpu")]
fn print_benchmark_results(results: &numrs2::gpu::benchmarks::BenchmarkResults) {
    println!("  GPU time: {:.3} ms", results.gpu_time_ms);

    if let Some(cpu_time) = results.cpu_time_ms {
        println!("  CPU time: {:.3} ms", cpu_time);
        if let Some(speedup) = results.speedup() {
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    if let Some(transfer_to) = results.transfer_to_gpu_ms {
        println!("  Transfer to GPU: {:.3} ms", transfer_to);
    }

    if let Some(transfer_from) = results.transfer_from_gpu_ms {
        println!("  Transfer from GPU: {:.3} ms", transfer_from);
    }

    println!("  GPU throughput: {:.2} GFLOPS", results.gpu_gflops());

    if let Some(effective) = results.effective_speedup() {
        println!("  Effective speedup (with transfers): {:.2}x", effective);
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU support is not enabled.");
    println!("Recompile with: cargo run --example gpu_acceleration --features gpu");
}
