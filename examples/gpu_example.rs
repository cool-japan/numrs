//! Example demonstrating GPU acceleration in NumRS2
//!
//! This example shows how to use GPU acceleration for array operations
//! in NumRS2. It compares the performance of CPU and GPU operations
//! for large arrays.
//!
//! Note: GPU acceleration requires the "gpu" feature to be enabled:
//! ```
//! cargo run --example gpu_example --features gpu
//! ```

use numrs2::array::Array;
use numrs2::error::Result;
use std::time::Instant;

// Conditionally import GPU modules when the feature is enabled
#[cfg(feature = "gpu")]
use numrs2::gpu;

fn main() -> Result<()> {
    println!("NumRS2 GPU Acceleration Example");
    println!("===============================");

    // Check if GPU support is enabled
    #[cfg(feature = "gpu")]
    {
        println!("\nDetecting GPU hardware...");
        if let Some(gpu_info) = gpu::get_gpu_info() {
            println!("✓ GPU detected: {}", gpu_info);
        } else {
            println!("✗ No compatible GPU detected");
            println!("GPU operations will fall back to CPU implementations");
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n✗ GPU support is not enabled in this build");
        println!("Recompile with --features gpu to enable GPU acceleration");
    }

    // Run with small arrays
    run_matrix_multiply_benchmark(100)?;

    // Run with larger arrays for better GPU utilization
    run_matrix_multiply_benchmark(1000)?;

    #[cfg(feature = "gpu")]
    run_matrix_multiply_benchmark(2000)?;

    println!("\nExample completed successfully!");
    Ok(())
}

/// Benchmark matrix multiplication with CPU and GPU operations
fn run_matrix_multiply_benchmark(size: usize) -> Result<()> {
    println!(
        "\nMatrix Multiplication Benchmark ({}x{} matrices)",
        size, size
    );
    println!("---------------------------------------------------");

    // Create random matrices on the CPU
    let a = create_random_matrix(size, size)?;
    let b = create_random_matrix(size, size)?;

    // CPU matrix multiplication
    println!("Running CPU matrix multiplication...");
    let cpu_start = Instant::now();
    let _cpu_result = a.dot(&b)?;
    let cpu_duration = cpu_start.elapsed();
    println!("CPU time: {:.2?}", cpu_duration);

    // GPU matrix multiplication (if available)
    #[cfg(feature = "gpu")]
    {
        println!("Running GPU matrix multiplication...");
        let gpu_start = Instant::now();

        // Transfer matrices to GPU
        let gpu_a = gpu::GpuArray::from_array(&a)?;
        let gpu_b = gpu::GpuArray::from_array(&b)?;

        // Perform matrix multiplication on GPU
        let gpu_result = gpu::matmul(&gpu_a, &gpu_b)?;

        // Transfer result back to CPU
        let _result = gpu_result.to_array()?;

        let gpu_duration = gpu_start.elapsed();
        println!("GPU time: {:.2?}", gpu_duration);

        // Calculate speedup
        if gpu_duration.as_secs_f64() > 0.0 {
            let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
            println!("Speedup: {:.2}x", speedup);
        }
    }

    Ok(())
}

/// Create a random matrix of the specified shape
fn create_random_matrix(rows: usize, cols: usize) -> Result<Array<f32>> {
    use numrs2::random::distributions::uniform;
    uniform(0.0, 1.0, &[rows, cols])
}
