//! Comprehensive GPU performance benchmarks
//!
//! This example compares performance across CPU (scalar), SIMD, and GPU implementations
//! for various operations in NumRS2.
//!
//! To run:
//! ```
//! cargo run --example gpu_benchmark --features "gpu scirs"
//! ```

use numrs2::error::Result;
use numrs2::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use std::time::Instant;

#[cfg(feature = "gpu")]
use numrs2::gpu;

#[cfg(feature = "scirs")]
use numrs2::optimized_ops::{
    enhanced_exp, enhanced_math, get_optimization_info, simd_elementwise_ops, simd_matmul,
};

/// Benchmark result struct
#[derive(Debug)]
struct BenchmarkResult {
    cpu_time: f64,
    #[cfg(feature = "scirs")]
    simd_time: Option<f64>,
    #[cfg(feature = "gpu")]
    gpu_time: Option<f64>,
}

impl BenchmarkResult {
    fn print_summary(&self, operation: &str, size: usize) {
        println!("\n{} (size: {})", operation, size);
        println!("{:-<50}", "");
        println!("CPU time:  {:>10.3} ms", self.cpu_time * 1000.0);

        #[cfg(feature = "scirs")]
        if let Some(simd_time) = self.simd_time {
            let speedup = self.cpu_time / simd_time;
            println!(
                "SIMD time: {:>10.3} ms (speedup: {:.2}x)",
                simd_time * 1000.0,
                speedup
            );
        }

        #[cfg(feature = "gpu")]
        if let Some(gpu_time) = self.gpu_time {
            let speedup = self.cpu_time / gpu_time;
            println!(
                "GPU time:  {:>10.3} ms (speedup: {:.2}x)",
                gpu_time * 1000.0,
                speedup
            );
        }
    }
}

fn benchmark_element_wise_operations(size: usize) -> Result<()> {
    println!("\n=== Element-wise Operations ===");

    // Create test data
    let a = Array::from_vec(vec![1.0f32; size]).reshape(&[size]);
    let b = Array::from_vec(vec![2.0f32; size]).reshape(&[size]);

    // Addition benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU addition
    let start = Instant::now();
    let _cpu_result = a.add(&b);
    result.cpu_time = start.elapsed().as_secs_f64();

    // SIMD addition
    #[cfg(feature = "scirs")]
    {
        // Convert f32 to f64 for SIMD operations
        let a_f64 = a.map(|x| x as f64);
        let b_f64 = b.map(|x| x as f64);
        let a_ndarray = Array1::from_vec(a_f64.to_vec());
        let b_ndarray = Array1::from_vec(b_f64.to_vec());
        let start = Instant::now();
        let simd_result = simd_elementwise_ops(&a_ndarray.view(), &b_ndarray.view())?;
        let _ = simd_result.add;
        result.simd_time = Some(start.elapsed().as_secs_f64());
    }

    // GPU addition
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_a = gpu::GpuArray::from_array(&a)?;
        let gpu_b = gpu::GpuArray::from_array(&b)?;
        let gpu_result = gpu::add(&gpu_a, &gpu_b)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Addition", size);

    // Multiplication benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU multiplication
    let start = Instant::now();
    let _cpu_result = a.multiply(&b);
    result.cpu_time = start.elapsed().as_secs_f64();

    // SIMD multiplication
    #[cfg(feature = "scirs")]
    {
        // Convert f32 to f64 for SIMD operations
        let a_f64 = a.map(|x| x as f64);
        let b_f64 = b.map(|x| x as f64);
        let a_ndarray = Array1::from_vec(a_f64.to_vec());
        let b_ndarray = Array1::from_vec(b_f64.to_vec());
        let start = Instant::now();
        let simd_result = simd_elementwise_ops(&a_ndarray.view(), &b_ndarray.view())?;
        let _ = simd_result.mul;
        result.simd_time = Some(start.elapsed().as_secs_f64());
    }

    // GPU multiplication
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_a = gpu::GpuArray::from_array(&a)?;
        let gpu_b = gpu::GpuArray::from_array(&b)?;
        let gpu_result = gpu::multiply(&gpu_a, &gpu_b)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Multiplication", size);

    Ok(())
}

fn benchmark_transcendental_functions(size: usize) -> Result<()> {
    println!("\n=== Transcendental Functions ===");

    // Create test data
    let data = Array::from_vec(
        (0..size)
            .map(|x| (x as f32 * 0.001).min(10.0))
            .collect::<Vec<_>>(),
    )
    .reshape(&[size]);

    // Exponential benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU exponential
    let start = Instant::now();
    let _cpu_result = data.map(|x| x.exp());
    result.cpu_time = start.elapsed().as_secs_f64();

    // SIMD exponential
    #[cfg(feature = "scirs")]
    {
        let data_f64 = data.map(|x| x as f64);
        let data_ndarray = Array1::from_vec(data_f64.to_vec());
        let start = Instant::now();
        let _ = enhanced_exp::parallel_exp(&data_ndarray.view());
        result.simd_time = Some(start.elapsed().as_secs_f64());
    }

    // GPU exponential
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_data = gpu::GpuArray::from_array(&data)?;
        let gpu_result = gpu::exp(&gpu_data)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Exponential", size);

    // Sine benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU sine
    let start = Instant::now();
    let _cpu_result = data.map(|x| x.sin());
    result.cpu_time = start.elapsed().as_secs_f64();

    // SIMD sine
    #[cfg(feature = "scirs")]
    {
        let data_f64 = data.map(|x| x as f64);
        let data_ndarray = Array1::from_vec(data_f64.to_vec());
        let start = Instant::now();
        let _ = enhanced_math::parallel_sin(&data_ndarray.view());
        result.simd_time = Some(start.elapsed().as_secs_f64());
    }

    // GPU sine
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_data = gpu::GpuArray::from_array(&data)?;
        let gpu_result = gpu::sin(&gpu_data)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Sine", size);

    // Square root benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU sqrt
    let start = Instant::now();
    let _cpu_result = data.map(|x| x.sqrt());
    result.cpu_time = start.elapsed().as_secs_f64();

    // SIMD sqrt
    #[cfg(feature = "scirs")]
    {
        let data_f64 = data.map(|x| x as f64);
        let data_ndarray = Array1::from_vec(data_f64.to_vec());
        let start = Instant::now();
        let _ = enhanced_exp::simd_sqrt(&data_ndarray.view());
        result.simd_time = Some(start.elapsed().as_secs_f64());
    }

    // GPU sqrt
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_data = gpu::GpuArray::from_array(&data)?;
        let gpu_result = gpu::sqrt(&gpu_data)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Square Root", size);

    Ok(())
}

fn benchmark_matrix_operations(size: usize) -> Result<()> {
    println!("\n=== Matrix Operations ===");

    // Create test matrices
    let a = Array::from_vec(
        (0..size * size)
            .map(|x| x as f32 * 0.001)
            .collect::<Vec<_>>(),
    )
    .reshape(&[size, size]);
    let b = Array::from_vec(
        (0..size * size)
            .map(|x| x as f32 * 0.002)
            .collect::<Vec<_>>(),
    )
    .reshape(&[size, size]);

    // Matrix multiplication benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU matmul
    let start = Instant::now();
    let _cpu_result = a.dot(&b)?;
    result.cpu_time = start.elapsed().as_secs_f64();

    // SIMD matmul
    #[cfg(feature = "scirs")]
    {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        let a_ndarray = Array2::from_shape_vec((size, size), a_vec).unwrap();
        let b_ndarray = Array2::from_shape_vec((size, size), b_vec).unwrap();
        let start = Instant::now();
        let _ = simd_matmul(&a_ndarray.view(), &b_ndarray.view())?;
        result.simd_time = Some(start.elapsed().as_secs_f64());
    }

    // GPU matmul
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_a = gpu::GpuArray::from_array(&a)?;
        let gpu_b = gpu::GpuArray::from_array(&b)?;
        let gpu_result = gpu::matmul(&gpu_a, &gpu_b)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Matrix Multiplication", size);

    // Matrix transpose benchmark
    let mut result = BenchmarkResult {
        cpu_time: 0.0,
        #[cfg(feature = "scirs")]
        simd_time: None,
        #[cfg(feature = "gpu")]
        gpu_time: None,
    };

    // CPU transpose
    let start = Instant::now();
    let _cpu_result = a.transpose();
    result.cpu_time = start.elapsed().as_secs_f64();

    // GPU transpose
    #[cfg(feature = "gpu")]
    {
        let start = Instant::now();
        let gpu_a = gpu::GpuArray::from_array(&a)?;
        let gpu_result = gpu::transpose(&gpu_a)?;
        let _ = gpu_result.to_array()?;
        result.gpu_time = Some(start.elapsed().as_secs_f64());
    }

    result.print_summary("Matrix Transpose", size);

    Ok(())
}

#[allow(dead_code)]
fn benchmark_memory_transfer(size: usize) -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        println!("\n=== Memory Transfer Overhead ===");

        // Create test data
        let data = Array::from_vec(vec![1.0f32; size]).reshape(&[size]);

        // Measure CPU to GPU transfer
        let start = Instant::now();
        let gpu_data = gpu::GpuArray::from_array(&data)?;
        let to_gpu_time = start.elapsed().as_secs_f64();

        // Measure GPU to CPU transfer
        let start = Instant::now();
        let _cpu_data = gpu_data.to_array()?;
        let to_cpu_time = start.elapsed().as_secs_f64();

        println!(
            "Data size: {} elements ({:.2} MB)",
            size,
            (size * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)
        );
        println!(
            "CPU → GPU transfer: {:.3} ms ({:.2} GB/s)",
            to_gpu_time * 1000.0,
            (size * std::mem::size_of::<f32>()) as f64 / to_gpu_time / 1e9
        );
        println!(
            "GPU → CPU transfer: {:.3} ms ({:.2} GB/s)",
            to_cpu_time * 1000.0,
            (size * std::mem::size_of::<f32>()) as f64 / to_cpu_time / 1e9
        );
    }

    Ok(())
}

fn print_performance_recommendations() {
    println!("\n=== Performance Recommendations ===");
    println!("Based on the benchmark results:");

    println!("\n1. Element-wise operations:");
    println!("   - Small arrays (< 1000): Use CPU");
    println!("   - Medium arrays (1000-100000): Use SIMD");
    println!("   - Large arrays (> 100000): Use GPU");

    println!("\n2. Matrix operations:");
    println!("   - Small matrices (< 100x100): Use CPU");
    println!("   - Medium matrices (100x100 - 500x500): Use SIMD");
    println!("   - Large matrices (> 500x500): Use GPU");

    println!("\n3. Transcendental functions:");
    println!("   - Always benefit from SIMD when available");
    println!("   - GPU provides best performance for large arrays");

    println!("\n4. Consider memory transfer overhead:");
    println!("   - GPU is most beneficial for operations that can be batched");
    println!("   - Avoid frequent CPU ↔ GPU transfers");
}

fn main() -> Result<()> {
    println!("NumRS2 GPU Performance Benchmarks");
    println!("=================================");

    // Print system information
    #[cfg(feature = "scirs")]
    println!("\n{}", get_optimization_info());

    #[cfg(feature = "gpu")]
    {
        if let Some(gpu_info) = gpu::get_gpu_info() {
            println!("GPU: {}", gpu_info);
        }
    }

    // Run benchmarks with different sizes
    let sizes = vec![
        100,       // Very small
        1_000,     // Small
        10_000,    // Medium
        100_000,   // Large
        1_000_000, // Very large
    ];

    for &size in &sizes {
        println!("\n\n╔══════════════════════════════════════════╗");
        println!("║  Benchmarking with {} elements", size);
        println!("╚══════════════════════════════════════════╝");

        benchmark_element_wise_operations(size)?;
        benchmark_transcendental_functions(size)?;

        // Matrix operations only for reasonable sizes
        if size <= 1000 {
            let matrix_size = (size as f64).sqrt() as usize;
            benchmark_matrix_operations(matrix_size)?;
        }

        #[cfg(feature = "gpu")]
        benchmark_memory_transfer(size)?;
    }

    print_performance_recommendations();

    println!("\n\nBenchmark completed successfully!");

    #[cfg(not(all(feature = "gpu", feature = "scirs")))]
    {
        println!("\nNote: To see all optimizations, run with:");
        println!("cargo run --example gpu_benchmark --features \"gpu scirs\"");
    }

    Ok(())
}
