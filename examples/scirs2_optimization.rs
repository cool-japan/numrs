//! Example demonstrating scirs2-core optimizations in NumRS2
//!
//! This example shows how to use SIMD, GPU, and parallel processing features
//! from scirs2-core for high-performance numerical computing.

use numrs2::prelude::*;
use numrs2::interop::ndarray_compat::to_ndarray;
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "scirs")]
    {
        println!("=== NumRS2 with SciRS2 Core Optimizations ===\n");
        
        // Display available optimizations
        println!("{}\n", get_optimization_info());
        
        // Benchmark SIMD operations
        benchmark_simd()?;
        
        // Benchmark parallel processing
        benchmark_parallel()?;
        
        // Demonstrate adaptive processing
        demo_adaptive_processing()?;
    }
    
    #[cfg(not(feature = "scirs"))]
    {
        println!("This example requires the 'scirs' feature to be enabled.");
        println!("Run with: cargo run --example scirs2_optimization --features scirs");
    }
    
    Ok(())
}

#[cfg(feature = "scirs")]
fn benchmark_simd() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== SIMD Benchmarks ===\n");
    
    // Matrix multiplication benchmark
    let sizes = vec![100, 500, 1000];
    
    for size in sizes {
        println!("Matrix multiplication {}x{}:", size, size);
        
        let a = Array::from_vec((0..size*size).map(|i| i as f32 * 0.001).collect())
            .reshape(&[size, size]);
        let b = Array::from_vec((0..size*size).map(|i| i as f32 * 0.002).collect())
            .reshape(&[size, size]);
        
        // Convert to ndarray for SIMD operations
        let a_nd = to_ndarray(&a).unwrap();
        let b_nd = to_ndarray(&b).unwrap();
        let a_view = a_nd.view().into_dimensionality().unwrap();
        let b_view = b_nd.view().into_dimensionality().unwrap();
        
        // Time SIMD matrix multiplication
        let start = Instant::now();
        let _result = simd_matmul(&a_view, &b_view)?;
        let simd_time = start.elapsed();
        
        // Time regular matrix multiplication
        let start = Instant::now();
        let _result = a.matmul(&b)?;
        let regular_time = start.elapsed();
        
        println!("  SIMD:    {:?}", simd_time);
        println!("  Regular: {:?}", regular_time);
        println!("  Speedup: {:.2}x\n", regular_time.as_secs_f64() / simd_time.as_secs_f64());
    }
    
    // Vector operations benchmark
    println!("Vector operations (10M elements):");
    let size = 10_000_000;
    let v = Array::from_vec((0..size).map(|i| i as f32 * 0.001).collect());
    let v_nd = to_ndarray(&v).unwrap();
    let v_view = v_nd.view().into_dimensionality().unwrap();
    
    let start = Instant::now();
    let result = simd_vector_ops(&v_view);
    let simd_time = start.elapsed();
    
    println!("  SIMD vector stats computed in: {:?}", simd_time);
    println!("  Sum: {:.2}, Mean: {:.2}, Norm: {:.2}", 
             result.sum, result.mean, result.norm);
    println!("  Min: {:.2}, Max: {:.2}\n", result.min, result.max);
    
    Ok(())
}

#[cfg(feature = "scirs")]
fn benchmark_parallel() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Parallel Processing Benchmarks ===\n");
    
    // Column statistics on large dataset
    let rows = 1000;
    let cols = 1000;
    
    println!("Computing statistics for {}x{} matrix:", rows, cols);
    
    let data = Array::from_vec((0..rows*cols).map(|i| i as f64 * 0.001).collect())
        .reshape(&[rows, cols]);
    let data_nd = to_ndarray(&data).unwrap();
    let data_view = data_nd.view().into_dimensionality().unwrap();
    
    let start = Instant::now();
    let stats = parallel_column_statistics(&data_view);
    let parallel_time = start.elapsed();
    
    println!("  Parallel processing completed in: {:?}", parallel_time);
    println!("  Computed statistics for {} columns", stats.len());
    println!("  Sample stats (column 0): mean={:.4}, sum={:.2}, min={:.4}, max={:.4}\n",
             stats[0].mean, stats[0].sum, stats[0].min, stats[0].max);
    
    // Chunked processing demonstration
    println!("Chunked processing (100M elements):");
    let size = 100_000_000;
    let data = Array::from_vec((0..size).map(|i| i as f64 * 0.001).collect());
    let data_nd = to_ndarray(&data).unwrap();
    let data_view = data_nd.view().into_dimensionality().unwrap();
    
    let chunk_size = 1_000_000;
    let start = Instant::now();
    let chunk_sums = chunked_array_processing(&data_view, chunk_size, |chunk| {
        chunk.iter().map(|&x| x * x).sum::<f64>()
    });
    let chunked_time = start.elapsed();
    
    println!("  Processed {} chunks in: {:?}", chunk_sums.len(), chunked_time);
    println!("  Total sum of squares: {:.2}\n", chunk_sums.iter().sum::<f64>());
    
    Ok(())
}

#[cfg(feature = "scirs")]
fn demo_adaptive_processing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Adaptive Processing ===\n");
    
    let sizes = vec![100, 10_000, 1_000_000, 10_000_000];
    
    for size in sizes {
        let data = Array::from_vec((0..size).map(|i| i as f64 * 0.001).collect());
        let data_nd = to_ndarray(&data).unwrap();
        let data_view = data_nd.view().into_dimensionality().unwrap();
        
        let start = Instant::now();
        let sum = adaptive_array_sum(&data_view);
        let time = start.elapsed();
        
        let method = if should_use_parallel(size) {
            "parallel"
        } else {
            "sequential/SIMD"
        };
        
        println!("Size: {:>10}, Method: {:>15}, Time: {:?}, Sum: {:.2}",
                 size, method, time, sum);
    }
    
    Ok(())
}