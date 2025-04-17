use numrs2::prelude::*;
use numrs2::memory_optimize::{
    LayoutStrategy, 
    PlacementStrategy,
    AlignmentStrategy,
    optimize_layout,
    optimize_placement,
    align_data,
    optimize_memory
};
use std::time::Instant;

fn main() {
    println!("NumRS Memory Layout Optimization Example");
    println!("======================================\n");

    // SECTION 1: Basic memory layout optimization
    println!("1. Basic Memory Layout Optimization");
    println!("----------------------------------");

    // Create a vector of floats
    let size = 1_000_000;
    let mut data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    
    // Measure performance with default layout
    let start = Instant::now();
    let sum = naive_sum(&data);
    let default_time = start.elapsed();
    println!("Sum with default layout: {} (time: {:?})", sum, default_time);
    
    // Optimize the layout for better cache efficiency
    optimize_layout(&mut data, LayoutStrategy::RowMajor);
    
    // Measure performance with optimized layout
    let start = Instant::now();
    let sum = naive_sum(&data);
    let optimized_time = start.elapsed();
    println!("Sum with optimized layout: {} (time: {:?})", sum, optimized_time);
    println!("Speedup: {:.2}x", default_time.as_secs_f64() / optimized_time.as_secs_f64());

    // SECTION 2: Memory Placement Strategies
    println!("\n2. Memory Placement Strategies");
    println!("-----------------------------");

    // Create a large matrix
    let n = 1_000;
    let mut matrix: Vec<f64> = (0..n*n).map(|i| i as f64).collect();
    
    // Measure performance with default placement
    let start = Instant::now();
    let sum = naive_matrix_sum(&matrix, n);
    let default_time = start.elapsed();
    println!("Matrix sum with default placement: {} (time: {:?})", sum, default_time);
    
    // Optimize the placement for better cache efficiency
    optimize_placement(&mut matrix, PlacementStrategy::CacheAware);
    
    // Measure performance with optimized placement
    let start = Instant::now();
    let sum = naive_matrix_sum(&matrix, n);
    let optimized_time = start.elapsed();
    println!("Matrix sum with optimized placement: {} (time: {:?})", sum, optimized_time);
    println!("Speedup: {:.2}x", default_time.as_secs_f64() / optimized_time.as_secs_f64());

    // SECTION 3: Alignment Optimization
    println!("\n3. Alignment Optimization");
    println!("-----------------------");

    // Create a vector for SIMD operations
    let size = 1_000_000;
    let mut data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    
    // Measure performance with default alignment
    let start = Instant::now();
    let sum = simd_like_sum(&data);
    let default_time = start.elapsed();
    println!("SIMD-like sum with default alignment: {} (time: {:?})", sum, default_time);
    
    // Optimize the alignment for SIMD operations
    align_data(&mut data, AlignmentStrategy::Simd);
    
    // Measure performance with optimized alignment
    let start = Instant::now();
    let sum = simd_like_sum(&data);
    let optimized_time = start.elapsed();
    println!("SIMD-like sum with optimized alignment: {} (time: {:?})", sum, optimized_time);
    println!("Speedup: {:.2}x", default_time.as_secs_f64() / optimized_time.as_secs_f64());

    // SECTION 4: Combined Optimization
    println!("\n4. Combined Optimization");
    println!("----------------------");

    // Create a large matrix
    let n = 1_000;
    let mut matrix: Vec<f64> = (0..n*n).map(|i| i as f64).collect();
    
    // Measure performance with no optimization
    let start = Instant::now();
    let sum = naive_matrix_sum(&matrix, n);
    let default_time = start.elapsed();
    println!("Matrix sum with no optimization: {} (time: {:?})", sum, default_time);
    
    // Apply combined optimization
    optimize_memory(&mut matrix, LayoutStrategy::Blocked(64), PlacementStrategy::Aligned(32));
    
    // Measure performance with combined optimization
    let start = Instant::now();
    let sum = naive_matrix_sum(&matrix, n);
    let optimized_time = start.elapsed();
    println!("Matrix sum with combined optimization: {} (time: {:?})", sum, optimized_time);
    println!("Speedup: {:.2}x", default_time.as_secs_f64() / optimized_time.as_secs_f64());

    // SECTION 5: Using Optimized Memory with NumRS Arrays
    println!("\n5. Using Optimized Memory with NumRS Arrays");
    println!("----------------------------------------");

    // Create a regular NumRS array
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Original array: {:?}", array);
    
    // We can't directly optimize the memory layout of the array,
    // but we can use the optimized functions when creating new arrays
    println!("For NumRS arrays, memory optimizations would be implemented at a lower level");
    println!("in the array implementation itself, ensuring optimal performance for all operations.");
}

// A simple sum function to benchmark performance
fn naive_sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

// A naive matrix sum function to benchmark performance
fn naive_matrix_sum(data: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += data[i * n + j];
        }
    }
    sum
}

// A function simulating SIMD operations
fn simd_like_sum(data: &[f32]) -> f32 {
    // In a real implementation, this would use actual SIMD instructions
    // For now, just use a regular sum with batching to simulate SIMD
    let batch_size = 4; // Simulate 4-wide SIMD
    let mut sum = 0.0;
    
    let full_batches = data.len() / batch_size;
    
    for i in 0..full_batches {
        let start = i * batch_size;
        for j in 0..batch_size {
            sum += data[start + j];
        }
    }
    
    // Handle remaining elements
    for i in (full_batches * batch_size)..data.len() {
        sum += data[i];
    }
    
    sum
}