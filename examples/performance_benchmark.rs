//! Performance benchmarks comparing standard vs optimized operations
//!
//! This example demonstrates the performance improvements from using
//! SIMD and parallel processing optimizations in NumRS2.

use ndarray::{Array1, Array2};

#[cfg(feature = "scirs")]
use numrs2::optimized_ops::{
    enhanced_exp, enhanced_math, get_optimization_info, process_large_array, simd_matmul,
    SimdMathOps,
};

use std::time::Instant;

fn benchmark_operation<F>(name: &str, size: usize, iterations: usize, mut op: F) -> f64
where
    F: FnMut(),
{
    // Warm up
    for _ in 0..5 {
        op();
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        op();
    }
    let duration = start.elapsed();

    let ops_per_sec = (iterations as f64 * size as f64) / duration.as_secs_f64();
    println!("{:<40} {:>15.2} ops/sec", name, ops_per_sec);

    ops_per_sec
}

#[allow(dead_code)]
fn benchmark_trigonometric_functions() {
    println!("\n=== Trigonometric Function Benchmarks ===");

    let sizes = vec![100, 1000, 10000, 100000];

    for &size in &sizes {
        println!("\nArray size: {}", size);

        let data = Array1::from_vec((0..size).map(|x| x as f64 * 0.01).collect::<Vec<_>>());
        let data_view = data.view();

        // Standard implementation
        let std_ops = benchmark_operation("Standard sin (scalar)", size, 100, || {
            let _ = data_view.map(|&x| x.sin());
        });

        // Parallel implementation
        #[cfg(feature = "scirs")]
        let par_ops = benchmark_operation("Parallel sin", size, 100, || {
            let _ = enhanced_math::parallel_sin(&data_view);
        });

        #[cfg(feature = "scirs")]
        println!("Speedup: {:.2}x", par_ops / std_ops);
    }
}

#[allow(dead_code)]
fn benchmark_exponential_functions() {
    println!("\n=== Exponential Function Benchmarks ===");

    let sizes = vec![100, 1000, 10000, 100000];

    for &size in &sizes {
        println!("\nArray size: {}", size);

        let data = Array1::from_vec(
            (0..size)
                .map(|x| (x as f64 * 0.01).min(10.0))
                .collect::<Vec<_>>(),
        );
        let data_view = data.view();

        // Standard sqrt
        let std_ops = benchmark_operation("Standard sqrt (scalar)", size, 100, || {
            let _ = data_view.map(|&x| x.sqrt());
        });

        // SIMD sqrt
        #[cfg(feature = "scirs")]
        let simd_ops = benchmark_operation("SIMD sqrt", size, 100, || {
            let _ = enhanced_exp::simd_sqrt(&data_view);
        });

        #[cfg(feature = "scirs")]
        println!("Speedup: {:.2}x", simd_ops / std_ops);
    }
}

#[allow(dead_code)]
fn benchmark_matrix_operations() {
    println!("\n=== Matrix Operation Benchmarks ===");

    let sizes = vec![10, 50, 100, 200];

    for &size in &sizes {
        println!("\nMatrix size: {}x{}", size, size);

        let a = Array2::from_shape_vec(
            (size, size),
            (0..size * size)
                .map(|x| x as f32 * 0.01)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let b = Array2::from_shape_vec(
            (size, size),
            (0..size * size)
                .map(|x| x as f32 * 0.02)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let a_view = a.view();
        let b_view = b.view();

        // Standard matmul
        let std_ops = benchmark_operation("Standard matmul", size * size, 10, || {
            let _ = a.dot(&b);
        });

        // SIMD matmul
        #[cfg(feature = "scirs")]
        let simd_ops = benchmark_operation("SIMD matmul", size * size, 10, || {
            let _ = simd_matmul(&a_view, &b_view);
        });

        #[cfg(feature = "scirs")]
        println!("Speedup: {:.2}x", simd_ops / std_ops);
    }
}

#[allow(dead_code)]
fn benchmark_large_array_processing() {
    println!("\n=== Large Array Processing Benchmarks ===");

    let size = 1_000_000;
    let chunk_sizes = vec![1000, 10000, 100000];

    let data = Array1::from_vec((0..size).map(|x| x as f64 * 0.001).collect::<Vec<_>>());
    let data_view = data.view();

    println!("\nProcessing {} elements", size);

    // Standard processing (all at once)
    let std_time = {
        let start = Instant::now();
        let _ = data_view.map(|&x| x.sin() + x.cos());
        start.elapsed().as_secs_f64()
    };
    println!("Standard processing: {:.3} seconds", std_time);

    // Chunked processing with different chunk sizes
    #[cfg(feature = "scirs")]
    for &chunk_size in &chunk_sizes {
        let chunked_time = {
            let start = Instant::now();
            let _ = process_large_array(&data_view, chunk_size, |chunk| {
                chunk.map(|&x| x.sin() + x.cos())
            });
            start.elapsed().as_secs_f64()
        };

        println!(
            "Chunked processing (chunk_size={}): {:.3} seconds (speedup: {:.2}x)",
            chunk_size,
            chunked_time,
            std_time / chunked_time
        );
    }
}

#[allow(dead_code)]
fn benchmark_adaptive_algorithms() {
    println!("\n=== Adaptive Algorithm Benchmarks ===");

    let sizes = vec![10, 100, 1000, 10000, 100000];

    println!("\nTesting adaptive algorithm selection for sqrt:");

    #[cfg(feature = "scirs")]
    for &size in &sizes {
        let data = Array1::from_vec((0..size).map(|x| (x + 1) as f64).collect::<Vec<_>>());
        let data_view = data.view();

        let time = {
            let start = Instant::now();
            let _ = SimdMathOps::adaptive_math_function(
                &data_view,
                |data| enhanced_exp::simd_sqrt(data),
                |x| x.sqrt(),
            );
            start.elapsed()
        };

        println!(
            "Size: {:>6} - Time: {:>8.3} ms - Rate: {:>10.2} Mops/sec",
            size,
            time.as_secs_f64() * 1000.0,
            (size as f64) / time.as_secs_f64() / 1_000_000.0
        );
    }
}

fn main() {
    println!("NumRS2 Performance Benchmark");
    println!("============================");

    #[cfg(feature = "scirs")]
    {
        println!("\n{}", get_optimization_info());

        benchmark_trigonometric_functions();
        benchmark_exponential_functions();
        benchmark_matrix_operations();
        benchmark_large_array_processing();
        benchmark_adaptive_algorithms();

        println!("\n=== Summary ===");
        println!("The benchmarks demonstrate significant performance improvements when using:");
        println!("- Parallel processing for trigonometric functions");
        println!("- SIMD operations for mathematical functions");
        println!("- Optimized matrix multiplication");
        println!("- Chunked processing for memory efficiency");
        println!("- Adaptive algorithm selection based on data size");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("\nNote: This example requires the 'scirs' feature to be enabled.");
        println!("Run with: cargo run --example performance_benchmark --features scirs");
    }
}
