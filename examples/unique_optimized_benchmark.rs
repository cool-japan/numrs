// Benchmark comparing original and optimized unique implementations
//
// This example compares performance between the standard unique function
// and the optimized version for different array sizes and scenarios.

use numrs2::error::Result;
use numrs2::prelude::*;
use std::time::{Duration, Instant};

// Import the optimized version
use numrs2::unique_optimized::unique_optimized;

fn benchmark<F, R>(name: &str, runs: u32, f: F) -> Duration
where
    F: Fn() -> R,
{
    // Warmup
    for _ in 0..5 {
        let _ = f();
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..runs {
        let _ = f();
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / runs as f64;

    println!("{}: avg {:.2} ms per run ({} runs)", name, avg_ms, runs);
    elapsed
}

fn main() -> Result<()> {
    println!("NumRS2 Unique Function Optimization Comparison");
    println!("===========================================");

    // Small array test
    println!("\nSmall array test (1,000 elements):");
    let small_array = Array::from_vec((0..1000).map(|i| i % 800).collect::<Vec<i32>>());

    let original_small = benchmark("Original unique - small array", 50, || {
        unique(&small_array, None, None, None, None)
    });

    let optimized_small = benchmark("Optimized unique - small array", 50, || {
        unique_optimized(&small_array, None, None, None, None)
    });

    print_comparison(original_small, optimized_small);

    // Medium array test
    println!("\nMedium array test (10,000 elements):");
    let medium_array = Array::from_vec((0..10000).map(|i| i % 1000).collect::<Vec<i32>>());

    let original_medium = benchmark("Original unique - medium array", 20, || {
        unique(&medium_array, None, None, None, None)
    });

    let optimized_medium = benchmark("Optimized unique - medium array", 20, || {
        unique_optimized(&medium_array, None, None, None, None)
    });

    print_comparison(original_medium, optimized_medium);

    // Large array test
    println!("\nLarge array test (100,000 elements):");
    let large_array = Array::from_vec((0..100000).map(|i| i % 5000).collect::<Vec<i32>>());

    let original_large = benchmark("Original unique - large array", 5, || {
        unique(&large_array, None, None, None, None)
    });

    let optimized_large = benchmark("Optimized unique - large array", 5, || {
        unique_optimized(&large_array, None, None, None, None)
    });

    print_comparison(original_large, optimized_large);

    // 2D array test with axis parameter
    println!("\n2D array test with axis parameter:");
    let mut array_2d_data = Vec::with_capacity(10000);
    for i in 0..1000 {
        for j in 0..10 {
            array_2d_data.push((i + j) % 500);
        }
    }
    let array_2d = Array::from_vec(array_2d_data).reshape(&[1000, 10]);

    let original_axis = benchmark("Original unique - with axis=0", 10, || {
        unique(&array_2d, Some(0), None, None, None)
    });

    let optimized_axis = benchmark("Optimized unique - with axis=0", 10, || {
        unique_optimized(&array_2d, Some(0), None, None, None)
    });

    print_comparison(original_axis, optimized_axis);

    // Test with all return options
    println!("\nTest with all return options:");
    let original_returns = benchmark("Original unique - all returns", 10, || {
        unique(&medium_array, None, Some(true), Some(true), Some(true))
    });

    let optimized_returns = benchmark("Optimized unique - all returns", 10, || {
        unique_optimized(&medium_array, None, Some(true), Some(true), Some(true))
    });

    print_comparison(original_returns, optimized_returns);

    // Summary
    println!("\nOptimization Summary:");
    println!("====================");
    println!(
        "Small array (1,000 elements): {}",
        format_improvement(original_small, optimized_small)
    );
    println!(
        "Medium array (10,000 elements): {}",
        format_improvement(original_medium, optimized_medium)
    );
    println!(
        "Large array (100,000 elements): {}",
        format_improvement(original_large, optimized_large)
    );
    println!(
        "With axis parameter: {}",
        format_improvement(original_axis, optimized_axis)
    );
    println!(
        "With all return options: {}",
        format_improvement(original_returns, optimized_returns)
    );

    println!("\nBenchmark complete!");

    Ok(())
}

fn print_comparison(original: Duration, optimized: Duration) {
    println!("Comparison: {}", format_improvement(original, optimized));
}

fn format_improvement(original: Duration, optimized: Duration) -> String {
    if optimized < original {
        let percentage = (1.0 - (optimized.as_nanos() as f64 / original.as_nanos() as f64)) * 100.0;
        format!("Optimized is {:.2}% faster", percentage)
    } else {
        let percentage = ((optimized.as_nanos() as f64 / original.as_nanos() as f64) - 1.0) * 100.0;
        format!("Optimized is {:.2}% slower", percentage)
    }
}
