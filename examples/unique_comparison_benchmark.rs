// Benchmark comparing the original and optimized unique functions
//
// This example measures the performance differences between the standard
// unique function and the optimized version.

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
    let avg_ns = elapsed.as_nanos() as f64 / runs as f64;

    println!("{}: avg {:.2} ns per run ({} runs)", name, avg_ns, runs);
    elapsed
}

fn main() -> Result<()> {
    println!("NumRS2 Unique Function Comparison Benchmark");
    println!("=========================================");

    // Setup random number generator with fixed seed for reproducibility
    let rng = RandomState::with_seed(42);

    println!("\nPart 1: Small Array Comparison (1,000 elements)");
    println!("--------------------------------------------");

    // Generate small array with duplicates
    let small_array = Array::from_vec(
        rng.uniform(0.0f64, 10.0f64, &[1000])?
            .map(|x| (x * 10.0f64).floor() as i32)
            .to_vec(),
    )
    .reshape(&[1000]);

    // Original implementation
    let original_small = benchmark("Original unique - small array", 100, || {
        unique(&small_array, None, None, None, None)
    });

    // Optimized implementation
    let optimized_small = benchmark("Optimized unique - small array", 100, || {
        unique_optimized(&small_array, None, None, None, None)
    });

    let small_improvement = if optimized_small < original_small {
        let percentage =
            (1.0 - (optimized_small.as_nanos() as f64 / original_small.as_nanos() as f64)) * 100.0;
        format!("{:.2}% faster", percentage)
    } else {
        let percentage =
            ((optimized_small.as_nanos() as f64 / original_small.as_nanos() as f64) - 1.0) * 100.0;
        format!("{:.2}% slower", percentage)
    };

    println!("Small array comparison: Optimized is {}", small_improvement);

    println!("\nPart 2: Medium Array Comparison (10,000 elements)");
    println!("----------------------------------------------");

    // Generate medium array with duplicates
    let medium_array = Array::from_vec(
        rng.uniform(0.0f64, 100.0f64, &[10000])?
            .map(|x| (x * 10.0f64).floor() as i32)
            .to_vec(),
    )
    .reshape(&[10000]);

    // Original implementation
    let original_medium = benchmark("Original unique - medium array", 20, || {
        unique(&medium_array, None, None, None, None)
    });

    // Optimized implementation
    let optimized_medium = benchmark("Optimized unique - medium array", 20, || {
        unique_optimized(&medium_array, None, None, None, None)
    });

    let medium_improvement = if optimized_medium < original_medium {
        let percentage = (1.0
            - (optimized_medium.as_nanos() as f64 / original_medium.as_nanos() as f64))
            * 100.0;
        format!("{:.2}% faster", percentage)
    } else {
        let percentage = ((optimized_medium.as_nanos() as f64 / original_medium.as_nanos() as f64)
            - 1.0)
            * 100.0;
        format!("{:.2}% slower", percentage)
    };

    println!(
        "Medium array comparison: Optimized is {}",
        medium_improvement
    );

    println!("\nPart 3: Large Array Comparison (100,000 elements)");
    println!("----------------------------------------------");

    // Generate large array with duplicates
    let large_array = Array::from_vec(
        rng.uniform(0.0f64, 1000.0f64, &[100000])?
            .map(|x| (x * 10.0f64).floor() as i32)
            .to_vec(),
    )
    .reshape(&[100000]);

    // Original implementation
    let original_large = benchmark("Original unique - large array", 5, || {
        unique(&large_array, None, None, None, None)
    });

    // Optimized implementation
    let optimized_large = benchmark("Optimized unique - large array", 5, || {
        unique_optimized(&large_array, None, None, None, None)
    });

    let large_improvement = if optimized_large < original_large {
        let percentage =
            (1.0 - (optimized_large.as_nanos() as f64 / original_large.as_nanos() as f64)) * 100.0;
        format!("{:.2}% faster", percentage)
    } else {
        let percentage =
            ((optimized_large.as_nanos() as f64 / original_large.as_nanos() as f64) - 1.0) * 100.0;
        format!("{:.2}% slower", percentage)
    };

    println!("Large array comparison: Optimized is {}", large_improvement);

    println!("\nPart 4: 2D Array Comparison with Axis Parameter");
    println!("--------------------------------------------");

    // Generate 2D array
    let array_2d = Array::from_vec(
        rng.uniform(0.0f64, 10.0f64, &[1000, 5])?
            .map(|x| (x * 5.0f64).floor() as i32)
            .to_vec(),
    )
    .reshape(&[1000, 5]);

    // Original implementation
    let original_2d = benchmark("Original unique - 2D array with axis=0", 20, || {
        unique(&array_2d, Some(0), None, None, None)
    });

    // Optimized implementation
    let optimized_2d = benchmark("Optimized unique - 2D array with axis=0", 20, || {
        unique_optimized(&array_2d, Some(0), None, None, None)
    });

    let axis_improvement = if optimized_2d < original_2d {
        let percentage =
            (1.0 - (optimized_2d.as_nanos() as f64 / original_2d.as_nanos() as f64)) * 100.0;
        format!("{:.2}% faster", percentage)
    } else {
        let percentage =
            ((optimized_2d.as_nanos() as f64 / original_2d.as_nanos() as f64) - 1.0) * 100.0;
        format!("{:.2}% slower", percentage)
    };

    println!(
        "2D array with axis comparison: Optimized is {}",
        axis_improvement
    );

    println!("\nPart 5: All Return Options Comparison");
    println!("---------------------------------");

    // Original implementation with all returns
    let original_all_returns = benchmark("Original unique - all returns", 20, || {
        unique(&medium_array, None, Some(true), Some(true), Some(true))
    });

    // Optimized implementation with all returns
    let optimized_all_returns = benchmark("Optimized unique - all returns", 20, || {
        unique_optimized(&medium_array, None, Some(true), Some(true), Some(true))
    });

    let returns_improvement = if optimized_all_returns < original_all_returns {
        let percentage = (1.0
            - (optimized_all_returns.as_nanos() as f64 / original_all_returns.as_nanos() as f64))
            * 100.0;
        format!("{:.2}% faster", percentage)
    } else {
        let percentage = ((optimized_all_returns.as_nanos() as f64
            / original_all_returns.as_nanos() as f64)
            - 1.0)
            * 100.0;
        format!("{:.2}% slower", percentage)
    };

    println!(
        "All returns comparison: Optimized is {}",
        returns_improvement
    );

    println!("\nSummary of Optimizations:");
    println!("------------------------");
    println!("Small arrays (1,000 elements): {}", small_improvement);
    println!("Medium arrays (10,000 elements): {}", medium_improvement);
    println!("Large arrays (100,000 elements): {}", large_improvement);
    println!("2D arrays with axis parameter: {}", axis_improvement);
    println!("With all return options: {}", returns_improvement);

    println!("\nBenchmark complete!");

    Ok(())
}
