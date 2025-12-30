#![allow(deprecated)]
#![allow(clippy::result_large_err)]

// Benchmark for the `unique` function in NumRS2
//
// This example measures the performance of the unique function on arrays of
// different sizes and types. It compares performance across different scenarios:
// - Small vs large arrays
// - With and without axis parameter
// - With different return options

use numrs2::error::Result;
use numrs2::prelude::*;
use std::time::{Duration, Instant};

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
    println!("NumRS2 Unique Function Benchmarks");
    println!("================================");

    // Setup random number generator with fixed seed for reproducibility
    let rng = RandomState::with_seed(42);

    println!("\nPart 1: Small Array Benchmarks");
    println!("-----------------------------");

    // Generate small array with duplicates
    let uniform_values = rng.uniform(0.0f64, 10.0f64, &[1000])?;
    let int_values: Vec<i32> = uniform_values
        .to_vec()
        .iter()
        .map(|&x| (x * 10.0f64).floor() as i32)
        .collect();
    let small_array = Array::from_vec(int_values).reshape(&[1000]);

    // Basic unique function
    benchmark("unique - small array", 1000, || {
        unique(&small_array, None, None, None, None)
    });

    // With all return options
    benchmark("unique - small array with all returns", 1000, || {
        unique(&small_array, None, Some(true), Some(true), Some(true))
    });

    println!("\nPart 2: Large Array Benchmarks");
    println!("-----------------------------");

    // Generate large array with duplicates
    let uniform_values_large = rng.uniform(0.0f64, 100.0f64, &[100000])?;
    let int_values_large: Vec<i32> = uniform_values_large
        .to_vec()
        .iter()
        .map(|&x| (x * 10.0f64).floor() as i32)
        .collect();
    let large_array = Array::from_vec(int_values_large).reshape(&[100000]);

    // Basic unique function
    benchmark("unique - large array", 100, || {
        unique(&large_array, None, None, None, None)
    });

    // With all return options
    benchmark("unique - large array with all returns", 100, || {
        unique(&large_array, None, Some(true), Some(true), Some(true))
    });

    println!("\nPart 3: 2D Array Benchmarks");
    println!("--------------------------");

    // Generate 2D array
    let uniform_values_2d = rng.uniform(0.0f64, 10.0f64, &[1000, 5])?;
    let int_values_2d: Vec<i32> = uniform_values_2d
        .to_vec()
        .iter()
        .map(|&x| (x * 5.0f64).floor() as i32)
        .collect();
    let array_2d = Array::from_vec(int_values_2d).reshape(&[1000, 5]);

    // Without axis (flattened)
    benchmark("unique - 2D array without axis", 1000, || {
        unique(&array_2d, None, None, None, None)
    });

    // With axis=0
    benchmark("unique - 2D array with axis=0", 100, || {
        unique(&array_2d, Some(0), None, None, None)
    });

    // With axis=1
    benchmark("unique - 2D array with axis=1", 100, || {
        unique(&array_2d, Some(1), None, None, None)
    });

    println!("\nPart 4: String Array Benchmarks - Skipped");
    println!("--------------------------------------");
    println!("// Skipped: String doesn't implement Zero trait required by unique function");

    println!("\nPart 5: Edge Cases");
    println!("----------------");

    // Array with mostly unique elements
    let mostly_unique = Array::from_vec((0..10000).collect::<Vec<i32>>());

    benchmark("unique - mostly unique elements", 100, || {
        unique(&mostly_unique, None, None, None, None)
    });

    // Array with all identical elements
    let all_same = Array::from_vec(vec![7; 10000]);

    benchmark("unique - all identical elements", 100, || {
        unique(&all_same, None, None, None, None)
    });

    println!("\nBenchmark complete!");

    Ok(())
}
