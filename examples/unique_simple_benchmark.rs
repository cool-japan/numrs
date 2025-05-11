// Simple benchmark for the unique function
//
// This example focuses on integer arrays which satisfy all trait bounds

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
    let avg_ms = elapsed.as_millis() as f64 / runs as f64;

    println!("{}: avg {:.2} ms per run ({} runs)", name, avg_ms, runs);
    elapsed
}

fn main() -> Result<()> {
    println!("NumRS2 Unique Function Benchmark");
    println!("===============================");

    // Small array test - few duplicates
    println!("\nSmall array test (1,000 elements):");
    let small_array = Array::from_vec((0..1000).map(|i| i % 800).collect::<Vec<i32>>());

    benchmark("unique - small array", 100, || {
        unique(&small_array, None, None, None, None)
    });

    // Medium array test
    println!("\nMedium array test (10,000 elements):");
    let medium_array = Array::from_vec((0..10000).map(|i| i % 1000).collect::<Vec<i32>>());

    benchmark("unique - medium array", 20, || {
        unique(&medium_array, None, None, None, None)
    });

    // Large array test
    println!("\nLarge array test (100,000 elements):");
    let large_array = Array::from_vec((0..100000).map(|i| i % 5000).collect::<Vec<i32>>());

    benchmark("unique - large array", 5, || {
        unique(&large_array, None, None, None, None)
    });

    // 2D array test
    println!("\n2D array test (1000x10 elements):");
    let mut array_2d_data = Vec::with_capacity(10000);
    for i in 0..1000 {
        for j in 0..10 {
            array_2d_data.push((i + j) % 500);
        }
    }
    let array_2d = Array::from_vec(array_2d_data).reshape(&[1000, 10]);

    // Test with axis parameter
    benchmark("unique - 2D array with axis=0", 10, || {
        unique(&array_2d, Some(0), None, None, None)
    });

    benchmark("unique - 2D array with axis=1", 10, || {
        unique(&array_2d, Some(1), None, None, None)
    });

    // Test with all return options
    println!("\nTesting all return options:");
    benchmark("unique - all returns", 10, || {
        unique(&medium_array, None, Some(true), Some(true), Some(true))
    });

    // Edge cases
    println!("\nEdge cases:");

    // Array with mostly unique elements
    let mostly_unique = Array::from_vec((0..10000).collect::<Vec<i32>>());
    benchmark("unique - mostly unique elements", 10, || {
        unique(&mostly_unique, None, None, None, None)
    });

    // Array with all identical elements
    let all_same = Array::from_vec(vec![7; 10000]);
    benchmark("unique - all identical elements", 10, || {
        unique(&all_same, None, None, None, None)
    });

    println!("\nBenchmark complete!");

    Ok(())
}
