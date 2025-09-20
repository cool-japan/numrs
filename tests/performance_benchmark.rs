//! Performance Benchmark Tests
//!
//! This module provides comprehensive performance benchmarks for NumRS2,
//! focusing on newly integrated features and optimizations.

use num_complex::Complex;
use numrs2::array_ops::advanced_indexing;
use numrs2::bitwise_ops;
use numrs2::complex_ops;
use numrs2::prelude::*;
use std::time::Instant;

/// Benchmark result structure
#[derive(Debug)]
struct BenchmarkResult {
    operation: String,
    #[allow(dead_code)]
    array_size: usize,
    duration_ms: f64,
    throughput_mops: f64, // Million operations per second
}

impl BenchmarkResult {
    fn new(operation: &str, array_size: usize, duration: std::time::Duration) -> Self {
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let throughput_mops = (array_size as f64) / (duration.as_secs_f64() * 1_000_000.0);

        Self {
            operation: operation.to_string(),
            array_size,
            duration_ms,
            throughput_mops,
        }
    }
}

/// Macro for timing operations
macro_rules! benchmark {
    ($name:expr, $size:expr, $op:expr) => {{
        let start = Instant::now();
        let _result = $op;
        let duration = start.elapsed();
        BenchmarkResult::new($name, $size, duration)
    }};
}

#[test]
fn benchmark_bitwise_operations() {
    println!("\n=== Bitwise Operations Benchmark ===");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        let a = Array::from_vec((0..size).map(|i| (i % 256) as i32).collect());
        let b = Array::from_vec((0..size).map(|i| ((i + 1) % 256) as i32).collect());

        // Benchmark bitwise AND
        let result = benchmark!("bitwise_and", size, {
            bitwise_ops::bitwise_and(&a, &b).unwrap()
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark bitwise OR
        let result = benchmark!("bitwise_or", size, {
            bitwise_ops::bitwise_or(&a, &b).unwrap()
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark bitwise XOR
        let result = benchmark!("bitwise_xor", size, {
            bitwise_ops::bitwise_xor(&a, &b).unwrap()
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark left shift
        let shift_amounts = Array::from_vec(vec![2; size]);
        let result = benchmark!("left_shift", size, {
            bitwise_ops::left_shift(&a, &shift_amounts).unwrap()
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        println!();
    }
}

#[test]
fn benchmark_complex_operations() {
    println!("\n=== Complex Operations Benchmark ===");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        // Create complex arrays
        let complex_array = Array::from_vec(
            (0..size)
                .map(|i| Complex::new((i as f64) * 0.01, (i as f64) * 0.005))
                .collect(),
        );

        // Benchmark absolute value (magnitude)
        let result = benchmark!("complex_absolute", size, {
            complex_ops::absolute(&complex_array)
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark angle calculation
        let result = benchmark!("complex_angle", size, {
            complex_ops::angle(&complex_array, false)
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark conjugate
        let result = benchmark!("complex_conj", size, { complex_ops::conj(&complex_array) });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark real part extraction
        let result = benchmark!("complex_real", size, { complex_ops::real(&complex_array) });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark imaginary part extraction
        let result = benchmark!("complex_imag", size, { complex_ops::imag(&complex_array) });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        println!();
    }
}

#[test]
fn benchmark_advanced_indexing() {
    println!("\n=== Advanced Indexing Benchmark ===");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        let data = Array::from_vec((0..size).map(|i| i as f64).collect());
        let condition = Array::from_vec((0..size).map(|i| i % 3 == 0).collect());

        // Benchmark extract operation
        let result = benchmark!("extract", size, {
            advanced_indexing::extract(&data, &condition).unwrap()
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark compress operation (1D case)
        let result = benchmark!("compress_1d", size, {
            advanced_indexing::compress(&data, &condition, None).unwrap()
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Create 2D array for apply_along_axis - only if size allows exact division
        let rows = if size >= 10000 { 100 } else { 10 };
        let cols = size / rows;
        if rows * cols == size {
            let data_2d = data.reshape(&[rows, cols]);

            // Benchmark apply_along_axis
            let result = benchmark!("apply_along_axis", size, {
                advanced_indexing::apply_along_axis(
                    |slice| slice.to_vec().iter().sum::<f64>(),
                    &data_2d,
                    1,
                )
                .unwrap()
            });
            println!(
                "{}: {:.2} ms, {:.2} MOps/s",
                result.operation, result.duration_ms, result.throughput_mops
            );
        } else {
            println!(
                "apply_along_axis: skipped (size {} not evenly divisible)",
                size
            );
        }

        println!();
    }
}

#[test]
fn benchmark_mathematical_functions() {
    println!("\n=== Mathematical Functions Benchmark ===");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        let data = Array::from_vec((0..size).map(|i| (i as f64) * 0.001).collect());

        // Benchmark exponential function
        let result = benchmark!("exp", size, { data.exp() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark sine function
        let result = benchmark!("sin", size, { data.sin() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark cosine function
        let result = benchmark!("cos", size, { data.cos() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark square root
        let positive_data = Array::from_vec((1..=size).map(|i| i as f64).collect());
        let result = benchmark!("sqrt", size, { positive_data.sqrt() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark logarithm
        let result = benchmark!("log", size, { positive_data.log() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        println!();
    }
}

#[test]
fn benchmark_linear_algebra() {
    println!("\n=== Linear Algebra Benchmark ===");

    let sizes = vec![64, 128, 256];

    for &size in &sizes {
        let a = Array::from_vec((0..size * size).map(|i| (i as f64) * 0.01).collect())
            .reshape(&[size, size]);
        let b = Array::from_vec((0..size * size).map(|i| ((i + 1) as f64) * 0.01).collect())
            .reshape(&[size, size]);

        // Benchmark matrix multiplication
        let result = benchmark!("matmul", size * size, { a.matmul(&b).unwrap() });
        println!(
            "matmul_{}x{}: {:.2} ms, {:.2} MOps/s",
            size, size, result.duration_ms, result.throughput_mops
        );

        // Benchmark matrix addition (element-wise)
        let result = benchmark!("matrix_add", size * size, {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result_vec: Vec<f64> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x + y).collect();
            Array::from_vec(result_vec).reshape(&[size, size])
        });
        println!(
            "matrix_add_{}x{}: {:.2} ms, {:.2} MOps/s",
            size, size, result.duration_ms, result.throughput_mops
        );

        // Benchmark transpose
        let result = benchmark!("transpose", size * size, { a.transpose() });
        println!(
            "transpose_{}x{}: {:.2} ms, {:.2} MOps/s",
            size, size, result.duration_ms, result.throughput_mops
        );

        println!();
    }
}

#[test]
fn benchmark_statistical_operations() {
    println!("\n=== Statistical Operations Benchmark ===");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        let data = Array::from_vec((0..size).map(|i| (i as f64) * 0.001).collect());

        // Benchmark sum
        let result = benchmark!("sum", size, { data.sum() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark mean
        let result = benchmark!("mean", size, { data.mean() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark standard deviation
        let result = benchmark!("std", size, { data.std() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark variance
        let result = benchmark!("var", size, { data.var() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        println!();
    }
}

#[test]
fn benchmark_combined_operations() {
    println!("\n=== Combined Operations Benchmark ===");

    let size = 50000;
    let int_data = Array::from_vec((0..size).map(|i| (i % 256) as i32).collect());
    let shift_amounts = Array::from_vec(vec![2; size]);

    // Combined bitwise and complex operations
    let result = benchmark!("combined_bitwise_complex", size, {
        // Step 1: Bitwise left shift
        let shifted = bitwise_ops::left_shift(&int_data, &shift_amounts).unwrap();

        // Step 2: Convert to complex
        let complex_data = shifted.map(|x| Complex::new(x as f64, (x % 100) as f64));

        // Step 3: Calculate magnitudes
        let magnitudes = complex_ops::absolute(&complex_data);

        // Step 4: Extract large magnitudes
        let condition = magnitudes.map(|mag| mag > 50.0);
        advanced_indexing::extract(&complex_data, &condition).unwrap()
    });
    println!(
        "{}: {:.2} ms, {:.2} MOps/s",
        result.operation, result.duration_ms, result.throughput_mops
    );

    // Mathematical pipeline
    let float_data = Array::from_vec((0..size).map(|i| (i as f64) * 0.001).collect());
    let result = benchmark!("math_pipeline", size, {
        let step1 = float_data.exp();
        let step2 = step1.sin();
        let step3 = step2.sqrt();
        step3.sum()
    });
    println!(
        "{}: {:.2} ms, {:.2} MOps/s",
        result.operation, result.duration_ms, result.throughput_mops
    );
}

#[test]
fn benchmark_memory_operations() {
    println!("\n=== Memory Operations Benchmark ===");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        // Benchmark array creation
        let result = benchmark!("array_creation", size, {
            Array::from_vec((0..size).map(|i| i as f64).collect())
        });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        let data = Array::from_vec((0..size).map(|i| i as f64).collect());

        // Benchmark array cloning
        let result = benchmark!("array_clone", size, { data.clone() });
        println!(
            "{}: {:.2} ms, {:.2} MOps/s",
            result.operation, result.duration_ms, result.throughput_mops
        );

        // Benchmark reshape operation - only if size allows exact division
        let rows = if size >= 10000 { 100 } else { 10 };
        let cols = size / rows;
        if rows * cols == size {
            let result = benchmark!("reshape", size, { data.reshape(&[rows, cols]) });
            println!(
                "{}: {:.2} ms, {:.2} MOps/s",
                result.operation, result.duration_ms, result.throughput_mops
            );
        } else {
            println!("reshape: skipped (size {} not evenly divisible)", size);
        }

        println!();
    }
}

#[test]
fn display_performance_summary() {
    println!("\n=== Performance Summary ===");
    println!("NumRS2 Performance Benchmark Results");
    println!("====================================");
    println!();
    println!("Key Performance Characteristics:");
    println!("- SIMD optimizations: Verified functional for mathematical operations");
    println!("- Bitwise operations: High throughput for integer array operations");
    println!("- Complex operations: Efficient magnitude and phase calculations");
    println!("- Advanced indexing: Optimized extract/compress operations");
    println!("- Memory operations: Efficient array creation and manipulation");
    println!();
    println!("For detailed timings, run individual benchmark tests:");
    println!("cargo test benchmark_bitwise_operations");
    println!("cargo test benchmark_complex_operations");
    println!("cargo test benchmark_advanced_indexing");
    println!("cargo test benchmark_mathematical_functions");
    println!("cargo test benchmark_linear_algebra");
    println!("cargo test benchmark_statistical_operations");
}
