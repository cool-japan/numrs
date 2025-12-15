//! Memory Usage Profiling Tests
//!
//! This module analyzes memory usage patterns in NumRS2 operations
//! to identify potential optimizations and ensure efficient memory management.

use numrs2::array_ops::advanced_indexing;
use numrs2::bitwise_ops;
use numrs2::complex_ops;
use numrs2::prelude::*;
use scirs2_core::Complex;
use std::time::Instant;

/// Simple memory usage tracker
struct MemoryTracker {
    start_time: Instant,
    operation_name: String,
}

impl MemoryTracker {
    fn new(operation: &str) -> Self {
        Self {
            start_time: Instant::now(),
            operation_name: operation.to_string(),
        }
    }

    fn finish(self, array_size: usize) -> MemoryProfile {
        let duration = self.start_time.elapsed();
        MemoryProfile {
            operation: self.operation_name,
            array_size,
            duration_ms: duration.as_secs_f64() * 1000.0,
            estimated_memory_mb: (array_size * std::mem::size_of::<f64>()) as f64
                / (1024.0 * 1024.0),
        }
    }
}

#[derive(Debug)]
struct MemoryProfile {
    #[allow(dead_code)]
    operation: String,
    #[allow(dead_code)]
    array_size: usize,
    duration_ms: f64,
    estimated_memory_mb: f64,
}

#[test]
fn test_memory_usage_array_creation() {
    println!("\n=== Array Creation Memory Analysis ===");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for &size in &sizes {
        // Test different array creation methods

        // zeros() creation
        let tracker = MemoryTracker::new("zeros_creation");
        let zeros_array: Array<f64> = Array::zeros(&[size]);
        let profile = tracker.finish(size);
        println!(
            "zeros({}): {:.2} ms, {:.2} MB estimated",
            size, profile.duration_ms, profile.estimated_memory_mb
        );

        // ones() creation
        let tracker = MemoryTracker::new("ones_creation");
        let ones_array: Array<f64> = Array::ones(&[size]);
        let profile = tracker.finish(size);
        println!(
            "ones({}): {:.2} ms, {:.2} MB estimated",
            size, profile.duration_ms, profile.estimated_memory_mb
        );

        // from_vec() creation
        let tracker = MemoryTracker::new("from_vec_creation");
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let from_vec_array = Array::from_vec(data);
        let profile = tracker.finish(size);
        println!(
            "from_vec({}): {:.2} ms, {:.2} MB estimated",
            size, profile.duration_ms, profile.estimated_memory_mb
        );

        // Verify arrays are valid
        assert_eq!(zeros_array.len(), size);
        assert_eq!(ones_array.len(), size);
        assert_eq!(from_vec_array.len(), size);

        println!();
    }
}

#[test]
fn test_memory_usage_mathematical_operations() {
    println!("\n=== Mathematical Operations Memory Analysis ===");

    let sizes = vec![10000, 100000, 500000];

    for &size in &sizes {
        let data = Array::from_vec((0..size).map(|i| (i as f64) * 0.001).collect());
        let memory_per_array_mb = (size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        println!(
            "Array size: {} elements ({:.2} MB)",
            size, memory_per_array_mb
        );

        // Test exp operation
        let tracker = MemoryTracker::new("exp");
        let exp_result = data.exp();
        let profile = tracker.finish(size);
        println!(
            "  exp: {:.2} ms, creates {:.2} MB",
            profile.duration_ms, memory_per_array_mb
        );

        // Test sin operation
        let tracker = MemoryTracker::new("sin");
        let sin_result = data.sin();
        let profile = tracker.finish(size);
        println!(
            "  sin: {:.2} ms, creates {:.2} MB",
            profile.duration_ms, memory_per_array_mb
        );

        // Test chained operations (memory efficiency)
        let tracker = MemoryTracker::new("chained_ops");
        let chained_result = data.exp().sin().exp();
        let profile = tracker.finish(size);
        println!(
            "  chained (exp->sin->exp): {:.2} ms, creates ~{:.2} MB total",
            profile.duration_ms,
            memory_per_array_mb * 3.0
        );

        // Verify results are valid
        assert_eq!(exp_result.len(), size);
        assert_eq!(sin_result.len(), size);
        assert_eq!(chained_result.len(), size);

        println!();
    }
}

#[test]
fn test_memory_usage_complex_operations() {
    println!("\n=== Complex Operations Memory Analysis ===");

    let sizes = vec![5000, 25000, 100000];

    for &size in &sizes {
        let complex_data = Array::from_vec(
            (0..size)
                .map(|i| Complex::new((i as f64) * 0.01, (i as f64) * 0.005))
                .collect(),
        );

        let memory_per_complex_mb =
            (size * std::mem::size_of::<Complex<f64>>()) as f64 / (1024.0 * 1024.0);
        let memory_per_real_mb = (size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        println!(
            "Complex array size: {} elements ({:.2} MB)",
            size, memory_per_complex_mb
        );

        // Test absolute value extraction
        let tracker = MemoryTracker::new("complex_absolute");
        let abs_result = complex_ops::absolute(&complex_data);
        let profile = tracker.finish(size);
        println!(
            "  absolute: {:.2} ms, creates {:.2} MB (f64 array)",
            profile.duration_ms, memory_per_real_mb
        );

        // Test real part extraction
        let tracker = MemoryTracker::new("complex_real");
        let real_result = complex_ops::real(&complex_data);
        let profile = tracker.finish(size);
        println!(
            "  real: {:.2} ms, creates {:.2} MB (f64 array)",
            profile.duration_ms, memory_per_real_mb
        );

        // Test angle calculation
        let tracker = MemoryTracker::new("complex_angle");
        let angle_result = complex_ops::angle(&complex_data, false);
        let profile = tracker.finish(size);
        println!(
            "  angle: {:.2} ms, creates {:.2} MB (f64 array)",
            profile.duration_ms, memory_per_real_mb
        );

        // Verify results
        assert_eq!(abs_result.len(), size);
        assert_eq!(real_result.len(), size);
        assert_eq!(angle_result.len(), size);

        println!();
    }
}

#[test]
fn test_memory_usage_bitwise_operations() {
    println!("\n=== Bitwise Operations Memory Analysis ===");

    let sizes = vec![10000, 50000, 200000];

    for &size in &sizes {
        let int_data_a = Array::from_vec((0..size).map(|i| (i % 256) as i32).collect());
        let int_data_b = Array::from_vec((0..size).map(|i| ((i + 1) % 256) as i32).collect());

        let memory_per_array_mb = (size * std::mem::size_of::<i32>()) as f64 / (1024.0 * 1024.0);

        println!(
            "Integer array size: {} elements ({:.2} MB each)",
            size, memory_per_array_mb
        );

        // Test bitwise AND
        let tracker = MemoryTracker::new("bitwise_and");
        let and_result = bitwise_ops::bitwise_and(&int_data_a, &int_data_b).unwrap();
        let profile = tracker.finish(size);
        println!(
            "  bitwise_and: {:.2} ms, creates {:.2} MB",
            profile.duration_ms, memory_per_array_mb
        );

        // Test left shift
        let shift_amounts = Array::from_vec(vec![2; size]);
        let tracker = MemoryTracker::new("left_shift");
        let shift_result = bitwise_ops::left_shift(&int_data_a, &shift_amounts).unwrap();
        let profile = tracker.finish(size);
        println!(
            "  left_shift: {:.2} ms, creates {:.2} MB",
            profile.duration_ms, memory_per_array_mb
        );

        // Verify results
        assert_eq!(and_result.len(), size);
        assert_eq!(shift_result.len(), size);

        println!();
    }
}

#[test]
fn test_memory_usage_advanced_indexing() {
    println!("\n=== Advanced Indexing Memory Analysis ===");

    let sizes = vec![10000, 50000, 200000];

    for &size in &sizes {
        let data = Array::from_vec((0..size).map(|i| i as f64).collect());
        let condition = Array::from_vec((0..size).map(|i| i % 3 == 0).collect());

        let memory_per_array_mb = (size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        println!(
            "Array size: {} elements ({:.2} MB)",
            size, memory_per_array_mb
        );

        // Test extract operation
        let tracker = MemoryTracker::new("extract");
        let extracted = advanced_indexing::extract(&data, &condition).unwrap();
        let profile = tracker.finish(extracted.len());
        println!(
            "  extract: {:.2} ms, creates {:.2} MB (~1/3 of original)",
            profile.duration_ms,
            extracted.len() as f64 * std::mem::size_of::<f64>() as f64 / (1024.0 * 1024.0)
        );

        // Test compress operation (1D)
        let tracker = MemoryTracker::new("compress");
        let compressed = advanced_indexing::compress(&data, &condition, None).unwrap();
        let profile = tracker.finish(compressed.len());
        println!(
            "  compress: {:.2} ms, creates {:.2} MB (~1/3 of original)",
            profile.duration_ms,
            compressed.len() as f64 * std::mem::size_of::<f64>() as f64 / (1024.0 * 1024.0)
        );

        // Verify results
        assert_eq!(extracted.len(), compressed.len()); // Should be equal for 1D case

        println!();
    }
}

#[test]
fn test_memory_usage_matrix_operations() {
    println!("\n=== Matrix Operations Memory Analysis ===");

    let sizes = vec![64, 128, 256];

    for &size in &sizes {
        let matrix_a = Array::from_vec((0..size * size).map(|i| (i as f64) * 0.01).collect())
            .reshape(&[size, size]);
        let matrix_b = Array::from_vec((0..size * size).map(|i| ((i + 1) as f64) * 0.01).collect())
            .reshape(&[size, size]);

        let memory_per_matrix_mb =
            (size * size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        println!(
            "Matrix size: {}x{} ({:.2} MB each)",
            size, size, memory_per_matrix_mb
        );

        // Test matrix multiplication
        let tracker = MemoryTracker::new("matmul");
        let matmul_result = matrix_a.matmul(&matrix_b).unwrap();
        let profile = tracker.finish(size * size);
        println!(
            "  matmul: {:.2} ms, creates {:.2} MB",
            profile.duration_ms, memory_per_matrix_mb
        );

        // Test transpose
        let tracker = MemoryTracker::new("transpose");
        let transpose_result = matrix_a.transpose();
        let profile = tracker.finish(size * size);
        println!(
            "  transpose: {:.2} ms, creates {:.2} MB",
            profile.duration_ms, memory_per_matrix_mb
        );

        // Test reshape (should be nearly free)
        let tracker = MemoryTracker::new("reshape");
        let reshaped = matrix_a.reshape(&[size * size]);
        let profile = tracker.finish(size * size);
        println!(
            "  reshape: {:.2} ms, minimal memory overhead",
            profile.duration_ms
        );

        // Verify results
        assert_eq!(matmul_result.shape(), &[size, size]);
        assert_eq!(transpose_result.shape(), &[size, size]);
        assert_eq!(reshaped.shape(), &[size * size]);

        println!();
    }
}

#[test]
fn test_memory_efficiency_recommendations() {
    println!("\n=== Memory Efficiency Recommendations ===");

    let test_size = 100000;
    let data = Array::from_vec((0..test_size).map(|i| (i as f64) * 0.001).collect());

    // Compare in-place vs copy operations where possible
    println!("Array size: {} elements", test_size);

    // Method 1: Multiple separate operations (creates multiple temporary arrays)
    let start = Instant::now();
    let step1 = data.exp();
    let step2 = step1.sin();
    let step3 = step2.exp(); // Changed from sqrt to exp to avoid NaN
    let multi_op_time = start.elapsed().as_secs_f64() * 1000.0;

    // Method 2: Chained operations (also creates temporaries but might be optimized)
    let start = Instant::now();
    let chained = data.exp().sin().exp(); // Changed from sqrt to exp to avoid NaN
    let chained_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Performance comparison:");
    println!("  Multiple operations: {:.2} ms", multi_op_time);
    println!("  Chained operations: {:.2} ms", chained_time);

    // Verify results are equivalent
    let step3_vec = step3.to_vec();
    let chained_vec = chained.to_vec();
    for (a, b) in step3_vec.iter().zip(chained_vec.iter()) {
        // Handle special cases: both inf, both -inf, or both NaN should be considered equal
        let equal = (a.is_nan() && b.is_nan())
            || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || (a - b).abs() < 1e-6;
        assert!(equal, "Values differ: {} vs {}", a, b);
    }

    println!("\nMemory optimization recommendations:");
    println!("1. Use chained operations where possible for potential optimization");
    println!("2. Consider in-place operations for large arrays when data can be modified");
    println!("3. Be aware that each operation creates a new array (current design)");
    println!("4. For very large datasets, consider processing in chunks to reduce peak memory");
    println!("5. Complex operations automatically handle memory layout efficiently");
    println!("6. SIMD operations may require aligned memory but handle this automatically");
}

#[test]
fn test_memory_patterns_large_arrays() {
    println!("\n=== Large Array Memory Patterns ===");

    // Test with increasingly large arrays to identify memory scaling
    let base_size = 10000;
    let multipliers = vec![1, 5, 10, 50];

    for &mult in &multipliers {
        let size = base_size * mult;
        let memory_mb = (size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        if memory_mb > 100.0 {
            println!(
                "Skipping size {} ({:.1} MB) to avoid excessive memory usage",
                size, memory_mb
            );
            continue;
        }

        println!(
            "Testing array size: {} elements ({:.2} MB)",
            size, memory_mb
        );

        let start = Instant::now();
        let large_array = Array::from_vec((0..size).map(|i| (i as f64) * 0.001).collect());
        let creation_time = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let _processed = large_array.exp();
        let processing_time = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "  Creation: {:.2} ms ({:.2} MB/s)",
            creation_time,
            memory_mb / (creation_time / 1000.0)
        );
        println!(
            "  Processing (exp): {:.2} ms ({:.2} MB/s)",
            processing_time,
            memory_mb / (processing_time / 1000.0)
        );

        // Verify scaling characteristics
        let mb_per_ms_creation = memory_mb / creation_time;
        let mb_per_ms_processing = memory_mb / processing_time;

        if mult > 1 {
            println!(
                "  Scaling efficiency: creation {:.1} MB/ms, processing {:.1} MB/ms",
                mb_per_ms_creation, mb_per_ms_processing
            );
        }

        println!();
    }

    println!("Memory pattern analysis complete.");
    println!("Good scaling indicates linear memory access patterns and efficient allocation.");
}
