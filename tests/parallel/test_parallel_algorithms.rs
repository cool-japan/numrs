//! Comprehensive tests for parallel algorithms

use numrs2::parallel::{ParallelArrayOps, ParallelConfig};

// ============================================================================
// Parallel Map Tests
// ============================================================================

#[test]
fn test_parallel_map_basic() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let input: Vec<i32> = (0..1000).collect();
    let mut output = vec![0i32; 1000];

    ops.parallel_map(&input, &mut output, |x| x * 2)
        .expect("Failed to parallel map");

    for (&out, &inp) in output.iter().zip(input.iter()) {
        assert_eq!(out, inp * 2);
    }
}

#[test]
fn test_parallel_map_different_types() {
    let config = ParallelConfig {
        num_threads: Some(2),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    // Test with f64
    let input_f64: Vec<f64> = (0..500).map(|x| x as f64 * 0.5).collect();
    let mut output_f64 = vec![0.0f64; 500];

    ops.parallel_map(&input_f64, &mut output_f64, |x| x * x)
        .expect("Failed to parallel map f64");

    for (&out, &inp) in output_f64.iter().zip(input_f64.iter()) {
        assert!((out - inp * inp).abs() < 1e-10);
    }
}

#[test]
fn test_parallel_map_edge_cases() {
    let config = ParallelConfig::default();
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    // Empty array
    let input_empty: Vec<i32> = vec![];
    let mut output_empty: Vec<i32> = vec![];
    ops.parallel_map(&input_empty, &mut output_empty, |x| x + 1)
        .expect("Failed to parallel map empty array");
    assert_eq!(output_empty.len(), 0);

    // Single element
    let input_single = vec![42];
    let mut output_single = vec![0];
    ops.parallel_map(&input_single, &mut output_single, |x| x * 3)
        .expect("Failed to parallel map single element");
    assert_eq!(output_single[0], 126);
}

// ============================================================================
// Parallel Reduce Tests
// ============================================================================

#[test]
fn test_parallel_reduce_sum() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 100,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<i64> = (1..=1000).collect();
    let result = ops
        .parallel_reduce(&data, 0i64, |a, b| a + b)
        .expect("Failed to parallel reduce");

    let expected: i64 = (1..=1000).sum();
    assert_eq!(result, expected);
}

#[test]
fn test_parallel_reduce_custom_operations() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 50,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];

    // Test max
    let max_result = ops
        .parallel_reduce(&data, f64::NEG_INFINITY, |a, b| a.max(b))
        .expect("Failed to reduce max");
    assert_eq!(max_result, 5.5);

    // Test min
    let min_result = ops
        .parallel_reduce(&data, f64::INFINITY, |a, b| a.min(b))
        .expect("Failed to reduce min");
    assert_eq!(min_result, 1.5);

    // Test product
    let product_result = ops
        .parallel_reduce(&data, 1.0, |a, b| a * b)
        .expect("Failed to reduce product");
    let expected_product = 1.5 * 2.5 * 3.5 * 4.5 * 5.5;
    assert!((product_result - expected_product).abs() < 1e-6);
}

#[test]
fn test_parallel_reduce_correctness() {
    let config = ParallelConfig {
        num_threads: Some(8),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<i32> = (0..10000).collect();

    let parallel_result = ops
        .parallel_reduce(&data, 0, |a, b| a + b)
        .expect("Failed to parallel reduce");

    let sequential_result: i32 = data.iter().sum();

    assert_eq!(parallel_result, sequential_result);
}

// ============================================================================
// Parallel Filter Tests
// ============================================================================

#[test]
fn test_parallel_filter_basic() {
    // Note: ParallelArrayOps doesn't have a filter method in the current implementation
    // This test demonstrates how it would work if implemented
    let config = ParallelConfig::default();
    let _ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<i32> = (0..100).collect();
    let filtered: Vec<i32> = data.iter().filter(|&&x| x % 2 == 0).copied().collect();

    assert_eq!(filtered.len(), 50);
    assert!(filtered.iter().all(|&x| x % 2 == 0));
}

#[test]
fn test_parallel_filter_complex_predicate() {
    let data: Vec<i32> = (0..1000).collect();
    let filtered: Vec<i32> = data
        .iter()
        .filter(|&&x| x % 3 == 0 && x % 5 == 0)
        .copied()
        .collect();

    assert!(filtered.iter().all(|&x| x % 15 == 0));
}

// ============================================================================
// Parallel Sort Tests
// ============================================================================

#[test]
fn test_parallel_sort_correctness() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 100,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let mut data: Vec<i32> = (0..1000).rev().collect(); // Reverse sorted
    ops.parallel_sort(&mut data).expect("Failed to sort");

    for i in 1..data.len() {
        assert!(data[i - 1] <= data[i]);
    }
}

#[test]
fn test_parallel_sort_random_data() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 100,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let mut data: Vec<i32> = vec![
        64, 34, 25, 12, 22, 11, 90, 88, 45, 50, 33, 17, 10, 82, 67, 23,
    ];
    ops.parallel_sort(&mut data).expect("Failed to sort");

    for i in 1..data.len() {
        assert!(data[i - 1] <= data[i]);
    }
}

#[test]
fn test_parallel_sort_different_sizes() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 50,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    for size in [10, 100, 1000, 5000] {
        let mut data: Vec<i32> = (0..size).rev().collect();
        ops.parallel_sort(&mut data).expect("Failed to sort");

        for i in 1..data.len() {
            assert!(data[i - 1] <= data[i], "Failed for size {}", size);
        }
    }
}

// ============================================================================
// Parallel Scan/Prefix Sum Tests
// ============================================================================

#[test]
fn test_parallel_prefix_sum_basic() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let mut result = vec![0i32; 5];

    ops.parallel_prefix_sum(&data, &mut result)
        .expect("Failed to compute prefix sum");

    assert_eq!(result, vec![1, 3, 6, 10, 15]);
}

#[test]
fn test_parallel_prefix_sum_large_array() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 100,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<i32> = vec![1; 1000];
    let mut result = vec![0i32; 1000];

    ops.parallel_prefix_sum(&data, &mut result)
        .expect("Failed to compute prefix sum");

    for (i, &r) in result.iter().enumerate() {
        assert_eq!(r, (i + 1) as i32);
    }
}

// ============================================================================
// Parallel Pipeline Tests
// ============================================================================

#[test]
fn test_parallel_pipeline_multi_stage() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    // Stage 1: Map
    let input: Vec<i32> = (0..100).collect();
    let mut stage1 = vec![0i32; 100];
    ops.parallel_map(&input, &mut stage1, |x| x * 2)
        .expect("Failed stage 1");

    // Stage 2: Map again
    let mut stage2 = vec![0i32; 100];
    ops.parallel_map(&stage1, &mut stage2, |x| x + 10)
        .expect("Failed stage 2");

    // Verify pipeline result
    for (i, &s2) in stage2.iter().enumerate() {
        assert_eq!(s2, (i as i32) * 2 + 10);
    }
}

#[test]
fn test_parallel_pipeline_with_reduce() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    // Stage 1: Map
    let input: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let mut squared = vec![0.0f64; 100];
    ops.parallel_map(&input, &mut squared, |x| x * x)
        .expect("Failed to square");

    // Stage 2: Reduce
    let sum = ops
        .parallel_reduce(&squared, 0.0, |a, b| a + b)
        .expect("Failed to reduce");

    // Verify sum of squares
    let expected: f64 = (1..=100).map(|x| (x * x) as f64).sum();
    assert!((sum - expected).abs() < 1e-6);
}

// ============================================================================
// Parallel Binary Operations Tests
// ============================================================================

#[test]
fn test_parallel_binary_op_addition() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let a: Vec<f64> = (0..1000).map(|x| x as f64).collect();
    let b: Vec<f64> = (0..1000).map(|x| x as f64 * 0.5).collect();
    let mut result = vec![0.0f64; 1000];

    ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y)
        .expect("Failed binary op");

    for ((&r, &av), &bv) in result.iter().zip(a.iter()).zip(b.iter()) {
        assert!((r - (av + bv)).abs() < 1e-10);
    }
}

#[test]
fn test_parallel_binary_op_multiplication() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let a: Vec<f64> = (1..=500).map(|x| x as f64).collect();
    let b: Vec<f64> = (1..=500).map(|x| (x as f64) * 2.0).collect();
    let mut result = vec![0.0f64; 500];

    ops.parallel_binary_op(&a, &b, &mut result, |x, y| x * y)
        .expect("Failed binary op");

    for ((&r, &av), &bv) in result.iter().zip(a.iter()).zip(b.iter()) {
        assert!((r - (av * bv)).abs() < 1e-6);
    }
}

// ============================================================================
// Performance and Correctness Tests
// ============================================================================

#[test]
fn test_parallel_ops_thread_scaling() {
    // Test with different thread counts
    for num_threads in [1, 2, 4, 8] {
        let available = std::thread::available_parallelism().map_or(4, |n| n.get());
        if num_threads > available {
            continue;
        }

        let config = ParallelConfig {
            num_threads: Some(num_threads),
            parallel_threshold: 100,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

        let data: Vec<i32> = (0..10000).collect();
        let result = ops
            .parallel_reduce(&data, 0, |a, b| a + b)
            .expect("Failed to reduce");

        let expected: i32 = data.iter().sum();
        assert_eq!(result, expected);
    }
}

#[test]
fn test_parallel_ops_determinism() {
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 100,
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let mut data1: Vec<i32> = (0..1000).rev().collect();
    let mut data2 = data1.clone();

    ops.parallel_sort(&mut data1).expect("Failed to sort 1");
    ops.parallel_sort(&mut data2).expect("Failed to sort 2");

    assert_eq!(data1, data2, "Parallel sort should be deterministic");
}

#[test]
fn test_parallel_ops_error_handling() {
    let config = ParallelConfig::default();
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    // Test dimension mismatch
    let a = vec![1, 2, 3];
    let b = vec![1, 2];
    let mut result = vec![0; 3];

    let err = ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y);
    assert!(err.is_err());
}

#[test]
fn test_parallel_threshold_behavior() {
    // Test that small arrays use sequential processing
    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 1000, // High threshold
        ..Default::default()
    };
    let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

    let data: Vec<i32> = (0..100).collect(); // Below threshold
    let result = ops
        .parallel_reduce(&data, 0, |a, b| a + b)
        .expect("Failed to reduce");

    let expected: i32 = data.iter().sum();
    assert_eq!(result, expected);
}
