//! Expression Templates Example
//!
//! This example demonstrates NumRS2's expression templates system for lazy evaluation
//! and optimization, including:
//!
//! 1. SharedArray - Reference-counted arrays with operator overloading
//! 2. SharedExpr - Lifetime-free expression templates
//! 3. CSE (Common Subexpression Elimination) - Automatic caching of repeated computations
//! 4. Memory access patterns - Cache-aware iteration strategies
//!
//! Run with: `cargo run --example expression_templates_example`

use numrs2::expr::{CachedExpr, ExprCache, SharedExpr, SharedExprBuilder};
use numrs2::memory_optimize::access_patterns::{
    cache_aware_binary_op, cache_aware_copy, cache_aware_transform, detect_layout, Block,
    BlockedIterator, OptimizationHints, StrideOptimizer, Tile2D, TiledIterator2D,
};
use numrs2::shared_array::SharedArray;

fn main() {
    println!("=== NumRS2 Expression Templates Example ===\n");

    // Part 1: SharedArray with Operator Overloading
    shared_array_example();

    // Part 2: SharedExpr Lazy Evaluation
    shared_expr_example();

    // Part 3: Common Subexpression Elimination (CSE)
    cse_example();

    // Part 4: Memory Access Pattern Optimization
    memory_patterns_example();

    println!("\n=== All examples completed successfully! ===");
}

/// Demonstrates SharedArray - reference-counted arrays with operator overloading
fn shared_array_example() {
    println!("--- Part 1: SharedArray with Operator Overloading ---\n");

    // Create shared arrays
    let a: SharedArray<f64> = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b: SharedArray<f64> = SharedArray::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

    println!("Array a: {:?}", a.to_vec());
    println!("Array b: {:?}", b.to_vec());

    // Cheap cloning (O(1) - just increments reference count)
    let a_clone = a.clone();
    println!("Reference count after clone: {}", a.ref_count());

    // Operator overloading - natural syntax!
    let sum = a.clone() + b.clone();
    let diff = a.clone() - b.clone();
    let product = a.clone() * b.clone();
    let quotient = b.clone() / a.clone();

    println!("\na + b = {:?}", sum.to_vec());
    println!("a - b = {:?}", diff.to_vec());
    println!("a * b = {:?}", product.to_vec());
    println!("b / a = {:?}", quotient.to_vec());

    // Scalar operations
    let scaled = a.clone() * 2.0;
    let shifted = a.clone() + 5.0;
    println!("\na * 2 = {:?}", scaled.to_vec());
    println!("a + 5 = {:?}", shifted.to_vec());

    // Chained operations - all computed efficiently
    let result = (a.clone() + b.clone()) * 2.0 - 5.0;
    println!("\n(a + b) * 2 - 5 = {:?}", result.to_vec());

    // Reference-based operations (avoid move)
    let ref_sum = &a_clone + &b;
    println!("&a + &b = {:?}", ref_sum.to_vec());

    println!();
}

/// Demonstrates SharedExpr - lifetime-free lazy evaluation
fn shared_expr_example() {
    println!("--- Part 2: SharedExpr Lazy Evaluation ---\n");

    // Create arrays
    let a: SharedArray<f64> = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b: SharedArray<f64> = SharedArray::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

    // Build expression using SharedExprBuilder
    // Expression: (a + b) * 2
    // First evaluate a + b, then scale by 2
    // Note: We need to evaluate add result first, then apply scalar multiplication
    let add_result = a.clone() + b.clone();
    let scaled_expr = SharedExprBuilder::from_shared_array(add_result).mul_scalar(2.0);

    println!("Expression built (mul_scalar is lazy)");
    println!(
        "Expression size: {} elements",
        SharedExpr::size(&scaled_expr.clone().into_expr())
    );

    // Evaluation happens here
    let result = scaled_expr.eval();
    println!("Evaluated result: {:?}", result.to_vec());
    // Expected: [22.0, 44.0, 66.0, 88.0]

    // Create unary expression (square each element)
    let c: SharedArray<f64> = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let expr_c = SharedExprBuilder::from_shared_array(c);
    let squared = expr_c.map(|x| x * x);
    let squared_result = squared.eval();
    println!("Squared: {:?}", squared_result.to_vec());
    // Expected: [1.0, 4.0, 9.0, 16.0]

    println!();
}

/// Demonstrates Common Subexpression Elimination (CSE)
fn cse_example() {
    println!("--- Part 3: Common Subexpression Elimination (CSE) ---\n");

    // Create arrays
    let a: SharedArray<f64> = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b: SharedArray<f64> = SharedArray::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

    // Create expression cache for CSE
    let cache: ExprCache<f64> = ExprCache::new();

    // Build expression: a + b
    let sum = a.clone() + b.clone();
    let sum_expr = SharedExprBuilder::from_shared_array(sum);

    // Wrap with caching - computation cached after first eval
    let cached_sum = CachedExpr::new(sum_expr.into_expr(), cache.clone());

    println!("First evaluation (computes and caches):");
    let result1 = cached_sum.eval();
    println!("Result: {:?}", result1.to_vec());

    println!("Second evaluation (uses cache):");
    let result2 = cached_sum.eval();
    println!("Result: {:?}", result2.to_vec());

    // Check cache hit
    println!("\nCache contains {} entries", cache.len());

    // Demonstrate cache invalidation
    cached_sum.invalidate();
    println!("After invalidation, cache contains {} entries", cache.len());

    println!();
}

/// Demonstrates memory access pattern optimization
fn memory_patterns_example() {
    println!("--- Part 4: Memory Access Pattern Optimization ---\n");

    // Create array dimensions
    let size = 1000;

    // Detect memory layout
    let layout = detect_layout(&[size], &[1]);
    println!("Memory layout for 1D contiguous array: {:?}", layout);

    // 2D C-contiguous layout
    let layout_2d = detect_layout(&[100, 100], &[100, 1]);
    println!("Memory layout for 2D C-contiguous: {:?}", layout_2d);

    // 2D F-contiguous layout
    let layout_f = detect_layout(&[100, 100], &[1, 100]);
    println!("Memory layout for 2D F-contiguous: {:?}", layout_f);

    // Get optimization hints
    let hints = OptimizationHints::default_for::<f64>(size);
    println!("\nOptimization hints for {} elements:", size);
    println!("  Layout: {:?}", hints.layout);
    println!("  Access pattern: {:?}", hints.access_pattern);
    println!("  Block size: {}", hints.block_size);
    println!("  Use parallel: {}", hints.use_parallel);
    println!("  Cache efficiency: {:.2}", hints.cache_efficiency);

    // Analyze specific shape/strides
    let hints_analyzed = OptimizationHints::analyze::<f64>(&[100, 100], &[100, 1]);
    println!("\nAnalyzed hints for 100x100 array:");
    println!("  Layout: {:?}", hints_analyzed.layout);
    println!("  Tile size: {:?}", hints_analyzed.tile_size);

    // Blocked iteration for cache efficiency
    println!("\nBlocked iteration (block_size = 64):");
    let block_iter = BlockedIterator::new(size, 64);
    let blocks: Vec<Block> = block_iter.collect();
    println!("  Total blocks: {}", blocks.len());
    println!("  First block: {:?}", blocks.first());
    println!("  Last block: {:?}", blocks.last());

    // 2D tiled iteration for matrix operations
    println!("\nTiled 2D iteration (100x100 matrix, 16x16 tiles):");
    let rows = 100;
    let cols = 100;
    let tile_iter = TiledIterator2D::new(rows, cols, 16, 16);
    let tiles: Vec<Tile2D> = tile_iter.collect();
    println!("  Total tiles: {}", tiles.len());
    println!("  First tile: {:?}", tiles.first());

    // Stride optimizer
    println!("\nStride optimization:");
    let shape = [8usize, 8];
    let strides = [8usize, 1]; // Row-major strides
    let stride_opt = StrideOptimizer::new(&shape, &strides);
    println!(
        "  Optimal iteration order: {:?}",
        stride_opt.optimal_iteration_order()
    );
    println!("  Should copy: {}", stride_opt.should_copy());
    println!(
        "  Bandwidth efficiency: {:.2}",
        stride_opt.bandwidth_efficiency()
    );

    // Cache-aware operations
    println!("\nCache-aware operations:");

    let src: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; 1000];

    // Cache-aware copy
    cache_aware_copy(&src, &mut dst);
    println!("  cache_aware_copy: copied {} elements", dst.len());
    assert_eq!(dst[0], 0.0);
    assert_eq!(dst[999], 999.0);

    // Cache-aware transform
    let mut transformed = vec![0.0f64; 1000];
    cache_aware_transform(&src, &mut transformed, |x| x * 2.0);
    println!(
        "  cache_aware_transform: first 5 values = {:?}",
        &transformed[0..5]
    );

    // Cache-aware binary operation
    let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..1000).map(|i| (i * 2) as f64).collect();
    let mut result = vec![0.0f64; 1000];
    cache_aware_binary_op(&a, &b, &mut result, |x, y| x + y);
    println!(
        "  cache_aware_binary_op: first 5 values = {:?}",
        &result[0..5]
    );

    println!();
}
