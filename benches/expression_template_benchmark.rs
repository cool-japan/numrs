//! Expression Template Performance Benchmarks
//!
//! Comprehensive benchmarks for expression template system enhancements:
//! - SIMD-optimized evaluation
//! - Operation fusion
//! - Buffer reuse
//! - Allocation reduction

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::expr::{
    ArrayExpr, BinaryExpr, BinaryOpType, BufferPool, Expr, FusedBinaryScalarExpr, ScalarExpr,
    SimdBinaryExpr, SimdExprEval, UnaryExpr,
};
use std::hint::black_box;

/// Benchmark basic expression evaluation
fn bench_basic_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_expression");

    for size in [1000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64) * 2.0).collect();

        let arr_a = Array::from_vec(a);
        let arr_b = Array::from_vec(b);

        group.bench_with_input(BenchmarkId::new("simple_add", size), size, |bencher, _| {
            bencher.iter(|| {
                let expr = BinaryExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    |x: f64, y: f64| x + y,
                )
                .expect("Expression creation should succeed");
                black_box(expr.eval());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd_add", size), size, |bencher, _| {
            bencher.iter(|| {
                let expr = SimdBinaryExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    BinaryOpType::Add,
                )
                .expect("Expression creation should succeed");
                black_box(expr.eval());
            });
        });
    }

    group.finish();
}

/// Benchmark complex expression chains
fn bench_complex_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_expression");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a: Vec<f64> = (0..*size).map(|i| i as f64 + 1.0).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64) * 2.0 + 1.0).collect();
        let c: Vec<f64> = (0..*size).map(|i| (i as f64) * 3.0 + 1.0).collect();
        let d: Vec<f64> = (0..*size).map(|i| (i as f64) * 4.0 + 1.0).collect();

        let arr_a = Array::from_vec(a);
        let arr_b = Array::from_vec(b);
        let arr_c = Array::from_vec(c);
        let arr_d = Array::from_vec(d);

        // Expression: (a + b) * (c - d)
        group.bench_with_input(
            BenchmarkId::new("four_array_expr", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let sum = BinaryExpr::new(
                        ArrayExpr::new(&arr_a),
                        ArrayExpr::new(&arr_b),
                        |x: f64, y: f64| x + y,
                    )
                    .expect("Sum expression creation should succeed");

                    let diff = BinaryExpr::new(
                        ArrayExpr::new(&arr_c),
                        ArrayExpr::new(&arr_d),
                        |x: f64, y: f64| x - y,
                    )
                    .expect("Diff expression creation should succeed");

                    let product = BinaryExpr::new(sum, diff, |x: f64, y: f64| x * y)
                        .expect("Product expression creation should succeed");

                    black_box(product.eval());
                });
            },
        );

        // SIMD version
        group.bench_with_input(
            BenchmarkId::new("four_array_expr_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let sum = SimdBinaryExpr::new(
                        ArrayExpr::new(&arr_a),
                        ArrayExpr::new(&arr_b),
                        BinaryOpType::Add,
                    )
                    .expect("Sum expression creation should succeed");

                    let diff = SimdBinaryExpr::new(
                        ArrayExpr::new(&arr_c),
                        ArrayExpr::new(&arr_d),
                        BinaryOpType::Sub,
                    )
                    .expect("Diff expression creation should succeed");

                    let product = SimdBinaryExpr::new(sum, diff, BinaryOpType::Mul)
                        .expect("Product expression creation should succeed");

                    black_box(product.eval());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark operation fusion
fn bench_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a: Vec<f64> = (0..*size).map(|i| i as f64 + 1.0).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64) * 2.0 + 1.0).collect();

        let arr_a = Array::from_vec(a);
        let arr_b = Array::from_vec(b);

        // Expression: (a + b) * 2.0

        // Non-fused version (two separate operations)
        group.bench_with_input(BenchmarkId::new("unfused", size), size, |bencher, _| {
            bencher.iter(|| {
                let sum = BinaryExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    |x: f64, y: f64| x + y,
                )
                .expect("Sum expression creation should succeed");

                let scaled = ScalarExpr::new(sum, 2.0, |x: f64, s: f64| x * s);

                black_box(scaled.eval());
            });
        });

        // Fused version (single operation)
        group.bench_with_input(BenchmarkId::new("fused", size), size, |bencher, _| {
            bencher.iter(|| {
                let fused = FusedBinaryScalarExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    2.0,
                    |x: f64, y: f64| x + y,
                    |x: f64, s: f64| x * s,
                )
                .expect("Fused expression creation should succeed");

                black_box(fused.eval());
            });
        });
    }

    group.finish();
}

/// Benchmark buffer reuse
fn bench_buffer_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_reuse");

    for size in [10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let arr = Array::from_vec(data);

        // Without buffer reuse (allocates every time)
        group.bench_with_input(BenchmarkId::new("no_reuse", size), size, |bencher, _| {
            bencher.iter(|| {
                let expr = ArrayExpr::new(&arr);
                black_box(expr.eval());
            });
        });

        // With buffer reuse
        group.bench_with_input(BenchmarkId::new("with_reuse", size), size, |bencher, _| {
            let mut pool = BufferPool::default();
            bencher.iter(|| {
                let expr = ArrayExpr::new(&arr);
                black_box(expr.eval_simd_optimized(&mut pool));
            });
        });
    }

    group.finish();
}

/// Benchmark chained operations
fn bench_chained_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_operations");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f64> = (0..*size).map(|i| (i as f64) + 1.0).collect();
        let arr = Array::from_vec(data);

        // Chain: x -> x * 2 -> x + 1 -> sqrt(x)
        group.bench_with_input(BenchmarkId::new("chain_4ops", size), size, |bencher, _| {
            bencher.iter(|| {
                let mul = ScalarExpr::new(ArrayExpr::new(&arr), 2.0, |x: f64, s: f64| x * s);
                let add = ScalarExpr::new(mul, 1.0, |x: f64, s: f64| x + s);
                let sqrt = UnaryExpr::new(add, |x: f64| x.sqrt());
                black_box(sqrt.eval());
            });
        });

        // SIMD version
        group.bench_with_input(
            BenchmarkId::new("chain_4ops_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let mul = ScalarExpr::new(ArrayExpr::new(&arr), 2.0, |x: f64, s: f64| x * s);
                    let add = ScalarExpr::new(mul, 1.0, |x: f64, s: f64| x + s);
                    let sqrt = UnaryExpr::new(add, |x: f64| x.sqrt());
                    black_box(sqrt.eval_simd());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark large array expressions
fn bench_large_arrays(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_arrays");
    group.sample_size(10); // Fewer samples for large arrays

    for size in [1_000_000, 10_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..*size).map(|i| (i as f64) + 1.0).collect();

        let arr_a = Array::from_vec(a);
        let arr_b = Array::from_vec(b);

        group.bench_with_input(BenchmarkId::new("add_large", size), size, |bencher, _| {
            bencher.iter(|| {
                let expr = BinaryExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    |x: f64, y: f64| x + y,
                )
                .expect("Expression creation should succeed");
                black_box(expr.eval());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("add_large_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let expr = SimdBinaryExpr::new(
                        ArrayExpr::new(&arr_a),
                        ArrayExpr::new(&arr_b),
                        BinaryOpType::Add,
                    )
                    .expect("Expression creation should succeed");
                    black_box(expr.eval());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    let size = 100_000;
    let a: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..size).map(|i| (i as f64) * 2.0).collect();

    let arr_a = Array::from_vec(a);
    let arr_b = Array::from_vec(b);

    // Direct evaluation (many allocations)
    group.bench_function("many_small_evals", |bencher| {
        bencher.iter(|| {
            for _ in 0..10 {
                let expr = BinaryExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    |x: f64, y: f64| x + y,
                )
                .expect("Expression creation should succeed");
                black_box(expr.eval());
            }
        });
    });

    // Batch evaluation with buffer reuse
    group.bench_function("batch_with_reuse", |bencher| {
        bencher.iter(|| {
            let mut pool = BufferPool::new(10);
            for _ in 0..10 {
                let expr = BinaryExpr::new(
                    ArrayExpr::new(&arr_a),
                    ArrayExpr::new(&arr_b),
                    |x: f64, y: f64| x + y,
                )
                .expect("Expression creation should succeed");
                black_box(expr.eval_simd_optimized(&mut pool));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_expression,
    bench_complex_expression,
    bench_fusion,
    bench_buffer_reuse,
    bench_chained_operations,
    bench_large_arrays,
    bench_allocation_patterns,
);
criterion_main!(benches);
