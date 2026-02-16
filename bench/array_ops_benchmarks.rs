//! Comprehensive Array Operations Benchmarks for NumRS2
//!
//! This benchmark suite tests all array operations including:
//! - Element-wise operations (add, sub, mul, div, pow)
//! - Broadcasting operations
//! - Reduction operations (sum, prod, min, max, argmin, argmax)
//! - Indexing and slicing
//! - Array reshaping and transposition
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::prelude::*;
use numrs2::ufuncs::sin;
use std::hint::black_box;

/// Benchmark element-wise addition
fn bench_element_wise_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_addition");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("add_1d", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a + &arr_b;
                    black_box(result);
                });
            }
        });

        // 2D arrays
        group.bench_with_input(BenchmarkId::new("add_2d", size), size, |bencher, &s| {
            let dim = (s as f64).sqrt() as usize;
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (
                rng.random::<f64>(&[dim, dim]),
                rng.random::<f64>(&[dim, dim]),
            ) {
                bencher.iter(|| {
                    let result = &a + &arr_b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark element-wise subtraction
fn bench_element_wise_subtraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_subtraction");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("sub", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a - &arr_b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark element-wise multiplication
fn bench_element_wise_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_multiplication");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("mul", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    let result = &a * &arr_b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark element-wise division
fn bench_element_wise_division(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_division");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("div", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) =
                (rng.random::<f64>(&[s]), rng.uniform::<f64>(0.1, 10.0, &[s]))
            {
                bencher.iter(|| {
                    let result = &a / &arr_b;
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark element-wise power
fn bench_element_wise_power(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_power");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("pow", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(a) = rng.uniform::<f64>(0.1, 10.0, &[s]) {
                bencher.iter(|| {
                    let result = power_scalar(&a, 2.5);
                    black_box(result);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark broadcasting operations
fn bench_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    // Broadcast scalar to array
    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("scalar_to_array", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.random::<f64>(&[s]) {
                    bencher.iter(|| {
                        let result = arr.add_scalar(5.0);
                        black_box(result);
                    });
                }
            },
        );
    }

    // Broadcast vector to matrix (row-wise)
    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("vector_to_matrix", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let (Ok(mat), Ok(vec)) = (rng.random::<f64>(&[s, s]), rng.random::<f64>(&[s])) {
                    bencher.iter(|| {
                        // Broadcast add
                        if let Ok(result) = mat.add_broadcast(&vec) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark reduction operations - sum
fn bench_reduction_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_sum");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("sum_all", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        // 2D sum along axis
        group.bench_with_input(BenchmarkId::new("sum_axis0", size), size, |bencher, &s| {
            let dim = (s as f64).sqrt() as usize;
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[dim, dim]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, Some(0), false) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("sum_axis1", size), size, |bencher, &s| {
            let dim = (s as f64).sqrt() as usize;
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[dim, dim]) {
                bencher.iter(|| {
                    if let Ok(result) = sum(&arr, Some(1), false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark reduction operations - product
fn bench_reduction_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_product");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("prod", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.uniform::<f64>(0.99, 1.01, &[s]) {
                bencher.iter(|| {
                    if let Ok(result) = prod(&arr, None, false, None) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark reduction operations - min/max
fn bench_reduction_minmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_minmax");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("min", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = min(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("max", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = max(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark argmin/argmax operations
fn bench_argminmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argminmax");

    for size in [100, 1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("argmin", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = argmin(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("argmax", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = argmax(&arr, None, false) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark array indexing
fn bench_array_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_indexing");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("get_element", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.random::<f64>(&[s]) {
                    bencher.iter(|| {
                        for i in 0..100 {
                            let idx = (i * 7) % s;
                            if let Ok(val) = arr.get(&[idx]) {
                                black_box(val);
                            }
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark array slicing
fn bench_array_slicing(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_slicing");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("slice_1d", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    let vec = arr.to_vec();
                    let sliced = &vec[0..s / 2];
                    black_box(sliced);
                });
            }
        });

        let dim = (size / 100).max(10);
        group.bench_with_input(BenchmarkId::new("slice_2d", size), size, |bencher, &_s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[dim, dim]) {
                bencher.iter(|| {
                    // For 2D slicing, convert to vec and create a subview
                    let vec = arr.to_vec();
                    let sliced = &vec[0..(dim * dim / 4)];
                    black_box(sliced);
                });
            }
        });
    }

    group.finish();
}

/// Benchmark array reshaping
fn bench_array_reshape(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_reshape");

    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("reshape_1d_to_2d", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.random::<f64>(&[s]) {
                    let dim = (s as f64).sqrt() as usize;
                    if dim * dim == s {
                        bencher.iter(|| {
                            let result = arr.reshape(&[dim, dim]);
                            black_box(result);
                        });
                    }
                }
            },
        );

        group.bench_with_input(
            BenchmarkId::new("flatten_2d_to_1d", size),
            size,
            |bencher, &s| {
                let dim = (s as f64).sqrt() as usize;
                if dim * dim == s {
                    let rng = random::default_rng();
                    if let Ok(arr) = rng.random::<f64>(&[dim, dim]) {
                        bencher.iter(|| {
                            let result = arr.reshape(&[s]);
                            black_box(result);
                        });
                    }
                }
            },
        );
    }

    group.finish();
}

/// Benchmark array transposition
fn bench_array_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_transpose");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("transpose_2d", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.random::<f64>(&[s, s]) {
                    bencher.iter(|| {
                        black_box(arr.transpose());
                    });
                }
            },
        );

        // Non-square transpose
        group.bench_with_input(
            BenchmarkId::new("transpose_rect", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.random::<f64>(&[s, s * 2]) {
                    bencher.iter(|| {
                        black_box(arr.transpose());
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark array concatenation
fn bench_array_concatenation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_concatenation");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("concat_1d", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    if let Ok(result) = concatenate(&[&a, &arr_b], 0) {
                        black_box(result);
                    }
                });
            }
        });

        let dim = (size / 10).max(10);
        group.bench_with_input(
            BenchmarkId::new("concat_2d_axis0", size),
            size,
            |bencher, &_s| {
                let rng = random::default_rng();
                if let (Ok(a), Ok(arr_b)) = (
                    rng.random::<f64>(&[dim, dim]),
                    rng.random::<f64>(&[dim, dim]),
                ) {
                    bencher.iter(|| {
                        if let Ok(result) = concatenate(&[&a, &arr_b], 0) {
                            black_box(result);
                        }
                    });
                }
            },
        );

        group.bench_with_input(
            BenchmarkId::new("concat_2d_axis1", size),
            size,
            |bencher, &_s| {
                let rng = random::default_rng();
                if let (Ok(a), Ok(arr_b)) = (
                    rng.random::<f64>(&[dim, dim]),
                    rng.random::<f64>(&[dim, dim]),
                ) {
                    bencher.iter(|| {
                        if let Ok(result) = concatenate(&[&a, &arr_b], 1) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark array stacking
fn bench_array_stacking(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_stacking");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("vstack", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    if let Ok(result) = vstack(&[&a, &arr_b]) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("hstack", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let (Ok(a), Ok(arr_b)) = (rng.random::<f64>(&[s]), rng.random::<f64>(&[s])) {
                bencher.iter(|| {
                    if let Ok(result) = hstack(&[&a, &arr_b]) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark array splitting
fn bench_array_splitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_splitting");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("split_equal", size),
            size,
            |bencher, &s| {
                let rng = random::default_rng();
                if let Ok(arr) = rng.random::<f64>(&[s]) {
                    bencher.iter(|| {
                        // Split into 4 equal parts - compute indices
                        let section_size = s / 4;
                        let indices = vec![section_size, section_size * 2, section_size * 3];
                        if let Ok(result) = split(&arr, &indices, 0) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark array tiling and repetition
fn bench_array_tiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_tiling");

    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("tile_1d", size), size, |bencher, &s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[s]) {
                bencher.iter(|| {
                    if let Ok(result) = tile(&arr, &[5]) {
                        black_box(result);
                    }
                });
            }
        });

        let dim = (size / 10).max(5);
        group.bench_with_input(BenchmarkId::new("tile_2d", size), size, |bencher, &_s| {
            let rng = random::default_rng();
            if let Ok(arr) = rng.random::<f64>(&[dim, dim]) {
                bencher.iter(|| {
                    if let Ok(result) = tile(&arr, &[3, 3]) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_element_wise_addition,
    bench_element_wise_subtraction,
    bench_element_wise_multiplication,
    bench_element_wise_division,
    bench_element_wise_power,
    bench_broadcasting,
    bench_reduction_sum,
    bench_reduction_product,
    bench_reduction_minmax,
    bench_argminmax,
    bench_array_indexing,
    bench_array_slicing,
    bench_array_reshape,
    bench_array_transpose,
    bench_array_concatenation,
    bench_array_stacking,
    bench_array_splitting,
    bench_array_tiling,
);

criterion_main!(benches);
