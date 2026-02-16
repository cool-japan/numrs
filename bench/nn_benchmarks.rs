//! Benchmarks for Neural Network operations
//!
//! This benchmark suite measures performance of NN primitives including:
//! - Activation functions (ReLU, GELU, Swish, Mish, etc.)
//! - Convolution operations (1D, 2D)
//! - Pooling operations (max, average)
//! - Normalization (batch norm, layer norm)
//! - Loss functions (MSE, cross-entropy, etc.)
//! - SIMD vs scalar performance comparison

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::nn::activation::*;
use numrs2::nn::conv::*;
use numrs2::nn::loss::*;
use numrs2::nn::normalization::*;
use numrs2::nn::pooling::*;
use numrs2::nn::simd_ops::*;
use numrs2::nn::ReductionMode;
use scirs2_core::ndarray::{Array1, Array2};
use std::hint::black_box;

// ================================
// Activation Function Benchmarks
// ================================

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    // Test sizes: small, medium, large
    let sizes = vec![100, 1000, 10000, 100000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let x_f32 = Array1::from_vec((0..size).map(|i| (i as f32) / 100.0 - 5.0).collect());
        let x_f64 = Array1::from_vec((0..size).map(|i| (i as f64) / 100.0 - 5.0).collect());

        // ReLU benchmarks
        group.bench_with_input(BenchmarkId::new("relu_f64", size), &size, |b, _| {
            b.iter(|| relu(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("simd_relu_f32", size), &size, |b, _| {
            b.iter(|| simd_relu_f32(&black_box(x_f32.view())))
        });

        // Sigmoid benchmarks
        group.bench_with_input(BenchmarkId::new("sigmoid_f64", size), &size, |b, _| {
            b.iter(|| sigmoid(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("simd_sigmoid_f32", size), &size, |b, _| {
            b.iter(|| simd_sigmoid_f32(&black_box(x_f32.view())))
        });

        // Tanh benchmarks
        group.bench_with_input(BenchmarkId::new("tanh_f64", size), &size, |b, _| {
            b.iter(|| tanh(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("simd_tanh_f32", size), &size, |b, _| {
            b.iter(|| simd_tanh_f32(&black_box(x_f32.view())))
        });

        // GELU benchmarks
        group.bench_with_input(BenchmarkId::new("gelu_f64", size), &size, |b, _| {
            b.iter(|| gelu(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("simd_gelu_f32", size), &size, |b, _| {
            b.iter(|| simd_gelu_f32(&black_box(x_f32.view())))
        });

        // Swish benchmarks
        group.bench_with_input(BenchmarkId::new("swish_f64", size), &size, |b, _| {
            b.iter(|| swish(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("simd_swish_f32", size), &size, |b, _| {
            b.iter(|| simd_swish_f32(&black_box(x_f32.view())))
        });

        // Mish benchmarks
        group.bench_with_input(BenchmarkId::new("mish_f64", size), &size, |b, _| {
            b.iter(|| mish(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("simd_mish_f32", size), &size, |b, _| {
            b.iter(|| simd_mish_f32(&black_box(x_f32.view())))
        });

        // Softmax benchmarks
        group.bench_with_input(BenchmarkId::new("softmax_f64", size), &size, |b, _| {
            b.iter(|| softmax(&black_box(x_f64.view())))
        });
    }

    group.finish();
}

// ================================
// Convolution Benchmarks
// ================================

fn bench_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution");

    // 1D convolution benchmarks
    let input_1d = Array1::from_vec((0..1000).map(|i| i as f64).collect());
    let kernel_1d_small = Array1::from_vec(vec![1.0, 0.0, -1.0]);
    let kernel_1d_large = Array1::from_vec(vec![1.0; 11]);

    group.bench_function("conv1d_small_kernel", |b| {
        b.iter(|| {
            conv1d(
                &black_box(input_1d.view()),
                &black_box(kernel_1d_small.view()),
                1,
            )
        })
    });

    group.bench_function("conv1d_large_kernel", |b| {
        b.iter(|| {
            conv1d(
                &black_box(input_1d.view()),
                &black_box(kernel_1d_large.view()),
                1,
            )
        })
    });

    // 2D convolution benchmarks
    let sizes = vec![(32, 32), (64, 64), (128, 128)];
    let kernel_sizes = vec![(3, 3), (5, 5)];

    for (h, w) in sizes {
        for (kh, kw) in &kernel_sizes {
            let input_2d = Array2::from_shape_fn((h, w), |(i, j)| (i + j) as f64);
            let kernel_2d =
                Array2::from_shape_fn((*kh, *kw), |(i, j)| if i == j { 1.0 } else { 0.0 });

            group.throughput(Throughput::Elements((h * w) as u64));
            group.bench_with_input(
                BenchmarkId::new("conv2d", format!("{}x{}_k{}x{}", h, w, kh, kw)),
                &(h, w),
                |b, _| {
                    b.iter(|| {
                        conv2d(
                            &black_box(input_2d.view()),
                            &black_box(kernel_2d.view()),
                            (1, 1),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// ================================
// Pooling Benchmarks
// ================================

fn bench_pooling(c: &mut Criterion) {
    let mut group = c.benchmark_group("pooling");

    let sizes = vec![(32, 32), (64, 64), (128, 128), (256, 256)];
    let pool_sizes = vec![(2, 2), (4, 4)];

    for (h, w) in sizes {
        for (ph, pw) in &pool_sizes {
            let input = Array2::from_shape_fn((h, w), |(i, j)| (i + j) as f64);

            group.throughput(Throughput::Elements((h * w) as u64));

            // Max pooling
            group.bench_with_input(
                BenchmarkId::new("max_pool2d", format!("{}x{}_p{}x{}", h, w, ph, pw)),
                &(h, w),
                |b, _| b.iter(|| max_pool2d(&black_box(input.view()), (*ph, *pw), (*ph, *pw))),
            );

            // Average pooling
            group.bench_with_input(
                BenchmarkId::new("avg_pool2d", format!("{}x{}_p{}x{}", h, w, ph, pw)),
                &(h, w),
                |b, _| b.iter(|| avg_pool2d(&black_box(input.view()), (*ph, *pw), (*ph, *pw))),
            );
        }
    }

    group.finish();
}

// ================================
// Normalization Benchmarks
// ================================

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    let batch_sizes = vec![32, 64, 128];
    let feature_sizes = vec![128, 256, 512];

    for batch_size in batch_sizes {
        for feature_size in &feature_sizes {
            let x = Array2::from_shape_fn((batch_size, *feature_size), |(i, j)| {
                (i * feature_size + j) as f64 / 100.0
            });
            let gamma = Array1::ones(*feature_size);
            let beta = Array1::zeros(*feature_size);
            let epsilon = 1e-5;

            group.throughput(Throughput::Elements((batch_size * feature_size) as u64));

            // Batch normalization
            group.bench_with_input(
                BenchmarkId::new(
                    "batch_norm_1d",
                    format!("b{}_f{}", batch_size, feature_size),
                ),
                &(batch_size, feature_size),
                |b, _| {
                    b.iter(|| {
                        batch_norm_1d(
                            &black_box(x.view()),
                            &black_box(gamma.view()),
                            &black_box(beta.view()),
                            epsilon,
                        )
                    })
                },
            );

            // Layer normalization
            group.bench_with_input(
                BenchmarkId::new("layer_norm", format!("b{}_f{}", batch_size, feature_size)),
                &(batch_size, feature_size),
                |b, _| {
                    b.iter(|| {
                        layer_norm(
                            &black_box(x.view()),
                            &black_box(gamma.view()),
                            &black_box(beta.view()),
                            epsilon,
                        )
                    })
                },
            );

            // Instance normalization - commented out, not implemented
            // group.bench_with_input(
            //     BenchmarkId::new(
            //         "instance_norm",
            //         format!("b{}_f{}", batch_size, feature_size),
            //     ),
            //     &(batch_size, feature_size),
            //     |b, _| b.iter(|| instance_norm(&black_box(x.view()), epsilon)),
            // );

            // Dropout (using dropout_2d for 2D arrays)
            group.bench_with_input(
                BenchmarkId::new("dropout", format!("b{}_f{}", batch_size, feature_size)),
                &(batch_size, feature_size),
                |b, _| b.iter(|| dropout_2d(&black_box(x.view()), 0.5, false)),
            );
        }
    }

    group.finish();
}

// ================================
// Loss Function Benchmarks
// ================================

fn bench_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_functions");

    let sizes = vec![100, 1000, 10000, 100000];

    for size in sizes {
        let y_true = Array1::from_vec((0..size).map(|i| (i % 10) as f64 / 10.0).collect());
        let y_pred = Array1::from_vec((0..size).map(|i| ((i + 1) % 10) as f64 / 10.0).collect());

        // For categorical cross entropy, we need 2D arrays (batch_size x num_classes)
        let batch_size = 100.min(size);
        let num_classes = size / batch_size;
        let y_true_2d = Array2::from_shape_vec(
            (batch_size, num_classes),
            (0..batch_size * num_classes)
                .map(|i| {
                    if i % num_classes == (i / num_classes) % num_classes {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((batch_size, num_classes)));
        let y_pred_2d = Array2::from_shape_vec(
            (batch_size, num_classes),
            (0..batch_size * num_classes)
                .map(|i| 1.0 / num_classes as f64)
                .collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((batch_size, num_classes)));

        group.throughput(Throughput::Elements(size as u64));

        // MSE loss
        group.bench_with_input(BenchmarkId::new("mse_loss", size), &size, |b, _| {
            b.iter(|| {
                mse_loss(
                    &black_box(y_true.view()),
                    &black_box(y_pred.view()),
                    ReductionMode::Mean,
                )
            })
        });

        // MAE loss
        group.bench_with_input(BenchmarkId::new("mae_loss", size), &size, |b, _| {
            b.iter(|| {
                mae_loss(
                    &black_box(y_true.view()),
                    &black_box(y_pred.view()),
                    ReductionMode::Mean,
                )
            })
        });

        // Huber loss
        group.bench_with_input(BenchmarkId::new("huber_loss", size), &size, |b, _| {
            b.iter(|| {
                huber_loss(
                    &black_box(y_true.view()),
                    &black_box(y_pred.view()),
                    1.0,
                    ReductionMode::Mean,
                )
            })
        });

        // Cross entropy loss (categorical - requires 2D arrays)
        group.bench_with_input(
            BenchmarkId::new("cross_entropy_loss", size),
            &size,
            |b, _| {
                b.iter(|| {
                    categorical_cross_entropy(
                        &black_box(y_true_2d.view()),
                        &black_box(y_pred_2d.view()),
                        ReductionMode::Mean,
                    )
                })
            },
        );

        // Binary cross entropy loss
        group.bench_with_input(
            BenchmarkId::new("binary_cross_entropy_loss", size),
            &size,
            |b, _| {
                b.iter(|| {
                    binary_cross_entropy(
                        &black_box(y_true.view()),
                        &black_box(y_pred.view()),
                        ReductionMode::Mean,
                    )
                })
            },
        );

        // Focal loss
        group.bench_with_input(BenchmarkId::new("focal_loss", size), &size, |b, _| {
            b.iter(|| {
                focal_loss(
                    &black_box(y_true.view()),
                    &black_box(y_pred.view()),
                    0.25,
                    2.0,
                    ReductionMode::Mean,
                )
            })
        });
    }

    group.finish();
}

// ================================
// SIMD vs Scalar Comparison
// ================================

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let x_f32 = Array1::from_vec((0..size).map(|i| (i as f32) / 100.0 - 5.0).collect());
        let x_f64 = Array1::from_vec((0..size).map(|i| (i as f64) / 100.0 - 5.0).collect());

        group.throughput(Throughput::Elements(size as u64));

        // ReLU comparison
        group.bench_with_input(BenchmarkId::new("relu_scalar_f64", size), &size, |b, _| {
            b.iter(|| relu(&black_box(x_f64.view())))
        });

        group.bench_with_input(BenchmarkId::new("relu_simd_f32", size), &size, |b, _| {
            b.iter(|| simd_relu_f32(&black_box(x_f32.view())))
        });

        // Sigmoid comparison
        group.bench_with_input(
            BenchmarkId::new("sigmoid_scalar_f64", size),
            &size,
            |b, _| b.iter(|| sigmoid(&black_box(x_f64.view()))),
        );

        group.bench_with_input(BenchmarkId::new("sigmoid_simd_f32", size), &size, |b, _| {
            b.iter(|| simd_sigmoid_f32(&black_box(x_f32.view())))
        });

        // Matrix multiplication comparison
        let n = (size as f64).sqrt() as usize;
        if n > 1 {
            let a_f32 = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f32);
            let b_f32 = Array2::from_shape_fn((n, n), |(i, j)| (i * 2 + j) as f32);

            group.throughput(Throughput::Elements((n * n * n) as u64));

            group.bench_with_input(
                BenchmarkId::new("matmul_simd_f32", format!("{}x{}", n, n)),
                &n,
                |b, _| {
                    b.iter(|| simd_matmul_f32(&black_box(a_f32.view()), &black_box(b_f32.view())))
                },
            );
        }
    }

    group.finish();
}

// ================================
// SIMD Reduction Operations
// ================================

fn bench_simd_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_reductions");

    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        let x = Array1::from_vec((0..size).map(|i| i as f32).collect());

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, _| {
            b.iter(|| simd_sum_f32(&black_box(x.view())))
        });

        group.bench_with_input(BenchmarkId::new("mean", size), &size, |b, _| {
            b.iter(|| simd_mean_f32(&black_box(x.view())))
        });

        group.bench_with_input(BenchmarkId::new("norm", size), &size, |b, _| {
            b.iter(|| simd_norm_f32(&black_box(x.view())))
        });

        group.bench_with_input(BenchmarkId::new("min", size), &size, |b, _| {
            b.iter(|| simd_min_f32(&black_box(x.view())))
        });

        group.bench_with_input(BenchmarkId::new("max", size), &size, |b, _| {
            b.iter(|| simd_max_f32(&black_box(x.view())))
        });

        let y = Array1::from_vec((0..size).map(|i| (i + 1) as f32).collect());
        group.bench_with_input(BenchmarkId::new("dot", size), &size, |b, _| {
            b.iter(|| simd_dot_f32(&black_box(x.view()), &black_box(y.view())))
        });
    }

    group.finish();
}

// ================================
// Combined Operations Benchmark
// ================================

fn bench_combined_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_operations");

    // Simulate a mini forward pass through a layer
    let batch_size = 128;
    let input_size = 512;
    let output_size = 256;

    let x = Array2::from_shape_fn((batch_size, input_size), |(i, j)| {
        (i * input_size + j) as f32 / 1000.0
    });
    let weights = Array2::from_shape_fn((input_size, output_size), |(i, j)| {
        ((i + j) as f32 / 1000.0) - 0.5
    });
    let gamma = Array1::ones(output_size);
    let beta = Array1::zeros(output_size);

    group.throughput(Throughput::Elements((batch_size * output_size) as u64));

    group.bench_function("forward_pass_linear_relu_batchnorm", |b| {
        b.iter(|| {
            // Matrix multiplication
            let linear_out = simd_matmul_f32(&black_box(x.view()), &black_box(weights.view()))
                .expect("matmul failed");

            // ReLU activation
            let relu_out = simd_relu_2d_f32(&black_box(linear_out.view()));

            // Batch normalization
            batch_norm_1d(
                &black_box(relu_out.view()),
                &black_box(gamma.view()),
                &black_box(beta.view()),
                1e-5,
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_activation_functions,
    bench_convolution,
    bench_pooling,
    bench_normalization,
    bench_loss_functions,
    bench_simd_vs_scalar,
    bench_simd_reductions,
    bench_combined_operations,
);

criterion_main!(benches);
