//! Neural Network Basics Example
//!
//! This example demonstrates the fundamental neural network primitives in NumRS2.
//! It shows how to:
//! - Build a simple feedforward network
//! - Apply different activation functions
//! - Use batch normalization and dropout
//! - Compute loss functions
//! - Implement a basic training loop structure
//!
//! This is a pedagogical example showing the building blocks - for production
//! neural networks, consider using a dedicated deep learning framework.

use numrs2::nn::activation::*;
use numrs2::nn::conv::*;
use numrs2::nn::loss::*;
use numrs2::nn::normalization::*;
use numrs2::nn::pooling::*;
use numrs2::nn::simd_ops::*;
use numrs2::nn::ReductionMode;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::*;

fn main() {
    println!("{}", "=".repeat(70));
    println!("NumRS2 Neural Network Basics Example");
    println!("{}", "=".repeat(70));
    println!();

    // Display SIMD capabilities
    println!("SIMD Capabilities:");
    println!("{}", get_simd_info());
    println!();

    // Demonstrate activation functions
    demo_activation_functions();
    println!();

    // Demonstrate normalization
    demo_normalization();
    println!();

    // Demonstrate convolution and pooling
    demo_conv_pooling();
    println!();

    // Demonstrate loss functions
    demo_loss_functions();
    println!();

    // Demonstrate a simple feedforward network
    demo_feedforward_network();
    println!();

    // Demonstrate a mini training loop
    demo_training_loop();
    println!();

    println!("{}", "=".repeat(70));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(70));
}

/// Demonstrate various activation functions
fn demo_activation_functions() {
    println!("Activation Functions Demo");
    println!("{}", "-".repeat(70));

    // Create sample input
    let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    println!("Input: {:?}", x.as_slice().expect("as_slice failed"));

    // ReLU
    let relu_out = relu(&x.view()).expect("relu failed");
    println!("ReLU: {:?}", relu_out.as_slice().expect("as_slice failed"));

    // Leaky ReLU
    let leaky_relu_out = leaky_relu(&x.view(), 0.01).expect("leaky_relu failed");
    println!(
        "Leaky ReLU (α=0.01): {:?}",
        leaky_relu_out.as_slice().expect("as_slice failed")
    );

    // Sigmoid
    let sigmoid_out = sigmoid(&x.view()).expect("sigmoid failed");
    println!(
        "Sigmoid: {:?}",
        sigmoid_out
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );

    // Tanh
    let tanh_out = tanh(&x.view()).expect("tanh failed");
    println!(
        "Tanh: {:?}",
        tanh_out
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );

    // GELU
    let gelu_out = gelu(&x.view()).expect("gelu failed");
    println!(
        "GELU: {:?}",
        gelu_out
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );

    // Swish/SiLU
    let swish_out = swish(&x.view()).expect("swish failed");
    println!(
        "Swish: {:?}",
        swish_out
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );

    // Softmax (for classification)
    let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let softmax_out = softmax(&logits.view()).expect("softmax failed");
    println!(
        "Softmax({}): {:?}",
        logits
            .iter()
            .map(|v| format!("{:.0}", v))
            .collect::<Vec<_>>()
            .join(", "),
        softmax_out
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );
    println!("  (sum = {:.4})", softmax_out.iter().sum::<f64>());

    // Compare SIMD vs scalar performance
    println!("\nSIMD Acceleration:");
    let x_f32 = Array1::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let simd_relu_out = simd_relu_f32(&x_f32.view());
    println!(
        "SIMD ReLU (f32): {:?}",
        simd_relu_out.as_slice().expect("as_slice failed")
    );
}

/// Demonstrate normalization techniques
fn demo_normalization() {
    println!("Normalization Demo");
    println!("{}", "-".repeat(70));

    // Create batch of samples (4 samples, 3 features each)
    let x = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("from_shape_vec failed");

    println!("Input (4 samples, 3 features):");
    for i in 0..x.nrows() {
        println!(
            "  Sample {}: {:?}",
            i,
            x.row(i).as_slice().expect("as_slice failed")
        );
    }

    // Batch normalization parameters
    let gamma = Array1::ones(3);
    let beta = Array1::zeros(3);
    let epsilon = 1e-5;

    // Batch normalization
    let batch_norm_out = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");
    println!("\nBatch Normalization:");
    for i in 0..batch_norm_out.nrows() {
        println!(
            "  Sample {}: {:?}",
            i,
            batch_norm_out
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }

    // Layer normalization
    let layer_norm_out =
        layer_norm(&x.view(), &gamma.view(), &beta.view(), epsilon).expect("layer_norm failed");
    println!("\nLayer Normalization:");
    for i in 0..layer_norm_out.nrows() {
        println!(
            "  Sample {}: {:?}",
            i,
            layer_norm_out
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }

    // Dropout (p=0.5)
    println!("\nDropout (p=0.5) - random mask:");
    for trial in 0..3 {
        let dropout_out = dropout_2d(&x.view(), 0.5, true).expect("dropout_2d failed");
        println!(
            "  Trial {}: Non-zero count = {}",
            trial,
            dropout_out
                .iter()
                .filter(|&&v: &&f64| v.abs() > 1e-10)
                .count()
        );
    }
}

/// Demonstrate convolution and pooling operations
fn demo_conv_pooling() {
    println!("Convolution and Pooling Demo");
    println!("{}", "-".repeat(70));

    // 1D Convolution example
    println!("1D Convolution (edge detection):");
    let signal = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    let kernel = Array1::from_vec(vec![-1.0, 0.0, 1.0]); // Edge detection kernel
    println!(
        "  Signal: {:?}",
        signal.as_slice().expect("as_slice failed")
    );
    println!(
        "  Kernel: {:?}",
        kernel.as_slice().expect("as_slice failed")
    );

    let conv_out = conv1d(&signal.view(), &kernel.view(), 1).expect("conv1d failed");
    println!(
        "  Output: {:?}",
        conv_out
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
    );

    // 2D Convolution example
    println!("\n2D Convolution:");
    let image = Array2::from_shape_vec(
        (5, 5),
        vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ],
    )
    .expect("from_shape_vec failed");

    let kernel_2d = Array2::from_shape_vec(
        (3, 3),
        vec![1.0, 1.0, 1.0, 1.0, -8.0, 1.0, 1.0, 1.0, 1.0], // Laplacian kernel
    )
    .expect("from_shape_vec failed");

    println!("  Input image (5x5):");
    for i in 0..image.nrows() {
        println!(
            "    {:?}",
            image.row(i).as_slice().expect("as_slice failed")
        );
    }

    let conv2d_out = conv2d(&image.view(), &kernel_2d.view(), (1, 1)).expect("conv2d failed");
    println!("  Output (3x3 - Laplacian edge detection):");
    for i in 0..conv2d_out.nrows() {
        println!(
            "    {:?}",
            conv2d_out
                .row(i)
                .iter()
                .map(|v| format!("{:5.1}", v))
                .collect::<Vec<_>>()
        );
    }

    // Pooling operations
    println!("\nPooling Operations:");
    let feature_map = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .expect("from_shape_vec failed");

    println!("  Input feature map (4x4):");
    for i in 0..feature_map.nrows() {
        println!(
            "    {:?}",
            feature_map.row(i).as_slice().expect("as_slice failed")
        );
    }

    let max_pool_out = max_pool2d(&feature_map.view(), (2, 2), (2, 2)).expect("max_pool2d failed");
    println!("  Max Pooling (2x2, stride 2):");
    for i in 0..max_pool_out.nrows() {
        println!(
            "    {:?}",
            max_pool_out.row(i).as_slice().expect("as_slice failed")
        );
    }

    let avg_pool_out = avg_pool2d(&feature_map.view(), (2, 2), (2, 2)).expect("avg_pool2d failed");
    println!("  Average Pooling (2x2, stride 2):");
    for i in 0..avg_pool_out.nrows() {
        println!(
            "    {:?}",
            avg_pool_out
                .row(i)
                .iter()
                .map(|v| format!("{:.1}", v))
                .collect::<Vec<_>>()
        );
    }
}

/// Demonstrate loss functions
fn demo_loss_functions() {
    println!("Loss Functions Demo");
    println!("{}", "-".repeat(70));

    // Regression example
    println!("Regression Losses:");
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Array1::from_vec(vec![1.1, 2.3, 2.8, 4.2, 4.9]);

    let mse =
        mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean).expect("mse_loss failed");
    println!("  MSE Loss: {:.6}", mse);

    let mae =
        mae_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean).expect("mae_loss failed");
    println!("  MAE Loss: {:.6}", mae);

    let huber = huber_loss(&y_true.view(), &y_pred.view(), 1.0, ReductionMode::Mean)
        .expect("huber_loss failed");
    println!("  Huber Loss (δ=1.0): {:.6}", huber);

    // Classification example
    println!("\nClassification Losses:");
    let y_true_class =
        Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("from_shape_vec failed"); // One-hot: class 0
    let y_pred_probs =
        Array2::from_shape_vec((1, 3), vec![0.7, 0.2, 0.1]).expect("from_shape_vec failed"); // Predicted probabilities

    let ce = categorical_cross_entropy(
        &y_true_class.view(),
        &y_pred_probs.view(),
        ReductionMode::Mean,
    )
    .expect("categorical_cross_entropy failed");
    println!("  Cross Entropy Loss: {:.6}", ce);

    // Binary classification
    let y_true_binary = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
    let y_pred_binary = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.95, 0.05]);

    let bce = binary_cross_entropy(
        &y_true_binary.view(),
        &y_pred_binary.view(),
        ReductionMode::Mean,
    )
    .expect("binary_cross_entropy failed");
    println!("  Binary Cross Entropy Loss: {:.6}", bce);

    // Focal loss (for imbalanced datasets)
    let focal = focal_loss(
        &y_true_binary.view(),
        &y_pred_binary.view(),
        0.25,
        2.0,
        ReductionMode::Mean,
    )
    .expect("focal_loss failed");
    println!("  Focal Loss (α=0.25, γ=2.0): {:.6}", focal);
}

/// Demonstrate a simple feedforward network
fn demo_feedforward_network() {
    println!("Simple Feedforward Network Demo");
    println!("{}", "-".repeat(70));

    // Network architecture: 4 -> 8 -> 8 -> 3
    println!("Architecture: Input(4) -> Hidden(8) -> Hidden(8) -> Output(3)");
    println!();

    // Create random input (batch_size=2, features=4)
    let input = Array2::from_shape_vec((2, 4), vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.4, -0.6, 0.9])
        .expect("from_shape_vec failed");

    println!("Input (2 samples, 4 features):");
    for i in 0..input.nrows() {
        println!(
            "  Sample {}: {:?}",
            i,
            input
                .row(i)
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>()
        );
    }

    // Layer 1: Linear(4 -> 8) + ReLU
    println!("\nLayer 1: Linear(4->8) + ReLU");
    let weights1 =
        Array2::from_shape_fn((4, 8), |(i, j)| ((i as f32 + j as f32) / 10.0 - 0.5) as f64);
    let linear1_out = simd_matmul_f64(&input.view(), &weights1.view()).expect("matmul failed");
    let relu1_out = relu_2d(&linear1_out.view()).expect("relu_2d failed");

    println!("  Output shape: {:?}", relu1_out.shape());
    println!(
        "  Mean activation: {:.4}",
        relu1_out.mean().expect("mean failed")
    );

    // Batch normalization
    let gamma1 = Array1::ones(8);
    let beta1 = Array1::zeros(8);
    let bn1_out = batch_norm_1d(&relu1_out.view(), &gamma1.view(), &beta1.view(), 1e-5)
        .expect("batch_norm_1d failed");

    println!(
        "  After batch norm mean: {:.4}",
        bn1_out.mean().expect("mean failed")
    );

    // Layer 2: Linear(8 -> 8) + ReLU
    println!("\nLayer 2: Linear(8->8) + ReLU");
    let weights2 = Array2::from_shape_fn((8, 8), |(i, j)| if i == j { 1.0 } else { 0.1 });
    let linear2_out = simd_matmul_f64(&bn1_out.view(), &weights2.view()).expect("matmul failed");
    let relu2_out = relu_2d(&linear2_out.view()).expect("relu_2d failed");

    println!("  Output shape: {:?}", relu2_out.shape());
    println!(
        "  Mean activation: {:.4}",
        relu2_out.mean().expect("mean failed")
    );

    // Output layer: Linear(8 -> 3) + Softmax
    println!("\nOutput Layer: Linear(8->3) + Softmax");
    let weights3 = Array2::from_shape_fn((8, 3), |(i, j)| (i + j) as f64 / 20.0);
    let linear3_out = simd_matmul_f64(&relu2_out.view(), &weights3.view()).expect("matmul failed");

    println!("  Logits:");
    for i in 0..linear3_out.nrows() {
        println!(
            "    Sample {}: {:?}",
            i,
            linear3_out
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }

    // Apply softmax to each sample (axis=1 for columns)
    let softmax_out = softmax_2d(&linear3_out.view(), 1).expect("softmax_2d failed");
    println!("  Probabilities (after softmax):");
    for i in 0..softmax_out.nrows() {
        println!(
            "    Sample {}: {:?} (sum={:.4})",
            i,
            softmax_out
                .row(i)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>(),
            softmax_out.row(i).iter().sum::<f64>()
        );
    }
}

/// Demonstrate a simplified training loop structure
fn demo_training_loop() {
    println!("Mini Training Loop Demo");
    println!("{}", "-".repeat(70));

    println!("Simulating training for 5 epochs...");
    println!();

    // Create synthetic data (batch_size=16, features=10)
    let batch_size = 16;
    let input_size = 10;
    let output_size = 3;

    // Initialize "model" parameters
    let weights = Array2::from_shape_fn((input_size, output_size), |(i, j)| {
        ((i as f32 + j as f32) / 20.0 - 0.5) as f64
    });
    let gamma = Array1::ones(output_size);
    let beta = Array1::zeros(output_size);

    for epoch in 1..=5 {
        // Generate random input and target
        let input = Array2::from_shape_fn((batch_size, input_size), |(i, j)| {
            ((i + j) as f64 / 100.0).sin()
        });

        // One-hot encoded targets (random for demo)
        let target_class = epoch % output_size;
        let mut target = Array2::zeros((batch_size, output_size));
        for i in 0..batch_size {
            target[[i, target_class]] = 1.0;
        }

        // Forward pass
        let linear_out = simd_matmul_f64(&input.view(), &weights.view()).expect("matmul failed");
        let bn_out = batch_norm_1d(&linear_out.view(), &gamma.view(), &beta.view(), 1e-5)
            .expect("batch_norm_1d failed");
        let output = softmax_2d(&bn_out.view(), 1).expect("softmax_2d failed");

        // Compute loss (using categorical cross entropy for the batch)
        let loss = categorical_cross_entropy(&target.view(), &output.view(), ReductionMode::Mean)
            .expect("categorical_cross_entropy failed");
        let avg_loss = loss;

        // Compute accuracy (for demo - just check max index)
        let mut correct = 0;
        for i in 0..batch_size {
            let pred_class = output
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("partial_cmp failed"))
                .map(|(idx, _)| idx)
                .expect("max_by failed");
            if pred_class == target_class {
                correct += 1;
            }
        }
        let accuracy = (correct as f64 / batch_size as f64) * 100.0;

        println!(
            "Epoch {}: Loss = {:.6}, Accuracy = {:.2}%",
            epoch, avg_loss, accuracy
        );

        // In a real training loop, you would:
        // 1. Compute gradients (backward pass)
        // 2. Update weights using optimizer
        // 3. Optionally update learning rate
    }

    println!();
    println!("Note: This is a simplified demo. Real training would include:");
    println!("  - Gradient computation (backward pass)");
    println!("  - Weight updates (optimizer)");
    println!("  - Validation set evaluation");
    println!("  - Early stopping and checkpointing");
}
