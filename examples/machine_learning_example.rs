//! Machine Learning Example for NumRS2
//!
//! This example demonstrates how NumRS2 can be used for machine learning tasks,
//! including linear regression, polynomial fitting, classification, and basic
//! neural network operations.
//!
//! Run with: cargo run --example machine_learning_example

use numrs2::autodiff::Dual;
use numrs2::prelude::*;

fn main() {
    println!("=== NumRS2 Machine Learning Examples ===\n");

    // Example 1: Linear Regression from Scratch
    linear_regression_example();

    // Example 2: Polynomial Fitting
    polynomial_fitting_example();

    // Example 3: Data Normalization
    data_normalization_example();

    // Example 4: Logistic Regression
    logistic_regression_example();

    // Example 5: Basic Neural Network Forward Pass
    neural_network_example();

    // Example 6: Gradient Descent Optimization
    gradient_descent_example();

    // Example 7: K-Means Clustering (Simple)
    kmeans_clustering_example();

    println!("\n=== All Examples Completed Successfully! ===");
}

/// Example 1: Linear Regression using Normal Equations
fn linear_regression_example() {
    println!("1. Linear Regression (Normal Equations)");
    println!("---------------------------------------");

    // Generate sample data: y = 2x + 1 + noise
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.2, 16.8, 19.1, 20.9];

    let x = Array::from_vec(x_data.clone());
    let y = Array::from_vec(y_data.clone());
    let n = x.size();

    // Add bias column (X becomes [1, x])
    let mut x_bias = Array::<f64>::ones(&[n, 2]);
    for i in 0..n {
        x_bias.set(&[i, 1], x.get(&[i]).unwrap()).unwrap();
    }

    // Normal equation: θ = (X^T X)^(-1) X^T y
    let xt = x_bias.transpose();
    let xtx = xt.matmul(&x_bias).unwrap();

    // Simple 2x2 inverse
    let a = xtx.get(&[0, 0]).unwrap();
    let b = xtx.get(&[0, 1]).unwrap();
    let c = xtx.get(&[1, 0]).unwrap();
    let d = xtx.get(&[1, 1]).unwrap();
    let det = a * d - b * c;

    let mut xtx_inv = Array::<f64>::zeros(&[2, 2]);
    xtx_inv.set(&[0, 0], d / det).unwrap();
    xtx_inv.set(&[0, 1], -b / det).unwrap();
    xtx_inv.set(&[1, 0], -c / det).unwrap();
    xtx_inv.set(&[1, 1], a / det).unwrap();

    let xty = matvec(&xt, &y).unwrap();
    let theta = matvec(&xtx_inv, &xty).unwrap();

    let intercept = theta.get(&[0]).unwrap();
    let slope = theta.get(&[1]).unwrap();

    println!("  Fitted model: y = {:.4}x + {:.4}", slope, intercept);
    println!("  (True model: y = 2x + 1)");

    // Calculate R² score
    let y_pred = x.multiply_scalar(slope).add_scalar(intercept);
    let y_mean = y.mean();
    let ss_tot: f64 = (0..n)
        .map(|i| {
            let diff = y.get(&[i]).unwrap() - y_mean;
            diff * diff
        })
        .sum();
    let ss_res: f64 = (0..n)
        .map(|i| {
            let diff = y.get(&[i]).unwrap() - y_pred.get(&[i]).unwrap();
            diff * diff
        })
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;
    println!("  R² Score: {:.6}\n", r2);
}

/// Helper for matrix-vector multiplication
#[allow(clippy::result_large_err)]
fn matvec(mat: &Array<f64>, vec: &Array<f64>) -> Result<Array<f64>> {
    let rows = mat.shape()[0];
    let cols = mat.shape()[1];
    let mut result = Array::<f64>::zeros(&[rows]);

    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += mat.get(&[i, j]).unwrap() * vec.get(&[j]).unwrap();
        }
        result.set(&[i], sum)?;
    }
    Ok(result)
}

/// Example 2: Polynomial Fitting
#[allow(deprecated)]
fn polynomial_fitting_example() {
    println!("2. Polynomial Fitting");
    println!("---------------------");

    // Generate quadratic data: y = 0.5x² - 2x + 3
    let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .map(|&x| 0.5 * x * x - 2.0 * x + 3.0 + 0.1 * x.sin())
        .collect();

    let x = Array::from_vec(x_data);
    let y = Array::from_vec(y_data);

    // Fit polynomial of degree 2
    match numrs2::new_modules::polynomial::polyfit(&x, &y, 2) {
        Ok(poly) => {
            let coeffs = poly.coefficients();
            println!("  Fitted polynomial coefficients:");
            println!("    a₀ (constant): {:.4}", coeffs[0]);
            println!("    a₁ (linear):   {:.4}", coeffs[1]);
            println!("    a₂ (quadratic): {:.4}", coeffs[2]);
            println!("  (True: a₀=3, a₁=-2, a₂=0.5)\n");
        }
        Err(e) => {
            println!("  Polynomial fitting failed: {:?}\n", e);
        }
    }
}

/// Example 3: Data Normalization
fn data_normalization_example() {
    println!("3. Data Normalization");
    println!("---------------------");

    // Create sample data
    let data = Array::from_vec(vec![100.0_f64, 200.0, 300.0, 400.0, 500.0]);

    // Z-score normalization (standardization)
    let mean = data.mean();
    let std = data.std();
    let normalized: Array<f64> = data.add_scalar(-mean).multiply_scalar(1.0 / std);

    println!("  Original data: {:?}", data.to_vec());
    println!("  Mean: {:.2}, Std: {:.2}", mean, std);
    println!("  Z-score normalized: {:?}", normalized.to_vec());

    // Min-Max normalization
    let min = data.min();
    let max = data.max();
    let range = max - min;
    let minmax_norm: Array<f64> = data.add_scalar(-min).multiply_scalar(1.0 / range);

    println!("  Min-Max normalized: {:?}\n", minmax_norm.to_vec());
}

/// Example 4: Logistic Regression (Binary Classification)
fn logistic_regression_example() {
    println!("4. Logistic Regression");
    println!("----------------------");

    // Simple 2D classification problem
    // Class 0: points with x1 + x2 < 0
    // Class 1: points with x1 + x2 >= 0
    let x_data: Vec<f64> = vec![
        -2.0, -1.0, // class 0
        -1.0, -2.0, // class 0
        -0.5, -0.5, // class 0
        1.0, 0.5, // class 1
        0.5, 1.0, // class 1
        2.0, 1.0, // class 1
    ];
    let y_labels: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let n_samples = 6_usize;
    let _n_features = 2_usize;

    // Initialize weights
    let mut w = Array::from_vec(vec![0.0_f64, 0.0]);
    let mut b = 0.0_f64;
    let learning_rate = 0.5_f64;
    let n_iterations = 100_usize;

    // Training loop
    for _iter in 0..n_iterations {
        let mut grad_w = Array::from_vec(vec![0.0_f64, 0.0]);
        let mut grad_b = 0.0_f64;

        for i in 0..n_samples {
            let x1 = x_data[i * 2];
            let x2 = x_data[i * 2 + 1];
            let y_true = y_labels[i];

            // Forward pass: z = w·x + b, sigmoid(z)
            let z = w.get(&[0]).unwrap() * x1 + w.get(&[1]).unwrap() * x2 + b;
            let y_pred = 1.0_f64 / (1.0_f64 + (-z).exp());

            // Compute gradients
            let error = y_pred - y_true;
            grad_w
                .set(&[0], grad_w.get(&[0]).unwrap() + error * x1)
                .unwrap();
            grad_w
                .set(&[1], grad_w.get(&[1]).unwrap() + error * x2)
                .unwrap();
            grad_b += error;
        }

        // Update weights
        for j in 0..2 {
            let new_w =
                w.get(&[j]).unwrap() - learning_rate * grad_w.get(&[j]).unwrap() / n_samples as f64;
            w.set(&[j], new_w).unwrap();
        }
        b -= learning_rate * grad_b / n_samples as f64;
    }

    println!(
        "  Learned weights: w1={:.4}, w2={:.4}, b={:.4}",
        w.get(&[0]).unwrap(),
        w.get(&[1]).unwrap(),
        b
    );

    // Test predictions
    println!("  Predictions:");
    for i in 0..n_samples {
        let x1 = x_data[i * 2];
        let x2 = x_data[i * 2 + 1];
        let z = w.get(&[0]).unwrap() * x1 + w.get(&[1]).unwrap() * x2 + b;
        let prob = 1.0_f64 / (1.0_f64 + (-z).exp());
        let pred = if prob >= 0.5 { 1 } else { 0 };
        println!(
            "    ({:.1}, {:.1}) -> prob: {:.4}, pred: {}, true: {}",
            x1, x2, prob, pred, y_labels[i] as i32
        );
    }
    println!();
}

/// Example 5: Basic Neural Network Forward Pass
fn neural_network_example() {
    println!("5. Basic Neural Network (2-Layer MLP)");
    println!("-------------------------------------");

    // Network architecture: 2 -> 3 -> 1 (input -> hidden -> output)
    let input = Array::from_vec(vec![0.5, 0.8]);

    // Initialize weights (normally would use random initialization)
    let w1 = Array::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(&[3, 2]); // 3x2
    let b1 = Array::from_vec(vec![0.1, 0.1, 0.1]);

    let w2 = Array::from_vec(vec![0.2, 0.3, 0.4]).reshape(&[1, 3]); // 1x3
    let b2 = Array::from_vec(vec![0.1]);

    // Forward pass - Layer 1
    let z1 = matvec(&w1, &input).unwrap();
    let z1_biased = array_add(&z1, &b1);

    // ReLU activation
    let h1 = relu(&z1_biased);
    println!("  Hidden layer (after ReLU): {:?}", h1.to_vec());

    // Forward pass - Layer 2
    let z2 = matvec(&w2, &h1).unwrap();
    let output = array_add(&z2, &b2);

    // Sigmoid activation
    let final_output = sigmoid_vec(&output);
    println!("  Output (after Sigmoid): {:?}\n", final_output.to_vec());
}

fn array_add(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
    let n = a.size();
    let mut result = Array::<f64>::zeros(&[n]);
    for i in 0..n {
        result
            .set(&[i], a.get(&[i]).unwrap() + b.get(&[i]).unwrap())
            .unwrap();
    }
    result
}

fn relu(arr: &Array<f64>) -> Array<f64> {
    let n = arr.size();
    let mut result = Array::<f64>::zeros(&[n]);
    for i in 0..n {
        let val = arr.get(&[i]).unwrap();
        result.set(&[i], if val > 0.0 { val } else { 0.0 }).unwrap();
    }
    result
}

fn sigmoid_vec(arr: &Array<f64>) -> Array<f64> {
    let n = arr.size();
    let mut result = Array::<f64>::zeros(&[n]);
    for i in 0..n {
        let val = arr.get(&[i]).unwrap();
        result.set(&[i], 1.0 / (1.0 + (-val).exp())).unwrap();
    }
    result
}

/// Example 6: Gradient Descent using Autodiff (Forward Mode with Dual Numbers)
fn gradient_descent_example() {
    println!("6. Gradient Descent Optimization");
    println!("--------------------------------");

    // Minimize f(x) = (x - 3)² using gradient descent with Dual numbers
    let target = 3.0_f64;
    let mut x = 0.0_f64; // Initial guess
    let learning_rate = 0.1_f64;

    println!("  Minimizing f(x) = (x - 3)²");
    println!("  Initial x: {:.4}", x);

    for iter in 0..20 {
        // Compute gradient using forward mode autodiff (Dual numbers)
        // Dual::variable(x) creates x + ε where ε² = 0
        // The derivative appears in the dual part
        let x_dual = Dual::variable(x);
        let target_dual = Dual::constant(target);
        let diff = x_dual - target_dual;
        let f = diff * diff; // f(x) = (x - target)²

        let grad = f.deriv(); // f'(x) = 2(x - target)

        // Update
        x -= learning_rate * grad;

        if (iter + 1) % 5 == 0 {
            println!(
                "  Iteration {:2}: x = {:.6}, f(x) = {:.6}",
                iter + 1,
                x,
                (x - target) * (x - target)
            );
        }
    }
    println!("  Final x: {:.6} (target: {})\n", x, target);
}

/// Example 7: K-Means Clustering (Simple Implementation)
fn kmeans_clustering_example() {
    println!("7. K-Means Clustering");
    println!("---------------------");

    // 2D data points (two clear clusters)
    let points: Vec<(f64, f64)> = vec![
        (1.0, 1.0),
        (1.5, 1.2),
        (1.2, 0.8),
        (5.0, 5.0),
        (5.5, 5.2),
        (4.8, 5.3),
    ];

    // Initialize centroids
    let mut centroids: Vec<(f64, f64)> = vec![(1.0, 1.0), (5.0, 5.0)];
    let k = 2;

    println!("  Initial centroids: {:?}", centroids);

    // Run K-means for a few iterations
    for iter in 0..5 {
        // Assign points to nearest centroid
        let mut clusters: Vec<Vec<(f64, f64)>> = vec![vec![], vec![]];

        for &point in &points {
            let mut min_dist = f64::MAX;
            let mut best_k = 0;

            for (i, centroid) in centroids.iter().enumerate() {
                let dx = point.0 - centroid.0;
                let dy = point.1 - centroid.1;
                let dist = dx * dx + dy * dy;
                if dist < min_dist {
                    min_dist = dist;
                    best_k = i;
                }
            }
            clusters[best_k].push(point);
        }

        // Update centroids
        for i in 0..k {
            if !clusters[i].is_empty() {
                let sum_x: f64 = clusters[i].iter().map(|p| p.0).sum();
                let sum_y: f64 = clusters[i].iter().map(|p| p.1).sum();
                let n = clusters[i].len() as f64;
                centroids[i] = (sum_x / n, sum_y / n);
            }
        }

        if iter == 4 {
            println!("  Final centroids after {} iterations:", iter + 1);
            for (i, centroid) in centroids.iter().enumerate() {
                println!("    Cluster {}: ({:.4}, {:.4})", i, centroid.0, centroid.1);
            }
            println!("  Cluster assignments:");
            for (i, cluster) in clusters.iter().enumerate() {
                println!("    Cluster {}: {:?}", i, cluster);
            }
        }
    }
}

use numrs2::error::Result;
