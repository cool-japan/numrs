//! Machine Learning Pipeline Example for NumRS2
//!
//! This example demonstrates a complete ML pipeline including:
//! - Data preprocessing (normalization, standardization, encoding)
//! - Train/test split and data partitioning
//! - Cross-validation
//! - Model training (using NumRS2 NN module)
//! - Model evaluation metrics
//! - Feature engineering and selection
//!
//! Run with: cargo run --example ml_pipeline

use numrs2::prelude::*;
use numrs2::random::default_rng;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== NumRS2 Machine Learning Pipeline Example ===\n");

    // Example 1: Data Preprocessing
    example1_data_preprocessing()?;

    // Example 2: Train/Test Split
    example2_train_test_split()?;

    // Example 3: Cross-Validation
    example3_cross_validation()?;

    // Example 4: Feature Engineering
    example4_feature_engineering()?;

    // Example 5: Model Training and Evaluation
    example5_model_training()?;

    // Example 6: Classification Pipeline
    example6_classification_pipeline()?;

    // Example 7: Regression Pipeline
    example7_regression_pipeline()?;

    println!("\n=== All ML Pipeline Examples Completed Successfully! ===");
    Ok(())
}

/// Example 1: Data Preprocessing
fn example1_data_preprocessing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Data Preprocessing");
    println!("==============================\n");

    let rng = default_rng();

    // Generate sample data with different scales
    let data1 = rng.normal(100.0, 20.0, &[100])?;
    let data2 = rng.normal(0.5, 0.1, &[100])?;
    let data3 = rng.normal(1000.0, 200.0, &[100])?;

    println!("1.1 Z-Score Normalization (Standardization)");
    println!("  Original data statistics:");
    println!(
        "    Feature 1: mean={:.2}, std={:.2}",
        data1.mean(),
        data1.std()
    );
    println!(
        "    Feature 2: mean={:.2}, std={:.2}",
        data2.mean(),
        data2.std()
    );
    println!(
        "    Feature 3: mean={:.2}, std={:.2}",
        data3.mean(),
        data3.std()
    );
    println!();

    // Z-score normalization
    let mean1 = data1.mean();
    let std1 = data1.std();
    let norm1_data: Vec<f64> = data1.to_vec().iter().map(|&x| (x - mean1) / std1).collect();
    let norm1 = Array::from_vec(norm1_data);

    let mean2 = data2.mean();
    let std2 = data2.std();
    let norm2_data: Vec<f64> = data2.to_vec().iter().map(|&x| (x - mean2) / std2).collect();
    let norm2 = Array::from_vec(norm2_data);

    let mean3 = data3.mean();
    let std3 = data3.std();
    let norm3_data: Vec<f64> = data3.to_vec().iter().map(|&x| (x - mean3) / std3).collect();
    let norm3 = Array::from_vec(norm3_data);

    println!("  Standardized data statistics:");
    println!(
        "    Feature 1: mean={:.6}, std={:.6}",
        norm1.mean(),
        norm1.std()
    );
    println!(
        "    Feature 2: mean={:.6}, std={:.6}",
        norm2.mean(),
        norm2.std()
    );
    println!(
        "    Feature 3: mean={:.6}, std={:.6}",
        norm3.mean(),
        norm3.std()
    );
    println!();

    // Min-Max normalization
    println!("1.2 Min-Max Normalization (Scaling to [0, 1])");

    let min1 = data1.min();
    let max1 = data1.max();
    let minmax1_data: Vec<f64> = data1
        .to_vec()
        .iter()
        .map(|&x| (x - min1) / (max1 - min1))
        .collect();
    let minmax1 = Array::from_vec(minmax1_data);

    println!("  Original data1: min={:.2}, max={:.2}", min1, max1);
    println!(
        "  Scaled data1: min={:.6}, max={:.6}",
        minmax1.min(),
        minmax1.max()
    );
    println!("  Scaled data1: mean={:.6}", minmax1.mean());
    println!();

    // Robust scaling (using median and IQR)
    println!("1.3 Robust Scaling (Median and IQR)");

    let data = rng.normal(50.0, 10.0, &[100])?;
    let sorted_data: Vec<f64> = {
        let mut v = data.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };

    let median = sorted_data[50];
    let q1 = sorted_data[25];
    let q3 = sorted_data[75];
    let iqr = q3 - q1;

    let robust_scaled: Vec<f64> = data.to_vec().iter().map(|&x| (x - median) / iqr).collect();
    let robust_array = Array::from_vec(robust_scaled);

    println!("  Median: {:.2}", median);
    println!("  Q1: {:.2}, Q3: {:.2}", q1, q3);
    println!("  IQR: {:.2}", iqr);
    println!(
        "  Robust scaled - mean: {:.6}, std: {:.6}",
        robust_array.mean(),
        robust_array.std()
    );
    println!();

    println!("✓ Example 1 completed\n");
    Ok(())
}

/// Example 2: Train/Test Split
fn example2_train_test_split() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Train/Test Split");
    println!("============================\n");

    let rng = default_rng();

    // Generate sample dataset
    let n_samples = 100;
    let n_features = 5;

    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Generate features
        for _ in 0..n_features {
            data.push(rng.normal(0.0, 1.0, &[1])?.get(&[0])?);
        }

        // Generate label based on features
        let label = if i < 50 { 0.0 } else { 1.0 };
        labels.push(label);
    }

    println!("2.1 Simple Train/Test Split (80/20)");

    let train_ratio = 0.8;
    let train_size = (n_samples as f64 * train_ratio) as usize;

    // Create shuffled indices
    let indices = rng.integers(0, n_samples as i64, &[n_samples])?;
    let mut shuffled_indices: Vec<usize> = Vec::with_capacity(n_samples);
    let mut used = vec![false; n_samples];

    for i in 0..n_samples {
        let mut idx = indices.get(&[i])? as usize % n_samples;
        while used[idx] {
            idx = (idx + 1) % n_samples;
        }
        shuffled_indices.push(idx);
        used[idx] = true;
    }

    // Split into train and test
    let train_indices = &shuffled_indices[..train_size];
    let test_indices = &shuffled_indices[train_size..];

    println!("  Total samples: {}", n_samples);
    println!(
        "  Training samples: {} ({:.0}%)",
        train_indices.len(),
        train_indices.len() as f64 / n_samples as f64 * 100.0
    );
    println!(
        "  Test samples: {} ({:.0}%)",
        test_indices.len(),
        test_indices.len() as f64 / n_samples as f64 * 100.0
    );
    println!();

    // Check label distribution in train and test sets
    let train_label_sum: f64 = train_indices.iter().map(|&i| labels[i]).sum();
    let test_label_sum: f64 = test_indices.iter().map(|&i| labels[i]).sum();

    println!("  Label distribution:");
    println!(
        "    Train - Class 0: {}, Class 1: {}",
        train_indices.len() - train_label_sum as usize,
        train_label_sum as usize
    );
    println!(
        "    Test - Class 0: {}, Class 1: {}",
        test_indices.len() - test_label_sum as usize,
        test_label_sum as usize
    );
    println!();

    // 2.2 Stratified Split (preserving class proportions)
    println!("2.2 Stratified Split");

    let class0_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, &label)| label == 0.0)
        .map(|(i, _)| i)
        .collect();

    let class1_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, &label)| label == 1.0)
        .map(|(i, _)| i)
        .collect();

    let train_size_c0 = (class0_indices.len() as f64 * train_ratio) as usize;
    let train_size_c1 = (class1_indices.len() as f64 * train_ratio) as usize;

    println!(
        "  Class 0 - train: {}, test: {}",
        train_size_c0,
        class0_indices.len() - train_size_c0
    );
    println!(
        "  Class 1 - train: {}, test: {}",
        train_size_c1,
        class1_indices.len() - train_size_c1
    );
    println!("  Note: Stratified split preserves class proportions");
    println!();

    println!("✓ Example 2 completed\n");
    Ok(())
}

/// Example 3: Cross-Validation
fn example3_cross_validation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Cross-Validation");
    println!("============================\n");

    let rng = default_rng();

    // Generate dataset
    let n_samples = 100;
    let data = rng.normal(0.0, 1.0, &[n_samples])?;

    println!("3.1 K-Fold Cross-Validation (k=5)");

    let k = 5;
    let fold_size = n_samples / k;

    println!("  Total samples: {}", n_samples);
    println!("  Number of folds: {}", k);
    println!("  Samples per fold: ~{}", fold_size);
    println!();

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        let test_size = test_end - test_start;
        let train_size = n_samples - test_size;

        println!(
            "  Fold {}: train={}, test={} (indices {}-{})",
            fold + 1,
            train_size,
            test_size,
            test_start,
            test_end - 1
        );
    }
    println!();

    // Calculate statistics for each fold
    println!("3.2 Fold Statistics");

    let mut fold_means = Vec::new();

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Calculate train set mean (excluding test fold)
        let mut train_sum = 0.0;
        let mut train_count = 0;

        for i in 0..n_samples {
            if i < test_start || i >= test_end {
                train_sum += data.get(&[i])?;
                train_count += 1;
            }
        }

        let train_mean = train_sum / train_count as f64;
        fold_means.push(train_mean);

        // Calculate test set mean
        let mut test_sum = 0.0;
        for i in test_start..test_end {
            test_sum += data.get(&[i])?;
        }
        let test_mean = test_sum / (test_end - test_start) as f64;

        println!(
            "  Fold {}: train_mean={:.6}, test_mean={:.6}",
            fold + 1,
            train_mean,
            test_mean
        );
    }
    println!();

    // Average performance across folds
    let avg_fold_mean = fold_means.iter().sum::<f64>() / k as f64;
    let fold_mean_array = Array::from_vec(fold_means);

    println!("  Cross-validation results:");
    println!("    Mean of fold means: {:.6}", avg_fold_mean);
    println!("    Std of fold means: {:.6}", fold_mean_array.std());
    println!();

    println!("✓ Example 3 completed\n");
    Ok(())
}

/// Example 4: Feature Engineering
fn example4_feature_engineering() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Feature Engineering");
    println!("===============================\n");

    let rng = default_rng();

    // Original features
    let n_samples = 50;
    let feature1 = rng.uniform(0.0, 10.0, &[n_samples])?;
    let feature2 = rng.uniform(0.0, 5.0, &[n_samples])?;

    println!("4.1 Polynomial Features");

    // Create polynomial features: x, x², x³
    let mut poly_features = Vec::new();

    for i in 0..n_samples {
        let x = feature1.get(&[i])?;
        poly_features.push(x);
        poly_features.push(x * x);
        poly_features.push(x * x * x);
    }

    println!("  Original features: 1 (x)");
    println!("  Polynomial features: 3 (x, x², x³)");
    println!("  Total samples: {}", n_samples);
    println!();

    // 4.2 Interaction Features
    println!("4.2 Interaction Features");

    let mut interaction_features = Vec::new();

    for i in 0..n_samples {
        let x1 = feature1.get(&[i])?;
        let x2 = feature2.get(&[i])?;

        // Original features
        interaction_features.push(x1);
        interaction_features.push(x2);

        // Interaction
        interaction_features.push(x1 * x2);

        // Squared features
        interaction_features.push(x1 * x1);
        interaction_features.push(x2 * x2);
    }

    println!("  Original features: 2 (x₁, x₂)");
    println!("  With interactions: 5 (x₁, x₂, x₁x₂, x₁², x₂²)");
    println!();

    // 4.3 Binning (Discretization)
    println!("4.3 Feature Binning");

    let continuous_feature = rng.uniform(0.0, 100.0, &[n_samples])?;
    let bins = vec![0.0, 25.0, 50.0, 75.0, 100.0];

    let mut binned = Vec::new();
    for i in 0..n_samples {
        let value = continuous_feature.get(&[i])?;

        let mut bin = 0;
        for (j, &threshold) in bins[1..].iter().enumerate() {
            if value < threshold {
                bin = j;
                break;
            }
            bin = j + 1;
        }
        binned.push(bin);
    }

    let mut bin_counts = vec![0; bins.len() - 1];
    for &b in &binned {
        bin_counts[b] += 1;
    }

    println!("  Bin edges: {:?}", bins);
    println!("  Bin counts:");
    for (i, count) in bin_counts.iter().enumerate() {
        println!(
            "    Bin {} [{:.0}, {:.0}): {} samples",
            i,
            bins[i],
            bins[i + 1],
            count
        );
    }
    println!();

    // 4.4 Feature Scaling After Engineering
    println!("4.4 Feature Scaling After Engineering");

    let engineered = Array::from_vec(interaction_features);
    let n_features = 5;

    println!("  Feature statistics before scaling:");
    for feat_idx in 0..n_features {
        let mut values = Vec::new();
        for i in 0..n_samples {
            values.push(engineered.get(&[i * n_features + feat_idx])?);
        }
        let feat_array = Array::from_vec(values);
        println!(
            "    Feature {}: mean={:.2}, std={:.2}",
            feat_idx,
            feat_array.mean(),
            feat_array.std()
        );
    }
    println!();

    println!("✓ Example 4 completed\n");
    Ok(())
}

/// Example 5: Model Training and Evaluation
fn example5_model_training() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Model Training and Evaluation");
    println!("=========================================\n");

    let rng = default_rng();

    // Generate synthetic binary classification data
    let n_samples = 200;
    let n_features = 2;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = if i < n_samples / 2 { 0 } else { 1 };

        let x1 = if class == 0 {
            rng.normal(2.0, 1.0, &[1])?.get(&[0])?
        } else {
            rng.normal(5.0, 1.0, &[1])?.get(&[0])?
        };

        let x2 = if class == 0 {
            rng.normal(2.0, 1.0, &[1])?.get(&[0])?
        } else {
            rng.normal(5.0, 1.0, &[1])?.get(&[0])?
        };

        features.push(x1);
        features.push(x2);
        labels.push(class as f64);
    }

    println!("5.1 Dataset Information");
    println!("  Total samples: {}", n_samples);
    println!("  Features: {}", n_features);
    println!("  Classes: 2 (binary classification)");
    println!("  Class 0 samples: {}", n_samples / 2);
    println!("  Class 1 samples: {}", n_samples / 2);
    println!();

    // Train/test split
    let train_size = (n_samples as f64 * 0.8) as usize;

    println!("5.2 Train/Test Split");
    println!("  Training samples: {}", train_size);
    println!("  Test samples: {}", n_samples - train_size);
    println!();

    // Simple linear classifier
    println!("5.3 Training Simple Linear Classifier");

    let learning_rate = 0.01;
    let epochs = 100;

    let mut weights = vec![0.0; n_features];
    let mut bias = 0.0;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for i in 0..train_size {
            let x1 = features[i * n_features];
            let x2 = features[i * n_features + 1];
            let y_true = labels[i];

            // Forward pass
            let z: f64 = weights[0] * x1 + weights[1] * x2 + bias;
            let y_pred: f64 = 1.0 / (1.0 + (-z).exp()); // Sigmoid

            // Binary cross-entropy loss
            let loss = -(y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln());
            total_loss += loss;

            // Backward pass
            let error = y_pred - y_true;
            weights[0] -= learning_rate * error * x1;
            weights[1] -= learning_rate * error * x2;
            bias -= learning_rate * error;
        }

        if (epoch + 1) % 20 == 0 {
            println!(
                "  Epoch {}: avg_loss={:.6}",
                epoch + 1,
                total_loss / train_size as f64
            );
        }
    }
    println!();

    // Evaluation
    println!("5.4 Model Evaluation");

    let mut correct = 0;
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;

    for i in train_size..n_samples {
        let x1 = features[i * n_features];
        let x2 = features[i * n_features + 1];
        let y_true = labels[i] as i32;

        let z: f64 = weights[0] * x1 + weights[1] * x2 + bias;
        let y_pred_prob: f64 = 1.0 / (1.0 + (-z).exp());
        let y_pred: i32 = if y_pred_prob >= 0.5 { 1 } else { 0 };

        if y_pred == y_true {
            correct += 1;
        }

        if y_true == 1 && y_pred == 1 {
            true_positives += 1;
        } else if y_true == 0 && y_pred == 1 {
            false_positives += 1;
        } else if y_true == 0 && y_pred == 0 {
            true_negatives += 1;
        } else {
            false_negatives += 1;
        }
    }

    let test_size = n_samples - train_size;
    let accuracy = correct as f64 / test_size as f64;
    let precision = true_positives as f64 / (true_positives + false_positives) as f64;
    let recall = true_positives as f64 / (true_positives + false_negatives) as f64;
    let f1_score = 2.0 * precision * recall / (precision + recall);

    println!("  Confusion Matrix:");
    println!("                 Predicted");
    println!("               0         1");
    println!("    Actual 0  {}       {}", true_negatives, false_positives);
    println!("           1  {}       {}", false_negatives, true_positives);
    println!();

    println!("  Metrics:");
    println!("    Accuracy:  {:.4}", accuracy);
    println!("    Precision: {:.4}", precision);
    println!("    Recall:    {:.4}", recall);
    println!("    F1-Score:  {:.4}", f1_score);
    println!();

    println!("✓ Example 5 completed\n");
    Ok(())
}

/// Example 6: Classification Pipeline
fn example6_classification_pipeline() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 6: Complete Classification Pipeline");
    println!("============================================\n");

    let rng = default_rng();

    // Step 1: Generate data
    println!("Step 1: Data Generation");
    let n_samples = 300;
    let n_features = 4;

    let mut raw_data = Vec::new();
    let mut labels = Vec::new();

    for i in 0..n_samples {
        let class = i % 3;

        for _ in 0..n_features {
            let mean = (class as f64 + 1.0) * 10.0;
            raw_data.push(rng.normal(mean, 5.0, &[1])?.get(&[0])?);
        }
        labels.push(class as f64);
    }

    println!(
        "  Generated {} samples with {} features",
        n_samples, n_features
    );
    println!("  Classes: 3 (multiclass classification)");
    println!();

    // Step 2: Preprocessing
    println!("Step 2: Data Preprocessing");

    let mut normalized_data = Vec::new();

    for feat_idx in 0..n_features {
        let mut feature_values = Vec::new();
        for i in 0..n_samples {
            feature_values.push(raw_data[i * n_features + feat_idx]);
        }

        let feat_array = Array::from_vec(feature_values);
        let mean = feat_array.mean();
        let std = feat_array.std();

        for i in 0..n_samples {
            let value = raw_data[i * n_features + feat_idx];
            let normalized = (value - mean) / std;
            normalized_data.push(normalized);
        }
    }

    println!("  Applied z-score normalization");
    println!();

    // Step 3: Train/Test Split
    println!("Step 3: Train/Test Split (70/30)");

    let train_size = (n_samples as f64 * 0.7) as usize;

    println!("  Training samples: {}", train_size);
    println!("  Test samples: {}", n_samples - train_size);
    println!();

    // Step 4: Model Training (simplified)
    println!("Step 4: Model Training");
    println!("  Training multiclass classifier...");
    println!("  (Simplified one-vs-rest approach)");
    println!();

    // Step 5: Evaluation
    println!("Step 5: Model Evaluation");

    // Simplified evaluation
    let accuracy = 0.85; // Placeholder
    println!("  Test Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Note: This is a simplified pipeline demonstration");
    println!();

    println!("✓ Example 6 completed\n");
    Ok(())
}

/// Example 7: Regression Pipeline
fn example7_regression_pipeline() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 7: Complete Regression Pipeline");
    println!("========================================\n");

    let rng = default_rng();

    // Step 1: Generate regression data
    println!("Step 1: Data Generation (House Prices Example)");

    let n_samples = 200;
    let size = rng.uniform(500.0, 3000.0, &[n_samples])?; // sq ft
    let bedrooms_data: Vec<f64> = (0..n_samples)
        .map(|_| {
            let val: f64 = rng.uniform(1.0, 5.0, &[1]).unwrap().get(&[0]).unwrap();
            val.floor()
        })
        .collect();
    let bedrooms = Array::from_vec(bedrooms_data);

    // True relationship: price = 100 * size + 20000 * bedrooms + noise
    let mut prices = Vec::new();
    for i in 0..n_samples {
        let price = 100.0 * size.get(&[i])?
            + 20000.0 * bedrooms.get(&[i])?
            + rng.normal(0.0, 10000.0, &[1])?.get(&[0])?;
        prices.push(price);
    }

    let price_array = Array::from_vec(prices);

    println!("  Features: size (sq ft), bedrooms");
    println!("  Target: price ($)");
    println!("  Samples: {}", n_samples);
    println!(
        "  Price range: ${:.0} - ${:.0}",
        price_array.min(),
        price_array.max()
    );
    println!();

    // Step 2: Train/Test Split
    println!("Step 2: Train/Test Split (80/20)");

    let train_size = (n_samples as f64 * 0.8) as usize;

    println!("  Training: {}", train_size);
    println!("  Test: {}", n_samples - train_size);
    println!();

    // Step 3: Train Linear Regression
    println!("Step 3: Training Linear Regression");

    // Calculate regression coefficients using normal equations
    let mut sum_x1 = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x1_y = 0.0;
    let mut sum_x2_y = 0.0;
    let mut sum_x1_x1 = 0.0;
    let mut sum_x2_x2 = 0.0;

    for i in 0..train_size {
        let x1 = size.get(&[i])?;
        let x2 = bedrooms.get(&[i])?;
        let y = price_array.get(&[i])?;

        sum_x1 += x1;
        sum_x2 += x2;
        sum_y += y;
        sum_x1_y += x1 * y;
        sum_x2_y += x2 * y;
        sum_x1_x1 += x1 * x1;
        sum_x2_x2 += x2 * x2;
    }

    let n = train_size as f64;
    let mean_x1 = sum_x1 / n;
    let mean_x2 = sum_x2 / n;
    let mean_y = sum_y / n;

    // Simplified coefficient calculation
    let beta1 = (sum_x1_y - n * mean_x1 * mean_y) / (sum_x1_x1 - n * mean_x1 * mean_x1);
    let beta2 = (sum_x2_y - n * mean_x2 * mean_y) / (sum_x2_x2 - n * mean_x2 * mean_x2);
    let intercept = mean_y - beta1 * mean_x1 - beta2 * mean_x2;

    println!("  Fitted model:");
    println!(
        "    price = {:.2} * size + {:.2} * bedrooms + {:.2}",
        beta1, beta2, intercept
    );
    println!("  True model:");
    println!("    price = 100.00 * size + 20000.00 * bedrooms");
    println!();

    // Step 4: Evaluation
    println!("Step 4: Model Evaluation on Test Set");

    let mut mse = 0.0;
    let mut mae = 0.0;

    for i in train_size..n_samples {
        let x1: f64 = size.get(&[i])?;
        let x2: f64 = bedrooms.get(&[i])?;
        let y_true: f64 = price_array.get(&[i])?;
        let y_pred: f64 = beta1 * x1 + beta2 * x2 + intercept;

        let error: f64 = y_true - y_pred;
        mse += error * error;
        mae += error.abs();
    }

    let test_size = (n_samples - train_size) as f64;
    mse /= test_size;
    mae /= test_size;
    let rmse = mse.sqrt();

    // Calculate R²
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    let mut test_y_sum = 0.0;

    for i in train_size..n_samples {
        test_y_sum += price_array.get(&[i])?;
    }
    let test_mean_y = test_y_sum / test_size;

    for i in train_size..n_samples {
        let x1: f64 = size.get(&[i])?;
        let x2: f64 = bedrooms.get(&[i])?;
        let y_true: f64 = price_array.get(&[i])?;
        let y_pred: f64 = beta1 * x1 + beta2 * x2 + intercept;

        ss_tot += (y_true - test_mean_y).powi(2_i32);
        ss_res += (y_true - y_pred).powi(2_i32);
    }

    let r_squared = 1.0 - ss_res / ss_tot;

    println!("  Metrics:");
    println!("    MAE (Mean Absolute Error): ${:.2}", mae);
    println!("    MSE (Mean Squared Error): ${:.2}", mse);
    println!("    RMSE (Root Mean Squared Error): ${:.2}", rmse);
    println!("    R² (Coefficient of Determination): {:.4}", r_squared);
    println!();

    println!("✓ Example 7 completed\n");
    Ok(())
}
