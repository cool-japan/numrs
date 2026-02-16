/// Tests for loss functions
use numrs2::nn::loss::*;
use numrs2::nn::ReductionMode;
use scirs2_core::ndarray::array;

const EPSILON: f64 = 1e-6;

#[test]
fn test_mse_loss_basic() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.1, 2.2, 2.9];

    let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mse_loss failed");

    // MSE = mean((0.1^2 + 0.2^2 + 0.1^2)) = mean(0.01 + 0.04 + 0.01) = 0.02
    assert!((loss - 0.02_f64).abs() < EPSILON);
}

#[test]
fn test_mse_loss_zero() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 3.0];

    let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mse_loss failed");

    // Perfect prediction should have zero loss
    let loss_val: f64 = loss;
    assert!(loss_val.abs() < EPSILON);
}

#[test]
fn test_mse_loss_sum_reduction() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.1, 2.2, 2.9];

    let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Sum)
        .expect("mse_loss failed");

    // Sum of squared errors = 0.01 + 0.04 + 0.01 = 0.06
    assert!((loss - 0.06_f64).abs() < EPSILON);
}

#[test]
fn test_mse_loss_dimension_mismatch() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0];

    let result = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean);
    assert!(result.is_err());
}

#[test]
fn test_mae_loss_basic() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.5, 2.5, 2.5];

    let loss = mae_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mae_loss failed");

    // MAE = mean(|0.5| + |0.5| + |0.5|) = 0.5
    assert!((loss - 0.5_f64).abs() < EPSILON);
}

#[test]
fn test_mae_loss_zero() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 3.0];

    let loss = mae_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mae_loss failed");

    let loss_val: f64 = loss;
    assert!(loss_val.abs() < EPSILON);
}

#[test]
fn test_huber_loss_basic() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.1, 2.2, 2.9];
    let delta = 1.0_f64;

    let loss = huber_loss(&y_true.view(), &y_pred.view(), delta, ReductionMode::Mean)
        .expect("huber_loss failed");

    // All errors are < delta, so it behaves like MSE * 0.5
    // 0.5 * mean(0.1^2 + 0.2^2 + 0.1^2) = 0.5 * 0.02 = 0.01
    assert!((loss - 0.01_f64).abs() < EPSILON);
}

#[test]
fn test_huber_loss_large_error() {
    let y_true = array![0.0, 0.0];
    let y_pred = array![2.0, -2.0];
    let delta = 1.0_f64;

    let loss = huber_loss(&y_true.view(), &y_pred.view(), delta, ReductionMode::Mean)
        .expect("huber_loss failed");

    // Both errors are > delta
    // For error = 2: delta * (|2| - 0.5 * delta) = 1.0 * (2.0 - 0.5) = 1.5
    // Mean = (1.5 + 1.5) / 2 = 1.5
    assert!((loss - 1.5).abs() < EPSILON);
}

#[test]
fn test_binary_cross_entropy_basic() {
    let y_true = array![1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.1, 0.8];

    let loss = binary_cross_entropy(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("binary_cross_entropy failed");

    // BCE = -mean(y*log(p) + (1-y)*log(1-p))
    // Should be small since predictions are close to truth
    assert!(loss < 0.3);
    assert!(loss > 0.0);
}

#[test]
fn test_binary_cross_entropy_perfect_prediction() {
    let y_true = array![1.0, 0.0, 1.0];
    let y_pred = array![0.9999, 0.0001, 0.9999];

    let loss = binary_cross_entropy(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("binary_cross_entropy failed");

    // Should be very small
    assert!(loss < 0.01);
}

#[test]
fn test_focal_loss_basic() {
    let y_true = array![1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.1, 0.8];
    let alpha = 0.25_f64;
    let gamma = 2.0_f64;

    let loss = focal_loss(&y_true.view(), &y_pred.view(), alpha, gamma, ReductionMode::Mean)
        .expect("focal_loss failed");

    // Focal loss should be positive
    assert!(loss > 0.0);
}

#[test]
fn test_loss_2d_arrays() {
    let y_true = array![[1.0, 2.0], [3.0, 4.0]];
    let y_pred = array![[1.1, 2.1], [2.9, 3.9]];

    let loss = mse_loss_2d(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mse_loss_2d failed");

    // MSE for all 4 elements
    assert!(loss > 0.0);
}

#[test]
fn test_loss_f32() {
    // Test f32 versions
    let y_true = array![1.0f32, 2.0, 3.0];
    let y_pred = array![1.1f32, 2.2, 2.9];

    let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mse_loss failed");

    assert!((loss - 0.02).abs() < 1e-5);
}

#[test]
fn test_loss_large_values() {
    let y_true = array![1000.0, 2000.0, 3000.0];
    let y_pred = array![1001.0, 2001.0, 3001.0];

    let loss: f64 = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
        .expect("mse_loss failed");

    // Should handle large values without overflow
    assert!(loss.is_finite());
    assert!((loss - 1.0).abs() < EPSILON);
}
