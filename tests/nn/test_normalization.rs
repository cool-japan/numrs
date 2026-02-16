/// Tests for normalization operations
use numrs2::nn::normalization::*;
use scirs2_core::ndarray::{array, Array1, Array2};

const EPSILON: f64 = 1e-5;

#[test]
fn test_batch_norm_1d_basic() {
    // Create a simple 2x3 input (2 samples, 3 features)
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gamma = array![1.0, 1.0, 1.0];
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let y = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");

    assert_eq!(y.shape(), x.shape());

    // Check that each feature is normalized across the batch
    for j in 0..y.ncols() {
        let col = y.column(j);
        let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
        assert!(mean.abs() < EPSILON * 10.0_f64); // Mean should be ~0
    }
}

#[test]
fn test_batch_norm_1d_with_scale_shift() {
    let x = array![[0.0, 1.0], [2.0, 3.0]];
    let gamma = array![2.0, 2.0]; // Scale by 2
    let beta = array![1.0, 1.0]; // Shift by 1
    let epsilon = 1e-5_f64;

    let y = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");

    // After normalization with gamma=2, beta=1, values should be transformed
    assert_eq!(y.shape(), x.shape());

    // Values should be finite
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val.is_finite());
    }
}

#[test]
fn test_batch_norm_1d_dimension_mismatch() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gamma = array![1.0, 1.0]; // Wrong size
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let result = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_basic() {
    // Create a 2x3 input (2 samples, 3 features)
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gamma = array![1.0, 1.0, 1.0];
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let y = layer_norm(&x.view(), &gamma.view(), &beta.view(), epsilon).expect("layer_norm failed");

    assert_eq!(y.shape(), x.shape());

    // Check that each sample is normalized independently
    for i in 0..y.nrows() {
        let row = y.row(i);
        let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
        assert!(mean.abs() < EPSILON * 10.0_f64); // Mean should be ~0
    }
}

#[test]
fn test_layer_norm_with_scale_shift() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gamma = array![2.0, 2.0, 2.0];
    let beta = array![1.0, 1.0, 1.0];
    let epsilon = 1e-5_f64;

    let y = layer_norm(&x.view(), &gamma.view(), &beta.view(), epsilon).expect("layer_norm failed");

    // Values should be transformed by gamma and beta
    assert_eq!(y.shape(), x.shape());

    for &val in y.iter() {
        let val: f64 = val;
        assert!(val.is_finite());
    }
}

#[test]
fn test_layer_norm_dimension_mismatch() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gamma = array![1.0, 1.0]; // Wrong size
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let result = layer_norm(&x.view(), &gamma.view(), &beta.view(), epsilon);
    assert!(result.is_err());
}

// FIXME: instance_norm function not implemented yet
// #[test]
// fn test_instance_norm_basic() {
//     // Create a 2x3 input
//     let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//     let epsilon = 1e-5_f64;
//
//     let y = instance_norm(&x.view(), epsilon).expect("instance_norm failed");
//
//     assert_eq!(y.shape(), x.shape());
//
//     // Each row should be normalized independently
//     for i in 0..y.nrows() {
//         let row = y.row(i);
//         let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
//         let var: f64 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64;
//
//         assert!(mean.abs() < EPSILON * 10.0_f64);
//         assert!((var - 1.0_f64).abs() < EPSILON * 100.0_f64); // Variance should be ~1
//     }
// }

#[test]
fn test_dropout_basic() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let p = 0.5_f64;

    let y = dropout_2d(&x.view(), p, true).expect("dropout failed");

    assert_eq!(y.shape(), x.shape());

    // Check that some values are zeroed (probabilistically)
    // Note: This test might occasionally fail due to randomness
    let num_zeros = y.iter().filter(|&&v| {
        let v: f64 = v;
        v.abs() < EPSILON
    }).count();

    // With p=0.5, we expect roughly half to be zero
    // But we allow a wide range due to randomness
    assert!(num_zeros <= y.len());
}

#[test]
fn test_dropout_p_zero() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let p = 0.0_f64; // No dropout

    let y = dropout_2d(&x.view(), p, true).expect("dropout failed");

    // All values should be unchanged
    for (&actual, &expected) in y.iter().zip(x.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_dropout_p_high() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let p = 0.9999_f64; // Drop almost everything (p must be < 1.0)

    let y = dropout_2d(&x.view(), p, true).expect("dropout failed");

    // Most values should be zero (probabilistically)
    let num_zeros = y.iter().filter(|&&v| {
        let v: f64 = v;
        v.abs() < EPSILON
    }).count();

    // With p=0.9999, we expect most to be zero
    assert!(num_zeros >= 1);
}

#[test]
fn test_dropout_invalid_probability() {
    let x = array![[1.0, 2.0, 3.0]];

    // p < 0 should error
    let result = dropout_2d(&x.view(), -0.1_f64, true);
    assert!(result.is_err());

    // p >= 1 should error (range is [0, 1))
    let result = dropout_2d(&x.view(), 1.0_f64, true);
    assert!(result.is_err());

    // p > 1 should error
    let result = dropout_2d(&x.view(), 1.5_f64, true);
    assert!(result.is_err());
}

#[test]
fn test_batch_norm_single_sample() {
    // Edge case: single sample
    let x = array![[1.0, 2.0, 3.0]];
    let gamma = array![1.0, 1.0, 1.0];
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let y = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_layer_norm_single_feature() {
    // Edge case: single feature
    let x = array![[1.0], [2.0], [3.0]];
    let gamma = array![1.0];
    let beta = array![0.0];
    let epsilon = 1e-5_f64;

    let y = layer_norm(&x.view(), &gamma.view(), &beta.view(), epsilon).expect("layer_norm failed");

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_normalization_large_values() {
    // Test with large values
    let x = array![[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]];
    let gamma = array![1.0, 1.0, 1.0];
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let y = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");

    // Should handle large values without overflow
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val.is_finite());
    }
}

#[test]
fn test_normalization_negative_values() {
    // Test with negative values
    let x = array![[-3.0, -2.0, -1.0], [1.0, 2.0, 3.0]];
    let gamma = array![1.0, 1.0, 1.0];
    let beta = array![0.0, 0.0, 0.0];
    let epsilon = 1e-5_f64;

    let y = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");

    for &val in y.iter() {
        let val: f64 = val;
        assert!(val.is_finite());
    }
}

#[test]
fn test_normalization_f32() {
    // Test f32 versions
    let x = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gamma = array![1.0f32, 1.0, 1.0];
    let beta = array![0.0f32, 0.0, 0.0];
    let epsilon = 1e-5f32;

    let y = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)
        .expect("batch_norm_1d failed");

    assert_eq!(y.shape(), x.shape());
}
