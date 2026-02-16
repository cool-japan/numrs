/// Tests for pooling operations
use numrs2::nn::pooling::*;
use scirs2_core::ndarray::array;

const EPSILON: f64 = 1e-6;

#[test]
fn test_max_pool2d_basic() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ];
    let pool_size = (2, 2);
    let stride = (2, 2);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // Output shape should be (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = max(1,2,5,6) = 6
    assert!((output[[0, 0]] - 6.0_f64).abs() < EPSILON);

    // output[0,1] = max(3,4,7,8) = 8
    assert!((output[[0, 1]] - 8.0_f64).abs() < EPSILON);

    // output[1,0] = max(9,10,13,14) = 14
    assert!((output[[1, 0]] - 14.0_f64).abs() < EPSILON);

    // output[1,1] = max(11,12,15,16) = 16
    assert!((output[[1, 1]] - 16.0_f64).abs() < EPSILON);
}

#[test]
fn test_max_pool2d_stride_1() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // Output shape should be (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = max(1,2,4,5) = 5
    assert!((output[[0, 0]] - 5.0_f64).abs() < EPSILON);

    // output[0,1] = max(2,3,5,6) = 6
    assert!((output[[0, 1]] - 6.0_f64).abs() < EPSILON);

    // output[1,0] = max(4,5,7,8) = 8
    assert!((output[[1, 0]] - 8.0_f64).abs() < EPSILON);

    // output[1,1] = max(5,6,8,9) = 9
    assert!((output[[1, 1]] - 9.0_f64).abs() < EPSILON);
}

#[test]
fn test_max_pool2d_zero_pool_size_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let pool_size = (0, 2);
    let stride = (1, 1);

    let result = max_pool2d(&x.view(), pool_size, stride);
    assert!(result.is_err());
}

#[test]
fn test_max_pool2d_zero_stride_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let pool_size = (2, 2);
    let stride = (0, 1);

    let result = max_pool2d(&x.view(), pool_size, stride);
    assert!(result.is_err());
}

#[test]
fn test_avg_pool2d_basic() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ];
    let pool_size = (2, 2);
    let stride = (2, 2);

    let output = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");

    // Output shape should be (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = avg(1,2,5,6) = 3.5
    assert!((output[[0, 0]] - 3.5_f64).abs() < EPSILON);

    // output[0,1] = avg(3,4,7,8) = 5.5
    assert!((output[[0, 1]] - 5.5_f64).abs() < EPSILON);

    // output[1,0] = avg(9,10,13,14) = 11.5
    assert!((output[[1, 0]] - 11.5_f64).abs() < EPSILON);

    // output[1,1] = avg(11,12,15,16) = 13.5
    assert!((output[[1, 1]] - 13.5_f64).abs() < EPSILON);
}

#[test]
fn test_avg_pool2d_stride_1() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");

    // Output shape should be (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = avg(1,2,4,5) = 3.0
    assert!((output[[0, 0]] - 3.0_f64).abs() < EPSILON);

    // output[0,1] = avg(2,3,5,6) = 4.0
    assert!((output[[0, 1]] - 4.0_f64).abs() < EPSILON);

    // output[1,0] = avg(4,5,7,8) = 6.0
    assert!((output[[1, 0]] - 6.0_f64).abs() < EPSILON);

    // output[1,1] = avg(5,6,8,9) = 7.0
    assert!((output[[1, 1]] - 7.0_f64).abs() < EPSILON);
}

#[test]
fn test_avg_pool2d_zero_pool_size_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let pool_size = (2, 0);
    let stride = (1, 1);

    let result = avg_pool2d(&x.view(), pool_size, stride);
    assert!(result.is_err());
}

#[test]
fn test_avg_pool2d_zero_stride_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let pool_size = (2, 2);
    let stride = (1, 0);

    let result = avg_pool2d(&x.view(), pool_size, stride);
    assert!(result.is_err());
}

#[test]
fn test_max_pool2d_single_element_pool() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let pool_size = (1, 1);
    let stride = (1, 1);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // Should be identity operation
    assert_eq!(output.shape(), x.shape());
    for (&actual, &expected) in output.iter().zip(x.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_avg_pool2d_single_element_pool() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let pool_size = (1, 1);
    let stride = (1, 1);

    let output = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");

    // Should be identity operation
    assert_eq!(output.shape(), x.shape());
    for (&actual, &expected) in output.iter().zip(x.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_max_pool2d_negative_values() {
    let x = array![[-5.0, -2.0], [-3.0, -1.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // max(-5, -2, -3, -1) = -1
    assert!((output[[0, 0]] - (-1.0_f64)).abs() < EPSILON);
}

#[test]
fn test_avg_pool2d_negative_values() {
    let x = array![[-4.0, -2.0], [-6.0, -8.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");

    // avg(-4, -2, -6, -8) = -5.0
    assert!((output[[0, 0]] - (-5.0_f64)).abs() < EPSILON);
}

#[test]
fn test_max_pool2d_mixed_sign_values() {
    let x = array![[-1.0, 2.0], [3.0, -4.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // max(-1, 2, 3, -4) = 3
    assert!((output[[0, 0]] - 3.0_f64).abs() < EPSILON);
}

#[test]
fn test_max_pool2d_non_square_pool() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let pool_size = (2, 3);
    let stride = (1, 1);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // Output shape should be (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = max(1,2,3,5,6,7) = 7
    assert!((output[[0, 0]] - 7.0_f64).abs() < EPSILON);
}

#[test]
fn test_avg_pool2d_non_square_pool() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let pool_size = (2, 3);
    let stride = (1, 1);

    let output = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");

    // Output shape should be (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = avg(1,2,3,5,6,7) = 4.0
    assert!((output[[0, 0]] - 4.0_f64).abs() < EPSILON);
}

#[test]
fn test_pooling_f32() {
    // Test f32 versions
    let x = array![[1.0f32, 2.0], [3.0, 4.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let max_out = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");
    assert!((max_out[[0, 0]] - 4.0_f32).abs() < 1e-6_f32);

    let avg_out = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");
    assert!((avg_out[[0, 0]] - 2.5_f32).abs() < 1e-6_f32);
}

#[test]
fn test_global_avg_pool() {
    // Test global average pooling (pool size = input size)
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = avg_pool2d(&x.view(), pool_size, stride).expect("avg_pool2d failed");

    // Should compute global average
    assert_eq!(output.shape(), &[1, 1]);
    assert!((output[[0, 0]] - 2.5_f64).abs() < EPSILON);
}

#[test]
fn test_max_pool_large_values() {
    let x = array![[1000.0, 2000.0], [3000.0, 4000.0]];
    let pool_size = (2, 2);
    let stride = (1, 1);

    let output = max_pool2d(&x.view(), pool_size, stride).expect("max_pool2d failed");

    // Should handle large values
    assert!((output[[0, 0]] - 4000.0_f64).abs() < EPSILON);
}
