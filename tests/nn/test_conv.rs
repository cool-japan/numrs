/// Tests for convolution operations
use numrs2::nn::conv::*;
use scirs2_core::ndarray::{array, Array1, Array2};

const EPSILON: f64 = 1e-6;

#[test]
fn test_conv1d_basic() {
    let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = array![1.0, 0.0, -1.0];
    let stride = 1;

    let output = conv1d(&input.view(), &kernel.view(), stride).expect("conv1d failed");

    // Output length should be (5 - 3) / 1 + 1 = 3
    assert_eq!(output.len(), 3);

    // Manual calculation:
    // output[0] = 1*1 + 2*0 + 3*(-1) = 1 - 3 = -2
    // output[1] = 2*1 + 3*0 + 4*(-1) = 2 - 4 = -2
    // output[2] = 3*1 + 4*0 + 5*(-1) = 3 - 5 = -2

    assert!((output[0] - (-2.0_f64)).abs() < EPSILON);
    assert!((output[1] - (-2.0_f64)).abs() < EPSILON);
    assert!((output[2] - (-2.0_f64)).abs() < EPSILON);
}

#[test]
fn test_conv1d_stride_2() {
    let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = array![1.0, 1.0];
    let stride = 2;

    let output = conv1d(&input.view(), &kernel.view(), stride).expect("conv1d failed");

    // Output length should be (5 - 2) / 2 + 1 = 2
    assert_eq!(output.len(), 2);

    // output[0] = 1*1 + 2*1 = 3
    // output[1] = 3*1 + 4*1 = 7
    assert!((output[0] - 3.0_f64).abs() < EPSILON);
    assert!((output[1] - 7.0_f64).abs() < EPSILON);
}

#[test]
fn test_conv1d_empty_kernel_error() {
    let input = array![1.0, 2.0, 3.0];
    let kernel: Array1<f64> = Array1::zeros(0);
    let stride = 1;

    let result = conv1d(&input.view(), &kernel.view(), stride);
    assert!(result.is_err());
}

#[test]
fn test_conv1d_zero_stride_error() {
    let input = array![1.0, 2.0, 3.0];
    let kernel = array![1.0, 1.0];
    let stride = 0;

    let result = conv1d(&input.view(), &kernel.view(), stride);
    assert!(result.is_err());
}

#[test]
fn test_conv1d_kernel_larger_than_input() {
    let input = array![1.0, 2.0];
    let kernel = array![1.0, 1.0, 1.0];
    let stride = 1;

    let result = conv1d(&input.view(), &kernel.view(), stride);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_basic() {
    let input = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    let kernel = array![[1.0, 0.0], [0.0, 1.0]];
    let stride = (1, 1);

    let output = conv2d(&input.view(), &kernel.view(), stride).expect("conv2d failed");

    // Output shape should be ((3-2)/1+1, (3-2)/1+1) = (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // Manual calculation for output[0,0]:
    // 1*1 + 2*0 + 4*0 + 5*1 = 1 + 5 = 6
    assert!((output[[0, 0]] - 6.0_f64).abs() < EPSILON);
}

#[test]
fn test_conv2d_stride_2() {
    let input = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ];
    let kernel = array![[1.0, 1.0], [1.0, 1.0]];
    let stride = (2, 2);

    let output = conv2d(&input.view(), &kernel.view(), stride).expect("conv2d failed");

    // Output shape should be ((4-2)/2+1, (4-2)/2+1) = (2, 2)
    assert_eq!(output.shape(), &[2, 2]);

    // output[0,0] = 1+2+5+6 = 14
    assert!((output[[0, 0]] - 14.0_f64).abs() < EPSILON);

    // output[0,1] = 3+4+7+8 = 22
    assert!((output[[0, 1]] - 22.0_f64).abs() < EPSILON);

    // output[1,0] = 9+10+13+14 = 46
    assert!((output[[1, 0]] - 46.0_f64).abs() < EPSILON);

    // output[1,1] = 11+12+15+16 = 54
    assert!((output[[1, 1]] - 54.0_f64).abs() < EPSILON);
}

#[test]
fn test_conv2d_empty_kernel_error() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let kernel: Array2<f64> = Array2::zeros((0, 0));
    let stride = (1, 1);

    let result = conv2d(&input.view(), &kernel.view(), stride);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_zero_stride_error() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let kernel = array![[1.0]];
    let stride = (0, 1);

    let result = conv2d(&input.view(), &kernel.view(), stride);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_kernel_larger_than_input() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let kernel = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    let stride = (1, 1);

    let result = conv2d(&input.view(), &kernel.view(), stride);
    assert!(result.is_err());
}

#[test]
fn test_conv1d_identity_kernel() {
    let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = array![1.0];
    let stride = 1;

    let output = conv1d(&input.view(), &kernel.view(), stride).expect("conv1d failed");

    // Identity kernel should preserve input
    assert_eq!(output.len(), input.len());
    for (&actual, &expected) in output.iter().zip(input.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_conv2d_identity_kernel() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let kernel = array![[1.0]];
    let stride = (1, 1);

    let output = conv2d(&input.view(), &kernel.view(), stride).expect("conv2d failed");

    // Identity kernel should preserve input
    assert_eq!(output.shape(), input.shape());
    for (&actual, &expected) in output.iter().zip(input.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_conv1d_average_filter() {
    let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = array![0.333333, 0.333333, 0.333333];
    let stride = 1;

    let output = conv1d(&input.view(), &kernel.view(), stride).expect("conv1d failed");

    // Average filter should smooth the signal
    // output[0] ≈ (1+2+3)/3 = 2.0
    assert!((output[0] - 2.0_f64).abs() < 0.01_f64);
    // output[1] ≈ (2+3+4)/3 = 3.0
    assert!((output[1] - 3.0_f64).abs() < 0.01_f64);
}

#[test]
fn test_conv2d_edge_detection() {
    let input = array![
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ];

    // Sobel-like horizontal edge detection
    let kernel = array![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let stride = (1, 1);

    let output = conv2d(&input.view(), &kernel.view(), stride).expect("conv2d failed");

    // Should detect vertical edges
    assert_eq!(output.shape(), &[3, 3]);
}

#[test]
fn test_conv_f32() {
    // Test f32 versions
    let input = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let kernel = array![1.0f32, 1.0];
    let stride = 1;

    let output = conv1d(&input.view(), &kernel.view(), stride).expect("conv1d failed");

    assert_eq!(output.len(), 4);
    assert!((output[0] - 3.0_f32).abs() < 1e-6_f32);
}
