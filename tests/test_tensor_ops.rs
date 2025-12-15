use num_traits::Float;
use numrs2::linalg::tensor_ops::{kron, tensordot};
use numrs2::prelude::*;

#[test]
fn test_kron_basic() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    let result = kron(&a, &b).unwrap();

    // Expected result: 4x4 matrix
    // [1*5, 1*6, 2*5, 2*6]   [5, 6, 10, 12]
    // [1*7, 1*8, 2*7, 2*8] = [7, 8, 14, 16]
    // [3*5, 3*6, 4*5, 4*6]   [15, 18, 20, 24]
    // [3*7, 3*8, 4*7, 4*8]   [21, 24, 28, 32]

    assert_eq!(result.shape(), &[4, 4]);
    let expected = [
        5.0, 6.0, 10.0, 12.0, 7.0, 8.0, 14.0, 16.0, 15.0, 18.0, 20.0, 24.0, 21.0, 24.0, 28.0, 32.0,
    ];

    let result_data = result.to_vec();
    for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            Float::abs(actual - expected) < 1e-10f64,
            "Mismatch at index {}: {} != {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_kron_identity() {
    let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]); // identity
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    let result = kron(&a, &b).unwrap();

    // Expected result: block diagonal matrix
    // [5, 6, 0, 0]
    // [7, 8, 0, 0]
    // [0, 0, 5, 6]
    // [0, 0, 7, 8]

    assert_eq!(result.shape(), &[4, 4]);
    let expected = [
        5.0, 6.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0,
    ];

    let result_data = result.to_vec();
    for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            Float::abs(actual - expected) < 1e-10f64,
            "Mismatch at index {}: {} != {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_tensordot_matrix_multiplication() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    // Contract axis 1 of a with axis 0 of b - this is matrix multiplication
    let result = tensordot(&a, &b, &[1, 0]).unwrap();

    // Expected result is the same as matrix multiplication a @ b
    let expected = a.matmul(&b).unwrap();

    assert_eq!(result.shape(), expected.shape());
    let result_data = result.to_vec();
    let expected_data = expected.to_vec();

    for (i, (&actual, &expected)) in result_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            Float::abs(actual - expected) < 1e-10f64,
            "Mismatch at index {}: {} != {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_tensordot_error_handling() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0]).reshape(&[3, 1]);

    // This should fail because the contracted dimensions don't match
    let result = tensordot(&a, &b, &[1, 0]);
    assert!(result.is_err());

    // Test with wrong number of axes
    let result = tensordot(&a, &b, &[1, 0, 1]);
    assert!(result.is_err());
}

#[test]
fn test_kron_error_handling() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3]); // 1D array
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    // This should fail because a is not 2D
    let result = kron(&a, &b);
    assert!(result.is_err());
}
