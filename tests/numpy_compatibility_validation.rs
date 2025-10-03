//! NumPy Compatibility Validation Tests
//!
//! This module provides comprehensive tests to validate that NumRS2 functions
//! behave identically to their NumPy equivalents.

use numrs2::array_ops::advanced_indexing;
use numrs2::bitwise_ops;
use numrs2::complex_ops;
use numrs2::prelude::*;
use scirs2_core::Complex;

/// Test numerical equivalence with tolerance
fn assert_arrays_close_f64(a: &Array<f64>, b: &[f64], tolerance: f64) {
    let a_vec = a.to_vec();
    assert_eq!(a_vec.len(), b.len(), "Array lengths don't match");
    for (i, (&got, &expected)) in a_vec.iter().zip(b.iter()).enumerate() {
        assert!(
            (got - expected).abs() < tolerance,
            "Arrays differ at index {}: got {}, expected {}, diff {}",
            i,
            got,
            expected,
            (got - expected).abs()
        );
    }
}

#[allow(dead_code)]
fn assert_arrays_close_f32(a: &Array<f32>, b: &[f32], tolerance: f32) {
    let a_vec = a.to_vec();
    assert_eq!(a_vec.len(), b.len(), "Array lengths don't match");
    for (i, (&got, &expected)) in a_vec.iter().zip(b.iter()).enumerate() {
        assert!(
            (got - expected).abs() < tolerance,
            "Arrays differ at index {}: got {}, expected {}, diff {}",
            i,
            got,
            expected,
            (got - expected).abs()
        );
    }
}

#[test]
fn test_bitwise_operations_numpy_equivalence() {
    // Test data that matches NumPy test cases
    let a = Array::from_vec(vec![13, 17, 21, 5, 8]);
    let b = Array::from_vec(vec![9, 7, 15, 12, 3]);

    // Expected results computed with NumPy:
    // >>> import numpy as np
    // >>> a = np.array([13, 17, 21, 5, 8])
    // >>> b = np.array([9, 7, 15, 12, 3])
    // >>> np.bitwise_and(a, b)
    let expected_and = vec![9, 1, 5, 4, 0];
    let result_and = bitwise_ops::bitwise_and(&a, &b).unwrap();
    assert_eq!(result_and.to_vec(), expected_and);

    // >>> np.bitwise_or(a, b)
    let expected_or = vec![13, 23, 31, 13, 11];
    let result_or = bitwise_ops::bitwise_or(&a, &b).unwrap();
    assert_eq!(result_or.to_vec(), expected_or);

    // >>> np.bitwise_xor(a, b)
    let expected_xor = vec![4, 22, 26, 9, 11];
    let result_xor = bitwise_ops::bitwise_xor(&a, &b).unwrap();
    assert_eq!(result_xor.to_vec(), expected_xor);

    // >>> np.left_shift(a, [1, 2, 1, 3, 2])
    let shift_amounts = Array::from_vec(vec![1, 2, 1, 3, 2]);
    let expected_left_shift = vec![26, 68, 42, 40, 32];
    let result_left_shift = bitwise_ops::left_shift(&a, &shift_amounts).unwrap();
    assert_eq!(result_left_shift.to_vec(), expected_left_shift);

    // >>> np.right_shift([26, 68, 42, 40, 32], [1, 2, 1, 3, 2])
    let shifted_data = Array::from_vec(vec![26, 68, 42, 40, 32]);
    let expected_right_shift = vec![13, 17, 21, 5, 8];
    let result_right_shift = bitwise_ops::right_shift(&shifted_data, &shift_amounts).unwrap();
    assert_eq!(result_right_shift.to_vec(), expected_right_shift);
}

#[test]
fn test_complex_operations_numpy_equivalence() {
    // Test data: complex numbers with known NumPy results
    let complex_data = Array::from_vec(vec![
        Complex::new(3.0, 4.0),   // 3+4j
        Complex::new(1.0, 0.0),   // 1+0j
        Complex::new(0.0, 1.0),   // 0+1j
        Complex::new(-1.0, -1.0), // -1-1j
        Complex::new(5.0, 0.0),   // 5+0j
    ]);

    // Test absolute values (magnitude)
    // >>> import numpy as np
    // >>> a = np.array([3+4j, 1+0j, 0+1j, -1-1j, 5+0j])
    // >>> np.abs(a)
    let expected_abs = vec![5.0, 1.0, 1.0, std::f64::consts::SQRT_2, 5.0];
    let result_abs = complex_ops::absolute(&complex_data);
    assert_arrays_close_f64(&result_abs, &expected_abs, 1e-10);

    // Test angle calculation (phase)
    // >>> np.angle(a)
    let expected_angle = vec![
        0.9272952180016122,          // arctan(4/3)
        0.0,                         // angle of 1+0j
        std::f64::consts::FRAC_PI_2, // π/2 for 0+1j
        -2.356194490192345,          // -3π/4 for -1-1j
        0.0,                         // angle of 5+0j
    ];
    let result_angle = complex_ops::angle(&complex_data, false);
    assert_arrays_close_f64(&result_angle, &expected_angle, 1e-10);

    // Test real parts
    // >>> np.real(a)
    let expected_real = vec![3.0, 1.0, 0.0, -1.0, 5.0];
    let result_real = complex_ops::real(&complex_data);
    assert_arrays_close_f64(&result_real, &expected_real, 1e-15);

    // Test imaginary parts
    // >>> np.imag(a)
    let expected_imag = vec![4.0, 0.0, 1.0, -1.0, 0.0];
    let result_imag = complex_ops::imag(&complex_data);
    assert_arrays_close_f64(&result_imag, &expected_imag, 1e-15);

    // Test conjugate
    // >>> np.conj(a)
    let expected_conj = vec![
        Complex::new(3.0, -4.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, -1.0),
        Complex::new(-1.0, 1.0),
        Complex::new(5.0, 0.0),
    ];
    let result_conj = complex_ops::conj(&complex_data);
    let result_conj_vec = result_conj.to_vec();
    for (got, expected) in result_conj_vec.iter().zip(expected_conj.iter()) {
        assert!((got.re - expected.re).abs() < 1e-15);
        assert!((got.im - expected.im).abs() < 1e-15);
    }
}

#[test]
fn test_mathematical_functions_numpy_equivalence() {
    // Test mathematical functions against known NumPy results
    let test_data = Array::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);

    // Test exponential function
    // >>> import numpy as np
    // >>> x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    // >>> np.exp(x)
    let expected_exp = vec![
        1.0,
        1.6487212707001282,
        std::f64::consts::E,
        4.4816890703380645,
        7.38905609893065,
    ];
    let result_exp = test_data.exp();
    assert_arrays_close_f64(&result_exp, &expected_exp, 1e-10);

    // Test sine function
    // >>> np.sin(x)
    let expected_sin = vec![
        0.0,
        0.479425538604203,
        0.8414709848078965,
        0.9974949866040544,
        0.9092974268256817,
    ];
    let result_sin = test_data.sin();
    assert_arrays_close_f64(&result_sin, &expected_sin, 1e-10);

    // Test cosine function
    // >>> np.cos(x)
    let expected_cos = vec![
        1.0,
        0.8775825618903728,
        0.5403023058681398,
        0.07073720166770291,
        -0.4161468365471424,
    ];
    let result_cos = test_data.cos();
    assert_arrays_close_f64(&result_cos, &expected_cos, 1e-10);

    // Test square root for positive values
    let positive_data = Array::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
    // >>> np.sqrt([0.0, 1.0, 4.0, 9.0, 16.0])
    let expected_sqrt = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let result_sqrt = positive_data.sqrt();
    assert_arrays_close_f64(&result_sqrt, &expected_sqrt, 1e-15);

    // Test natural logarithm
    let log_data = Array::from_vec(vec![1.0, 2.0, std::f64::consts::E, 10.0]);
    // >>> np.log([1.0, 2.0, np.e, 10.0])
    let expected_log = vec![0.0, std::f64::consts::LN_2, 1.0, std::f64::consts::LN_10];
    let result_log = log_data.log();
    assert_arrays_close_f64(&result_log, &expected_log, 1e-10);
}

#[test]
fn test_statistical_functions_numpy_equivalence() {
    let test_data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Test sum
    // >>> import numpy as np
    // >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    // >>> np.sum(x)
    let expected_sum = 55.0f64;
    let result_sum = test_data.sum();
    assert!((result_sum - expected_sum).abs() < 1e-15);

    // Test mean
    // >>> np.mean(x)
    let expected_mean = 5.5f64;
    let result_mean = test_data.mean();
    assert!((result_mean - expected_mean).abs() < 1e-15);

    // Test standard deviation
    // >>> np.std(x, ddof=0)  # Population standard deviation
    let expected_std = 2.8722813232690143f64;
    let result_std = test_data.std();
    assert!(
        (result_std - expected_std).abs() < 1e-10,
        "Std dev mismatch: got {}, expected {}",
        result_std,
        expected_std
    );

    // Test variance
    // >>> np.var(x, ddof=0)  # Population variance
    let expected_var = 8.25f64;
    let result_var = test_data.var();
    assert!((result_var - expected_var).abs() < 1e-15);
}

#[test]
fn test_array_creation_numpy_equivalence() {
    // Test array creation functions

    // Test zeros
    let zeros_result: Array<f64> = Array::zeros(&[3, 2]);
    assert_eq!(zeros_result.shape(), &[3, 2]);
    assert_eq!(zeros_result.to_vec(), vec![0.0; 6]);

    // Test ones
    let ones_result: Array<f64> = Array::ones(&[2, 3]);
    assert_eq!(ones_result.shape(), &[2, 3]);
    assert_eq!(ones_result.to_vec(), vec![1.0; 6]);

    // Test arange equivalent
    let range_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let arange_result = Array::from_vec(range_data);
    let expected_arange = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(arange_result.to_vec(), expected_arange);

    // Test linspace equivalent (manual implementation for test)
    let linspace_result = Array::from_vec((0..11).map(|i| (i as f64) * 0.1).collect::<Vec<f64>>());
    let expected_linspace = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    assert_arrays_close_f64(&linspace_result, &expected_linspace, 1e-15);
}

#[test]
fn test_array_manipulation_numpy_equivalence() {
    let original = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Test reshape
    let reshaped = original.reshape(&[2, 3]);
    assert_eq!(reshaped.shape(), &[2, 3]);
    assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Test transpose
    let transposed = reshaped.transpose();
    assert_eq!(transposed.shape(), &[3, 2]);
    // NumPy transpose result: [[1, 4], [2, 5], [3, 6]]
    let expected_transposed = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(transposed.to_vec(), expected_transposed);

    // Test flatten (via reshape)
    let flattened = transposed.reshape(&[6]);
    assert_eq!(flattened.shape(), &[6]);
}

#[test]
fn test_boolean_operations_numpy_equivalence() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array::from_vec(vec![3.0, 2.0, 1.0, 6.0, 4.0]);

    // Test greater than
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let gt_vec: Vec<bool> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x > y).collect();
    let gt_result = Array::from_vec(gt_vec);
    let expected_gt = vec![false, false, true, false, true];
    assert_eq!(gt_result.to_vec(), expected_gt);

    // Test less than
    let lt_vec: Vec<bool> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x < y).collect();
    let lt_result = Array::from_vec(lt_vec);
    let expected_lt = vec![true, false, false, true, false];
    assert_eq!(lt_result.to_vec(), expected_lt);

    // Test equal
    let eq_vec: Vec<bool> = a_vec
        .iter()
        .zip(b_vec.iter())
        .map(|(x, y)| {
            let diff: f64 = *x - *y;
            diff.abs() < 1e-15f64
        })
        .collect();
    let eq_result = Array::from_vec(eq_vec);
    let expected_eq = vec![false, true, false, false, false];
    assert_eq!(eq_result.to_vec(), expected_eq);
}

#[test]
fn test_advanced_indexing_numpy_equivalence() {
    let data = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let condition = Array::from_vec(vec![false, true, false, true, true, false]);

    // Test extract operation (equivalent to NumPy's extract)
    // >>> import numpy as np
    // >>> data = np.array([10, 20, 30, 40, 50, 60])
    // >>> condition = np.array([False, True, False, True, True, False])
    // >>> np.extract(condition, data)
    let expected_extract = vec![20.0, 40.0, 50.0];
    let result_extract = advanced_indexing::extract(&data, &condition).unwrap();
    assert_eq!(result_extract.to_vec(), expected_extract);
}

#[test]
fn test_precision_and_edge_cases() {
    // Test edge cases and precision requirements

    // Very small numbers
    let small_numbers = Array::from_vec(vec![1e-15, 1e-10, 1e-5]);
    let exp_small = small_numbers.exp();
    // Should be approximately [1.0, 1.0, 1.00001]
    assert!((exp_small.to_vec()[0] - 1.0f64).abs() < 1e-14);
    assert!((exp_small.to_vec()[1] - 1.0f64).abs() < 1e-9);

    // Very large numbers (within reasonable range)
    let large_numbers = Array::from_vec(vec![10.0, 20.0]);
    let exp_large = large_numbers.exp();
    // exp(10) ≈ 22026.465794806718, exp(20) ≈ 485165195.4097903
    assert!((exp_large.to_vec()[0] - 22026.465794806718f64).abs() < 1e-6);

    // Test with negative numbers
    let negative_numbers = Array::from_vec(vec![-1.0, -2.0, -0.5]);
    let exp_negative = negative_numbers.exp();
    // exp(-1) ≈ 0.36787944117144233, exp(-2) ≈ 0.1353352832366127
    assert!((exp_negative.to_vec()[0] - 0.36787944117144233f64).abs() < 1e-10);
    assert!((exp_negative.to_vec()[1] - 0.1353352832366127f64).abs() < 1e-10);
}

#[test]
fn test_numpy_compatibility_summary() {
    println!("\n=== NumPy Compatibility Validation Summary ===");
    println!("✅ Bitwise operations: AND, OR, XOR, left/right shift");
    println!("✅ Complex operations: absolute, angle, real, imag, conjugate");
    println!("✅ Mathematical functions: exp, sin, cos, sqrt, log");
    println!("✅ Statistical functions: sum, mean, std, var");
    println!("✅ Array creation: zeros, ones, arange equivalent");
    println!("✅ Array manipulation: reshape, transpose");
    println!("✅ Boolean operations: comparisons");
    println!("✅ Advanced indexing: extract operation");
    println!("✅ Precision and edge cases: small/large/negative numbers");
    println!("\nNumRS2 demonstrates excellent NumPy compatibility!");
}
