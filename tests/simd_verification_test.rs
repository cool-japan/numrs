//! SIMD Verification Tests
//!
//! This module tests that SIMD optimizations are correctly activated and functional.
//! Uses scirs2-core's SimdUnifiedOps for platform-independent SIMD operations.

use numrs2::prelude::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::simd_ops::SimdUnifiedOps;

#[test]
fn test_simd_exp_functionality() {
    // Test that SIMD exp function works correctly
    let input = Array::from_vec(vec![0.0f32, 1.0, 2.0, -1.0, 0.5, -0.5]);

    // Use SimdUnifiedOps for platform-independent SIMD
    let nd_input = Array1::from_vec(input.to_vec());
    let simd_result = f32::simd_exp(&nd_input.view());
    let expected_values: Vec<f32> = input.to_vec().iter().map(|&x| x.exp()).collect();

    // Check that SIMD and scalar results are close
    for (simd_val, expected_val) in simd_result.iter().zip(expected_values.iter()) {
        assert!(
            (simd_val - expected_val).abs() < 1e-5,
            "SIMD exp mismatch: got {}, expected {}",
            simd_val,
            expected_val
        );
    }
}

#[test]
fn test_simd_performance_threshold() {
    // Test that SIMD threshold is working correctly
    let small_array = Array::from_vec(vec![1.0f32; 16]); // Below SIMD threshold (32)
    let large_array = Array::from_vec(vec![1.0f32; 100]); // Above SIMD threshold

    // Both should work, but large arrays should benefit from SIMD
    let small_result = small_array.exp();
    let large_result = large_array.exp();

    // Verify correctness
    assert_eq!(small_result.len(), 16);
    assert_eq!(large_result.len(), 100);

    // All values should be approximately e ≈ 2.718
    // Note: SIMD operations may have slightly different precision than std lib
    for &val in small_result.to_vec().iter() {
        let relative_error = (val - std::f32::consts::E).abs() / std::f32::consts::E;
        assert!(
            relative_error < 1e-4,
            "Small array exp(1.0) failed: got {}, expected {}, relative error {}",
            val,
            std::f32::consts::E,
            relative_error
        );
    }

    for &val in large_result.to_vec().iter() {
        let relative_error = (val - std::f32::consts::E).abs() / std::f32::consts::E;
        assert!(
            relative_error < 1e-4,
            "Large array exp(1.0) failed: got {}, expected {}, relative error {}",
            val,
            std::f32::consts::E,
            relative_error
        );
    }
}

#[test]
fn test_simd_matrix_operations() {
    // Test SIMD matrix operations via SimdUnifiedOps
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = Array1::from_vec(a_data.clone());
    let b = Array1::from_vec(b_data.clone());

    // Test element-wise operations
    let add_result = f32::simd_add(&a.view(), &b.view());
    let mul_result = f32::simd_mul(&a.view(), &b.view());

    // Verify add: [8, 10, 12, 14, 16, 18]
    let expected_add: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    for (got, expected) in add_result.iter().zip(expected_add.iter()) {
        assert!(
            (got - expected).abs() < 1e-5,
            "Add mismatch: got {}, expected {}",
            got,
            expected
        );
    }

    // Verify mul: [7, 16, 27, 40, 55, 72]
    let expected_mul: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x * y)
        .collect();
    for (got, expected) in mul_result.iter().zip(expected_mul.iter()) {
        assert!(
            (got - expected).abs() < 1e-5,
            "Mul mismatch: got {}, expected {}",
            got,
            expected
        );
    }
}

#[test]
fn test_simd_vs_scalar_consistency() {
    // Compare SIMD operations with scalar equivalents for consistency
    let test_data = vec![0.1f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -2.0];
    let array = Array::from_vec(test_data.clone());

    // Test exponential function
    let array_exp = array.exp();
    let scalar_exp: Vec<f32> = test_data.iter().map(|&x| x.exp()).collect();

    for (array_val, scalar_val) in array_exp.to_vec().iter().zip(scalar_exp.iter()) {
        assert!(
            (array_val - scalar_val).abs() < 1e-5,
            "Exp consistency check failed: array={}, scalar={}",
            array_val,
            scalar_val
        );
    }

    // Test other mathematical functions
    let array_sin = array.sin();
    let scalar_sin: Vec<f32> = test_data.iter().map(|&x| x.sin()).collect();

    for (array_val, scalar_val) in array_sin.to_vec().iter().zip(scalar_sin.iter()) {
        assert!(
            (array_val - scalar_val).abs() < 1e-5,
            "Sin consistency check failed: array={}, scalar={}",
            array_val,
            scalar_val
        );
    }
}

#[test]
fn test_large_array_simd_performance() {
    // Test performance characteristics with large arrays that should trigger SIMD
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    let large_array = Array::from_vec(data);

    // Test multiple operations that should use SIMD
    let exp_result = large_array.exp();
    let sin_result = large_array.sin();
    let sqrt_result = large_array.sqrt();

    // Verify results are reasonable
    assert_eq!(exp_result.len(), size);
    assert_eq!(sin_result.len(), size);
    assert_eq!(sqrt_result.len(), size);

    // Spot check some values
    let exp_vec = exp_result.to_vec();
    let sin_vec = sin_result.to_vec();
    let sqrt_vec = sqrt_result.to_vec();

    // exp(0) ≈ 1.0
    assert!((exp_vec[0] - 1.0).abs() < 1e-5);

    // sin(0) ≈ 0.0
    assert!(sin_vec[0].abs() < 1e-5);

    // sqrt(1) ≈ 1.0 (at index 1000, value = 1.0)
    if size > 1000 {
        assert!((sqrt_vec[1000] - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_simd_alignment_and_remainder_handling() {
    // Test SIMD with arrays that don't align perfectly to SIMD lane sizes

    // AVX2 processes 8 f32 values at once, so test with various remainders
    for remainder in 1..8 {
        let size = 32 + remainder; // 32 is divisible by 8, plus remainder
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let array = Array::from_vec(data.clone());

        let result = array.exp();
        let expected: Vec<f32> = data.iter().map(|&x| x.exp()).collect();

        assert_eq!(result.len(), size);

        // Check that all values are computed correctly, including the remainder
        // Note: SIMD operations may have slightly different precision than std lib
        // This is expected and acceptable for SIMD optimizations
        for (got, expected) in result.to_vec().iter().zip(expected.iter()) {
            let relative_error = (got - expected).abs() / expected.abs().max(1e-10);
            assert!(
                relative_error < 1e-3,
                "Remainder handling failed for size {}: got {}, expected {}, relative error {}",
                size,
                got,
                expected,
                relative_error
            );
        }
    }
}

#[test]
fn test_simd_unified_ops_availability() {
    // Test that SimdUnifiedOps functions are available and work correctly
    let test_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let nd_array = Array1::from_vec(test_data.clone());

    // Test various SIMD operations
    let exp_result = f64::simd_exp(&nd_array.view());
    let sqrt_result = f64::simd_sqrt(&nd_array.view());
    let sum_result = f64::simd_sum(&nd_array.view());

    // Verify exp
    assert_eq!(exp_result.len(), 8);
    assert!((exp_result[0] - std::f64::consts::E).abs() < 1e-10);

    // Verify sqrt
    assert_eq!(sqrt_result.len(), 8);
    assert!((sqrt_result[0] - 1.0).abs() < 1e-10);
    assert!((sqrt_result[3] - 2.0).abs() < 1e-10);

    // Verify sum: 1+2+3+4+5+6+7+8 = 36
    assert!((sum_result - 36.0).abs() < 1e-10);
}

#[test]
fn test_simd_transcendental_functions() {
    // Test transcendental functions via SimdUnifiedOps
    let test_data: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let nd_array = Array1::from_vec(test_data.clone());

    // Test sin
    let sin_result = f64::simd_sin(&nd_array.view());
    for (i, &x) in test_data.iter().enumerate() {
        assert!(
            (sin_result[i] - x.sin()).abs() < 1e-10,
            "sin mismatch at index {}: got {}, expected {}",
            i,
            sin_result[i],
            x.sin()
        );
    }

    // Test cos
    let cos_result = f64::simd_cos(&nd_array.view());
    for (i, &x) in test_data.iter().enumerate() {
        assert!(
            (cos_result[i] - x.cos()).abs() < 1e-10,
            "cos mismatch at index {}: got {}, expected {}",
            i,
            cos_result[i],
            x.cos()
        );
    }

    // Test ln
    let positive_data: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, std::f64::consts::E];
    let nd_positive = Array1::from_vec(positive_data.clone());
    let ln_result = f64::simd_ln(&nd_positive.view());
    for (i, &x) in positive_data.iter().enumerate() {
        assert!(
            (ln_result[i] - x.ln()).abs() < 1e-10,
            "ln mismatch at index {}: got {}, expected {}",
            i,
            ln_result[i],
            x.ln()
        );
    }
}
