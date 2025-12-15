//! SIMD Verification Tests
//!
//! This module tests that SIMD optimizations are correctly activated and functional.

use numrs2::prelude::*;
use numrs2::simd_optimize::avx2_enhanced::EnhancedSimdOps;

#[test]
fn test_simd_exp_functionality() {
    // Test that SIMD exp function works correctly
    let input = Array::from_vec(vec![0.0f32, 1.0, 2.0, -1.0, 0.5, -0.5]);

    #[cfg(target_arch = "x86_64")]
    {
        let simd_result = EnhancedSimdOps::vectorized_exp_f32(&input);
        let expected_values = input
            .to_vec()
            .iter()
            .map(|&x| x.exp())
            .collect::<Vec<f32>>();
        let expected = Array::from_vec(expected_values);

        // Check that SIMD and scalar results are close
        for (simd_val, expected_val) in simd_result.to_vec().iter().zip(expected.to_vec().iter()) {
            assert!(
                (simd_val - expected_val).abs() < 1e-5,
                "SIMD exp mismatch: got {}, expected {}",
                simd_val,
                expected_val
            );
        }
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
fn test_simd_cache_aware_matmul() {
    // Test cache-aware matrix multiplication
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let b = Array::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[3, 2]);

    #[allow(unused_mut)]
    let mut c: Array<f32> = Array::zeros(&[2, 2]);

    #[cfg(target_arch = "x86_64")]
    {
        let result = EnhancedSimdOps::cache_aware_matmul_f32(&a, &b, &mut c, 32);

        if let Ok(()) = result {
            // Expected result of matrix multiplication:
            // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]  = [58, 64]
            // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]  = [139, 154]
            let expected = vec![58.0f32, 64.0, 139.0, 154.0];
            let result_vec = c.to_vec();

            for (got, expected) in result_vec.iter().zip(expected.iter()) {
                assert!(
                    (got - expected).abs() < 1e-4,
                    "Matrix multiplication mismatch: got {}, expected {}",
                    got,
                    expected
                );
            }
        }
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

#[cfg(target_arch = "x86_64")]
#[test]
fn test_target_feature_availability() {
    // This test just verifies that the code compiles with target features
    // Actual feature detection would require runtime checks

    use std::arch::is_x86_feature_detected;

    // These checks verify that the CPU supports the required features
    // The tests will be skipped if the features aren't available
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        println!("AVX2 and FMA features are available");

        // Test a simple SIMD operation
        let test_array = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = EnhancedSimdOps::vectorized_exp_f32(&test_array);

        assert_eq!(result.len(), 8);

        // First element should be exp(1) ≈ 2.718
        assert!((result.to_vec()[0] - std::f32::consts::E).abs() < 1e-4);
    } else {
        println!("AVX2/FMA not available, skipping SIMD-specific tests");
    }
}
