//! Tests for NEON SIMD operations
//!
//! This module contains comprehensive tests for all NEON operations.

use crate::array::Array;
use crate::simd_optimize::neon_enhanced::{NeonEnhancedOps, NeonFeatureDetector};
use approx::assert_relative_eq;

#[test]
fn test_neon_feature_detection() {
    let features = NeonFeatureDetector::detect_neon_features();
    println!("NEON features: {:?}", features);

    let ops = features.recommended_operations();

    // On x86_64, NEON won't be available, so ops might be empty
    #[cfg(target_arch = "aarch64")]
    assert!(!ops.is_empty());

    #[cfg(not(target_arch = "aarch64"))]
    println!("NEON not available on this architecture: {}", ops.len());
}

#[test]
fn test_neon_exp() {
    let input = Array::from_vec(vec![0.0, 1.0, 2.0, -1.0]);
    let result = NeonEnhancedOps::neon_exp_f32(&input);

    assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(result.to_vec()[1], std::f32::consts::E, epsilon = 1e-4);
    assert_relative_eq!(
        result.to_vec()[2],
        std::f32::consts::E.powi(2),
        epsilon = 5e-3
    );
    assert_relative_eq!(
        result.to_vec()[3],
        1.0 / std::f32::consts::E,
        epsilon = 2e-5
    );
}

#[test]
fn test_neon_sum() {
    let input = Array::from_vec(vec![1.0f32; 100]);
    let result = NeonEnhancedOps::neon_sum_f32(&input);
    assert_relative_eq!(result, 100.0, epsilon = 1e-6);
}

#[test]
fn test_neon_dot() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
    let result = NeonEnhancedOps::neon_dot_f32(&a, &b)
        .expect("neon_dot_f32 should succeed with equal-length vectors");
    assert_relative_eq!(result, 70.0, epsilon = 1e-6); // 1*5 + 2*6 + 3*7 + 4*8 = 70
}

#[test]
fn test_optimal_block_size() {
    let block_size = NeonFeatureDetector::optimal_block_size();
    assert!(block_size >= 16);
    assert!(block_size <= 64);
}

// =========================================================================
// Tests for f64 NEON operations (used by ufuncs)
// =========================================================================

#[test]
fn test_vectorized_abs_f64() {
    let input = Array::from_vec(vec![-1.0, -2.0, 3.0, -4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_abs_f64(&input);
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_sqrt_f64() {
    let input = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    let result = NeonEnhancedOps::vectorized_sqrt_f64(&input);
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_square_f64() {
    let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_square_f64(&input);
    let expected = [1.0, 4.0, 9.0, 16.0, 25.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_exp_f64() {
    let input = Array::from_vec(vec![0.0, 1.0, 2.0]);
    let result = NeonEnhancedOps::vectorized_exp_f64(&input);
    assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(result.to_vec()[1], std::f64::consts::E, epsilon = 1e-4);
    assert_relative_eq!(
        result.to_vec()[2],
        std::f64::consts::E.powi(2),
        epsilon = 1e-2
    );
}

#[test]
fn test_vectorized_floor_ceil_round_f64() {
    let input = Array::from_vec(vec![1.3, 2.7, -1.3, -2.7]);

    let floor_result = NeonEnhancedOps::vectorized_floor_f64(&input);
    let ceil_result = NeonEnhancedOps::vectorized_ceil_f64(&input);
    let round_result = NeonEnhancedOps::vectorized_round_f64(&input);

    assert_relative_eq!(floor_result.to_vec()[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(floor_result.to_vec()[1], 2.0, epsilon = 1e-10);
    assert_relative_eq!(floor_result.to_vec()[2], -2.0, epsilon = 1e-10);
    assert_relative_eq!(floor_result.to_vec()[3], -3.0, epsilon = 1e-10);

    assert_relative_eq!(ceil_result.to_vec()[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(ceil_result.to_vec()[1], 3.0, epsilon = 1e-10);
    assert_relative_eq!(ceil_result.to_vec()[2], -1.0, epsilon = 1e-10);
    assert_relative_eq!(ceil_result.to_vec()[3], -2.0, epsilon = 1e-10);

    assert_relative_eq!(round_result.to_vec()[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(round_result.to_vec()[1], 3.0, epsilon = 1e-10);
    assert_relative_eq!(round_result.to_vec()[2], -1.0, epsilon = 1e-10);
    assert_relative_eq!(round_result.to_vec()[3], -3.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_sign_f64() {
    let input = Array::from_vec(vec![-5.0, 0.0, 3.0, -0.0]);
    let result = NeonEnhancedOps::vectorized_sign_f64(&input);
    assert_relative_eq!(result.to_vec()[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(result.to_vec()[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.to_vec()[2], 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.to_vec()[3], 0.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_large_array_f64() {
    // Test with array larger than SIMD threshold (32) to ensure NEON path is taken
    let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let arr = Array::from_vec(input.clone());

    let abs_result = NeonEnhancedOps::vectorized_abs_f64(&arr);
    let square_result = NeonEnhancedOps::vectorized_square_f64(&arr);

    assert_eq!(abs_result.len(), 100);
    assert_eq!(square_result.len(), 100);

    // Check a few values
    assert_relative_eq!(square_result.to_vec()[10], 100.0, epsilon = 1e-10);
    assert_relative_eq!(square_result.to_vec()[5], 25.0, epsilon = 1e-10);
}

// =========================================================================
// Tests for High-Priority NEON f64 Operations
// =========================================================================

#[test]
fn test_vectorized_add_arrays_f64() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let result = NeonEnhancedOps::vectorized_add_arrays_f64(&a, &b);
    let expected = [11.0, 22.0, 33.0, 44.0, 55.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_sub_arrays_f64() {
    let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_sub_arrays_f64(&a, &b);
    let expected = [9.0, 18.0, 27.0, 36.0, 45.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_mul_arrays_f64() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = NeonEnhancedOps::vectorized_mul_arrays_f64(&a, &b);
    let expected = [2.0, 6.0, 12.0, 20.0, 30.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_div_arrays_f64() {
    let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let b = Array::from_vec(vec![2.0, 4.0, 5.0, 8.0, 10.0]);
    let result = NeonEnhancedOps::vectorized_div_arrays_f64(&a, &b);
    let expected = [5.0, 5.0, 6.0, 5.0, 5.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_add_scalar_f64() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_add_scalar_f64(&a, 10.0);
    let expected = [11.0, 12.0, 13.0, 14.0, 15.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_mul_scalar_f64() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_mul_scalar_f64(&a, 3.0);
    let expected = [3.0, 6.0, 9.0, 12.0, 15.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_sum_f64() {
    let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_sum_f64(&input);
    assert_relative_eq!(result, 15.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_prod_f64() {
    let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_prod_f64(&input);
    assert_relative_eq!(result, 120.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_max_min_f64() {
    let input = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
    let max_result = NeonEnhancedOps::vectorized_max_f64(&input);
    let min_result = NeonEnhancedOps::vectorized_min_f64(&input);
    assert_relative_eq!(max_result, 9.0, epsilon = 1e-10);
    assert_relative_eq!(min_result, 1.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_mean_f64() {
    let input = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    let result = NeonEnhancedOps::vectorized_mean_f64(&input);
    assert_relative_eq!(result, 6.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_fma_f64() {
    // FMA: a * b + c
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
    let c = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
    let result = NeonEnhancedOps::vectorized_fma_f64(&a, &b, &c);
    // 1*2+10=12, 2*3+20=26, 3*4+30=42, 4*5+40=60
    let expected = [12.0, 26.0, 42.0, 60.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_dot_f64() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
    let result = NeonEnhancedOps::vectorized_dot_f64(&a, &b);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert_relative_eq!(result, 70.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_norm_l2_f64() {
    let input = Array::from_vec(vec![3.0, 4.0]); // sqrt(9+16) = 5
    let result = NeonEnhancedOps::vectorized_norm_l2_f64(&input);
    assert_relative_eq!(result, 5.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_norm_l1_f64() {
    let input = Array::from_vec(vec![-3.0, 4.0, -5.0]); // |-3|+|4|+|-5| = 12
    let result = NeonEnhancedOps::vectorized_norm_l1_f64(&input);
    assert_relative_eq!(result, 12.0, epsilon = 1e-10);
}

#[test]
fn test_vectorized_maximum_minimum_f64() {
    let a = Array::from_vec(vec![1.0, 5.0, 3.0, 7.0]);
    let b = Array::from_vec(vec![2.0, 4.0, 6.0, 5.0]);
    let max_result = NeonEnhancedOps::vectorized_maximum_f64(&a, &b);
    let min_result = NeonEnhancedOps::vectorized_minimum_f64(&a, &b);

    let max_expected = [2.0, 5.0, 6.0, 7.0];
    let min_expected = [1.0, 4.0, 3.0, 5.0];

    for (r, e) in max_result.to_vec().iter().zip(max_expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
    for (r, e) in min_result.to_vec().iter().zip(min_expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_negative_f64() {
    let input = Array::from_vec(vec![1.0, -2.0, 3.0, -4.0]);
    let result = NeonEnhancedOps::vectorized_negative_f64(&input);
    let expected = [-1.0, 2.0, -3.0, 4.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_reciprocal_f64() {
    let input = Array::from_vec(vec![1.0, 2.0, 4.0, 5.0]);
    let result = NeonEnhancedOps::vectorized_reciprocal_f64(&input);
    let expected = [1.0, 0.5, 0.25, 0.2];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_high_priority_large_arrays_f64() {
    // Test with arrays larger than SIMD threshold to ensure NEON path is taken
    let input_a: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let input_b: Vec<f64> = (101..=200).map(|i| i as f64).collect();
    let arr_a = Array::from_vec(input_a);
    let arr_b = Array::from_vec(input_b);

    // Test array operations
    let add_result = NeonEnhancedOps::vectorized_add_arrays_f64(&arr_a, &arr_b);
    let mul_result = NeonEnhancedOps::vectorized_mul_arrays_f64(&arr_a, &arr_b);

    assert_eq!(add_result.len(), 100);
    assert_eq!(mul_result.len(), 100);

    // First element: 1 + 101 = 102, 1 * 101 = 101
    assert_relative_eq!(add_result.to_vec()[0], 102.0, epsilon = 1e-10);
    assert_relative_eq!(mul_result.to_vec()[0], 101.0, epsilon = 1e-10);

    // Last element: 100 + 200 = 300, 100 * 200 = 20000
    assert_relative_eq!(add_result.to_vec()[99], 300.0, epsilon = 1e-10);
    assert_relative_eq!(mul_result.to_vec()[99], 20000.0, epsilon = 1e-10);

    // Test reductions
    let sum = NeonEnhancedOps::vectorized_sum_f64(&arr_a);
    assert_relative_eq!(sum, 5050.0, epsilon = 1e-10); // sum of 1..100

    let max = NeonEnhancedOps::vectorized_max_f64(&arr_a);
    let min = NeonEnhancedOps::vectorized_min_f64(&arr_a);
    assert_relative_eq!(max, 100.0, epsilon = 1e-10);
    assert_relative_eq!(min, 1.0, epsilon = 1e-10);

    // Test dot product
    let simple_a = Array::from_vec(vec![1.0f64; 100]);
    let simple_b = Array::from_vec(vec![2.0f64; 100]);
    let dot = NeonEnhancedOps::vectorized_dot_f64(&simple_a, &simple_b);
    assert_relative_eq!(dot, 200.0, epsilon = 1e-10);
}

// =========================================================================
// Tests for Medium-Priority NEON f64 Operations
// =========================================================================

#[test]
fn test_vectorized_clamp_f64() {
    let input = Array::from_vec(vec![-2.0, 0.5, 2.0, 5.0, 10.0]);
    let result = NeonEnhancedOps::vectorized_clamp_f64(&input, 0.0, 3.0);
    let expected = [0.0, 0.5, 2.0, 3.0, 3.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_pow_f64() {
    let base = Array::from_vec(vec![2.0, 3.0, 4.0]);
    let exp = Array::from_vec(vec![2.0, 2.0, 0.5]);
    let result = NeonEnhancedOps::vectorized_pow_f64(&base, &exp);
    let expected = [4.0, 9.0, 2.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_pow_scalar_f64() {
    let base = Array::from_vec(vec![2.0, 3.0, 4.0]);
    let result = NeonEnhancedOps::vectorized_pow_scalar_f64(&base, 3.0);
    let expected = [8.0, 27.0, 64.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_cbrt_f64() {
    let input = Array::from_vec(vec![8.0, 27.0, 64.0]);
    let result = NeonEnhancedOps::vectorized_cbrt_f64(&input);
    let expected = [2.0, 3.0, 4.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_log2_log10_f64() {
    let input = Array::from_vec(vec![1.0, 2.0, 4.0, 8.0]);
    let log2_result = NeonEnhancedOps::vectorized_log2_f64(&input);
    let expected_log2 = [0.0, 1.0, 2.0, 3.0];
    for (r, e) in log2_result.to_vec().iter().zip(expected_log2.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }

    let input10 = Array::from_vec(vec![1.0, 10.0, 100.0]);
    let log10_result = NeonEnhancedOps::vectorized_log10_f64(&input10);
    let expected_log10 = [0.0, 1.0, 2.0];
    for (r, e) in log10_result.to_vec().iter().zip(expected_log10.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_exp2_f64() {
    let input = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let result = NeonEnhancedOps::vectorized_exp2_f64(&input);
    let expected = [1.0, 2.0, 4.0, 8.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_expm1_log1p_f64() {
    // exp(0) - 1 = 0, exp(small) - 1 ~ small for small values
    let input = Array::from_vec(vec![0.0, 1e-10]);
    let expm1_result = NeonEnhancedOps::vectorized_expm1_f64(&input);
    assert_relative_eq!(expm1_result.to_vec()[0], 0.0, epsilon = 1e-15);
    assert_relative_eq!(expm1_result.to_vec()[1], 1e-10, epsilon = 1e-18);

    // log(1 + 0) = 0, log(1 + small) ~ small for small values
    let log1p_result = NeonEnhancedOps::vectorized_log1p_f64(&input);
    assert_relative_eq!(log1p_result.to_vec()[0], 0.0, epsilon = 1e-15);
    assert_relative_eq!(log1p_result.to_vec()[1], 1e-10, epsilon = 1e-18);
}

#[test]
fn test_vectorized_copysign_f64() {
    let magnitude = Array::from_vec(vec![1.0, -2.0, 3.0, -4.0]);
    let sign = Array::from_vec(vec![-1.0, 1.0, -1.0, 1.0]);
    let result = NeonEnhancedOps::vectorized_copysign_f64(&magnitude, &sign);
    let expected = [-1.0, 2.0, -3.0, 4.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_hypot_f64() {
    let x = Array::from_vec(vec![3.0, 5.0, 8.0]);
    let y = Array::from_vec(vec![4.0, 12.0, 15.0]);
    let result = NeonEnhancedOps::vectorized_hypot_f64(&x, &y);
    let expected = [5.0, 13.0, 17.0];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_atan2_f64() {
    let y = Array::from_vec(vec![0.0, 1.0, 0.0, -1.0]);
    let x = Array::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
    let result = NeonEnhancedOps::vectorized_atan2_f64(&y, &x);
    // atan2(0,1)=0, atan2(1,0)=pi/2, atan2(0,-1)=pi, atan2(-1,0)=-pi/2
    let expected = [
        0.0,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        -std::f64::consts::FRAC_PI_2,
    ];
    for (r, e) in result.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_sub_div_scalar_f64() {
    let input = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

    let sub_result = NeonEnhancedOps::vectorized_sub_scalar_f64(&input, 5.0);
    let sub_expected = [5.0, 15.0, 25.0, 35.0];
    for (r, e) in sub_result.to_vec().iter().zip(sub_expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }

    let div_result = NeonEnhancedOps::vectorized_div_scalar_f64(&input, 10.0);
    let div_expected = [1.0, 2.0, 3.0, 4.0];
    for (r, e) in div_result.to_vec().iter().zip(div_expected.iter()) {
        assert_relative_eq!(r, e, epsilon = 1e-10);
    }
}

#[test]
fn test_vectorized_variance_std_f64() {
    // Simple variance/std test: [1, 2, 3, 4, 5] => mean=3, variance=2, std=sqrt(2)
    let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let variance = NeonEnhancedOps::vectorized_variance_f64(&input);
    let std_dev = NeonEnhancedOps::vectorized_std_f64(&input);

    // variance = [(1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2] / 5
    //          = [4 + 1 + 0 + 1 + 4] / 5 = 10/5 = 2
    assert_relative_eq!(variance, 2.0, epsilon = 1e-10);
    assert_relative_eq!(std_dev, 2.0f64.sqrt(), epsilon = 1e-10);
}

#[test]
fn test_vectorized_medium_priority_large_arrays_f64() {
    // Test with arrays larger than SIMD threshold
    let input: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let arr = Array::from_vec(input);

    // Test clamp on large array
    let clamped = NeonEnhancedOps::vectorized_clamp_f64(&arr, 20.0, 80.0);
    assert_eq!(clamped.len(), 100);
    assert_relative_eq!(clamped.to_vec()[0], 20.0, epsilon = 1e-10); // 1 clamped to 20
    assert_relative_eq!(clamped.to_vec()[50], 51.0, epsilon = 1e-10); // 51 unchanged
    assert_relative_eq!(clamped.to_vec()[99], 80.0, epsilon = 1e-10); // 100 clamped to 80

    // Test variance/std on large array
    // For 1..100, mean = 50.5
    // variance = sum((x - 50.5)^2) / 100 = 833.25
    let variance = NeonEnhancedOps::vectorized_variance_f64(&arr);
    assert_relative_eq!(variance, 833.25, epsilon = 1e-10);

    // Test hypot on large arrays
    let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let y: Vec<f64> = (1..=100).map(|i| (i * 2) as f64).collect();
    let arr_x = Array::from_vec(x);
    let arr_y = Array::from_vec(y);
    let hypot = NeonEnhancedOps::vectorized_hypot_f64(&arr_x, &arr_y);
    // hypot(1, 2) = sqrt(5), hypot(100, 200) = sqrt(50000)
    assert_relative_eq!(hypot.to_vec()[0], 5.0f64.sqrt(), epsilon = 1e-10);
    assert_relative_eq!(hypot.to_vec()[99], 50000.0f64.sqrt(), epsilon = 1e-8);
}
