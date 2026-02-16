//! WebAssembly statistics tests
//!
//! This module tests NumRS2's WASM statistics bindings using wasm-bindgen-test.
//! All operations use scirs2-stats for implementation following SCIRS2 policy.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// Import WASM bindings
use numrs2::wasm::WasmArray;
use numrs2::wasm::stats::*;

const TOLERANCE: f64 = 1e-10;

#[wasm_bindgen_test]
fn test_mean() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let m = mean(&arr);
    assert_eq!(m, 3.0);
}

#[wasm_bindgen_test]
fn test_mean_negative() {
    let arr = WasmArray::from_vec(&vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5])
        .expect("Failed to create array");

    let m = mean(&arr);
    assert_eq!(m, 0.0);
}

#[wasm_bindgen_test]
fn test_mean_2d() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create array");

    let m = mean(&arr);
    assert_eq!(m, 3.5);
}

#[wasm_bindgen_test]
fn test_median_odd_count() {
    let arr = WasmArray::from_vec(&vec![1.0, 3.0, 2.0, 5.0, 4.0], &[5])
        .expect("Failed to create array");

    let med = median(&arr);
    assert_eq!(med, 3.0);
}

#[wasm_bindgen_test]
fn test_median_even_count() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[4])
        .expect("Failed to create array");

    let med = median(&arr);
    assert_eq!(med, 2.5);
}

#[wasm_bindgen_test]
fn test_median_single_element() {
    let arr = WasmArray::from_vec(&vec![42.0], &[1])
        .expect("Failed to create array");

    let med = median(&arr);
    assert_eq!(med, 42.0);
}

#[wasm_bindgen_test]
fn test_variance() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let var = variance(&arr);
    // Variance of [1,2,3,4,5] is 2.0
    assert!((var - 2.0).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_variance_constant() {
    let arr = WasmArray::full(&[5], 3.0);

    let var = variance(&arr);
    // Variance of constant array should be 0
    assert!(var.abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_std_dev() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let std = std_dev(&arr);
    // Std dev is sqrt(variance) = sqrt(2.0) ≈ 1.414
    assert!((std - 1.4142135623730951).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_std_dev_constant() {
    let arr = WasmArray::full(&[10], 5.0);

    let std = std_dev(&arr);
    // Std dev of constant array should be 0
    assert!(std.abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_minimum() {
    let arr = WasmArray::from_vec(&vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6])
        .expect("Failed to create array");

    let min = minimum(&arr);
    assert_eq!(min, 1.0);
}

#[wasm_bindgen_test]
fn test_minimum_negative() {
    let arr = WasmArray::from_vec(&vec![3.0, -5.0, 2.0, 7.0], &[4])
        .expect("Failed to create array");

    let min = minimum(&arr);
    assert_eq!(min, -5.0);
}

#[wasm_bindgen_test]
fn test_maximum() {
    let arr = WasmArray::from_vec(&vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6])
        .expect("Failed to create array");

    let max = maximum(&arr);
    assert_eq!(max, 9.0);
}

#[wasm_bindgen_test]
fn test_maximum_negative() {
    let arr = WasmArray::from_vec(&vec![-3.0, -5.0, -2.0, -7.0], &[4])
        .expect("Failed to create array");

    let max = maximum(&arr);
    assert_eq!(max, -2.0);
}

#[wasm_bindgen_test]
fn test_sum() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let s = sum(&arr);
    assert_eq!(s, 15.0);
}

#[wasm_bindgen_test]
fn test_sum_negative() {
    let arr = WasmArray::from_vec(&vec![-1.0, 2.0, -3.0, 4.0], &[4])
        .expect("Failed to create array");

    let s = sum(&arr);
    assert_eq!(s, 2.0);
}

#[wasm_bindgen_test]
fn test_prod() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let p = product(&arr);
    assert_eq!(p, 120.0);
}

#[wasm_bindgen_test]
fn test_percentile_0() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let p0 = compute_percentile(&arr, 0.0).expect("Percentile failed");
    assert_eq!(p0, 1.0);
}

#[wasm_bindgen_test]
fn test_percentile_25() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let p25 = compute_percentile(&arr, 0.25).expect("Percentile failed");
    assert_eq!(p25, 2.0);
}

#[wasm_bindgen_test]
fn test_percentile_50() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let p50 = compute_percentile(&arr, 0.5).expect("Percentile failed");
    assert_eq!(p50, 3.0);
}

#[wasm_bindgen_test]
fn test_percentile_75() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let p75 = compute_percentile(&arr, 0.75).expect("Percentile failed");
    assert_eq!(p75, 4.0);
}

#[wasm_bindgen_test]
fn test_percentile_100() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let p100 = compute_percentile(&arr, 1.0).expect("Percentile failed");
    assert_eq!(p100, 5.0);
}

#[wasm_bindgen_test]
fn test_percentile_invalid() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0], &[3])
        .expect("Failed to create array");

    // Percentile outside [0, 1] should fail
    let result = compute_percentile(&arr, 1.5);
    assert!(result.is_err());

    let result = compute_percentile(&arr, -0.1);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_histogram() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], &[7])
        .expect("Failed to create array");

    let result = compute_histogram(&arr, 4).expect("Histogram failed");

    // Should return (counts, bin_edges)
    assert_eq!(result.0.size(), 4);
    assert_eq!(result.1.size(), 5); // n+1 bin edges
}

#[wasm_bindgen_test]
fn test_covariance() {
    let x = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let y = WasmArray::from_vec(&vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5])
        .expect("Failed to create array");

    let cov = covariance(&x, &y).expect("Covariance failed");

    // Perfect positive correlation should give cov = 4.0
    assert!((cov - 4.0).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_covariance_independent() {
    let x = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[4])
        .expect("Failed to create array");

    let y = WasmArray::from_vec(&vec![1.0, -1.0, 1.0, -1.0], &[4])
        .expect("Failed to create array");

    let cov = covariance(&x, &y).expect("Covariance failed");

    // Should be close to 0 for independent variables
    assert!(cov.abs() < 0.1);
}

#[wasm_bindgen_test]
fn test_covariance_incompatible() {
    let x = WasmArray::ones(&[5]);
    let y = WasmArray::ones(&[4]);

    let result = covariance(&x, &y);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_correlation() {
    let x = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let y = WasmArray::from_vec(&vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5])
        .expect("Failed to create array");

    let corr = correlation(&x, &y).expect("Correlation failed");

    // Perfect positive correlation should give corr = 1.0
    assert!((corr - 1.0).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_correlation_negative() {
    let x = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
        .expect("Failed to create array");

    let y = WasmArray::from_vec(&vec![10.0, 8.0, 6.0, 4.0, 2.0], &[5])
        .expect("Failed to create array");

    let corr = correlation(&x, &y).expect("Correlation failed");

    // Perfect negative correlation should give corr = -1.0
    assert!((corr - (-1.0)).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_correlation_incompatible() {
    let x = WasmArray::ones(&[5]);
    let y = WasmArray::ones(&[3]);

    let result = correlation(&x, &y);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_corrcoef_matrix() {
    // Create data matrix: 100 samples, 3 features
    let data = WasmArray::random(&[100, 3]);

    let corr_matrix = correlation_coefficient(&data).expect("Corrcoef failed");

    assert_eq!(corr_matrix.shape(), vec![3, 3]);

    // Diagonal should be 1.0 (self-correlation)
    assert!((corr_matrix.get(&[0, 0]).expect("Get failed") - 1.0).abs() < TOLERANCE);
    assert!((corr_matrix.get(&[1, 1]).expect("Get failed") - 1.0).abs() < TOLERANCE);
    assert!((corr_matrix.get(&[2, 2]).expect("Get failed") - 1.0).abs() < TOLERANCE);

    // Matrix should be symmetric
    let corr_01 = corr_matrix.get(&[0, 1]).expect("Get failed");
    let corr_10 = corr_matrix.get(&[1, 0]).expect("Get failed");
    assert!((corr_01 - corr_10).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_random_normal() {
    let arr = random_normal(&[1000]).expect("Random normal failed");

    assert_eq!(arr.size(), 1000);

    // Check that mean is close to 0 and std is close to 1
    let m = mean(&arr);
    let s = std_dev(&arr);

    assert!(m.abs() < 0.1); // Mean should be close to 0
    assert!((s - 1.0).abs() < 0.1); // Std should be close to 1
}

#[wasm_bindgen_test]
fn test_random_uniform() {
    let arr = random_uniform(&[1000]).expect("Random uniform failed");

    assert_eq!(arr.size(), 1000);

    // All values should be in [0, 1)
    let min = minimum(&arr);
    let max = maximum(&arr);

    assert!(min >= 0.0);
    assert!(max < 1.0);

    // Mean should be close to 0.5
    let m = mean(&arr);
    assert!((m - 0.5).abs() < 0.1);
}

#[wasm_bindgen_test]
fn test_random_uniform_range() {
    let arr = random_uniform_range(&[1000], -5.0, 5.0).expect("Random uniform range failed");

    assert_eq!(arr.size(), 1000);

    // All values should be in [-5, 5)
    let min = minimum(&arr);
    let max = maximum(&arr);

    assert!(min >= -5.0);
    assert!(max < 5.0);

    // Mean should be close to 0
    let m = mean(&arr);
    assert!(m.abs() < 0.5);
}

#[wasm_bindgen_test]
fn test_random_uniform_invalid_range() {
    // High < Low should fail
    let result = random_uniform_range(&[10], 5.0, 0.0);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_statistics_chain() {
    // Test chaining multiple statistical operations
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[10])
        .expect("Failed to create array");

    let m = mean(&arr);
    let med = median(&arr);
    let var = variance(&arr);
    let std = std_dev(&arr);
    let min = minimum(&arr);
    let max = maximum(&arr);

    assert_eq!(m, 5.5);
    assert_eq!(med, 5.5);
    assert!((var - 8.25).abs() < TOLERANCE);
    assert!((std - 2.8722813232690143).abs() < TOLERANCE);
    assert_eq!(min, 1.0);
    assert_eq!(max, 10.0);
}

#[wasm_bindgen_test]
fn test_empty_array_stats() {
    let arr = WasmArray::zeros(&[0]);

    // Operations on empty arrays should handle gracefully
    // (Behavior may vary - could return NaN, 0, or error)
    let m = mean(&arr);
    assert!(m.is_nan() || m == 0.0);
}

#[wasm_bindgen_test]
fn test_single_element_stats() {
    let arr = WasmArray::full(&[1], 42.0);

    assert_eq!(mean(&arr), 42.0);
    assert_eq!(median(&arr), 42.0);
    assert_eq!(minimum(&arr), 42.0);
    assert_eq!(maximum(&arr), 42.0);
    assert_eq!(sum(&arr), 42.0);
}

#[wasm_bindgen_test]
fn test_2d_array_stats() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create array");

    // Stats should work on flattened array
    assert_eq!(mean(&arr), 3.5);
    assert_eq!(sum(&arr), 21.0);
    assert_eq!(minimum(&arr), 1.0);
    assert_eq!(maximum(&arr), 6.0);
}
