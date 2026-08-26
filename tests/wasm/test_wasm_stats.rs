//! WebAssembly statistics tests
//!
//! This module tests NumRS2's WASM statistics bindings using wasm-bindgen-test.
//! All operations use scirs2-stats for implementation following SCIRS2 policy.

#![cfg(target_arch = "wasm32")]

// `wasm_bindgen_test_configure!(run_in_browser)` must appear exactly once per
// test binary (see `tests/wasm_integration.rs`, which wires this file in),
// not once per module -- so it does not appear here.
use wasm_bindgen_test::*;

// Import WASM bindings
use numrs2::wasm::stats::*;
use numrs2::wasm::WasmArray;

const TOLERANCE: f64 = 1e-10;

#[wasm_bindgen_test]
fn test_mean() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

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
    let arr =
        WasmArray::from_vec(&vec![1.0, 3.0, 2.0, 5.0, 4.0], &[5]).expect("Failed to create array");

    let med = median(&arr);
    assert_eq!(med, 3.0);
}

#[wasm_bindgen_test]
fn test_median_even_count() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Failed to create array");

    let med = median(&arr);
    assert_eq!(med, 2.5);
}

#[wasm_bindgen_test]
fn test_median_single_element() {
    let arr = WasmArray::from_vec(&vec![42.0], &[1]).expect("Failed to create array");

    let med = median(&arr);
    assert_eq!(med, 42.0);
}

#[wasm_bindgen_test]
fn test_variance() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

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
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

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
    let arr =
        WasmArray::from_vec(&vec![3.0, -5.0, 2.0, 7.0], &[4]).expect("Failed to create array");

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
    let arr =
        WasmArray::from_vec(&vec![-3.0, -5.0, -2.0, -7.0], &[4]).expect("Failed to create array");

    let max = maximum(&arr);
    assert_eq!(max, -2.0);
}

#[wasm_bindgen_test]
fn test_sum() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let s = sum(&arr);
    assert_eq!(s, 15.0);
}

#[wasm_bindgen_test]
fn test_sum_negative() {
    let arr =
        WasmArray::from_vec(&vec![-1.0, 2.0, -3.0, 4.0], &[4]).expect("Failed to create array");

    let s = sum(&arr);
    assert_eq!(s, 2.0);
}

#[wasm_bindgen_test]
fn test_prod() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let p = product(&arr);
    assert_eq!(p, 120.0);
}

#[wasm_bindgen_test]
fn test_percentile_0() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let p0 = compute_percentile(&arr, 0.0).expect("Percentile failed");
    assert_eq!(p0, 1.0);
}

#[wasm_bindgen_test]
fn test_percentile_25() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let p25 = compute_percentile(&arr, 0.25).expect("Percentile failed");
    assert_eq!(p25, 2.0);
}

#[wasm_bindgen_test]
fn test_percentile_50() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let p50 = compute_percentile(&arr, 0.5).expect("Percentile failed");
    assert_eq!(p50, 3.0);
}

#[wasm_bindgen_test]
fn test_percentile_75() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let p75 = compute_percentile(&arr, 0.75).expect("Percentile failed");
    assert_eq!(p75, 4.0);
}

#[wasm_bindgen_test]
fn test_percentile_100() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let p100 = compute_percentile(&arr, 1.0).expect("Percentile failed");
    assert_eq!(p100, 5.0);
}

#[wasm_bindgen_test]
fn test_percentile_invalid() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0], &[3]).expect("Failed to create array");

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

    // `HistogramResult` exposes `counts`/`bin_edges` via `#[wasm_bindgen(getter)]`
    // methods, not as a tuple struct.
    assert_eq!(result.counts().size(), 4);
    assert_eq!(result.bin_edges().size(), 5); // n+1 bin edges
}

#[wasm_bindgen_test]
fn test_covariance() {
    let x =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let y =
        WasmArray::from_vec(&vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5]).expect("Failed to create array");

    // `covariance(x, Some(y))` for two 1-D inputs always returns the full
    // 2x2 covariance matrix (see `stats::correlation::cov`'s doc comment:
    // "Estimate covariance matrix of variables"), never a bare scalar --
    // element [0,1] (== [1,0], symmetric) is the x-y cross-covariance.
    let cov_matrix = covariance(&x, Some(y)).expect("Covariance failed");
    let cov_xy = cov_matrix.get(&[0, 1]).expect("Failed to get cov[0,1]");

    // Perfect positive correlation should give cov = 4.0
    assert!((cov_xy - 4.0).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_covariance_independent() {
    let x = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Failed to create array");

    let y = WasmArray::from_vec(&vec![1.0, -1.0, 1.0, -1.0], &[4]).expect("Failed to create array");

    // See the comment in `test_covariance` on why this is a matrix element,
    // not the return value directly.
    let cov_matrix = covariance(&x, Some(y)).expect("Covariance failed");
    let cov_xy = cov_matrix.get(&[0, 1]).expect("Failed to get cov[0,1]");

    // Should be close to 0 for independent variables
    assert!(cov_xy.abs() < 0.1);
}

#[wasm_bindgen_test]
fn test_covariance_incompatible() {
    let x = WasmArray::ones(&[5]);
    let y = WasmArray::ones(&[4]);

    let result = covariance(&x, Some(y));
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_correlation() {
    let x =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let y =
        WasmArray::from_vec(&vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5]).expect("Failed to create array");

    // Like `covariance`, `correlation(x, Some(y))` for two 1-D inputs
    // returns the full 2x2 correlation matrix, not a bare scalar -- the
    // diagonal is always 1.0 (self-correlation), so [0,1] (== [1,0]) is the
    // actual x-y correlation coefficient.
    let corr_matrix = correlation(&x, Some(y)).expect("Correlation failed");
    let corr_xy = corr_matrix.get(&[0, 1]).expect("Failed to get corr[0,1]");

    // Perfect positive correlation should give corr = 1.0
    assert!((corr_xy - 1.0).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_correlation_negative() {
    let x =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("Failed to create array");

    let y =
        WasmArray::from_vec(&vec![10.0, 8.0, 6.0, 4.0, 2.0], &[5]).expect("Failed to create array");

    // See the comment in `test_correlation` on why this is a matrix
    // element, not the return value directly.
    let corr_matrix = correlation(&x, Some(y)).expect("Correlation failed");
    let corr_xy = corr_matrix.get(&[0, 1]).expect("Failed to get corr[0,1]");

    // Perfect negative correlation should give corr = -1.0
    assert!((corr_xy - (-1.0)).abs() < TOLERANCE);
}

#[wasm_bindgen_test]
fn test_correlation_incompatible() {
    let x = WasmArray::ones(&[5]);
    let y = WasmArray::ones(&[3]);

    let result = correlation(&x, Some(y));
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_corrcoef_matrix() {
    // `correlation(x, None)` forwards to `stats::correlation::corrcoef`,
    // whose `rowvar` parameter defaults to `true` when passed `None` (see
    // `cov`'s doc comment: "each row represents a variable, each column an
    // observation") -- so the *rows*, not columns, are the 3 features here.
    // Layout is 3 rows x 100 columns, row-major flat: 100 values for
    // feature 0, then 100 for feature 1, then 100 for feature 2.
    //
    // Three differently-shaped, non-constant features (deterministic rather
    // than `WasmArray::random`, which has no such constructor on
    // `WasmArray`) so the correlation matrix is well-defined (non-zero
    // variance) without being degenerate (perfectly co-linear) in every
    // entry.
    let n = 100;
    let mut raw = Vec::with_capacity(3 * n);
    for i in 0..n {
        raw.push(i as f64); // feature 0
    }
    for i in 0..n {
        raw.push(i as f64 * 0.7 + (i % 7) as f64); // feature 1
    }
    for i in 0..n {
        raw.push((i as f64 * 1.3).sin() * 10.0); // feature 2
    }
    let data = WasmArray::from_vec(&raw, &[3, n]).expect("Failed to create array");

    // The "correlation matrix" mode of `correlation` (the single free
    // function this module exposes -- there is no separate
    // `correlation_coefficient`) is selected by passing `None` for `y`.
    let corr_matrix = correlation(&data, None).expect("Corrcoef failed");

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

// `test_random_normal`, `test_random_uniform`, `test_random_uniform_range`
// and `test_random_uniform_invalid_range` were removed here (W6-TESTS
// orphaned-test pass): they called `random_normal`/`random_uniform`/
// `random_uniform_range`, none of which exist anywhere under `src/wasm/`
// (nor does a `WasmArray::random` constructor). This is a real gap in the
// WASM binding surface -- the module doc-comment at `tests/wasm/mod.rs`
// describes random-generation coverage that was apparently planned but
// never implemented -- not a test bug, and it is out of this pass's file
// ownership (tests/** only) to add that binding to `src/wasm/stats.rs`.

#[wasm_bindgen_test]
fn test_statistics_chain() {
    // Test chaining multiple statistical operations
    let arr = WasmArray::from_vec(
        &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[10],
    )
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
