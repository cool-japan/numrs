//! Shape-statistic tests for `skew`/`kurtosis`.
//!
//! Extracted verbatim from `mod.rs`; see that file's "Test modules" note.

use super::*;
use crate::array::Array;

#[test]
fn test_skew_symmetric_data() {
    // Symmetric data should have skewness ≈ 0
    let data = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
    let s = skew(&data, None).expect("skew should succeed");
    assert!(
        s.to_vec()[0].abs() < 1e-10,
        "symmetric distribution skewness should be ~0, got {}",
        s.to_vec()[0]
    );
}

#[test]
fn test_skew_right_skewed() {
    // Right-skewed data (long tail on the right) should have positive skewness
    let data = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 10.0]);
    let s = skew(&data, None).expect("skew should succeed");
    assert!(
        s.to_vec()[0] > 0.0,
        "right-skewed data should have positive skewness"
    );
}

#[test]
fn test_skew_constant_array_returns_zero() {
    let data = Array::from_vec(vec![5.0_f64, 5.0, 5.0, 5.0]);
    let s = skew(&data, None).expect("constant array skew should succeed");
    assert_eq!(s.to_vec()[0], 0.0);
}

#[test]
fn test_skew_too_few_elements_errors() {
    let data = Array::from_vec(vec![42.0_f64]);
    assert!(
        skew(&data, None).is_err(),
        "skew with 1 element should return error"
    );
}

#[test]
fn test_kurtosis_uniform_like_data() {
    // For uniform-like data the excess kurtosis should be negative (platykurtic)
    let data = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
    let k = kurtosis(&data, None).expect("kurtosis should succeed");
    assert!(
        k.to_vec()[0] < 0.0,
        "uniform-like data should have negative excess kurtosis, got {}",
        k.to_vec()[0]
    );
}

#[test]
fn test_kurtosis_constant_array_returns_zero() {
    let data = Array::from_vec(vec![3.0_f64, 3.0, 3.0, 3.0]);
    let k = kurtosis(&data, None).expect("constant array kurtosis should succeed");
    assert_eq!(k.to_vec()[0], 0.0);
}

#[test]
fn test_kurtosis_too_few_elements_errors() {
    let data = Array::from_vec(vec![99.0_f64]);
    assert!(
        kurtosis(&data, None).is_err(),
        "kurtosis with 1 element should return error"
    );
}

#[test]
fn test_skew_empty_array_errors() {
    let data: Array<f64> = Array::from_vec(vec![]);
    assert!(
        skew(&data, None).is_err(),
        "skew on empty array should fail"
    );
}

#[test]
fn test_kurtosis_empty_array_errors() {
    let data: Array<f64> = Array::from_vec(vec![]);
    assert!(
        kurtosis(&data, None).is_err(),
        "kurtosis on empty array should fail"
    );
}
