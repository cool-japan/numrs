//! `ddof` semantics for `var`/`std`: population at `ddof = 0`, sample at `ddof = 1`,
//! checked against hand-computed ground truth rather than against another kernel.
//!
//! Extracted verbatim from `mod.rs`; see that file's "Test modules" note.

use super::*;
use crate::array::Array;

/// Hand-computed (no kernel code involved) population/sample mean+variance for an
/// arbitrary `f64` slice, used as independent ground truth below.
fn hand_mean_var(data: &[f64], ddof: usize) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let sum_sq_dev: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
    (mean, sum_sq_dev / (n - ddof as f64))
}

/// Regression test for the population-vs-sample `var`/`std` bug (see this module's
/// `var` doc comment and `kernels::reduce`'s "never use `simd_variance`/`simd_std`"
/// module docs): at `n == 100` (past the old `len() >= 64` threshold that used to gate
/// the buggy sample-variance SIMD branch), `math::statistics::var`/`std` with `ddof = 0`
/// must match NumPy's population convention, not sample.
#[test]
fn var_std_ddof_0_is_population_at_n_100() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let (_, expected_pop_var) = hand_mean_var(&data, 0);
    // Closed form for a discrete uniform {0, ..., 99}: (n^2 - 1) / 12 = 833.25.
    assert!(
        (expected_pop_var - 833.25).abs() < 1e-9,
        "sanity: {expected_pop_var}"
    );

    let arr = Array::from_vec(data);
    let got_var = var(&arr, None, 0, false)
        .expect("var should succeed")
        .to_vec()[0];
    let got_std = std(&arr, None, 0, false)
        .expect("std should succeed")
        .to_vec()[0];

    assert!(
        (got_var - expected_pop_var).abs() < 1e-9,
        "population variance mismatch at n=100, ddof=0: got {got_var}, expected {expected_pop_var}"
    );
    assert!(
        (got_std - expected_pop_var.sqrt()).abs() < 1e-9,
        "population std mismatch at n=100, ddof=0: got {got_std}, expected {}",
        expected_pop_var.sqrt()
    );
}

/// The `ddof = 1` (sample variance, divisor `n - 1`) companion to the test above, at the
/// same `n == 100` -- explicitly required alongside the `ddof = 0` regression test, since
/// a fix that hardcodes population semantics (divisor always `n`, ignoring `ddof`) would
/// pass the `ddof = 0` test above but silently break every other `ddof` value.
#[test]
fn var_std_ddof_1_is_sample_at_n_100() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let (_, expected_sample_var) = hand_mean_var(&data, 1);
    // n/(n-1) times the population variance: 833.25 * 100/99.
    assert!(
        (expected_sample_var - 833.25 * 100.0 / 99.0).abs() < 1e-9,
        "sanity: {expected_sample_var}"
    );
    // Population and sample variance must differ by more than float noise at this n,
    // otherwise this test can't distinguish "honors ddof" from "always population".
    assert!((expected_sample_var - 833.25).abs() > 1e-3);

    let arr = Array::from_vec(data);
    let got_var = var(&arr, None, 1, false)
        .expect("var should succeed")
        .to_vec()[0];
    let got_std = std(&arr, None, 1, false)
        .expect("std should succeed")
        .to_vec()[0];

    assert!(
        (got_var - expected_sample_var).abs() < 1e-9,
        "sample variance mismatch at n=100, ddof=1: got {got_var}, expected {expected_sample_var}"
    );
    assert!(
        (got_std - expected_sample_var.sqrt()).abs() < 1e-9,
        "sample std mismatch at n=100, ddof=1: got {got_std}, expected {}",
        expected_sample_var.sqrt()
    );
}

/// Same `ddof = 1` check for `f32`, which shares the `T == f32` kernel-dispatch branch.
#[test]
fn var_std_ddof_1_is_sample_for_f32_at_n_100() {
    let data_f64: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let (_, expected_sample_var) = hand_mean_var(&data_f64, 1);

    let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();
    let arr = Array::from_vec(data_f32);
    let got_var = var(&arr, None, 1, false)
        .expect("var should succeed")
        .to_vec()[0];

    assert!(
        (got_var as f64 - expected_sample_var).abs() < 1e-1,
        "f32 sample variance mismatch at n=100, ddof=1: got {got_var}, expected {expected_sample_var}"
    );
}

/// Below the old 64-element SIMD threshold, sanity-check `ddof=1` too (small-n path).
#[test]
fn var_std_ddof_1_small_n() {
    // Sample variance (ddof=1) of [1,2,3,4,5] is 2.5 (population, ddof=0, is 2.0).
    let arr = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
    let got_var = var(&arr, None, 1, false)
        .expect("var should succeed")
        .to_vec()[0];
    assert!((got_var - 2.5).abs() < 1e-12);
}

#[test]
fn var_ddof_equal_to_n_errors() {
    let arr = Array::from_vec(vec![1.0f64, 2.0, 3.0]);
    assert!(var(&arr, None, 3, false).is_err());
    assert!(var(&arr, None, 10, false).is_err());
}
