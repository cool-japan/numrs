//! NumPy-pinned reference tests for `numrs2::stats::quantile` method=
//! interpolations (Hyndman & Fan taxonomy, NumPy >= 1.22).
//!
//! Reference values were generated with:
//! ```text
//! python3 -c 'import numpy as np; print(np.quantile(data, q, method=m))'
//! ```
//! using numpy 2.4.2. See the data/q sets below for the exact inputs.

use numrs2::array::Array;
use numrs2::stats::quantile::quantile;

const EPS: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64, ctx: &str) {
    assert!(
        (actual - expected).abs() < EPS,
        "{ctx}: expected {expected}, got {actual}"
    );
}

fn check_method(data: &[f64], qs: &[f64], method: &str, expected: &[f64], ctx: &str) {
    let a = Array::from_vec(data.to_vec());
    let q = Array::from_vec(qs.to_vec());
    let result =
        quantile(&a, &q, Some(method)).unwrap_or_else(|e| panic!("{ctx} ({method}) errored: {e}"));
    let got = result.to_vec();
    assert_eq!(
        got.len(),
        expected.len(),
        "{ctx} ({method}) length mismatch"
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close(
            g,
            e,
            &format!("{ctx} ({method}) at index {i} (q={})", qs[i]),
        );
    }
}

// n=5 (odd), sorted = [1, 2, 4, 7, 9]
const DATA_ODD: [f64; 5] = [2.0, 7.0, 1.0, 9.0, 4.0];
// n=6 (even), sorted = [1, 2, 4, 5, 7, 9]
const DATA_EVEN: [f64; 6] = [2.0, 7.0, 1.0, 9.0, 4.0, 5.0];
const QS: [f64; 7] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

#[test]
fn test_inverted_cdf_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "inverted_cdf",
        &[1.0, 1.0, 2.0, 4.0, 7.0, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "inverted_cdf",
        &[1.0, 1.0, 2.0, 4.0, 7.0, 9.0, 9.0],
        "even_n6",
    );
}

#[test]
fn test_averaged_inverted_cdf_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "averaged_inverted_cdf",
        &[1.0, 1.0, 2.0, 4.0, 7.0, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "averaged_inverted_cdf",
        &[1.0, 1.0, 2.0, 4.5, 7.0, 9.0, 9.0],
        "even_n6",
    );
}

#[test]
fn test_closest_observation_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "closest_observation",
        &[1.0, 1.0, 1.0, 2.0, 7.0, 7.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "closest_observation",
        &[1.0, 1.0, 2.0, 4.0, 5.0, 7.0, 9.0],
        "even_n6",
    );
}

#[test]
fn test_interpolated_inverted_cdf_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "interpolated_inverted_cdf",
        &[1.0, 1.0, 1.25, 3.0, 6.25, 8.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "interpolated_inverted_cdf",
        &[1.0, 1.0, 1.5, 4.0, 6.0, 7.800000000000001, 9.0],
        "even_n6",
    );
}

#[test]
fn test_hazen_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "hazen",
        &[1.0, 1.0, 1.75, 4.0, 7.5, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "hazen",
        &[1.0, 1.1, 2.0, 4.5, 7.0, 8.8, 9.0],
        "even_n6",
    );
}

#[test]
fn test_weibull_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "weibull",
        &[1.0, 1.0, 1.5, 4.0, 8.0, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "weibull",
        &[1.0, 1.0, 1.75, 4.5, 7.5, 9.0, 9.0],
        "even_n6",
    );
}

#[test]
fn test_linear_odd_even_is_default() {
    check_method(
        &DATA_ODD,
        &QS,
        "linear",
        &[1.0, 1.4, 2.0, 4.0, 7.0, 8.2, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "linear",
        &[1.0, 1.5, 2.5, 4.5, 6.5, 8.0, 9.0],
        "even_n6",
    );

    // Default method (None) must equal explicit "linear".
    let a = Array::from_vec(DATA_ODD.to_vec());
    let q = Array::from_vec(QS.to_vec());
    let default_result = quantile(&a, &q, None).expect("default quantile should succeed");
    let linear_result = quantile(&a, &q, Some("linear")).expect("linear quantile should succeed");
    assert_eq!(default_result.to_vec(), linear_result.to_vec());
}

#[test]
fn test_median_unbiased_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "median_unbiased",
        &[
            1.0,
            1.0,
            1.6666666666666667,
            4.0,
            7.666666666666666,
            9.0,
            9.0,
        ],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "median_unbiased",
        &[
            1.0,
            1.0,
            1.9166666666666667,
            4.5,
            7.166666666666666,
            9.0,
            9.0,
        ],
        "even_n6",
    );
}

#[test]
fn test_normal_unbiased_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "normal_unbiased",
        &[1.0, 1.0, 1.6875, 4.0, 7.625, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "normal_unbiased",
        &[1.0, 1.0, 1.9375, 4.5, 7.125, 9.0, 9.0],
        "even_n6",
    );
}

#[test]
fn test_legacy_lower_higher_midpoint_nearest_odd_even() {
    check_method(
        &DATA_ODD,
        &QS,
        "lower",
        &[1.0, 1.0, 2.0, 4.0, 7.0, 7.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "lower",
        &[1.0, 1.0, 2.0, 4.0, 5.0, 7.0, 9.0],
        "even_n6",
    );

    check_method(
        &DATA_ODD,
        &QS,
        "higher",
        &[1.0, 2.0, 2.0, 4.0, 7.0, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "higher",
        &[1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 9.0],
        "even_n6",
    );

    check_method(
        &DATA_ODD,
        &QS,
        "midpoint",
        &[1.0, 1.5, 2.0, 4.0, 7.0, 8.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "midpoint",
        &[1.0, 1.5, 3.0, 4.5, 6.0, 8.0, 9.0],
        "even_n6",
    );

    check_method(
        &DATA_ODD,
        &QS,
        "nearest",
        &[1.0, 1.0, 2.0, 4.0, 7.0, 9.0, 9.0],
        "odd_n5",
    );
    check_method(
        &DATA_EVEN,
        &QS,
        "nearest",
        &[1.0, 1.0, 2.0, 4.0, 7.0, 7.0, 9.0],
        "even_n6",
    );
}

/// `"nearest"` must break exact ties via round-half-to-even, matching
/// `numpy.around` -- NOT round-half-away-from-zero.
#[test]
fn test_nearest_round_half_to_even_ties() {
    let data = [10.0, 20.0, 30.0, 40.0, 50.0]; // n=5, (n-1)=4
                                               // (n-1)*q = 0.5, 1.5, 2.5, 3.5 -> round to even indices 0, 2, 2, 4
    check_method(
        &data,
        &[0.125, 0.375, 0.625, 0.875],
        "nearest",
        &[10.0, 30.0, 30.0, 50.0],
        "tie_break",
    );
}

#[test]
fn test_single_element_all_methods() {
    let methods = [
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "median_unbiased",
        "normal_unbiased",
    ];
    for m in methods {
        check_method(
            &[42.0],
            &[0.0, 0.5, 1.0],
            m,
            &[42.0, 42.0, 42.0],
            "single_elem",
        );
    }
}

#[test]
fn test_q_edges_zero_and_one_all_methods() {
    // q=0 must always be the minimum, q=1 must always be the maximum,
    // for every method, on both odd and even n.
    let methods = [
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "median_unbiased",
        "normal_unbiased",
    ];
    for m in methods {
        check_method(&DATA_ODD, &[0.0, 1.0], m, &[1.0, 9.0], "odd_n5_edges");
        check_method(&DATA_EVEN, &[0.0, 1.0], m, &[1.0, 9.0], "even_n6_edges");
    }
}

#[test]
fn test_nan_input_propagates_to_all_quantiles_regardless_of_method() {
    // NumPy: any NaN in the input makes every requested quantile NaN,
    // regardless of q or method.
    let methods = ["linear", "lower", "higher", "nearest", "inverted_cdf"];
    for m in methods {
        let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
        let q = Array::from_vec(vec![0.0, 0.5, 1.0]);
        let result = quantile(&a, &q, Some(m)).expect("quantile with NaN input should not error");
        for (i, v) in result.to_vec().into_iter().enumerate() {
            assert!(v.is_nan(), "method {m} index {i}: expected NaN, got {v}");
        }
    }
}

#[test]
fn test_nan_q_is_rejected() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let q = Array::from_vec(vec![f64::NAN]);
    assert!(quantile(&a, &q, None).is_err(), "NaN q should be rejected");
}

#[test]
fn test_out_of_range_q_is_rejected() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let too_high = Array::from_vec(vec![1.5]);
    assert!(quantile(&a, &too_high, None).is_err());
    let too_low = Array::from_vec(vec![-0.1]);
    assert!(quantile(&a, &too_low, None).is_err());
}

#[test]
fn test_invalid_method_is_rejected() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let q = Array::from_vec(vec![0.5]);
    assert!(quantile(&a, &q, Some("not_a_real_method")).is_err());
}

#[test]
fn test_empty_array_is_rejected() {
    let a: Array<f64> = Array::from_vec(vec![]);
    let q = Array::from_vec(vec![0.5]);
    assert!(quantile(&a, &q, None).is_err());
}
