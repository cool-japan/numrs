//! NumPy-pinned reference tests for `numrs2`'s `pad` function: all 11
//! `numpy.pad` modes, `reflect_type='odd'`, asymmetric `end_values`, and
//! N-D corner-cascading semantics.
//!
//! Reference values were generated with `numpy 2.4.2` via
//! `python3 -c 'import numpy as np; print(np.pad(...))'`.

use numrs2::array::Array;
use numrs2::array_ops::manipulation::pad;

const EPS: f64 = 1e-9;

fn assert_vec_close(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < EPS,
            "{ctx} at index {i}: expected {e}, got {a} (full actual={actual:?}, expected={expected:?})"
        );
    }
}

// ---------------------------------------------------------------------
// Pre-existing modes: constant, edge, wrap (even/basic), reflect, symmetric
// ---------------------------------------------------------------------

#[test]
fn test_constant_mode_unchanged_behavior() {
    let a = Array::from_vec(vec![1, 2, 3]);
    let result =
        pad(&a, &[(2, 3)], "constant", Some((0, 0)), None, None).expect("pad should succeed");
    assert_eq!(result.to_vec(), vec![0, 0, 1, 2, 3, 0, 0, 0]);
}

#[test]
fn test_constant_mode_asymmetric_values() {
    // np.pad([1,2,3,4,5], (2,3), 'constant', constant_values=(4,6))
    let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
    let result =
        pad(&a, &[(2, 3)], "constant", Some((4, 6)), None, None).expect("pad should succeed");
    assert_eq!(result.to_vec(), vec![4, 4, 1, 2, 3, 4, 5, 6, 6, 6]);
}

#[test]
fn test_edge_mode_2d_corners() {
    // np.pad([[1,2],[3,4]], ((1,1),(1,1)), mode='edge')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let result = pad(&a, &[(1, 1), (1, 1)], "edge", None, None, None).expect("pad should succeed");
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "edge_2d");
}

// ---------------------------------------------------------------------
// New stat modes: maximum, mean, median, minimum -- 2D discriminating
// corner test (rows and columns have different stats, so a correct
// implementation must reproduce NumPy's axis-cascading corner behavior).
// ---------------------------------------------------------------------

const STAT_INPUT: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]; // shape (2,3)

#[test]
fn test_mean_mode_2d_corner_cascade() {
    let a = Array::from_vec(STAT_INPUT.to_vec()).reshape(&[2, 3]);
    let result = pad(&a, &[(1, 1), (1, 1)], "mean", None, None, None).expect("pad should succeed");
    let expected = [
        4.0, 2.5, 3.5, 6.0, 4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 6.0, 4.0, 5.0, 9.0, 6.0, 4.0, 2.5, 3.5,
        6.0, 4.0,
    ];
    assert_eq!(result.shape(), vec![4, 5]);
    assert_vec_close(&result.to_vec(), &expected, "mean_2d");
}

#[test]
fn test_maximum_mode_2d_corner_cascade() {
    let a = Array::from_vec(STAT_INPUT.to_vec()).reshape(&[2, 3]);
    let result =
        pad(&a, &[(1, 1), (1, 1)], "maximum", None, None, None).expect("pad should succeed");
    let expected = [
        9.0, 4.0, 5.0, 9.0, 9.0, 3.0, 1.0, 2.0, 3.0, 3.0, 9.0, 4.0, 5.0, 9.0, 9.0, 9.0, 4.0, 5.0,
        9.0, 9.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "maximum_2d");
}

#[test]
fn test_median_mode_2d_corner_cascade() {
    let a = Array::from_vec(STAT_INPUT.to_vec()).reshape(&[2, 3]);
    let result =
        pad(&a, &[(1, 1), (1, 1)], "median", None, None, None).expect("pad should succeed");
    let expected = [
        3.5, 2.5, 3.5, 6.0, 3.5, 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 9.0, 5.0, 3.5, 2.5, 3.5,
        6.0, 3.5,
    ];
    assert_vec_close(&result.to_vec(), &expected, "median_2d");
}

#[test]
fn test_minimum_mode_2d_corner_cascade() {
    let a = Array::from_vec(STAT_INPUT.to_vec()).reshape(&[2, 3]);
    let result =
        pad(&a, &[(1, 1), (1, 1)], "minimum", None, None, None).expect("pad should succeed");
    let expected = [
        1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 4.0, 4.0, 5.0, 9.0, 4.0, 1.0, 1.0, 2.0,
        3.0, 1.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "minimum_2d");
}

// ---------------------------------------------------------------------
// stat modes (maximum, mean, median, minimum): NaN propagates, matching
// NumPy, regardless of NaN's position in the lane.
// ---------------------------------------------------------------------

#[test]
fn test_stat_modes_propagate_nan_regardless_of_position() {
    // np.pad([1.0, 5.0, nan, 3.0], (1,1), mode=...) for each stat mode ->
    // [nan, 1.0, 5.0, nan, 3.0, nan]: only the *padding* slots (indices 0
    // and 5) take the computed statistic -- which is NaN, since the whole
    // original axis is one NaN-tainted lane (`stat_length=None`
    // semantics, no partial window) -- while the original data (indices
    // 1..=4, including its own interior NaN at index 2) passes through
    // unchanged, exactly like every other pad mode.
    let a = Array::from_vec(vec![1.0, 5.0, f64::NAN, 3.0]);
    let expected_nan_at = [true, false, false, true, false, true];
    let expected_finite = [f64::NAN, 1.0, 5.0, f64::NAN, 3.0, f64::NAN];
    for mode in ["maximum", "minimum", "median", "mean"] {
        let result =
            pad(&a, &[(1, 1)], mode, None, None, None).unwrap_or_else(|e| panic!("{mode}: {e}"));
        let data = result.to_vec();
        assert_eq!(data.len(), 6, "{mode}: length");
        for (i, &v) in data.iter().enumerate() {
            if expected_nan_at[i] {
                assert!(v.is_nan(), "{mode} at index {i}: expected NaN, got {v}");
            } else {
                assert!(
                    (v - expected_finite[i]).abs() < EPS,
                    "{mode} at index {i}: expected {}, got {v}",
                    expected_finite[i]
                );
            }
        }
    }
}

#[test]
fn test_maximum_minimum_nan_not_only_when_first() {
    // Regression test: a naive `fold(lane[0], |acc, v| if v > acc {v} else
    // {acc})` only surfaces NaN in the *computed statistic* when NaN
    // happens to be the very first element (every other position's
    // `v > acc`/`v < acc` comparison against NaN is false, silently
    // dropping it instead of propagating). NaN at a non-leading position
    // must still poison the padding, matching
    // `np.pad([1.0, 3.0, nan], (1,1), mode='maximum')` /
    // `mode='minimum')` -> `[nan, 1.0, 3.0, nan, nan]` for both: original
    // data (indices 1..=3, including its own interior NaN at index 3)
    // passes through unchanged; only the two padding slots -- indices 0
    // and 4 -- take the poisoned statistic.
    let a = Array::from_vec(vec![1.0, 3.0, f64::NAN]);
    for mode in ["maximum", "minimum"] {
        let result =
            pad(&a, &[(1, 1)], mode, None, None, None).unwrap_or_else(|e| panic!("{mode}: {e}"));
        let data = result.to_vec();
        assert_eq!(data.len(), 5, "{mode}: length");
        assert!(
            data[0].is_nan(),
            "{mode} padding (before): expected NaN, got {}",
            data[0]
        );
        assert!(
            (data[1] - 1.0).abs() < EPS,
            "{mode} original[0]: expected 1.0, got {}",
            data[1]
        );
        assert!(
            (data[2] - 3.0).abs() < EPS,
            "{mode} original[1]: expected 3.0, got {}",
            data[2]
        );
        assert!(
            data[3].is_nan(),
            "{mode} original[2] (NaN passthrough): expected NaN, got {}",
            data[3]
        );
        assert!(
            data[4].is_nan(),
            "{mode} padding (after): expected NaN, got {}",
            data[4]
        );
    }
}

// ---------------------------------------------------------------------
// linear_ramp
// ---------------------------------------------------------------------

#[test]
fn test_linear_ramp_1d_default_end_value_zero() {
    // np.pad([1,2,3,4,5], (2,3), mode='linear_ramp')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = pad(&a, &[(2, 3)], "linear_ramp", None, None, None).expect("pad should succeed");
    let expected = [
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        3.3333333333333335,
        1.6666666666666667,
        0.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "linear_ramp_1d");
}

#[test]
fn test_linear_ramp_1d_asymmetric_end_values() {
    // np.pad([1,2,3,4,5], (2,3), 'linear_ramp', end_values=(5,-4))
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = pad(&a, &[(2, 3)], "linear_ramp", None, Some((5.0, -4.0)), None)
        .expect("pad should succeed");
    let expected = [5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, -1.0, -4.0];
    assert_vec_close(&result.to_vec(), &expected, "linear_ramp_asymmetric");
}

#[test]
fn test_linear_ramp_2d() {
    // np.pad([[1,2],[3,4]], ((1,1),(2,2)), mode='linear_ramp', end_values=0)
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let result = pad(
        &a,
        &[(1, 1), (2, 2)],
        "linear_ramp",
        None,
        Some((0.0, 0.0)),
        None,
    )
    .expect("pad should succeed");
    let expected = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.5, 1.0, 2.0, 1.0, 0.0, //
        0.0, 1.5, 3.0, 4.0, 2.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(result.shape(), vec![4, 6]);
    assert_vec_close(&result.to_vec(), &expected, "linear_ramp_2d");
}

// ---------------------------------------------------------------------
// reflect / symmetric, even and odd, including multi-period cases
// ---------------------------------------------------------------------

#[test]
fn test_reflect_even_basic() {
    // np.pad([1,2,3,4,5], (2,3), mode='reflect')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = pad(&a, &[(2, 3)], "reflect", None, None, None).expect("pad should succeed");
    assert_vec_close(
        &result.to_vec(),
        &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],
        "reflect_even",
    );
}

#[test]
fn test_symmetric_even_basic() {
    // np.pad([1,2,3,4,5], (2,3), mode='symmetric')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = pad(&a, &[(2, 3)], "symmetric", None, None, None).expect("pad should succeed");
    assert_vec_close(
        &result.to_vec(),
        &[2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0],
        "symmetric_even",
    );
}

#[test]
fn test_reflect_odd_multi_period() {
    // np.pad([1,2,3], (7,7), mode='reflect', reflect_type='odd')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let result =
        pad(&a, &[(7, 7)], "reflect", None, None, Some("odd")).expect("pad should succeed");
    let expected: Vec<f64> = (-6..=10).map(|v| v as f64).collect();
    assert_vec_close(&result.to_vec(), &expected, "reflect_odd_multiperiod");
}

#[test]
fn test_reflect_even_multi_period() {
    // np.pad([1,2,3], (7,7), mode='reflect', reflect_type='even')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let result =
        pad(&a, &[(7, 7)], "reflect", None, None, Some("even")).expect("pad should succeed");
    let expected = [
        2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "reflect_even_multiperiod");
}

#[test]
fn test_symmetric_odd_multi_period() {
    // np.pad([1,2,3], (7,7), mode='symmetric', reflect_type='odd')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let result =
        pad(&a, &[(7, 7)], "symmetric", None, None, Some("odd")).expect("pad should succeed");
    let expected = [
        -3.0, -3.0, -2.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "symmetric_odd_multiperiod");
}

#[test]
fn test_symmetric_even_multi_period() {
    // np.pad([1,2,3], (7,7), mode='symmetric', reflect_type='even')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let result = pad(&a, &[(7, 7)], "symmetric", None, None, None).expect("pad should succeed");
    let expected = [
        1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "symmetric_even_multiperiod");
}

#[test]
fn test_reflect_symmetric_axis_size_one_degenerates_to_edge() {
    // np.pad([5.0], 3, mode='reflect') == np.pad([5.0], 3, mode='symmetric')
    // == np.pad([5.0], 3, mode='reflect', reflect_type='odd') == all 5s.
    let a = Array::from_vec(vec![5.0]);
    for (mode, reflect_type) in [
        ("reflect", None),
        ("symmetric", None),
        ("reflect", Some("odd")),
        ("symmetric", Some("odd")),
    ] {
        let result = pad(&a, &[(3, 3)], mode, None, None, reflect_type).unwrap_or_else(|e| {
            panic!("mode {mode} (reflect_type {reflect_type:?}) should succeed: {e}")
        });
        assert_vec_close(
            &result.to_vec(),
            &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            &format!("axis_size_one_{mode}_{reflect_type:?}"),
        );
    }
}

#[test]
fn test_reflect_type_only_validated_for_reflect_and_symmetric_modes() {
    // An irrelevant / bogus reflect_type must not reject an unrelated mode.
    let a = Array::from_vec(vec![1, 2, 3]);
    assert!(pad(
        &a,
        &[(1, 1)],
        "constant",
        None,
        None,
        Some("not_even_or_odd")
    )
    .is_ok());

    // But it IS validated when the mode actually consults it.
    assert!(pad(
        &a,
        &[(1, 1)],
        "reflect",
        None,
        None,
        Some("not_even_or_odd")
    )
    .is_err());
}

// ---------------------------------------------------------------------
// wrap, including the "before is a multiple of axis_size" edge case that
// a naive single-modulo implementation gets wrong.
// ---------------------------------------------------------------------

#[test]
fn test_wrap_basic() {
    // np.pad([1,2,3,4,5], (2,3), mode='wrap')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = pad(&a, &[(2, 3)], "wrap", None, None, None).expect("pad should succeed");
    assert_vec_close(
        &result.to_vec(),
        &[4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0],
        "wrap_basic",
    );
}

#[test]
fn test_wrap_before_multiple_of_axis_size() {
    // np.pad([1,2,3], (3,0), mode='wrap')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let result = pad(&a, &[(3, 0)], "wrap", None, None, None).expect("pad should succeed");
    assert_vec_close(
        &result.to_vec(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "wrap_multiple",
    );
}

#[test]
fn test_wrap_multi_period() {
    // np.pad([1,2,3], (7,5), mode='wrap')
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let result = pad(&a, &[(7, 5)], "wrap", None, None, None).expect("pad should succeed");
    let expected = [
        3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0,
    ];
    assert_vec_close(&result.to_vec(), &expected, "wrap_multiperiod");
}

// ---------------------------------------------------------------------
// empty mode: only shape + original-data placement are guaranteed.
// ---------------------------------------------------------------------

#[test]
fn test_empty_mode_shape_and_original_preserved() {
    let a = Array::from_vec(vec![1, 2, 3]);
    let result = pad(&a, &[(2, 2)], "empty", None, None, None).expect("pad should succeed");
    assert_eq!(result.shape(), vec![7]);
    let data = result.to_vec();
    assert_eq!(&data[2..5], &[1, 2, 3]);
}

// ---------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------

#[test]
fn test_unknown_mode_is_rejected() {
    let a = Array::from_vec(vec![1, 2, 3]);
    assert!(pad(&a, &[(1, 1)], "not_a_real_mode", None, None, None).is_err());
}

#[test]
fn test_mismatched_pad_width_length_is_rejected() {
    let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    assert!(pad(&a, &[(1, 1)], "constant", None, None, None).is_err());
}

#[test]
fn test_empty_axis_rejected_for_non_constant_non_empty_modes() {
    let a: Array<f64> =
        Array::from_vec_shape(vec![], &[0, 3]).expect("zero-size array should build");
    assert!(pad(&a, &[(1, 0), (0, 0)], "edge", None, None, None).is_err());
    assert!(pad(&a, &[(1, 0), (0, 0)], "constant", None, None, None).is_ok());
}
