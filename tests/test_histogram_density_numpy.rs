//! NumPy-pinned reference tests for `numrs2::stats` histogram functions:
//! `density=`, the `histogramdd` canonical entry point (with `histogram_dd`
//! kept as a backward-compatible alias), bin-edge semantics (right-open
//! except the last bin), and degenerate-range handling.
//!
//! Reference values were generated with `numpy 2.4.2` via
//! `python3 -c 'import numpy as np; print(np.histogram(...))'`.

use numrs2::array::Array;
use numrs2::stats::{histogram, histogram2d, histogram_dd, histogramdd};

const EPS: f64 = 1e-9;

fn assert_vec_close(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < EPS,
            "{ctx} at index {i}: expected {e}, got {a}"
        );
    }
}

// ---------------------------------------------------------------------
// histogram: density
// ---------------------------------------------------------------------

#[test]
fn test_histogram_density_matches_numpy() {
    // np.histogram([1,2,2,3,3,3,4], bins=4, density=True)
    let data = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]);
    let (counts, edges) = histogram(&data, 4, None, None, None).expect("counts should succeed");
    assert_vec_close(&counts.to_vec(), &[1.0, 2.0, 3.0, 1.0], "counts");
    assert_vec_close(&edges.to_vec(), &[1.0, 1.75, 2.5, 3.25, 4.0], "edges");

    let (density, edges_d) =
        histogram(&data, 4, None, None, Some(true)).expect("density should succeed");
    assert_vec_close(&edges_d.to_vec(), &edges.to_vec(), "density_edges");
    let expected_density = [
        0.19047619047619047,
        0.38095238095238093,
        0.5714285714285714,
        0.19047619047619047,
    ];
    assert_vec_close(&density.to_vec(), &expected_density, "density");

    // Density must integrate to 1 over the range.
    let widths: Vec<f64> = edges_d.to_vec().windows(2).map(|w| w[1] - w[0]).collect();
    let integral: f64 = density
        .to_vec()
        .iter()
        .zip(widths.iter())
        .map(|(&d, &w)| d * w)
        .sum();
    assert!(
        (integral - 1.0).abs() < EPS,
        "density should integrate to 1, got {integral}"
    );
}

#[test]
fn test_histogram_weighted_density_matches_numpy() {
    // np.histogram(data, bins=4, weights=w, density=True)
    let data = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]);
    let w = Array::from_vec(vec![1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0]);
    let (density, _) =
        histogram(&data, 4, None, Some(&w), Some(true)).expect("weighted density should succeed");
    let expected = [
        0.1212121212121212,
        0.36363636363636365,
        0.6060606060606061,
        0.2424242424242424,
    ];
    assert_vec_close(&density.to_vec(), &expected, "weighted_density");
}

#[test]
fn test_histogram_density_with_zero_total_is_nan_like_numpy() {
    // np.histogram([100.0, 200.0], bins=3, range=(0,3), density=True)
    // -> [nan, nan, nan] (0/0), not zeros: NumPy does not special-case an
    // empty-in-range histogram, it just lets the float division produce NaN.
    let data: Array<f64> = Array::from_vec(vec![100.0, 200.0]);
    let (density, _) = histogram(&data, 3, Some((0.0, 3.0)), None, Some(true))
        .expect("density should not error even when total is zero");
    for v in density.to_vec() {
        assert!(v.is_nan(), "expected NaN, got {v}");
    }
}

#[test]
fn test_histogram_density_false_or_none_equal_raw_counts() {
    let data = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0]);
    let (none_counts, _) = histogram(&data, 3, None, None, None).expect("ok");
    let (false_counts, _) = histogram(&data, 3, None, None, Some(false)).expect("ok");
    assert_eq!(none_counts.to_vec(), false_counts.to_vec());
}

// ---------------------------------------------------------------------
// histogram: bin-edge semantics (right-open except the last bin)
// ---------------------------------------------------------------------

#[test]
fn test_histogram_bin_edges_right_open_except_last() {
    // np.histogram([1,2,1], bins=[0,1,2,3]) == (array([0, 2, 1]), edges)
    // Reproduced here with uniform bins covering the same [0,3] range in 3
    // equal-width bins of width 1, so edges line up with the pinned values.
    let data = Array::from_vec(vec![1.0, 2.0, 1.0]);
    let (counts, edges) =
        histogram(&data, 3, Some((0.0, 3.0)), None, None).expect("histogram should succeed");
    assert_vec_close(&edges.to_vec(), &[0.0, 1.0, 2.0, 3.0], "edges");
    // value 1.0 falls in bin [1,2) -> bin index 1 (two occurrences);
    // value 2.0 falls in the *last*, closed bin [2,3] -> bin index 2.
    assert_vec_close(&counts.to_vec(), &[0.0, 2.0, 1.0], "counts");
}

#[test]
fn test_histogram_value_at_last_edge_included_in_last_bin() {
    // np.histogram([0,1,2,3], bins=3, range=(0,3)) == ([1,1,2], [0,1,2,3])
    // (the value 3.0 sits exactly on the last edge, and 2.0 also lands in
    // the closed last bin [2,3]).
    let data = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let (counts, _) =
        histogram(&data, 3, Some((0.0, 3.0)), None, None).expect("histogram should succeed");
    assert_vec_close(&counts.to_vec(), &[1.0, 1.0, 2.0], "counts_last_edge");
}

// ---------------------------------------------------------------------
// histogram: degenerate range (min == max)
// ---------------------------------------------------------------------

#[test]
fn test_histogram_degenerate_range_expands_by_half() {
    // np.histogram([5.0,5.0,5.0], bins=3) ==
    //   (array([0, 3, 0]), array([4.5, 4.8333.., 5.1666.., 5.5]))
    let data = Array::from_vec(vec![5.0, 5.0, 5.0]);
    let (counts, edges) = histogram(&data, 3, None, None, None).expect("should not panic or error");
    assert_vec_close(
        &edges.to_vec(),
        &[4.5, 4.833333333333333, 5.166666666666667, 5.5],
        "degenerate_edges",
    );
    assert_vec_close(&counts.to_vec(), &[0.0, 3.0, 0.0], "degenerate_counts");
}

#[test]
fn test_histogram_explicit_degenerate_range_is_expanded_not_rejected() {
    // NumPy expands an explicit degenerate range too, rather than
    // rejecting it (only min > max is an error).
    let data = Array::from_vec(vec![5.0, 5.0, 5.0]);
    let result = histogram(&data, 2, Some((5.0, 5.0)), None, None);
    assert!(
        result.is_ok(),
        "explicit min==max range should be expanded, not rejected"
    );
}

#[test]
fn test_histogram_min_greater_than_max_is_still_rejected() {
    let data = Array::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(histogram(&data, 2, Some((5.0, 1.0)), None, None).is_err());
}

// ---------------------------------------------------------------------
// histogram / histogram2d / histogram_bin_edges: NaN/infinite range must
// be a clean error, matching NumPy's `_get_outer_edges` (which raises
// "... range of [.., ..] is not finite" for both an auto-detected and an
// explicit non-finite range), and matching the same handling already
// implemented for histogramdd.
// ---------------------------------------------------------------------

#[test]
fn test_histogram_auto_range_nan_is_a_clean_error_not_a_panic() {
    // np.histogram([1.0, nan, 3.0], bins=3) raises
    // "autodetected range of [nan, nan] is not finite".
    let data = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
    let result = histogram(&data, 3, None, None, None);
    assert!(
        result.is_err(),
        "auto-range over NaN data should error, not panic or silently zero out"
    );
}

#[test]
fn test_histogram_explicit_nan_range_bound_is_rejected() {
    // np.histogram([1,2,3], bins=3, range=(nan, 5.0)) raises
    // "supplied range of [nan, 5.0] is not finite".
    let data = Array::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(histogram(&data, 3, Some((f64::NAN, 5.0)), None, None).is_err());
    assert!(histogram(&data, 3, Some((0.0, f64::NAN)), None, None).is_err());
}

#[test]
fn test_histogram_explicit_finite_range_with_nan_sample_excludes_it_not_an_error() {
    // A NaN *sample* (as opposed to a NaN *range bound*) is merely outside
    // any finite range and is silently excluded, matching
    // np.histogram([1.0, nan, 3.0], bins=3, range=(0,5)) ==
    //   (array([1, 1, 0]), array([0., 1.667, 3.333, 5.])).
    let data = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
    let (counts, edges) =
        histogram(&data, 3, Some((0.0, 5.0)), None, None).expect("finite range should succeed");
    assert_vec_close(
        &counts.to_vec(),
        &[1.0, 1.0, 0.0],
        "nan_sample_excluded_counts",
    );
    assert_vec_close(
        &edges.to_vec(),
        &[0.0, 1.6666666666666667, 3.3333333333333335, 5.0],
        "nan_sample_excluded_edges",
    );
}

#[test]
fn test_histogram2d_auto_range_nan_in_either_axis_is_a_clean_error() {
    // np.histogram2d([1.0, nan, 3.0], [1,2,3], bins=3) raises
    // "autodetected range of [nan, nan] is not finite".
    let x = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
    let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(
        histogram2d(&x, &y, 3, None, None, None).is_err(),
        "NaN in x with auto-range should error"
    );
    assert!(
        histogram2d(&y, &x, 3, None, None, None).is_err(),
        "NaN in y with auto-range should error"
    );
}

// ---------------------------------------------------------------------
// histogram2d: density
// ---------------------------------------------------------------------

#[test]
fn test_histogram2d_density_matches_numpy() {
    // np.histogram2d([1,2,2,3], [1,1,2,2], bins=2, density=True)
    let x = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0]);
    let y = Array::from_vec(vec![1.0, 1.0, 2.0, 2.0]);
    let (counts, x_edges, y_edges) =
        histogram2d(&x, &y, 2, None, None, None).expect("counts should succeed");
    assert_vec_close(&counts.to_vec(), &[1.0, 0.0, 1.0, 2.0], "h2d_counts");
    assert_vec_close(&x_edges.to_vec(), &[1.0, 2.0, 3.0], "h2d_x_edges");
    assert_vec_close(&y_edges.to_vec(), &[1.0, 1.5, 2.0], "h2d_y_edges");

    let (density, _, _) =
        histogram2d(&x, &y, 2, None, None, Some(true)).expect("density should succeed");
    assert_vec_close(&density.to_vec(), &[0.5, 0.0, 0.5, 1.0], "h2d_density");
}

// ---------------------------------------------------------------------
// histogramdd: canonical entry point, density, and histogram_dd delegation
// ---------------------------------------------------------------------

#[test]
fn test_histogramdd_matches_numpy_counts_and_edges() {
    let sample = Array::from_vec(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.3, 0.7]).reshape(&[4, 2]);
    let (hist, edges) =
        histogramdd(&sample, &[2, 2], None, None, None).expect("histogramdd should succeed");
    assert_eq!(hist.shape(), vec![2, 2]);
    assert_vec_close(&hist.to_vec(), &[1.0, 1.0, 0.0, 2.0], "histogramdd_counts");
    assert_vec_close(
        &edges[0].to_vec(),
        &[0.0, 0.5, 1.0],
        "histogramdd_edges_dim0",
    );
    assert_vec_close(
        &edges[1].to_vec(),
        &[0.0, 0.5, 1.0],
        "histogramdd_edges_dim1",
    );
}

#[test]
fn test_histogramdd_density_matches_histogram2d_density() {
    // Same data as the histogram2d density test above, fed through
    // histogramdd, must agree exactly.
    let sample = Array::from_vec(vec![1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0]).reshape(&[4, 2]);
    let (density, _) =
        histogramdd(&sample, &[2, 2], None, None, Some(true)).expect("density should succeed");
    assert_vec_close(
        &density.to_vec(),
        &[0.5, 0.0, 0.5, 1.0],
        "histogramdd_density",
    );
}

#[test]
fn test_histogram_dd_alias_delegates_to_histogramdd() {
    // histogram_dd (old 4-arg name) must produce identical results to
    // histogramdd(..., density=None).
    let sample = Array::from_vec(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.3, 0.7]).reshape(&[4, 2]);
    let (hist_alias, edges_alias) =
        histogram_dd(&sample, &[2, 2], None, None).expect("histogram_dd should succeed");
    let (hist_canonical, edges_canonical) =
        histogramdd(&sample, &[2, 2], None, None, None).expect("histogramdd should succeed");
    assert_eq!(hist_alias.to_vec(), hist_canonical.to_vec());
    for (a, c) in edges_alias.iter().zip(edges_canonical.iter()) {
        assert_eq!(a.to_vec(), c.to_vec());
    }
}

#[test]
fn test_histogramdd_no_longer_shifts_bin_edges_by_epsilon() {
    // Regression test: histogramdd's auto-range used to add a 1e-10
    // epsilon to the max, which shifted every bin edge away from the true
    // data range. Edges must now exactly match the data's min/max.
    let sample = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).reshape(&[3, 2]);
    let (_, edges) =
        histogramdd(&sample, &[2, 2], None, None, None).expect("histogramdd should succeed");
    assert_eq!(edges[0].to_vec(), vec![0.0, 1.0, 2.0]);
    assert_eq!(edges[1].to_vec(), vec![0.0, 1.0, 2.0]);
}

// ---------------------------------------------------------------------
// histogramdd: NaN handling must not panic.
// ---------------------------------------------------------------------

#[test]
fn test_histogramdd_nan_sample_excluded_with_explicit_range() {
    // np.histogramdd([[0,0],[nan,0.5],[1,1]], bins=2, range=[(0,1),(0,1)])
    // -> counts sum to 2 (the NaN row is silently excluded), not 3, and
    // (critically) this must not panic.
    let sample = Array::from_vec(vec![0.0, 0.0, f64::NAN, 0.5, 1.0, 1.0]).reshape(&[3, 2]);
    let (hist, _) = histogramdd(
        &sample,
        &[2, 2],
        Some(vec![(0.0, 1.0), (0.0, 1.0)]),
        None,
        None,
    )
    .expect("NaN row should be excluded, not panic");
    let total: f64 = hist.to_vec().iter().sum();
    assert_vec_close(&[total], &[2.0], "nan_excluded_total");
    assert_vec_close(&hist.to_vec(), &[1.0, 0.0, 0.0, 1.0], "nan_excluded_counts");
}

#[test]
fn test_histogramdd_nan_with_auto_range_is_a_clean_error_not_a_panic() {
    // np.histogramdd([[0,0],[nan,0.5],[1,1]], bins=2) raises
    // "autodetected range ... is not finite"; NumRs2 must return `Err`,
    // not panic, when the range has to be auto-detected from NaN data.
    let sample = Array::from_vec(vec![0.0, 0.0, f64::NAN, 0.5, 1.0, 1.0]).reshape(&[3, 2]);
    let result = histogramdd(&sample, &[2, 2], None, None, None);
    assert!(
        result.is_err(),
        "auto-range over NaN data should error, not panic or succeed"
    );
}

#[test]
fn test_histogramdd_explicit_nan_range_bound_is_rejected() {
    // np.histogramdd([[1,1],[2,2]], bins=2,
    //                 range=[(nan, 5.0), (0.0, 5.0)])
    // raises "supplied range of [nan, 5.0] is not finite".
    let sample = Array::from_vec(vec![1.0, 1.0, 2.0, 2.0]).reshape(&[2, 2]);
    let result = histogramdd(
        &sample,
        &[2, 2],
        Some(vec![(f64::NAN, 5.0), (0.0, 5.0)]),
        None,
        None,
    );
    assert!(
        result.is_err(),
        "explicit NaN range bound should error, not panic or succeed"
    );
}
