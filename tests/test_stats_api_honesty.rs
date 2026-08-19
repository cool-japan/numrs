//! Regression tests for LANE W1-G: stats/math API honesty.
//!
//! These tests lock in the fixes for four dishonest APIs that accepted parameters and then
//! silently discarded them (or exposed an inconsistent type), verified against real NumPy
//! (numpy 2.4.2) where a numeric answer is asserted:
//!
//! 1. `stats::basic::average`'s dead `returned` parameter has been removed; the correct
//!    `returned=True` behavior lives in `average_with_weights`, which really does return the
//!    weight sum.
//! 2. `cumsum`/`cumprod` (and their `cumulative_sum`/`cumulative_prod` aliases) now honor
//!    `out`: shape-validate it, write the result into it, and return it — instead of accepting
//!    and discarding it.
//! 3. `argsort`/`sort`/`partition`/`argpartition` now honor `kind` (stability is genuinely
//!    honored for `sort`/`argsort`; `partition`/`argpartition` validate `kind` against the one
//!    algorithm they actually implement) and reject `order` with a clear "not implemented"
//!    error instead of silently ignoring structured-array sort keys that plain `Array<T>` has
//!    no way to honor.
//! 4. `argmax`/`argmin` now take `Option<isize>` for `axis`, consistent with every other
//!    axis-taking reduction in this crate, with negative axes normalized the same way.

use numrs2::error::NumRs2Error;
use numrs2::prelude::*;
use numrs2::stats::average_with_weights;

const EPS: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < EPS,
        "{msg}: expected {expected}, got {actual}"
    );
}

// ===========================================================================
// 1. average() — dead `returned` parameter removed; average_with_weights honest
// ===========================================================================

#[test]
fn test_average_weighted_matches_numpy() {
    // NumPy: np.average([1,2,3,4,5], weights=[5,4,3,2,1]) == 2.3333333333333335
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let weights = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

    // average() no longer takes a `returned` parameter — this is the whole signature now.
    let avg = average(&data, Some(&weights), None).expect("average should succeed");
    assert_close(avg.to_vec()[0], 2.3333333333333335, "weighted average");
}

#[test]
fn test_average_unweighted_matches_numpy() {
    // NumPy: np.average([1,2,3,4,5]) == 3.0
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let avg = average(&data, None, None).expect("average should succeed");
    assert_close(avg.to_vec()[0], 3.0, "unweighted average");
}

#[test]
fn test_average_with_weights_returns_correct_weight_sum() {
    // NumPy: np.average([1,2,3,4,5], weights=[5,4,3,2,1], returned=True)
    //        == (2.3333333333333335, 15.0)
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let weights = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

    // Note the parameter order: (a, axis, weights) — different from average()'s (a, weights, axis).
    let (avg, wsum) = average_with_weights(&data, None, Some(&weights))
        .expect("average_with_weights should succeed");
    assert_close(
        avg.to_vec()[0],
        2.3333333333333335,
        "average_with_weights avg",
    );
    assert_close(wsum.to_vec()[0], 15.0, "average_with_weights weight sum");
}

#[test]
fn test_average_with_weights_uniform_weight_sum_is_element_count() {
    // NumPy: np.average([1,2,3,4,5], returned=True) == (3.0, 5.0)
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (avg, wsum) =
        average_with_weights(&data, None, None).expect("average_with_weights should succeed");
    assert_close(avg.to_vec()[0], 3.0, "uniform average");
    assert_close(wsum.to_vec()[0], 5.0, "uniform weight sum (element count)");
}

// ===========================================================================
// 2. cumsum / cumprod / cumulative_sum / cumulative_prod — `out` honored
// ===========================================================================

#[test]
fn test_cumsum_out_is_written_and_returned() {
    // NumPy: np.cumsum([1,2,3,4]) == [1, 3, 6, 10]
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

    let returned = cumsum(&a, None, Some(&mut out)).expect("cumsum should succeed");
    assert_eq!(
        out.to_vec(),
        vec![1.0, 3.0, 6.0, 10.0],
        "out must contain the result"
    );
    assert_eq!(
        returned.to_vec(),
        vec![1.0, 3.0, 6.0, 10.0],
        "return value must match out"
    );
}

#[test]
fn test_cumsum_out_shape_mismatch_errors() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = Array::from_vec(vec![0.0, 0.0, 0.0]); // wrong length: 3 vs 4

    let err = cumsum(&a, None, Some(&mut out)).expect_err("shape mismatch must error");
    assert!(
        matches!(err, NumRs2Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn test_cumsum_out_honored_along_axis() {
    // NumPy: np.cumsum([[1,2,3],[4,5,6]], axis=1) == [[1,3,6],[4,9,15]]
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let mut out = Array::from_vec(vec![0.0; 6]).reshape(&[2, 3]);

    cumsum(&a, Some(1), Some(&mut out)).expect("cumsum should succeed");
    assert_eq!(out.to_vec(), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn test_cumprod_out_is_written_and_returned() {
    // NumPy: np.cumprod([1,2,3,4]) == [1, 2, 6, 24]
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

    let returned = cumprod(&a, None, Some(&mut out)).expect("cumprod should succeed");
    assert_eq!(
        out.to_vec(),
        vec![1.0, 2.0, 6.0, 24.0],
        "out must contain the result"
    );
    assert_eq!(
        returned.to_vec(),
        vec![1.0, 2.0, 6.0, 24.0],
        "return value must match out"
    );
}

#[test]
fn test_cumprod_out_shape_mismatch_errors() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]); // wrong length: 5 vs 4

    let err = cumprod(&a, None, Some(&mut out)).expect_err("shape mismatch must error");
    assert!(
        matches!(err, NumRs2Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn test_cumulative_sum_alias_out_is_written_and_returned() {
    // aggregation::cumulative_sum is a thin alias over statistics::cumsum; it must forward
    // `out` honestly, not just accept-and-drop it.
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

    let returned = cumulative_sum(&a, None, Some(&mut out)).expect("cumulative_sum should succeed");
    assert_eq!(out.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
    assert_eq!(returned.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn test_cumulative_prod_alias_out_is_written_and_returned() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

    let returned =
        cumulative_prod(&a, None, Some(&mut out)).expect("cumulative_prod should succeed");
    assert_eq!(out.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
    assert_eq!(returned.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
}

// ===========================================================================
// 3a. argsort — kind honored (stability observable via indices) + order rejected
// ===========================================================================

#[test]
fn test_argsort_kind_stable_preserves_input_order_for_ties() {
    // NumPy: np.argsort([5,3,5,1,3,5], kind='stable') == [3, 1, 4, 0, 2, 5]
    // (exact tie order is *required* for a stable sort: equal keys keep input order)
    let a = Array::from_vec(vec![5.0, 3.0, 5.0, 1.0, 3.0, 5.0]);
    let expected = vec![3usize, 1, 4, 0, 2, 5];

    for kind in [Some("stable"), Some("mergesort")] {
        let indices = argsort(&a, None, kind, None).expect("argsort should succeed");
        assert_eq!(indices.to_vec(), expected, "kind={kind:?}");
    }
}

#[test]
fn test_argsort_kind_quicksort_produces_correctly_sorted_output() {
    // Unstable sorts don't guarantee tie order, so only assert the defined contract:
    // a valid permutation whose values are non-decreasing.
    let a = Array::from_vec(vec![5.0, 3.0, 5.0, 1.0, 3.0, 5.0]);
    let data = a.to_vec();

    for kind in [None, Some("quicksort"), Some("heapsort")] {
        let indices = argsort(&a, None, kind, None).expect("argsort should succeed");
        let idx = indices.to_vec();

        let mut sorted_idx = idx.clone();
        sorted_idx.sort_unstable();
        assert_eq!(
            sorted_idx,
            vec![0, 1, 2, 3, 4, 5],
            "kind={kind:?} must be a permutation"
        );

        for w in idx.windows(2) {
            assert!(
                data[w[0]] <= data[w[1]],
                "kind={kind:?} must sort correctly"
            );
        }
    }
}

#[test]
fn test_argsort_invalid_kind_errors() {
    let a = Array::from_vec(vec![3.0, 1.0, 2.0]);
    let err = argsort(&a, None, Some("bogus"), None).expect_err("invalid kind must error");
    assert!(
        matches!(err, NumRs2Error::InvalidOperation(_)),
        "expected InvalidOperation, got {err:?}"
    );
}

#[test]
fn test_argsort_order_is_unsupported() {
    let a = Array::from_vec(vec![3.0, 1.0, 2.0]);
    let err = argsort(&a, None, None, Some(&["field"])).expect_err("order must error");
    assert!(
        matches!(err, NumRs2Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
}

// ===========================================================================
// 3b. sort — kind honored (correctness across kinds) + order rejected
// ===========================================================================

#[test]
fn test_sort_kind_stable_and_quicksort_correctness() {
    // NumPy: np.sort([5,3,5,1,3,5]) == [1, 3, 3, 5, 5, 5]
    let a = Array::from_vec(vec![5.0, 3.0, 5.0, 1.0, 3.0, 5.0]);
    let expected = vec![1.0, 3.0, 3.0, 5.0, 5.0, 5.0];

    for kind in [
        None,
        Some("quicksort"),
        Some("heapsort"),
        Some("mergesort"),
        Some("stable"),
    ] {
        let sorted = sort(&a, None, kind, None).expect("sort should succeed");
        assert_eq!(sorted.to_vec(), expected, "kind={kind:?}");
    }
}

#[test]
fn test_sort_invalid_kind_errors() {
    let a = Array::from_vec(vec![3.0, 1.0, 2.0]);
    let err = sort(&a, None, Some("bogus"), None).expect_err("invalid kind must error");
    assert!(
        matches!(err, NumRs2Error::InvalidOperation(_)),
        "expected InvalidOperation, got {err:?}"
    );
}

#[test]
fn test_sort_order_is_unsupported() {
    let a = Array::from_vec(vec![3.0, 1.0, 2.0]);
    let err = sort(&a, None, None, Some(&["field"])).expect_err("order must error");
    assert!(
        matches!(err, NumRs2Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
}

// ===========================================================================
// 3c. partition / argpartition — kind validated against the one algorithm used,
//     order rejected
// ===========================================================================

#[test]
fn test_partition_kind_none_and_introselect_are_accepted_and_correct() {
    // NumPy: np.partition([3,4,2,1,9,0], 2)[2] == 2.0 (the kth-smallest value)
    let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0, 9.0, 0.0]);
    let kth = 2;

    for kind in [None, Some("introselect")] {
        let result = partition(&a, kth, None, kind, None).expect("partition should succeed");
        let v = result.to_vec();
        let pivot = v[kth];
        assert_close(pivot, 2.0, "kth-smallest value");
        for &x in &v[..kth] {
            assert!(x <= pivot, "kind={kind:?}: left side must be <= pivot");
        }
        for &x in &v[kth + 1..] {
            assert!(x >= pivot, "kind={kind:?}: right side must be >= pivot");
        }
    }
}

#[test]
fn test_partition_invalid_kind_errors() {
    let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
    let err = partition(&a, 1, None, Some("quicksort"), None).expect_err("invalid kind must error");
    assert!(
        matches!(err, NumRs2Error::InvalidOperation(_)),
        "expected InvalidOperation, got {err:?}"
    );
}

#[test]
fn test_partition_order_is_unsupported() {
    let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
    let err = partition(&a, 1, None, None, Some(&["field"])).expect_err("order must error");
    assert!(
        matches!(err, NumRs2Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
}

#[test]
fn test_argpartition_kind_none_and_introselect_are_accepted_and_correct() {
    // NumPy: a[np.argpartition([3,4,2,1,9,0], 2)][2] == 2.0
    let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0, 9.0, 0.0]);
    let data = a.to_vec();
    let kth = 2;

    for kind in [None, Some("introselect")] {
        let indices = argpartition(&a, kth, None, kind, None).expect("argpartition should succeed");
        let idx = indices.to_vec();
        let pivot = data[idx[kth]];
        assert_close(pivot, 2.0, "kth-smallest value via indices");
        for &i in &idx[..kth] {
            assert!(
                data[i] <= pivot,
                "kind={kind:?}: left side must be <= pivot"
            );
        }
        for &i in &idx[kth + 1..] {
            assert!(
                data[i] >= pivot,
                "kind={kind:?}: right side must be >= pivot"
            );
        }
    }
}

#[test]
fn test_argpartition_invalid_kind_errors() {
    let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
    let err =
        argpartition(&a, 1, None, Some("mergesort"), None).expect_err("invalid kind must error");
    assert!(
        matches!(err, NumRs2Error::InvalidOperation(_)),
        "expected InvalidOperation, got {err:?}"
    );
}

#[test]
fn test_argpartition_order_is_unsupported() {
    let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
    let err = argpartition(&a, 1, None, None, Some(&["field"])).expect_err("order must error");
    assert!(
        matches!(err, NumRs2Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
}

// ===========================================================================
// 4. argmax / argmin — axis is now Option<isize>, negative axis normalized
// ===========================================================================

#[test]
fn test_argmax_negative_axis_matches_positive_axis() {
    // NumPy: np.argmax([[1,3,2],[4,5,1]], axis=1) == np.argmax(..., axis=-1) == [1, 1]
    // NumPy: np.argmax([[1,3,2],[4,5,1]], axis=0) == np.argmax(..., axis=-2) == [1, 1, 0]
    let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0]).reshape(&[2, 3]);

    let pos1 = argmax(&a, Some(1), false).expect("argmax should succeed");
    let neg1 = argmax(&a, Some(-1), false).expect("argmax should succeed");
    assert_eq!(pos1.to_vec(), neg1.to_vec());
    assert_eq!(pos1.to_vec(), vec![1, 1]);

    let pos0 = argmax(&a, Some(0), false).expect("argmax should succeed");
    let neg0 = argmax(&a, Some(-2), false).expect("argmax should succeed");
    assert_eq!(pos0.to_vec(), neg0.to_vec());
    assert_eq!(pos0.to_vec(), vec![1, 1, 0]);
}

#[test]
fn test_argmin_negative_axis_matches_positive_axis() {
    // NumPy: np.argmin([[5,3,2],[4,1,6]], axis=1) == np.argmin(..., axis=-1) == [2, 1]
    let a = Array::from_vec(vec![5.0, 3.0, 2.0, 4.0, 1.0, 6.0]).reshape(&[2, 3]);

    let pos = argmin(&a, Some(1), false).expect("argmin should succeed");
    let neg = argmin(&a, Some(-1), false).expect("argmin should succeed");
    assert_eq!(pos.to_vec(), neg.to_vec());
    assert_eq!(pos.to_vec(), vec![2, 1]);
}
