//! Regression tests for `numrs2::stride_tricks`.
//!
//! `as_strided` and `sliding_window_view` are general N-D strided-gather
//! operations. Every expected value below was hand-computed against the
//! documented row-major (C-order) semantics and cross-checked against the
//! equivalent NumPy call (`numpy.lib.stride_tricks.as_strided` /
//! `numpy.lib.stride_tricks.sliding_window_view`), noted per test.
//!
//! Convention used throughout: `strides` for `as_strided` are ELEMENT
//! offsets into the flattened (row-major) buffer of the input array, i.e.
//! `flat_offset = sum(idx[d] * strides[d])`.

use numrs2::array::Array;
use numrs2::stride_tricks::{as_strided, broadcast_arrays, broadcast_to, sliding_window_view};

// ---------------------------------------------------------------------
// as_strided: the corrected rustdoc example
// ---------------------------------------------------------------------

// NumPy reference:
//   a = np.arange(1, 10).reshape(3, 3)          # strides (elements) = (3, 1)
//   as_strided(a, shape=(2, 2), strides=(2*3, 2*1))  # -> [[1, 3], [7, 9]]
#[test]
fn as_strided_doc_example_subsamples_corners() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

    let strided = as_strided(&array, &[2, 2], &[6, 2]).expect("as_strided should succeed");

    assert_eq!(strided.shape(), vec![2, 2]);
    assert_eq!(strided.to_vec(), vec![1, 3, 7, 9]);
}

// Pins down the literal `strides = [2, 2]` reading so the "strides are raw
// element offsets, not per-axis subsample factors" contract is unambiguous
// and independently regression-tested (as opposed to only the [6, 2] case
// above, which happens to equal 2x the array's natural strides).
#[test]
fn as_strided_literal_two_two_strides_are_element_offsets() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

    let strided = as_strided(&array, &[2, 2], &[2, 2]).expect("as_strided should succeed");

    // offset(i,j) = i*2 + j*2 into flat [1..=9]:
    // (0,0)->0->1, (0,1)->2->3, (1,0)->2->3, (1,1)->4->5
    assert_eq!(strided.shape(), vec![2, 2]);
    assert_eq!(strided.to_vec(), vec![1, 3, 3, 5]);
}

// ---------------------------------------------------------------------
// as_strided: overlapping strided views
// ---------------------------------------------------------------------

// NumPy reference: sliding_window_view(np.array([1,2,3,4,5]), 2)
//   -> [[1,2],[2,3],[3,4],[4,5]]
// built manually via as_strided(a, shape=(4,2), strides=(1,1)).
#[test]
fn as_strided_overlapping_windows_from_1d() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5]);

    let strided = as_strided(&array, &[4, 2], &[1, 1]).expect("as_strided should succeed");

    assert_eq!(strided.shape(), vec![4, 2]);
    assert_eq!(strided.to_vec(), vec![1, 2, 2, 3, 3, 4, 4, 5]);
}

// ---------------------------------------------------------------------
// as_strided: stride 0 (NumPy allows this -- it is the mechanism NumPy
// itself uses internally for broadcast_to / broadcast_arrays: a stride of
// 0 along an axis repeats the same element for every index on that axis).
// ---------------------------------------------------------------------

// NumPy reference:
//   a = np.array([10, 20, 30])
//   as_strided(a, shape=(3, 3), strides=(0, 8))  # element strides (0, 1)
//   -> [[10,20,30],[10,20,30],[10,20,30]]
#[test]
fn as_strided_stride_zero_repeats_row() {
    let array = Array::from_vec(vec![10, 20, 30]);

    let strided = as_strided(&array, &[3, 3], &[0, 1]).expect("as_strided should succeed");

    assert_eq!(strided.shape(), vec![3, 3]);
    assert_eq!(strided.to_vec(), vec![10, 20, 30, 10, 20, 30, 10, 20, 30]);
}

// When a dimension has size 1, its stride value can never be multiplied by
// a nonzero index, so it is effectively "don't care" -- NumPy accepts
// arbitrary (including negative) strides there too.
#[test]
fn as_strided_degenerate_dimension_stride_is_dont_care() {
    let array = Array::from_vec(vec![1, 2, 3]);

    let strided = as_strided(&array, &[1, 3], &[-5, 1]).expect("as_strided should succeed");

    assert_eq!(strided.shape(), vec![1, 3]);
    assert_eq!(strided.to_vec(), vec![1, 2, 3]);
}

// ---------------------------------------------------------------------
// as_strided: bounds-violation returns Err, never panics/garbage
// ---------------------------------------------------------------------

#[test]
fn as_strided_positive_bounds_violation_returns_err() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5]);

    // max reachable offset = (10-1)*1 = 9, buffer only has 5 elements.
    let result = as_strided(&array, &[10], &[1]);

    assert!(result.is_err());
}

#[test]
fn as_strided_negative_bounds_violation_returns_err() {
    let array = Array::from_vec(vec![1, 2, 3]);

    // min reachable offset = (5-1)*(-1) = -4, which is negative.
    let result = as_strided(&array, &[5], &[-1]);

    assert!(result.is_err());
}

#[test]
fn as_strided_dimension_mismatch_returns_err() {
    let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);

    let result = as_strided(&array, &[2, 2], &[1]);

    assert!(result.is_err());
}

// Zero-size output dimensions must produce an empty array, not panic --
// verified empirically (reshape() panics on size mismatch, so this proves
// the 0-length Vec path is actually exercised end to end).
#[test]
fn as_strided_zero_size_output_returns_empty() {
    let array = Array::from_vec(vec![1, 2, 3]);

    let strided = as_strided(&array, &[0, 3], &[1, 1]).expect("as_strided should succeed");

    assert_eq!(strided.shape(), vec![0, 3]);
    assert_eq!(strided.to_vec(), Vec::<i32>::new());
}

// ---------------------------------------------------------------------
// as_strided: general N-D (3-D case, hand-computed)
// ---------------------------------------------------------------------

// Logical layout: treat the 24 flat elements as if shaped [4,3,2]
// (natural element strides [6,2,1]) and pick out planes 0 and 2 along the
// first axis via strides=(12,2,1) (12 = 2*6, i.e. skip one plane).
//
// plane i=0 (flat[0..6])  -> values 1..=6
// plane i=1 (flat[6..12]) -> values 7..=12   (skipped)
// plane i=2 (flat[12..18])-> values 13..=18
// plane i=3 (flat[18..24])-> values 19..=24  (skipped)
#[test]
fn as_strided_3d_subsamples_planes() {
    let array = Array::from_vec((1..=24).collect::<Vec<i32>>());

    let strided = as_strided(&array, &[2, 3, 2], &[12, 2, 1]).expect("as_strided should succeed");

    assert_eq!(strided.shape(), vec![2, 3, 2]);
    assert_eq!(
        strided.to_vec(),
        vec![1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18]
    );
}

// ---------------------------------------------------------------------
// sliding_window_view: 1-D / 2-D behavior (must match the previously-real
// special-cased implementation, and match NumPy).
// ---------------------------------------------------------------------

// NumPy reference: sliding_window_view(np.array([1,2,3,4,5]), 2)
#[test]
fn sliding_window_view_1d_matches_reference() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5]);

    let windows =
        sliding_window_view(&array, &[2], None).expect("sliding_window_view should succeed");

    assert_eq!(windows.shape(), vec![4, 2]);
    assert_eq!(windows.to_vec(), vec![1, 2, 2, 3, 3, 4, 4, 5]);
}

#[test]
fn sliding_window_view_1d_with_step() {
    let array = Array::from_vec((1..=7).collect::<Vec<i32>>());

    let windows =
        sliding_window_view(&array, &[2], Some(&[2])).expect("sliding_window_view should succeed");

    // n_windows = (7-2)/2 + 1 = 3, starts at 0, 2, 4
    assert_eq!(windows.shape(), vec![3, 2]);
    assert_eq!(windows.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

// NumPy reference (numpy.lib.stride_tricks.sliding_window_view docs, shifted
// by +1 since this array is 1-indexed):
//   x = np.arange(1, 10).reshape(3, 3)
//   sliding_window_view(x, (2, 2))
//   -> [[[[1,2],[4,5]], [[2,3],[5,6]]],
//       [[[4,5],[7,8]], [[5,6],[8,9]]]]
#[test]
fn sliding_window_view_2d_matches_numpy_docs() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

    let windows =
        sliding_window_view(&array, &[2, 2], None).expect("sliding_window_view should succeed");

    assert_eq!(windows.shape(), vec![2, 2, 2, 2]);
    assert_eq!(
        windows.to_vec(),
        vec![1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9]
    );
}

// ---------------------------------------------------------------------
// sliding_window_view: general N-D (3-D case, hand-computed)
// ---------------------------------------------------------------------

// Input shaped [3,2,2] (values 1..=12, natural element strides [4,2,1]):
//   plane i=0 = [[1,2],[3,4]]
//   plane i=1 = [[5,6],[7,8]]
//   plane i=2 = [[9,10],[11,12]]
// window_shape=[2,2,2], step=None -> n_windows=[2,1,1], output shape
// [2,1,1,2,2,2]:
//   window(0,0,0) = planes {0,1} = [[1,2],[3,4]],[[5,6],[7,8]]
//   window(1,0,0) = planes {1,2} = [[5,6],[7,8]],[[9,10],[11,12]]
#[test]
fn sliding_window_view_3d_hand_computed() {
    let array = Array::from_vec((1..=12).collect::<Vec<i32>>()).reshape(&[3, 2, 2]);

    let windows =
        sliding_window_view(&array, &[2, 2, 2], None).expect("sliding_window_view should succeed");

    assert_eq!(windows.shape(), vec![2, 1, 1, 2, 2, 2]);
    assert_eq!(
        windows.to_vec(),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12]
    );
}

// ---------------------------------------------------------------------
// sliding_window_view: error paths
// ---------------------------------------------------------------------

#[test]
fn sliding_window_view_errors_when_window_exceeds_dim() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

    let result = sliding_window_view(&array, &[4, 2], None);

    assert!(result.is_err());
}

#[test]
fn sliding_window_view_errors_on_window_ndim_mismatch() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

    let result = sliding_window_view(&array, &[2], None);

    assert!(result.is_err());
}

// Regression test for a latent division-by-zero panic in the previous
// implementation: `(dim_size - window_size) / step_size` with
// `step_size == 0` must be a clean Err, not a crash.
#[test]
fn sliding_window_view_errors_on_zero_step_instead_of_panicking() {
    let array = Array::from_vec(vec![1, 2, 3, 4, 5]);

    let result = sliding_window_view(&array, &[2], Some(&[0]));

    assert!(result.is_err());
}

// ---------------------------------------------------------------------
// broadcast_to / broadcast_arrays: these are implemented in terms of
// as_strided internally, and were silently broken (byte strides fed into
// an element-stride-based as_strided) until as_strided became real. Lock
// in correct NumPy-verified values now that both are fixed.
// ---------------------------------------------------------------------

// NumPy reference: np.broadcast_to(np.array([[1,2,3]]), (3,3))
#[test]
fn broadcast_to_row_vector_replicates_values() {
    let array = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);

    let result = broadcast_to(&array, &[3, 3]).expect("broadcast_to should succeed");

    assert_eq!(result.shape(), vec![3, 3]);
    assert_eq!(result.to_vec(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
}

// NumPy reference: np.broadcast_to(np.array([[4],[5],[6]]), (3,3))
#[test]
fn broadcast_to_column_vector_replicates_values() {
    let array = Array::from_vec(vec![4, 5, 6]).reshape(&[3, 1]);

    let result = broadcast_to(&array, &[3, 3]).expect("broadcast_to should succeed");

    assert_eq!(result.shape(), vec![3, 3]);
    assert_eq!(result.to_vec(), vec![4, 4, 4, 5, 5, 5, 6, 6, 6]);
}

// NumPy reference:
//   a = np.array([[1,2,3]])      # shape (1,3)
//   b = np.array([[4],[5],[6]])  # shape (3,1)
//   np.broadcast_arrays(a, b)
//   -> [[[1,2,3],[1,2,3],[1,2,3]], [[4,4,4],[5,5,5],[6,6,6]]]
#[test]
fn broadcast_arrays_outer_pattern_matches_numpy() {
    let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]);
    let b = Array::from_vec(vec![4, 5, 6]).reshape(&[3, 1]);

    let result = broadcast_arrays(&[&a, &b]).expect("broadcast_arrays should succeed");

    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), vec![3, 3]);
    assert_eq!(result[1].shape(), vec![3, 3]);
    assert_eq!(result[0].to_vec(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    assert_eq!(result[1].to_vec(), vec![4, 4, 4, 5, 5, 5, 6, 6, 6]);
}
