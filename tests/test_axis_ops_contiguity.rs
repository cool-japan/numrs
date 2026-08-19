//! Regression tests for LANE W1-K item 3: `src/axis_ops.rs`.
//!
//! `cumsum_axis` / `cumprod_axis` had two distinct bugs, both exercised
//! here:
//!
//! 1. **Correctness** (not merely a panic): the inner loop only ever wrote
//!    to `stride` many flat positions per sequence (missing an `elem` loop
//!    over the `stride` positions within each block), so accumulating along
//!    any axis but the last silently produced wrong results for arrays with
//!    more than one row along that axis's orthogonal dimensions. Verified
//!    against NumPy (`np.cumsum` / `np.cumprod`) on a plain contiguous
//!    array below -- this fails with the pre-fix code even though the array
//!    is perfectly contiguous.
//! 2. **Panic hygiene**: `.as_slice_mut()` / `.as_slice()` were used
//!    unconditionally with `.expect("Array must be contiguous ...")`,
//!    reachable by calling these methods on a non-contiguous view (e.g. the
//!    result of `transpose_axis`). Fixed by snapshotting elements in
//!    logical order via `.iter()` instead.

use numrs2::array::Array;
use numrs2::prelude::*;

// ---------------------------------------------------------------------
// cumsum_axis: correctness on a plain contiguous array (bug #1 above).
// ---------------------------------------------------------------------

#[test]
fn cumsum_axis_0_on_contiguous_2x3_matches_numpy() {
    // a = [[1,2,3],[4,5,6]]
    // NumPy reference: np.cumsum(a, axis=0) -> [[1,2,3],[5,7,9]]
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let result = a.cumsum_axis(0).expect("cumsum_axis(0) should succeed");
    assert_eq!(result.shape(), vec![2, 3]);
    assert_eq!(result.to_vec(), vec![1, 2, 3, 5, 7, 9]);
}

#[test]
fn cumsum_axis_1_on_contiguous_2x3_matches_numpy() {
    // NumPy reference: np.cumsum(a, axis=1) -> [[1,3,6],[4,9,15]]
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let result = a.cumsum_axis(1).expect("cumsum_axis(1) should succeed");
    assert_eq!(result.shape(), vec![2, 3]);
    assert_eq!(result.to_vec(), vec![1, 3, 6, 4, 9, 15]);
}

#[test]
fn cumsum_axis_0_on_larger_contiguous_matrix_matches_numpy() {
    // a = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] (3x4)
    // NumPy reference: np.cumsum(a, axis=0)
    // -> [[1,2,3,4],[6,8,10,12],[15,18,21,24]]
    let a = Array::from_vec((1..=12).collect::<Vec<i64>>()).reshape(&[3, 4]);
    let result = a.cumsum_axis(0).expect("cumsum_axis(0) should succeed");
    assert_eq!(
        result.to_vec(),
        vec![1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24]
    );
}

/// The bug fixed in `cumsum_axis`/`cumprod_axis` (missing inner loop over
/// `stride`) and the one fixed in `argmin_axis`/`argmax_axis`
/// (over-allocated `result_data`) are only fully exercised when *both*
/// `n_sequences > 1` and `elements_per_sequence > 1` simultaneously -- a
/// plain 2D matrix only ever has one of those `> 1` for a given axis. This
/// uses a 3D array with the accumulation/reduction axis in the *middle*
/// (`axis=1` of a `[2,3,4]` array: `n_sequences=2`, `elements_per_sequence
/// (== stride) = 4`), which pins down the flat-index arithmetic in both
/// dimensions at once.
#[test]
fn cumsum_axis_1_on_contiguous_3d_matrix_matches_numpy() {
    // a[i,j,k] = i*12 + j*4 + k, shape [2,3,4]:
    // a[0] = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    // a[1] = [[12,13,14,15],[16,17,18,19],[20,21,22,23]]
    // NumPy reference: np.cumsum(a, axis=1) ->
    // [[[0,1,2,3],[4,6,8,10],[12,15,18,21]],
    //  [[12,13,14,15],[28,30,32,34],[48,51,54,57]]]
    let a = Array::from_vec((0..24).collect::<Vec<i64>>()).reshape(&[2, 3, 4]);
    let result = a.cumsum_axis(1).expect("cumsum_axis(1) should succeed");
    assert_eq!(result.shape(), vec![2, 3, 4]);
    assert_eq!(
        result.to_vec(),
        vec![
            0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 12, 13, 14, 15, 28, 30, 32, 34, 48, 51, 54,
            57,
        ]
    );
}

#[test]
fn argmin_axis_1_on_contiguous_3d_matrix_matches_numpy() {
    // Hand-picked so the argmin along axis=1 (size 3) varies independently
    // for every (i, k) combination -- an ascending-value array would give
    // argmin == 0 everywhere and would not distinguish a correct
    // implementation from one with broken index arithmetic.
    //
    // a[0,:,:] = [[5,1,7,9],[2,9,3,8],[8,4,6,0]]
    // a[1,:,:] = [[4,2,3,8],[6,0,1,5],[1,5,2,9]]
    //
    // Per (i, k), the column (j=0,1,2) minimum and its index:
    // i=0: k0 [5,2,8]->j1  k1 [1,9,4]->j0  k2 [7,3,6]->j1  k3 [9,8,0]->j2
    // i=1: k0 [4,6,1]->j2  k1 [2,0,5]->j1  k2 [3,1,2]->j1  k3 [8,5,9]->j1
    // NumPy reference: np.argmin(a, axis=1) -> [[1,0,1,2],[2,1,1,1]]
    let a = Array::from_vec(vec![
        5, 1, 7, 9, 2, 9, 3, 8, 8, 4, 6, 0, 4, 2, 3, 8, 6, 0, 1, 5, 1, 5, 2, 9,
    ])
    .reshape(&[2, 3, 4]);
    let result = a.argmin_axis(1).expect("argmin_axis(1) should succeed");
    assert_eq!(result.shape(), vec![2, 4]);
    assert_eq!(result.to_vec(), vec![1, 0, 1, 2, 2, 1, 1, 1]);
}

#[test]
fn cumprod_axis_0_on_contiguous_2x3_matches_numpy() {
    // NumPy reference: np.cumprod([[1,2,3],[4,5,6]], axis=0)
    // -> [[1,2,3],[4,10,18]]
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let result = a.cumprod_axis(0).expect("cumprod_axis(0) should succeed");
    assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 10, 18]);
}

#[test]
fn cumprod_axis_1_on_contiguous_2x3_matches_numpy() {
    // NumPy reference: np.cumprod([[1,2,3],[4,5,6]], axis=1)
    // -> [[1,2,6],[4,20,120]]
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let result = a.cumprod_axis(1).expect("cumprod_axis(1) should succeed");
    assert_eq!(result.to_vec(), vec![1, 2, 6, 4, 20, 120]);
}

// ---------------------------------------------------------------------
// cumsum_axis / cumprod_axis / argmin_axis / argmax_axis: must not panic
// on a non-contiguous (transposed) array, and must give logically-correct
// results (bug #2 above).
// ---------------------------------------------------------------------

/// Builds `b`, the transpose of `[[1,2,3],[4,5,6]]`, i.e. logically
/// `[[1,4],[2,5],[3,6]]` with shape `[3, 2]`, backed by a non-contiguous
/// memory layout (same buffer as the original 2x3 array, reinterpreted via
/// `transpose_axis`, so `.as_slice()`/`.as_slice_mut()` return `None`).
fn transposed_non_contiguous() -> Array<i64> {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let b = a.transpose_axis(0, 1);
    assert_eq!(b.shape(), vec![3, 2]);
    assert_eq!(b.to_vec(), vec![1, 4, 2, 5, 3, 6]); // sanity check on b itself
    assert!(
        b.array().as_slice().is_none(),
        "test fixture must actually be non-contiguous"
    );
    b
}

#[test]
fn cumsum_axis_on_non_contiguous_array_does_not_panic_and_matches_numpy() {
    let b = transposed_non_contiguous();
    // b = [[1,4],[2,5],[3,6]]
    // NumPy reference: np.cumsum(b, axis=0) -> [[1,4],[3,9],[6,15]]
    let result = b
        .cumsum_axis(0)
        .expect("must not panic on non-contiguous input");
    assert_eq!(result.shape(), vec![3, 2]);
    assert_eq!(result.to_vec(), vec![1, 4, 3, 9, 6, 15]);
}

#[test]
fn cumsum_axis_1_on_non_contiguous_array_matches_numpy() {
    let b = transposed_non_contiguous();
    // NumPy reference: np.cumsum(b, axis=1) -> [[1,5],[2,7],[3,9]]
    let result = b
        .cumsum_axis(1)
        .expect("must not panic on non-contiguous input");
    assert_eq!(result.to_vec(), vec![1, 5, 2, 7, 3, 9]);
}

#[test]
fn cumprod_axis_on_non_contiguous_array_does_not_panic_and_matches_numpy() {
    let b = transposed_non_contiguous();
    // NumPy reference: np.cumprod(b, axis=0) -> [[1,4],[2,20],[6,120]]
    let result = b
        .cumprod_axis(0)
        .expect("must not panic on non-contiguous input");
    assert_eq!(result.to_vec(), vec![1, 4, 2, 20, 6, 120]);
}

#[test]
fn argmin_axis_on_non_contiguous_array_does_not_panic_and_matches_numpy() {
    let b = transposed_non_contiguous();
    // NumPy reference: np.argmin(b, axis=0) -> [0, 0] (column mins: 1, 4)
    let result = b
        .argmin_axis(0)
        .expect("must not panic on non-contiguous input");
    assert_eq!(result.to_vec(), vec![0, 0]);
}

#[test]
fn argmax_axis_on_non_contiguous_array_does_not_panic_and_matches_numpy() {
    let b = transposed_non_contiguous();
    // NumPy reference: np.argmax(b, axis=0) -> [2, 2] (column maxes: 3, 6)
    let result = b
        .argmax_axis(0)
        .expect("must not panic on non-contiguous input");
    assert_eq!(result.to_vec(), vec![2, 2]);
}

#[test]
fn argmin_argmax_axis_1_on_non_contiguous_array_matches_numpy() {
    let b = transposed_non_contiguous();
    // b = [[1,4],[2,5],[3,6]]; NumPy reference:
    // np.argmin(b, axis=1) -> [0, 0, 0]  (each row's min is column 0)
    // np.argmax(b, axis=1) -> [1, 1, 1]  (each row's max is column 1)
    let argmin = b
        .argmin_axis(1)
        .expect("must not panic on non-contiguous input");
    let argmax = b
        .argmax_axis(1)
        .expect("must not panic on non-contiguous input");
    assert_eq!(argmin.to_vec(), vec![0, 0, 0]);
    assert_eq!(argmax.to_vec(), vec![1, 1, 1]);
}
