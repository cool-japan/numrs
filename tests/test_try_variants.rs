//! Regression tests for LANE W1-K: panic hygiene.
//!
//! Item 1 (src/array/creation.rs, src/array/manipulation.rs): every
//! public-API function that used to panic on bad user input now has a
//! non-panicking `try_*` counterpart that returns `Result`, while the
//! original panicking entry point is kept (for NumPy-like ergonomics) and
//! documented with `# Panics`. These tests assert that:
//!   - the `try_*` variant returns `Err` (never panics) on invalid input,
//!   - the `try_*` variant returns the *same, NumPy-verified* value as the
//!     panicking original on valid input (no behavior regression), and
//!   - the panicking original still panics on the same invalid input
//!     (`#[should_panic]`), so callers relying on NumPy-like ergonomics are
//!     unaffected.
//!
//! Also covers `Array::broadcast_to`, which already returned `Result` but
//! used to panic internally on: an empty source array, and a shape it could
//! not actually broadcast to (silently tiling instead of erring). It is now
//! implemented via `ndarray`'s own broadcasting and never panics.

use numrs2::array::Array;
use numrs2::error::NumRs2Error;

// ---------------------------------------------------------------------
// tril / try_tril
// ---------------------------------------------------------------------

#[test]
fn try_tril_errs_on_non_2d_array() {
    let a: Array<i32> = Array::from_vec(vec![1, 2, 3, 4]); // 1D
    let result = a.try_tril(0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        NumRs2Error::DimensionMismatch(_)
    ));
}

#[test]
fn try_tril_matches_panicking_original_on_valid_input() {
    // NumPy reference: np.tril([[1,2,3],[4,5,6],[7,8,9]], k=0)
    // -> [[1,0,0],[4,5,0],[7,8,9]]
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    let expected = a.tril(0);
    let actual = a
        .try_tril(0)
        .expect("try_tril should succeed on a 2D array");
    assert_eq!(actual.to_vec(), expected.to_vec());
    assert_eq!(actual.to_vec(), vec![1, 0, 0, 4, 5, 0, 7, 8, 9]);
}

#[test]
#[should_panic(expected = "tril requires a 2D array")]
fn tril_still_panics_on_non_2d_array() {
    let a: Array<i32> = Array::from_vec(vec![1, 2, 3, 4]);
    let _ = a.tril(0);
}

// ---------------------------------------------------------------------
// triu / try_triu
// ---------------------------------------------------------------------

#[test]
fn try_triu_errs_on_non_2d_array() {
    let a: Array<i32> = Array::from_vec(vec![1, 2, 3, 4]); // 1D
    let result = a.try_triu(0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        NumRs2Error::DimensionMismatch(_)
    ));
}

#[test]
fn try_triu_matches_panicking_original_on_valid_input() {
    // NumPy reference: np.triu([[1,2,3],[4,5,6],[7,8,9]], k=0)
    // -> [[1,2,3],[0,5,6],[0,0,9]]
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    let expected = a.triu(0);
    let actual = a
        .try_triu(0)
        .expect("try_triu should succeed on a 2D array");
    assert_eq!(actual.to_vec(), expected.to_vec());
    assert_eq!(actual.to_vec(), vec![1, 2, 3, 0, 5, 6, 0, 0, 9]);
}

#[test]
#[should_panic(expected = "triu requires a 2D array")]
fn triu_still_panics_on_non_2d_array() {
    let a: Array<i32> = Array::from_vec(vec![1, 2, 3, 4]);
    let _ = a.triu(0);
}

// ---------------------------------------------------------------------
// create_diagonal_matrix_helper / try_create_diagonal_matrix_helper
// ---------------------------------------------------------------------

#[test]
fn try_create_diagonal_matrix_helper_errs_on_non_1d_array() {
    let v: Array<i32> = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]); // 2D
    let result = Array::<i32>::try_create_diagonal_matrix_helper(&v, 0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        NumRs2Error::DimensionMismatch(_)
    ));
}

#[test]
fn try_create_diagonal_matrix_helper_matches_panicking_original() {
    // NumPy reference: np.diag([1, 2, 3]) -> [[1,0,0],[0,2,0],[0,0,3]]
    let v = Array::from_vec(vec![1, 2, 3]);
    let expected = Array::<i32>::create_diagonal_matrix_helper(&v, 0);
    let actual = Array::<i32>::try_create_diagonal_matrix_helper(&v, 0)
        .expect("try_create_diagonal_matrix_helper should succeed on a 1D array");
    assert_eq!(actual.to_vec(), expected.to_vec());
    assert_eq!(actual.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
}

#[test]
#[should_panic(expected = "diag requires a 1D array")]
fn create_diagonal_matrix_helper_still_panics_on_non_1d_array() {
    let v: Array<i32> = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    let _ = Array::<i32>::create_diagonal_matrix_helper(&v, 0);
}

// ---------------------------------------------------------------------
// create_diagonal_matrix / try_create_diagonal_matrix
// ---------------------------------------------------------------------

#[test]
fn try_create_diagonal_matrix_errs_on_3d_array() {
    let v: Array<i32> = Array::from_vec(vec![1; 8]).reshape(&[2, 2, 2]); // 3D
    let result = Array::<i32>::try_create_diagonal_matrix(&v, 0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        NumRs2Error::DimensionMismatch(_)
    ));
}

#[test]
fn try_create_diagonal_matrix_matches_panicking_original_1d_and_2d() {
    // 1D -> construct diagonal matrix.
    let v1 = Array::from_vec(vec![1, 2, 3]);
    let expected1 = Array::<i32>::create_diagonal_matrix(&v1, 0);
    let actual1 = Array::<i32>::try_create_diagonal_matrix(&v1, 0)
        .expect("try_create_diagonal_matrix should succeed on a 1D array");
    assert_eq!(actual1.to_vec(), expected1.to_vec());
    assert_eq!(actual1.to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);

    // 2D -> extract diagonal. NumPy reference:
    // np.diag([[1,2,3],[4,5,6],[7,8,9]]) -> [1, 5, 9]
    let v2 = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    let expected2 = Array::<i32>::create_diagonal_matrix(&v2, 0);
    let actual2 = Array::<i32>::try_create_diagonal_matrix(&v2, 0)
        .expect("try_create_diagonal_matrix should succeed on a 2D array");
    assert_eq!(actual2.to_vec(), expected2.to_vec());
    assert_eq!(actual2.to_vec(), vec![1, 5, 9]);
}

#[test]
#[should_panic(expected = "diag requires a 1D or 2D array")]
fn create_diagonal_matrix_still_panics_on_3d_array() {
    let v: Array<i32> = Array::from_vec(vec![1; 8]).reshape(&[2, 2, 2]);
    let _ = Array::<i32>::create_diagonal_matrix(&v, 0);
}

// ---------------------------------------------------------------------
// reshape / try_reshape
// ---------------------------------------------------------------------

#[test]
fn try_reshape_errs_on_size_mismatch() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]); // size 6
    let result = a.try_reshape(&[4, 4]); // size 16
    assert!(result.is_err());
}

#[test]
fn try_reshape_matches_panicking_original_on_valid_input() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    let expected = a.reshape(&[2, 3]);
    let actual = a
        .try_reshape(&[2, 3])
        .expect("try_reshape should succeed when sizes match");
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.to_vec(), expected.to_vec());
    assert_eq!(actual.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
#[should_panic]
fn reshape_still_panics_on_size_mismatch() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    let _ = a.reshape(&[4, 4]);
}

/// `try_reshape` must never panic on a non-contiguous array: it falls back
/// to a logical-order copy instead of relying on `into_shape_with_order`
/// succeeding directly. Uses a transposed (non-contiguous) array as input.
#[test]
fn try_reshape_succeeds_on_non_contiguous_array() {
    // a: [[1,2,3],[4,5,6]] (2x3, C-contiguous)
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    // b: transpose -> logically [[1,4],[2,5],[3,6]] (3x2), non-contiguous
    // memory layout (still backed by a's original C-order buffer).
    let b = a.transpose_axis(0, 1);
    assert_eq!(b.shape(), vec![3, 2]);
    assert_eq!(b.to_vec(), vec![1, 4, 2, 5, 3, 6]);

    // Reshape the non-contiguous array to a flat [6] shape: must reflect
    // *logical* order (NumPy: b.reshape(6) on the transposed array), not
    // the physical backing buffer's order.
    let flat = b
        .try_reshape(&[6])
        .expect("try_reshape must not panic on a non-contiguous array");
    assert_eq!(flat.to_vec(), vec![1, 4, 2, 5, 3, 6]);
}

// ---------------------------------------------------------------------
// reshape_with / try_reshape_with
// ---------------------------------------------------------------------

#[test]
fn try_reshape_with_errs_on_size_mismatch() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    assert!(a.try_reshape_with(&[4, 4], true).is_err());
    assert!(a.try_reshape_with(&[4, 4], false).is_err());
}

#[test]
fn try_reshape_with_matches_panicking_original_on_valid_input() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    for copy in [true, false] {
        let expected = a.reshape_with(&[3, 2], copy);
        let actual = a
            .try_reshape_with(&[3, 2], copy)
            .expect("try_reshape_with should succeed when sizes match");
        assert_eq!(actual.to_vec(), expected.to_vec());
        assert_eq!(actual.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }
}

#[test]
#[should_panic]
fn reshape_with_still_panics_on_size_mismatch() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    let _ = a.reshape_with(&[4, 4], true);
}

// ---------------------------------------------------------------------
// flatten / try_flatten
// ---------------------------------------------------------------------

#[test]
fn try_flatten_errs_on_invalid_order() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let result = a.try_flatten(Some("Z"));
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NumRs2Error::InvalidInput(_)));
}

#[test]
fn try_flatten_matches_panicking_original_on_valid_input() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let expected = a.flatten(Some("C"));
    let actual = a
        .try_flatten(Some("C"))
        .expect("try_flatten should succeed with a valid order");
    assert_eq!(actual.to_vec(), expected.to_vec());
    assert_eq!(actual.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
#[should_panic(expected = "Invalid order parameter")]
fn flatten_still_panics_on_invalid_order() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let _ = a.flatten(Some("Z"));
}

/// `try_flatten`'s `"F"` branch also goes through `try_reshape` now (it
/// used to call the panicking `reshape`); confirm this doesn't turn a
/// previously-succeeding call into a spurious `Err` for a non-contiguous,
/// multi-dimensional array (`try_reshape`'s non-contiguous fallback must
/// still apply here). Note: the `"F"` branch does not actually implement
/// Fortran-order raveling for ndim > 1 (a pre-existing gap noted in the
/// source, out of scope for this panic-hygiene fix) -- this test only
/// asserts that it succeeds and preserves the element count, not that the
/// order is truly column-major.
#[test]
fn try_flatten_f_order_succeeds_on_non_contiguous_array() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let b = a.transpose_axis(0, 1); // non-contiguous, shape [3, 2]
    let result = b
        .try_flatten(Some("F"))
        .expect("try_flatten(\"F\") must not panic or error on a non-contiguous array");
    assert_eq!(result.shape(), vec![6]);
    assert_eq!(result.size(), 6);
}

// ---------------------------------------------------------------------
// transpose_axis / try_transpose_axis
// ---------------------------------------------------------------------

#[test]
fn try_transpose_axis_errs_on_out_of_bounds_axis() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let result = a.try_transpose_axis(0, 5);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        NumRs2Error::IndexOutOfBounds(_)
    ));
}

#[test]
fn try_transpose_axis_matches_panicking_original_on_valid_input() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let expected = a.transpose_axis(0, 1);
    let actual = a
        .try_transpose_axis(0, 1)
        .expect("try_transpose_axis should succeed for valid axes");
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.to_vec(), expected.to_vec());
    assert_eq!(actual.shape(), vec![3, 2]);
    assert_eq!(actual.to_vec(), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
#[should_panic(expected = "Axis out of bounds")]
fn transpose_axis_still_panics_on_out_of_bounds_axis() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    let _ = a.transpose_axis(0, 5);
}

// ---------------------------------------------------------------------
// broadcast_to: already returned Result, but used to panic internally.
// ---------------------------------------------------------------------

#[test]
fn broadcast_to_errs_instead_of_panicking_on_empty_array() {
    let empty: Array<f64> = Array::from_vec(vec![]);
    // Broadcasting an empty (size-0) array to a non-empty shape is not a
    // valid NumPy broadcast (0 is neither 1 nor equal to the target dim);
    // this must return `Err`, not panic trying to read a template element.
    let result = empty.broadcast_to(&[3]);
    assert!(result.is_err());
}

#[test]
fn broadcast_to_errs_on_incompatible_shape_instead_of_silently_tiling() {
    // NumPy reference: np.broadcast_to(np.array([1, 2]), (4,)) raises
    // ValueError ("cannot broadcast ... shape (2,) into shape (4,)") because
    // 2 is neither 1 nor equal to 4. A prior implementation here computed
    // `dim % current_shape[i]`, which would have *silently tiled* this into
    // [1, 2, 1, 2] instead of erroring -- that is not NumPy broadcasting.
    let a = Array::from_vec(vec![1, 2]);
    let result = a.broadcast_to(&[4]);
    assert!(
        result.is_err(),
        "broadcasting [2] to [4] is not valid NumPy broadcasting and must error, got {:?}",
        result
    );
}

#[test]
fn broadcast_to_still_succeeds_on_valid_broadcast() {
    // NumPy reference: np.broadcast_to(np.array([1, 2, 3]), (3, 3))
    // -> [[1,2,3],[1,2,3],[1,2,3]]
    let a = Array::from_vec(vec![1, 2, 3]);
    let result = a
        .broadcast_to(&[3, 3])
        .expect("broadcasting [3] to [3, 3] is valid NumPy broadcasting");
    assert_eq!(result.shape(), vec![3, 3]);
    assert_eq!(result.to_vec(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
}

#[test]
fn broadcast_to_scalar_like_one_element_array() {
    // NumPy reference: np.broadcast_to(np.array([5]), (2, 3))
    // -> [[5,5,5],[5,5,5]]
    let a = Array::from_vec(vec![5]);
    let result = a
        .broadcast_to(&[2, 3])
        .expect("broadcasting [1] to [2, 3] is valid NumPy broadcasting");
    assert_eq!(result.shape(), vec![2, 3]);
    assert_eq!(result.to_vec(), vec![5, 5, 5, 5, 5, 5]);
}
