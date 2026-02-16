//! Tests for the indexing module

use super::*;
use crate::array::Array;

// ==================== ELLIPSIS TESTS ====================

#[test]
fn test_ellipsis_2d_first_position() {
    // For a 2D array (3x4), arr[..., 0] should get the first column
    let arr = Array::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ])
    .reshape(&[3, 4]);

    let result = arr
        .index(&[IndexSpec::Ellipsis, IndexSpec::Index(0)])
        .expect("indexing with ellipsis should succeed");

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn test_ellipsis_2d_last_position() {
    // For a 2D array (3x4), arr[0, ...] should get the first row
    let arr = Array::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ])
    .reshape(&[3, 4]);

    let result = arr
        .index(&[IndexSpec::Index(0), IndexSpec::Ellipsis])
        .expect("indexing with ellipsis should succeed");

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_ellipsis_3d_middle() {
    // For a 3D array (2x3x4), arr[0, ..., 0] should get first and last dim slices
    let arr: Array<f64> = Array::from_vec((0..24).map(|x| x as f64).collect()).reshape(&[2, 3, 4]);

    let result = arr
        .index(&[
            IndexSpec::Index(0),
            IndexSpec::Ellipsis,
            IndexSpec::Index(0),
        ])
        .expect("indexing with ellipsis should succeed");

    // Should be arr[0, :, 0] = [0, 4, 8]
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![0.0, 4.0, 8.0]);
}

#[test]
fn test_ellipsis_alone() {
    // arr[...] should return a copy of the entire array
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

    let result = arr
        .index(&[IndexSpec::Ellipsis])
        .expect("indexing with ellipsis should succeed");

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), arr.to_vec());
}

#[test]
fn test_ellipsis_with_slice() {
    // For a 2D array, arr[..., 1:3] should slice the last dimension
    let arr = Array::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ])
    .reshape(&[3, 4]);

    let result = arr
        .index(&[IndexSpec::Ellipsis, IndexSpec::Slice(1, Some(3), None)])
        .expect("indexing with ellipsis and slice should succeed");

    assert_eq!(result.shape(), &[3, 2]);
    // First row [1,2,3,4] -> [2,3], Second row [5,6,7,8] -> [6,7], etc
    assert_eq!(result.to_vec(), vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
}

// ==================== NEWAXIS TESTS ====================

#[test]
fn test_newaxis_1d_to_2d_row() {
    // arr[np.newaxis, :] on (3,) should give (1, 3)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);

    let result = arr
        .index(&[IndexSpec::NewAxis, IndexSpec::All])
        .expect("indexing with newaxis should succeed");

    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_newaxis_1d_to_2d_column() {
    // arr[:, np.newaxis] on (3,) should give (3, 1)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);

    let result = arr
        .index(&[IndexSpec::All, IndexSpec::NewAxis])
        .expect("indexing with newaxis should succeed");

    assert_eq!(result.shape(), &[3, 1]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_newaxis_2d_add_front() {
    // arr[np.newaxis, :, :] on (2, 3) should give (1, 2, 3)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = arr
        .index(&[IndexSpec::NewAxis, IndexSpec::All, IndexSpec::All])
        .expect("indexing with newaxis should succeed");

    assert_eq!(result.shape(), &[1, 2, 3]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_newaxis_2d_add_middle() {
    // arr[:, np.newaxis, :] on (2, 3) should give (2, 1, 3)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = arr
        .index(&[IndexSpec::All, IndexSpec::NewAxis, IndexSpec::All])
        .expect("indexing with newaxis should succeed");

    assert_eq!(result.shape(), &[2, 1, 3]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_newaxis_2d_add_end() {
    // arr[:, :, np.newaxis] on (2, 3) should give (2, 3, 1)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = arr
        .index(&[IndexSpec::All, IndexSpec::All, IndexSpec::NewAxis])
        .expect("indexing with newaxis should succeed");

    assert_eq!(result.shape(), &[2, 3, 1]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_newaxis_multiple() {
    // arr[np.newaxis, :, np.newaxis] on (3,) should give (1, 3, 1)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);

    let result = arr
        .index(&[IndexSpec::NewAxis, IndexSpec::All, IndexSpec::NewAxis])
        .expect("indexing with multiple newaxis should succeed");

    assert_eq!(result.shape(), &[1, 3, 1]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_newaxis_with_ellipsis() {
    // arr[np.newaxis, ...] on (2, 3) should give (1, 2, 3)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = arr
        .index(&[IndexSpec::NewAxis, IndexSpec::Ellipsis])
        .expect("indexing with newaxis and ellipsis should succeed");

    assert_eq!(result.shape(), &[1, 2, 3]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_ellipsis_with_newaxis() {
    // arr[..., np.newaxis] on (2, 3) should give (2, 3, 1)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = arr
        .index(&[IndexSpec::Ellipsis, IndexSpec::NewAxis])
        .expect("indexing with ellipsis and newaxis should succeed");

    assert_eq!(result.shape(), &[2, 3, 1]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_newaxis_with_index() {
    // arr[0, np.newaxis, :] on (2, 3) should select first row and add axis: (1, 3)
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = arr
        .index(&[IndexSpec::Index(0), IndexSpec::NewAxis, IndexSpec::All])
        .expect("indexing with index and newaxis should succeed");

    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_helper_insert_newaxis() {
    // Test the insert_newaxis helper directly
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    // Insert at position 0
    let result = insert_newaxis(&arr, &[0]).expect("insert_newaxis at position 0 should succeed");
    assert_eq!(result.shape(), &[1, 2, 3]);

    // Insert at position 1
    let result = insert_newaxis(&arr, &[1]).expect("insert_newaxis at position 1 should succeed");
    assert_eq!(result.shape(), &[2, 1, 3]);

    // Insert at position 2 (end)
    let result = insert_newaxis(&arr, &[2]).expect("insert_newaxis at position 2 should succeed");
    assert_eq!(result.shape(), &[2, 3, 1]);

    // Insert at multiple positions
    let result =
        insert_newaxis(&arr, &[0, 2]).expect("insert_newaxis at multiple positions should succeed");
    assert_eq!(result.shape(), &[1, 2, 1, 3]);
}
