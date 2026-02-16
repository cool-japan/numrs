//! MATLAB .mat file I/O format tests

use numrs2::prelude::*;
use tempfile::TempDir;

#[cfg(feature = "matlab")]
use numrs2::io::matlab::{read_mat, write_mat};

#[test]
#[cfg(feature = "matlab")]
fn test_matlab_f64() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test.mat");

    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    write_mat(&array, &path, "mydata").expect("Failed to write .mat file");
    let loaded: Array<f64> = read_mat(&path, "mydata").expect("Failed to read .mat file");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}

#[test]
#[cfg(feature = "matlab")]
fn test_matlab_i32() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test_int.mat");

    let array = Array::from_vec(vec![10, 20, 30, 40]).reshape(&[2, 2]);

    write_mat(&array, &path, "integers").expect("Failed to write .mat file");
    let loaded: Array<i32> = read_mat(&path, "integers").expect("Failed to read .mat file");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}
