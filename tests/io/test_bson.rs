//! BSON I/O format tests

use numrs2::prelude::*;
use tempfile::TempDir;

#[cfg(feature = "bson")]
use numrs2::io::bson_format::{from_bson_file, to_bson_file};

#[test]
#[cfg(feature = "bson")]
fn test_bson_f64() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test.bson");

    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    to_bson_file(&array, &path).expect("Failed to write BSON");
    let loaded: Array<f64> = from_bson_file(&path).expect("Failed to read BSON");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}

#[test]
#[cfg(feature = "bson")]
fn test_bson_i32() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test_int.bson");

    let array = Array::from_vec(vec![10, 20, 30, 40]).reshape(&[2, 2]);

    to_bson_file(&array, &path).expect("Failed to write BSON");
    let loaded: Array<i32> = from_bson_file(&path).expect("Failed to read BSON");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}
