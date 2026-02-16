//! MessagePack I/O format tests

use numrs2::prelude::*;
use tempfile::TempDir;

#[cfg(feature = "messagepack")]
use numrs2::io::messagepack::{from_messagepack, to_messagepack};

#[test]
#[cfg(feature = "messagepack")]
fn test_messagepack_f64() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test.msgpack");

    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    to_messagepack(&array, &path).expect("Failed to write MessagePack");
    let loaded: Array<f64> = from_messagepack(&path).expect("Failed to read MessagePack");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}

#[test]
#[cfg(feature = "messagepack")]
fn test_messagepack_i32() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test_int.msgpack");

    let array = Array::from_vec(vec![10, 20, 30, 40]).reshape(&[2, 2]);

    to_messagepack(&array, &path).expect("Failed to write MessagePack");
    let loaded: Array<i32> = from_messagepack(&path).expect("Failed to read MessagePack");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}
