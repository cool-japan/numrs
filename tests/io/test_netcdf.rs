//! NetCDF I/O format tests

use numrs2::prelude::*;
use tempfile::TempDir;

#[cfg(feature = "netcdf")]
use numrs2::io::netcdf::{read_netcdf, write_netcdf};

#[test]
#[cfg(feature = "netcdf")]
fn test_netcdf_f64() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test.nc");

    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    write_netcdf(&array, &path, "temperature", None).expect("Failed to write NetCDF");
    let loaded: Array<f64> = read_netcdf(&path, "temperature").expect("Failed to read NetCDF");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}

#[test]
#[cfg(feature = "netcdf")]
fn test_netcdf_i32() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().join("test_int.nc");

    let array = Array::from_vec(vec![10, 20, 30, 40]).reshape(&[2, 2]);

    write_netcdf(&array, &path, "data", None).expect("Failed to write NetCDF");
    let loaded: Array<i32> = read_netcdf(&path, "data").expect("Failed to read NetCDF");

    assert_eq!(array.shape(), loaded.shape());
    assert_eq!(array.to_vec(), loaded.to_vec());
}
