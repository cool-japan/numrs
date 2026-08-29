//! Integration tests proving `numrs2::io::netcdf` produces and consumes
//! **real** NetCDF-3 classic files.
//!
//! Before the fix these tests guard against, `write_netcdf` wrote a JSON
//! metadata sidecar (`<path>.nc.meta`) plus a raw binary blob named `.nc`
//! -- unreadable by any actual NetCDF tool, and the `netcdf3` crate
//! dependency was never used. This file is gated on the `netcdf` feature
//! (see `#![cfg(feature = "netcdf")]` below) so it compiles to nothing --
//! not even a warning -- when that feature is off.

#![cfg(feature = "netcdf")]

use numrs2::io::netcdf::{read_netcdf, write_netcdf};
use numrs2::prelude::*;
use std::collections::HashMap;
use std::io::Read;
use tempfile::TempDir;

/// The NetCDF-3 classic magic number: bytes `"CDF"` followed by version
/// byte `0x01`. A JSON sidecar (the previous behavior) would start with
/// `{`, not this.
const NETCDF3_CLASSIC_MAGIC: &[u8; 4] = b"CDF\x01";

#[test]
fn test_written_file_starts_with_real_netcdf3_classic_magic_bytes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("magic.nc");

    let array = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    write_netcdf(&array, &path, "data", None).expect("failed to write netcdf");

    let mut file = std::fs::File::open(&path).expect("failed to open written file");
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .expect("file is shorter than the NetCDF magic header");

    assert_eq!(
        &magic, NETCDF3_CLASSIC_MAGIC,
        "file does not start with the real NetCDF-3 classic magic bytes -- \
         got {:?}, this is not a genuine NetCDF file",
        magic
    );
}

#[test]
fn test_no_json_meta_sidecar_is_written() {
    // The previous implementation wrote a `<path>.nc.meta` JSON sidecar
    // alongside the raw binary `.nc` file. The real codec needs no such
    // sidecar: everything lives inside the single `.nc` file.
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("no_sidecar.nc");

    let array = Array::from_vec(vec![1.0f64, 2.0, 3.0]);
    write_netcdf(&array, &path, "data", None).expect("failed to write netcdf");

    let sidecar = path.with_extension("nc.meta");
    assert!(
        !sidecar.exists(),
        "no .nc.meta sidecar file should be written by the real NetCDF-3 codec"
    );
}

#[test]
fn test_roundtrip_f64_through_this_module() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("f64.nc");

    let array = Array::from_vec(vec![1.5f64, -2.25, 3.0, 4.75, 5.0, 6.125]).reshape(&[2, 3]);
    write_netcdf(&array, &path, "temperature", None).expect("failed to write netcdf");
    let loaded: Array<f64> = read_netcdf(&path, "temperature").expect("failed to read netcdf back");

    assert_eq!(loaded.shape(), array.shape());
    assert_eq!(loaded.to_vec(), array.to_vec());
}

#[test]
fn test_roundtrip_f32_through_this_module() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("f32.nc");

    let array = Array::from_vec(vec![1.5f32, -2.25, 3.0, 4.75]).reshape(&[2, 2]);
    write_netcdf(&array, &path, "v", None).expect("failed to write netcdf");
    let loaded: Array<f32> = read_netcdf(&path, "v").expect("failed to read netcdf back");

    assert_eq!(loaded.shape(), array.shape());
    assert_eq!(loaded.to_vec(), array.to_vec());
}

#[test]
fn test_roundtrip_i32_through_this_module() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("i32.nc");

    let array = Array::from_vec(vec![-10i32, 20, -30, 40, 50, -60]).reshape(&[2, 3]);
    write_netcdf(&array, &path, "counts", None).expect("failed to write netcdf");
    let loaded: Array<i32> = read_netcdf(&path, "counts").expect("failed to read netcdf back");

    assert_eq!(loaded.shape(), array.shape());
    assert_eq!(loaded.to_vec(), array.to_vec());
}

#[test]
fn test_roundtrip_i16_i8_u8_through_this_module() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    let path_i16 = temp_dir.path().join("i16.nc");
    let array_i16 = Array::from_vec(vec![-300i16, 300, -1, 0, 1]);
    write_netcdf(&array_i16, &path_i16, "v", None).expect("failed to write i16 netcdf");
    let loaded_i16: Array<i16> = read_netcdf(&path_i16, "v").expect("failed to read i16 netcdf");
    assert_eq!(loaded_i16.to_vec(), array_i16.to_vec());

    let path_i8 = temp_dir.path().join("i8.nc");
    let array_i8 = Array::from_vec(vec![-128i8, 127, 0, -1, 1]);
    write_netcdf(&array_i8, &path_i8, "v", None).expect("failed to write i8 netcdf");
    let loaded_i8: Array<i8> = read_netcdf(&path_i8, "v").expect("failed to read i8 netcdf");
    assert_eq!(loaded_i8.to_vec(), array_i8.to_vec());

    let path_u8 = temp_dir.path().join("u8.nc");
    let array_u8 = Array::from_vec(vec![0u8, 128, 255, 42]);
    write_netcdf(&array_u8, &path_u8, "v", None).expect("failed to write u8 netcdf");
    let loaded_u8: Array<u8> = read_netcdf(&path_u8, "v").expect("failed to read u8 netcdf");
    assert_eq!(loaded_u8.to_vec(), array_u8.to_vec());
}

#[test]
fn test_roundtrip_3d_shape_through_this_module() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("tensor.nc");

    let data: Vec<f64> = (0..24).map(|x| x as f64 * 0.5).collect();
    let array = Array::from_vec(data).reshape(&[2, 3, 4]);
    write_netcdf(&array, &path, "tensor", None).expect("failed to write netcdf");
    let loaded: Array<f64> = read_netcdf(&path, "tensor").expect("failed to read netcdf back");

    assert_eq!(loaded.shape(), array.shape());
    assert_eq!(loaded.to_vec(), array.to_vec());
}

#[test]
fn test_roundtrip_with_attributes_through_this_module() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("attrs.nc");

    let array = Array::from_vec(vec![10.0f64, 20.0, 30.0]);
    let mut attrs = HashMap::new();
    attrs.insert("units".to_string(), "meters".to_string());
    attrs.insert("long_name".to_string(), "surface temperature".to_string());

    write_netcdf(&array, &path, "temp", Some(attrs)).expect("failed to write netcdf");
    let loaded: Array<f64> = read_netcdf(&path, "temp").expect("failed to read netcdf back");

    assert_eq!(loaded.shape(), array.shape());
    assert_eq!(loaded.to_vec(), array.to_vec());
}

#[test]
fn test_variable_name_roundtrips() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("named.nc");

    let array = Array::from_vec(vec![1.0f64, 2.0, 3.0]);
    write_netcdf(&array, &path, "sea_surface_temperature", None).expect("failed to write netcdf");

    // Reading back under the correct name succeeds...
    let loaded: Array<f64> =
        read_netcdf(&path, "sea_surface_temperature").expect("failed to read netcdf back");
    assert_eq!(loaded.to_vec(), array.to_vec());

    // ...but an unrelated name must not silently succeed.
    let wrong_name: numrs2::error::Result<Array<f64>> = read_netcdf(&path, "not_the_right_name");
    assert!(wrong_name.is_err());
}

#[test]
fn test_i64_returns_clear_error_not_silent_narrowing() {
    // Classic NetCDF-3 has no native 64-bit integer type. The old JSON
    // sidecar implementation would happily "support" this by round-tripping
    // through its own private binary format; the real codec must instead
    // fail loudly rather than silently narrow to a 32-bit type.
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("i64.nc");

    let array = Array::from_vec(vec![1i64, 2, 3, i64::MAX]);
    let result = write_netcdf(&array, &path, "v", None);
    assert!(
        result.is_err(),
        "i64 has no native NetCDF-3 classic type and must return an error"
    );
}

#[test]
fn test_u32_returns_clear_error() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("u32.nc");

    let array = Array::from_vec(vec![1u32, 2, 3]);
    let result = write_netcdf(&array, &path, "v", None);
    assert!(
        result.is_err(),
        "u32 has no native NetCDF-3 classic type and must return an error"
    );
}

#[test]
fn test_u64_returns_clear_error() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("u64.nc");

    let array = Array::from_vec(vec![1u64, 2, 3]);
    let result = write_netcdf(&array, &path, "v", None);
    assert!(
        result.is_err(),
        "u64 has no native NetCDF-3 classic type and must return an error"
    );
}

#[test]
fn test_reading_nonexistent_variable_fails_cleanly() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("solo_var.nc");

    let array = Array::from_vec(vec![1.0f64, 2.0]);
    write_netcdf(&array, &path, "only_var", None).expect("failed to write netcdf");

    let result: numrs2::error::Result<Array<f64>> = read_netcdf(&path, "does_not_exist");
    assert!(result.is_err());
}
