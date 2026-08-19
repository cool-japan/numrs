//! NetCDF-3 format support for NumRS2 arrays
//!
//! This module provides a pure Rust implementation for reading and writing
//! NumRS2 arrays to/from real NetCDF-3 classic files, backed by the
//! [`netcdf3`](https://docs.rs/netcdf3) crate.
//!
//! NetCDF (Network Common Data Form) is a self-describing, machine-independent
//! data format widely used in scientific computing, especially for climate and
//! atmospheric data. Files written by this module are genuine NetCDF-3
//! classic files (they start with the `CDF\x01` magic bytes) and can be
//! read by any conforming NetCDF-3 reader, not just this module.
//!
//! # Type mapping
//!
//! The classic NetCDF-3 format has exactly six native variable types:
//! `NC_BYTE` (i8), `NC_CHAR` (u8), `NC_SHORT` (i16), `NC_INT` (i32),
//! `NC_FLOAT` (f32) and `NC_DOUBLE` (f64). NumRS2's `i8`, `u8`, `i16`,
//! `i32`, `f32` and `f64` element types map directly onto these. There is
//! no native 64-bit integer or unsigned 32/64-bit integer type in classic
//! NetCDF-3, so `write_netcdf`/`read_netcdf` for `i64`, `u32` and `u64`
//! return a clear [`NumRs2Error::TypeCastError`] instead of silently
//! narrowing or widening the data.
//!
//! Each array axis becomes its own fixed-size NetCDF dimension, named
//! `dim0`, `dim1`, ... in axis order.
//!
//! # Features
//! - Read/write NumRS2 arrays to/from real NetCDF-3 classic files
//! - Dimensions derived from the array's shape; variable names round-trip
//! - String-valued variable attributes
//! - Pure Rust implementation (no C dependencies)
//!
//! # Example
//! ```no_run
//! use numrs2::prelude::*;
//! use numrs2::io::netcdf::{write_netcdf, read_netcdf};
//! use std::path::Path;
//!
//! // Create an array
//! let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//!
//! // Write to NetCDF file
//! write_netcdf(&array, Path::new("data.nc"), "variable_name", None)
//!     .expect("Failed to write NetCDF file");
//!
//! // Read from NetCDF file
//! let loaded: Array<f64> = read_netcdf(Path::new("data.nc"), "variable_name")
//!     .expect("Failed to read NetCDF file");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use netcdf3::{DataSet, DataType as Nc3DataType, FileReader, FileWriter, Version};
use std::collections::HashMap;
use std::path::Path;

/// Write a NumRS2 array to a NetCDF file
///
/// # Arguments
/// * `array` - The array to write
/// * `path` - Path to the output NetCDF file
/// * `var_name` - Name for the variable in the NetCDF file
/// * `attrs` - Optional attributes to attach to the variable
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(NumRs2Error)` if writing fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::netcdf::write_netcdf;
/// use std::path::Path;
/// use std::collections::HashMap;
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let mut attrs = HashMap::new();
/// attrs.insert("units".to_string(), "meters".to_string());
///
/// write_netcdf(&array, Path::new("output.nc"), "temperature", Some(attrs))
///     .expect("Failed to write NetCDF file");
/// ```
pub fn write_netcdf<T, P>(
    array: &Array<T>,
    path: P,
    var_name: &str,
    attrs: Option<HashMap<String, String>>,
) -> Result<()>
where
    T: Clone + NetCdfWritable,
    P: AsRef<Path>,
{
    T::write_to_netcdf(array, path.as_ref(), var_name, attrs)
}

/// Read a NumRS2 array from a NetCDF file
///
/// # Arguments
/// * `path` - Path to the input NetCDF file
/// * `var_name` - Name of the variable to read
///
/// # Returns
/// * `Ok(Array<T>)` containing the loaded array
/// * `Err(NumRs2Error)` if reading fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::netcdf::read_netcdf;
/// use std::path::Path;
///
/// let array: Array<f64> = read_netcdf(Path::new("input.nc"), "temperature")
///     .expect("Failed to read NetCDF file");
/// ```
pub fn read_netcdf<T, P>(path: P, var_name: &str) -> Result<Array<T>>
where
    T: Clone + NetCdfReadable,
    P: AsRef<Path>,
{
    T::read_from_netcdf(path.as_ref(), var_name)
}

/// Trait for types that can be written to NetCDF format
pub trait NetCdfWritable: Clone {
    fn write_to_netcdf(
        array: &Array<Self>,
        path: &Path,
        var_name: &str,
        attrs: Option<HashMap<String, String>>,
    ) -> Result<()>;
}

/// Trait for types that can be read from NetCDF format
pub trait NetCdfReadable: Clone {
    fn read_from_netcdf(path: &Path, var_name: &str) -> Result<Array<Self>>;
}

/// Name of the NetCDF dimension for a given (0-based) array axis.
fn dim_name_for_axis(axis: usize) -> String {
    format!("dim{}", axis)
}

/// Builds a [`DataSet`] describing a single variable `var_name` of type
/// `data_type` over an array of the given `shape`, with one fixed-size
/// dimension per axis (see [`dim_name_for_axis`]).
fn build_data_set(shape: &[usize], var_name: &str, data_type: Nc3DataType) -> Result<DataSet> {
    if !netcdf3::is_valid_name(var_name) {
        return Err(NumRs2Error::InvalidInput(format!(
            "'{}' is not a valid NetCDF-3 variable name",
            var_name
        )));
    }

    let mut data_set = DataSet::new();
    let mut dim_names: Vec<String> = Vec::with_capacity(shape.len());
    for (axis, &size) in shape.iter().enumerate() {
        if size == 0 {
            return Err(NumRs2Error::InvalidInput(format!(
                "NetCDF-3 classic does not support a zero-length dimension (axis {} of variable '{}')",
                axis, var_name
            )));
        }
        let dim_name = dim_name_for_axis(axis);
        data_set.add_fixed_dim(&dim_name, size).map_err(|e| {
            NumRs2Error::IOError(format!(
                "Failed to define NetCDF dimension '{}': {:?}",
                dim_name, e
            ))
        })?;
        dim_names.push(dim_name);
    }

    data_set
        .add_var(var_name, &dim_names, data_type)
        .map_err(|e| {
            NumRs2Error::IOError(format!(
                "Failed to define NetCDF variable '{}': {:?}",
                var_name, e
            ))
        })?;

    Ok(data_set)
}

/// Attaches string-valued attributes to `var_name` in `data_set`.
fn apply_attrs(
    data_set: &mut DataSet,
    var_name: &str,
    attrs: &HashMap<String, String>,
) -> Result<()> {
    for (key, value) in attrs {
        if !netcdf3::is_valid_name(key) {
            return Err(NumRs2Error::InvalidInput(format!(
                "'{}' is not a valid NetCDF-3 attribute name",
                key
            )));
        }
        data_set
            .add_var_attr_string(var_name, key, value)
            .map_err(|e| {
                NumRs2Error::IOError(format!(
                    "Failed to set NetCDF attribute '{}' on variable '{}': {:?}",
                    key, var_name, e
                ))
            })?;
    }
    Ok(())
}

// Implements `NetCdfWritable`/`NetCdfReadable` for a Rust numeric type that
// has a direct, native NetCDF-3 classic counterpart.
macro_rules! impl_netcdf_native {
    ($rust_type:ty, $nc_type:expr, $write_var_fn:ident, $read_var_fn:ident) => {
        impl NetCdfWritable for $rust_type {
            fn write_to_netcdf(
                array: &Array<Self>,
                path: &Path,
                var_name: &str,
                attrs: Option<HashMap<String, String>>,
            ) -> Result<()> {
                let shape = array.shape();
                let mut data_set = build_data_set(&shape, var_name, $nc_type)?;
                if let Some(attrs) = &attrs {
                    apply_attrs(&mut data_set, var_name, attrs)?;
                }
                let data = array.to_vec();

                // `data_set` must outlive `writer`, which borrows it (see
                // `FileWriter::set_def`); declaring it first ensures that,
                // since locals drop in reverse declaration order.
                let mut writer = FileWriter::open(path).map_err(|e| {
                    NumRs2Error::IOError(format!(
                        "Failed to open NetCDF file '{}' for writing: {:?}",
                        path.display(),
                        e
                    ))
                })?;
                writer
                    .set_def(&data_set, Version::Classic, 0)
                    .map_err(|e| {
                        NumRs2Error::IOError(format!("Failed to write NetCDF header: {:?}", e))
                    })?;
                writer.$write_var_fn(var_name, &data).map_err(|e| {
                    NumRs2Error::IOError(format!(
                        "Failed to write NetCDF variable '{}': {:?}",
                        var_name, e
                    ))
                })?;
                writer.close().map_err(|e| {
                    NumRs2Error::IOError(format!("Failed to finalize NetCDF file: {:?}", e))
                })?;
                Ok(())
            }
        }

        impl NetCdfReadable for $rust_type {
            fn read_from_netcdf(path: &Path, var_name: &str) -> Result<Array<Self>> {
                let mut reader = FileReader::open(path).map_err(|e| {
                    NumRs2Error::IOError(format!(
                        "Failed to open NetCDF file '{}': {:?}",
                        path.display(),
                        e
                    ))
                })?;

                // Scoped so the immutable borrow of `reader` (via
                // `data_set()`) ends before the mutable `$read_var_fn` call
                // below.
                let (actual_type, shape): (Nc3DataType, Vec<usize>) = {
                    let var = reader.data_set().get_var(var_name).ok_or_else(|| {
                        NumRs2Error::DeserializationError(format!(
                            "Variable '{}' not found in NetCDF file",
                            var_name
                        ))
                    })?;
                    let shape = var.get_dims().iter().map(|d| d.size()).collect();
                    (var.data_type(), shape)
                };
                if actual_type != $nc_type {
                    return Err(NumRs2Error::DeserializationError(format!(
                        "Variable '{}' has NetCDF type {:?}, expected {:?}",
                        var_name, actual_type, $nc_type
                    )));
                }

                let data = reader.$read_var_fn(var_name).map_err(|e| {
                    NumRs2Error::DeserializationError(format!(
                        "Failed to read NetCDF variable '{}': {:?}",
                        var_name, e
                    ))
                })?;

                Ok(Array::from_vec(data).reshape(&shape))
            }
        }
    };
}

// Implements `NetCdfWritable`/`NetCdfReadable` for a Rust numeric type that
// has *no* native NetCDF-3 classic representation, returning a clear error
// instead of silently narrowing/widening the data.
macro_rules! impl_netcdf_unsupported {
    ($rust_type:ty, $type_label:expr) => {
        impl NetCdfWritable for $rust_type {
            fn write_to_netcdf(
                _array: &Array<Self>,
                _path: &Path,
                _var_name: &str,
                _attrs: Option<HashMap<String, String>>,
            ) -> Result<()> {
                Err(NumRs2Error::TypeCastError(format!(
                    "NetCDF-3 classic has no native {} type; supported numeric types are i8, u8, i16, i32, f32, f64",
                    $type_label
                )))
            }
        }

        impl NetCdfReadable for $rust_type {
            fn read_from_netcdf(_path: &Path, _var_name: &str) -> Result<Array<Self>> {
                Err(NumRs2Error::TypeCastError(format!(
                    "NetCDF-3 classic has no native {} type; supported numeric types are i8, u8, i16, i32, f32, f64",
                    $type_label
                )))
            }
        }
    };
}

impl_netcdf_native!(f64, Nc3DataType::F64, write_var_f64, read_var_f64);
impl_netcdf_native!(f32, Nc3DataType::F32, write_var_f32, read_var_f32);
impl_netcdf_native!(i8, Nc3DataType::I8, write_var_i8, read_var_i8);
impl_netcdf_native!(i16, Nc3DataType::I16, write_var_i16, read_var_i16);
impl_netcdf_native!(i32, Nc3DataType::I32, write_var_i32, read_var_i32);
impl_netcdf_native!(u8, Nc3DataType::U8, write_var_u8, read_var_u8);

// Classic NetCDF-3 has no native 64-bit integer or unsigned 32/64-bit
// integer type (see the module-level docs' "Type mapping" section).
impl_netcdf_unsupported!(i64, "i64");
impl_netcdf_unsupported!(u32, "u32");
impl_netcdf_unsupported!(u64, "u64");

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_netcdf_roundtrip_f64() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test.nc");

        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        // Write
        write_netcdf(&array, &path, "test_var", None).expect("Failed to write NetCDF");

        // Read
        let loaded: Array<f64> = read_netcdf(&path, "test_var").expect("Failed to read NetCDF");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_netcdf_with_attributes() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test_attrs.nc");

        let array = Array::from_vec(vec![10.0, 20.0, 30.0]);

        let mut attrs = HashMap::new();
        attrs.insert("units".to_string(), "meters".to_string());
        attrs.insert("long_name".to_string(), "temperature".to_string());

        // Write
        write_netcdf(&array, &path, "temp", Some(attrs)).expect("Failed to write NetCDF");

        // Read
        let loaded: Array<f64> = read_netcdf(&path, "temp").expect("Failed to read NetCDF");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_netcdf_multidimensional() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test_3d.nc");

        let array = Array::from_vec(vec![1.0; 24]).reshape(&[2, 3, 4]);

        // Write
        write_netcdf(&array, &path, "data", None).expect("Failed to write NetCDF");

        // Read
        let loaded: Array<f64> = read_netcdf(&path, "data").expect("Failed to read NetCDF");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_netcdf_overwrite_existing_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test_overwrite.nc");

        let first = Array::from_vec(vec![1.0, 2.0]);
        write_netcdf(&first, &path, "v", None).expect("Failed to write NetCDF");

        let second = Array::from_vec(vec![9.0, 8.0, 7.0]);
        write_netcdf(&second, &path, "v", None).expect("Failed to overwrite NetCDF file");

        let loaded: Array<f64> = read_netcdf(&path, "v").expect("Failed to read NetCDF");
        assert_eq!(loaded.to_vec(), vec![9.0, 8.0, 7.0]);
    }
}
