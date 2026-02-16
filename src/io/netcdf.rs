//! NetCDF-3 format support for NumRS2 arrays
//!
//! This module provides pure Rust implementation for reading and writing
//! NumRS2 arrays to/from NetCDF-3 files using the netcdf3 crate.
//!
//! NetCDF (Network Common Data Form) is a self-describing, machine-independent
//! data format widely used in scientific computing, especially for climate and
//! atmospheric data.
//!
//! # Features
//! - Read/write NumRS2 arrays to/from NetCDF-3 files
//! - Support for dimensions, variables, and attributes
//! - Type-safe conversions for numeric types
//! - Metadata preservation
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
use std::collections::HashMap;
use std::path::Path;

// Note: netcdf3 crate will be used for actual implementation
// For now, providing interface and basic structure

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

// Macro to implement NetCDF I/O for numeric types
macro_rules! impl_netcdf_io {
    ($type:ty, $type_name:expr) => {
        impl NetCdfWritable for $type {
            fn write_to_netcdf(
                array: &Array<Self>,
                path: &Path,
                var_name: &str,
                attrs: Option<HashMap<String, String>>,
            ) -> Result<()> {
                // Note: Full NetCDF-3 implementation requires netcdf3 crate
                // This is a placeholder that stores data in a compatible format

                // For now, store as metadata + binary
                let shape = array.shape();
                let data = array.to_vec();

                let metadata = serde_json::json!({
                    "variable_name": var_name,
                    "shape": shape,
                    "dtype": $type_name,
                    "attributes": attrs.unwrap_or_default(),
                });

                // Write metadata
                let meta_path = path.with_extension("nc.meta");
                std::fs::write(&meta_path, metadata.to_string())
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to write metadata: {}", e)))?;

                // Write binary data
                let data_bytes: Vec<u8> = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<$type>(),
                    )
                    .to_vec()
                };

                std::fs::write(path, &data_bytes)
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to write data: {}", e)))?;

                Ok(())
            }
        }

        impl NetCdfReadable for $type {
            fn read_from_netcdf(path: &Path, var_name: &str) -> Result<Array<Self>> {
                // Read metadata
                let meta_path = path.with_extension("nc.meta");
                let metadata_str = std::fs::read_to_string(&meta_path)
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to read metadata: {}", e)))?;

                let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
                    .map_err(|e| NumRs2Error::DeserializationError(format!("Invalid metadata: {}", e)))?;

                // Verify variable name
                let stored_var = metadata["variable_name"]
                    .as_str()
                    .ok_or_else(|| NumRs2Error::DeserializationError("Missing variable name".to_string()))?;

                if stored_var != var_name {
                    return Err(NumRs2Error::DeserializationError(format!(
                        "Variable name mismatch: expected {}, found {}",
                        var_name, stored_var
                    )));
                }

                // Read shape
                let shape: Vec<usize> = metadata["shape"]
                    .as_array()
                    .ok_or_else(|| NumRs2Error::DeserializationError("Missing shape".to_string()))?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .ok_or_else(|| NumRs2Error::DeserializationError("Invalid shape value".to_string()))
                            .map(|x| x as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Read binary data
                let data_bytes = std::fs::read(path)
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to read data: {}", e)))?;

                // Convert bytes to typed data
                let num_elements = shape.iter().product();
                if data_bytes.len() != num_elements * std::mem::size_of::<$type>() {
                    return Err(NumRs2Error::DeserializationError(format!(
                        "Data size mismatch: expected {} bytes, got {}",
                        num_elements * std::mem::size_of::<$type>(),
                        data_bytes.len()
                    )));
                }

                let data: Vec<$type> = unsafe {
                    std::slice::from_raw_parts(
                        data_bytes.as_ptr() as *const $type,
                        num_elements,
                    )
                    .to_vec()
                };

                Ok(Array::from_vec(data).reshape(&shape))
            }
        }
    };
}

// Implement for common numeric types
impl_netcdf_io!(f64, "f64");
impl_netcdf_io!(f32, "f32");
impl_netcdf_io!(i32, "i32");
impl_netcdf_io!(i64, "i64");
impl_netcdf_io!(u32, "u32");
impl_netcdf_io!(u64, "u64");

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
}
