//! BSON format support for NumRS2 arrays
//!
//! This module provides pure Rust implementation for serializing and deserializing
//! NumRS2 arrays to/from BSON format (Binary JSON), commonly used with MongoDB.
//!
//! # Features
//! - Serialize/deserialize NumRS2 arrays to/from BSON
//! - MongoDB-compatible format
//! - Type-safe conversions
//! - Shape and metadata preservation
//! - Pure Rust implementation (no C dependencies)
//!
//! # Example
//! ```no_run
//! use numrs2::prelude::*;
//! use numrs2::io::bson_format::{to_bson_file, from_bson_file};
//! use std::path::Path;
//!
//! // Create an array
//! let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//!
//! // Serialize to BSON file
//! to_bson_file(&array, Path::new("data.bson"))
//!     .expect("Failed to write BSON file");
//!
//! // Deserialize from BSON file
//! let loaded: Array<f64> = from_bson_file(Path::new("data.bson"))
//!     .expect("Failed to read BSON file");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use bson::{doc, Bson, Document};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Internal representation of an Array for BSON serialization
#[derive(Serialize, Deserialize)]
struct BsonArray<T> {
    /// Shape of the array
    shape: Vec<i64>,
    /// Flattened data
    data: Vec<T>,
    /// Data type name for validation
    dtype: String,
}

/// Serialize a NumRS2 array to BSON format and write to file
///
/// # Arguments
/// * `array` - The array to serialize
/// * `path` - Path to the output file
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(NumRs2Error)` if serialization fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::bson_format::to_bson_file;
/// use std::path::Path;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// to_bson_file(&array, Path::new("output.bson"))
///     .expect("Failed to write BSON file");
/// ```
pub fn to_bson_file<T, P>(array: &Array<T>, path: P) -> Result<()>
where
    T: Clone + Into<Bson>,
    P: AsRef<Path>,
{
    let file = File::create(path.as_ref())
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;

    let writer = BufWriter::new(file);

    to_bson_writer(array, writer)
}

/// Serialize a NumRS2 array to BSON format and write to a writer
///
/// # Arguments
/// * `array` - The array to serialize
/// * `writer` - Writer to write the serialized data to
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(NumRs2Error)` if serialization fails
pub fn to_bson_writer<T, W>(array: &Array<T>, mut writer: W) -> Result<()>
where
    T: Clone + Into<Bson>,
    W: std::io::Write,
{
    let shape: Vec<i64> = array.shape().iter().map(|&x| x as i64).collect();
    let data: Vec<Bson> = array.to_vec().into_iter().map(|x| x.into()).collect();

    let doc = doc! {
        "shape": shape,
        "data": data,
        "dtype": std::any::type_name::<T>(),
    };

    doc.to_writer(&mut writer)
        .map_err(|e| NumRs2Error::SerializationError(format!("BSON serialization error: {}", e)))?;

    Ok(())
}

/// Serialize a NumRS2 array to BSON Document
///
/// # Arguments
/// * `array` - The array to serialize
///
/// # Returns
/// * `Ok(Document)` containing the serialized data
/// * `Err(NumRs2Error)` if serialization fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::bson_format::to_bson_document;
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let doc = to_bson_document(&array)
///     .expect("Failed to serialize to BSON document");
/// ```
pub fn to_bson_document<T>(array: &Array<T>) -> Result<Document>
where
    T: Clone + Into<Bson>,
{
    let shape: Vec<i64> = array.shape().iter().map(|&x| x as i64).collect();
    let data: Vec<Bson> = array.to_vec().into_iter().map(|x| x.into()).collect();

    Ok(doc! {
        "shape": shape,
        "data": data,
        "dtype": std::any::type_name::<T>(),
    })
}

/// Local trait for converting from BSON values to numeric types
/// This trait avoids orphan trait violations by defining conversions locally
pub trait BsonConvertible: Sized {
    /// Convert from a BSON value to this type
    fn try_from_bson(value: Bson) -> std::result::Result<Self, String>;
}

/// Implement BsonConvertible for f64
impl BsonConvertible for f64 {
    fn try_from_bson(value: Bson) -> std::result::Result<Self, String> {
        match value {
            Bson::Double(d) => Ok(d),
            Bson::Int32(i) => Ok(i as f64),
            Bson::Int64(i) => Ok(i as f64),
            _ => Err(format!("Cannot convert {:?} to f64", value)),
        }
    }
}

/// Implement BsonConvertible for f32
impl BsonConvertible for f32 {
    fn try_from_bson(value: Bson) -> std::result::Result<Self, String> {
        match value {
            Bson::Double(d) => Ok(d as f32),
            Bson::Int32(i) => Ok(i as f32),
            Bson::Int64(i) => Ok(i as f32),
            _ => Err(format!("Cannot convert {:?} to f32", value)),
        }
    }
}

/// Implement BsonConvertible for i32
impl BsonConvertible for i32 {
    fn try_from_bson(value: Bson) -> std::result::Result<Self, String> {
        match value {
            Bson::Int32(i) => Ok(i),
            Bson::Int64(i) => Ok(i as i32),
            _ => Err(format!("Cannot convert {:?} to i32", value)),
        }
    }
}

/// Implement BsonConvertible for i64
impl BsonConvertible for i64 {
    fn try_from_bson(value: Bson) -> std::result::Result<Self, String> {
        match value {
            Bson::Int64(i) => Ok(i),
            Bson::Int32(i) => Ok(i as i64),
            _ => Err(format!("Cannot convert {:?} to i64", value)),
        }
    }
}

/// Deserialize a NumRS2 array from BSON format file
///
/// # Arguments
/// * `path` - Path to the input file
///
/// # Returns
/// * `Ok(Array<T>)` containing the deserialized array
/// * `Err(NumRs2Error)` if deserialization fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::bson_format::from_bson_file;
/// use std::path::Path;
///
/// let array: Array<f64> = from_bson_file(Path::new("input.bson"))
///     .expect("Failed to read BSON file");
/// ```
pub fn from_bson_file<T, P>(path: P) -> Result<Array<T>>
where
    T: Clone + BsonConvertible,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref())
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;

    let reader = BufReader::new(file);

    from_bson_reader(reader)
}

/// Deserialize a NumRS2 array from BSON format reader
///
/// # Arguments
/// * `reader` - Reader to read the serialized data from
///
/// # Returns
/// * `Ok(Array<T>)` containing the deserialized array
/// * `Err(NumRs2Error)` if deserialization fails
pub fn from_bson_reader<T, R>(mut reader: R) -> Result<Array<T>>
where
    T: Clone + BsonConvertible,
    R: std::io::Read,
{
    let doc = Document::from_reader(&mut reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("BSON deserialization error: {}", e))
    })?;

    from_bson_document(&doc)
}

/// Deserialize a NumRS2 array from BSON Document
///
/// # Arguments
/// * `doc` - BSON Document containing the array data
///
/// # Returns
/// * `Ok(Array<T>)` containing the deserialized array
/// * `Err(NumRs2Error)` if deserialization fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::bson_format::{to_bson_document, from_bson_document};
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let doc = to_bson_document(&array).expect("valid bson serialization");
///
/// let loaded: Array<f64> = from_bson_document(&doc)
///     .expect("Failed to deserialize from BSON document");
/// ```
pub fn from_bson_document<T>(doc: &Document) -> Result<Array<T>>
where
    T: Clone + BsonConvertible,
{
    // Extract shape
    let shape_bson = doc.get("shape").ok_or_else(|| {
        NumRs2Error::DeserializationError("Missing 'shape' field in BSON document".to_string())
    })?;

    let shape: Vec<usize> = if let Bson::Array(arr) = shape_bson {
        arr.iter()
            .map(|b| {
                if let Bson::Int64(i) = b {
                    Ok(*i as usize)
                } else if let Bson::Int32(i) = b {
                    Ok(*i as usize)
                } else {
                    Err(NumRs2Error::DeserializationError(
                        "Invalid shape value type".to_string(),
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        return Err(NumRs2Error::DeserializationError(
            "Shape field is not an array".to_string(),
        ));
    };

    // Extract data
    let data_bson = doc.get("data").ok_or_else(|| {
        NumRs2Error::DeserializationError("Missing 'data' field in BSON document".to_string())
    })?;

    let data: Vec<T> = if let Bson::Array(arr) = data_bson {
        arr.iter()
            .map(|b| {
                T::try_from_bson(b.clone()).map_err(|e| {
                    NumRs2Error::DeserializationError(format!(
                        "Failed to convert BSON value: {}",
                        e
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        return Err(NumRs2Error::DeserializationError(
            "Data field is not an array".to_string(),
        ));
    };

    let array = Array::from_vec(data).reshape(&shape);

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_bson_roundtrip_i32() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Write
        to_bson_file(&array, path).expect("Failed to write BSON");

        // Read
        let loaded: Array<i32> = from_bson_file(path).expect("Failed to read BSON");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_bson_roundtrip_f64() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Write
        to_bson_file(&array, path).expect("Failed to write BSON");

        // Read
        let loaded: Array<f64> = from_bson_file(path).expect("Failed to read BSON");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_bson_document() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        // Serialize to document
        let doc = to_bson_document(&array).expect("Failed to serialize to BSON document");

        // Deserialize from document
        let loaded: Array<f64> =
            from_bson_document(&doc).expect("Failed to deserialize from BSON document");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_bson_1d_array() {
        let array = Array::from_vec(vec![10, 20, 30, 40, 50]);

        let doc = to_bson_document(&array).expect("Failed to serialize");
        let loaded: Array<i32> = from_bson_document(&doc).expect("Failed to deserialize");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_bson_3d_array() {
        let array = Array::from_vec(vec![1.0; 24]).reshape(&[2, 3, 4]);

        let doc = to_bson_document(&array).expect("Failed to serialize");
        let loaded: Array<f64> = from_bson_document(&doc).expect("Failed to deserialize");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }
}
