//! MessagePack serialization support for NumRS2 arrays
//!
//! This module provides pure Rust implementation for serializing and deserializing
//! NumRS2 arrays to/from MessagePack format using rmp-serde.
//!
//! MessagePack is a compact binary serialization format that is more efficient
//! than JSON while maintaining language-agnostic compatibility.
//!
//! # Features
//! - Serialize/deserialize NumRS2 arrays to/from MessagePack
//! - Compact binary format (smaller than JSON)
//! - Type-safe conversions
//! - Shape and metadata preservation
//! - Pure Rust implementation (no C dependencies)
//!
//! # Example
//! ```no_run
//! use numrs2::prelude::*;
//! use numrs2::io::messagepack::{to_messagepack, from_messagepack};
//! use std::path::Path;
//!
//! // Create an array
//! let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//!
//! // Serialize to MessagePack file
//! to_messagepack(&array, Path::new("data.msgpack"))
//!     .expect("Failed to write MessagePack file");
//!
//! // Deserialize from MessagePack file
//! let loaded: Array<f64> = from_messagepack(Path::new("data.msgpack"))
//!     .expect("Failed to read MessagePack file");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Internal representation of an Array for MessagePack serialization
#[derive(Serialize, Deserialize)]
struct MessagePackArray<T> {
    /// Shape of the array
    shape: Vec<usize>,
    /// Flattened data
    data: Vec<T>,
    /// Data type name for validation
    dtype: String,
}

/// Serialize a NumRS2 array to MessagePack format and write to file
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
/// use numrs2::io::messagepack::to_messagepack;
/// use std::path::Path;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// to_messagepack(&array, Path::new("output.msgpack"))
///     .expect("Failed to write MessagePack file");
/// ```
pub fn to_messagepack<T, P>(array: &Array<T>, path: P) -> Result<()>
where
    T: Clone + Serialize,
    P: AsRef<Path>,
{
    let file = File::create(path.as_ref())
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;

    let mut writer = BufWriter::new(file);

    to_messagepack_writer(array, &mut writer)
}

/// Serialize a NumRS2 array to MessagePack format and write to a writer
///
/// # Arguments
/// * `array` - The array to serialize
/// * `writer` - Writer to write the serialized data to
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(NumRs2Error)` if serialization fails
pub fn to_messagepack_writer<T, W>(array: &Array<T>, writer: &mut W) -> Result<()>
where
    T: Clone + Serialize,
    W: Write,
{
    let msgpack_array = MessagePackArray {
        shape: array.shape(),
        data: array.to_vec(),
        dtype: std::any::type_name::<T>().to_string(),
    };

    rmp_serde::encode::write(writer, &msgpack_array).map_err(|e| {
        NumRs2Error::SerializationError(format!("MessagePack serialization error: {}", e))
    })?;

    Ok(())
}

/// Serialize a NumRS2 array to MessagePack bytes
///
/// # Arguments
/// * `array` - The array to serialize
///
/// # Returns
/// * `Ok(Vec<u8>)` containing the serialized bytes
/// * `Err(NumRs2Error)` if serialization fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::messagepack::to_messagepack_bytes;
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let bytes = to_messagepack_bytes(&array)
///     .expect("Failed to serialize to bytes");
/// ```
pub fn to_messagepack_bytes<T>(array: &Array<T>) -> Result<Vec<u8>>
where
    T: Clone + Serialize,
{
    let msgpack_array = MessagePackArray {
        shape: array.shape(),
        data: array.to_vec(),
        dtype: std::any::type_name::<T>().to_string(),
    };

    rmp_serde::to_vec(&msgpack_array).map_err(|e| {
        NumRs2Error::SerializationError(format!("MessagePack serialization error: {}", e))
    })
}

/// Deserialize a NumRS2 array from MessagePack format file
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
/// use numrs2::io::messagepack::from_messagepack;
/// use std::path::Path;
///
/// let array: Array<i32> = from_messagepack(Path::new("input.msgpack"))
///     .expect("Failed to read MessagePack file");
/// ```
pub fn from_messagepack<T, P>(path: P) -> Result<Array<T>>
where
    T: Clone + for<'de> Deserialize<'de>,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref())
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;

    let mut reader = BufReader::new(file);

    from_messagepack_reader(&mut reader)
}

/// Deserialize a NumRS2 array from MessagePack format reader
///
/// # Arguments
/// * `reader` - Reader to read the serialized data from
///
/// # Returns
/// * `Ok(Array<T>)` containing the deserialized array
/// * `Err(NumRs2Error)` if deserialization fails
pub fn from_messagepack_reader<T, R>(reader: &mut R) -> Result<Array<T>>
where
    T: Clone + for<'de> Deserialize<'de>,
    R: Read,
{
    let msgpack_array: MessagePackArray<T> = rmp_serde::from_read(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("MessagePack deserialization error: {}", e))
    })?;

    let array = Array::from_vec(msgpack_array.data).reshape(&msgpack_array.shape);

    Ok(array)
}

/// Deserialize a NumRS2 array from MessagePack bytes
///
/// # Arguments
/// * `bytes` - Byte slice containing MessagePack data
///
/// # Returns
/// * `Ok(Array<T>)` containing the deserialized array
/// * `Err(NumRs2Error)` if deserialization fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::messagepack::{to_messagepack_bytes, from_messagepack_bytes};
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let bytes = to_messagepack_bytes(&array).unwrap();
///
/// let loaded: Array<f64> = from_messagepack_bytes(&bytes)
///     .expect("Failed to deserialize from bytes");
/// ```
pub fn from_messagepack_bytes<T>(bytes: &[u8]) -> Result<Array<T>>
where
    T: Clone + for<'de> Deserialize<'de>,
{
    let msgpack_array: MessagePackArray<T> = rmp_serde::from_slice(bytes).map_err(|e| {
        NumRs2Error::DeserializationError(format!("MessagePack deserialization error: {}", e))
    })?;

    let array = Array::from_vec(msgpack_array.data).reshape(&msgpack_array.shape);

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_messagepack_roundtrip_i32() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Write
        to_messagepack(&array, path).expect("Failed to write MessagePack");

        // Read
        let loaded: Array<i32> = from_messagepack(path).expect("Failed to read MessagePack");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_messagepack_roundtrip_f64() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Write
        to_messagepack(&array, path).expect("Failed to write MessagePack");

        // Read
        let loaded: Array<f64> = from_messagepack(path).expect("Failed to read MessagePack");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_messagepack_bytes() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        // Serialize to bytes
        let bytes = to_messagepack_bytes(&array).expect("Failed to serialize to bytes");

        // Deserialize from bytes
        let loaded: Array<f64> =
            from_messagepack_bytes(&bytes).expect("Failed to deserialize from bytes");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_messagepack_1d_array() {
        let array = Array::from_vec(vec![10, 20, 30, 40, 50]);

        let bytes = to_messagepack_bytes(&array).expect("Failed to serialize");
        let loaded: Array<i32> = from_messagepack_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }

    #[test]
    fn test_messagepack_3d_array() {
        let array = Array::from_vec(vec![1.0; 24]).reshape(&[2, 3, 4]);

        let bytes = to_messagepack_bytes(&array).expect("Failed to serialize");
        let loaded: Array<f64> = from_messagepack_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(array.shape(), loaded.shape());
        assert_eq!(array.to_vec(), loaded.to_vec());
    }
}
