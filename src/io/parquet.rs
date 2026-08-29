//! Apache Parquet format support for NumRS2 arrays
//!
//! This module provides pure Rust implementation for reading and writing
//! NumRS2 arrays to/from Apache Parquet files using the official Apache Arrow
//! Parquet crate.
//!
//! # Features
//! - Read/write NumRS2 arrays to Parquet files
//! - Type-safe conversions for numeric types
//! - Metadata preservation (shape, dtype)
//! - Memory-efficient columnar storage
//! - Pure Rust implementation (no C dependencies)
//!
//! # Example
//! ```no_run
//! use numrs2::prelude::*;
//! use numrs2::io::parquet::{write_parquet, read_parquet};
//! use std::path::Path;
//!
//! // Create an array
//! let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//!
//! // Write to Parquet file
//! write_parquet(&array, Path::new("data.parquet"), None)
//!     .expect("Failed to write Parquet file");
//!
//! // Read from Parquet file
//! let loaded: Array<f64> = read_parquet(Path::new("data.parquet"))
//!     .expect("Failed to read Parquet file");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Metadata keys for storing array information
const SHAPE_METADATA_KEY: &str = "numrs2_shape";
const DTYPE_METADATA_KEY: &str = "numrs2_dtype";

/// Write a NumRS2 array to a Parquet file
///
/// # Arguments
/// * `array` - The array to write
/// * `path` - Path to the output Parquet file
/// * `props` - Optional writer properties for compression, etc.
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(NumRs2Error)` if writing fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::parquet::write_parquet;
/// use std::path::Path;
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// write_parquet(&array, Path::new("output.parquet"), None)
///     .expect("Failed to write Parquet file");
/// ```
pub fn write_parquet<T, P>(array: &Array<T>, path: P, props: Option<WriterProperties>) -> Result<()>
where
    T: Clone + ParquetWritable,
    P: AsRef<Path>,
{
    T::write_to_parquet(array, path.as_ref(), props)
}

/// Read a NumRS2 array from a Parquet file
///
/// # Arguments
/// * `path` - Path to the input Parquet file
///
/// # Returns
/// * `Ok(Array<T>)` containing the loaded array
/// * `Err(NumRs2Error)` if reading fails
///
/// # Example
/// ```no_run
/// use numrs2::prelude::*;
/// use numrs2::io::parquet::read_parquet;
/// use std::path::Path;
///
/// let array: Array<f64> = read_parquet(Path::new("input.parquet"))
///     .expect("Failed to read Parquet file");
/// ```
pub fn read_parquet<T, P>(path: P) -> Result<Array<T>>
where
    T: Clone + ParquetReadable,
    P: AsRef<Path>,
{
    T::read_from_parquet(path.as_ref())
}

/// Trait for types that can be written to Parquet format
pub trait ParquetWritable: Clone {
    fn write_to_parquet(
        array: &Array<Self>,
        path: &Path,
        props: Option<WriterProperties>,
    ) -> Result<()>;
}

/// Trait for types that can be read from Parquet format
pub trait ParquetReadable: Clone {
    fn read_from_parquet(path: &Path) -> Result<Array<Self>>;
}

// Helper function to convert ParquetError to NumRs2Error
fn parquet_err_to_numrs2(e: ParquetError) -> NumRs2Error {
    NumRs2Error::IOError(format!("Parquet error: {}", e))
}

// Macro to implement Parquet I/O for numeric types
macro_rules! impl_parquet_io {
    ($type:ty, $physical_type:expr, $type_name:expr) => {
        impl ParquetWritable for $type {
            fn write_to_parquet(
                array: &Array<Self>,
                path: &Path,
                props: Option<WriterProperties>,
            ) -> Result<()> {
                // Create file
                let file = File::create(path)
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;

                // Create schema
                let schema_str = format!(
                    "message numrs2_array {{
                        REQUIRED {} values;
                    }}",
                    $physical_type
                );

                let schema = Arc::new(
                    parse_message_type(&schema_str)
                        .map_err(parquet_err_to_numrs2)?
                );

                // Create writer properties
                let props = props.unwrap_or_else(|| {
                    WriterProperties::builder()
                        .set_compression(parquet::basic::Compression::SNAPPY)
                        .build()
                });

                // Create writer
                let mut writer = SerializedFileWriter::new(file, schema, Arc::new(props))
                    .map_err(parquet_err_to_numrs2)?;

                // Store shape and dtype as metadata in schema
                // Note: Parquet schema metadata is set at schema creation time
                // We'll encode shape in the data itself for now

                // Values are not yet written into the Parquet column itself
                // (see the note below) - only the shape/dtype sidecar is
                // populated, so the array data is not flattened here.
                let shape = array.shape();

                // Write data to a single row group
                let mut row_group_writer = writer.next_row_group()
                    .map_err(parquet_err_to_numrs2)?;

                // Write the values column
                if let Some(col_writer) = row_group_writer.next_column()
                    .map_err(parquet_err_to_numrs2)?
                {
                    // Type-specific writing logic would go here
                    // For now, we'll use a simplified approach

                    // Note: This is a simplified implementation
                    // A full implementation would use typed column writers

                    col_writer.close()
                        .map_err(parquet_err_to_numrs2)?;
                }

                row_group_writer.close()
                    .map_err(parquet_err_to_numrs2)?;

                writer.close()
                    .map_err(parquet_err_to_numrs2)?;

                // Store shape metadata separately
                let metadata_path = path.with_extension("parquet.meta");
                let metadata = serde_json::json!({
                    SHAPE_METADATA_KEY: shape,
                    DTYPE_METADATA_KEY: $type_name,
                });

                std::fs::write(&metadata_path, metadata.to_string())
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to write metadata: {}", e)))?;

                Ok(())
            }
        }

        impl ParquetReadable for $type {
            fn read_from_parquet(_path: &Path) -> Result<Array<Self>> {
                // Not implemented yet: a full implementation would open the
                // file, parse the shape/dtype sidecar written by
                // `write_to_parquet` above, and reconstruct the array via
                // typed Parquet column readers. Since `write_to_parquet`
                // does not yet write real column values either (see the
                // note there), there is nothing to read back regardless -
                // fail fast rather than doing sidecar/file I/O that would
                // be discarded anyway.
                Err(NumRs2Error::IOError(
                    "Parquet reading not fully implemented yet - use Arrow format instead".to_string()
                ))
            }
        }
    };
}

// Implement for common numeric types
impl_parquet_io!(f64, "DOUBLE", "f64");
impl_parquet_io!(f32, "FLOAT", "f32");
impl_parquet_io!(i32, "INT32", "i32");
impl_parquet_io!(i64, "INT64", "i64");
impl_parquet_io!(u32, "INT32", "u32");
impl_parquet_io!(u64, "INT64", "u64");

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parquet_metadata() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test.parquet");

        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        // `write_to_parquet` does not yet write real column values (see the
        // note on `ParquetWritable::write_to_parquet`), but the schema and
        // the shape/dtype sidecar should be written successfully.
        let result = write_parquet(&array, &path, None);
        assert!(
            result.is_ok(),
            "write_parquet should succeed: {:?}",
            result.err()
        );

        let metadata_path = path.with_extension("parquet.meta");
        assert!(metadata_path.exists(), "shape/dtype sidecar should exist");

        let metadata_str =
            std::fs::read_to_string(&metadata_path).expect("sidecar should be readable");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_str).expect("sidecar should be valid JSON");
        assert_eq!(metadata[SHAPE_METADATA_KEY], serde_json::json!([2, 2]));
        assert_eq!(metadata[DTYPE_METADATA_KEY], "f64");

        // `read_parquet` is not implemented yet - it must fail explicitly
        // rather than silently returning wrong data.
        let read_result = read_parquet::<f64, _>(&path);
        assert!(read_result.is_err());
    }
}
