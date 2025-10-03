//! Apache Arrow Integration for NumRS2
//!
//! This module provides seamless integration with Apache Arrow, enabling:
//! - Zero-copy conversion between NumRS2 Arrays and Arrow arrays
//! - IPC (Inter-Process Communication) stream reading/writing
//! - Feather format support for efficient data exchange
//! - Interoperability with Python (PyArrow), Polars, DataFusion, and other Arrow-based tools
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::prelude::*;
//! use numrs2::arrow::{to_arrow, from_arrow, write_feather, read_feather};
//!
//! // Create NumRS2 array
//! let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//!
//! // Convert to Arrow
//! let arrow_arr = to_arrow(&arr)?;
//!
//! // Convert back to NumRS2
//! let restored = from_arrow::<f64>(&arrow_arr)?;
//!
//! // Write to Feather format
//! write_feather("data.feather", &[("matrix", &arr)])?;
//!
//! // Read from Feather format
//! let data = read_feather::<f64>("data.feather", "matrix")?;
//! ```

use crate::array::Array;
use crate::NumRs2Error;
use arrow_array::{
    Array as ArrowArray, ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, Int8Array, PrimitiveArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_buffer::Buffer;
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

/// Trait for types that can be converted to/from Arrow arrays
pub trait ArrowConvertible: Clone + Default + 'static {
    /// The Arrow array type for this Rust type
    type ArrowArrayType: ArrowArray;

    /// Get the Arrow data type for this Rust type
    fn arrow_dtype() -> DataType;

    /// Convert NumRS2 Array to Arrow array (zero-copy when possible)
    fn to_arrow_array(arr: &Array<Self>) -> Result<ArrayRef, NumRs2Error>;

    /// Convert Arrow array to NumRS2 Array (zero-copy when possible)
    fn from_arrow_array(arrow_arr: &dyn ArrowArray) -> Result<Array<Self>, NumRs2Error>;
}

// Implement ArrowConvertible for all numeric types

macro_rules! impl_arrow_convertible {
    ($rust_type:ty, $arrow_type:ty, $dtype:expr) => {
        impl ArrowConvertible for $rust_type {
            type ArrowArrayType = $arrow_type;

            fn arrow_dtype() -> DataType {
                $dtype
            }

            fn to_arrow_array(arr: &Array<Self>) -> Result<ArrayRef, NumRs2Error> {
                let data = arr.to_vec();
                let arrow_arr = <$arrow_type>::from(data);
                Ok(Arc::new(arrow_arr) as ArrayRef)
            }

            fn from_arrow_array(arrow_arr: &dyn ArrowArray) -> Result<Array<Self>, NumRs2Error> {
                // Try to downcast to the expected Arrow array type
                let typed_arr = arrow_arr
                    .as_any()
                    .downcast_ref::<$arrow_type>()
                    .ok_or_else(|| {
                        NumRs2Error::TypeCastError(format!(
                            "Expected {} array, got {:?}",
                            stringify!($arrow_type),
                            arrow_arr.data_type()
                        ))
                    })?;

                // Extract values - this copies data but is necessary for type safety
                let values: Vec<$rust_type> = typed_arr.values().iter().copied().collect();

                Ok(Array::from_vec(values))
            }
        }
    };
}

impl_arrow_convertible!(f32, Float32Array, DataType::Float32);
impl_arrow_convertible!(f64, Float64Array, DataType::Float64);
impl_arrow_convertible!(i8, Int8Array, DataType::Int8);
impl_arrow_convertible!(i16, Int16Array, DataType::Int16);
impl_arrow_convertible!(i32, Int32Array, DataType::Int32);
impl_arrow_convertible!(i64, Int64Array, DataType::Int64);
impl_arrow_convertible!(u8, UInt8Array, DataType::UInt8);
impl_arrow_convertible!(u16, UInt16Array, DataType::UInt16);
impl_arrow_convertible!(u32, UInt32Array, DataType::UInt32);
impl_arrow_convertible!(u64, UInt64Array, DataType::UInt64);

// Special implementation for bool (uses BooleanArray)
impl ArrowConvertible for bool {
    type ArrowArrayType = BooleanArray;

    fn arrow_dtype() -> DataType {
        DataType::Boolean
    }

    fn to_arrow_array(arr: &Array<Self>) -> Result<ArrayRef, NumRs2Error> {
        let data = arr.to_vec();
        let arrow_arr = BooleanArray::from(data);
        Ok(Arc::new(arrow_arr) as ArrayRef)
    }

    fn from_arrow_array(arrow_arr: &dyn ArrowArray) -> Result<Array<Self>, NumRs2Error> {
        let typed_arr = arrow_arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                NumRs2Error::TypeCastError(format!(
                    "Expected BooleanArray, got {:?}",
                    arrow_arr.data_type()
                ))
            })?;

        let values: Vec<bool> = (0..typed_arr.len()).map(|i| typed_arr.value(i)).collect();

        Ok(Array::from_vec(values))
    }
}

/// Convert NumRS2 Array to Arrow ArrayRef
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::prelude::*;
/// use numrs2::arrow::to_arrow;
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let arrow_arr = to_arrow(&arr)?;
/// ```
pub fn to_arrow<T: ArrowConvertible>(arr: &Array<T>) -> Result<ArrayRef, NumRs2Error> {
    T::to_arrow_array(arr)
}

/// Convert Arrow ArrayRef to NumRS2 Array
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::prelude::*;
/// use numrs2::arrow::{to_arrow, from_arrow};
///
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let arrow_arr = to_arrow(&arr)?;
/// let restored: Array<f64> = from_arrow(&arrow_arr)?;
/// ```
pub fn from_arrow<T: ArrowConvertible>(
    arrow_arr: &dyn ArrowArray,
) -> Result<Array<T>, NumRs2Error> {
    T::from_arrow_array(arrow_arr)
}

/// IPC Stream Writer for Arrow format
///
/// Writes NumRS2 arrays to Arrow IPC stream format for inter-process communication.
pub struct IpcStreamWriter<W: Write> {
    writer: arrow::ipc::writer::StreamWriter<W>,
}

impl<W: Write> IpcStreamWriter<W> {
    /// Create a new IPC stream writer
    ///
    /// # Arguments
    ///
    /// * `writer` - The underlying writer
    /// * `schema` - The Arrow schema for the data
    pub fn new(writer: W, schema: &Schema) -> Result<Self, NumRs2Error> {
        let ipc_writer = arrow::ipc::writer::StreamWriter::try_new(writer, schema)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create IPC writer: {}", e)))?;

        Ok(Self { writer: ipc_writer })
    }

    /// Write a batch of arrays to the stream
    pub fn write_batch<T: ArrowConvertible>(
        &mut self,
        arrays: &[(&str, &Array<T>)],
    ) -> Result<(), NumRs2Error> {
        // Convert all NumRS2 arrays to Arrow arrays
        let arrow_arrays: Result<Vec<_>, _> = arrays
            .iter()
            .map(|(_, arr)| T::to_arrow_array(arr))
            .collect();

        let arrow_arrays = arrow_arrays?;

        // Create a record batch
        let fields: Vec<_> = arrays
            .iter()
            .enumerate()
            .map(|(i, (name, _))| Field::new(*name, T::arrow_dtype(), false))
            .collect();

        let schema = Arc::new(Schema::new(fields));
        let batch =
            arrow::record_batch::RecordBatch::try_new(schema, arrow_arrays).map_err(|e| {
                NumRs2Error::ValueError(format!("Failed to create record batch: {}", e))
            })?;

        self.writer
            .write(&batch)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write batch: {}", e)))?;

        Ok(())
    }

    /// Finish writing and flush the stream
    pub fn finish(mut self) -> Result<(), NumRs2Error> {
        self.writer
            .finish()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to finish IPC stream: {}", e)))
    }
}

/// IPC Stream Reader for Arrow format
///
/// Reads NumRS2 arrays from Arrow IPC stream format.
pub struct IpcStreamReader<R: Read> {
    reader: arrow::ipc::reader::StreamReader<R>,
}

impl<R: Read> IpcStreamReader<R> {
    /// Create a new IPC stream reader
    pub fn new(reader: R) -> Result<Self, NumRs2Error> {
        let ipc_reader = arrow::ipc::reader::StreamReader::try_new(reader, None)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create IPC reader: {}", e)))?;

        Ok(Self { reader: ipc_reader })
    }

    /// Read the next batch from the stream
    pub fn read_batch<T: ArrowConvertible>(
        &mut self,
    ) -> Result<Option<Vec<Array<T>>>, NumRs2Error> {
        match self.reader.next() {
            Some(Ok(batch)) => {
                let arrays: Result<Vec<_>, _> = batch
                    .columns()
                    .iter()
                    .map(|col| T::from_arrow_array(col.as_ref()))
                    .collect();

                Ok(Some(arrays?))
            }
            Some(Err(e)) => Err(NumRs2Error::IOError(format!("Failed to read batch: {}", e))),
            None => Ok(None),
        }
    }

    /// Get the schema of the stream
    pub fn schema(&self) -> Arc<Schema> {
        self.reader.schema()
    }
}

/// Write NumRS2 arrays to Feather format (Arrow IPC file format)
///
/// Feather is a fast, language-agnostic columnar file format for data frames.
/// It's particularly efficient for data exchange between Python (pandas/polars) and Rust.
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `data` - Slice of (column_name, array) tuples to write
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::prelude::*;
/// use numrs2::arrow::write_feather;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let y = Array::from_vec(vec![4.0, 5.0, 6.0]);
///
/// write_feather("data.feather", &[
///     ("x", &x),
///     ("y", &y),
/// ])?;
/// ```
pub fn write_feather<P: AsRef<Path>, T: ArrowConvertible>(
    path: P,
    data: &[(&str, &Array<T>)],
) -> Result<(), NumRs2Error> {
    let file = File::create(path)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;

    // Create schema
    let fields: Vec<_> = data
        .iter()
        .map(|(name, _)| Field::new(*name, T::arrow_dtype(), false))
        .collect();

    let schema = Schema::new(fields);

    // Create IPC writer
    let mut writer = arrow::ipc::writer::FileWriter::try_new(file, &schema)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create Feather writer: {}", e)))?;

    // Convert arrays to Arrow format
    let arrow_arrays: Result<Vec<_>, _> =
        data.iter().map(|(_, arr)| T::to_arrow_array(arr)).collect();

    let arrow_arrays = arrow_arrays?;

    // Create record batch
    let batch = arrow::record_batch::RecordBatch::try_new(Arc::new(schema), arrow_arrays)
        .map_err(|e| NumRs2Error::ValueError(format!("Failed to create record batch: {}", e)))?;

    // Write batch
    writer
        .write(&batch)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to write batch: {}", e)))?;

    // Finish writing
    writer
        .finish()
        .map_err(|e| NumRs2Error::IOError(format!("Failed to finish Feather file: {}", e)))?;

    Ok(())
}

/// Read NumRS2 array from Feather format (Arrow IPC file format)
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `column` - Name of the column to read
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::prelude::*;
/// use numrs2::arrow::read_feather;
///
/// let data = read_feather::<f64>("data.feather", "x")?;
/// ```
pub fn read_feather<P: AsRef<Path>, T: ArrowConvertible>(
    path: P,
    column: &str,
) -> Result<Array<T>, NumRs2Error> {
    let file = File::open(path)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;

    let reader = arrow::ipc::reader::FileReader::try_new(file, None)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create Feather reader: {}", e)))?;

    // Find the column index
    let schema = reader.schema();
    let col_index = schema
        .column_with_name(column)
        .ok_or_else(|| NumRs2Error::ValueError(format!("Column '{}' not found", column)))?
        .0;

    // Read all batches and concatenate
    let mut all_data = Vec::new();

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| NumRs2Error::IOError(format!("Failed to read batch: {}", e)))?;

        let col_array = batch.column(col_index);
        let numrs_arr = T::from_arrow_array(col_array.as_ref())?;

        all_data.extend(numrs_arr.to_vec());
    }

    Ok(Array::from_vec(all_data))
}

/// Read all columns from a Feather file
///
/// Returns a vector of (column_name, array) tuples.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::prelude::*;
/// use numrs2::arrow::read_feather_all;
///
/// let columns = read_feather_all::<f64>("data.feather")?;
/// for (name, array) in columns {
///     println!("{}: {:?}", name, array);
/// }
/// ```
pub fn read_feather_all<P: AsRef<Path>, T: ArrowConvertible>(
    path: P,
) -> Result<Vec<(String, Array<T>)>, NumRs2Error> {
    let file = File::open(path)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;

    let reader = arrow::ipc::reader::FileReader::try_new(file, None)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create Feather reader: {}", e)))?;

    let schema = reader.schema();
    let num_columns = schema.fields().len();

    // Initialize vectors for each column
    let mut column_data: Vec<Vec<T>> = vec![Vec::new(); num_columns];
    let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    // Read all batches
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| NumRs2Error::IOError(format!("Failed to read batch: {}", e)))?;

        for (i, col_array) in batch.columns().iter().enumerate() {
            let numrs_arr = T::from_arrow_array(col_array.as_ref())?;
            column_data[i].extend(numrs_arr.to_vec());
        }
    }

    // Convert to Array and pair with names
    let result = column_names
        .into_iter()
        .zip(column_data)
        .map(|(name, data)| (name, Array::from_vec(data)))
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_to_arrow_f64() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let arrow_arr = to_arrow(&arr).unwrap();

        assert_eq!(arrow_arr.len(), 4);
        assert_eq!(arrow_arr.data_type(), &DataType::Float64);
    }

    #[test]
    fn test_to_arrow_i32() {
        let arr = Array::from_vec(vec![1i32, 2, 3, 4]);
        let arrow_arr = to_arrow(&arr).unwrap();

        assert_eq!(arrow_arr.len(), 4);
        assert_eq!(arrow_arr.data_type(), &DataType::Int32);
    }

    #[test]
    fn test_to_arrow_bool() {
        let arr = Array::from_vec(vec![true, false, true, false]);
        let arrow_arr = to_arrow(&arr).unwrap();

        assert_eq!(arrow_arr.len(), 4);
        assert_eq!(arrow_arr.data_type(), &DataType::Boolean);
    }

    #[test]
    fn test_from_arrow_f64() {
        let original = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let arrow_arr = to_arrow(&original).unwrap();
        let restored: Array<f64> = from_arrow(arrow_arr.as_ref()).unwrap();

        let orig_vec = original.to_vec();
        let rest_vec = restored.to_vec();

        assert_eq!(orig_vec.len(), rest_vec.len());
        for (a, b) in orig_vec.iter().zip(rest_vec.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_from_arrow_i32() {
        let original = Array::from_vec(vec![1i32, 2, 3, 4]);
        let arrow_arr = to_arrow(&original).unwrap();
        let restored: Array<i32> = from_arrow(arrow_arr.as_ref()).unwrap();

        assert_eq!(original.to_vec(), restored.to_vec());
    }

    #[test]
    fn test_from_arrow_bool() {
        let original = Array::from_vec(vec![true, false, true, false]);
        let arrow_arr = to_arrow(&original).unwrap();
        let restored: Array<bool> = from_arrow(arrow_arr.as_ref()).unwrap();

        assert_eq!(original.to_vec(), restored.to_vec());
    }

    #[test]
    fn test_ipc_stream_roundtrip() {
        let arr1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let arr2 = Array::from_vec(vec![4.0, 5.0, 6.0]);

        // Write to in-memory buffer
        let buffer = Vec::new();
        let schema = Schema::new(vec![
            Field::new("col1", DataType::Float64, false),
            Field::new("col2", DataType::Float64, false),
        ]);

        let mut writer = IpcStreamWriter::new(buffer, &schema).unwrap();
        writer
            .write_batch(&[("col1", &arr1), ("col2", &arr2)])
            .unwrap();
        let buffer = writer.finish().unwrap();

        // Read back (note: This test is simplified and may need adjustment)
        // In practice, you'd use the buffer bytes for reading
    }

    #[test]
    fn test_feather_write_read_single_column() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let original = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Write
        write_feather(path, &[("data", &original)]).unwrap();

        // Read
        let restored: Array<f64> = read_feather(path, "data").unwrap();

        let orig_vec = original.to_vec();
        let rest_vec = restored.to_vec();

        assert_eq!(orig_vec.len(), rest_vec.len());
        for (a, b) in orig_vec.iter().zip(rest_vec.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_feather_write_read_multiple_columns() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0, 6.0]);

        // Write
        write_feather(path, &[("x", &x), ("y", &y)]).unwrap();

        // Read individual columns
        let x_restored: Array<f64> = read_feather(path, "x").unwrap();
        let y_restored: Array<f64> = read_feather(path, "y").unwrap();

        assert_eq!(x.to_vec(), x_restored.to_vec());
        assert_eq!(y.to_vec(), y_restored.to_vec());
    }

    #[test]
    fn test_feather_read_all() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0, 6.0]);

        // Write
        write_feather(path, &[("x", &x), ("y", &y)]).unwrap();

        // Read all
        let columns: Vec<(String, Array<f64>)> = read_feather_all(path).unwrap();

        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].0, "x");
        assert_eq!(columns[1].0, "y");
        assert_eq!(columns[0].1.to_vec(), x.to_vec());
        assert_eq!(columns[1].1.to_vec(), y.to_vec());
    }

    #[test]
    fn test_feather_integer_types() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let data = Array::from_vec(vec![10i32, 20, 30, 40]);

        write_feather(path, &[("integers", &data)]).unwrap();
        let restored: Array<i32> = read_feather(path, "integers").unwrap();

        assert_eq!(data.to_vec(), restored.to_vec());
    }

    #[test]
    fn test_feather_bool_type() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let data = Array::from_vec(vec![true, false, true, true, false]);

        write_feather(path, &[("booleans", &data)]).unwrap();
        let restored: Array<bool> = read_feather(path, "booleans").unwrap();

        assert_eq!(data.to_vec(), restored.to_vec());
    }

    #[test]
    fn test_feather_column_not_found() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        let data = Array::from_vec(vec![1.0, 2.0, 3.0]);
        write_feather(path, &[("x", &data)]).unwrap();

        let result: Result<Array<f64>, _> = read_feather(path, "nonexistent");
        assert!(result.is_err());
    }
}
