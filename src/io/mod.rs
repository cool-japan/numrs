use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

mod npy_npz;

pub use npy_npz::*;

/// Module for input/output operations with NumRS arrays.
///
/// This module provides functionality for:
/// 1. Serializing/deserializing arrays to/from various formats
/// 2. Reading and writing arrays from/to files
/// 3. Converting between arrays and other data formats
///    
/// Internal representation of an Array for serialization
#[derive(Serialize, Deserialize)]
struct SerializedArray<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

/// Enum representing different serialization formats
#[derive(Clone, Copy, Debug)]
pub enum SerializeFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Binary format
    Binary,
    /// NumPy NPY format (*.npy)
    Npy,
    /// NumPy NPZ format (zipped NPY files, *.npz)
    Npz,
}

impl<T: Clone + Serialize> Array<T> {
    /// Serialize the array to a string in the specified format
    ///
    /// # Arguments
    ///
    /// * `format` - The format to serialize to
    ///
    /// # Returns
    ///
    /// A Result containing the serialized string or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::io::SerializeFormat;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let json = array.to_string(SerializeFormat::Json).unwrap();
    /// ```
    pub fn to_string(&self, format: SerializeFormat) -> Result<String> {
        let serialized = SerializedArray {
            shape: self.shape(),
            data: self.to_vec(),
        };

        match format {
            SerializeFormat::Json => serde_json::to_string(&serialized).map_err(|e| {
                NumRs2Error::SerializationError(format!("JSON serialization error: {}", e))
            }),
            SerializeFormat::Csv => {
                // For CSV, we'll just write the flattened data as there's no standard
                // for multidimensional arrays in CSV
                let mut writer = csv::Writer::from_writer(vec![]);
                let data = self.to_vec();
                writer.serialize(&data).map_err(|e| {
                    NumRs2Error::SerializationError(format!("CSV serialization error: {}", e))
                })?;

                let csv_bytes = writer.into_inner().map_err(|e| {
                    NumRs2Error::SerializationError(format!("CSV serialization error: {}", e))
                })?;

                String::from_utf8(csv_bytes).map_err(|e| {
                    NumRs2Error::SerializationError(format!("CSV serialization error: {}", e))
                })
            }
            SerializeFormat::Binary => Err(NumRs2Error::SerializationError(
                "Binary serialization to string not supported".to_string(),
            )),
            SerializeFormat::Npy => Err(NumRs2Error::SerializationError(
                "NPY format serialization to string not supported".to_string(),
            )),
            SerializeFormat::Npz => Err(NumRs2Error::SerializationError(
                "NPZ format serialization to string not supported".to_string(),
            )),
        }
    }

    /// Serialize the array to a file in the specified format
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file
    /// * `format` - The format to serialize to
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::io::SerializeFormat;
    /// use std::path::Path;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// // Uncomment to actually write to file:
    /// // array.to_file(Path::new("array.json"), SerializeFormat::Json).unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: SerializeFormat) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        let serialized = SerializedArray {
            shape: self.shape(),
            data: self.to_vec(),
        };

        match format {
            SerializeFormat::Json => {
                let json = serde_json::to_string(&serialized).map_err(|e| {
                    NumRs2Error::SerializationError(format!("JSON serialization error: {}", e))
                })?;
                writer
                    .write_all(json.as_bytes())
                    .map_err(|e| NumRs2Error::IOError(format!("Failed to write to file: {}", e)))?;
            }
            SerializeFormat::Csv => {
                let mut csv_writer = csv::Writer::from_writer(writer);
                for row in self.to_row_vectors()? {
                    csv_writer.serialize(row).map_err(|e| {
                        NumRs2Error::SerializationError(format!("CSV serialization error: {}", e))
                    })?;
                }
                csv_writer.flush().map_err(|e| {
                    NumRs2Error::IOError(format!("Failed to flush CSV writer: {}", e))
                })?;
            }
            SerializeFormat::Binary => {
                bincode::serialize_into(&mut writer, &serialized).map_err(|e| {
                    NumRs2Error::SerializationError(format!("Binary serialization error: {}", e))
                })?;
            }
            SerializeFormat::Npy | SerializeFormat::Npz => {
                // Delegate to the NPY/NPZ module
                npy_npz::serialize_to_file(self, &mut writer, format)?;
            }
        }

        Ok(())
    }

    /// Convert the array to a vector of row vectors (for CSV export)
    ///
    /// # Returns
    ///
    /// A Result containing the rows or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let rows = array.to_row_vectors().unwrap();
    /// assert_eq!(rows, vec![vec![1, 2], vec![3, 4]]);
    /// ```
    pub fn to_row_vectors(&self) -> Result<Vec<Vec<T>>> {
        if self.ndim() == 1 {
            // For 1D arrays, return a single row
            return Ok(vec![self.to_vec()]);
        } else if self.ndim() == 2 {
            let shape = self.shape();
            let rows = shape[0];
            let cols = shape[1];
            let data = self.to_vec();

            let mut result = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols {
                    let idx = i * cols + j;
                    row.push(data[idx].clone());
                }
                result.push(row);
            }
            return Ok(result);
        }

        // For higher dimensional arrays, flatten to 2D first
        Err(NumRs2Error::DimensionMismatch(
            "Cannot convert arrays with more than 2 dimensions to CSV rows".to_string(),
        ))
    }
}

impl<T: Clone + for<'a> Deserialize<'a>> Array<T> {
    /// Deserialize an array from a string
    ///
    /// # Arguments
    ///
    /// * `s` - The string to deserialize from
    /// * `format` - The format of the string
    ///
    /// # Returns
    ///
    /// A Result containing the deserialized array or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::io::SerializeFormat;
    ///
    /// let json = r#"{"shape":[2,2],"data":[1,2,3,4]}"#;
    /// let array = Array::<i32>::from_string(json, SerializeFormat::Json).unwrap();
    /// assert_eq!(array.shape(), vec![2, 2]);
    /// assert_eq!(array.to_vec(), vec![1, 2, 3, 4]);
    /// ```
    pub fn from_string(s: &str, format: SerializeFormat) -> Result<Self> {
        match format {
            SerializeFormat::Json => {
                let serialized: SerializedArray<T> = serde_json::from_str(s).map_err(|e| {
                    NumRs2Error::DeserializationError(format!("JSON deserialization error: {}", e))
                })?;

                Ok(Array::from_vec(serialized.data).reshape(&serialized.shape))
            }
            SerializeFormat::Csv => {
                let mut reader = csv::Reader::from_reader(s.as_bytes());
                let mut data = Vec::new();

                for result in reader.deserialize() {
                    let record: Vec<T> = result.map_err(|e| {
                        NumRs2Error::DeserializationError(format!(
                            "CSV deserialization error: {}",
                            e
                        ))
                    })?;
                    data.extend(record);
                }

                // For CSV, we assume 1D array since we don't have shape information
                Ok(Array::from_vec(data))
            }
            SerializeFormat::Binary => Err(NumRs2Error::DeserializationError(
                "Binary deserialization from string not supported".to_string(),
            )),
            SerializeFormat::Npy => Err(NumRs2Error::DeserializationError(
                "NPY format deserialization from string not supported".to_string(),
            )),
            SerializeFormat::Npz => Err(NumRs2Error::DeserializationError(
                "NPZ format deserialization from string not supported".to_string(),
            )),
        }
    }

    /// Deserialize an array from a file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file
    /// * `format` - The format of the file
    ///
    /// # Returns
    ///
    /// A Result containing the deserialized array or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::io::SerializeFormat;
    /// use std::path::Path;
    ///
    /// // Uncomment to actually read from file:
    /// // let array = Array::<i32>::from_file(Path::new("array.json"), SerializeFormat::Json).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P, format: SerializeFormat) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);

        match format {
            SerializeFormat::Json => {
                let serialized: SerializedArray<T> =
                    serde_json::from_reader(reader).map_err(|e| {
                        NumRs2Error::DeserializationError(format!(
                            "JSON deserialization error: {}",
                            e
                        ))
                    })?;

                Ok(Array::from_vec(serialized.data).reshape(&serialized.shape))
            }
            SerializeFormat::Csv => {
                let mut csv_reader = csv::Reader::from_reader(reader);
                // Read all rows first to determine shape
                let mut all_rows: Vec<Vec<T>> = Vec::new();

                for result in csv_reader.deserialize() {
                    let record: Vec<T> = result.map_err(|e| {
                        NumRs2Error::DeserializationError(format!(
                            "CSV deserialization error: {}",
                            e
                        ))
                    })?;
                    all_rows.push(record);
                }

                if all_rows.is_empty() {
                    return Err(NumRs2Error::DeserializationError(
                        "CSV file contained no data".to_string(),
                    ));
                }

                // Check if all rows have the same length
                let row_length = all_rows[0].len();
                for (i, row) in all_rows.iter().enumerate().skip(1) {
                    if row.len() != row_length {
                        return Err(NumRs2Error::DeserializationError(
                            format!("CSV file has inconsistent row lengths: row 0 has length {}, row {} has length {}", 
                                    row_length, i, row.len())
                        ));
                    }
                }

                // Store the rows count before moving the data
                let rows_count = all_rows.len();

                // Flatten the rows into a single vector
                let mut data = Vec::with_capacity(rows_count * row_length);
                for row in all_rows {
                    data.extend(row);
                }

                // Create a 2D array with the appropriate shape
                Ok(Array::from_vec(data).reshape(&[rows_count, row_length]))
            }
            SerializeFormat::Binary => {
                let serialized: SerializedArray<T> =
                    bincode::deserialize_from(reader).map_err(|e| {
                        NumRs2Error::DeserializationError(format!(
                            "Binary deserialization error: {}",
                            e
                        ))
                    })?;

                Ok(Array::from_vec(serialized.data).reshape(&serialized.shape))
            }
            SerializeFormat::Npy | SerializeFormat::Npz => {
                // Delegate to the NPY/NPZ module for these formats
                npy_npz::deserialize_from_file(reader, format)
            }
        }
    }
}

// Conversion functions for standard Rust data structures

/// Convert a Vec to an Array
///
/// # Arguments
///
/// * `vec` - The vector to convert
/// * `shape` - Optional shape for the resulting array. If not provided, a 1D array is created.
///
/// # Returns
///
/// A new Array containing the vector's data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::io::vec_to_array;
///
/// let vec = vec![1, 2, 3, 4];
/// let array = vec_to_array(vec, Some(&[2, 2])).unwrap();
/// assert_eq!(array.shape(), vec![2, 2]);
/// ```
pub fn vec_to_array<T: Clone>(vec: Vec<T>, shape: Option<&[usize]>) -> Result<Array<T>> {
    let array = Array::from_vec(vec);
    match shape {
        Some(shape) => Ok(array.reshape(shape)),
        None => Ok(array),
    }
}

/// Convert a 2D Vec (`Vec<Vec<T>>`) to an Array
///
/// # Arguments
///
/// * `vec` - The 2D vector to convert
///
/// # Returns
///
/// A new 2D Array containing the vector's data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::io::vec2d_to_array;
///
/// let vec = vec![vec![1, 2], vec![3, 4]];
/// let array = vec2d_to_array(vec).unwrap();
/// assert_eq!(array.shape(), vec![2, 2]);
/// ```
pub fn vec2d_to_array<T: Clone>(vec: Vec<Vec<T>>) -> Result<Array<T>> {
    if vec.is_empty() {
        return Ok(Array::from_vec(Vec::new()));
    }

    let rows = vec.len();
    let cols = vec[0].len();

    // Check that all rows have the same length
    for (i, row) in vec.iter().enumerate() {
        if row.len() != cols {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Row 0 has length {}, but row {} has length {}",
                cols,
                i,
                row.len()
            )));
        }
    }

    // Flatten the 2D vector into a 1D vector
    let mut data = Vec::with_capacity(rows * cols);
    for row in vec {
        data.extend(row);
    }

    // Create a 2D array with the appropriate shape
    Ok(Array::from_vec(data).reshape(&[rows, cols]))
}

/// Convert an Array to a 2D Vec (`Vec<Vec<T>>`)
///
/// # Arguments
///
/// * `array` - The array to convert
///
/// # Returns
///
/// A 2D vector containing the array's data
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::io::array_to_vec2d;
///
/// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let vec = array_to_vec2d(&array).unwrap();
/// assert_eq!(vec, vec![vec![1, 2], vec![3, 4]]);
/// ```
pub fn array_to_vec2d<T: Clone>(array: &Array<T>) -> Result<Vec<Vec<T>>> {
    if array.ndim() != 2 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Expected 2D array, got {}D",
            array.ndim()
        )));
    }

    let shape = array.shape();
    let rows = shape[0];
    let cols = shape[1];
    let data = array.to_vec();

    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let idx = i * cols + j;
            row.push(data[idx].clone());
        }
        result.push(row);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_serialization() {
        let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let json = array.to_string(SerializeFormat::Json).unwrap();
        let expected = r#"{"shape":[2,2],"data":[1,2,3,4]}"#;
        assert_eq!(json, expected);

        let deserialized = Array::<i32>::from_string(&json, SerializeFormat::Json).unwrap();
        assert_eq!(deserialized.shape(), vec![2, 2]);
        assert_eq!(deserialized.to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_vec_to_array() {
        let vec = vec![1, 2, 3, 4];
        let array = vec_to_array(vec, Some(&[2, 2])).unwrap();
        assert_eq!(array.shape(), vec![2, 2]);
        assert_eq!(array.to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_vec2d_to_array() {
        let vec = vec![vec![1, 2], vec![3, 4]];
        let array = vec2d_to_array(vec).unwrap();
        assert_eq!(array.shape(), vec![2, 2]);
        assert_eq!(array.to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_array_to_vec2d() {
        let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let vec = array_to_vec2d(&array).unwrap();
        assert_eq!(vec, vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn test_csv_serialization() {
        let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
        let rows = array.to_row_vectors().unwrap();
        assert_eq!(rows, vec![vec![1, 2], vec![3, 4]]);
    }
}
