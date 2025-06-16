//! Structured arrays for heterogeneous data
//!
//! This module provides data types for working with heterogeneous data,
//! similar to NumPy's structured arrays and record arrays.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Represents a data type in the structured array system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// Boolean type
    Bool,
    /// 8-bit integer
    Int8,
    /// 16-bit integer
    Int16,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// String with a fixed length (bytes)
    String(usize),
    /// Complex number with 32-bit components
    Complex32,
    /// Complex number with 64-bit components
    Complex64,
    /// A structure with named fields
    Struct(Vec<Field>),
}

impl DType {
    /// Returns the size in bytes of this data type
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::UInt8 => 1,
            DType::UInt16 => 2,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::String(len) => *len,
            DType::Complex32 => 8,  // 2 * Float32
            DType::Complex64 => 16, // 2 * Float64
            DType::Struct(fields) => fields.iter().map(|f| f.dtype.size_in_bytes()).sum(),
        }
    }

    /// Returns true if this is a numeric data type
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DType::Bool
                | DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
                | DType::Float32
                | DType::Float64
                | DType::Complex32
                | DType::Complex64
        )
    }

    /// Returns true if this is a floating point data type
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            DType::Float32 | DType::Float64 | DType::Complex32 | DType::Complex64
        )
    }

    /// Returns true if this is a complex data type
    pub fn is_complex(&self) -> bool {
        matches!(self, DType::Complex32 | DType::Complex64)
    }

    /// Returns true if this is a string data type
    pub fn is_string(&self) -> bool {
        matches!(self, DType::String(_))
    }

    /// Returns true if this is a struct data type
    pub fn is_struct(&self) -> bool {
        matches!(self, DType::Struct(_))
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int8 => write!(f, "int8"),
            DType::Int16 => write!(f, "int16"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::UInt8 => write!(f, "uint8"),
            DType::UInt16 => write!(f, "uint16"),
            DType::UInt32 => write!(f, "uint32"),
            DType::UInt64 => write!(f, "uint64"),
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
            DType::String(len) => write!(f, "S{}", len),
            DType::Complex32 => write!(f, "complex64"),
            DType::Complex64 => write!(f, "complex128"),
            DType::Struct(fields) => {
                write!(f, "struct{{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", field.name, field.dtype)?;
                }
                write!(f, "}}")?;
                Ok(())
            }
        }
    }
}

/// Represents a field in a structured data type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field {
    /// The name of the field
    pub name: String,
    /// The data type of the field
    pub dtype: DType,
}

impl Field {
    /// Create a new field with the given name and data type
    pub fn new<S: Into<String>>(name: S, dtype: DType) -> Self {
        Self {
            name: name.into(),
            dtype,
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.dtype)
    }
}

/// A structured array with heterogeneous data types
#[derive(Debug, Clone)]
pub struct StructuredArray {
    /// The shape of the array
    shape: Vec<usize>,
    /// The data type descriptor
    dtype: DType,
    /// The raw data as bytes
    data: Vec<u8>,
}

impl StructuredArray {
    /// Create a new structured array with the given shape and data type
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        let size = shape.iter().product::<usize>();
        let byte_size = size * dtype.size_in_bytes();
        let data = vec![0; byte_size];

        Self {
            shape: shape.to_vec(),
            dtype,
            data,
        }
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the array
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Get the raw data of the array
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the size (total number of elements) of the array
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions of the array
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get a reference to a field as a standard NumRS Array
    pub fn field<T: Clone + Default + 'static>(&self, field_name: &str) -> Result<Array<T>> {
        if let DType::Struct(fields) = &self.dtype {
            // Find the field
            let field = fields
                .iter()
                .find(|f| f.name == field_name)
                .ok_or_else(|| {
                    NumRs2Error::IndexError(format!("Field '{}' not found", field_name))
                })?;

            // Calculate the offset and size
            let mut offset = 0;
            for f in fields.iter() {
                if f.name == field_name {
                    break;
                }
                offset += f.dtype.size_in_bytes();
            }

            // Extract the field data
            let field_size = field.dtype.size_in_bytes();
            let element_size = self.dtype.size_in_bytes();
            let mut field_data = Vec::with_capacity(self.size());

            // For each element in the array, extract the field
            for i in 0..self.size() {
                let start = i * element_size + offset;
                let end = start + field_size;
                let bytes = &self.data[start..end];

                // Convert bytes to the target type using the new implementation
                let value = bytes_to_value::<T>(bytes, &field.dtype)?;
                field_data.push(value);
            }

            // Create a NumRS Array
            let arr = Array::from_vec(field_data).reshape(&self.shape);
            Ok(arr)
        } else {
            Err(NumRs2Error::ValueError(
                "Not a structured array".to_string(),
            ))
        }
    }

    /// Set a field value at the given index
    pub fn set_field<T: Clone + 'static>(
        &mut self,
        index: &[usize],
        field_name: &str,
        value: T,
    ) -> Result<()> {
        if let DType::Struct(fields) = &self.dtype {
            // Check if the index is valid
            if index.len() != self.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Expected {} dimensions, got {}",
                    self.ndim(),
                    index.len()
                )));
            }
            for (i, &idx) in index.iter().enumerate() {
                if idx >= self.shape[i] {
                    return Err(NumRs2Error::IndexError(format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        idx, i, self.shape[i]
                    )));
                }
            }

            // Find the field
            let field = fields
                .iter()
                .find(|f| f.name == field_name)
                .ok_or_else(|| {
                    NumRs2Error::IndexError(format!("Field '{}' not found", field_name))
                })?;

            // Calculate the offset and size
            let mut offset = 0;
            for f in fields.iter() {
                if f.name == field_name {
                    break;
                }
                offset += f.dtype.size_in_bytes();
            }

            // Calculate the flat index
            let mut flat_index = 0;
            let mut stride = 1;
            for i in (0..self.ndim()).rev() {
                flat_index += index[i] * stride;
                stride *= self.shape[i];
            }

            // Calculate the byte position
            let element_size = self.dtype.size_in_bytes();
            let start = flat_index * element_size + offset;
            let end = start + field.dtype.size_in_bytes();

            // Convert value to bytes and store using the new implementation
            let bytes = value_to_bytes(&value, &field.dtype)?;
            if bytes.len() != field.dtype.size_in_bytes() {
                return Err(NumRs2Error::ValueError(format!(
                    "Expected {} bytes, got {}",
                    field.dtype.size_in_bytes(),
                    bytes.len()
                )));
            }
            self.data[start..end].copy_from_slice(&bytes);

            Ok(())
        } else {
            Err(NumRs2Error::ValueError(
                "Not a structured array".to_string(),
            ))
        }
    }

    /// Create a structured array from a set of NumRS Arrays with the same shape
    pub fn from_arrays<T: Clone + Default + 'static>(
        arrays: &HashMap<String, Array<T>>,
        shape: &[usize],
    ) -> Result<Self> {
        // Check that all arrays have the same shape
        for (name, arr) in arrays.iter() {
            if arr.shape() != shape {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Array '{}' has shape {:?}, expected {:?}",
                    name,
                    arr.shape(),
                    shape
                )));
            }
        }

        // Create fields for the dtype
        let fields = arrays
            .keys()
            .map(|name| {
                Field::new(name.clone(), DType::Float64) // Assuming T is f64 for simplicity
            })
            .collect();

        let dtype = DType::Struct(fields);
        let mut result = Self::new(shape, dtype);

        // Fill in the data
        let size = shape.iter().product::<usize>();
        for i in 0..size {
            let index = flat_to_index(i, shape);
            for (name, _arr) in arrays.iter() {
                // This is a simplification - we're assuming T can be converted to f64
                // Placeholder - in real implementation we would get the value
                let value = T::clone(&T::default());
                result.set_field(&index, name, value)?;
            }
        }

        Ok(result)
    }
}

/// A record array is a structured array where fields can be accessed by name
#[derive(Debug, Clone)]
pub struct RecordArray {
    /// The underlying structured array
    array: StructuredArray,
    /// Cache of field arrays
    field_cache: HashMap<String, Array<f64>>, // Simplified to only support f64
}

impl RecordArray {
    /// Create a new record array with the given shape and fields
    pub fn new(shape: &[usize], fields: Vec<Field>) -> Self {
        let dtype = DType::Struct(fields.clone());
        let array = StructuredArray::new(shape, dtype);
        
        // Initialize field cache with empty arrays for each field
        let mut field_cache = HashMap::new();
        for field in &fields {
            // Create an array filled with zeros for each field
            let field_array = Array::zeros(shape);
            field_cache.insert(field.name.clone(), field_array);
        }

        Self {
            array,
            field_cache,
        }
    }

    /// Create a record array from a set of NumRS Arrays with the same shape
    pub fn from_arrays(arrays: &HashMap<String, Array<f64>>, shape: &[usize]) -> Result<Self> {
        let array = StructuredArray::from_arrays(arrays, shape)?;
        let mut field_cache = HashMap::new();

        // Copy the arrays to the cache
        for (name, arr) in arrays.iter() {
            field_cache.insert(name.clone(), arr.clone());
        }

        Ok(Self { array, field_cache })
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Get the data type of the array
    pub fn dtype(&self) -> &DType {
        self.array.dtype()
    }

    /// Get the size (total number of elements) of the array
    pub fn size(&self) -> usize {
        self.array.size()
    }

    /// Get the number of dimensions of the array
    pub fn ndim(&self) -> usize {
        self.array.ndim()
    }

    /// Get a field by name
    pub fn field(&self, field_name: &str) -> Result<&Array<f64>> {
        if self.field_cache.contains_key(field_name) {
            Ok(&self.field_cache[field_name])
        } else {
            Err(NumRs2Error::IndexError(format!(
                "Field '{}' not found",
                field_name
            )))
        }
    }

    /// Get a mutable reference to a field by name
    pub fn field_mut(&mut self, field_name: &str) -> Result<&mut Array<f64>> {
        if self.field_cache.contains_key(field_name) {
            Ok(self.field_cache.get_mut(field_name).unwrap())
        } else {
            Err(NumRs2Error::IndexError(format!(
                "Field '{}' not found",
                field_name
            )))
        }
    }

    /// Set a field value at the given index
    pub fn set_field(&mut self, index: &[usize], field_name: &str, value: f64) -> Result<()> {
        // Update the cache if it exists
        if let Some(arr) = self.field_cache.get_mut(field_name) {
            arr.set(index, value)?;
        }

        // Update the underlying structured array
        self.array.set_field(index, field_name, value)
    }

    /// Add a new field to the record array
    pub fn add_field(&mut self, field_name: &str, data: Array<f64>) -> Result<()> {
        // Check if the field already exists
        if self.field_cache.contains_key(field_name) {
            return Err(NumRs2Error::ValueError(format!(
                "Field '{}' already exists",
                field_name
            )));
        }

        // Check if the shape matches
        if data.shape() != self.array.shape() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Array has shape {:?}, expected {:?}",
                data.shape(),
                self.array.shape()
            )));
        }

        // Add the field to the cache
        self.field_cache
            .insert(field_name.to_string(), data.clone());

        // Create new fields list with the added field
        let mut new_fields = Vec::new();
        if let DType::Struct(ref fields) = &self.array.dtype {
            new_fields.extend(fields.clone());
        }
        new_fields.push(Field::new(field_name, DType::Float64));

        // Create a new structured array with the updated fields
        let new_dtype = DType::Struct(new_fields);
        let mut new_array = StructuredArray::new(self.array.shape(), new_dtype);

        // Copy existing data to the new array
        for existing_field_name in self.field_cache.keys() {
            if existing_field_name != field_name {
                let size = self.array.size();
                for i in 0..size {
                    let index = flat_to_index(i, self.array.shape());
                    // Get value from cache instead of structured array to avoid issues
                    if let Some(field_array) = self.field_cache.get(existing_field_name) {
                        // Use dynamic indexing for multi-dimensional arrays
                        let value = match index.len() {
                            1 => field_array.array()[[index[0]]],
                            2 => field_array.array()[[index[0], index[1]]],
                            3 => field_array.array()[[index[0], index[1], index[2]]],
                            _ => return Err(NumRs2Error::NotImplemented("More than 3 dimensions not supported in add_field".to_string())),
                        };
                        new_array.set_field(&index, existing_field_name, value)?;
                    }
                }
            }
        }

        // Add the new field data
        let size = self.array.size();
        for i in 0..size {
            let index = flat_to_index(i, self.array.shape());
            // Use dynamic indexing for multi-dimensional arrays
            let value = match index.len() {
                1 => data.array()[[index[0]]],
                2 => data.array()[[index[0], index[1]]],
                3 => data.array()[[index[0], index[1], index[2]]],
                _ => return Err(NumRs2Error::NotImplemented("More than 3 dimensions not supported in add_field".to_string())),
            };
            new_array.set_field(&index, field_name, value)?;
        }

        // Replace the array
        self.array = new_array;

        Ok(())
    }

    /// Remove a field from the record array
    pub fn remove_field(&mut self, field_name: &str) -> Result<Array<f64>> {
        // Check if the field exists
        if !self.field_cache.contains_key(field_name) {
            return Err(NumRs2Error::IndexError(format!(
                "Field '{}' not found",
                field_name
            )));
        }

        // Remove the field from the cache
        let arr = self.field_cache.remove(field_name).unwrap();

        // Update the dtype of the structured array
        if let DType::Struct(ref mut fields) = &mut self.array.dtype {
            fields.retain(|f| f.name != field_name);
        }

        Ok(arr)
    }

    /// List all field names
    pub fn field_names(&self) -> Vec<String> {
        self.field_cache.keys().cloned().collect()
    }
}

impl fmt::Display for StructuredArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StructuredArray(shape={:?}, dtype={})",
            self.shape, self.dtype
        )
    }
}

impl fmt::Display for RecordArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RecordArray(shape={:?}, fields={})",
            self.shape(),
            self.field_names().join(", ")
        )
    }
}

/// Convert a slice of bytes to a value of type T
///
/// This implementation handles the conversion based on the size of T and assumes little-endian.
fn bytes_to_value<T: Clone + Default + 'static>(bytes: &[u8], dtype: &DType) -> Result<T> {
    use std::any::TypeId;
    
    let type_id = TypeId::of::<T>();
    
    // Handle different data types
    match dtype {
        DType::Bool => {
            if type_id == TypeId::of::<bool>() {
                let value = bytes[0] != 0;
                // SAFETY: We've verified T is bool
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Bool".to_string()))
            }
        }
        DType::Int8 => {
            if type_id == TypeId::of::<i8>() {
                let value = i8::from_le_bytes([bytes[0]]);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int8".to_string()))
            }
        }
        DType::Int16 => {
            if type_id == TypeId::of::<i16>() {
                let mut buf = [0u8; 2];
                buf.copy_from_slice(&bytes[0..2]);
                let value = i16::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int16".to_string()))
            }
        }
        DType::Int32 => {
            if type_id == TypeId::of::<i32>() {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&bytes[0..4]);
                let value = i32::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int32".to_string()))
            }
        }
        DType::Int64 => {
            if type_id == TypeId::of::<i64>() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[0..8]);
                let value = i64::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int64".to_string()))
            }
        }
        DType::UInt8 => {
            if type_id == TypeId::of::<u8>() {
                let value = bytes[0];
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt8".to_string()))
            }
        }
        DType::UInt16 => {
            if type_id == TypeId::of::<u16>() {
                let mut buf = [0u8; 2];
                buf.copy_from_slice(&bytes[0..2]);
                let value = u16::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt16".to_string()))
            }
        }
        DType::UInt32 => {
            if type_id == TypeId::of::<u32>() {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&bytes[0..4]);
                let value = u32::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt32".to_string()))
            }
        }
        DType::UInt64 => {
            if type_id == TypeId::of::<u64>() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[0..8]);
                let value = u64::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt64".to_string()))
            }
        }
        DType::Float32 => {
            if type_id == TypeId::of::<f32>() {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&bytes[0..4]);
                let value = f32::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Float32".to_string()))
            }
        }
        DType::Float64 => {
            if type_id == TypeId::of::<f64>() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[0..8]);
                let value = f64::from_le_bytes(buf);
                Ok(unsafe { std::mem::transmute_copy(&value) })
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Float64".to_string()))
            }
        }
        DType::String(_) => {
            if type_id == TypeId::of::<String>() {
                // Find the null terminator or use the full length
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                let value = String::from_utf8_lossy(&bytes[0..end]).to_string();
                // SAFETY: We've verified T is String, using ptr casting instead of transmute_copy
                let ptr = &value as *const String as *const T;
                let result = unsafe { std::ptr::read(ptr) };
                std::mem::forget(value); // Prevent double free for non-Copy types
                Ok(result)
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for String".to_string()))
            }
        }
        DType::Complex32 => {
            if type_id == TypeId::of::<Complex<f32>>() {
                let mut real_buf = [0u8; 4];
                let mut imag_buf = [0u8; 4];
                real_buf.copy_from_slice(&bytes[0..4]);
                imag_buf.copy_from_slice(&bytes[4..8]);
                let real = f32::from_le_bytes(real_buf);
                let imag = f32::from_le_bytes(imag_buf);
                let value = Complex::new(real, imag);
                // SAFETY: We've verified T is Complex<f32>, using ptr casting instead of transmute_copy
                let ptr = &value as *const Complex<f32> as *const T;
                let result = unsafe { std::ptr::read(ptr) };
                let _ = value; // Prevent double free
                Ok(result)
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Complex32".to_string()))
            }
        }
        DType::Complex64 => {
            if type_id == TypeId::of::<Complex<f64>>() {
                let mut real_buf = [0u8; 8];
                let mut imag_buf = [0u8; 8];
                real_buf.copy_from_slice(&bytes[0..8]);
                imag_buf.copy_from_slice(&bytes[8..16]);
                let real = f64::from_le_bytes(real_buf);
                let imag = f64::from_le_bytes(imag_buf);
                let value = Complex::new(real, imag);
                // SAFETY: We've verified T is Complex<f64>, using ptr casting instead of transmute_copy
                let ptr = &value as *const Complex<f64> as *const T;
                let result = unsafe { std::ptr::read(ptr) };
                let _ = value; // Prevent double free
                Ok(result)
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Complex64".to_string()))
            }
        }
        DType::Struct(_) => {
            Err(NumRs2Error::ValueError("Cannot convert struct to single value".to_string()))
        }
    }
}

/// Convert a value of type T to a slice of bytes
///
/// This implementation handles the conversion based on the type and assumes little-endian.
fn value_to_bytes<T: Clone + 'static>(value: &T, dtype: &DType) -> Result<Vec<u8>> {
    use std::any::TypeId;
    
    let type_id = TypeId::of::<T>();
    
    // Handle different data types
    match dtype {
        DType::Bool => {
            if type_id == TypeId::of::<bool>() {
                let bool_value: &bool = unsafe { std::mem::transmute(value) };
                Ok(vec![if *bool_value { 1u8 } else { 0u8 }])
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Bool".to_string()))
            }
        }
        DType::Int8 => {
            if type_id == TypeId::of::<i8>() {
                let int_value: &i8 = unsafe { std::mem::transmute(value) };
                Ok(int_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int8".to_string()))
            }
        }
        DType::Int16 => {
            if type_id == TypeId::of::<i16>() {
                let int_value: &i16 = unsafe { std::mem::transmute(value) };
                Ok(int_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int16".to_string()))
            }
        }
        DType::Int32 => {
            if type_id == TypeId::of::<i32>() {
                let int_value: &i32 = unsafe { std::mem::transmute(value) };
                Ok(int_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int32".to_string()))
            }
        }
        DType::Int64 => {
            if type_id == TypeId::of::<i64>() {
                let int_value: &i64 = unsafe { std::mem::transmute(value) };
                Ok(int_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Int64".to_string()))
            }
        }
        DType::UInt8 => {
            if type_id == TypeId::of::<u8>() {
                let uint_value: &u8 = unsafe { std::mem::transmute(value) };
                Ok(vec![*uint_value])
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt8".to_string()))
            }
        }
        DType::UInt16 => {
            if type_id == TypeId::of::<u16>() {
                let uint_value: &u16 = unsafe { std::mem::transmute(value) };
                Ok(uint_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt16".to_string()))
            }
        }
        DType::UInt32 => {
            if type_id == TypeId::of::<u32>() {
                let uint_value: &u32 = unsafe { std::mem::transmute(value) };
                Ok(uint_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt32".to_string()))
            }
        }
        DType::UInt64 => {
            if type_id == TypeId::of::<u64>() {
                let uint_value: &u64 = unsafe { std::mem::transmute(value) };
                Ok(uint_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for UInt64".to_string()))
            }
        }
        DType::Float32 => {
            if type_id == TypeId::of::<f32>() {
                let float_value: &f32 = unsafe { std::mem::transmute(value) };
                Ok(float_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Float32".to_string()))
            }
        }
        DType::Float64 => {
            if type_id == TypeId::of::<f64>() {
                let float_value: &f64 = unsafe { std::mem::transmute(value) };
                Ok(float_value.to_le_bytes().to_vec())
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Float64".to_string()))
            }
        }
        DType::String(max_len) => {
            if type_id == TypeId::of::<String>() {
                let string_value: &String = unsafe { std::mem::transmute(value) };
                let mut bytes = string_value.as_bytes().to_vec();
                
                // Pad with zeros or truncate to the specified length
                match bytes.len().cmp(max_len) {
                    std::cmp::Ordering::Less => bytes.resize(*max_len, 0),
                    std::cmp::Ordering::Greater => bytes.truncate(*max_len),
                    std::cmp::Ordering::Equal => {},
                }
                
                Ok(bytes)
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for String".to_string()))
            }
        }
        DType::Complex32 => {
            if type_id == TypeId::of::<Complex<f32>>() {
                let complex_value: &Complex<f32> = unsafe { std::mem::transmute(value) };
                let mut bytes = Vec::with_capacity(8);
                bytes.extend_from_slice(&complex_value.re.to_le_bytes());
                bytes.extend_from_slice(&complex_value.im.to_le_bytes());
                Ok(bytes)
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Complex32".to_string()))
            }
        }
        DType::Complex64 => {
            if type_id == TypeId::of::<Complex<f64>>() {
                let complex_value: &Complex<f64> = unsafe { std::mem::transmute(value) };
                let mut bytes = Vec::with_capacity(16);
                bytes.extend_from_slice(&complex_value.re.to_le_bytes());
                bytes.extend_from_slice(&complex_value.im.to_le_bytes());
                Ok(bytes)
            } else {
                Err(NumRs2Error::TypeCastError("Type mismatch for Complex64".to_string()))
            }
        }
        DType::Struct(_) => {
            Err(NumRs2Error::ValueError("Cannot convert single value to struct".to_string()))
        }
    }
}

/// Convert a flat index to a multi-dimensional index
fn flat_to_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    let mut remainder = flat_index;

    for i in (0..shape.len()).rev() {
        index[i] = remainder % shape[i];
        remainder /= shape[i];
    }

    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Bool.size_in_bytes(), 1);
        assert_eq!(DType::Int32.size_in_bytes(), 4);
        assert_eq!(DType::Float64.size_in_bytes(), 8);
        assert_eq!(DType::String(10).size_in_bytes(), 10);

        let fields = vec![
            Field::new("a", DType::Int32),
            Field::new("b", DType::Float64),
        ];
        let struct_type = DType::Struct(fields);
        assert_eq!(struct_type.size_in_bytes(), 12); // 4 + 8
    }

    #[test]
    fn test_dtype_properties() {
        assert!(DType::Int32.is_numeric());
        assert!(DType::Float64.is_floating_point());
        assert!(DType::Complex64.is_complex());
        assert!(DType::String(10).is_string());

        let fields = vec![
            Field::new("a", DType::Int32),
            Field::new("b", DType::Float64),
        ];
        let struct_type = DType::Struct(fields);
        assert!(struct_type.is_struct());
    }

    #[test]
    fn test_field_creation() {
        let field = Field::new("test", DType::Int32);
        assert_eq!(field.name, "test");
        assert_eq!(field.dtype, DType::Int32);
    }

    // More tests would be added for StructuredArray and RecordArray functionality
}
