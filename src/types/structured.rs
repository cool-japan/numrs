//! Structured arrays for heterogeneous data
//!
//! This module provides data types for working with heterogeneous data,
//! similar to NumPy's structured arrays and record arrays.

use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::array::Array;
use crate::error::{NumRs2Error, Result};

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
            DType::Complex32 => 8, // 2 * Float32
            DType::Complex64 => 16, // 2 * Float64
            DType::Struct(fields) => {
                fields.iter().map(|f| f.dtype.size_in_bytes()).sum()
            }
        }
    }
    
    /// Returns true if this is a numeric data type
    pub fn is_numeric(&self) -> bool {
        match self {
            DType::Bool | 
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 |
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 |
            DType::Float32 | DType::Float64 |
            DType::Complex32 | DType::Complex64 => true,
            _ => false,
        }
    }
    
    /// Returns true if this is a floating point data type
    pub fn is_floating_point(&self) -> bool {
        match self {
            DType::Float32 | DType::Float64 |
            DType::Complex32 | DType::Complex64 => true,
            _ => false,
        }
    }
    
    /// Returns true if this is a complex data type
    pub fn is_complex(&self) -> bool {
        match self {
            DType::Complex32 | DType::Complex64 => true,
            _ => false,
        }
    }
    
    /// Returns true if this is a string data type
    pub fn is_string(&self) -> bool {
        match self {
            DType::String(_) => true,
            _ => false,
        }
    }
    
    /// Returns true if this is a struct data type
    pub fn is_struct(&self) -> bool {
        match self {
            DType::Struct(_) => true,
            _ => false,
        }
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
    pub fn field<T: Clone>(&self, field_name: &str) -> Result<Array<T>> {
        if let DType::Struct(fields) = &self.dtype {
            // Find the field
            let field = fields.iter().find(|f| f.name == field_name)
                .ok_or_else(|| NumRs2Error::IndexError(format!("Field '{}' not found", field_name)))?;
            
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
                
                // Convert bytes to the target type
                // This is a simplification - in practice, you would need to handle
                // different types and endianness correctly
                let value = bytes_to_value::<T>(bytes);
                field_data.push(value);
            }
            
            // Create a NumRS Array
            let arr = Array::from_vec(field_data).reshape(&self.shape);
            Ok(arr)
        } else {
            Err(NumRs2Error::ValueError("Not a structured array".to_string()))
        }
    }
    
    /// Set a field value at the given index
    pub fn set_field<T: Clone>(&mut self, index: &[usize], field_name: &str, value: T) -> Result<()> {
        if let DType::Struct(fields) = &self.dtype {
            // Check if the index is valid
            if index.len() != self.ndim() {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Expected {} dimensions, got {}", self.ndim(), index.len())
                ));
            }
            for (i, &idx) in index.iter().enumerate() {
                if idx >= self.shape[i] {
                    return Err(NumRs2Error::IndexError(
                        format!("Index {} out of bounds for dimension {} with size {}", 
                                idx, i, self.shape[i])
                    ));
                }
            }
            
            // Find the field
            let field = fields.iter().find(|f| f.name == field_name)
                .ok_or_else(|| NumRs2Error::IndexError(format!("Field '{}' not found", field_name)))?;
            
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
            
            // Convert value to bytes and store
            // This is a simplification - in practice, you would need to handle
            // different types and endianness correctly
            let bytes = value_to_bytes(&value);
            self.data[start..end].copy_from_slice(&bytes);
            
            Ok(())
        } else {
            Err(NumRs2Error::ValueError("Not a structured array".to_string()))
        }
    }
    
    /// Create a structured array from a set of NumRS Arrays with the same shape
    pub fn from_arrays<T: Clone + Default>(arrays: &HashMap<String, Array<T>>, shape: &[usize]) -> Result<Self> {
        // Check that all arrays have the same shape
        for (name, arr) in arrays.iter() {
            if arr.shape() != shape {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Array '{}' has shape {:?}, expected {:?}", name, arr.shape(), shape)
                ));
            }
        }
        
        // Create fields for the dtype
        let fields = arrays.keys().map(|name| {
            Field::new(name.clone(), DType::Float64) // Assuming T is f64 for simplicity
        }).collect();
        
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
        let dtype = DType::Struct(fields);
        let array = StructuredArray::new(shape, dtype);
        
        Self {
            array,
            field_cache: HashMap::new(),
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
        
        Ok(Self {
            array,
            field_cache,
        })
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
            Err(NumRs2Error::IndexError(format!("Field '{}' not found", field_name)))
        }
    }
    
    /// Get a mutable reference to a field by name
    pub fn field_mut(&mut self, field_name: &str) -> Result<&mut Array<f64>> {
        if self.field_cache.contains_key(field_name) {
            Ok(self.field_cache.get_mut(field_name).unwrap())
        } else {
            Err(NumRs2Error::IndexError(format!("Field '{}' not found", field_name)))
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
            return Err(NumRs2Error::ValueError(format!("Field '{}' already exists", field_name)));
        }
        
        // Check if the shape matches
        if data.shape() != self.array.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Array has shape {:?}, expected {:?}", data.shape(), self.array.shape())
            ));
        }
        
        // Add the field to the cache
        self.field_cache.insert(field_name.to_string(), data.clone());
        
        // Update the dtype of the structured array
        if let DType::Struct(ref mut fields) = &mut self.array.dtype {
            fields.push(Field::new(field_name, DType::Float64));
        }
        
        // Fill the structured array with the data
        let size = self.array.size();
        for i in 0..size {
            let index = flat_to_index(i, self.array.shape());
            let value = data.get(&index)?;
            self.array.set_field(&index, field_name, value)?;
        }
        
        Ok(())
    }
    
    /// Remove a field from the record array
    pub fn remove_field(&mut self, field_name: &str) -> Result<Array<f64>> {
        // Check if the field exists
        if !self.field_cache.contains_key(field_name) {
            return Err(NumRs2Error::IndexError(format!("Field '{}' not found", field_name)));
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
        write!(f, "StructuredArray(shape={:?}, dtype={})", self.shape, self.dtype)
    }
}

impl fmt::Display for RecordArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RecordArray(shape={:?}, fields={})", self.shape(), self.field_names().join(", "))
    }
}

/// Convert a slice of bytes to a value of type T
/// 
/// This is a simplified implementation for demonstration purposes.
/// In practice, you would need to handle different types and endianness correctly.
fn bytes_to_value<T: Clone>(_bytes: &[u8]) -> T {
    // This is just a placeholder - in a real implementation, you would
    // convert the bytes to the appropriate type
    unimplemented!("bytes_to_value is not implemented")
}

/// Convert a value of type T to a slice of bytes
/// 
/// This is a simplified implementation for demonstration purposes.
/// In practice, you would need to handle different types and endianness correctly.
fn value_to_bytes<T: Clone>(_value: &T) -> Vec<u8> {
    // This is just a placeholder - in a real implementation, you would
    // convert the value to bytes
    unimplemented!("value_to_bytes is not implemented")
}

/// Convert a flat index to a multi-dimensional index
fn flat_to_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    let mut remainder = flat_index;
    
    for i in (0..shape.len()).rev() {
        let divisor = if i == 0 { 1 } else { shape[i] };
        index[i] = remainder % divisor;
        remainder /= divisor;
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
