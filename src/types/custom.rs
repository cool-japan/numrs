//! Custom data types for NumRS
//!
//! This module provides functionality for creating custom data types,
//! allowing users to extend the type system with their own types.

use std::fmt;
use std::any::Any;
use std::marker::PhantomData;
use serde::{Serialize, Deserialize};

/// Trait for custom data types
pub trait TypeDescriptor: fmt::Debug + Clone + Send + Sync + 'static {
    /// Returns the name of the type
    fn name(&self) -> &str;
    
    /// Returns the size in bytes of this type
    fn size_in_bytes(&self) -> usize;
    
    /// Returns true if this is a numeric type
    fn is_numeric(&self) -> bool;
    
    /// Returns a boxed value of this type, initialized to zero
    fn default_value(&self) -> Box<dyn Any>;
    
    /// Converts a value to bytes
    fn to_bytes(&self, value: &dyn Any) -> Vec<u8>;
    
    /// Converts bytes to a value
    fn from_bytes(&self, bytes: &[u8]) -> Box<dyn Any>;
}

/// A custom data type for NumRS
#[derive(Clone, Serialize, Deserialize)]
pub struct CustomDType<T: 'static> {
    /// The name of the type
    name: String,
    /// The size in bytes of this type
    size: usize,
    /// Whether this is a numeric type
    numeric: bool,
    /// Phantom data to track the type parameter
    #[serde(skip)]
    phantom: PhantomData<T>,
}

impl<T: Clone + Default + Send + Sync + 'static> CustomDType<T> {
    /// Create a new custom data type
    pub fn new<S: Into<String>>(name: S, size: usize, numeric: bool) -> Self {
        Self {
            name: name.into(),
            size,
            numeric,
            phantom: PhantomData,
        }
    }
}

impl<T: fmt::Debug + Clone + Default + Send + Sync + 'static> TypeDescriptor for CustomDType<T> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn size_in_bytes(&self) -> usize {
        self.size
    }
    
    fn is_numeric(&self) -> bool {
        self.numeric
    }
    
    fn default_value(&self) -> Box<dyn Any> {
        Box::new(T::default())
    }
    
    fn to_bytes(&self, value: &dyn Any) -> Vec<u8> {
        if let Some(_val) = value.downcast_ref::<T>() {
            // This is a simplification - in a real implementation, you would
            // serialize the value to bytes properly
            vec![0; self.size] // Placeholder
        } else {
            vec![0; self.size] // Placeholder for error case
        }
    }
    
    fn from_bytes(&self, _bytes: &[u8]) -> Box<dyn Any> {
        // This is a simplification - in a real implementation, you would
        // deserialize the bytes to the appropriate type
        Box::new(T::default())
    }
}

impl<T: fmt::Debug + Clone + Default + Send + Sync + 'static> fmt::Debug for CustomDType<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomDType")
            .field("name", &self.name)
            .field("size", &self.size)
            .field("numeric", &self.numeric)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug, Clone, Default)]
    struct TestType {
        value: i32,
    }
    
    #[test]
    fn test_custom_dtype() {
        let dtype = CustomDType::<TestType>::new("TestType", 4, true);
        
        assert_eq!(dtype.name(), "TestType");
        assert_eq!(dtype.size_in_bytes(), 4);
        assert!(dtype.is_numeric());
        
        let default_value = dtype.default_value();
        let _: &TestType = default_value.downcast_ref().unwrap();
    }
}
