//! Custom data types for NumRS
//!
//! This module provides functionality for creating custom data types,
//! allowing users to extend the type system with their own types.

use crate::error::{NumRs2Error, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;
use std::marker::PhantomData;

/// Codec used by [`CustomDType`] to convert values of `T` to and from a byte
/// representation.
///
/// A blanket implementation is provided for any type that implements
/// [`Serialize`]/[`DeserializeOwned`] (encoded via `oxicode`, the COOLJAPAN
/// pure-Rust replacement for `bincode`), so most user types only need to
/// `#[derive(Serialize, Deserialize)]` to be usable as `CustomDType<T>`.
/// Types that cannot derive serde support may implement this trait by hand
/// to plug in a custom binary representation.
pub trait CustomDTypeCodec: fmt::Debug + Clone + Default + Send + Sync + 'static {
    /// Serializes `self` into a byte vector.
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserializes a value from a byte slice.
    ///
    /// Implementations must reject byte slices that are malformed or the
    /// wrong length by returning an error rather than silently falling back
    /// to a default value.
    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

impl<T> CustomDTypeCodec for T
where
    T: fmt::Debug + Clone + Default + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let config = oxicode::config::standard();
        oxicode::serde::encode_to_vec(self, config)
            .map_err(|e| NumRs2Error::SerializationError(e.to_string()))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let config = oxicode::config::standard();
        let (value, consumed) = oxicode::serde::decode_owned_from_slice(bytes, config)
            .map_err(|e| NumRs2Error::DeserializationError(e.to_string()))?;

        // `decode_owned_from_slice` happily decodes a valid prefix and
        // silently ignores trailing bytes. Requiring the whole slice be
        // consumed turns "wrong-size input" into an error instead of a
        // value quietly reconstructed from a truncated/garbled buffer.
        if consumed != bytes.len() {
            return Err(NumRs2Error::DeserializationError(format!(
                "expected exactly {} byte(s) to decode `{}`, but only {} were consumed",
                bytes.len(),
                std::any::type_name::<T>(),
                consumed
            )));
        }

        Ok(value)
    }
}

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
    fn to_bytes(&self, value: &dyn Any) -> Result<Vec<u8>>;

    /// Converts bytes to a value
    fn parse_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Any>>;
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

impl<T: CustomDTypeCodec> CustomDType<T> {
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

impl<T: CustomDTypeCodec> TypeDescriptor for CustomDType<T> {
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

    fn to_bytes(&self, value: &dyn Any) -> Result<Vec<u8>> {
        let value = value.downcast_ref::<T>().ok_or_else(|| {
            NumRs2Error::TypeCastError(format!(
                "value is not an instance of `{}`",
                std::any::type_name::<T>()
            ))
        })?;
        value.to_bytes()
    }

    fn parse_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Any>> {
        let value = T::from_bytes(bytes)?;
        Ok(Box::new(value))
    }
}

impl<T: CustomDTypeCodec> fmt::Debug for CustomDType<T> {
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

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
        let _: &TestType = default_value
            .downcast_ref()
            .expect("default_value should downcast to TestType");
    }

    #[test]
    fn test_round_trip_via_codec_trait() {
        let original = TestType { value: 42 };
        let bytes = original.to_bytes().expect("encode should succeed");
        let decoded = TestType::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_round_trip_via_type_descriptor() {
        let dtype = CustomDType::<TestType>::new("TestType", 4, true);
        let original: Box<dyn Any> = Box::new(TestType { value: -7 });

        let bytes = dtype
            .to_bytes(original.as_ref())
            .expect("to_bytes should succeed for a matching value");
        let parsed = dtype
            .parse_bytes(&bytes)
            .expect("parse_bytes should succeed for well-formed bytes");
        let parsed: &TestType = parsed
            .downcast_ref()
            .expect("parsed value should downcast to TestType");

        assert_eq!(parsed, &TestType { value: -7 });
    }

    #[test]
    fn test_to_bytes_rejects_wrong_type() {
        let dtype = CustomDType::<TestType>::new("TestType", 4, true);
        let wrong: Box<dyn Any> = Box::new(123i32);

        let result = dtype.to_bytes(wrong.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_bytes_rejects_wrong_size() {
        let dtype = CustomDType::<TestType>::new("TestType", 4, true);
        let original = TestType { value: 1234 };
        let mut bytes = original.to_bytes().expect("encode should succeed");

        // Corrupt the byte length so the decoder cannot possibly reconstruct
        // a valid value: truncating leaves a length-prefixed/varint payload
        // short, which must surface as an error, not `TestType::default()`.
        bytes.truncate(bytes.len().saturating_sub(1));

        let result = dtype.parse_bytes(&bytes);
        assert!(result.is_err(), "truncated bytes must not decode silently");

        // Extra trailing bytes must also be rejected (not silently ignored).
        let mut original_bytes = original.to_bytes().expect("encode should succeed");
        original_bytes.push(0xFF);
        let result = dtype.parse_bytes(&original_bytes);
        assert!(
            result.is_err(),
            "trailing garbage bytes must not decode silently"
        );
    }

    #[test]
    fn test_empty_bytes_error_not_default() {
        let dtype = CustomDType::<TestType>::new("TestType", 4, true);
        let result = dtype.parse_bytes(&[]);
        assert!(result.is_err());
    }
}
