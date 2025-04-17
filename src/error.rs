use thiserror::Error;
use std::io;

/// NumRS2 error types
#[derive(Error, Debug)]
pub enum NumRs2Error {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Value error: {0}")]
    ValueError(String),
    
    #[error("Index error: {0}")]
    IndexError(String),
    
    #[error("BLAS error: code {0}")]
    BlasError(i32),
    
    #[error("LAPACK error: {0}")]
    LapackError(String),
    
    #[error("Conversion error: {0}")]
    ConversionError(String),
    
    #[error("Type cast error: {0}")]
    TypeCastError(String),
    
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("I/O error: {0}")]
    IOError(String),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Result type for NumRS2 operations
pub type Result<T> = std::result::Result<T, NumRs2Error>;

/// Implement From<io::Error> for NumRs2Error
impl From<io::Error> for NumRs2Error {
    fn from(err: io::Error) -> Self {
        NumRs2Error::IOError(err.to_string())
    }
}

/// Implement From<Box<bincode::ErrorKind>> for NumRs2Error
impl From<Box<bincode::ErrorKind>> for NumRs2Error {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        NumRs2Error::DeserializationError(err.to_string())
    }
}