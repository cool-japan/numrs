//! Legacy error system for backward compatibility
//!
//! This maintains the original NumRS2Error enum structure to ensure
//! existing code continues to work without modification.

use super::{ErrorCategory, ErrorContext, ErrorLocation, ErrorSeverity, OperationContext};
use std::io;
use thiserror::Error;

/// NumRS2 error types (legacy structure for backward compatibility)
#[derive(Error, Debug, Clone)]
pub enum NumRs2Error {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Value error: {0}")]
    ValueError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

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

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    #[error("Distributed computing error: {0}")]
    DistributedComputing(String),

    #[error("Control systems error: {0}")]
    ControlError(String),

    // New hierarchical variants that integrate with the legacy system
    #[error("{0}")]
    Core(#[from] super::hierarchical::CoreError),

    #[error("{0}")]
    Computation(#[from] super::hierarchical::ComputationError),

    #[error("{0}")]
    Memory(#[from] super::hierarchical::MemoryError),

    #[error("{0}")]
    IO(#[from] super::hierarchical::IOError),
}

/// Result type for NumRS2 operations
pub type Result<T> = std::result::Result<T, NumRs2Error>;

/// Implement From<io::Error> for NumRs2Error
impl From<io::Error> for NumRs2Error {
    fn from(err: io::Error) -> Self {
        NumRs2Error::IOError(err.to_string())
    }
}

// Note: oxicode changed its error types; we map them explicitly at call sites
// using map_err instead of a blanket From implementation.

impl NumRs2Error {
    /// Get the error category for this error
    pub fn category(&self) -> ErrorCategory {
        match self {
            NumRs2Error::Core(_) => ErrorCategory::Core,
            NumRs2Error::Computation(_) => ErrorCategory::Computation,
            NumRs2Error::Memory(_) => ErrorCategory::Memory,
            NumRs2Error::IO(_) => ErrorCategory::IO,

            // Legacy variants mapped to categories
            NumRs2Error::DimensionMismatch(_)
            | NumRs2Error::InvalidOperation(_)
            | NumRs2Error::ValueError(_)
            | NumRs2Error::InvalidInput(_)
            | NumRs2Error::IndexError(_)
            | NumRs2Error::IndexOutOfBounds(_)
            | NumRs2Error::ConversionError(_)
            | NumRs2Error::TypeCastError(_)
            | NumRs2Error::ShapeMismatch { .. } => ErrorCategory::Core,

            NumRs2Error::NumericalError(_) => ErrorCategory::Computation,

            NumRs2Error::ComputationError(_)
            | NumRs2Error::BlasError(_)
            | NumRs2Error::LapackError(_) => ErrorCategory::Computation,

            NumRs2Error::AllocationFailed(_) => ErrorCategory::Memory,

            NumRs2Error::SerializationError(_)
            | NumRs2Error::DeserializationError(_)
            | NumRs2Error::IOError(_) => ErrorCategory::IO,

            NumRs2Error::NotImplemented(_)
            | NumRs2Error::RuntimeError(_)
            | NumRs2Error::FeatureNotEnabled(_)
            | NumRs2Error::DistributedComputing(_)
            | NumRs2Error::ControlError(_) => ErrorCategory::Core,
        }
    }

    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NumRs2Error::Core(e) => e.severity(),
            NumRs2Error::Computation(e) => e.severity(),
            NumRs2Error::Memory(e) => e.severity(),
            NumRs2Error::IO(e) => e.severity(),

            // Legacy variants with default severities
            NumRs2Error::AllocationFailed(_) => ErrorSeverity::Critical,
            NumRs2Error::LapackError(_) | NumRs2Error::BlasError(_) => ErrorSeverity::High,
            NumRs2Error::DimensionMismatch(_) | NumRs2Error::ShapeMismatch { .. } => {
                ErrorSeverity::High
            }
            NumRs2Error::NumericalError(_) => ErrorSeverity::High,
            NumRs2Error::InvalidInput(_) => ErrorSeverity::Medium,
            NumRs2Error::IndexOutOfBounds(_) | NumRs2Error::IndexError(_) => ErrorSeverity::Medium,
            NumRs2Error::NotImplemented(_) | NumRs2Error::FeatureNotEnabled(_) => {
                ErrorSeverity::Low
            }
            _ => ErrorSeverity::Medium,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self.severity() {
            ErrorSeverity::Critical => false,
            ErrorSeverity::High => false,
            ErrorSeverity::Medium => true,
            ErrorSeverity::Low => true,
        }
    }

    /// Create a new error with context information
    pub fn with_context<C: Into<OperationContext>>(self, context: C) -> ErrorContext<Self> {
        ErrorContext::new(self, context.into())
    }

    /// Create a new error with location information
    pub fn at_location(self, location: ErrorLocation) -> ErrorContext<Self> {
        ErrorContext::new(self, OperationContext::default()).with_location(location)
    }
}
