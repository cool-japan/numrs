//! NumRS2 Error System
//!
//! This module provides both the new hierarchical error system and backward
//! compatibility with the existing flat error structure.

// Submodules for hierarchical error system
pub mod context;
pub mod core;
pub mod computation;
pub mod memory;
pub mod io;

// Integration modules
mod legacy;
mod hierarchical;

use std::fmt;
use thiserror::Error;

// Re-export the hierarchical error system
pub use self::hierarchical::*;

// Keep the original flat structure for backward compatibility
pub use legacy::{NumRs2Error, Result};

// Re-export core types for convenience
pub use context::{ErrorContext, ErrorLocation, ErrorSeverity, OperationContext};
pub use core::CoreError;
pub use computation::ComputationError;
pub use memory::MemoryError;
pub use io::IOError;

// Error category classification for hierarchical system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Core library errors (shape, indexing, operations)
    Core,
    /// Numerical computation errors
    Computation,
    /// Memory management errors
    Memory,
    /// Input/output errors
    IO,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Core => write!(f, "Core"),
            ErrorCategory::Computation => write!(f, "Computation"),
            ErrorCategory::Memory => write!(f, "Memory"),
            ErrorCategory::IO => write!(f, "I/O"),
        }
    }
}

/// Enhanced result type with context
pub type ContextResult<T> = std::result::Result<T, ErrorContext<NumRs2Error>>;

// For new code, recommend using the hierarchical system
pub mod prelude {
    pub use super::{
        CoreError, ComputationError, MemoryError, IOError,
        ErrorContext, ErrorLocation, ErrorSeverity, OperationContext,
        NumRs2Error, Result, ErrorCategory, ContextResult
    };
}