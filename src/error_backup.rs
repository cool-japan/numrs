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

// Re-export the hierarchical error system
pub use self::hierarchical::*;

// Keep the original flat structure for backward compatibility
pub use legacy::{NumRs2Error, Result};

// For new code, recommend using the hierarchical system
pub mod prelude {
    pub use super::hierarchical::{
        CoreError, ComputationError, MemoryError, IOError,
        ErrorContext, ErrorLocation, ErrorSeverity, OperationContext,
    };
    pub use super::{NumRs2Error, Result};
}
