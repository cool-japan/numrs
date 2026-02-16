//! Error conversion utilities for Python bindings
//!
//! Converts NumRS2 errors to Python exceptions with proper error types.

use crate::NumRs2Error;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Convert NumRS2 error to appropriate Python exception
impl From<NumRs2Error> for PyErr {
    fn from(err: NumRs2Error) -> PyErr {
        match err {
            // Shape and dimension errors
            NumRs2Error::ShapeMismatch { .. } => PyValueError::new_err(format!("{}", err)),
            NumRs2Error::DimensionMismatch(_) => PyValueError::new_err(format!("{}", err)),

            // Index errors
            NumRs2Error::IndexOutOfBounds { .. } => PyIndexError::new_err(format!("{}", err)),

            // Type errors
            NumRs2Error::ConversionError(_) => PyTypeError::new_err(format!("{}", err)),

            // Computation errors
            NumRs2Error::InvalidOperation(msg) if msg.contains("singular") => {
                PyValueError::new_err("Singular matrix")
            }
            NumRs2Error::InvalidOperation(msg) if msg.contains("converge") => {
                PyRuntimeError::new_err("Algorithm did not converge")
            }

            // I/O errors
            NumRs2Error::IOError(_) => PyRuntimeError::new_err(format!("{}", err)),

            // Generic errors
            _ => PyRuntimeError::new_err(format!("{}", err)),
        }
    }
}

/// Helper to create PyValueError from string
pub fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Helper to create PyTypeError from string
pub fn type_error(msg: impl Into<String>) -> PyErr {
    PyTypeError::new_err(msg.into())
}

/// Helper to create PyIndexError from string
pub fn index_error(msg: impl Into<String>) -> PyErr {
    PyIndexError::new_err(msg.into())
}
