//! Python bindings for NumRS2 via PyO3
//!
//! This module provides comprehensive Python bindings for NumRS2's functionality,
//! enabling seamless integration with Python and NumPy.
//!
//! ## Module Organization
//!
//! - `array` - Core array operations and creation
//! - `linalg` - Linear algebra operations
//! - `stats` - Statistical functions and distributions
//! - `optimize` - Optimization algorithms
//! - `nn` - Neural network primitives
//! - `symbolic` - Symbolic computation
//! - `io` - Data I/O formats
//! - `error` - Error conversion utilities

use pyo3::prelude::*;

pub mod array;
pub mod error;
pub mod io;
pub mod linalg;
pub mod nn;
pub mod optimize;
pub mod stats;
pub mod symbolic;

/// NumRS2 Python module
#[pymodule]
fn _numrs2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add array class and functions
    array::register(m)?;

    // Add linear algebra functions
    linalg::register(m)?;

    // Add statistics functions
    stats::register(m)?;

    // Add optimization functions
    optimize::register(m)?;

    // Add neural network functions
    nn::register(m)?;

    // Add symbolic computation functions
    symbolic::register(m)?;

    // Add I/O functions
    io::register(m)?;

    Ok(())
}
