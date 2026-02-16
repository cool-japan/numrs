//! WebAssembly tests module
//!
//! This module contains tests for NumRS2's WebAssembly bindings.
//! Tests are executed using wasm-bindgen-test framework in a browser environment.
//!
//! # Running Tests
//!
//! To run WASM tests:
//!
//! ```bash
//! # Test in Firefox (headless)
//! wasm-pack test --headless --firefox --features wasm
//!
//! # Test in Chrome (headless)
//! wasm-pack test --headless --chrome --features wasm
//!
//! # Test in browser (interactive)
//! wasm-pack test --firefox --features wasm
//! ```
//!
//! # Test Coverage
//!
//! - **Array Operations** (`test_wasm_array.rs`):
//!   - Array creation (zeros, ones, full, from_vec)
//!   - Shape manipulation (reshape, transpose)
//!   - Element access (get, set)
//!   - Arithmetic operations (add, subtract, multiply, divide)
//!   - Scalar operations
//!   - Statistical reductions (sum, mean, min, max, std)
//!   - Matrix multiplication
//!
//! - **Linear Algebra** (`test_wasm_linalg.rs`):
//!   - Matrix multiplication (matmul)
//!   - Dot product
//!   - Outer product
//!   - Norm computation (L1, L2, infinity)
//!   - Trace
//!   - Determinant (with lapack feature)
//!   - Matrix inverse (with lapack feature)
//!   - SVD (with lapack feature)
//!   - QR decomposition (with lapack feature)
//!   - Eigenvalues (with lapack feature)
//!
//! - **Statistics** (`test_wasm_stats.rs`):
//!   - Basic statistics (mean, median, variance, std)
//!   - Min/max operations
//!   - Sum and product
//!   - Percentiles
//!   - Histogram
//!   - Covariance and correlation
//!   - Random number generation (normal, uniform)
//!
//! # SCIRS2 Integration
//!
//! All WASM bindings follow the SCIRS2 ecosystem policy:
//! - Use `scirs2_core::ndarray` for array operations
//! - Use `scirs2_core::random` for random number generation
//! - Use `scirs2-linalg` for linear algebra (pure Rust via OxiBLAS)
//! - Use `scirs2-stats` for statistical functions
//! - No direct dependencies on C/C++ libraries
//!
//! # Error Handling
//!
//! All public WASM API functions:
//! - Return `Result` types for operations that can fail
//! - Convert Rust errors to JavaScript `JsValue` errors
//! - Do NOT use `unwrap()` or `expect()` in production code
//! - Provide meaningful error messages

#![cfg(target_arch = "wasm32")]

// Test modules
mod test_wasm_array;
mod test_wasm_linalg;
mod test_wasm_stats;
