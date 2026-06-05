//! WebAssembly bindings for NumRS2
//!
//! This module provides JavaScript-friendly bindings for NumRS2 functionality,
//! enabling the use of NumRS2 in web browsers and Node.js environments.
//!
//! # Features
//!
//! - **Array Operations**: Create and manipulate N-dimensional arrays from JavaScript
//! - **Linear Algebra**: Matrix operations, decompositions, and solvers
//! - **Statistics**: Statistical functions and probability distributions
//! - **SIMD Optimization**: Automatic SIMD acceleration where supported
//!
//! # Usage
//!
//! This module is only available when compiling with the `wasm` feature flag:
//!
//! ```toml
//! [dependencies]
//! numrs2 = { version = "0.3.0", features = ["wasm"] }
//! ```
//!
//! # Building
//!
//! To build for WebAssembly:
//!
//! ```bash
//! wasm-pack build --target web --features wasm
//! ```
//!
//! # Example
//!
//! ```javascript
//! import init, { WasmArray } from './pkg/numrs2.js';
//!
//! async function main() {
//!     await init();
//!
//!     // Create a 2x2 array
//!     const arr = WasmArray.zeros([2, 2]);
//!     console.log(arr.shape());
//! }
//! ```
//!
//! # SCIRS2 Integration
//!
//! All WebAssembly bindings follow the SCIRS2 ecosystem policy:
//! - Use `scirs2_core::ndarray` for array operations
//! - Use `scirs2_core::random` for random number generation
//! - Use `scirs2-linalg` for linear algebra (pure Rust via OxiBLAS)
//! - No direct dependencies on C/C++ libraries
//!
//! # Performance
//!
//! NumRS2's WebAssembly bindings leverage:
//! - SIMD instructions where supported by the browser
//! - Efficient memory layout for minimal JavaScript ↔ Rust boundary crossings
//! - Zero-copy data transfer where possible
//!
//! # Error Handling
//!
//! All public API functions return `Result` types and do not use `unwrap()`.
//! Errors are properly converted to JavaScript exceptions with meaningful messages.

use wasm_bindgen::prelude::*;

// Public submodules
pub mod array;
pub mod linalg;
pub mod stats;
pub mod utils;

// Re-export commonly used types
pub use array::WasmArray;
pub use utils::{set_panic_hook, version};

/// Initialize the WebAssembly module
///
/// This function should be called before using any NumRS2 functionality
/// in JavaScript. It sets up panic hooks for better error messages.
///
/// # Example
///
/// ```javascript
/// import init from './pkg/numrs2.js';
/// await init();
/// ```
#[wasm_bindgen(start)]
pub fn init_wasm() {
    utils::set_panic_hook();
}

/// Get the NumRS2 version string
///
/// Returns the version of the NumRS2 library.
///
/// # Example
///
/// ```javascript
/// import { get_version } from './pkg/numrs2.js';
/// console.log(get_version()); // "0.3.0"
/// ```
#[wasm_bindgen]
pub fn get_version() -> String {
    version()
}

/// Check if SIMD is available in the current environment
///
/// Returns true if SIMD operations are available in the WebAssembly runtime.
///
/// # Example
///
/// ```javascript
/// import { is_simd_available } from './pkg/numrs2.js';
/// if (is_simd_available()) {
///     console.log("SIMD acceleration available!");
/// }
/// ```
#[wasm_bindgen]
pub fn is_simd_available() -> bool {
    // Check for WASM SIMD support
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_version();
        assert!(!version.is_empty());
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_simd_check() {
        // Should not panic
        let _simd = is_simd_available();
    }
}
