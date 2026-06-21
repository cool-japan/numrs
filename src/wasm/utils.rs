//! Utility functions for WebAssembly integration
//!
//! This module provides helper functions for memory management, error handling,
//! logging, and diagnostics in the WebAssembly environment.

use crate::error::NumRs2Error;
use wasm_bindgen::prelude::*;

/// Set up panic hook for better error messages in WebAssembly
///
/// This function installs a panic hook that logs panics to the browser console
/// with full stack traces. Call this once at the start of your application.
///
/// # Example
/// ```javascript
/// import { set_panic_hook } from './pkg/numrs2.js';
/// set_panic_hook();
/// ```
#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "wasm")]
    {
        console_error_panic_hook::set_once();
    }
}

/// Get the NumRS2 version string
///
/// Returns the version of the NumRS2 library.
///
/// # Returns
/// Version string (e.g., "0.3.0")
///
/// # Example
/// ```javascript
/// import { version } from './pkg/numrs2.js';
/// console.log(`NumRS2 version: ${version()}`);
/// ```
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get library name
///
/// Returns the name of the library.
///
/// # Example
/// ```javascript
/// import { library_name } from './pkg/numrs2.js';
/// console.log(library_name()); // "numrs2"
/// ```
#[wasm_bindgen]
pub fn library_name() -> String {
    "numrs2".to_string()
}

/// Get full library information
///
/// Returns a JSON string with library metadata.
///
/// # Returns
/// JSON string containing library information
///
/// # Example
/// ```javascript
/// import { library_info } from './pkg/numrs2.js';
/// const info = JSON.parse(library_info());
/// console.log(info);
/// ```
#[wasm_bindgen]
pub fn library_info() -> String {
    format!(
        r#"{{"name":"{}","version":"{}","description":"{}","features":{}}}"#,
        library_name(),
        version(),
        "NumRS2: High-Performance Numerical Computing in Rust for WebAssembly",
        get_feature_flags()
    )
}

/// Get enabled feature flags
///
/// Returns a JSON array of enabled features.
///
/// # Returns
/// JSON string array of feature names
fn get_feature_flags() -> String {
    let mut features = Vec::new();

    #[cfg(feature = "wasm")]
    features.push("wasm");

    #[cfg(feature = "lapack")]
    features.push("lapack");

    #[cfg(feature = "gpu")]
    features.push("gpu");

    #[cfg(feature = "python")]
    features.push("python");

    #[cfg(feature = "arrow")]
    features.push("arrow");

    #[cfg(feature = "matrix_decomp")]
    features.push("matrix_decomp");

    format!("{:?}", features)
}

/// Log a message to the browser console
///
/// # Parameters
/// - `message`: Message to log
///
/// # Example
/// ```javascript
/// import { log } from './pkg/numrs2.js';
/// log("Computing array operations...");
/// ```
#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}

/// Log a warning to the browser console
///
/// # Parameters
/// - `message`: Warning message
///
/// # Example
/// ```javascript
/// import { warn } from './pkg/numrs2.js';
/// warn("Large array allocation detected");
/// ```
#[wasm_bindgen]
pub fn warn(message: &str) {
    web_sys::console::warn_1(&JsValue::from_str(message));
}

/// Log an error to the browser console
///
/// # Parameters
/// - `message`: Error message
///
/// # Example
/// ```javascript
/// import { error } from './pkg/numrs2.js';
/// error("Operation failed");
/// ```
#[wasm_bindgen]
pub fn error(message: &str) {
    web_sys::console::error_1(&JsValue::from_str(message));
}

/// Get current WebAssembly memory usage in bytes
///
/// Returns the current memory usage of the WebAssembly module.
///
/// # Returns
/// Memory usage in bytes
///
/// # Example
/// ```javascript
/// import { memory_usage } from './pkg/numrs2.js';
/// console.log(`Memory usage: ${memory_usage()} bytes`);
/// ```
#[wasm_bindgen]
pub fn memory_usage() -> usize {
    // Get the WebAssembly memory as a JsValue and cast it to ArrayBuffer
    let memory_value = wasm_bindgen::memory();

    // Convert JsValue to ArrayBuffer and get byte length
    let array_buffer: js_sys::ArrayBuffer = memory_value.unchecked_into();
    array_buffer.byte_length() as usize
}

/// Check if WebAssembly SIMD is supported
///
/// Returns true if the WebAssembly runtime supports SIMD operations.
///
/// # Returns
/// true if SIMD is supported, false otherwise
///
/// # Example
/// ```javascript
/// import { has_simd_support } from './pkg/numrs2.js';
/// if (has_simd_support()) {
///     console.log("SIMD acceleration available!");
/// }
/// ```
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

/// Check if WebAssembly threads are supported
///
/// Returns true if the WebAssembly runtime supports threads (SharedArrayBuffer).
///
/// # Returns
/// true if threads are supported, false otherwise
///
/// # Example
/// ```javascript
/// import { has_threads_support } from './pkg/numrs2.js';
/// if (has_threads_support()) {
///     console.log("Multi-threading available!");
/// }
/// ```
#[wasm_bindgen]
pub fn has_threads_support() -> bool {
    #[cfg(target_feature = "atomics")]
    {
        true
    }
    #[cfg(not(target_feature = "atomics"))]
    {
        false
    }
}

/// Convert NumRs2Error to JsValue
///
/// Internal utility function to convert NumRS2 errors to JavaScript exceptions.
pub(crate) fn error_to_js(error: NumRs2Error) -> JsValue {
    JsValue::from_str(&format!("NumRS2 Error: {}", error))
}

/// Performance timer for benchmarking
///
/// Simple performance timer using the Performance API.
#[wasm_bindgen]
pub struct PerfTimer {
    start_time: f64,
}

impl Default for PerfTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl PerfTimer {
    /// Create a new performance timer
    ///
    /// # Example
    /// ```javascript
    /// const timer = PerfTimer.new();
    /// // ... do some work ...
    /// console.log(`Elapsed: ${timer.elapsed()}ms`);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> PerfTimer {
        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        PerfTimer { start_time }
    }

    /// Get elapsed time in milliseconds
    ///
    /// # Returns
    /// Elapsed time since timer creation in milliseconds
    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() - self.start_time)
            .unwrap_or(0.0)
    }

    /// Reset the timer
    ///
    /// Resets the start time to the current time.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.start_time = web_sys::window()
            .and_then(|w: web_sys::Window| w.performance())
            .map(|p: web_sys::Performance| p.now())
            .unwrap_or(0.0);
    }
}

/// Memory allocation helper using dlmalloc
///
/// This is automatically used when the wasm feature is enabled
/// to provide a maintained pure-Rust allocator optimized for WebAssembly.
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
        assert_eq!(ver, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_library_name() {
        assert_eq!(library_name(), "numrs2");
    }

    #[test]
    fn test_library_info() {
        let info = library_info();
        assert!(info.contains("numrs2"));
        assert!(info.contains("version"));
    }

    #[test]
    fn test_feature_flags() {
        let flags = get_feature_flags();
        assert!(flags.contains("wasm"));
    }

    #[test]
    fn test_simd_support() {
        // Should not panic
        let _has_simd = has_simd_support();
    }

    #[test]
    fn test_threads_support() {
        // Should not panic
        let _has_threads = has_threads_support();
    }
}
