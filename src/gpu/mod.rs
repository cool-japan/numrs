//! GPU acceleration for NumRS2
//!
//! This module provides GPU-accelerated versions of NumRS2 array operations using WGPU.
//! The implementation focuses on maintaining the same API as CPU-based operations
//! while providing significant performance improvements for large data sets.
//!
//! ## Feature Flag
//!
//! GPU acceleration is enabled via the "gpu" feature flag in Cargo.toml:
//!
//! ```toml
//! [dependencies]
//! numrs2 = { version = "0.1.0-beta.1", features = ["gpu"] }
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use numrs2::array::Array;
//! use numrs2::gpu;
//!
//! #[cfg(feature = "gpu")]
//! fn main() -> numrs2::error::Result<()> {
//!     // Create two arrays on the CPU (using f32 for better GPU compatibility)
//!     let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).reshape(&[5]);
//!     let b = Array::from_vec(vec![5.0f32, 4.0, 3.0, 2.0, 1.0]).reshape(&[5]);
//!
//!     // Create GPU arrays from CPU arrays
//!     let gpu_a = gpu::GpuArray::from_array(&a)?;
//!     let gpu_b = gpu::GpuArray::from_array(&b)?;
//!
//!     // Perform GPU-accelerated addition
//!     let gpu_result = gpu::add(&gpu_a, &gpu_b)?;
//!
//!     // Convert back to CPU array
//!     let result = gpu_result.to_array()?;
//!     
//!     // Should be [6.0, 6.0, 6.0, 6.0, 6.0]
//!     println!("Result: {:?}", result);
//!     
//!     Ok(())
//! }
//!
//! #[cfg(not(feature = "gpu"))]
//! fn main() {
//!     println!("GPU support is not enabled. Recompile with --features gpu");
//! }
//! ```
//!
//! ## Supported Operations
//!
//! - Basic arithmetic: add, subtract, multiply, divide
//! - Element-wise functions: exp, log, sin, cos, etc.
//! - Matrix operations: matmul, transpose
//! - Reduction operations: sum, mean, min, max
//!
//! ## Limitations
//!
//! - GPU arrays must be of the same data type (f32 or f64)
//! - Operations between CPU and GPU arrays are not directly supported
//! - Not all NumRS2 operations are currently accelerated
//! - Performance benefits are most noticeable for large arrays

// Re-export public types
pub use array::GpuArray;
pub use context::GpuContext;
pub use ops::*;
#[cfg(feature = "gpu")]
pub use util::get_gpu_info;

// Conditionally include GPU modules when the feature is enabled
#[cfg(feature = "gpu")]
mod array;
#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
mod ops;
#[cfg(feature = "gpu")]
mod shaders;
#[cfg(feature = "gpu")]
pub mod util;

// Placeholder stubs for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub struct GpuArray;

#[cfg(not(feature = "gpu"))]
pub struct GpuContext;
