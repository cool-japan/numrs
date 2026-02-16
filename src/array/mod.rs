//! Multi-dimensional array type and operations
//!
//! This module provides the `Array<T>` type, a NumPy-like multi-dimensional array
//! with support for:
//!
//! - Array creation (zeros, ones, eye, tri, diagonal, etc.)
//! - Shape manipulation (reshape, transpose, broadcast)
//! - Element-wise operations (map, par_map)
//! - Aggregation operations (sum, product)
//! - Arithmetic with automatic broadcasting
//! - Linear algebra operations (matmul, dot, norms)
//!
//! # Example
//!
//! ```
//! use numrs2::prelude::*;
//!
//! // Create arrays
//! let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//! let b = Array::<f64>::eye(2, 2, 0);
//!
//! // Matrix multiplication
//! let c = a.matmul(&b).expect("matrix multiplication should succeed");
//!
//! // Element-wise operations with broadcasting
//! let d = &a + 10.0;
//! ```

mod arithmetic;
mod core;
mod creation;
mod linalg;
mod manipulation;
mod operations;
mod operations_optimized;

// Re-export the main Array struct and ArrayFlags from core
pub use self::core::{Array, ArrayFlags};

// Re-export LstsqResult type when features are enabled
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
pub use self::core::LstsqResult;
