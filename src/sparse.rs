//! Sparse Matrix Operations
//!
//! This module provides functionality for working with sparse matrices,
//! including various sparse matrix formats and operations optimized for
//! matrices with many zero elements.
//!
//! ## Features
//!
//! - Compressed Sparse Row (CSR) format
//! - Compressed Sparse Column (CSC) format  
//! - Coordinate (COO) format
//! - Sparse matrix arithmetic operations
//! - Conversion between sparse and dense formats
//! - Memory-efficient storage and computation
//! - Advanced iterative solvers (CG, BiCGSTAB)
//! - Sparse matrix decompositions (ILU)
//! - Optimized sparse-dense matrix operations

// Re-export functionality from the former new_modules
pub use crate::new_modules::sparse::*;

// Re-export enhanced sparse operations
pub use crate::sparse_enhanced::SparseOpsAdvanced;