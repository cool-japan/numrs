//! Extended Linear Algebra Operations
//!
//! This module provides advanced linear algebra functionality including
//! eigenvalue decomposition, matrix decompositions, and other sophisticated
//! linear algebra operations built on top of the core linalg module.
//!
//! ## Features
//!
//! - Eigenvalue and eigenvector computation
//! - Matrix decompositions (SVD, QR, Cholesky, LU)
//! - Advanced matrix analysis
//! - Numerical stability enhancements

/// Matrix decomposition operations
pub mod decomposition {
    pub use crate::new_modules::matrix_decomp::*;
}

/// Eigenvalue and eigenvector operations  
pub mod eigenvalue {
    pub use crate::new_modules::eigenvalues::*;
}

// Re-export key functionality at module level for convenience
pub use decomposition::*;
pub use eigenvalue::*;