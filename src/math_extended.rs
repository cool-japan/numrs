//! Extended Mathematical Functions
//!
//! This module provides advanced mathematical functions including special functions,
//! polynomial operations, and other mathematical utilities that extend beyond
//! basic element-wise operations.
//!
//! ## Features
//!
//! - Special functions (error functions, gamma functions, Bessel functions, etc.)
//! - Polynomial operations (evaluation, roots, interpolation)
//! - Advanced mathematical utilities
//! - Numerical analysis functions

/// Special mathematical functions
pub mod special {
    //! Special functions from mathematical physics and engineering
    //! 
    //! Provides implementations similar to scipy.special, including:
    //! - Error functions (erf, erfc)
    //! - Gamma and related functions
    //! - Bessel functions
    //! - Hypergeometric functions
    pub use crate::new_modules::special::*;
}

/// Polynomial operations and utilities
pub mod polynomial {
    //! Polynomial manipulation and evaluation functions
    //!
    //! Provides functionality for:
    //! - Polynomial evaluation and arithmetic
    //! - Root finding
    //! - Interpolation
    //! - Polynomial fitting
    pub use crate::new_modules::polynomial::*;
}

// Re-export key functionality at module level for convenience
pub use special::*;
pub use polynomial::*;