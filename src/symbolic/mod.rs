//! Symbolic Computation for NumRS2
//!
//! This module provides symbolic mathematics capabilities including:
//! - Expression tree representation
//! - Symbolic differentiation
//! - Expression simplification and algebraic transformations
//! - Symbolic linear algebra operations
//! - LaTeX and string representation of expressions
//! - Numerical evaluation of symbolic expressions
//!
//! # Overview
//!
//! Symbolic computation allows manipulation of mathematical expressions in their
//! symbolic form before numerical evaluation. This is useful for:
//! - Computing exact derivatives for optimization
//! - Simplifying complex mathematical expressions
//! - Understanding mathematical relationships
//! - Generating efficient numerical code
//!
//! # Integration with Automatic Differentiation
//!
//! This module complements the `autodiff` module by providing:
//! - Symbolic derivatives that can be inspected and simplified
//! - Conversion between symbolic and numeric representations
//! - Combination of symbolic and numeric differentiation techniques
//!
//! # Examples
//!
//! ## Basic Expression Creation and Differentiation
//!
//! ```rust,ignore
//! use numrs2::symbolic::*;
//!
//! // Create symbolic expression: f(x) = x² + 2x + 1
//! let x = Expr::var("x");
//! let f = x.clone() * x.clone() + x.clone() * 2.0 + 1.0;
//!
//! // Compute symbolic derivative: f'(x) = 2x + 2
//! let df = differentiate(&f, "x")?;
//! println!("Derivative: {}", df);
//!
//! // Simplify the result
//! let simplified = simplify(&df);
//! println!("Simplified: {}", simplified);
//!
//! // Evaluate at x = 3
//! let mut vars = std::collections::HashMap::new();
//! vars.insert("x".to_string(), 3.0);
//! let result = f.eval(&vars)?;
//! println!("f(3) = {}", result); // 16.0
//! ```
//!
//! ## LaTeX Output
//!
//! ```rust,ignore
//! use numrs2::symbolic::*;
//!
//! let x = Expr::var("x");
//! let f = (x.clone() + 1.0).pow(2.0);
//! println!("LaTeX: {}", f.to_latex());
//! // Output: (x + 1)^{2}
//! ```
//!
//! # Features
//!
//! - **Expression Types**: Constants, variables, arithmetic operations, transcendental functions
//! - **Differentiation**: Automatic symbolic differentiation using chain rule
//! - **Simplification**: Algebraic simplification including constant folding and identity operations
//! - **Linear Algebra**: Symbolic matrix operations for small systems
//! - **Multiple Representations**: LaTeX, string, and Python-compatible output
//!
//! # Performance Considerations
//!
//! Symbolic computation is designed for relatively small expressions and systems.
//! For large-scale numerical computations, use the numeric methods in other modules.
//! Symbolic linear algebra is limited to small matrices (typically < 10x10) due to
//! exponential growth in expression complexity.

// Re-export main types and functions
pub mod differentiation;
pub mod expr;
pub mod linalg;
pub mod simplify;

pub use differentiation::{differentiate, directional_derivative, gradient, hessian, jacobian};
pub use expr::Expr;
pub use linalg::{
    determinant, inverse, matrix_add, matrix_mul, matrix_sub, solve, trace, transpose,
    SymbolicMatrix,
};
pub use simplify::{expand, simplify};

// Re-export commonly used types
use crate::error::{NumRs2Error, Result};
use std::collections::HashMap;
