//! Iterative solvers for linear systems
//!
//! This module provides iterative methods for solving large linear systems Ax = b,
//! which are more efficient than direct methods for sparse or very large matrices.
//!
//! # Available Solvers
//!
//! - **Conjugate Gradient (CG)**: For symmetric positive definite systems
//! - **Preconditioned CG (PCG)**: CG with various preconditioners
//! - **GMRES**: For general non-symmetric systems
//! - **FGMRES**: Flexible GMRES with variable preconditioning
//! - **BiCGSTAB**: For non-symmetric systems with better stability
//! - **MINRES**: For symmetric indefinite systems
//!
//! # Preconditioners
//!
//! - **Identity**: No preconditioning
//! - **Jacobi**: Diagonal scaling
//! - **SSOR**: Symmetric Successive Over-Relaxation
//! - **Incomplete Cholesky (IC(0))**: For SPD matrices
//! - **Custom**: User-provided preconditioner function
//!
//! # Iterative Refinement
//!
//! For improving solutions from direct solvers or iterative methods.
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::linalg::iterative_solvers::*;
//!
//! // Solve Ax = b using Conjugate Gradient
//! let a = Array::from_vec(vec![
//!     4.0, 1.0,
//!     1.0, 3.0,
//! ]).reshape(&[2, 2]);
//! let b = Array::from_vec(vec![1.0, 2.0]);
//!
//! let result = conjugate_gradient(&a, &b, None, Some(1e-6), Some(100));
//! assert!(result.is_ok());
//! ```

// Submodules
mod bicgstab;
mod cg;
mod core;
mod gmres;
mod minres;
mod preconditioners;
mod refinement;

// Re-export core types
pub use self::core::{SolverConfig, SolverResult};

// Re-export helper functions (for internal use and testing)
pub use self::core::{compute_norm, compute_norm_vec, dot_vec, matvec, validate_system};

// Re-export Conjugate Gradient methods
pub use self::cg::{conjugate_gradient, pcg, pcg_ichol, pcg_jacobi, pcg_ssor};

// Re-export GMRES methods
pub use self::gmres::{fgmres, fgmres_jacobi, gmres, gmres_jacobi, gmres_precond};

// Re-export BiCGSTAB
pub use self::bicgstab::bicgstab;

// Re-export MINRES
pub use self::minres::minres;

// Re-export preconditioners
pub use self::preconditioners::{
    CustomPreconditioner, IdentityPreconditioner, IncompleteCholeskyPreconditioner,
    JacobiPreconditioner, Preconditioner, SSORPreconditioner,
};

// Re-export iterative refinement
pub use self::refinement::{
    iterative_refinement, iterative_refinement_bicgstab, iterative_refinement_cg, RefinementConfig,
    RefinementResult,
};
