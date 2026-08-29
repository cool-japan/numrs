//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
//!
//! CMA-ES is widely regarded as the gold standard for derivative-free continuous
//! optimization. It adapts a multivariate normal distribution to search the
//! objective function landscape, learning both the optimal mean and covariance
//! structure of the search distribution.
//!
//! # Algorithm Overview
//!
//! CMA-ES maintains a search distribution N(m, sigma^2 C) where:
//! - **m** is the mean (distribution center)
//! - **sigma** is the step-size (global scale)
//! - **C** is the covariance matrix (shape of the distribution)
//!
//! Each generation:
//! 1. Sample lambda candidate solutions from the distribution
//! 2. Evaluate the objective function on each candidate
//! 3. Rank candidates by fitness and select the best mu
//! 4. Update the mean m toward the best solutions (weighted recombination)
//! 5. Adapt the covariance matrix C via rank-1 and rank-mu updates
//! 6. Adapt the step-size sigma via cumulative step-size adaptation (CSA)
//!
//! # Features
//!
//! - **Covariance matrix adaptation**: Learns second-order structure of the landscape
//! - **Step-size adaptation**: CSA (Cumulative Step-size Adaptation)
//! - **Weighted recombination**: Rank-mu update with log-linear weights
//! - **Population-based search**: (mu/mu_w, lambda) selection scheme
//! - **IPOP restart strategy**: Increasing population size on stagnation
//! - **Box constraints**: Penalty-based boundary handling with repair
//! - **Eigenvalue decomposition**: Jacobi method for efficient sampling
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::cma_es::{cma_es, CMAESConfig};
//!
//! // Minimize the Rosenbrock function
//! let f = |x: &[f64]| {
//!     let (x0, x1) = (x[0], x[1]);
//!     (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
//! };
//!
//! let x0 = vec![0.0, 0.0];
//! let config = CMAESConfig::default();
//! let result = cma_es(f, &x0, config).expect("CMA-ES should succeed");
//! assert!(result.success);
//! ```
//!
//! # References
//!
//! - Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."
//! - Hansen, N., & Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation
//!   in Evolution Strategies." Evolutionary Computation, 9(2), 159-195.
//! - Auger, A., & Hansen, N. (2005). "A Restart CMA Evolution Strategy with
//!   Increasing Population Size." CEC 2005.

mod algorithm;
pub(crate) mod config;
pub(crate) mod eigen;
pub(crate) mod state;
pub(crate) mod types;

#[cfg(test)]
mod tests;

// ============================================================================
// Public re-exports
// ============================================================================

// Main entry point
pub use algorithm::cma_es;

// Configuration
pub use config::CMAESConfig;

// Result types
pub use types::{CMAESResult, CmaEsConfig, CmaEsResult, TerminationReason};
