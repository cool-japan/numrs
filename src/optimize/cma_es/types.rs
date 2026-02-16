//! CMA-ES result types and auxiliary types.

use scirs2_core::random::{thread_rng, Distribution, Normal, Rng};
use scirs2_core::rngs::SmallRng;
use scirs2_core::SeedableRng;

use super::CMAESConfig;

// ============================================================================
// Backward-compatibility aliases (used by mod.rs re-exports)
// ============================================================================

/// Backward-compatible alias for [`CMAESConfig`].
pub type CmaEsConfig = CMAESConfig;

/// Backward-compatible alias for [`CMAESResult`].
pub type CmaEsResult = CMAESResult;

// ============================================================================
// Result
// ============================================================================

/// Result of CMA-ES optimization.
///
/// Contains the best solution found, convergence diagnostics,
/// and detailed information about the optimization run.
#[derive(Debug, Clone)]
pub struct CMAESResult {
    /// Best solution found (parameter vector)
    pub x: Vec<f64>,
    /// Objective function value at the best solution
    pub fun: f64,
    /// Number of iterations (generations) executed
    pub nit: usize,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Whether the optimizer converged
    pub success: bool,
    /// Termination status message
    pub message: String,
    /// History of best objective values per generation
    pub history: Vec<f64>,
    /// Final step-size sigma
    pub final_sigma: f64,
    /// Final condition number of the covariance matrix
    pub final_condition_number: f64,
    /// Number of restarts performed (IPOP)
    pub restarts: usize,
    /// Specific termination reason
    pub termination_reason: TerminationReason,
}

// Backward-compatible field accessors
impl CMAESResult {
    /// Backward-compatible accessor: best solution vector.
    pub fn x_best(&self) -> &[f64] {
        &self.x
    }

    /// Backward-compatible accessor: best function value.
    pub fn f_best(&self) -> f64 {
        self.fun
    }

    /// Backward-compatible accessor: number of generations.
    pub fn generations(&self) -> usize {
        self.nit
    }

    /// Backward-compatible accessor: number of function evaluations.
    pub fn function_evaluations(&self) -> usize {
        self.nfev
    }

    /// Backward-compatible accessor: whether the optimizer converged.
    pub fn converged(&self) -> bool {
        self.success
    }

    /// Backward-compatible accessor: convergence history.
    pub fn convergence_history(&self) -> &[f64] {
        &self.history
    }
}

/// Reason for CMA-ES termination.
#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    /// Converged by function value tolerance
    FunctionTolerance,
    /// Converged by parameter tolerance
    ParameterTolerance,
    /// Reached maximum number of generations
    MaxGenerations,
    /// Covariance matrix condition number too large
    ConditionNumber,
    /// Step-size became effectively zero or infinity
    StepSizeDiverged,
    /// All eigenvalues became degenerate
    EigenvalueDegenerate,
    /// No improvement after restarts
    NoImprovementAfterRestarts,
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationReason::FunctionTolerance => write!(f, "function value tolerance reached"),
            TerminationReason::ParameterTolerance => write!(f, "parameter tolerance reached"),
            TerminationReason::MaxGenerations => write!(f, "maximum generations reached"),
            TerminationReason::ConditionNumber => {
                write!(f, "covariance matrix condition number too large")
            }
            TerminationReason::StepSizeDiverged => write!(f, "step-size diverged"),
            TerminationReason::EigenvalueDegenerate => write!(f, "eigenvalues degenerate"),
            TerminationReason::NoImprovementAfterRestarts => {
                write!(f, "no improvement after restarts")
            }
        }
    }
}

// ============================================================================
// RNG wrapper
// ============================================================================

/// Wrapper to allow both seeded and thread-local RNG usage.
pub(crate) enum RngSource {
    Seeded(SmallRng),
    ThreadLocal,
}

impl RngSource {
    pub(crate) fn create(seed: Option<u64>) -> Self {
        match seed {
            Some(s) => RngSource::Seeded(SmallRng::seed_from_u64(s)),
            None => RngSource::ThreadLocal,
        }
    }

    pub(crate) fn sample_normal(&mut self) -> f64 {
        let normal = match Normal::new(0.0, 1.0) {
            Ok(n) => n,
            Err(_) => return 0.0, // Should never happen for N(0,1)
        };
        match self {
            RngSource::Seeded(rng) => normal.sample(rng),
            RngSource::ThreadLocal => {
                let mut rng = thread_rng();
                normal.sample(&mut rng)
            }
        }
    }

    pub(crate) fn sample_normal_with_std(&mut self, sigma: f64) -> f64 {
        let normal = match Normal::new(0.0, sigma) {
            Ok(n) => n,
            Err(_) => return 0.0,
        };
        match self {
            RngSource::Seeded(rng) => normal.sample(rng),
            RngSource::ThreadLocal => {
                let mut rng = thread_rng();
                normal.sample(&mut rng)
            }
        }
    }
}
