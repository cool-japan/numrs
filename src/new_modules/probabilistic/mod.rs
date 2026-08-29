//! # Probabilistic Programming Module
//!
//! This module provides comprehensive probabilistic programming infrastructure for NumRS2,
//! including advanced inference algorithms, Bayesian utilities, and probabilistic graphical models.
//!
//! ## Overview
//!
//! The probabilistic module offers production-ready implementations of:
//!
//! - **MCMC Inference**: Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo (HMC),
//!   No-U-Turn Sampler (NUTS), and Parallel Tempering
//! - **Variational Inference**: Mean-field VI, Automatic Differentiation VI (ADVI), ELBO optimization
//! - **Bayesian Utilities**: Conjugate priors, posterior computation, model comparison (BIC, DIC, WAIC),
//!   credible intervals, hypothesis testing
//! - **Graphical Models**: Bayesian networks, Markov Random Fields, Hidden Markov Models,
//!   Gaussian Processes
//! - **Extended Distributions**: Beta, Gamma, Dirichlet, Student's t, Wishart, Inverse-Wishart,
//!   Von Mises, and more
//!
//! ## Mathematical Background
//!
//! ### Bayesian Inference
//!
//! Bayesian inference provides a principled framework for updating beliefs about parameters θ
//! given observed data D using Bayes' theorem:
//!
//! ```text
//! p(θ|D) = p(D|θ)p(θ) / p(D)
//! ```
//!
//! where:
//! - p(θ|D) is the posterior distribution
//! - p(D|θ) is the likelihood
//! - p(θ) is the prior distribution
//! - p(D) is the marginal likelihood (evidence)
//!
//! ### Markov Chain Monte Carlo (MCMC)
//!
//! When the posterior distribution cannot be computed analytically, MCMC methods construct
//! a Markov chain whose stationary distribution is the target posterior. Common algorithms include:
//!
//! - **Metropolis-Hastings**: Generic MCMC using proposal distributions
//! - **Gibbs Sampling**: Samples from conditional distributions
//! - **Hamiltonian Monte Carlo**: Uses gradient information for efficient exploration
//! - **NUTS**: Adaptive HMC with automatic step size tuning
//!
//! ### Variational Inference
//!
//! Variational inference approximates the posterior p(θ|D) with a simpler distribution q(θ)
//! by minimizing the Kullback-Leibler divergence:
//!
//! ```text
//! KL(q||p) = ∫ q(θ) log(q(θ)/p(θ|D)) dθ
//! ```
//!
//! This is equivalent to maximizing the Evidence Lower BOund (ELBO):
//!
//! ```text
//! ELBO = E_q[log p(D,θ)] - E_q[log q(θ)]
//! ```
//!
//! ## SCIRS2 Policy Compliance
//!
//! This module strictly follows SCIRS2 ecosystem policies:
//!
//! - **Random Number Generation**: ALWAYS use `scirs2_core::random` (NEVER direct rand/rand_distr)
//! - **Array Operations**: ALWAYS use `scirs2_core::ndarray` (NEVER direct ndarray)
//! - **Parallel Processing**: ALWAYS use `scirs2_core::parallel_ops` (NEVER direct rayon)
//! - **Statistical Functions**: Use `scirs2_stats` for statistical computations
//! - **Linear Algebra**: Use `scirs2_linalg` for matrix operations (Pure Rust via OxiBLAS)
//!
//! ## Usage Examples
//!
//! ### Example 1: Metropolis-Hastings Sampling
//!
//! ```rust,ignore
//! use numrs2::new_modules::probabilistic::{MetropolisHastings, GaussianProposal};
//! use scirs2_core::random::default_rng;
//!
//! // Define log-posterior function
//! let log_posterior = |theta: &[f64]| -> f64 {
//!     // Log-likelihood + log-prior
//!     -0.5 * theta[0].powi(2) // Standard normal prior
//! };
//!
//! // Create sampler with Gaussian proposal
//! let mut rng = default_rng();
//! let proposal = GaussianProposal::new(0.5)?; // Proposal std dev
//! let mut sampler = MetropolisHastings::new(log_posterior, proposal);
//!
//! // Run MCMC for 10,000 iterations
//! let initial_state = vec![0.0];
//! let samples = sampler.sample(&initial_state, 10000, 1000, &mut rng)?;
//! ```
//!
//! ### Example 2: Bayesian Linear Regression
//!
//! ```rust,ignore
//! use numrs2::new_modules::probabilistic::{BayesianLinearRegression, NormalInverseGammaPrior};
//! use numrs2::prelude::*;
//!
//! // Data: y = 2*x + 1 + noise
//! let x = linspace(0.0, 10.0, 100).reshape(&[100, 1]);
//! let y = x.multiply_scalar(2.0).add_scalar(1.0).add(&randn(&[100]));
//!
//! // Set up conjugate prior
//! let prior = NormalInverseGammaPrior::default();
//!
//! // Compute posterior
//! let posterior = prior.update(&x, &y)?;
//!
//! // Sample from posterior predictive
//! let x_new = linspace(10.0, 15.0, 50).reshape(&[50, 1]);
//! let y_pred = posterior.predict(&x_new)?;
//! ```
//!
//! ### Example 3: Model Comparison with WAIC
//!
//! ```rust,ignore
//! use numrs2::new_modules::probabilistic::{ModelComparison, waic};
//!
//! // Compute WAIC for model selection
//! let log_likelihood_samples = /* MCMC samples of log-likelihood */;
//! let waic_score = waic(&log_likelihood_samples)?;
//! println!("WAIC: {}", waic_score.waic);
//! println!("Effective parameters: {}", waic_score.p_waic);
//! ```
//!
//! ## Performance Considerations
//!
//! - **SIMD Optimization**: Distribution operations use SIMD when applicable
//! - **Parallel MCMC**: Multiple chains can run in parallel using `scirs2_core::parallel_ops`
//! - **Memory Efficiency**: Streaming algorithms for large-scale inference
//! - **Numerical Stability**: Log-space computations to prevent underflow
//!
//! ## References
//!
//! - Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
//! - Neal, R. M. (2011). MCMC using Hamiltonian dynamics. *Handbook of Markov Chain Monte Carlo*.
//! - Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. *Journal of Machine Learning Research*.
//! - Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians.
//! - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

use crate::error::NumRs2Error;
use std::fmt;

// Module declarations
pub mod bayesian;
pub mod distributions;
pub mod graphical;
pub mod inference;

#[cfg(test)]
mod tests;

// Re-exports from submodules
pub use bayesian::*;
pub use distributions::*;
pub use graphical::*;
pub use inference::*;

/// Result type for probabilistic operations
pub type Result<T> = std::result::Result<T, ProbabilisticError>;

/// Comprehensive error type for probabilistic programming operations
#[derive(Debug, Clone)]
pub enum ProbabilisticError {
    /// Invalid parameter value
    InvalidParameter { parameter: String, message: String },

    /// Dimension mismatch in array operations
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },

    /// Numerical error (overflow, underflow, NaN, etc.)
    NumericalError { message: String },

    /// Convergence failure in iterative algorithms
    ConvergenceError {
        algorithm: String,
        iterations: usize,
        message: String,
    },

    /// Invalid probability distribution
    InvalidDistribution {
        distribution: String,
        reason: String,
    },

    /// MCMC sampling error
    SamplingError {
        sampler: String,
        iteration: usize,
        message: String,
    },

    /// Variational inference error
    VariationalInferenceError { message: String },

    /// Graphical model error
    GraphicalModelError { model_type: String, message: String },

    /// Integration error with NumRS2
    NumRs2IntegrationError { source: Box<NumRs2Error> },

    /// Generic error with custom message
    Other { message: String },
}

impl fmt::Display for ProbabilisticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbabilisticError::InvalidParameter { parameter, message } => {
                write!(f, "Invalid parameter '{}': {}", parameter, message)
            }
            ProbabilisticError::DimensionMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Dimension mismatch in {}: expected {:?}, got {:?}",
                    operation, expected, actual
                )
            }
            ProbabilisticError::NumericalError { message } => {
                write!(f, "Numerical error: {}", message)
            }
            ProbabilisticError::ConvergenceError {
                algorithm,
                iterations,
                message,
            } => {
                write!(
                    f,
                    "Convergence failure in {} after {} iterations: {}",
                    algorithm, iterations, message
                )
            }
            ProbabilisticError::InvalidDistribution {
                distribution,
                reason,
            } => {
                write!(f, "Invalid distribution '{}': {}", distribution, reason)
            }
            ProbabilisticError::SamplingError {
                sampler,
                iteration,
                message,
            } => {
                write!(
                    f,
                    "Sampling error in {} at iteration {}: {}",
                    sampler, iteration, message
                )
            }
            ProbabilisticError::VariationalInferenceError { message } => {
                write!(f, "Variational inference error: {}", message)
            }
            ProbabilisticError::GraphicalModelError {
                model_type,
                message,
            } => {
                write!(f, "Graphical model error in {}: {}", model_type, message)
            }
            ProbabilisticError::NumRs2IntegrationError { source } => {
                write!(f, "NumRS2 integration error: {}", source)
            }
            ProbabilisticError::Other { message } => {
                write!(f, "Probabilistic error: {}", message)
            }
        }
    }
}

impl std::error::Error for ProbabilisticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProbabilisticError::NumRs2IntegrationError { source } => Some(source),
            _ => None,
        }
    }
}

impl From<NumRs2Error> for ProbabilisticError {
    fn from(error: NumRs2Error) -> Self {
        ProbabilisticError::NumRs2IntegrationError {
            source: Box::new(error),
        }
    }
}

/// Helper function to validate probability values
///
/// Ensures that a probability is in the valid range [0, 1].
///
/// # Arguments
///
/// * `p` - Probability value to validate
/// * `name` - Parameter name for error messages
///
/// # Returns
///
/// * `Ok(())` if the probability is valid
/// * `Err(ProbabilisticError)` if the probability is invalid
///
/// # Examples
///
/// ```rust,ignore
/// validate_probability(0.5, "p")?; // OK
/// validate_probability(-0.1, "p")?; // Error: probability out of range
/// validate_probability(1.5, "p")?; // Error: probability out of range
/// ```
pub fn validate_probability(p: f64, name: &str) -> Result<()> {
    if !p.is_finite() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: name.to_string(),
            message: format!("probability must be finite, got {}", p),
        });
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: name.to_string(),
            message: format!("probability must be in [0, 1], got {}", p),
        });
    }
    Ok(())
}

/// Helper function to validate positive parameter values
///
/// Ensures that a parameter is strictly positive.
///
/// # Arguments
///
/// * `value` - Parameter value to validate
/// * `name` - Parameter name for error messages
///
/// # Returns
///
/// * `Ok(())` if the value is valid
/// * `Err(ProbabilisticError)` if the value is invalid
pub fn validate_positive(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: name.to_string(),
            message: format!("value must be finite, got {}", value),
        });
    }
    if value <= 0.0 {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: name.to_string(),
            message: format!("value must be positive, got {}", value),
        });
    }
    Ok(())
}

/// Helper function to validate non-negative parameter values
///
/// Ensures that a parameter is non-negative.
///
/// # Arguments
///
/// * `value` - Parameter value to validate
/// * `name` - Parameter name for error messages
///
/// # Returns
///
/// * `Ok(())` if the value is valid
/// * `Err(ProbabilisticError)` if the value is invalid
pub fn validate_non_negative(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: name.to_string(),
            message: format!("value must be finite, got {}", value),
        });
    }
    if value < 0.0 {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: name.to_string(),
            message: format!("value must be non-negative, got {}", value),
        });
    }
    Ok(())
}

/// Helper function to validate array shapes match
///
/// # Arguments
///
/// * `expected` - Expected shape
/// * `actual` - Actual shape
/// * `operation` - Operation name for error messages
///
/// # Returns
///
/// * `Ok(())` if shapes match
/// * `Err(ProbabilisticError)` if shapes don't match
pub fn validate_shape(expected: &[usize], actual: &[usize], operation: &str) -> Result<()> {
    if expected != actual {
        return Err(ProbabilisticError::DimensionMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
            operation: operation.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod module_tests {
    use super::*;

    #[test]
    fn test_validate_probability() {
        assert!(validate_probability(0.0, "p").is_ok());
        assert!(validate_probability(0.5, "p").is_ok());
        assert!(validate_probability(1.0, "p").is_ok());
        assert!(validate_probability(-0.1, "p").is_err());
        assert!(validate_probability(1.1, "p").is_err());
        assert!(validate_probability(f64::NAN, "p").is_err());
        assert!(validate_probability(f64::INFINITY, "p").is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(0.1, "x").is_ok());
        assert!(validate_positive(1.0, "x").is_ok());
        assert!(validate_positive(100.0, "x").is_ok());
        assert!(validate_positive(0.0, "x").is_err());
        assert!(validate_positive(-1.0, "x").is_err());
        assert!(validate_positive(f64::NAN, "x").is_err());
    }

    #[test]
    fn test_validate_non_negative() {
        assert!(validate_non_negative(0.0, "x").is_ok());
        assert!(validate_non_negative(0.1, "x").is_ok());
        assert!(validate_non_negative(1.0, "x").is_ok());
        assert!(validate_non_negative(-0.1, "x").is_err());
        assert!(validate_non_negative(f64::NAN, "x").is_err());
    }

    #[test]
    fn test_validate_shape() {
        assert!(validate_shape(&[2, 3], &[2, 3], "test").is_ok());
        assert!(validate_shape(&[2], &[2], "test").is_ok());
        assert!(validate_shape(&[2, 3], &[3, 2], "test").is_err());
        assert!(validate_shape(&[2, 3], &[2], "test").is_err());
    }

    #[test]
    fn test_error_display() {
        let err = ProbabilisticError::InvalidParameter {
            parameter: "alpha".to_string(),
            message: "must be positive".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("alpha"));
        assert!(display.contains("positive"));
    }

    #[test]
    fn test_error_from_numrs2() {
        let numrs2_err = NumRs2Error::DimensionMismatch("expected 2x3, got 3x2".to_string());
        let prob_err: ProbabilisticError = numrs2_err.into();

        match prob_err {
            ProbabilisticError::NumRs2IntegrationError { .. } => {}
            _ => panic!("Expected NumRs2IntegrationError"),
        }
    }
}
