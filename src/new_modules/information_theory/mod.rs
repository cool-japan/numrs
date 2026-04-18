//! Information Theory Module
//!
//! This module provides comprehensive information-theoretic measures and concepts
//! for NumRS2, following the SciRS2 ecosystem integration policy.
//!
//! # Overview
//!
//! Information theory, founded by Claude Shannon in 1948, provides a mathematical
//! framework for quantifying information, uncertainty, and communication. This module
//! implements fundamental information-theoretic measures used in machine learning,
//! statistics, signal processing, and data compression.
//!
//! # Mathematical Foundations
//!
//! ## Entropy
//!
//! Entropy measures the average uncertainty or information content of a random variable.
//! For a discrete random variable X with probability mass function p(x):
//!
//! ```text
//! Shannon Entropy: H(X) = -Σ p(x) log₂ p(x)
//!
//! Joint Entropy: H(X,Y) = -Σ p(x,y) log₂ p(x,y)
//!
//! Conditional Entropy: H(X|Y) = H(X,Y) - H(Y)
//!
//! Rényi Entropy: H_α(X) = (1/(1-α)) log₂(Σ p(x)^α)
//! ```
//!
//! ## Divergence
//!
//! Divergence measures quantify the "distance" or dissimilarity between probability
//! distributions (note: most are not true metrics):
//!
//! ```text
//! KL Divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
//!
//! Jensen-Shannon Divergence: JSD(P||Q) = [D_KL(P||M) + D_KL(Q||M)]/2
//!   where M = (P+Q)/2
//!
//! Hellinger Distance: H(P,Q) = √(1 - Σ√(p(x)q(x)))
//! ```
//!
//! ## Mutual Information
//!
//! Mutual information measures the amount of information shared between random variables:
//!
//! ```text
//! Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
//!                           = D_KL(P(X,Y) || P(X)P(Y))
//!
//! Conditional Mutual Information: I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
//!
//! Variation of Information: VI(X,Y) = H(X|Y) + H(Y|X)
//! ```
//!
//! # Module Organization
//!
//! - [`entropy`]: Entropy measures (Shannon, Rényi, differential, cross-entropy)
//! - [`divergence`]: Divergence and distance measures (KL, JS, Hellinger, etc.)
//! - [`mod@mutual_information`]: Mutual information and related measures
//! - [`coding`]: Information-theoretic bounds and model selection criteria
//!
//! # Examples
//!
//! ```rust
//! use numrs2::new_modules::information_theory::{entropy, mutual_information};
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Compute Shannon entropy of a discrete distribution
//! let probs = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
//! let h = entropy::shannon_entropy(&probs, entropy::LogBase::Bits).expect("valid probability distribution");
//! // Uniform distribution has maximum entropy: H = log₂(4) = 2 bits
//!
//! // Compute mutual information between two variables
//! let joint_probs = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).expect("valid shape and data length");
//! let mi = mutual_information::mutual_information(&joint_probs).expect("valid probability distribution");
//! // Independent variables have zero mutual information
//! ```
//!
//! # SCIRS2 Integration Policy
//!
//! This module strictly follows NumRS2's SCIRS2 integration policy:
//! - **Arrays**: Use `scirs2_core::ndarray` (NEVER direct ndarray)
//! - **Random Numbers**: Use `scirs2_core::random` (NEVER direct rand)
//! - **Error Handling**: Base on `crate::error::NumRs2Error`
//! - **No Unwrap**: All error handling via `Result<T, NumRs2Error>`
//! - **Pure Rust**: 100% Pure Rust implementation
//!
//! # References
//!
//! - Shannon, C. E. (1948). "A Mathematical Theory of Communication"
//! - Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory" (2nd ed.)
//! - MacKay, D. J. (2003). "Information Theory, Inference and Learning Algorithms"

// Submodules
pub mod coding;
pub mod divergence;
pub mod entropy;
pub mod mutual_information;

// Re-exports for convenience
pub use coding::{
    aic, bic, binary_symmetric_channel_capacity, entropy_rate, mdl, rate_distortion_binary,
};
pub use divergence::{
    bhattacharyya_coefficient, bhattacharyya_distance, chi_squared_divergence, hellinger_distance,
    jensen_shannon_divergence, kl_divergence, total_variation_distance,
};
pub use entropy::{
    conditional_entropy, cross_entropy, differential_entropy, joint_entropy, renyi_entropy,
    shannon_entropy, LogBase,
};
pub use mutual_information::{
    adjusted_mutual_information, conditional_mutual_information, mutual_information,
    normalized_mutual_information, pointwise_mutual_information, variation_of_information,
    NormalizationType,
};

use crate::error::NumRs2Error;
use std::fmt;
use thiserror::Error;

/// Information Theory specific error types
#[derive(Error, Debug, Clone)]
pub enum InfoTheoryError {
    /// Invalid probability distribution (e.g., negative values, doesn't sum to 1)
    #[error("Invalid probability distribution: {0}")]
    InvalidDistribution(String),

    /// Empty input data
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Numerical instability or invalid result
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

impl From<InfoTheoryError> for NumRs2Error {
    fn from(err: InfoTheoryError) -> Self {
        match err {
            InfoTheoryError::InvalidDistribution(msg) => NumRs2Error::ValueError(msg),
            InfoTheoryError::EmptyInput(msg) => NumRs2Error::InvalidInput(msg),
            InfoTheoryError::InvalidParameter(msg) => NumRs2Error::ValueError(msg),
            InfoTheoryError::NumericalError(msg) => NumRs2Error::NumericalError(msg),
            InfoTheoryError::DimensionMismatch(msg) => NumRs2Error::DimensionMismatch(msg),
            InfoTheoryError::ComputationError(msg) => NumRs2Error::ComputationError(msg),
        }
    }
}

/// Result type for information theory operations
pub type InfoTheoryResult<T> = Result<T, InfoTheoryError>;

// Internal utilities used across submodules

/// Validates that a probability distribution is valid (non-negative, finite)
pub(crate) fn validate_distribution(
    probs: &scirs2_core::ndarray::ArrayView1<f64>,
) -> InfoTheoryResult<()> {
    if probs.is_empty() {
        return Err(InfoTheoryError::EmptyInput(
            "Probability array is empty".to_string(),
        ));
    }

    for &p in probs.iter() {
        if p < 0.0 {
            return Err(InfoTheoryError::InvalidDistribution(format!(
                "Negative probability: {}",
                p
            )));
        }
        if !p.is_finite() {
            return Err(InfoTheoryError::InvalidDistribution(format!(
                "Non-finite probability: {}",
                p
            )));
        }
    }

    Ok(())
}

/// Normalizes a probability distribution (in-place if possible)
pub(crate) fn normalize_distribution(
    probs: &scirs2_core::ndarray::Array1<f64>,
) -> InfoTheoryResult<scirs2_core::ndarray::Array1<f64>> {
    use scirs2_core::ndarray::Array1;

    let sum: f64 = probs.iter().sum();

    if sum <= 0.0 || !sum.is_finite() {
        return Err(InfoTheoryError::InvalidDistribution(format!(
            "Invalid distribution sum: {}",
            sum
        )));
    }

    if (sum - 1.0).abs() < 1e-10 {
        // Already normalized
        Ok(probs.clone())
    } else {
        // Normalize
        Ok(probs / sum)
    }
}

/// Convention for handling 0*log(0) = 0
pub(crate) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if y > 0.0 {
        x * y.ln()
    } else {
        f64::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_validate_distribution() {
        let valid = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        assert!(validate_distribution(&valid.view()).is_ok());

        let negative = Array1::from_vec(vec![0.5, -0.1, 0.6]);
        assert!(validate_distribution(&negative.view()).is_err());

        let infinite = Array1::from_vec(vec![0.5, f64::INFINITY, 0.2]);
        assert!(validate_distribution(&infinite.view()).is_err());

        let empty: Array1<f64> = Array1::from_vec(vec![]);
        assert!(validate_distribution(&empty.view()).is_err());
    }

    #[test]
    fn test_normalize_distribution() {
        let unnormalized = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let normalized = normalize_distribution(&unnormalized).expect("normalization failed");
        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((normalized[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((normalized[1] - 2.0 / 6.0).abs() < 1e-10);
        assert!((normalized[2] - 3.0 / 6.0).abs() < 1e-10);

        let already_normalized = Array1::from_vec(vec![0.25, 0.25, 0.5]);
        let result = normalize_distribution(&already_normalized).expect("normalization failed");
        let sum2: f64 = result.iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_xlogy() {
        // 0*log(0) = 0 convention
        assert_eq!(xlogy(0.0, 0.0), 0.0);
        assert_eq!(xlogy(0.0, 1.0), 0.0);

        // Regular case
        let result = xlogy(2.0, std::f64::consts::E);
        assert!((result - 2.0).abs() < 1e-10);

        // Negative log
        let result2 = xlogy(1.0, 0.5);
        assert!(result2 < 0.0);
    }
}
