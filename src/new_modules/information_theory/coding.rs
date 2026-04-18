//! Information-Theoretic Coding and Model Selection
//!
//! This module implements information-theoretic bounds, channel capacity computations,
//! and model selection criteria based on information theory.
//!
//! # Mathematical Background
//!
//! ## Shannon's Source Coding Theorem
//!
//! For a source with entropy H(X), the average code length L satisfies:
//!
//! ```text
//! H(X) ≤ L < H(X) + 1
//! ```
//!
//! ## Channel Capacity
//!
//! The maximum rate at which information can be reliably transmitted:
//!
//! ```text
//! C = max I(X;Y)
//!     p(x)
//! ```
//!
//! For specific channels:
//! - Binary Symmetric Channel (BSC): C = 1 - H(p)
//! - Binary Erasure Channel (BEC): C = 1 - ε
//!
//! ## Model Selection Criteria
//!
//! Information-theoretic criteria balance model fit with complexity:
//!
//! ```text
//! AIC = -2 ln(L) + 2k
//! BIC = -2 ln(L) + k ln(n)
//! MDL = -ln(L) + (k/2) ln(n)
//! ```
//!
//! where L is likelihood, k is parameters, n is sample size.

use super::entropy::{shannon_entropy, LogBase};
use super::{validate_distribution, InfoTheoryError, InfoTheoryResult};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Estimate entropy rate of a discrete-time sequence
///
/// # Arguments
///
/// * `sequence` - Discrete sequence of symbols
/// * `order` - Order of Markov approximation (0 = memoryless)
///
/// # Returns
///
/// Estimated entropy rate in bits per symbol
///
/// # Mathematical Formula
///
/// For order k Markov source:
/// ```text
/// H_rate = H(X_n | X_{n-1}, ..., X_{n-k})
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::entropy_rate;
/// use scirs2_core::ndarray::Array1;
///
/// // IID sequence
/// let sequence = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0, 1]);
/// let rate = entropy_rate(&sequence, 0).expect("valid sequence and order");
/// assert!(rate > 0.9 && rate < 1.1); // Approximately 1 bit per symbol
/// ```
pub fn entropy_rate(sequence: &Array1<usize>, order: usize) -> Result<f64, NumRs2Error> {
    if sequence.is_empty() {
        return Err(NumRs2Error::InvalidInput("Sequence is empty".to_string()));
    }

    if order >= sequence.len() {
        return Err(NumRs2Error::InvalidInput(format!(
            "Order {} too large for sequence length {}",
            order,
            sequence.len()
        )));
    }

    // Find alphabet size
    let max_symbol = sequence
        .iter()
        .max()
        .ok_or_else(|| NumRs2Error::InvalidInput("Empty sequence".to_string()))?;
    let alphabet_size = max_symbol + 1;

    if order == 0 {
        // Memoryless source: H_rate = H(X)
        let mut counts = vec![0usize; alphabet_size];
        for &symbol in sequence.iter() {
            counts[symbol] += 1;
        }

        let total = sequence.len() as f64;
        let probs: Vec<f64> = counts.iter().map(|&c| (c as f64) / total).collect();
        let probs_array = Array1::from_vec(probs);

        return shannon_entropy(&probs_array, LogBase::Bits);
    }

    // For order k > 0, estimate H(X_n | X_{n-1}, ..., X_{n-k})
    // Count (k+1)-grams and k-grams
    let context_size = alphabet_size.pow(order as u32);
    let ngram_size = alphabet_size.pow((order + 1) as u32);

    let mut context_counts = vec![0usize; context_size];
    let mut ngram_counts = vec![0usize; ngram_size];

    for i in order..sequence.len() {
        // Compute context index
        let mut context_idx = 0;
        for j in 0..order {
            context_idx = context_idx * alphabet_size + sequence[i - order + j];
        }
        context_counts[context_idx] += 1;

        // Compute (k+1)-gram index
        let ngram_idx = context_idx * alphabet_size + sequence[i];
        ngram_counts[ngram_idx] += 1;
    }

    // Compute H(X_n | context) = H(X_n, context) - H(context)
    let total = (sequence.len() - order) as f64;

    // H(context)
    let context_probs: Vec<f64> = context_counts.iter().map(|&c| (c as f64) / total).collect();
    let context_probs_array = Array1::from_vec(context_probs);
    let h_context = shannon_entropy(&context_probs_array, LogBase::Bits)?;

    // H(X_n, context)
    let ngram_probs: Vec<f64> = ngram_counts.iter().map(|&c| (c as f64) / total).collect();
    let ngram_probs_array = Array1::from_vec(ngram_probs);
    let h_ngram = shannon_entropy(&ngram_probs_array, LogBase::Bits)?;

    let rate = h_ngram - h_context;
    Ok(rate.max(0.0))
}

/// Compute capacity of Binary Symmetric Channel (BSC)
///
/// # Arguments
///
/// * `error_prob` - Crossover probability (0 ≤ p ≤ 0.5)
///
/// # Returns
///
/// Channel capacity in bits
///
/// # Mathematical Formula
///
/// ```text
/// C = 1 - H(p) = 1 + p log₂(p) + (1-p) log₂(1-p)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::binary_symmetric_channel_capacity;
///
/// // Perfect channel (p=0)
/// let c = binary_symmetric_channel_capacity(0.0).expect("valid channel parameter");
/// assert!((c - 1.0).abs() < 1e-10); // C = 1 bit
///
/// // Maximally noisy (p=0.5)
/// let c2 = binary_symmetric_channel_capacity(0.5).expect("valid channel parameter");
/// assert!(c2.abs() < 1e-10); // C = 0 bits
/// ```
pub fn binary_symmetric_channel_capacity(error_prob: f64) -> Result<f64, NumRs2Error> {
    if !(0.0..=0.5).contains(&error_prob) {
        return Err(NumRs2Error::ValueError(format!(
            "Error probability must be in [0, 0.5], got {}",
            error_prob
        )));
    }

    if error_prob == 0.0 {
        return Ok(1.0);
    }

    if (error_prob - 0.5).abs() < 1e-10 {
        return Ok(0.0);
    }

    // C = 1 - H(p)
    let probs = Array1::from_vec(vec![error_prob, 1.0 - error_prob]);
    let h_p = shannon_entropy(&probs, LogBase::Bits)?;

    Ok(1.0 - h_p)
}

/// Compute capacity of Binary Erasure Channel (BEC)
///
/// # Arguments
///
/// * `erasure_prob` - Erasure probability (0 ≤ ε ≤ 1)
///
/// # Returns
///
/// Channel capacity in bits
///
/// # Mathematical Formula
///
/// ```text
/// C = 1 - ε
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::binary_erasure_channel_capacity;
///
/// // Perfect channel (ε=0)
/// let c = binary_erasure_channel_capacity(0.0).expect("valid channel parameter");
/// assert!((c - 1.0).abs() < 1e-10); // C = 1 bit
///
/// // Half erasures (ε=0.5)
/// let c2 = binary_erasure_channel_capacity(0.5).expect("valid channel parameter");
/// assert!((c2 - 0.5).abs() < 1e-10); // C = 0.5 bits
/// ```
pub fn binary_erasure_channel_capacity(erasure_prob: f64) -> Result<f64, NumRs2Error> {
    if !(0.0..=1.0).contains(&erasure_prob) {
        return Err(NumRs2Error::ValueError(format!(
            "Erasure probability must be in [0, 1], got {}",
            erasure_prob
        )));
    }

    // C = 1 - ε
    Ok(1.0 - erasure_prob)
}

/// Compute rate-distortion function for binary source
///
/// # Arguments
///
/// * `source_prob` - Probability of source symbol 1
/// * `distortion` - Maximum allowed distortion (0 ≤ D ≤ min(p, 1-p))
///
/// # Returns
///
/// Rate-distortion R(D) in bits
///
/// # Mathematical Formula
///
/// For binary source with p = P(X=1):
/// ```text
/// R(D) = H(p) - H(D)  if D ≤ min(p, 1-p)
///      = 0            if D ≥ min(p, 1-p)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::rate_distortion_binary;
///
/// // Fair coin, zero distortion
/// let r = rate_distortion_binary(0.5, 0.0).expect("valid distribution and distortion");
/// assert!((r - 1.0).abs() < 1e-10); // R(0) = H(0.5) = 1 bit
///
/// // Fair coin, maximum distortion
/// let r2 = rate_distortion_binary(0.5, 0.5).expect("valid distribution and distortion");
/// assert!(r2.abs() < 1e-10); // R(0.5) = 0 bits
/// ```
pub fn rate_distortion_binary(source_prob: f64, distortion: f64) -> Result<f64, NumRs2Error> {
    if !(0.0..=1.0).contains(&source_prob) {
        return Err(NumRs2Error::ValueError(format!(
            "Source probability must be in [0, 1], got {}",
            source_prob
        )));
    }

    if distortion < 0.0 {
        return Err(NumRs2Error::ValueError(format!(
            "Distortion must be non-negative, got {}",
            distortion
        )));
    }

    let p = source_prob;
    let d_max = p.min(1.0 - p);

    if distortion >= d_max {
        return Ok(0.0);
    }

    // R(D) = H(p) - H(D)
    let source_probs = Array1::from_vec(vec![p, 1.0 - p]);
    let h_p = shannon_entropy(&source_probs, LogBase::Bits)?;

    let distortion_probs = Array1::from_vec(vec![distortion, 1.0 - distortion]);
    let h_d = shannon_entropy(&distortion_probs, LogBase::Bits)?;

    Ok(h_p - h_d)
}

/// Compute Akaike Information Criterion (AIC)
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `num_parameters` - Number of model parameters (k)
///
/// # Returns
///
/// AIC value (lower is better)
///
/// # Mathematical Formula
///
/// ```text
/// AIC = -2 ln(L) + 2k
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::aic;
///
/// let log_likelihood = -100.0;
/// let k = 5;
/// let aic_val = aic(log_likelihood, k);
/// assert!((aic_val - 210.0).abs() < 1e-10); // AIC = -2*(-100) + 2*5 = 210
/// ```
pub fn aic(log_likelihood: f64, num_parameters: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * (num_parameters as f64)
}

/// Compute corrected AIC for small sample sizes (AICc)
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `num_parameters` - Number of model parameters (k)
/// * `sample_size` - Sample size (n)
///
/// # Returns
///
/// AICc value (lower is better)
///
/// # Mathematical Formula
///
/// ```text
/// AICc = AIC + 2k(k+1)/(n-k-1)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::aicc;
///
/// let log_likelihood = -100.0;
/// let k = 5;
/// let n = 50;
/// let aicc_val = aicc(log_likelihood, k, n).expect("valid AICc parameters");
/// // AICc = AIC + correction
/// assert!(aicc_val > 210.0); // Larger than AIC due to correction
/// ```
pub fn aicc(
    log_likelihood: f64,
    num_parameters: usize,
    sample_size: usize,
) -> Result<f64, NumRs2Error> {
    if sample_size <= num_parameters + 1 {
        return Err(NumRs2Error::ValueError(format!(
            "Sample size {} must be greater than k+1 = {}",
            sample_size,
            num_parameters + 1
        )));
    }

    let aic_val = aic(log_likelihood, num_parameters);
    let k = num_parameters as f64;
    let n = sample_size as f64;

    let correction = (2.0 * k * (k + 1.0)) / (n - k - 1.0);

    Ok(aic_val + correction)
}

/// Compute Bayesian Information Criterion (BIC)
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `num_parameters` - Number of model parameters (k)
/// * `sample_size` - Sample size (n)
///
/// # Returns
///
/// BIC value (lower is better)
///
/// # Mathematical Formula
///
/// ```text
/// BIC = -2 ln(L) + k ln(n)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::bic;
///
/// let log_likelihood = -100.0;
/// let k = 5;
/// let n = 100;
/// let bic_val = bic(log_likelihood, k, n);
/// // BIC = -2*(-100) + 5*ln(100) ≈ 200 + 23.03 = 223.03
/// assert!((bic_val - 223.03).abs() < 0.1);
/// ```
pub fn bic(log_likelihood: f64, num_parameters: usize, sample_size: usize) -> f64 {
    let k = num_parameters as f64;
    let n = sample_size as f64;

    -2.0 * log_likelihood + k * n.ln()
}

/// Compute Minimum Description Length (MDL) criterion
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `num_parameters` - Number of model parameters (k)
/// * `sample_size` - Sample size (n)
///
/// # Returns
///
/// MDL value (lower is better)
///
/// # Mathematical Formula
///
/// ```text
/// MDL = -ln(L) + (k/2) ln(n)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::coding::mdl;
///
/// let log_likelihood = -100.0;
/// let k = 5;
/// let n = 100;
/// let mdl_val = mdl(log_likelihood, k, n);
/// // MDL = -(-100) + (5/2)*ln(100) ≈ 100 + 11.51 = 111.51
/// assert!((mdl_val - 111.51).abs() < 0.1);
/// ```
pub fn mdl(log_likelihood: f64, num_parameters: usize, sample_size: usize) -> f64 {
    let k = num_parameters as f64;
    let n = sample_size as f64;

    -log_likelihood + (k / 2.0) * n.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_entropy_rate_iid() {
        // IID binary sequence (fair coin)
        let sequence = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        let rate = entropy_rate(&sequence, 0).expect("entropy rate failed");
        // For fair coin, H = 1 bit
        assert!((rate - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_entropy_rate_deterministic() {
        // Deterministic sequence
        let sequence = Array1::from_vec(vec![0, 0, 0, 0, 0, 0, 0, 0]);
        let rate = entropy_rate(&sequence, 0).expect("entropy rate failed");
        // H = 0 for deterministic
        assert!(rate.abs() < EPSILON);
    }

    #[test]
    fn test_entropy_rate_markov() {
        // Simple Markov chain: 0 -> 1, 1 -> 0 (alternating)
        let sequence = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        let rate = entropy_rate(&sequence, 1).expect("entropy rate failed");
        // For deterministic transitions, H_rate = 0
        assert!(rate.abs() < 0.1);
    }

    #[test]
    fn test_bsc_capacity() {
        // Perfect channel
        let c = binary_symmetric_channel_capacity(0.0).expect("bsc capacity failed");
        assert!((c - 1.0).abs() < EPSILON);

        // Maximally noisy
        let c2 = binary_symmetric_channel_capacity(0.5).expect("bsc capacity failed");
        assert!(c2.abs() < EPSILON);

        // Intermediate
        let c3 = binary_symmetric_channel_capacity(0.1).expect("bsc capacity failed");
        assert!(c3 > 0.5 && c3 < 1.0);
    }

    #[test]
    fn test_bec_capacity() {
        // Perfect channel
        let c = binary_erasure_channel_capacity(0.0).expect("bec capacity failed");
        assert!((c - 1.0).abs() < EPSILON);

        // Half erasures
        let c2 = binary_erasure_channel_capacity(0.5).expect("bec capacity failed");
        assert!((c2 - 0.5).abs() < EPSILON);

        // Full erasures
        let c3 = binary_erasure_channel_capacity(1.0).expect("bec capacity failed");
        assert!(c3.abs() < EPSILON);
    }

    #[test]
    fn test_rate_distortion_binary() {
        // Fair coin, zero distortion
        let r = rate_distortion_binary(0.5, 0.0).expect("rate distortion failed");
        assert!((r - 1.0).abs() < EPSILON); // R(0) = H(0.5) = 1 bit

        // Fair coin, maximum distortion
        let r2 = rate_distortion_binary(0.5, 0.5).expect("rate distortion failed");
        assert!(r2.abs() < EPSILON); // R(0.5) = 0 bits

        // Biased coin
        let r3 = rate_distortion_binary(0.2, 0.0).expect("rate distortion failed");
        let expected = -0.2 * 0.2_f64.log2() - 0.8 * 0.8_f64.log2();
        assert!((r3 - expected).abs() < 0.01);
    }

    #[test]
    fn test_aic() {
        let log_likelihood = -100.0;
        let k = 5;
        let aic_val = aic(log_likelihood, k);
        assert!((aic_val - 210.0).abs() < EPSILON); // AIC = -2*(-100) + 2*5 = 210
    }

    #[test]
    fn test_aicc() {
        let log_likelihood = -100.0;
        let k = 5;
        let n = 50;
        let aicc_val = aicc(log_likelihood, k, n).expect("aicc failed");

        let aic_val = aic(log_likelihood, k);
        let correction = (2.0 * 5.0 * 6.0) / (50.0 - 5.0 - 1.0);
        let expected = aic_val + correction;

        assert!((aicc_val - expected).abs() < EPSILON);
        assert!(aicc_val > aic_val); // AICc > AIC due to correction
    }

    #[test]
    fn test_bic() {
        let log_likelihood = -100.0;
        let k = 5;
        let n = 100;
        let bic_val = bic(log_likelihood, k, n);

        let expected = -2.0 * log_likelihood + (k as f64) * (n as f64).ln();
        assert!((bic_val - expected).abs() < EPSILON);
    }

    #[test]
    fn test_mdl() {
        let log_likelihood = -100.0;
        let k = 5;
        let n = 100;
        let mdl_val = mdl(log_likelihood, k, n);

        let expected = -log_likelihood + (k as f64 / 2.0) * (n as f64).ln();
        assert!((mdl_val - expected).abs() < EPSILON);
    }

    #[test]
    fn test_model_selection_comparison() {
        let log_likelihood = -100.0;
        let k = 5;
        let n = 100;

        let aic_val = aic(log_likelihood, k);
        let bic_val = bic(log_likelihood, k, n);

        // BIC penalizes complexity more heavily than AIC for large n
        assert!(bic_val > aic_val);
    }

    #[test]
    fn test_coding_errors() {
        // Invalid error probability for BSC
        assert!(binary_symmetric_channel_capacity(-0.1).is_err());
        assert!(binary_symmetric_channel_capacity(0.6).is_err());

        // Invalid erasure probability for BEC
        assert!(binary_erasure_channel_capacity(-0.1).is_err());
        assert!(binary_erasure_channel_capacity(1.1).is_err());

        // Invalid distortion for rate-distortion
        assert!(rate_distortion_binary(0.5, -0.1).is_err());

        // Invalid sample size for AICc
        assert!(aicc(-100.0, 5, 5).is_err());
    }
}
