//! Entropy Measures
//!
//! This module implements various entropy measures from information theory,
//! which quantify the uncertainty or information content of probability distributions.
//!
//! # Mathematical Background
//!
//! ## Shannon Entropy
//!
//! Shannon entropy measures the average information content:
//!
//! ```text
//! H(X) = -Σ p(x) log_b p(x)
//! ```
//!
//! where b is the logarithm base (2 for bits, e for nats, 10 for hartleys).
//!
//! Properties:
//! - H(X) ≥ 0 (non-negative)
//! - H(X) = 0 iff X is deterministic
//! - H(X) ≤ log_b(n) for n-valued random variable (maximum for uniform)
//!
//! ## Joint and Conditional Entropy
//!
//! ```text
//! Joint: H(X,Y) = -Σ p(x,y) log p(x,y)
//! Conditional: H(X|Y) = H(X,Y) - H(Y) = -Σ p(x,y) log p(x|y)
//! ```
//!
//! ## Cross-Entropy
//!
//! Measures the average number of bits needed to encode events from P using Q:
//!
//! ```text
//! H(P,Q) = -Σ p(x) log q(x)
//! ```
//!
//! Always H(P,Q) ≥ H(P), with equality iff P = Q.
//!
//! ## Rényi Entropy
//!
//! Generalized entropy parameterized by α ≥ 0, α ≠ 1:
//!
//! ```text
//! H_α(X) = (1/(1-α)) log_b(Σ p(x)^α)
//! ```
//!
//! Special cases:
//! - α → 1: Shannon entropy
//! - α = 0: log(n) (Hartley entropy)
//! - α = 2: -log(Σ p(x)^2) (collision entropy)
//! - α → ∞: -log(max p(x)) (min-entropy)

use super::{validate_distribution, xlogy, InfoTheoryError, InfoTheoryResult};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::f64::consts::E;

/// Logarithm base for entropy calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogBase {
    /// Base 2 (entropy in bits, standard in computer science)
    Bits,
    /// Natural logarithm (entropy in nats, standard in mathematics)
    Nats,
    /// Base 10 (entropy in hartleys or bans)
    Hartleys,
    /// Custom base
    Custom(f64),
}

impl LogBase {
    /// Convert natural logarithm to the specified base
    fn convert(&self, ln_value: f64) -> f64 {
        match self {
            LogBase::Bits => ln_value / 2_f64.ln(),
            LogBase::Nats => ln_value,
            LogBase::Hartleys => ln_value / 10_f64.ln(),
            LogBase::Custom(base) => {
                if *base <= 0.0 || *base == 1.0 {
                    f64::NAN
                } else {
                    ln_value / base.ln()
                }
            }
        }
    }
}

/// Compute Shannon entropy of a discrete probability distribution
///
/// # Arguments
///
/// * `probs` - Probability distribution (will be normalized if sum != 1)
/// * `base` - Logarithm base for entropy computation
///
/// # Returns
///
/// Shannon entropy H(X) in the specified units
///
/// # Mathematical Formula
///
/// ```text
/// H(X) = -Σ p(x) log_b p(x)
/// ```
///
/// with the convention 0*log(0) = 0.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::entropy::{shannon_entropy, LogBase};
/// use scirs2_core::ndarray::Array1;
///
/// // Uniform distribution has maximum entropy
/// let uniform = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
/// let h = shannon_entropy(&uniform, LogBase::Bits).unwrap();
/// assert!((h - 2.0).abs() < 1e-10); // log2(4) = 2 bits
///
/// // Deterministic distribution has zero entropy
/// let deterministic = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
/// let h = shannon_entropy(&deterministic, LogBase::Bits).unwrap();
/// assert!(h.abs() < 1e-10);
/// ```
pub fn shannon_entropy(probs: &Array1<f64>, base: LogBase) -> Result<f64, NumRs2Error> {
    validate_distribution(&probs.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let normalized =
        super::normalize_distribution(probs).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // H(X) = -Σ p(x) log p(x)
    let mut entropy = 0.0;
    for &p in normalized.iter() {
        entropy -= xlogy(p, p);
    }

    Ok(base.convert(entropy))
}

/// Compute joint entropy of a joint probability distribution
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y)
/// * `base` - Logarithm base for entropy computation
///
/// # Returns
///
/// Joint entropy H(X,Y) in the specified units
///
/// # Mathematical Formula
///
/// ```text
/// H(X,Y) = -Σ p(x,y) log p(x,y)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::entropy::{joint_entropy, LogBase};
/// use scirs2_core::ndarray::Array2;
///
/// // Independent uniform variables
/// let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).unwrap();
/// let h = joint_entropy(&joint, LogBase::Bits).unwrap();
/// assert!((h - 2.0).abs() < 1e-10); // log2(4) = 2 bits
/// ```
pub fn joint_entropy(joint_probs: &Array2<f64>, base: LogBase) -> Result<f64, NumRs2Error> {
    if joint_probs.is_empty() {
        return Err(NumRs2Error::InvalidInput(
            "Joint probability array is empty".to_string(),
        ));
    }

    // Flatten and validate
    let flat = joint_probs.iter().cloned().collect::<Vec<_>>();
    let flat_array = Array1::from_vec(flat);

    validate_distribution(&flat_array.view())
        .map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let normalized = super::normalize_distribution(&flat_array)
        .map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // H(X,Y) = -Σ p(x,y) log p(x,y)
    let mut entropy = 0.0;
    for &p in normalized.iter() {
        entropy -= xlogy(p, p);
    }

    Ok(base.convert(entropy))
}

/// Compute conditional entropy H(X|Y) from joint distribution
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y)
/// * `base` - Logarithm base for entropy computation
///
/// # Returns
///
/// Conditional entropy H(X|Y) in the specified units
///
/// # Mathematical Formula
///
/// ```text
/// H(X|Y) = H(X,Y) - H(Y)
///        = -Σ p(x,y) log p(x|y)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::entropy::{conditional_entropy, LogBase};
/// use scirs2_core::ndarray::Array2;
///
/// // For independent variables, H(X|Y) = H(X)
/// let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).unwrap();
/// let h_cond = conditional_entropy(&joint, LogBase::Bits).unwrap();
/// assert!((h_cond - 1.0).abs() < 1e-10); // Each variable has 1 bit
/// ```
pub fn conditional_entropy(joint_probs: &Array2<f64>, base: LogBase) -> Result<f64, NumRs2Error> {
    // H(X|Y) = H(X,Y) - H(Y)
    let h_xy = joint_entropy(joint_probs, base)?;

    // Compute marginal H(Y) by summing over X (rows)
    let marginal_y: Array1<f64> = joint_probs.sum_axis(Axis(0));
    let h_y = shannon_entropy(&marginal_y, base)?;

    Ok(h_xy - h_y)
}

/// Compute cross-entropy between two probability distributions
///
/// # Arguments
///
/// * `p` - True probability distribution
/// * `q` - Estimated/model probability distribution
/// * `base` - Logarithm base for entropy computation
///
/// # Returns
///
/// Cross-entropy H(P,Q) in the specified units
///
/// # Mathematical Formula
///
/// ```text
/// H(P,Q) = -Σ p(x) log q(x)
/// ```
///
/// Always H(P,Q) ≥ H(P), with equality iff P = Q.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::entropy::{cross_entropy, LogBase};
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.4, 0.6]);
/// let h_cross = cross_entropy(&p, &q, LogBase::Bits).unwrap();
/// assert!(h_cross > 1.0); // Greater than H(P) = 1 bit
/// ```
pub fn cross_entropy(p: &Array1<f64>, q: &Array1<f64>, base: LogBase) -> Result<f64, NumRs2Error> {
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    validate_distribution(&p.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    validate_distribution(&q.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // H(P,Q) = -Σ p(x) log q(x)
    let mut cross_ent = 0.0;
    for i in 0..p_norm.len() {
        cross_ent -= xlogy(p_norm[i], q_norm[i]);
    }

    Ok(base.convert(cross_ent))
}

/// Compute Rényi entropy of a probability distribution
///
/// # Arguments
///
/// * `probs` - Probability distribution
/// * `alpha` - Order parameter (α ≥ 0, α ≠ 1)
/// * `base` - Logarithm base for entropy computation
///
/// # Returns
///
/// Rényi entropy H_α(X) in the specified units
///
/// # Mathematical Formula
///
/// ```text
/// H_α(X) = (1/(1-α)) log(Σ p(x)^α)
/// ```
///
/// Special cases:
/// - α → 0: log(n) (Hartley entropy)
/// - α → 1: Shannon entropy (limit)
/// - α = 2: -log(Σ p(x)^2) (collision entropy)
/// - α → ∞: -log(max p(x)) (min-entropy)
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::entropy::{renyi_entropy, LogBase};
/// use scirs2_core::ndarray::Array1;
///
/// let probs = Array1::from_vec(vec![0.5, 0.3, 0.2]);
///
/// // α = 2 (collision entropy)
/// let h2 = renyi_entropy(&probs, 2.0, LogBase::Bits).unwrap();
/// assert!(h2 > 0.0);
///
/// // As α → 1, should approach Shannon entropy
/// let h_shannon = renyi_entropy(&probs, 0.999, LogBase::Bits).unwrap();
/// let h_shannon2 = renyi_entropy(&probs, 1.001, LogBase::Bits).unwrap();
/// assert!((h_shannon - h_shannon2).abs() < 0.1);
/// ```
pub fn renyi_entropy(probs: &Array1<f64>, alpha: f64, base: LogBase) -> Result<f64, NumRs2Error> {
    if alpha < 0.0 {
        return Err(NumRs2Error::ValueError(format!(
            "Alpha must be non-negative, got {}",
            alpha
        )));
    }

    if (alpha - 1.0).abs() < 1e-10 {
        // α = 1: return Shannon entropy
        return shannon_entropy(probs, base);
    }

    validate_distribution(&probs.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let normalized =
        super::normalize_distribution(probs).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    if alpha == 0.0 {
        // H_0 = log(n) (Hartley entropy - count of non-zero elements)
        let n = normalized.iter().filter(|&&p| p > 0.0).count() as f64;
        return Ok(base.convert(n.ln()));
    }

    if alpha.is_infinite() {
        // H_∞ = -log(max p(x)) (min-entropy)
        let max_p = normalized.iter().cloned().fold(0.0, f64::max);
        if max_p <= 0.0 {
            return Err(NumRs2Error::NumericalError(
                "All probabilities are zero".to_string(),
            ));
        }
        return Ok(base.convert(-max_p.ln()));
    }

    // General case: H_α = (1/(1-α)) log(Σ p^α)
    let sum_p_alpha: f64 = normalized.iter().map(|&p| p.powf(alpha)).sum();

    if sum_p_alpha <= 0.0 || !sum_p_alpha.is_finite() {
        return Err(NumRs2Error::NumericalError(format!(
            "Invalid sum of p^α: {}",
            sum_p_alpha
        )));
    }

    let renyi = sum_p_alpha.ln() / (1.0 - alpha);
    Ok(base.convert(renyi))
}

/// Compute differential entropy for continuous distributions
///
/// # Arguments
///
/// * `pdf_values` - Probability density function values at sample points
/// * `dx` - Spacing between sample points (bin width)
/// * `base` - Logarithm base for entropy computation
///
/// # Returns
///
/// Differential entropy h(X) in the specified units
///
/// # Mathematical Formula
///
/// For continuous random variable with PDF f(x):
///
/// ```text
/// h(X) = -∫ f(x) log f(x) dx ≈ -Σ f(xi) log f(xi) Δx
/// ```
///
/// Note: Differential entropy can be negative (unlike discrete entropy).
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::entropy::{differential_entropy, LogBase};
/// use scirs2_core::ndarray::Array1;
///
/// // Approximate uniform distribution on [0, 1]
/// let n = 100;
/// let pdf = Array1::from_elem(n, 1.0); // Uniform PDF = 1
/// let dx = 1.0 / (n as f64);
/// let h = differential_entropy(&pdf, dx, LogBase::Nats).unwrap();
/// // For uniform on [0,1], h(X) = 0 nats
/// assert!(h.abs() < 0.1);
/// ```
pub fn differential_entropy(
    pdf_values: &Array1<f64>,
    dx: f64,
    base: LogBase,
) -> Result<f64, NumRs2Error> {
    if pdf_values.is_empty() {
        return Err(NumRs2Error::InvalidInput(
            "PDF values array is empty".to_string(),
        ));
    }

    if dx <= 0.0 {
        return Err(NumRs2Error::ValueError(format!(
            "dx must be positive, got {}",
            dx
        )));
    }

    // Check for valid PDF values (non-negative)
    for &f in pdf_values.iter() {
        if f < 0.0 {
            return Err(NumRs2Error::ValueError(format!(
                "PDF value cannot be negative: {}",
                f
            )));
        }
        if !f.is_finite() {
            return Err(NumRs2Error::ValueError(format!(
                "PDF value must be finite: {}",
                f
            )));
        }
    }

    // h(X) = -∫ f(x) log f(x) dx ≈ -Σ f(xi) log f(xi) Δx
    let mut entropy = 0.0;
    for &f in pdf_values.iter() {
        entropy -= xlogy(f, f) * dx;
    }

    Ok(base.convert(entropy))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_log_base_conversion() {
        let ln_val = 2_f64.ln(); // ln(2)

        assert!((LogBase::Bits.convert(ln_val) - 1.0).abs() < EPSILON);
        assert!((LogBase::Nats.convert(ln_val) - 2_f64.ln()).abs() < EPSILON);
        assert!((LogBase::Hartleys.convert(ln_val) - 2_f64.log10()).abs() < EPSILON);
        assert!((LogBase::Custom(2.0).convert(ln_val) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution has maximum entropy
        let uniform = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let h = shannon_entropy(&uniform, LogBase::Bits).expect("entropy failed");
        assert!((h - 2.0).abs() < EPSILON); // log2(4) = 2 bits
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        // Deterministic distribution has zero entropy
        let deterministic = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let h = shannon_entropy(&deterministic, LogBase::Bits).expect("entropy failed");
        assert!(h.abs() < EPSILON);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        // Binary entropy function
        let p = 0.3;
        let probs = Array1::from_vec(vec![p, 1.0 - p]);
        let h = shannon_entropy(&probs, LogBase::Bits).expect("entropy failed");

        // H(p) = -p*log2(p) - (1-p)*log2(1-p)
        let expected = -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
        assert!((h - expected).abs() < EPSILON);
    }

    #[test]
    fn test_shannon_entropy_normalization() {
        // Should normalize if sum != 1
        let unnormalized = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let h = shannon_entropy(&unnormalized, LogBase::Bits).expect("entropy failed");
        assert!((h - 2.0).abs() < EPSILON); // Should be same as [0.25, 0.25, 0.25, 0.25]
    }

    #[test]
    fn test_joint_entropy() {
        // Independent uniform variables
        let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25])
            .expect("array creation failed");
        let h = joint_entropy(&joint, LogBase::Bits).expect("entropy failed");
        assert!((h - 2.0).abs() < EPSILON); // log2(4) = 2 bits
    }

    #[test]
    fn test_conditional_entropy_independent() {
        // For independent variables, H(X|Y) = H(X)
        let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25])
            .expect("array creation failed");
        let h_cond = conditional_entropy(&joint, LogBase::Bits).expect("entropy failed");
        assert!((h_cond - 1.0).abs() < EPSILON); // Each variable has 1 bit
    }

    #[test]
    fn test_conditional_entropy_dependent() {
        // Perfectly correlated: X = Y
        let joint = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5])
            .expect("array creation failed");
        let h_cond = conditional_entropy(&joint, LogBase::Bits).expect("entropy failed");
        assert!(h_cond.abs() < EPSILON); // H(X|Y) = 0 for perfect correlation
    }

    #[test]
    fn test_cross_entropy() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);

        // H(P,P) = H(P)
        let h_self = cross_entropy(&p, &q, LogBase::Bits).expect("cross entropy failed");
        let h_shannon = shannon_entropy(&p, LogBase::Bits).expect("shannon entropy failed");
        assert!((h_self - h_shannon).abs() < EPSILON);

        // H(P,Q) > H(P) when P != Q
        let q2 = Array1::from_vec(vec![0.4, 0.6]);
        let h_cross = cross_entropy(&p, &q2, LogBase::Bits).expect("cross entropy failed");
        assert!(h_cross > h_shannon);
    }

    #[test]
    fn test_renyi_entropy_alpha_2() {
        // α = 2 (collision entropy)
        let probs = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let h2 = renyi_entropy(&probs, 2.0, LogBase::Bits).expect("renyi entropy failed");

        // H_2 = -log2(Σ p^2)
        let sum_p2: f64 = probs.iter().map(|&p| p * p).sum();
        let expected = -sum_p2.log2();
        assert!((h2 - expected).abs() < EPSILON);
    }

    #[test]
    fn test_renyi_entropy_alpha_0() {
        // α = 0 (Hartley entropy)
        let probs = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let h0 = renyi_entropy(&probs, 0.0, LogBase::Bits).expect("renyi entropy failed");

        // H_0 = log2(n) for n non-zero probabilities
        let expected = (3.0_f64).log2();
        assert!((h0 - expected).abs() < EPSILON);
    }

    #[test]
    fn test_renyi_entropy_alpha_infinity() {
        // α → ∞ (min-entropy)
        let probs = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let h_inf =
            renyi_entropy(&probs, f64::INFINITY, LogBase::Bits).expect("renyi entropy failed");

        // H_∞ = -log2(max p)
        let expected = -(0.5_f64).log2();
        assert!((h_inf - expected).abs() < EPSILON);
    }

    #[test]
    fn test_renyi_entropy_limit_to_shannon() {
        // As α → 1, Rényi entropy → Shannon entropy
        let probs = Array1::from_vec(vec![0.5, 0.3, 0.2]);

        let h_shannon = shannon_entropy(&probs, LogBase::Bits).expect("shannon entropy failed");
        let h_renyi_low =
            renyi_entropy(&probs, 0.9999, LogBase::Bits).expect("renyi entropy failed");
        let h_renyi_high =
            renyi_entropy(&probs, 1.0001, LogBase::Bits).expect("renyi entropy failed");

        assert!((h_renyi_low - h_shannon).abs() < 0.01);
        assert!((h_renyi_high - h_shannon).abs() < 0.01);
    }

    #[test]
    fn test_differential_entropy_uniform() {
        // Uniform distribution on [0, 1] has h(X) = 0 nats
        let n = 1000;
        let pdf = Array1::from_elem(n, 1.0); // PDF = 1 for uniform on [0,1]
        let dx = 1.0 / (n as f64);
        let h = differential_entropy(&pdf, dx, LogBase::Nats).expect("differential entropy failed");
        assert!(h.abs() < 0.01); // Should be close to 0
    }

    #[test]
    fn test_differential_entropy_gaussian() {
        // Gaussian has h(X) = 0.5 * log(2πeσ²)
        // For σ = 1: h(X) = 0.5 * log(2πe) ≈ 1.4189 nats
        let n = 1000;
        let sigma = 1.0;
        let x_vals: Vec<f64> = (0..n)
            .map(|i| -5.0 + 10.0 * (i as f64) / (n as f64))
            .collect();
        let pdf: Array1<f64> = Array1::from_vec(
            x_vals
                .iter()
                .map(|&x| {
                    let z = x / sigma;
                    (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()
                })
                .collect(),
        );
        let dx = 10.0 / (n as f64);
        let h = differential_entropy(&pdf, dx, LogBase::Nats).expect("differential entropy failed");

        let expected = 0.5 * (2.0 * std::f64::consts::PI * E).ln();
        assert!((h - expected).abs() < 0.1); // Approximate due to discretization
    }

    #[test]
    fn test_entropy_errors() {
        // Empty array
        let empty: Array1<f64> = Array1::from_vec(vec![]);
        assert!(shannon_entropy(&empty, LogBase::Bits).is_err());

        // Negative probability
        let negative = Array1::from_vec(vec![0.5, -0.1, 0.6]);
        assert!(shannon_entropy(&negative, LogBase::Bits).is_err());

        // Dimension mismatch in cross-entropy
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.3, 0.3, 0.4]);
        assert!(cross_entropy(&p, &q, LogBase::Bits).is_err());

        // Invalid alpha for Rényi
        let probs = Array1::from_vec(vec![0.5, 0.5]);
        assert!(renyi_entropy(&probs, -1.0, LogBase::Bits).is_err());

        // Invalid dx for differential entropy
        let pdf = Array1::from_vec(vec![1.0, 1.0]);
        assert!(differential_entropy(&pdf, -0.1, LogBase::Bits).is_err());
    }
}
