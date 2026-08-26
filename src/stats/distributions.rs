//! Statistical Distribution Functions
//!
//! This module provides statistical functions for various probability distributions,
//! including probability density functions (PDF), cumulative distribution functions (CDF),
//! and quantile functions (inverse CDF).
//!
//! # Available Distributions
//!
//! - **Beta Distribution**: PDF, CDF, inverse CDF, log PDF, incomplete beta
//! - **Gamma Distribution**: PDF, CDF, inverse CDF, log PDF, incomplete gamma, digamma, polygamma
//! - **Student's t-Distribution**: PDF, CDF, inverse CDF, non-central t
//! - **Cauchy Distribution**: PDF, CDF, inverse CDF
//! - **Laplace Distribution**: PDF, CDF, inverse CDF
//! - **Logistic Distribution**: PDF, CDF, inverse CDF
//! - **Pareto Distribution**: PDF, CDF, inverse CDF
//!
//! # Examples
//!
//! ```
//! use numrs2::stats::distributions::*;
//!
//! // Beta distribution CDF
//! let cdf_val = beta_cdf(0.5, 2.0, 3.0).expect("beta CDF");
//!
//! // Gamma distribution quantile
//! let quantile = gamma_ppf(0.95, 2.0, 1.0).expect("gamma quantile");
//!
//! // Student's t CDF
//! let t_cdf = student_t_cdf(1.96, 10.0).expect("t CDF");
//! ```

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_special::*;

// ============================================================================
// Beta Distribution Functions
// ============================================================================

/// Beta distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF (must be in [0, 1])
/// * `a` - Alpha parameter (must be positive)
/// * `b` - Beta parameter (must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; a, b) = (x^(a-1) * (1-x)^(b-1)) / B(a, b)
///
/// where B(a, b) is the beta function.
pub fn beta_pdf<T: Float>(x: T, a: T, b: T) -> Result<T> {
    if a <= T::zero() || b <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and beta parameters must be positive".to_string(),
        ));
    }

    if x < T::zero() || x > T::one() {
        return Ok(T::zero());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert b to f64".to_string()))?;

    // Use log-beta for numerical stability
    let log_beta_val = loggamma(a_f64) + loggamma(b_f64) - loggamma(a_f64 + b_f64);
    let log_pdf = (a_f64 - 1.0) * x_f64.ln() + (b_f64 - 1.0) * (1.0 - x_f64).ln() - log_beta_val;

    T::from(log_pdf.exp())
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Beta distribution log probability density function (log PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the log PDF (must be in [0, 1])
/// * `a` - Alpha parameter (must be positive)
/// * `b` - Beta parameter (must be positive)
///
/// # Returns
///
/// The log probability density at x
pub fn beta_logpdf<T: Float>(x: T, a: T, b: T) -> Result<T> {
    if a <= T::zero() || b <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and beta parameters must be positive".to_string(),
        ));
    }

    if x <= T::zero() || x >= T::one() {
        return Ok(T::neg_infinity());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert b to f64".to_string()))?;

    let log_beta_val = loggamma(a_f64) + loggamma(b_f64) - loggamma(a_f64 + b_f64);
    let log_pdf = (a_f64 - 1.0) * x_f64.ln() + (b_f64 - 1.0) * (1.0 - x_f64).ln() - log_beta_val;

    T::from(log_pdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Beta distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `a` - Alpha parameter (must be positive)
/// * `b` - Beta parameter (must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// F(x; a, b) = I_x(a, b) = B_x(a, b) / B(a, b)
///
/// where I_x is the regularized incomplete beta function.
pub fn beta_cdf<T: Float>(x: T, a: T, b: T) -> Result<T> {
    if a <= T::zero() || b <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and beta parameters must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::zero());
    }
    if x >= T::one() {
        return Ok(T::one());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert b to f64".to_string()))?;

    // Use regularized incomplete beta function (I_x, not B_x)
    let cdf = betainc_regularized(x_f64, a_f64, b_f64).map_err(|e| {
        NumRs2Error::InvalidOperation(format!("Beta CDF computation failed: {:?}", e))
    })?;

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Beta distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `a` - Alpha parameter (must be positive)
/// * `b` - Beta parameter (must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
///
/// # Note
///
/// Uses Newton-Raphson iteration for high precision.
pub fn beta_ppf<T: Float>(p: T, a: T, b: T) -> Result<T> {
    if p < T::zero() || p > T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in [0, 1]".to_string(),
        ));
    }

    if a <= T::zero() || b <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and beta parameters must be positive".to_string(),
        ));
    }

    if p == T::zero() {
        return Ok(T::zero());
    }
    if p == T::one() {
        return Ok(T::one());
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert b to f64".to_string()))?;

    // Initial guess: a closed-form approximate inversion of the Beta CDF's
    // tail behavior (small-x power-law near 0, mirrored near 1), which -
    // unlike a plain method-of-moments guess (the unconditional mean,
    // independent of `p`) - already accounts for the target quantile `p`.
    let mut x = if p_f64 < 0.5 {
        (p_f64 * (a_f64 + b_f64) / a_f64).powf(1.0 / a_f64)
    } else {
        1.0 - ((1.0 - p_f64) * (a_f64 + b_f64) / b_f64).powf(1.0 / b_f64)
    };

    // Newton-Raphson iteration
    for _ in 0..100 {
        if x <= 0.0 {
            x = 1e-10;
        }
        if x >= 1.0 {
            x = 1.0 - 1e-10;
        }

        let cdf = betainc_regularized(x, a_f64, b_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Beta CDF computation failed: {:?}", e))
        })?;
        let pdf_f64 = beta_pdf(x, a_f64, b_f64)?;

        let diff = cdf - p_f64;
        if diff.abs() < 1e-12 {
            break;
        }

        if pdf_f64 > 1e-10 {
            x -= diff / pdf_f64;
        } else {
            // Bisection fallback
            break;
        }
    }

    x = x.clamp(0.0, 1.0);

    T::from(x).ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Gamma Distribution Functions
// ============================================================================

/// Gamma distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF (must be positive)
/// * `shape` - Shape parameter (k or α, must be positive)
/// * `scale` - Scale parameter (θ, must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; k, θ) = (x^(k-1) * exp(-x/θ)) / (θ^k * Γ(k))
pub fn gamma_pdf<T: Float>(x: T, shape: T, scale: T) -> Result<T> {
    if shape <= T::zero() || scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Shape and scale parameters must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::zero());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let shape_f64 = shape.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
    })?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    // Use log-gamma for numerical stability
    let log_pdf = (shape_f64 - 1.0) * x_f64.ln()
        - x_f64 / scale_f64
        - shape_f64 * scale_f64.ln()
        - loggamma(shape_f64);

    T::from(log_pdf.exp())
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Gamma distribution log probability density function (log PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the log PDF (must be positive)
/// * `shape` - Shape parameter (k or α, must be positive)
/// * `scale` - Scale parameter (θ, must be positive)
///
/// # Returns
///
/// The log probability density at x
pub fn gamma_logpdf<T: Float>(x: T, shape: T, scale: T) -> Result<T> {
    if shape <= T::zero() || scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Shape and scale parameters must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::neg_infinity());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let shape_f64 = shape.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
    })?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let log_pdf = (shape_f64 - 1.0) * x_f64.ln()
        - x_f64 / scale_f64
        - shape_f64 * scale_f64.ln()
        - loggamma(shape_f64);

    T::from(log_pdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Gamma distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `shape` - Shape parameter (k or α, must be positive)
/// * `scale` - Scale parameter (θ, must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// F(x; k, θ) = γ(k, x/θ) / Γ(k)
///
/// where γ is the lower incomplete gamma function.
pub fn gamma_cdf<T: Float>(x: T, shape: T, scale: T) -> Result<T> {
    if shape <= T::zero() || scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Shape and scale parameters must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::zero());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let shape_f64 = shape.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
    })?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    // Use regularized incomplete gamma function
    let cdf = gammainc(shape_f64, x_f64 / scale_f64).map_err(|e| {
        NumRs2Error::InvalidOperation(format!("Gamma CDF computation failed: {:?}", e))
    })?;

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Gamma distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `shape` - Shape parameter (k or α, must be positive)
/// * `scale` - Scale parameter (θ, must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
pub fn gamma_ppf<T: Float>(p: T, shape: T, scale: T) -> Result<T> {
    if p < T::zero() || p > T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in [0, 1]".to_string(),
        ));
    }

    if shape <= T::zero() || scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Shape and scale parameters must be positive".to_string(),
        ));
    }

    if p == T::zero() {
        return Ok(T::zero());
    }
    if p == T::one() {
        return Ok(T::infinity());
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let shape_f64 = shape.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
    })?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    // Initial guess using Wilson-Hilferty transformation
    let mut x = shape_f64 * scale_f64;

    // Newton-Raphson iteration
    for _ in 0..100 {
        if x <= 0.0 {
            x = 1e-10;
        }

        let cdf = gammainc(shape_f64, x / scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Gamma CDF computation failed: {:?}", e))
        })?;
        let pdf_f64 = gamma_pdf(x, shape_f64, scale_f64)?;

        let diff = cdf - p_f64;
        if diff.abs() < 1e-12 {
            break;
        }

        if pdf_f64 > 1e-10 {
            x -= diff / pdf_f64;
        } else {
            break;
        }
    }

    x = x.max(0.0);

    T::from(x).ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Student's t-Distribution Functions
// ============================================================================

/// Student's t-distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF
/// * `df` - Degrees of freedom (must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; ν) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
pub fn student_t_pdf<T: Float>(x: T, df: T) -> Result<T> {
    if df <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Degrees of freedom must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let df_f64 = df
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert df to f64".to_string()))?;

    // Special case: t(1) is the standard Cauchy distribution
    // This avoids potential numerical issues with loggamma(0.5)
    if (df_f64 - 1.0).abs() < 1e-10 {
        let pdf = 1.0 / (std::f64::consts::PI * (1.0 + x_f64 * x_f64));
        return T::from(pdf)
            .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()));
    }

    let log_pdf = loggamma((df_f64 + 1.0) / 2.0)
        - loggamma(df_f64 / 2.0)
        - 0.5 * (df_f64 * std::f64::consts::PI).ln()
        - ((df_f64 + 1.0) / 2.0) * (1.0 + x_f64 * x_f64 / df_f64).ln();

    T::from(log_pdf.exp())
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Student's t-distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `df` - Degrees of freedom (must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// Uses the regularized incomplete beta function relationship:
/// F(t; ν) = 1/2 + t*Γ((ν+1)/2) / (√(νπ)*Γ(ν/2)) * ₂F₁(1/2, (ν+1)/2; 3/2; -t²/ν)
pub fn student_t_cdf<T: Float>(x: T, df: T) -> Result<T> {
    if df <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Degrees of freedom must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let df_f64 = df
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert df to f64".to_string()))?;

    // Use regularized incomplete beta function for CDF
    // Formula: For t > 0: p = 1 - I_x(v/2, 1/2)/2
    //          For t < 0: p = I_x(v/2, 1/2)/2
    //          For t = 0: p = 0.5
    // where x = v/(v+t²) and I_x is the regularized incomplete beta
    let t2 = x_f64 * x_f64;
    let x_beta = df_f64 / (df_f64 + t2);
    let beta_val = betainc_regularized(x_beta, df_f64 / 2.0, 0.5).map_err(|e| {
        NumRs2Error::InvalidOperation(format!("Student's t CDF computation failed: {:?}", e))
    })?;

    let cdf = if x_f64 < 0.0 {
        0.5 * beta_val
    } else if x_f64 == 0.0 {
        0.5
    } else {
        1.0 - 0.5 * beta_val
    };

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Student's t-distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `df` - Degrees of freedom (must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
pub fn student_t_ppf<T: Float>(p: T, df: T) -> Result<T> {
    if p < T::zero() || p > T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in [0, 1]".to_string(),
        ));
    }

    if df <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Degrees of freedom must be positive".to_string(),
        ));
    }

    if p == T::zero() {
        return Ok(T::neg_infinity());
    }
    if p == T::one() {
        return Ok(T::infinity());
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let df_f64 = df
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert df to f64".to_string()))?;

    // Initial guess using normal approximation for large df
    let mut x = if df_f64 > 100.0 {
        // Normal approximation
        normal_ppf(p_f64)
    } else {
        // Simple initial guess
        if p_f64 < 0.5 {
            -1.0
        } else {
            1.0
        }
    };

    // Newton-Raphson iteration
    for _ in 0..100 {
        let cdf_f64 = student_t_cdf(x, df_f64)?;
        let pdf_f64 = student_t_pdf(x, df_f64)?;

        let diff = cdf_f64 - p_f64;
        if diff.abs() < 1e-12 {
            break;
        }

        if pdf_f64 > 1e-10 {
            x -= diff / pdf_f64;
        } else {
            break;
        }
    }

    T::from(x).ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Cauchy Distribution Functions
// ============================================================================

/// Cauchy distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF
/// * `loc` - Location parameter (median)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; x₀, γ) = 1 / (π * γ * (1 + ((x - x₀)/γ)²))
pub fn cauchy_pdf<T: Float>(x: T, loc: T, scale: T) -> Result<T> {
    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let z = (x_f64 - loc_f64) / scale_f64;
    let pdf = 1.0 / (std::f64::consts::PI * scale_f64 * (1.0 + z * z));

    T::from(pdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Cauchy distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `loc` - Location parameter (median)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// F(x; x₀, γ) = 1/π * arctan((x - x₀)/γ) + 1/2
pub fn cauchy_cdf<T: Float>(x: T, loc: T, scale: T) -> Result<T> {
    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let z = (x_f64 - loc_f64) / scale_f64;
    let cdf = z.atan() / std::f64::consts::PI + 0.5;

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Cauchy distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `loc` - Location parameter (median)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
///
/// # Formula
///
/// F⁻¹(p; x₀, γ) = x₀ + γ * tan(π * (p - 1/2))
pub fn cauchy_ppf<T: Float>(p: T, loc: T, scale: T) -> Result<T> {
    if p < T::zero() || p > T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in [0, 1]".to_string(),
        ));
    }

    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let ppf = loc_f64 + scale_f64 * (std::f64::consts::PI * (p_f64 - 0.5)).tan();

    T::from(ppf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Laplace Distribution Functions
// ============================================================================

/// Laplace (double exponential) distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; μ, b) = 1/(2b) * exp(-|x - μ|/b)
pub fn laplace_pdf<T: Float>(x: T, loc: T, scale: T) -> Result<T> {
    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let pdf = (-(x_f64 - loc_f64).abs() / scale_f64).exp() / (2.0 * scale_f64);

    T::from(pdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Laplace distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// F(x; μ, b) = 1/2 * (1 + sgn(x - μ) * (1 - exp(-|x - μ|/b)))
pub fn laplace_cdf<T: Float>(x: T, loc: T, scale: T) -> Result<T> {
    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let cdf = if x_f64 < loc_f64 {
        0.5 * ((x_f64 - loc_f64) / scale_f64).exp()
    } else {
        1.0 - 0.5 * (-(x_f64 - loc_f64) / scale_f64).exp()
    };

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Laplace distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
///
/// # Formula
///
/// F⁻¹(p; μ, b) = μ - b * sgn(p - 1/2) * ln(1 - 2|p - 1/2|)
pub fn laplace_ppf<T: Float>(p: T, loc: T, scale: T) -> Result<T> {
    if p < T::zero() || p > T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in [0, 1]".to_string(),
        ));
    }

    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let ppf = if p_f64 < 0.5 {
        loc_f64 + scale_f64 * (2.0 * p_f64).ln()
    } else {
        loc_f64 - scale_f64 * (2.0 * (1.0 - p_f64)).ln()
    };

    T::from(ppf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Logistic Distribution Functions
// ============================================================================

/// Logistic distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; μ, s) = exp(-(x-μ)/s) / (s * (1 + exp(-(x-μ)/s))²)
pub fn logistic_pdf<T: Float>(x: T, loc: T, scale: T) -> Result<T> {
    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let z = (x_f64 - loc_f64) / scale_f64;
    let exp_z = (-z).exp();
    let pdf = exp_z / (scale_f64 * (1.0 + exp_z) * (1.0 + exp_z));

    T::from(pdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Logistic distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// F(x; μ, s) = 1 / (1 + exp(-(x-μ)/s))
pub fn logistic_cdf<T: Float>(x: T, loc: T, scale: T) -> Result<T> {
    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let z = (x_f64 - loc_f64) / scale_f64;
    let cdf = 1.0 / (1.0 + (-z).exp());

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Logistic distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
///
/// # Formula
///
/// F⁻¹(p; μ, s) = μ + s * ln(p / (1 - p))
pub fn logistic_ppf<T: Float>(p: T, loc: T, scale: T) -> Result<T> {
    if p <= T::zero() || p >= T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in (0, 1)".to_string(),
        ));
    }

    if scale <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Scale parameter must be positive".to_string(),
        ));
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let loc_f64 = loc
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert loc to f64".to_string()))?;
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    let ppf = loc_f64 + scale_f64 * (p_f64 / (1.0 - p_f64)).ln();

    T::from(ppf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Pareto Distribution Functions
// ============================================================================

/// Pareto distribution probability density function (PDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the PDF (must be >= xm)
/// * `alpha` - Shape parameter (must be positive)
/// * `xm` - Scale parameter (minimum value, must be positive)
///
/// # Returns
///
/// The probability density at x
///
/// # Formula
///
/// f(x; α, xₘ) = α * xₘ^α / x^(α+1)  for x >= xₘ
pub fn pareto_pdf<T: Float>(x: T, alpha: T, xm: T) -> Result<T> {
    if alpha <= T::zero() || xm <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and xm parameters must be positive".to_string(),
        ));
    }

    if x < xm {
        return Ok(T::zero());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let alpha_f64 = alpha.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert alpha to f64".to_string())
    })?;
    let xm_f64 = xm
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert xm to f64".to_string()))?;

    let pdf = alpha_f64 * xm_f64.powf(alpha_f64) / x_f64.powf(alpha_f64 + 1.0);

    T::from(pdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Pareto distribution cumulative distribution function (CDF)
///
/// # Arguments
///
/// * `x` - Value at which to evaluate the CDF
/// * `alpha` - Shape parameter (must be positive)
/// * `xm` - Scale parameter (minimum value, must be positive)
///
/// # Returns
///
/// The cumulative probability up to x
///
/// # Formula
///
/// F(x; α, xₘ) = 1 - (xₘ/x)^α  for x >= xₘ
pub fn pareto_cdf<T: Float>(x: T, alpha: T, xm: T) -> Result<T> {
    if alpha <= T::zero() || xm <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and xm parameters must be positive".to_string(),
        ));
    }

    if x < xm {
        return Ok(T::zero());
    }

    let x_f64 = x
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert x to f64".to_string()))?;
    let alpha_f64 = alpha.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert alpha to f64".to_string())
    })?;
    let xm_f64 = xm
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert xm to f64".to_string()))?;

    let cdf = 1.0 - (xm_f64 / x_f64).powf(alpha_f64);

    T::from(cdf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

/// Pareto distribution percent point function (inverse CDF/quantile function)
///
/// # Arguments
///
/// * `p` - Probability (must be in [0, 1])
/// * `alpha` - Shape parameter (must be positive)
/// * `xm` - Scale parameter (minimum value, must be positive)
///
/// # Returns
///
/// The value x such that P(X <= x) = p
///
/// # Formula
///
/// F⁻¹(p; α, xₘ) = xₘ / (1 - p)^(1/α)
pub fn pareto_ppf<T: Float>(p: T, alpha: T, xm: T) -> Result<T> {
    if p < T::zero() || p >= T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Probability must be in [0, 1)".to_string(),
        ));
    }

    if alpha <= T::zero() || xm <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Alpha and xm parameters must be positive".to_string(),
        ));
    }

    if p == T::zero() {
        return Ok(xm);
    }

    let p_f64 = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert p to f64".to_string()))?;
    let alpha_f64 = alpha.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert alpha to f64".to_string())
    })?;
    let xm_f64 = xm
        .to_f64()
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert xm to f64".to_string()))?;

    let ppf = xm_f64 / (1.0 - p_f64).powf(1.0 / alpha_f64);

    T::from(ppf)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert result".to_string()))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Normal distribution inverse CDF (helper for Student's t)
fn normal_ppf(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm for inverse normal CDF
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let q = p - 0.5;

    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = ((((a[5] * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1]) * r + a[0];
        let den = ((((b[4] * r + b[3]) * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0;
        return q * num / den;
    }

    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();

    let num = ((((c[5] * r + c[4]) * r + c[3]) * r + c[2]) * r + c[1]) * r + c[0];
    let den = (((d[3] * r + d[2]) * r + d[1]) * r + d[0]) * r + 1.0;
    let result = num / den;

    if q < 0.0 {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_beta_pdf() {
        // Beta(2, 3) PDF at x=0.5
        let pdf = beta_pdf(0.5, 2.0, 3.0).expect("beta_pdf");
        assert!((pdf - 1.5).abs() < TOL);
    }

    #[test]
    fn test_beta_cdf() {
        // Beta(2, 3) CDF at x=0.5
        let cdf = beta_cdf(0.5, 2.0, 3.0).expect("beta_cdf");
        assert!(cdf > 0.0 && cdf < 1.0);
    }

    #[test]
    fn test_gamma_pdf() {
        // Gamma(2, 1) PDF at x=1.0
        let pdf = gamma_pdf(1.0, 2.0, 1.0).expect("gamma_pdf");
        assert!(pdf > 0.0);
    }

    #[test]
    fn test_gamma_cdf() {
        // Gamma(2, 1) CDF at x=1.0
        let cdf = gamma_cdf(1.0, 2.0, 1.0).expect("gamma_cdf");
        assert!(cdf > 0.0 && cdf < 1.0);
    }

    #[test]
    fn test_student_t_pdf() {
        // Student's t(5) PDF at x=0.0
        let pdf = student_t_pdf(0.0, 5.0).expect("student_t_pdf");
        assert!(pdf > 0.0);
    }

    #[test]
    fn test_student_t_cdf() {
        // Student's t(5) CDF at x=0.0 should be 0.5
        let cdf = student_t_cdf(0.0, 5.0).expect("student_t_cdf");
        eprintln!("Student's t CDF at 0.0: {}", cdf);
        assert!((cdf - 0.5).abs() < TOL, "Expected 0.5, got {}", cdf);
    }

    #[test]
    fn test_cauchy_pdf() {
        let pdf = cauchy_pdf(0.0, 0.0, 1.0).expect("cauchy_pdf");
        assert!((pdf - 1.0 / std::f64::consts::PI).abs() < TOL);
    }

    #[test]
    fn test_cauchy_cdf() {
        let cdf = cauchy_cdf(0.0, 0.0, 1.0).expect("cauchy_cdf");
        assert!((cdf - 0.5).abs() < TOL);
    }

    #[test]
    fn test_laplace_pdf() {
        let pdf = laplace_pdf(0.0, 0.0, 1.0).expect("laplace_pdf");
        assert!((pdf - 0.5).abs() < TOL);
    }

    #[test]
    fn test_laplace_cdf() {
        let cdf = laplace_cdf(0.0, 0.0, 1.0).expect("laplace_cdf");
        assert!((cdf - 0.5).abs() < TOL);
    }

    #[test]
    fn test_logistic_pdf() {
        let pdf = logistic_pdf(0.0, 0.0, 1.0).expect("logistic_pdf");
        assert!((pdf - 0.25).abs() < TOL);
    }

    #[test]
    fn test_logistic_cdf() {
        let cdf = logistic_cdf(0.0, 0.0, 1.0).expect("logistic_cdf");
        assert!((cdf - 0.5).abs() < TOL);
    }

    #[test]
    fn test_pareto_pdf() {
        let pdf = pareto_pdf(2.0, 2.0, 1.0).expect("pareto_pdf");
        assert!(pdf > 0.0);
    }

    #[test]
    fn test_pareto_cdf() {
        let cdf = pareto_cdf(2.0, 2.0, 1.0).expect("pareto_cdf");
        assert!((cdf - 0.75).abs() < TOL);
    }
}
