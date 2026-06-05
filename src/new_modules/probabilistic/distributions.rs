//! # Extended Probability Distributions
//!
//! This module provides extended probability distributions beyond the standard distributions
//! available in `scirs2_core::random`. All distributions strictly follow SCIRS2 policy compliance.
//!
//! ## Available Distributions
//!
//! - **Beta Distribution**: Beta(α, β) - Continuous distribution on [0, 1]
//! - **Gamma Distribution**: Gamma(α, β) - Continuous distribution on (0, ∞)
//! - **Dirichlet Distribution**: Dir(α) - Multivariate generalization of Beta
//! - **Student's t Distribution**: t(ν) - Heavy-tailed alternative to normal
//! - **Wishart Distribution**: Wishart(V, n) - Distribution over positive definite matrices
//! - **Inverse-Wishart Distribution**: IW(Ψ, ν) - Conjugate prior for covariance matrices
//! - **Von Mises Distribution**: VM(μ, κ) - Circular normal distribution
//! - **Log-Normal Distribution**: LogNormal(μ, σ) - Distribution of exp(Normal)
//!
//! ## SCIRS2 Policy Compliance
//!
//! - All RNG operations use `scirs2_core::random` (NEVER direct rand/rand_distr)
//! - All array operations use `scirs2_core::ndarray` (NEVER direct ndarray)
//! - All statistical functions use `scirs2_stats` where applicable
//!
//! ## Mathematical Background
//!
//! ### Beta Distribution
//!
//! The Beta distribution Beta(α, β) has probability density function:
//!
//! ```text
//! f(x; α, β) = (1/B(α,β)) x^(α-1) (1-x)^(β-1)
//! ```
//!
//! where B(α,β) = Γ(α)Γ(β)/Γ(α+β) is the Beta function.
//!
//! ### Gamma Distribution
//!
//! The Gamma distribution Gamma(α, β) (shape-rate parameterization) has PDF:
//!
//! ```text
//! f(x; α, β) = (β^α / Γ(α)) x^(α-1) exp(-βx)
//! ```
//!
//! ### Dirichlet Distribution
//!
//! The Dirichlet distribution Dir(α) is a multivariate distribution over the probability simplex:
//!
//! ```text
//! f(x; α) = (1/B(α)) ∏_i x_i^(α_i - 1)
//! ```
//!
//! where ∑_i x_i = 1 and B(α) is the multivariate Beta function.

use crate::array::Array;
use crate::new_modules::probabilistic::{
    validate_non_negative, validate_positive, ProbabilisticError, Result,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Distribution, Rng, RngExt};
use std::f64::consts::PI;

/// Beta distribution Beta(α, β)
///
/// The Beta distribution is a continuous probability distribution on [0, 1]
/// defined by two positive shape parameters α and β.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::new_modules::probabilistic::BetaDistribution;
/// use scirs2_core::random::default_rng;
///
/// let beta = BetaDistribution::new(2.0, 5.0)?;
/// let mut rng = default_rng();
/// let sample = beta.sample(&mut rng)?;
/// ```
#[derive(Debug, Clone)]
pub struct BetaDistribution {
    /// Shape parameter α > 0
    alpha: f64,
    /// Shape parameter β > 0
    beta: f64,
    /// Cached normalization constant: log B(α, β)
    log_beta: f64,
}

impl BetaDistribution {
    /// Create a new Beta distribution with parameters α and β
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter α (must be > 0)
    /// * `beta` - Shape parameter β (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(BetaDistribution)` if parameters are valid
    /// * `Err(ProbabilisticError)` if parameters are invalid
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        validate_positive(alpha, "alpha")?;
        validate_positive(beta, "beta")?;

        // Compute log Beta function: log B(α, β) = log Γ(α) + log Γ(β) - log Γ(α+β)
        let log_beta = gamma_ln(alpha) + gamma_ln(beta) - gamma_ln(alpha + beta);

        Ok(Self {
            alpha,
            beta,
            log_beta,
        })
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the beta parameter
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Probability density function
    ///
    /// Computes f(x; α, β) = (1/B(α,β)) x^(α-1) (1-x)^(β-1)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the PDF (must be in [0, 1])
    ///
    /// # Returns
    ///
    /// * PDF value at x
    pub fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 {
            return if self.alpha < 1.0 {
                f64::INFINITY
            } else if self.alpha == 1.0 {
                self.beta
            } else {
                0.0
            };
        }
        if x == 1.0 {
            return if self.beta < 1.0 {
                f64::INFINITY
            } else if self.beta == 1.0 {
                self.alpha
            } else {
                0.0
            };
        }

        (self.log_pdf(x)).exp()
    }

    /// Log probability density function
    ///
    /// Computes log f(x; α, β) for numerical stability
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the log PDF
    ///
    /// # Returns
    ///
    /// * Log PDF value at x
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 || x == 1.0 {
            return f64::NEG_INFINITY; // Handle boundary carefully in practice
        }

        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - self.log_beta
    }

    /// Sample from the Beta distribution
    ///
    /// Uses the fact that if X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then
    /// X/(X+Y) ~ Beta(α, β)
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator (SCIRS2 compliant)
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Sample from Beta(α, β)
    /// * `Err(ProbabilisticError)` - If sampling fails
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<f64> {
        // Use gamma distribution to sample Beta
        let gamma1 = GammaDistribution::new(self.alpha, 1.0)?;
        let gamma2 = GammaDistribution::new(self.beta, 1.0)?;

        let x = gamma1.sample(rng)?;
        let y = gamma2.sample(rng)?;

        Ok(x / (x + y))
    }

    /// Mean of the distribution
    ///
    /// E\[X\] = α / (α + β)
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the distribution
    ///
    /// Var\[X\] = αβ / ((α+β)²(α+β+1))
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }
}

/// Gamma distribution Gamma(α, β) with shape-rate parameterization
///
/// The Gamma distribution is a continuous probability distribution on (0, ∞).
///
/// # Parameterization
///
/// This uses the shape-rate parameterization where:
/// - α is the shape parameter
/// - β is the rate parameter (inverse of scale)
///
/// Mean = α/β, Variance = α/β²
#[derive(Debug, Clone)]
pub struct GammaDistribution {
    /// Shape parameter α > 0
    alpha: f64,
    /// Rate parameter β > 0 (inverse scale)
    beta: f64,
    /// Cached log normalization constant
    log_norm: f64,
}

impl GammaDistribution {
    /// Create a new Gamma distribution
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter (must be > 0)
    /// * `beta` - Rate parameter (must be > 0)
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        validate_positive(alpha, "alpha")?;
        validate_positive(beta, "beta")?;

        let log_norm = alpha * beta.ln() - gamma_ln(alpha);

        Ok(Self {
            alpha,
            beta,
            log_norm,
        })
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the beta parameter
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.log_pdf(x).exp()
    }

    /// Log probability density function
    pub fn log_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        self.log_norm + (self.alpha - 1.0) * x.ln() - self.beta * x
    }

    /// Sample from Gamma distribution using Marsaglia-Tsang method
    ///
    /// Reference: Marsaglia, G., & Tsang, W. W. (2000). A simple method for generating
    /// gamma variables. ACM Transactions on Mathematical Software, 26(3), 363-372.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<f64> {
        // For α < 1, use the transformation Gamma(α,β) = Gamma(α+1,β) * U^(1/α)
        let (alpha_adj, correction) = if self.alpha < 1.0 {
            (self.alpha + 1.0, rng.random::<f64>().powf(1.0 / self.alpha))
        } else {
            (self.alpha, 1.0)
        };

        // Marsaglia-Tsang method for α >= 1
        let d = alpha_adj - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let z = sample_standard_normal(rng);
            let v = (1.0 + c * z).powi(3);

            if v <= 0.0 {
                continue;
            }

            let u: f64 = rng.random();
            let z2 = z * z;

            // Acceptance test
            if u < 1.0 - 0.0331 * z2 * z2 {
                return Ok(d * v * correction / self.beta);
            }

            if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
                return Ok(d * v * correction / self.beta);
            }
        }
    }

    /// Mean of the distribution
    pub fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Variance of the distribution
    pub fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }
}

/// Dirichlet distribution Dir(α)
///
/// The Dirichlet distribution is a multivariate probability distribution over the probability
/// simplex. It is a conjugate prior for the categorical/multinomial distribution.
#[derive(Debug, Clone)]
pub struct DirichletDistribution {
    /// Concentration parameters α_i > 0
    alpha: Vec<f64>,
    /// Cached sum of alphas
    alpha_sum: f64,
}

impl DirichletDistribution {
    /// Create a new Dirichlet distribution
    ///
    /// # Arguments
    ///
    /// * `alpha` - Concentration parameters (all must be > 0)
    pub fn new(alpha: Vec<f64>) -> Result<Self> {
        if alpha.is_empty() {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "alpha".to_string(),
                message: "alpha vector cannot be empty".to_string(),
            });
        }

        for (i, &a) in alpha.iter().enumerate() {
            validate_positive(a, &format!("alpha[{}]", i))?;
        }

        let alpha_sum = alpha.iter().sum();

        Ok(Self { alpha, alpha_sum })
    }

    /// Get the alpha parameters
    pub fn alpha(&self) -> &Vec<f64> {
        &self.alpha
    }

    /// Log probability density function
    ///
    /// Computes log f(x; α) where x is on the probability simplex
    pub fn log_pdf(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.alpha.len() {
            return Err(ProbabilisticError::DimensionMismatch {
                expected: vec![self.alpha.len()],
                actual: vec![x.len()],
                operation: "Dirichlet log_pdf".to_string(),
            });
        }

        // Check simplex constraint
        let sum: f64 = x.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "x".to_string(),
                message: format!("x must sum to 1, got sum = {}", sum),
            });
        }

        // Check all elements are non-negative
        for (i, &xi) in x.iter().enumerate() {
            if xi < 0.0 {
                return Err(ProbabilisticError::InvalidParameter {
                    parameter: format!("x[{}]", i),
                    message: "elements must be non-negative".to_string(),
                });
            }
        }

        // Compute log PDF
        let mut log_prob = gamma_ln(self.alpha_sum);
        for i in 0..self.alpha.len() {
            log_prob -= gamma_ln(self.alpha[i]);
            if x[i] > 0.0 {
                log_prob += (self.alpha[i] - 1.0) * x[i].ln();
            } else if self.alpha[i] > 1.0 {
                return Ok(f64::NEG_INFINITY);
            }
        }

        Ok(log_prob)
    }

    /// Sample from Dirichlet distribution
    ///
    /// Uses the property that if X_i ~ Gamma(α_i, 1) independently, then
    /// (X_1/S, ..., X_k/S) ~ Dir(α) where S = ∑X_i
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<Vec<f64>> {
        let mut samples = Vec::with_capacity(self.alpha.len());
        let mut sum = 0.0;

        // Sample from Gamma distributions
        for &alpha_i in &self.alpha {
            let gamma = GammaDistribution::new(alpha_i, 1.0)?;
            let x = gamma.sample(rng)?;
            samples.push(x);
            sum += x;
        }

        // Normalize to sum to 1
        for sample in &mut samples {
            *sample /= sum;
        }

        Ok(samples)
    }

    /// Mean of the distribution
    ///
    /// E\[X_i\] = α_i / ∑α_j
    pub fn mean(&self) -> Vec<f64> {
        self.alpha.iter().map(|&a| a / self.alpha_sum).collect()
    }
}

/// Student's t-distribution t(ν)
///
/// The Student's t-distribution is a continuous probability distribution that generalizes
/// the standard normal distribution. It has heavier tails than the normal distribution.
#[derive(Debug, Clone)]
pub struct StudentTDistribution {
    /// Degrees of freedom ν > 0
    nu: f64,
    /// Cached normalization constant
    log_norm: f64,
}

impl StudentTDistribution {
    /// Create a new Student's t-distribution
    ///
    /// # Arguments
    ///
    /// * `nu` - Degrees of freedom (must be > 0)
    pub fn new(nu: f64) -> Result<Self> {
        validate_positive(nu, "nu")?;

        // Log normalization: log(Γ((ν+1)/2) / (√(νπ) Γ(ν/2)))
        let log_norm = gamma_ln((nu + 1.0) / 2.0) - 0.5 * (nu * PI).ln() - gamma_ln(nu / 2.0);

        Ok(Self { nu, log_norm })
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Log probability density function
    pub fn log_pdf(&self, x: f64) -> f64 {
        self.log_norm - ((self.nu + 1.0) / 2.0) * (1.0 + x * x / self.nu).ln()
    }

    /// Sample from Student's t-distribution
    ///
    /// Uses the representation t(ν) = Z / √(V/ν) where Z ~ N(0,1) and V ~ χ²(ν)
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<f64> {
        let z = sample_standard_normal(rng);
        let chi_squared = GammaDistribution::new(self.nu / 2.0, 0.5)?;
        let v = chi_squared.sample(rng)?;

        Ok(z / (v / self.nu).sqrt())
    }

    /// Mean of the distribution (undefined for ν ≤ 1)
    pub fn mean(&self) -> Option<f64> {
        if self.nu > 1.0 {
            Some(0.0)
        } else {
            None
        }
    }

    /// Variance of the distribution (undefined for ν ≤ 2)
    pub fn variance(&self) -> Option<f64> {
        if self.nu > 2.0 {
            Some(self.nu / (self.nu - 2.0))
        } else {
            None
        }
    }
}

/// Log-Normal distribution LogNormal(μ, σ)
///
/// If X ~ Normal(μ, σ²), then Y = exp(X) ~ LogNormal(μ, σ)
#[derive(Debug, Clone)]
pub struct LogNormalDistribution {
    /// Location parameter μ (mean of underlying normal)
    mu: f64,
    /// Scale parameter σ > 0 (std dev of underlying normal)
    sigma: f64,
}

impl LogNormalDistribution {
    /// Create a new Log-Normal distribution
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        validate_positive(sigma, "sigma")?;
        Ok(Self { mu, sigma })
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.log_pdf(x).exp()
    }

    /// Log probability density function
    pub fn log_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let log_x = x.ln();
        -log_x
            - 0.5 * (2.0 * PI).ln()
            - self.sigma.ln()
            - 0.5 * ((log_x - self.mu) / self.sigma).powi(2)
    }

    /// Sample from Log-Normal distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<f64> {
        let z = sample_standard_normal(rng);
        Ok((self.mu + self.sigma * z).exp())
    }

    /// Mean of the distribution
    pub fn mean(&self) -> f64 {
        (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }

    /// Variance of the distribution
    pub fn variance(&self) -> f64 {
        let exp_sigma2 = (self.sigma * self.sigma).exp();
        (2.0 * self.mu).exp() * exp_sigma2 * (exp_sigma2 - 1.0)
    }
}

/// Von Mises distribution VM(μ, κ)
///
/// The Von Mises distribution is a continuous probability distribution on the circle
/// (circular normal distribution). It is used for modeling circular/directional data.
#[derive(Debug, Clone)]
pub struct VonMisesDistribution {
    /// Mean direction μ ∈ [0, 2π)
    mu: f64,
    /// Concentration parameter κ ≥ 0
    kappa: f64,
}

impl VonMisesDistribution {
    /// Create a new Von Mises distribution
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean direction in radians
    /// * `kappa` - Concentration parameter (≥ 0)
    pub fn new(mu: f64, kappa: f64) -> Result<Self> {
        validate_non_negative(kappa, "kappa")?;

        // Normalize mu to [0, 2π)
        let mu_normalized = mu.rem_euclid(2.0 * PI);

        Ok(Self {
            mu: mu_normalized,
            kappa,
        })
    }

    /// Probability density function
    ///
    /// f(x; μ, κ) = exp(κ cos(x - μ)) / (2π I₀(κ))
    ///
    /// where I₀ is the modified Bessel function of the first kind of order 0
    pub fn pdf(&self, x: f64) -> f64 {
        let x_normalized = x.rem_euclid(2.0 * PI);
        let i0_kappa = bessel_i0(self.kappa);
        (self.kappa * (x_normalized - self.mu).cos()).exp() / (2.0 * PI * i0_kappa)
    }

    /// Log probability density function
    pub fn log_pdf(&self, x: f64) -> f64 {
        let x_normalized = x.rem_euclid(2.0 * PI);
        let log_i0_kappa = bessel_i0(self.kappa).ln();
        self.kappa * (x_normalized - self.mu).cos() - (2.0 * PI).ln() - log_i0_kappa
    }

    /// Sample from Von Mises distribution using Best-Fisher algorithm
    ///
    /// Reference: Best, D. J., & Fisher, N. I. (1979). Efficient simulation of the
    /// von Mises distribution. Applied Statistics, 152-157.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<f64> {
        if self.kappa < 1e-6 {
            // For small kappa, distribution is nearly uniform
            return Ok(rng.random::<f64>() * 2.0 * PI);
        }

        // Best-Fisher algorithm
        let tau = 1.0 + (1.0 + 4.0 * self.kappa * self.kappa).sqrt();
        let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * self.kappa);
        let r = (1.0 + rho * rho) / (2.0 * rho);

        loop {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let u3: f64 = rng.random();

            // Guard against division by zero
            if u2 < 1e-10 {
                continue;
            }

            let z = (u1 - 0.5) / u2;
            let f = (1.0 + r * z) / (r + z);

            // Check if f is valid
            if !f.is_finite() || !(-1.0..=1.0).contains(&f) {
                continue;
            }

            let c = self.kappa * (r - f);

            // Best-Fisher acceptance criterion (fixed typo: c.ln() - c.ln() -> c.ln() - c)
            if c * (2.0 - c) - u3 > 0.0 || (c > 0.0 && c.ln() - c + 1.0 - u3 >= 0.0) {
                let theta = self.mu + f.acos() * if rng.random::<bool>() { 1.0 } else { -1.0 };
                return Ok(theta.rem_euclid(2.0 * PI));
            }
        }
    }

    /// Mean direction
    pub fn mean_direction(&self) -> f64 {
        self.mu
    }

    /// Circular variance
    ///
    /// Var = 1 - I₁(κ)/I₀(κ)
    ///
    /// where I₁ and I₀ are modified Bessel functions
    pub fn circular_variance(&self) -> f64 {
        1.0 - bessel_i1(self.kappa) / bessel_i0(self.kappa)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Log-gamma function — natural logarithm of the Gamma function: ln Γ(x)
///
/// Delegates to `scirs2_special::loggamma` for positive arguments, which is a
/// highly accurate (≈ 15-digit) Lanczos-based implementation.  For x ≤ 0 the
/// Euler reflection formula is used:
///
///   ln Γ(x) = ln(π / sin(π x)) − ln Γ(1 − x)
///
/// This covers the full real domain (excluding non-positive integers where the
/// function has poles).
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        // Reflection formula; returns NaN / ±Inf at the poles (integer ≤ 0)
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == 0.0 {
            return f64::INFINITY; // pole
        }
        return (pi / sin_pi_x.abs()).ln() - gamma_ln(1.0 - x);
    }

    // For x in (0, 0.5) the reflection formula gives better numerical
    // conditioning than direct evaluation at small arguments.
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        if sin_pi_x == 0.0 {
            return f64::INFINITY;
        }
        return (pi / sin_pi_x).ln() - gamma_ln(1.0 - x);
    }

    // Delegate to the Lanczos implementation in scirs2-special (g=7, 9 coefficients,
    // accurate to ~15 significant digits across the positive real line).
    scirs2_special::loggamma(x)
}

/// Sample from standard normal distribution N(0,1)
///
/// Uses Box-Muller transform with SCIRS2-compliant RNG
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;

    r * theta.cos()
}

/// Modified Bessel function of the first kind of order 0: I₀(x)
///
/// Uses series expansion for small x and asymptotic expansion for large x
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 3.75 {
        // Series expansion for small x
        let t = (x / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        // Asymptotic expansion for large x
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

/// Modified Bessel function of the first kind of order 1: I₁(x)
fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 3.75 {
        // Series expansion
        let t = (x / 3.75).powi(2);
        let result = ax
            * (0.5
                + t * (0.87890594
                    + t * (0.51498869
                        + t * (0.15084934
                            + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        if x < 0.0 {
            -result
        } else {
            result
        }
    } else {
        // Asymptotic expansion
        let t = 3.75 / ax;
        let result = (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (-0.03988024
                    + t * (-0.00362018
                        + t * (0.00163801
                            + t * (-0.01031555
                                + t * (0.02282967
                                    + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))))))));
        if x < 0.0 {
            -result
        } else {
            result
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_beta_distribution_creation() {
        let beta = BetaDistribution::new(2.0, 5.0);
        assert!(beta.is_ok());

        let beta = BetaDistribution::new(-1.0, 5.0);
        assert!(beta.is_err());

        let beta = BetaDistribution::new(2.0, 0.0);
        assert!(beta.is_err());
    }

    #[test]
    fn test_beta_distribution_pdf() {
        let beta =
            BetaDistribution::new(2.0, 5.0).expect("test: valid Beta distribution parameters");

        // PDF should be 0 outside [0,1]
        assert_eq!(beta.pdf(-0.1), 0.0);
        assert_eq!(beta.pdf(1.1), 0.0);

        // PDF should be positive in (0,1)
        assert!(beta.pdf(0.5) > 0.0);
        assert!(beta.pdf(0.3) > 0.0);
    }

    #[test]
    fn test_beta_distribution_moments() {
        let beta =
            BetaDistribution::new(2.0, 5.0).expect("test: valid Beta distribution parameters");

        // Mean = α/(α+β) = 2/7
        assert_relative_eq!(beta.mean(), 2.0 / 7.0, epsilon = 1e-10);

        // Variance = αβ/((α+β)²(α+β+1))
        let expected_var = (2.0 * 5.0) / (49.0 * 8.0);
        assert_relative_eq!(beta.variance(), expected_var, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_distribution_sampling() {
        let beta =
            BetaDistribution::new(2.0, 5.0).expect("test: valid Beta distribution parameters");
        let mut rng = thread_rng();

        // Sample multiple times
        for _ in 0..100 {
            let sample = beta.sample(&mut rng).expect("test: valid Beta sample");
            assert!((0.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_gamma_distribution_creation() {
        let gamma = GammaDistribution::new(2.0, 1.0);
        assert!(gamma.is_ok());

        let gamma = GammaDistribution::new(0.0, 1.0);
        assert!(gamma.is_err());
    }

    #[test]
    fn test_gamma_distribution_pdf() {
        let gamma =
            GammaDistribution::new(2.0, 1.0).expect("test: valid Gamma distribution parameters");

        // PDF should be 0 for x <= 0
        assert_eq!(gamma.pdf(0.0), 0.0);
        assert_eq!(gamma.pdf(-1.0), 0.0);

        // PDF should be positive for x > 0
        assert!(gamma.pdf(1.0) > 0.0);
        assert!(gamma.pdf(2.0) > 0.0);
    }

    #[test]
    fn test_gamma_distribution_moments() {
        let gamma =
            GammaDistribution::new(2.0, 1.0).expect("test: valid Gamma distribution parameters");

        // Mean = α/β = 2/1 = 2
        assert_relative_eq!(gamma.mean(), 2.0, epsilon = 1e-10);

        // Variance = α/β² = 2/1 = 2
        assert_relative_eq!(gamma.variance(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_distribution_sampling() {
        let gamma =
            GammaDistribution::new(2.0, 1.0).expect("test: valid Gamma distribution parameters");
        let mut rng = thread_rng();

        // Sample multiple times
        for _ in 0..100 {
            let sample = gamma.sample(&mut rng).expect("test: valid Gamma sample");
            assert!(sample > 0.0);
        }
    }

    #[test]
    fn test_dirichlet_distribution_creation() {
        let dir = DirichletDistribution::new(vec![1.0, 2.0, 3.0]);
        assert!(dir.is_ok());

        let dir = DirichletDistribution::new(vec![1.0, -1.0, 3.0]);
        assert!(dir.is_err());

        let dir = DirichletDistribution::new(vec![]);
        assert!(dir.is_err());
    }

    #[test]
    fn test_dirichlet_distribution_mean() {
        let dir = DirichletDistribution::new(vec![1.0, 2.0, 3.0])
            .expect("test: valid Dirichlet distribution parameters");
        let mean = dir.mean();

        assert_eq!(mean.len(), 3);
        assert_relative_eq!(mean[0], 1.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 2.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(mean[2], 3.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirichlet_distribution_sampling() {
        let dir = DirichletDistribution::new(vec![1.0, 2.0, 3.0])
            .expect("test: valid Dirichlet distribution parameters");
        let mut rng = thread_rng();

        for _ in 0..100 {
            let sample = dir.sample(&mut rng).expect("test: valid Dirichlet sample");

            // Check simplex constraint
            let sum: f64 = sample.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

            // Check all elements non-negative
            for &x in &sample {
                assert!(x >= 0.0);
            }
        }
    }

    #[test]
    fn test_student_t_distribution_creation() {
        let t = StudentTDistribution::new(5.0);
        assert!(t.is_ok());

        let t = StudentTDistribution::new(0.0);
        assert!(t.is_err());
    }

    #[test]
    fn test_student_t_distribution_moments() {
        let t =
            StudentTDistribution::new(5.0).expect("test: valid Student-T distribution parameters");

        // Mean should be 0 for ν > 1
        assert_eq!(t.mean(), Some(0.0));

        // Variance = ν/(ν-2) for ν > 2
        assert_relative_eq!(
            t.variance().expect("test: variance exists for df>2"),
            5.0 / 3.0,
            epsilon = 1e-10
        );

        // For ν = 1 (Cauchy), mean is undefined
        let cauchy =
            StudentTDistribution::new(1.0).expect("test: valid Student-T distribution parameters");
        assert_eq!(cauchy.mean(), None);
        assert_eq!(cauchy.variance(), None);
    }

    #[test]
    fn test_student_t_distribution_sampling() {
        let t =
            StudentTDistribution::new(5.0).expect("test: valid Student-T distribution parameters");
        let mut rng = thread_rng();

        // Just verify sampling works
        for _ in 0..100 {
            let _sample = t.sample(&mut rng).expect("test: valid Student-T sample");
        }
    }

    #[test]
    fn test_lognormal_distribution_creation() {
        let ln = LogNormalDistribution::new(0.0, 1.0);
        assert!(ln.is_ok());

        let ln = LogNormalDistribution::new(0.0, -1.0);
        assert!(ln.is_err());
    }

    #[test]
    fn test_lognormal_distribution_pdf() {
        let ln = LogNormalDistribution::new(0.0, 1.0)
            .expect("test: valid LogNormal distribution parameters");

        // PDF should be 0 for x <= 0
        assert_eq!(ln.pdf(0.0), 0.0);
        assert_eq!(ln.pdf(-1.0), 0.0);

        // PDF should be positive for x > 0
        assert!(ln.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_lognormal_distribution_sampling() {
        let ln = LogNormalDistribution::new(0.0, 1.0)
            .expect("test: valid LogNormal distribution parameters");
        let mut rng = thread_rng();

        for _ in 0..100 {
            let sample = ln.sample(&mut rng).expect("test: valid LogNormal sample");
            assert!(sample > 0.0);
        }
    }

    #[test]
    fn test_von_mises_distribution_creation() {
        let vm = VonMisesDistribution::new(0.0, 1.0);
        assert!(vm.is_ok());

        let vm = VonMisesDistribution::new(0.0, -1.0);
        assert!(vm.is_err());
    }

    #[test]
    fn test_von_mises_distribution_pdf() {
        let vm = VonMisesDistribution::new(0.0, 1.0)
            .expect("test: valid VonMises distribution parameters");

        // PDF should be positive
        assert!(vm.pdf(0.0) > 0.0);
        assert!(vm.pdf(PI) > 0.0);
    }

    #[test]
    fn test_von_mises_distribution_sampling() {
        let vm = VonMisesDistribution::new(0.0, 1.0)
            .expect("test: valid VonMises distribution parameters");
        let mut rng = thread_rng();

        for _ in 0..100 {
            let sample = vm.sample(&mut rng).expect("test: valid VonMises sample");
            // Sample should be finite and when normalized should be in [0, 2π)
            assert!(sample.is_finite());
            let normalized = sample.rem_euclid(2.0 * PI);
            assert!((0.0..2.0 * PI).contains(&normalized));
        }
    }

    #[test]
    fn test_helper_functions() {
        // Test gamma_ln
        let g1 = gamma_ln(1.0);
        assert_relative_eq!(g1, 0.0, epsilon = 1e-6); // Γ(1) = 1, ln(1) = 0

        let g2 = gamma_ln(2.0);
        assert_relative_eq!(g2, 0.0, epsilon = 1e-6); // Γ(2) = 1, ln(1) = 0

        // Test bessel_i0
        let i0_0 = bessel_i0(0.0);
        assert_relative_eq!(i0_0, 1.0, epsilon = 1e-6); // I₀(0) = 1

        // Test bessel_i1
        let i1_0 = bessel_i1(0.0);
        assert_relative_eq!(i1_0, 0.0, epsilon = 1e-6); // I₁(0) = 0
    }

    #[test]
    fn test_sample_standard_normal() {
        let mut rng = thread_rng();

        // Test that we can sample
        for _ in 0..100 {
            let _sample = sample_standard_normal(&mut rng);
        }
    }

    /// High-precision verification of the Lanczos-based `gamma_ln` helper.
    ///
    /// Reference values (accurate to at least 16 digits):
    ///   ln Γ(0.5) = ln(√π)  ≈  0.572_364_942_924_700
    ///   ln Γ(1.0) = 0.0      (exact)
    ///   ln Γ(5.0) = ln(4!)  = ln(24)  ≈  3.178_053_830_347_946
    ///   ln Γ(10.0) = ln(9!) = ln(362880)  ≈  12.801_827_480_081_469
    #[test]
    fn test_gamma_ln_known_values() {
        // ln Γ(0.5) = 0.5 * ln(π)
        let expected_half = 0.5f64 * std::f64::consts::PI.ln();
        let computed_half = gamma_ln(0.5);
        assert!(
            (computed_half - expected_half).abs() < 1e-10,
            "ln Γ(0.5): expected {:.15}, got {:.15}, diff {:.2e}",
            expected_half,
            computed_half,
            (computed_half - expected_half).abs()
        );

        // ln Γ(1.0) = ln(1) = 0
        let computed_one = gamma_ln(1.0);
        assert!(
            computed_one.abs() < 1e-14,
            "ln Γ(1.0): expected 0.0, got {:.15e}",
            computed_one
        );

        // ln Γ(5.0) = ln(4!) = ln(24)
        let expected_five = 24.0f64.ln();
        let computed_five = gamma_ln(5.0);
        assert!(
            (computed_five - expected_five).abs() < 1e-10,
            "ln Γ(5.0): expected {:.15}, got {:.15}, diff {:.2e}",
            expected_five,
            computed_five,
            (computed_five - expected_five).abs()
        );

        // ln Γ(10.0) = ln(9!) = ln(362880)
        let expected_ten = 362880.0f64.ln();
        let computed_ten = gamma_ln(10.0);
        assert!(
            (computed_ten - expected_ten).abs() < 1e-10,
            "ln Γ(10.0): expected {:.15}, got {:.15}, diff {:.2e}",
            expected_ten,
            computed_ten,
            (computed_ten - expected_ten).abs()
        );
    }
}
