//! # Bayesian Inference Utilities
//!
//! This module provides comprehensive Bayesian inference utilities including conjugate priors,
//! posterior computation, model comparison, credible intervals, and hypothesis testing.
//!
//! ## Features
//!
//! - **Conjugate Priors**: Beta-Binomial, Gamma-Poisson, Normal-Normal, Dirichlet-Multinomial
//! - **Posterior Computation**: Analytical and numerical methods
//! - **Model Comparison**: BIC, DIC, WAIC, LOO-CV, Bayes factors
//! - **Credible Intervals**: HPD intervals, equal-tailed intervals
//! - **Hypothesis Testing**: Bayes factors, posterior probabilities
//!
//! ## Mathematical Background
//!
//! ### Conjugate Priors
//!
//! A prior distribution p(θ) is conjugate to a likelihood p(D|θ) if the posterior
//! p(θ|D) is in the same distributional family as the prior.
//!
//! Examples:
//! - Beta is conjugate to Binomial
//! - Gamma is conjugate to Poisson
//! - Normal is conjugate to Normal (with known variance)
//! - Dirichlet is conjugate to Multinomial
//!
//! ### Model Comparison Criteria
//!
//! **BIC (Bayesian Information Criterion)**:
//! ```text
//! BIC = -2 log L + k log n
//! ```
//! where k is the number of parameters and n is the sample size.
//!
//! **DIC (Deviance Information Criterion)**:
//! ```text
//! DIC = D̄ + p_D
//! ```
//! where D̄ is the posterior mean deviance and p_D is the effective number of parameters.
//!
//! **WAIC (Widely Applicable Information Criterion)**:
//! ```text
//! WAIC = -2(lppd - p_WAIC)
//! ```
//! where lppd is the log pointwise predictive density and p_WAIC measures model complexity.
//!
//! ## SCIRS2 Policy Compliance
//!
//! - All statistical computations use `scirs2_stats`
//! - All array operations use `scirs2_core::ndarray`
//! - All RNG operations use `scirs2_core::random`

use crate::array::Array;
use crate::new_modules::probabilistic::{
    validate_positive, validate_probability, BetaDistribution, DirichletDistribution,
    GammaDistribution, ProbabilisticError, Result,
};
use scirs2_core::random::{thread_rng, Rng};
use std::f64::consts::PI;

// ============================================================================
// Conjugate Prior Distributions
// ============================================================================

/// Beta-Binomial conjugate prior
///
/// Prior: Beta(α, β)
/// Likelihood: Binomial(n, θ)
/// Posterior: Beta(α + k, β + n - k)
///
/// where k is the number of successes in n trials
#[derive(Debug, Clone)]
pub struct BetaBinomialConjugate {
    /// Prior Beta distribution
    prior: BetaDistribution,
}

impl BetaBinomialConjugate {
    /// Create new Beta-Binomial conjugate prior
    ///
    /// # Arguments
    ///
    /// * `alpha` - Prior parameter α (pseudo-successes)
    /// * `beta` - Prior parameter β (pseudo-failures)
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        let prior = BetaDistribution::new(alpha, beta)?;
        Ok(Self { prior })
    }

    /// Compute posterior distribution given data
    ///
    /// # Arguments
    ///
    /// * `n_successes` - Number of observed successes
    /// * `n_trials` - Total number of trials
    ///
    /// # Returns
    ///
    /// Posterior Beta distribution
    pub fn update(&self, n_successes: usize, n_trials: usize) -> Result<BetaDistribution> {
        if n_successes > n_trials {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "n_successes".to_string(),
                message: "n_successes cannot exceed n_trials".to_string(),
            });
        }

        let alpha_post = self.prior.alpha() + n_successes as f64;
        let beta_post = self.prior.beta() + (n_trials - n_successes) as f64;

        BetaDistribution::new(alpha_post, beta_post)
    }

    /// Compute posterior predictive probability
    ///
    /// P(next trial is success | data) = (α + k) / (α + β + n)
    pub fn posterior_predictive(&self, n_successes: usize, n_trials: usize) -> Result<f64> {
        let posterior = self.update(n_successes, n_trials)?;
        Ok(posterior.mean())
    }
}

/// Gamma-Poisson conjugate prior
///
/// Prior: Gamma(α, β)
/// Likelihood: Poisson(λ)
/// Posterior: Gamma(α + ∑x_i, β + n)
#[derive(Debug, Clone)]
pub struct GammaPoissonConjugate {
    /// Prior Gamma distribution
    prior: GammaDistribution,
}

impl GammaPoissonConjugate {
    /// Create new Gamma-Poisson conjugate prior
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter (prior total count)
    /// * `beta` - Rate parameter (prior number of observations)
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        let prior = GammaDistribution::new(alpha, beta)?;
        Ok(Self { prior })
    }

    /// Compute posterior distribution given data
    ///
    /// # Arguments
    ///
    /// * `data` - Observed Poisson counts
    pub fn update(&self, data: &[f64]) -> Result<GammaDistribution> {
        let sum_x: f64 = data.iter().sum();
        let n = data.len() as f64;

        let alpha_post = self.prior.alpha() + sum_x;
        let beta_post = self.prior.beta() + n;

        GammaDistribution::new(alpha_post, beta_post)
    }

    /// Posterior predictive distribution for next observation
    pub fn posterior_predictive_mean(&self, data: &[f64]) -> Result<f64> {
        let posterior = self.update(data)?;
        Ok(posterior.mean())
    }
}

/// Normal-Normal conjugate prior (known variance)
///
/// Prior: N(μ₀, σ₀²)
/// Likelihood: N(μ, σ²) with known σ²
/// Posterior: N(μ_n, σ_n²)
///
/// where:
/// - μ_n = (σ²μ₀ + nσ₀²x̄) / (σ² + nσ₀²)
/// - σ_n² = σ²σ₀² / (σ² + nσ₀²)
#[derive(Debug, Clone)]
pub struct NormalNormalConjugate {
    /// Prior mean μ₀
    prior_mean: f64,
    /// Prior variance σ₀²
    prior_variance: f64,
    /// Known likelihood variance σ²
    likelihood_variance: f64,
}

impl NormalNormalConjugate {
    /// Create new Normal-Normal conjugate prior
    pub fn new(prior_mean: f64, prior_variance: f64, likelihood_variance: f64) -> Result<Self> {
        validate_positive(prior_variance, "prior_variance")?;
        validate_positive(likelihood_variance, "likelihood_variance")?;

        Ok(Self {
            prior_mean,
            prior_variance,
            likelihood_variance,
        })
    }

    /// Compute posterior distribution given data
    pub fn update(&self, data: &[f64]) -> Result<(f64, f64)> {
        if data.is_empty() {
            return Ok((self.prior_mean, self.prior_variance));
        }

        let n = data.len() as f64;
        let sample_mean = data.iter().sum::<f64>() / n;

        // Compute posterior mean
        let precision_prior = 1.0 / self.prior_variance;
        let precision_likelihood = n / self.likelihood_variance;
        let precision_post = precision_prior + precision_likelihood;

        let mean_post = (precision_prior * self.prior_mean + precision_likelihood * sample_mean)
            / precision_post;

        let variance_post = 1.0 / precision_post;

        Ok((mean_post, variance_post))
    }

    /// Compute 95% credible interval
    pub fn credible_interval_95(&self, data: &[f64]) -> Result<(f64, f64)> {
        let (mean, variance) = self.update(data)?;
        let std = variance.sqrt();
        // 95% CI for normal: μ ± 1.96σ
        Ok((mean - 1.96 * std, mean + 1.96 * std))
    }
}

/// Dirichlet-Multinomial conjugate prior
///
/// Prior: Dir(α)
/// Likelihood: Multinomial(n, θ)
/// Posterior: Dir(α + counts)
#[derive(Debug, Clone)]
pub struct DirichletMultinomialConjugate {
    /// Prior Dirichlet distribution
    prior: DirichletDistribution,
}

impl DirichletMultinomialConjugate {
    /// Create new Dirichlet-Multinomial conjugate prior
    pub fn new(alpha: Vec<f64>) -> Result<Self> {
        let prior = DirichletDistribution::new(alpha)?;
        Ok(Self { prior })
    }

    /// Compute posterior distribution given counts
    ///
    /// # Arguments
    ///
    /// * `counts` - Observed counts for each category
    pub fn update(&self, counts: &[usize]) -> Result<DirichletDistribution> {
        if counts.len() != self.prior.alpha().len() {
            return Err(ProbabilisticError::DimensionMismatch {
                expected: vec![self.prior.alpha().len()],
                actual: vec![counts.len()],
                operation: "Dirichlet-Multinomial update".to_string(),
            });
        }

        let mut alpha_post = self.prior.alpha().clone();
        for (i, &count) in counts.iter().enumerate() {
            alpha_post[i] += count as f64;
        }

        DirichletDistribution::new(alpha_post)
    }

    /// Posterior predictive probabilities
    pub fn posterior_predictive(&self, counts: &[usize]) -> Result<Vec<f64>> {
        let posterior = self.update(counts)?;
        Ok(posterior.mean())
    }
}

// ============================================================================
// Model Comparison
// ============================================================================

/// Bayesian Information Criterion (BIC)
///
/// BIC = -2 log L + k log n
///
/// Lower values indicate better model fit. Penalizes model complexity.
///
/// # Arguments
///
/// * `log_likelihood` - Maximum log-likelihood
/// * `n_parameters` - Number of model parameters
/// * `n_observations` - Number of observations
pub fn bic(log_likelihood: f64, n_parameters: usize, n_observations: usize) -> f64 {
    -2.0 * log_likelihood + (n_parameters as f64) * (n_observations as f64).ln()
}

/// Akaike Information Criterion (AIC)
///
/// AIC = -2 log L + 2k
///
/// # Arguments
///
/// * `log_likelihood` - Maximum log-likelihood
/// * `n_parameters` - Number of model parameters
pub fn aic(log_likelihood: f64, n_parameters: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * (n_parameters as f64)
}

/// Deviance Information Criterion (DIC)
///
/// DIC = D̄ + p_D
///
/// where D̄ is posterior mean deviance and p_D is effective number of parameters
#[derive(Debug, Clone)]
pub struct DICResult {
    /// DIC value
    pub dic: f64,
    /// Posterior mean deviance D̄
    pub d_bar: f64,
    /// Effective number of parameters p_D
    pub p_d: f64,
}

/// Compute DIC from MCMC samples
///
/// # Arguments
///
/// * `log_likelihood_samples` - Log-likelihood values for each MCMC sample
/// * `log_likelihood_at_mean` - Log-likelihood at posterior mean parameters
pub fn dic(log_likelihood_samples: &[f64], log_likelihood_at_mean: f64) -> Result<DICResult> {
    if log_likelihood_samples.is_empty() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: "log_likelihood_samples".to_string(),
            message: "samples cannot be empty".to_string(),
        });
    }

    // Deviance = -2 * log likelihood
    let deviances: Vec<f64> = log_likelihood_samples.iter().map(|&ll| -2.0 * ll).collect();

    // Posterior mean deviance
    let d_bar = deviances.iter().sum::<f64>() / deviances.len() as f64;

    // Deviance at posterior mean
    let d_hat = -2.0 * log_likelihood_at_mean;

    // Effective number of parameters
    let p_d = d_bar - d_hat;

    // DIC
    let dic = d_bar + p_d;

    Ok(DICResult { dic, d_bar, p_d })
}

/// Widely Applicable Information Criterion (WAIC)
#[derive(Debug, Clone)]
pub struct WAICResult {
    /// WAIC value
    pub waic: f64,
    /// Log pointwise predictive density
    pub lppd: f64,
    /// Effective number of parameters
    pub p_waic: f64,
}

/// Compute WAIC from pointwise log-likelihood samples
///
/// # Arguments
///
/// * `pointwise_log_likelihood` - Matrix of log-likelihoods: [n_samples × n_observations]
///   where each row is an MCMC sample and each column is a data point
pub fn waic(pointwise_log_likelihood: &[Vec<f64>]) -> Result<WAICResult> {
    if pointwise_log_likelihood.is_empty() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: "pointwise_log_likelihood".to_string(),
            message: "samples cannot be empty".to_string(),
        });
    }

    let n_samples = pointwise_log_likelihood.len();
    let n_obs = pointwise_log_likelihood[0].len();

    // Compute lppd (log pointwise predictive density)
    let mut lppd = 0.0;
    for i in 0..n_obs {
        // For observation i, compute log(mean(exp(ll)))
        let mut mean_likelihood = 0.0;
        for sample in pointwise_log_likelihood {
            mean_likelihood += sample[i].exp();
        }
        mean_likelihood /= n_samples as f64;
        lppd += mean_likelihood.ln();
    }

    // Compute p_WAIC (effective number of parameters)
    let mut p_waic = 0.0;
    for i in 0..n_obs {
        // Variance of log-likelihood for observation i
        let ll_i: Vec<f64> = pointwise_log_likelihood.iter().map(|s| s[i]).collect();
        let mean_ll = ll_i.iter().sum::<f64>() / n_samples as f64;
        let var_ll = ll_i.iter().map(|&ll| (ll - mean_ll).powi(2)).sum::<f64>() / n_samples as f64;
        p_waic += var_ll;
    }

    // WAIC = -2(lppd - p_WAIC)
    let waic = -2.0 * (lppd - p_waic);

    Ok(WAICResult { waic, lppd, p_waic })
}

// ============================================================================
// Credible Intervals
// ============================================================================

/// Compute equal-tailed credible interval
///
/// Returns the interval [q_α/2, q_{1-α/2}] where q_p is the p-th quantile.
/// For example, α=0.05 gives a 95% credible interval.
///
/// # Arguments
///
/// * `samples` - MCMC samples
/// * `alpha` - Significance level (e.g., 0.05 for 95% CI)
pub fn equal_tailed_interval(samples: &[f64], alpha: f64) -> Result<(f64, f64)> {
    validate_probability(alpha, "alpha")?;

    if samples.is_empty() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: "samples".to_string(),
            message: "samples cannot be empty".to_string(),
        });
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;

    let lower = sorted[lower_idx.min(n - 1)];
    let upper = sorted[upper_idx.min(n - 1)];

    Ok((lower, upper))
}

/// Compute Highest Posterior Density (HPD) interval
///
/// The HPD interval is the shortest interval containing (1-α)% of the posterior mass.
/// For unimodal distributions, this is the most informative credible interval.
///
/// # Arguments
///
/// * `samples` - MCMC samples
/// * `alpha` - Significance level (e.g., 0.05 for 95% HPD)
pub fn hpd_interval(samples: &[f64], alpha: f64) -> Result<(f64, f64)> {
    validate_probability(alpha, "alpha")?;

    if samples.is_empty() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: "samples".to_string(),
            message: "samples cannot be empty".to_string(),
        });
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let interval_size = ((1.0 - alpha) * n as f64).ceil() as usize;

    if interval_size >= n {
        return Ok((sorted[0], sorted[n - 1]));
    }

    // Find the shortest interval containing interval_size points
    let mut min_width = f64::INFINITY;
    let mut best_lower = sorted[0];
    let mut best_upper = sorted[n - 1];

    for i in 0..=(n - interval_size) {
        let lower = sorted[i];
        let upper = sorted[i + interval_size - 1];
        let width = upper - lower;

        if width < min_width {
            min_width = width;
            best_lower = lower;
            best_upper = upper;
        }
    }

    Ok((best_lower, best_upper))
}

// ============================================================================
// Bayes Factors
// ============================================================================

/// Compute Bayes factor from marginal likelihoods
///
/// BF_{10} = p(D|M1) / p(D|M0)
///
/// Interpretation (Jeffreys' scale):
/// - BF < 1: Evidence for M0
/// - 1-3: Barely worth mentioning
/// - 3-10: Substantial evidence for M1
/// - 10-30: Strong evidence for M1
/// - 30-100: Very strong evidence for M1
/// - >100: Decisive evidence for M1
///
/// # Arguments
///
/// * `log_marginal_likelihood_m1` - Log marginal likelihood for model 1
/// * `log_marginal_likelihood_m0` - Log marginal likelihood for model 0 (null)
pub fn bayes_factor(log_marginal_likelihood_m1: f64, log_marginal_likelihood_m0: f64) -> f64 {
    (log_marginal_likelihood_m1 - log_marginal_likelihood_m0).exp()
}

/// Estimate marginal likelihood using harmonic mean estimator
///
/// WARNING: The harmonic mean estimator is known to be unstable and should be used with caution.
/// Consider using bridge sampling or other more stable methods for production use.
///
/// # Arguments
///
/// * `log_likelihood_samples` - Log-likelihood values for MCMC samples
pub fn harmonic_mean_marginal_likelihood(log_likelihood_samples: &[f64]) -> Result<f64> {
    if log_likelihood_samples.is_empty() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: "log_likelihood_samples".to_string(),
            message: "samples cannot be empty".to_string(),
        });
    }

    // Harmonic mean: 1/n ∑ 1/L_i = 1/n ∑ exp(-log L_i)
    let n = log_likelihood_samples.len() as f64;
    let sum_inv: f64 = log_likelihood_samples.iter().map(|&ll| (-ll).exp()).sum();

    let harmonic_mean = n / sum_inv;
    Ok(harmonic_mean.ln())
}

// ============================================================================
// Posterior Predictive Checks
// ============================================================================

/// Compute posterior predictive p-value
///
/// p = P(T(y_rep) ≥ T(y) | y)
///
/// where T is a test statistic, y is observed data, and y_rep is replicated data
/// from the posterior predictive distribution.
///
/// # Arguments
///
/// * `test_statistic_observed` - Test statistic on observed data
/// * `test_statistic_replicated` - Test statistics on posterior predictive samples
pub fn posterior_predictive_pvalue(
    test_statistic_observed: f64,
    test_statistic_replicated: &[f64],
) -> Result<f64> {
    if test_statistic_replicated.is_empty() {
        return Err(ProbabilisticError::InvalidParameter {
            parameter: "test_statistic_replicated".to_string(),
            message: "replicated statistics cannot be empty".to_string(),
        });
    }

    let n_greater = test_statistic_replicated
        .iter()
        .filter(|&&t| t >= test_statistic_observed)
        .count();

    Ok(n_greater as f64 / test_statistic_replicated.len() as f64)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beta_binomial_conjugate() {
        let prior = BetaBinomialConjugate::new(1.0, 1.0).unwrap(); // Uniform prior

        // Observe 7 successes in 10 trials
        let posterior = prior.update(7, 10).unwrap();

        // Posterior should be Beta(8, 4)
        assert_relative_eq!(posterior.alpha(), 8.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.beta(), 4.0, epsilon = 1e-10);

        // Posterior mean = 8/12 = 2/3
        assert_relative_eq!(posterior.mean(), 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_poisson_conjugate() {
        let prior = GammaPoissonConjugate::new(1.0, 1.0).unwrap();

        // Observe data: [2, 3, 4, 3, 2]
        let data = vec![2.0, 3.0, 4.0, 3.0, 2.0];
        let posterior = prior.update(&data).unwrap();

        // Sum = 14, n = 5
        // Posterior: Gamma(1 + 14, 1 + 5) = Gamma(15, 6)
        assert_relative_eq!(posterior.alpha(), 15.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.beta(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_normal_conjugate() {
        let prior = NormalNormalConjugate::new(0.0, 1.0, 1.0).unwrap();

        // Observe data near 2.0
        let data = vec![1.8, 2.0, 2.2, 1.9, 2.1];
        let (mean_post, var_post) = prior.update(&data).unwrap();

        // Posterior mean should be between prior mean (0) and sample mean (2)
        assert!(mean_post > 0.0 && mean_post < 2.0);

        // Posterior variance should be smaller than prior
        assert!(var_post < 1.0);
    }

    #[test]
    fn test_dirichlet_multinomial_conjugate() {
        let prior = DirichletMultinomialConjugate::new(vec![1.0, 1.0, 1.0]).unwrap();

        // Observe counts
        let counts = vec![10, 20, 15];
        let posterior = prior.update(&counts).unwrap();

        // Posterior alpha should be [11, 21, 16]
        assert_relative_eq!(posterior.alpha()[0], 11.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.alpha()[1], 21.0, epsilon = 1e-10);
        assert_relative_eq!(posterior.alpha()[2], 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bic() {
        let log_lik = -100.0;
        let bic_value = bic(log_lik, 5, 100);

        // BIC = -2*(-100) + 5*log(100) = 200 + 5*4.605 ≈ 223.03
        assert!(bic_value > 220.0 && bic_value < 225.0);
    }

    #[test]
    fn test_aic() {
        let log_lik = -100.0;
        let aic_value = aic(log_lik, 5);

        // AIC = -2*(-100) + 2*5 = 200 + 10 = 210
        assert_relative_eq!(aic_value, 210.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dic() {
        let log_lik_samples = vec![-10.0, -12.0, -11.0, -10.5, -11.5];
        let log_lik_at_mean = -11.0;

        let dic_result = dic(&log_lik_samples, log_lik_at_mean).unwrap();

        assert!(dic_result.dic.is_finite());
        assert!(dic_result.d_bar > 0.0);
        assert!(dic_result.p_d.is_finite());
    }

    #[test]
    fn test_waic() {
        // Simple example with 3 samples and 4 observations
        let pointwise_ll = vec![
            vec![-1.0, -2.0, -1.5, -2.5],
            vec![-1.2, -1.8, -1.4, -2.3],
            vec![-0.9, -2.1, -1.6, -2.4],
        ];

        let waic_result = waic(&pointwise_ll).unwrap();

        assert!(waic_result.waic.is_finite());
        assert!(waic_result.lppd.is_finite());
        assert!(waic_result.p_waic >= 0.0);
    }

    #[test]
    fn test_equal_tailed_interval() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (lower, upper) = equal_tailed_interval(&samples, 0.1).unwrap();

        // 90% interval should exclude bottom 5% and top 5%
        assert!(lower >= 1.0);
        assert!(upper <= 10.0);
        assert!(lower < upper);
    }

    #[test]
    fn test_hpd_interval() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (lower, upper) = hpd_interval(&samples, 0.1).unwrap();

        // 90% HPD should be a valid interval
        assert!(lower < upper);
        assert!(lower >= 1.0);
        assert!(upper <= 10.0);
    }

    #[test]
    fn test_bayes_factor() {
        let log_ml_m1 = -100.0;
        let log_ml_m0 = -110.0;

        let bf = bayes_factor(log_ml_m1, log_ml_m0);

        // BF = exp(-100 - (-110)) = exp(10) ≈ 22026
        assert_relative_eq!(bf, 10.0_f64.exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_harmonic_mean_marginal_likelihood() {
        let log_lik_samples = vec![-10.0, -11.0, -12.0, -10.5, -11.5];
        let log_ml = harmonic_mean_marginal_likelihood(&log_lik_samples).unwrap();

        assert!(log_ml.is_finite());
        assert!(log_ml < 0.0); // Log of value less than 1
    }

    #[test]
    fn test_posterior_predictive_pvalue() {
        let observed = 5.0;
        let replicated = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let pvalue = posterior_predictive_pvalue(observed, &replicated).unwrap();

        // 6 out of 8 values are >= 5.0
        assert_relative_eq!(pvalue, 6.0 / 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_credible_interval_95() {
        let prior = NormalNormalConjugate::new(0.0, 1.0, 1.0).unwrap();
        let data = vec![1.0, 2.0, 1.5, 1.8, 2.1];

        let (lower, upper) = prior.credible_interval_95(&data).unwrap();

        assert!(lower < upper);
        let (mean, _) = prior.update(&data).unwrap();
        assert!(lower < mean && mean < upper);
    }
}
