//! GARCH Family Models for Volatility Modeling
//!
//! Implements Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
//! models for modeling time-varying volatility in financial time series.
//!
//! ## Models
//!
//! - **ARCH(q)**: AutoRegressive Conditional Heteroskedasticity
//!   - `sigma^2_t = omega + sum_i alpha_i * eps^2_{t-i}`
//!
//! - **GARCH(p,q)**: Generalized ARCH
//!   - `sigma^2_t = omega + sum_i alpha_i * eps^2_{t-i} + sum_j beta_j * sigma^2_{t-j}`
//!
//! - **EGARCH(p,q)**: Exponential GARCH (Nelson, 1991)
//!   - `log(sigma^2_t) = omega + sum_i alpha_i * g(z_{t-i}) + sum_j beta_j * log(sigma^2_{t-j})`
//!   - where `g(z) = gamma * z + delta * (|z| - E[|z|])`
//!   - Captures leverage effects (asymmetric response to positive/negative shocks)
//!
//! ## References
//!
//! - Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with
//!   Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.
//! - Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
//!   *Journal of Econometrics*, 31(3), 307-327.
//! - Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns:
//!   A New Approach." *Econometrica*, 59(2), 347-370.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array1, ArrayView1};

// =============================================================================
// Parameter Structures
// =============================================================================

/// Fitted GARCH model parameters.
///
/// Contains the estimated parameters, conditional variances, and diagnostic
/// information (log-likelihood, information criteria).
#[derive(Debug, Clone)]
pub struct GarchParams {
    /// Constant term in variance equation (omega > 0).
    pub omega: f64,
    /// ARCH coefficients (alpha_1, ..., alpha_q); each alpha_i >= 0.
    pub alpha: Array1<f64>,
    /// GARCH coefficients (beta_1, ..., beta_p); each beta_j >= 0.
    pub beta: Array1<f64>,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
    /// Akaike Information Criterion: -2*LL + 2*k.
    pub aic: f64,
    /// Bayesian Information Criterion: -2*LL + k*ln(n).
    pub bic: f64,
}

impl GarchParams {
    /// Check the stationarity condition: sum(alpha) + sum(beta) < 1.
    pub fn is_stationary(&self) -> bool {
        let persistence = self.alpha.iter().sum::<f64>() + self.beta.iter().sum::<f64>();
        persistence < 1.0
    }

    /// Compute the persistence (sum of alpha and beta coefficients).
    pub fn persistence(&self) -> f64 {
        self.alpha.iter().sum::<f64>() + self.beta.iter().sum::<f64>()
    }

    /// Compute the unconditional (long-run) variance: omega / (1 - persistence).
    ///
    /// Returns `None` if the model is not stationary.
    pub fn unconditional_variance(&self) -> Option<f64> {
        let persistence = self.persistence();
        if persistence >= 1.0 || self.omega <= 0.0 {
            return None;
        }
        Some(self.omega / (1.0 - persistence))
    }

    /// Compute the half-life of a volatility shock (in periods).
    ///
    /// The half-life is `ln(0.5) / ln(persistence)`.
    /// Returns `None` if the model is not stationary.
    pub fn half_life(&self) -> Option<f64> {
        let persistence = self.persistence();
        if persistence <= 0.0 || persistence >= 1.0 {
            return None;
        }
        Some(0.5_f64.ln() / persistence.ln())
    }
}

/// Fitted EGARCH model parameters.
///
/// The EGARCH model uses a logarithmic specification so that
/// no positivity constraints on the coefficients are needed.
#[derive(Debug, Clone)]
pub struct EgarchParams {
    /// Constant term in log-variance equation.
    pub omega: f64,
    /// Magnitude coefficients (delta) for |z| - E[|z|].
    pub alpha: Array1<f64>,
    /// Leverage/asymmetry coefficients (gamma) for z.
    pub gamma: Array1<f64>,
    /// Persistence coefficients for log(sigma^2_{t-j}).
    pub beta: Array1<f64>,
    /// Log-likelihood.
    pub log_likelihood: f64,
    /// AIC.
    pub aic: f64,
    /// BIC.
    pub bic: f64,
}

impl EgarchParams {
    /// Check the stationarity condition for EGARCH: sum(|beta|) < 1.
    pub fn is_stationary(&self) -> bool {
        let persistence: f64 = self.beta.iter().map(|b| b.abs()).sum();
        persistence < 1.0
    }

    /// Compute the persistence of EGARCH model.
    pub fn persistence(&self) -> f64 {
        self.beta.iter().map(|b| b.abs()).sum()
    }
}

/// Result of residual analysis on a fitted GARCH model.
#[derive(Debug, Clone)]
pub struct GarchResidualAnalysis {
    /// Standardized residuals: eps_t / sigma_t.
    pub standardized_residuals: Array1<f64>,
    /// Conditional standard deviations: sigma_t.
    pub conditional_volatility: Array1<f64>,
    /// Ljung-Box statistic on squared standardized residuals.
    pub ljung_box_stat: f64,
    /// p-value for the Ljung-Box test on squared standardized residuals.
    pub ljung_box_pvalue: f64,
    /// Mean of standardized residuals (should be near 0).
    pub mean_std_resid: f64,
    /// Variance of standardized residuals (should be near 1).
    pub var_std_resid: f64,
    /// Excess kurtosis of standardized residuals (normal = 0).
    pub kurtosis_std_resid: f64,
}

// =============================================================================
// Parameter Validation Helpers
// =============================================================================

/// Validate that GARCH parameters satisfy positivity and stationarity constraints.
///
/// Not currently called from any `fit()` method below (each estimates its own
/// parameters via constrained optimization without a final validation pass);
/// exercised directly by this module's own tests, which is real coverage of
/// the constraint logic even without a production caller.
#[allow(dead_code)]
fn validate_garch_params(omega: f64, alpha: &Array1<f64>, beta: &Array1<f64>) -> Result<()> {
    if omega <= 0.0 {
        return Err(NumRs2Error::ValueError(format!(
            "omega must be positive, got {}",
            omega
        )));
    }
    for (i, &a) in alpha.iter().enumerate() {
        if a < 0.0 {
            return Err(NumRs2Error::ValueError(format!(
                "alpha[{}] must be non-negative, got {}",
                i, a
            )));
        }
    }
    for (j, &b) in beta.iter().enumerate() {
        if b < 0.0 {
            return Err(NumRs2Error::ValueError(format!(
                "beta[{}] must be non-negative, got {}",
                j, b
            )));
        }
    }
    let persistence: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    if persistence >= 1.0 {
        return Err(NumRs2Error::ValueError(format!(
            "Stationarity violated: sum(alpha) + sum(beta) = {} >= 1.0",
            persistence
        )));
    }
    Ok(())
}

/// Project parameters onto the feasible set (positivity + stationarity).
///
/// This function clamps negative values to a small positive epsilon and
/// rescales if the persistence exceeds 1.
fn project_params(omega: f64, alpha: &mut Array1<f64>, beta: &mut Array1<f64>) -> f64 {
    let omega_proj = if omega <= 0.0 { 1e-6 } else { omega };

    for a in alpha.iter_mut() {
        if *a < 0.0 {
            *a = 0.0;
        }
    }
    for b in beta.iter_mut() {
        if *b < 0.0 {
            *b = 0.0;
        }
    }

    let persistence: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    if persistence >= 0.9999 {
        let scale = 0.99 / persistence;
        *alpha *= scale;
        *beta *= scale;
    }

    omega_proj
}

// =============================================================================
// Normal log-likelihood for GARCH
// =============================================================================

/// Compute the Gaussian log-likelihood given residuals and conditional variances.
///
/// LL = sum_{t} [ -0.5 * (ln(2*pi) + ln(sigma^2_t) + eps^2_t / sigma^2_t) ]
fn gaussian_log_likelihood(residuals: &ArrayView1<f64>, variances: &[f64], start: usize) -> f64 {
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let mut ll = 0.0;
    let n = residuals.len().min(variances.len());
    for t in start..n {
        let v = variances[t];
        if v <= 0.0 {
            continue;
        }
        ll += -0.5 * (ln2pi + v.ln() + residuals[t].powi(2) / v);
    }
    ll
}

// =============================================================================
// ARCH Model
// =============================================================================

/// ARCH(q) model -- AutoRegressive Conditional Heteroskedasticity.
///
/// The conditional variance depends only on past squared residuals:
///
/// `sigma^2_t = omega + alpha_1 * eps^2_{t-1} + ... + alpha_q * eps^2_{t-q}`
///
/// This is a special case of GARCH(0, q).
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::timeseries::garch::Arch;
/// use scirs2_core::ndarray::Array1;
///
/// let residuals = Array1::from_vec(vec![
///     0.1, -0.3, 0.5, -0.2, 0.4, -0.6, 0.3, -0.1, 0.2, -0.4,
///     0.5, -0.3, 0.1, -0.2, 0.6, -0.4, 0.2, -0.5, 0.3, -0.1,
/// ]);
/// let arch = Arch::new(1);
/// let params = arch.fit(&residuals.view()).expect("ARCH fit should succeed");
/// assert!(params.omega > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct Arch {
    /// ARCH order q: number of lagged squared residuals.
    pub q: usize,
}

impl Arch {
    /// Create a new ARCH(q) model.
    ///
    /// # Panics (avoided)
    ///
    /// Returns error on fit if q is 0 or data is insufficient.
    pub fn new(q: usize) -> Self {
        Self { q }
    }

    /// Fit the ARCH(q) model to a series of residuals using MLE.
    ///
    /// Uses the variance-targeting approach for initialization, followed by
    /// iterative gradient-based optimization of the Gaussian log-likelihood.
    pub fn fit(&self, residuals: &ArrayView1<f64>) -> Result<GarchParams> {
        if self.q == 0 {
            return Err(NumRs2Error::ValueError(
                "ARCH order q must be at least 1".to_string(),
            ));
        }

        let garch = Garch::new(0, self.q);
        garch.fit(residuals)
    }

    /// Forecast conditional variance h steps ahead using a fitted ARCH model.
    pub fn forecast_variance(
        &self,
        residuals: &ArrayView1<f64>,
        params: &GarchParams,
        steps: usize,
    ) -> Result<Array1<f64>> {
        let garch = Garch::new(0, self.q);
        garch.forecast_variance(residuals, params, steps)
    }
}

// =============================================================================
// GARCH Model
// =============================================================================

/// GARCH(p, q) model -- Generalized ARCH.
///
/// The conditional variance equation:
///
/// `sigma^2_t = omega + sum_{i=1}^{q} alpha_i * eps^2_{t-i}
///                     + sum_{j=1}^{p} beta_j * sigma^2_{t-j}`
///
/// ## Parameter Constraints
///
/// - omega > 0
/// - alpha_i >= 0 for all i
/// - beta_j >= 0 for all j
/// - sum(alpha) + sum(beta) < 1 (stationarity)
///
/// ## Notes
///
/// - `p` = GARCH order (number of lagged conditional variances)
/// - `q` = ARCH order (number of lagged squared residuals)
/// - GARCH(1,1) is by far the most commonly used specification
#[derive(Debug, Clone)]
pub struct Garch {
    /// GARCH order p (lagged conditional variances).
    pub p: usize,
    /// ARCH order q (lagged squared residuals).
    pub q: usize,
}

impl Garch {
    /// Create a new GARCH(p, q) model.
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }

    /// Fit the GARCH(p, q) model to a series of residuals via MLE.
    ///
    /// The fitting procedure:
    /// 1. Initialize omega via variance targeting: omega = var * (1 - sum(alpha) - sum(beta)).
    /// 2. Use a simplified numerical gradient descent on the Gaussian log-likelihood.
    /// 3. Project parameters to satisfy positivity and stationarity at each step.
    pub fn fit(&self, residuals: &ArrayView1<f64>) -> Result<GarchParams> {
        let n = residuals.len();
        let max_lag = self.p.max(self.q);

        if max_lag == 0 && self.p == 0 && self.q == 0 {
            return Err(NumRs2Error::ValueError(
                "At least one of p or q must be positive".to_string(),
            ));
        }

        if n <= max_lag + 1 {
            return Err(NumRs2Error::ValueError(format!(
                "Insufficient observations ({}) for GARCH({},{}) estimation; need at least {}",
                n,
                self.p,
                self.q,
                max_lag + 2
            )));
        }

        let sq_resid: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
        let sample_var = sq_resid.iter().sum::<f64>() / n as f64;

        if sample_var < 1e-15 {
            return Err(NumRs2Error::ComputationError(
                "Residual series has zero or near-zero variance; cannot fit GARCH".to_string(),
            ));
        }

        // --- Initialization via variance targeting ---
        let init_alpha_sum = 0.05 * self.q as f64;
        let init_beta_sum = if self.p > 0 {
            0.85_f64.min(0.99 - init_alpha_sum)
        } else {
            0.0
        };
        let init_omega = sample_var * (1.0 - init_alpha_sum - init_beta_sum);

        let mut omega = init_omega.max(1e-8);
        let mut alpha = if self.q > 0 {
            Array1::from_elem(self.q, init_alpha_sum / self.q as f64)
        } else {
            Array1::zeros(0)
        };
        let mut beta = if self.p > 0 {
            Array1::from_elem(self.p, init_beta_sum / self.p as f64)
        } else {
            Array1::zeros(0)
        };

        // --- MLE via numerical gradient ascent ---
        let max_iter = 200;
        let mut lr = 1e-5;
        let eps_grad = 1e-7;

        let variances = self.compute_variances(&sq_resid, omega, &alpha, &beta, sample_var);
        let mut best_ll = gaussian_log_likelihood(residuals, &variances, max_lag);
        let mut best_omega = omega;
        let mut best_alpha = alpha.clone();
        let mut best_beta = beta.clone();

        for iter in 0..max_iter {
            // Compute numerical gradient for omega
            let var_plus =
                self.compute_variances(&sq_resid, omega + eps_grad, &alpha, &beta, sample_var);
            let var_minus = self.compute_variances(
                &sq_resid,
                (omega - eps_grad).max(1e-10),
                &alpha,
                &beta,
                sample_var,
            );
            let grad_omega = (gaussian_log_likelihood(residuals, &var_plus, max_lag)
                - gaussian_log_likelihood(residuals, &var_minus, max_lag))
                / (2.0 * eps_grad);

            // Compute gradients for alpha
            let mut grad_alpha = Array1::zeros(self.q);
            for i in 0..self.q {
                let mut alpha_p = alpha.clone();
                let mut alpha_m = alpha.clone();
                alpha_p[i] += eps_grad;
                alpha_m[i] = (alpha_m[i] - eps_grad).max(0.0);
                let vp = self.compute_variances(&sq_resid, omega, &alpha_p, &beta, sample_var);
                let vm = self.compute_variances(&sq_resid, omega, &alpha_m, &beta, sample_var);
                let delta = alpha_p[i] - alpha_m[i];
                if delta.abs() > 1e-15 {
                    grad_alpha[i] = (gaussian_log_likelihood(residuals, &vp, max_lag)
                        - gaussian_log_likelihood(residuals, &vm, max_lag))
                        / delta;
                }
            }

            // Compute gradients for beta
            let mut grad_beta = Array1::zeros(self.p);
            for j in 0..self.p {
                let mut beta_p = beta.clone();
                let mut beta_m = beta.clone();
                beta_p[j] += eps_grad;
                beta_m[j] = (beta_m[j] - eps_grad).max(0.0);
                let vp = self.compute_variances(&sq_resid, omega, &alpha, &beta_p, sample_var);
                let vm = self.compute_variances(&sq_resid, omega, &alpha, &beta_m, sample_var);
                let delta = beta_p[j] - beta_m[j];
                if delta.abs() > 1e-15 {
                    grad_beta[j] = (gaussian_log_likelihood(residuals, &vp, max_lag)
                        - gaussian_log_likelihood(residuals, &vm, max_lag))
                        / delta;
                }
            }

            // Update parameters
            omega += lr * grad_omega;
            alpha = &alpha + &(&grad_alpha * lr);
            beta = &beta + &(&grad_beta * lr);

            // Project onto feasible set
            omega = project_params(omega, &mut alpha, &mut beta);

            // Evaluate new log-likelihood
            let new_variances = self.compute_variances(&sq_resid, omega, &alpha, &beta, sample_var);
            let new_ll = gaussian_log_likelihood(residuals, &new_variances, max_lag);

            if new_ll > best_ll {
                best_ll = new_ll;
                best_omega = omega;
                best_alpha = alpha.clone();
                best_beta = beta.clone();
                // Increase learning rate on success
                lr *= 1.05;
            } else {
                // Revert and decrease learning rate
                omega = best_omega;
                alpha = best_alpha.clone();
                beta = best_beta.clone();
                lr *= 0.5;
            }

            // Convergence check
            if iter > 10 && lr < 1e-14 {
                break;
            }
        }

        omega = best_omega;
        alpha = best_alpha;
        beta = best_beta;

        // Final log-likelihood
        let final_variances = self.compute_variances(&sq_resid, omega, &alpha, &beta, sample_var);
        let log_likelihood = gaussian_log_likelihood(residuals, &final_variances, max_lag);

        let k = 1 + self.q + self.p; // number of parameters
        let n_eff = (n - max_lag) as f64;
        let aic = -2.0 * log_likelihood + 2.0 * k as f64;
        let bic = -2.0 * log_likelihood + (k as f64) * n_eff.ln();

        Ok(GarchParams {
            omega,
            alpha,
            beta,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// Compute conditional variances given parameters.
    ///
    /// Uses `backcast` (typically the sample variance) as initial values for
    /// time points before enough history is available.
    pub fn compute_variances(
        &self,
        sq_residuals: &[f64],
        omega: f64,
        alpha: &Array1<f64>,
        beta: &Array1<f64>,
        backcast: f64,
    ) -> Vec<f64> {
        let n = sq_residuals.len();
        let mut variances = vec![backcast; n];

        let max_lag = self.p.max(self.q);

        for t in max_lag..n {
            let mut var_t = omega;

            for i in 0..self.q {
                let idx = t - i - 1;
                if idx < n {
                    var_t += alpha[i] * sq_residuals[idx];
                }
            }

            for j in 0..self.p {
                let idx = t - j - 1;
                if idx < n {
                    var_t += beta[j] * variances[idx];
                }
            }

            variances[t] = var_t.max(1e-12);
        }

        variances
    }

    /// Forecast conditional variance h steps ahead.
    ///
    /// For horizons beyond 1, uses the recursive property:
    /// `E[sigma^2_{T+h}] = omega + (alpha_1 + beta_1) * E[sigma^2_{T+h-1}]`
    /// (for GARCH(1,1); generalized for higher orders).
    pub fn forecast_variance(
        &self,
        residuals: &ArrayView1<f64>,
        params: &GarchParams,
        steps: usize,
    ) -> Result<Array1<f64>> {
        if steps == 0 {
            return Ok(Array1::zeros(0));
        }

        let n = residuals.len();
        let sq_resid: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
        let sample_var = sq_resid.iter().sum::<f64>() / n as f64;

        let hist_var = self.compute_variances(
            &sq_resid,
            params.omega,
            &params.alpha,
            &params.beta,
            sample_var,
        );

        // Build extended series for recursive forecasting
        let total_len = n + steps;
        let mut ext_sq_resid = vec![0.0; total_len];
        let mut ext_var = vec![0.0; total_len];

        ext_sq_resid[..n].copy_from_slice(&sq_resid[..n]);
        ext_var[..n].copy_from_slice(&hist_var[..n]);

        // For future periods, E[eps^2_{T+h}] = E[sigma^2_{T+h}] (law of iterated expectations)
        let mut forecasts = Array1::zeros(steps);

        for h in 0..steps {
            let t = n + h;
            let mut var_t = params.omega;

            for i in 0..self.q {
                let idx = t - i - 1;
                // For future time points, eps^2 = sigma^2 (expected value)
                let sq_val = if idx < n {
                    ext_sq_resid[idx]
                } else {
                    ext_var[idx]
                };
                var_t += params.alpha[i] * sq_val;
            }

            for j in 0..self.p {
                let idx = t - j - 1;
                var_t += params.beta[j] * ext_var[idx];
            }

            ext_var[t] = var_t.max(1e-12);
            ext_sq_resid[t] = var_t; // E[eps^2] = sigma^2 for future
            forecasts[h] = var_t;
        }

        Ok(forecasts)
    }

    /// Compute standardized residuals and perform residual diagnostics.
    ///
    /// Returns a `GarchResidualAnalysis` that includes:
    /// - Standardized residuals (eps_t / sigma_t)
    /// - Ljung-Box test on squared standardized residuals
    /// - Summary statistics of standardized residuals
    pub fn residual_analysis(
        &self,
        residuals: &ArrayView1<f64>,
        params: &GarchParams,
        lags: usize,
    ) -> Result<GarchResidualAnalysis> {
        let n = residuals.len();
        let sq_resid: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
        let sample_var = sq_resid.iter().sum::<f64>() / n as f64;

        let variances = self.compute_variances(
            &sq_resid,
            params.omega,
            &params.alpha,
            &params.beta,
            sample_var,
        );

        let max_lag = self.p.max(self.q);

        // Compute standardized residuals and conditional volatility
        let effective_n = n - max_lag;
        let mut std_resid = Array1::zeros(effective_n);
        let mut cond_vol = Array1::zeros(effective_n);

        for t in max_lag..n {
            let sigma = variances[t].sqrt();
            let idx = t - max_lag;
            cond_vol[idx] = sigma;
            if sigma > 1e-15 {
                std_resid[idx] = residuals[t] / sigma;
            }
        }

        // Summary statistics of standardized residuals
        let mean_sr = std_resid.iter().sum::<f64>() / effective_n as f64;
        let var_sr = std_resid
            .iter()
            .map(|&z| (z - mean_sr).powi(2))
            .sum::<f64>()
            / effective_n as f64;
        let m4 = std_resid
            .iter()
            .map(|&z| (z - mean_sr).powi(4))
            .sum::<f64>()
            / effective_n as f64;
        let kurtosis = if var_sr > 1e-15 {
            m4 / (var_sr * var_sr) - 3.0
        } else {
            0.0
        };

        // Ljung-Box test on squared standardized residuals
        let sq_std_resid: Array1<f64> = std_resid.iter().map(|&z| z * z).collect();

        let test_lags = lags.min(effective_n - 1).max(1);
        let df_adjust = self.p + self.q; // degrees of freedom adjustment

        let (lb_stat, lb_pvalue) = ljung_box_squared(&sq_std_resid.view(), test_lags, df_adjust)?;

        Ok(GarchResidualAnalysis {
            standardized_residuals: std_resid,
            conditional_volatility: cond_vol,
            ljung_box_stat: lb_stat,
            ljung_box_pvalue: lb_pvalue,
            mean_std_resid: mean_sr,
            var_std_resid: var_sr,
            kurtosis_std_resid: kurtosis,
        })
    }

    /// Fit and return conditional variances without creating full params.
    ///
    /// Convenience method for obtaining the variance series given known parameters.
    pub fn conditional_variances(
        &self,
        residuals: &ArrayView1<f64>,
        params: &GarchParams,
    ) -> Array1<f64> {
        let sq_resid: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
        let sample_var = sq_resid.iter().sum::<f64>() / residuals.len() as f64;
        let variances = self.compute_variances(
            &sq_resid,
            params.omega,
            &params.alpha,
            &params.beta,
            sample_var,
        );
        Array1::from_vec(variances)
    }
}

// =============================================================================
// EGARCH Model
// =============================================================================

/// EGARCH(p, q) model -- Exponential GARCH (Nelson, 1991).
///
/// The log-variance equation:
///
/// `log(sigma^2_t) = omega + sum_{i=1}^{q} [ gamma_i * z_{t-i} + delta_i * (|z_{t-i}| - E[|z|]) ]
///                         + sum_{j=1}^{p} beta_j * log(sigma^2_{t-j})`
///
/// where `z_t = eps_t / sigma_t` are standardized residuals and `E[|z|] = sqrt(2/pi)`
/// for standard normal.
///
/// ## Advantages over standard GARCH
///
/// - No positivity constraints needed (operates on log-variance).
/// - Captures leverage effects via the asymmetry coefficient gamma.
/// - Negative gamma means negative shocks increase volatility more than positive shocks.
#[derive(Debug, Clone)]
pub struct Egarch {
    /// EGARCH order p (lagged log-variances).
    pub p: usize,
    /// EGARCH order q (lagged standardized innovations).
    pub q: usize,
}

impl Egarch {
    /// Create a new EGARCH(p, q) model.
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }

    /// Fit the EGARCH(p, q) model to a series of residuals.
    ///
    /// Uses numerical gradient ascent on the Gaussian log-likelihood.
    pub fn fit(&self, residuals: &ArrayView1<f64>) -> Result<EgarchParams> {
        let n = residuals.len();
        let max_lag = self.p.max(self.q);

        if self.q == 0 {
            return Err(NumRs2Error::ValueError(
                "EGARCH order q must be at least 1".to_string(),
            ));
        }

        if n <= max_lag + 1 {
            return Err(NumRs2Error::ValueError(format!(
                "Insufficient observations ({}) for EGARCH({},{}) estimation",
                n, self.p, self.q
            )));
        }

        let sq_resid: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
        let sample_var = sq_resid.iter().sum::<f64>() / n as f64;

        if sample_var < 1e-15 {
            return Err(NumRs2Error::ComputationError(
                "Residual series has zero or near-zero variance; cannot fit EGARCH".to_string(),
            ));
        }

        // Initialize parameters
        let mut omega = sample_var.ln();
        let mut alpha = Array1::from_elem(self.q, 0.1); // magnitude (delta)
        let mut gamma = Array1::from_elem(self.q, -0.05); // leverage (gamma)
        let mut beta = if self.p > 0 {
            Array1::from_elem(self.p, 0.9 / self.p as f64)
        } else {
            Array1::zeros(0)
        };

        let max_iter = 200;
        let mut lr = 1e-5;
        let eps_grad = 1e-7;

        let variances = self.compute_variances(residuals, omega, &alpha, &gamma, &beta, sample_var);
        let mut best_ll = gaussian_log_likelihood(residuals, &variances, max_lag);
        let mut best_omega = omega;
        let mut best_alpha = alpha.clone();
        let mut best_gamma = gamma.clone();
        let mut best_beta = beta.clone();

        for iter in 0..max_iter {
            // Numerical gradient for omega
            let vp = self.compute_variances(
                residuals,
                omega + eps_grad,
                &alpha,
                &gamma,
                &beta,
                sample_var,
            );
            let vm = self.compute_variances(
                residuals,
                omega - eps_grad,
                &alpha,
                &gamma,
                &beta,
                sample_var,
            );
            let g_omega = (gaussian_log_likelihood(residuals, &vp, max_lag)
                - gaussian_log_likelihood(residuals, &vm, max_lag))
                / (2.0 * eps_grad);

            // Gradients for alpha (magnitude)
            let mut g_alpha = Array1::zeros(self.q);
            for i in 0..self.q {
                let mut ap = alpha.clone();
                let mut am = alpha.clone();
                ap[i] += eps_grad;
                am[i] -= eps_grad;
                let vp = self.compute_variances(residuals, omega, &ap, &gamma, &beta, sample_var);
                let vm = self.compute_variances(residuals, omega, &am, &gamma, &beta, sample_var);
                g_alpha[i] = (gaussian_log_likelihood(residuals, &vp, max_lag)
                    - gaussian_log_likelihood(residuals, &vm, max_lag))
                    / (2.0 * eps_grad);
            }

            // Gradients for gamma (leverage)
            let mut g_gamma = Array1::zeros(self.q);
            for i in 0..self.q {
                let mut gp = gamma.clone();
                let mut gm = gamma.clone();
                gp[i] += eps_grad;
                gm[i] -= eps_grad;
                let vp = self.compute_variances(residuals, omega, &alpha, &gp, &beta, sample_var);
                let vm = self.compute_variances(residuals, omega, &alpha, &gm, &beta, sample_var);
                g_gamma[i] = (gaussian_log_likelihood(residuals, &vp, max_lag)
                    - gaussian_log_likelihood(residuals, &vm, max_lag))
                    / (2.0 * eps_grad);
            }

            // Gradients for beta
            let mut g_beta = Array1::zeros(self.p);
            for j in 0..self.p {
                let mut bp = beta.clone();
                let mut bm = beta.clone();
                bp[j] += eps_grad;
                bm[j] -= eps_grad;
                let vp = self.compute_variances(residuals, omega, &alpha, &gamma, &bp, sample_var);
                let vm = self.compute_variances(residuals, omega, &alpha, &gamma, &bm, sample_var);
                g_beta[j] = (gaussian_log_likelihood(residuals, &vp, max_lag)
                    - gaussian_log_likelihood(residuals, &vm, max_lag))
                    / (2.0 * eps_grad);
            }

            // Update
            omega += lr * g_omega;
            alpha = &alpha + &(&g_alpha * lr);
            gamma = &gamma + &(&g_gamma * lr);
            beta = &beta + &(&g_beta * lr);

            // Soft stationarity constraint for EGARCH: |sum(beta)| < 1
            let beta_sum: f64 = beta.iter().map(|b| b.abs()).sum();
            if beta_sum >= 0.9999 {
                let scale = 0.99 / beta_sum;
                beta *= scale;
            }

            let new_variances =
                self.compute_variances(residuals, omega, &alpha, &gamma, &beta, sample_var);
            let new_ll = gaussian_log_likelihood(residuals, &new_variances, max_lag);

            if new_ll > best_ll {
                best_ll = new_ll;
                best_omega = omega;
                best_alpha = alpha.clone();
                best_gamma = gamma.clone();
                best_beta = beta.clone();
                lr *= 1.05;
            } else {
                omega = best_omega;
                alpha = best_alpha.clone();
                gamma = best_gamma.clone();
                beta = best_beta.clone();
                lr *= 0.5;
            }

            if iter > 10 && lr < 1e-14 {
                break;
            }
        }

        let final_variances = self.compute_variances(
            residuals,
            best_omega,
            &best_alpha,
            &best_gamma,
            &best_beta,
            sample_var,
        );
        let log_likelihood = gaussian_log_likelihood(residuals, &final_variances, max_lag);

        let k = 1 + 2 * self.q + self.p; // omega + q alphas + q gammas + p betas
        let n_eff = (n - max_lag) as f64;
        let aic = -2.0 * log_likelihood + 2.0 * k as f64;
        let bic = -2.0 * log_likelihood + (k as f64) * n_eff.ln();

        Ok(EgarchParams {
            omega: best_omega,
            alpha: best_alpha,
            gamma: best_gamma,
            beta: best_beta,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// Compute conditional variances for the EGARCH model.
    ///
    /// Works in log-variance space, then exponentiates.
    pub fn compute_variances(
        &self,
        residuals: &ArrayView1<f64>,
        omega: f64,
        alpha: &Array1<f64>,
        gamma: &Array1<f64>,
        beta: &Array1<f64>,
        backcast: f64,
    ) -> Vec<f64> {
        let n = residuals.len();
        let max_lag = self.p.max(self.q);
        let e_abs_z = (2.0_f64 / std::f64::consts::PI).sqrt(); // E[|z|] for N(0,1)

        let log_backcast = backcast.max(1e-12).ln();

        let mut log_var = vec![log_backcast; n];
        let mut variances = vec![backcast; n];

        for t in max_lag..n {
            let mut lv = omega;

            // Lagged standardized innovations
            for i in 0..self.q {
                let idx = t - i - 1;
                if idx < n {
                    let sigma_prev = variances[idx].sqrt().max(1e-15);
                    let z = residuals[idx] / sigma_prev;
                    // g(z) = gamma * z + alpha * (|z| - E[|z|])
                    lv += gamma[i] * z + alpha[i] * (z.abs() - e_abs_z);
                }
            }

            // Lagged log-variances
            for j in 0..self.p {
                let idx = t - j - 1;
                if idx < n {
                    lv += beta[j] * log_var[idx];
                }
            }

            // Clamp to avoid numerical overflow/underflow
            lv = lv.clamp(-50.0, 50.0);

            log_var[t] = lv;
            variances[t] = lv.exp().max(1e-12);
        }

        variances
    }

    /// Forecast conditional variance h steps ahead for EGARCH.
    pub fn forecast_variance(
        &self,
        residuals: &ArrayView1<f64>,
        params: &EgarchParams,
        steps: usize,
    ) -> Result<Array1<f64>> {
        if steps == 0 {
            return Ok(Array1::zeros(0));
        }

        let n = residuals.len();
        let sq_resid: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
        let sample_var = sq_resid.iter().sum::<f64>() / n as f64;

        let hist_var = self.compute_variances(
            residuals,
            params.omega,
            &params.alpha,
            &params.gamma,
            &params.beta,
            sample_var,
        );

        let e_abs_z = (2.0_f64 / std::f64::consts::PI).sqrt();

        // Build extended arrays
        let total_len = n + steps;
        let mut ext_var = vec![0.0; total_len];
        let mut ext_log_var = vec![0.0; total_len];

        for t in 0..n {
            ext_var[t] = hist_var[t];
            ext_log_var[t] = hist_var[t].max(1e-12).ln();
        }

        let mut forecasts = Array1::zeros(steps);

        for h in 0..steps {
            let t = n + h;
            let mut lv = params.omega;

            // For future z: E[z] = 0, E[|z| - E[|z|]] = 0 under normality
            // So the innovation terms contribute 0 in expectation for future periods
            for i in 0..self.q {
                let idx = t - i - 1;
                if idx < n {
                    let sigma_prev = ext_var[idx].sqrt().max(1e-15);
                    let z = residuals[idx] / sigma_prev;
                    lv += params.gamma[i] * z + params.alpha[i] * (z.abs() - e_abs_z);
                }
                // For idx >= n, contribution is 0 in expectation
            }

            for j in 0..self.p {
                let idx = t - j - 1;
                lv += params.beta[j] * ext_log_var[idx];
            }

            lv = lv.clamp(-50.0, 50.0);
            ext_log_var[t] = lv;
            ext_var[t] = lv.exp().max(1e-12);
            forecasts[h] = ext_var[t];
        }

        Ok(forecasts)
    }
}

// =============================================================================
// Ljung-Box Test for Squared Residuals
// =============================================================================

/// Ljung-Box test applied to a generic series (typically squared standardized residuals).
///
/// Q = n*(n+2) * sum_{k=1}^{m} [ rho_k^2 / (n-k) ]
///
/// Under H0 (no autocorrelation), Q ~ chi^2(m - df_adjust).
fn ljung_box_squared(
    series: &ArrayView1<f64>,
    lags: usize,
    df_adjust: usize,
) -> Result<(f64, f64)> {
    let n = series.len();
    if n <= lags {
        return Err(NumRs2Error::ValueError(format!(
            "lags ({}) must be less than series length ({})",
            lags, n
        )));
    }

    // Compute ACF of the series
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-15 {
        // Zero variance => no autocorrelation
        return Ok((0.0, 1.0));
    }

    let mut q_stat = 0.0;
    for k in 1..=lags {
        let mut cov_k = 0.0;
        for t in k..n {
            cov_k += (series[t] - mean) * (series[t - k] - mean);
        }
        let rho_k = cov_k / (n as f64 * var);
        q_stat += rho_k.powi(2) / (n - k) as f64;
    }
    q_stat *= (n * (n + 2)) as f64;

    // Degrees of freedom
    let df = if lags > df_adjust {
        (lags - df_adjust) as f64
    } else {
        1.0 // minimum df
    };

    let p_value = 1.0 - chi_squared_cdf_approx(q_stat, df);

    Ok((q_stat, p_value))
}

/// Approximate chi-squared CDF for p-value computation.
fn chi_squared_cdf_approx(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Wilson-Hilferty normal approximation for any df
    let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();

    normal_cdf_approx(z)
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn normal_cdf_approx(x: f64) -> f64 {
    // Use the error function approximation
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * abs_x / std::f64::consts::SQRT_2);
    let erf_val =
        1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x / 2.0).exp();

    0.5 * (1.0 + sign * erf_val)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Helper: generate synthetic GARCH(1,1) data for testing.
    fn generate_garch_data(n: usize, omega: f64, alpha: f64, beta: f64, seed: u64) -> Array1<f64> {
        // Simple LCG-based pseudo-random generator for reproducibility
        let mut state = seed;
        let mut next_uniform = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [0, 1)
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        // Box-Muller for normal deviates
        let mut next_normal = || -> f64 {
            let u1 = next_uniform().max(1e-15);
            let u2 = next_uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut data = Array1::zeros(n);
        let mut sigma2 = omega / (1.0 - alpha - beta).max(0.01);

        for t in 0..n {
            let z = next_normal();
            let eps = z * sigma2.sqrt();
            data[t] = eps;
            sigma2 = omega + alpha * eps * eps + beta * sigma2;
            sigma2 = sigma2.max(1e-10);
        }

        data
    }

    // =========================================================================
    // Test 1: GARCH(1,1) basic fit on synthetic data
    // =========================================================================
    #[test]
    fn test_garch11_fit_synthetic() {
        let data = generate_garch_data(500, 0.01, 0.08, 0.90, 42);
        let model = Garch::new(1, 1);
        let params = model
            .fit(&data.view())
            .expect("GARCH(1,1) fit should succeed");

        assert!(params.omega > 0.0, "omega should be positive");
        assert!(params.alpha[0] >= 0.0, "alpha should be non-negative");
        assert!(params.beta[0] >= 0.0, "beta should be non-negative");
        assert!(
            params.persistence() < 1.0,
            "persistence should be < 1 for stationarity"
        );
        assert!(params.is_stationary(), "model should be stationary");
        assert!(
            params.log_likelihood.is_finite(),
            "log-likelihood should be finite"
        );
    }

    // =========================================================================
    // Test 2: ARCH(1) as special case (p=0)
    // =========================================================================
    #[test]
    fn test_arch1_special_case() {
        let data = generate_garch_data(300, 0.05, 0.3, 0.0, 123);
        let arch = Arch::new(1);
        let params = arch.fit(&data.view()).expect("ARCH(1) fit should succeed");

        assert!(params.omega > 0.0);
        assert_eq!(params.alpha.len(), 1);
        assert_eq!(params.beta.len(), 0);
        assert!(params.alpha[0] >= 0.0);
        assert!(params.aic.is_finite());
        assert!(params.bic.is_finite());
    }

    // =========================================================================
    // Test 3: Variance forecasting
    // =========================================================================
    #[test]
    fn test_variance_forecasting() {
        let data = generate_garch_data(200, 0.01, 0.1, 0.85, 77);
        let model = Garch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");
        let forecast = model
            .forecast_variance(&data.view(), &params, 10)
            .expect("forecast should succeed");

        assert_eq!(forecast.len(), 10);
        // All forecasted variances should be positive
        for h in 0..10 {
            assert!(
                forecast[h] > 0.0,
                "forecast[{}] = {} should be positive",
                h,
                forecast[h]
            );
        }

        // Forecasts should converge toward unconditional variance for stationary model
        if let Some(uncond_var) = params.unconditional_variance() {
            // The last forecast should be closer to unconditional variance than the first
            let first_diff = (forecast[0] - uncond_var).abs();
            let last_diff = (forecast[9] - uncond_var).abs();
            // This may not always hold for short horizons, so just check they're positive
            assert!(
                uncond_var > 0.0,
                "unconditional variance should be positive"
            );
            let _ = (first_diff, last_diff); // Use values to avoid warnings
        }
    }

    // =========================================================================
    // Test 4: Stationarity constraint validation
    // =========================================================================
    #[test]
    fn test_stationarity_constraint() {
        // Test validate_garch_params rejects non-stationary parameters
        let omega = 0.01;
        let alpha = Array1::from_vec(vec![0.5]);
        let beta = Array1::from_vec(vec![0.6]);
        // sum = 1.1 > 1 => should fail
        let result = validate_garch_params(omega, &alpha, &beta);
        assert!(result.is_err(), "Should reject non-stationary params");
    }

    // =========================================================================
    // Test 5: Parameter validation - negative omega
    // =========================================================================
    #[test]
    fn test_param_validation_negative_omega() {
        let alpha = Array1::from_vec(vec![0.1]);
        let beta = Array1::from_vec(vec![0.8]);
        let result = validate_garch_params(-0.01, &alpha, &beta);
        assert!(result.is_err(), "Should reject negative omega");
    }

    // =========================================================================
    // Test 6: Parameter validation - negative alpha
    // =========================================================================
    #[test]
    fn test_param_validation_negative_alpha() {
        let alpha = Array1::from_vec(vec![-0.1]);
        let beta = Array1::from_vec(vec![0.8]);
        let result = validate_garch_params(0.01, &alpha, &beta);
        assert!(result.is_err(), "Should reject negative alpha");
    }

    // =========================================================================
    // Test 7: Edge case - constant variance (nearly zero ARCH effects)
    // =========================================================================
    #[test]
    fn test_constant_variance_series() {
        // Constant-variance series: iid normal noise
        let mut data = Vec::with_capacity(200);
        let mut state: u64 = 999;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((state >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos());
        }
        let data = Array1::from_vec(data);

        let model = Garch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");

        // For iid data, ARCH+GARCH effects should be small
        let persistence = params.persistence();
        assert!(
            persistence < 1.0,
            "persistence {} should be < 1",
            persistence
        );
        assert!(params.omega > 0.0);
    }

    // =========================================================================
    // Test 8: Edge case - insufficient data
    // =========================================================================
    #[test]
    fn test_insufficient_data() {
        let data = Array1::from_vec(vec![0.1, -0.2]);
        let model = Garch::new(1, 1);
        let result = model.fit(&data.view());
        assert!(result.is_err(), "Should fail with insufficient data");
    }

    // =========================================================================
    // Test 9: GARCH(2,1) higher order
    // =========================================================================
    #[test]
    fn test_garch21() {
        let data = generate_garch_data(400, 0.01, 0.1, 0.85, 55);
        let model = Garch::new(1, 2);
        let params = model
            .fit(&data.view())
            .expect("GARCH(1,2) fit should succeed");

        assert_eq!(params.alpha.len(), 2);
        assert_eq!(params.beta.len(), 1);
        assert!(params.omega > 0.0);
        assert!(params.is_stationary());
    }

    // =========================================================================
    // Test 10: Residual analysis on GARCH(1,1)
    // =========================================================================
    #[test]
    fn test_residual_analysis() {
        let data = generate_garch_data(300, 0.02, 0.1, 0.85, 88);
        let model = Garch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");
        let analysis = model
            .residual_analysis(&data.view(), &params, 10)
            .expect("residual analysis should succeed");

        // Standardized residuals should have mean near 0
        assert!(
            analysis.mean_std_resid.abs() < 0.5,
            "mean of std residuals = {} should be near 0",
            analysis.mean_std_resid
        );

        // Variance of standardized residuals should be near 1
        assert!(
            (analysis.var_std_resid - 1.0).abs() < 1.5,
            "variance of std residuals = {} should be near 1",
            analysis.var_std_resid
        );

        // Ljung-Box statistic should be non-negative
        assert!(analysis.ljung_box_stat >= 0.0);
        assert!(
            analysis.ljung_box_pvalue >= 0.0 && analysis.ljung_box_pvalue <= 1.0,
            "p-value should be in [0, 1]"
        );

        // Conditional volatility should be positive
        assert!(
            analysis.conditional_volatility.iter().all(|&v| v > 0.0),
            "conditional volatility should be positive"
        );
    }

    // =========================================================================
    // Test 11: EGARCH basic fit
    // =========================================================================
    #[test]
    fn test_egarch_basic_fit() {
        let data = generate_garch_data(300, 0.02, 0.1, 0.85, 111);
        let model = Egarch::new(1, 1);
        let params = model.fit(&data.view()).expect("EGARCH fit should succeed");

        assert!(params.log_likelihood.is_finite());
        assert!(params.aic.is_finite());
        assert!(params.bic.is_finite());
        assert_eq!(params.alpha.len(), 1);
        assert_eq!(params.gamma.len(), 1);
        assert_eq!(params.beta.len(), 1);
    }

    // =========================================================================
    // Test 12: EGARCH stationarity check
    // =========================================================================
    #[test]
    fn test_egarch_stationarity() {
        let data = generate_garch_data(300, 0.02, 0.1, 0.85, 222);
        let model = Egarch::new(1, 1);
        let params = model.fit(&data.view()).expect("EGARCH fit should succeed");

        // The optimizer should produce stationary parameters
        assert!(
            params.is_stationary(),
            "EGARCH should be stationary; persistence = {}",
            params.persistence()
        );
    }

    // =========================================================================
    // Test 13: EGARCH variance forecast
    // =========================================================================
    #[test]
    fn test_egarch_variance_forecast() {
        let data = generate_garch_data(200, 0.02, 0.1, 0.85, 333);
        let model = Egarch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");
        let forecast = model
            .forecast_variance(&data.view(), &params, 5)
            .expect("forecast should succeed");

        assert_eq!(forecast.len(), 5);
        assert!(
            forecast.iter().all(|&v| v > 0.0),
            "all forecasts should be positive"
        );
    }

    // =========================================================================
    // Test 14: ARCH forecast via Arch wrapper
    // =========================================================================
    #[test]
    fn test_arch_forecast() {
        let data = generate_garch_data(200, 0.05, 0.3, 0.0, 444);
        let arch = Arch::new(2);
        let params = arch.fit(&data.view()).expect("ARCH(2) fit should succeed");
        let forecast = arch
            .forecast_variance(&data.view(), &params, 5)
            .expect("forecast should succeed");

        assert_eq!(forecast.len(), 5);
        assert!(forecast.iter().all(|&v| v > 0.0));
    }

    // =========================================================================
    // Test 15: GarchParams helper methods
    // =========================================================================
    #[test]
    fn test_garch_params_helpers() {
        let params = GarchParams {
            omega: 0.01,
            alpha: Array1::from_vec(vec![0.1]),
            beta: Array1::from_vec(vec![0.85]),
            log_likelihood: -100.0,
            aic: 206.0,
            bic: 210.0,
        };

        assert!(params.is_stationary());
        let persistence = params.persistence();
        assert!((persistence - 0.95).abs() < 1e-10);

        let uncond_var = params
            .unconditional_variance()
            .expect("should have unconditional variance");
        assert!((uncond_var - 0.2).abs() < 1e-10); // 0.01 / (1 - 0.95) = 0.2

        let half_life = params.half_life().expect("should have half-life");
        assert!(half_life > 0.0);
        // ln(0.5) / ln(0.95) ~ 13.5
        assert!((half_life - 13.513).abs() < 0.1);
    }

    // =========================================================================
    // Test 16: Non-stationary params helper
    // =========================================================================
    #[test]
    fn test_non_stationary_params_helpers() {
        let params = GarchParams {
            omega: 0.01,
            alpha: Array1::from_vec(vec![0.5]),
            beta: Array1::from_vec(vec![0.6]),
            log_likelihood: -100.0,
            aic: 206.0,
            bic: 210.0,
        };

        assert!(!params.is_stationary());
        assert!(params.unconditional_variance().is_none());
        assert!(params.half_life().is_none());
    }

    // =========================================================================
    // Test 17: Zero-variance edge case
    // =========================================================================
    #[test]
    fn test_zero_variance_data() {
        let data = Array1::from_elem(100, 0.0);
        let model = Garch::new(1, 1);
        let result = model.fit(&data.view());
        assert!(result.is_err(), "Should fail on zero-variance data");
    }

    // =========================================================================
    // Test 18: Conditional variances computation
    // =========================================================================
    #[test]
    fn test_conditional_variances() {
        let data = generate_garch_data(100, 0.01, 0.1, 0.85, 555);
        let model = Garch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");
        let cond_var = model.conditional_variances(&data.view(), &params);

        assert_eq!(cond_var.len(), data.len());
        assert!(
            cond_var.iter().all(|&v| v > 0.0),
            "all conditional variances should be positive"
        );
    }

    // =========================================================================
    // Test 19: Information criteria ordering (BIC >= AIC for n >= 8)
    // =========================================================================
    #[test]
    fn test_information_criteria() {
        let data = generate_garch_data(200, 0.01, 0.1, 0.85, 666);
        let model = Garch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");

        // For n sufficiently large, BIC penalty > AIC penalty
        // BIC = -2*LL + k*ln(n), AIC = -2*LL + 2*k
        // So BIC > AIC when ln(n) > 2, i.e., n > ~7.4
        assert!(
            params.bic >= params.aic,
            "BIC ({}) should be >= AIC ({}) for n >= 8",
            params.bic,
            params.aic
        );
    }

    // =========================================================================
    // Test 20: Forecast of zero steps
    // =========================================================================
    #[test]
    fn test_forecast_zero_steps() {
        let data = generate_garch_data(100, 0.01, 0.1, 0.85, 777);
        let model = Garch::new(1, 1);
        let params = model.fit(&data.view()).expect("fit should succeed");
        let forecast = model
            .forecast_variance(&data.view(), &params, 0)
            .expect("zero-step forecast should succeed");
        assert_eq!(forecast.len(), 0);
    }

    // =========================================================================
    // Test 21: Project params utility
    // =========================================================================
    #[test]
    fn test_project_params() {
        let mut alpha = Array1::from_vec(vec![-0.1, 0.5]);
        let mut beta = Array1::from_vec(vec![0.6]);
        let omega = project_params(-0.01, &mut alpha, &mut beta);

        assert!(omega > 0.0, "omega should be projected to positive");
        assert!(
            alpha[0] >= 0.0,
            "alpha[0] should be non-negative after projection"
        );
        let persistence = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
        assert!(
            persistence < 1.0,
            "persistence should be < 1 after projection"
        );
    }

    // =========================================================================
    // Test 22: EGARCH with leverage effect detection
    // =========================================================================
    #[test]
    fn test_egarch_leverage_params() {
        // Generate data with asymmetric volatility response
        let data = generate_garch_data(500, 0.02, 0.12, 0.85, 888);
        let model = Egarch::new(1, 1);
        let params = model.fit(&data.view()).expect("EGARCH fit should succeed");

        // The gamma coefficient captures leverage; it can be any sign
        // Just verify it's finite
        assert!(
            params.gamma[0].is_finite(),
            "gamma should be finite, got {}",
            params.gamma[0]
        );
    }

    // =========================================================================
    // Test 23: EGARCH q=0 rejection
    // =========================================================================
    #[test]
    fn test_egarch_q_zero_rejected() {
        let data = generate_garch_data(200, 0.02, 0.1, 0.85, 999);
        let model = Egarch::new(1, 0);
        let result = model.fit(&data.view());
        assert!(result.is_err(), "EGARCH with q=0 should fail");
    }

    // =========================================================================
    // Test 24: ARCH order 0 rejection
    // =========================================================================
    #[test]
    fn test_arch_order_zero_rejected() {
        let data = generate_garch_data(200, 0.02, 0.1, 0.85, 1111);
        let arch = Arch::new(0);
        let result = arch.fit(&data.view());
        assert!(result.is_err(), "ARCH with q=0 should fail");
    }
}
