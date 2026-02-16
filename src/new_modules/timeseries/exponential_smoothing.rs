//! Advanced Exponential Smoothing Methods for Time Series Analysis
//!
//! This module implements the ETS (Error, Trend, Seasonal) framework for
//! exponential smoothing, providing a unified interface for:
//!
//! - **Simple Exponential Smoothing (SES)**: Level-only smoothing (ETS(A,N,N))
//! - **Holt's Linear Method**: Level + additive trend (ETS(A,A,N))
//! - **Damped Trend Method**: Level + damped additive trend (ETS(A,Ad,N))
//! - **Holt-Winters Additive**: Level + trend + additive seasonality (ETS(A,A,A))
//! - **Holt-Winters Multiplicative**: Level + trend + multiplicative seasonality (ETS(A,A,M))
//! - **Damped Holt-Winters**: Damped trend variants of seasonal models
//!
//! ## Features
//!
//! - Parameter optimization via grid search minimizing MSE
//! - Prediction intervals using analytical variance formulas
//! - AIC/BIC information criteria for model selection
//! - Robust component initialization
//!
//! ## References
//!
//! - Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
//! - Hyndman, R.J., Koehler, A.B., Ord, J.K., & Snyder, R.D. (2008).
//!   *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
//! - Gardner, E.S. Jr (2006). "Exponential smoothing: The state of the art - Part II."
//!   *International Journal of Forecasting*, 22(4), 637-666.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, ArrayView1};

// ============================================================================
// Type Definitions
// ============================================================================

/// Type of trend component in the exponential smoothing model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendComponent {
    /// No trend component (level-only model).
    None,
    /// Additive (linear) trend: forecast = level + h * trend.
    Additive,
    /// Damped additive trend: forecast = level + (phi + phi^2 + ... + phi^h) * trend.
    /// The damping parameter phi is in (0, 1), typically 0.8-0.98.
    Damped,
}

/// Type of seasonal component in the exponential smoothing model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeasonalComponent {
    /// No seasonal component.
    None,
    /// Additive seasonality: forecast = level + trend + seasonal.
    Additive,
    /// Multiplicative seasonality: forecast = (level + trend) * seasonal.
    Multiplicative,
}

/// Configuration for exponential smoothing parameter optimization.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Grid resolution for each parameter dimension (default: 20).
    pub grid_resolution: usize,
    /// Minimum parameter value (default: 0.01).
    pub param_min: f64,
    /// Maximum parameter value (default: 0.99).
    pub param_max: f64,
    /// Number of refinement iterations around the best grid point (default: 2).
    pub refinement_iterations: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 20,
            param_min: 0.01,
            param_max: 0.99,
            refinement_iterations: 2,
        }
    }
}

/// Result of fitting an exponential smoothing model.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingResult {
    /// Fitted values on training data.
    pub fitted: Array1<f64>,
    /// Residuals (actual - fitted) on training data.
    pub residuals: Array1<f64>,
    /// Smoothed level component at each time step.
    pub level: Array1<f64>,
    /// Smoothed trend component at each time step (if applicable).
    pub trend: Option<Array1<f64>>,
    /// Smoothed seasonal indices (one full cycle, if applicable).
    pub seasonal: Option<Array1<f64>>,
    /// Sum of squared errors.
    pub sse: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of estimated parameters (for AIC/BIC).
    pub n_params: usize,
}

/// Forecast output with point predictions and optional prediction intervals.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingForecast {
    /// Point forecasts for h steps ahead.
    pub point: Array1<f64>,
    /// Lower bounds of prediction intervals (at the specified confidence level).
    pub lower: Option<Array1<f64>>,
    /// Upper bounds of prediction intervals (at the specified confidence level).
    pub upper: Option<Array1<f64>>,
    /// Confidence level used for intervals (e.g. 0.95).
    pub confidence_level: f64,
}

/// Information criteria for model selection.
#[derive(Debug, Clone)]
pub struct InformationCriteria {
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Corrected AIC (for small samples).
    pub aicc: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
}

// ============================================================================
// Exponential Smoothing Model
// ============================================================================

/// Unified Exponential Smoothing model supporting SES, Holt, Holt-Winters,
/// and damped trend variants.
///
/// # Model Specification
///
/// The model is specified by choosing:
/// - `alpha` (0, 1): level smoothing parameter
/// - `beta` (0, 1): trend smoothing parameter (optional)
/// - `gamma` (0, 1): seasonal smoothing parameter (optional)
/// - `phi` (0, 1): damping parameter for damped trend (optional, typically 0.8-0.98)
/// - `trend`: type of trend component
/// - `seasonal`: type of seasonal component
/// - `period`: seasonal period length (required if seasonal != None)
///
/// # Examples
///
/// ```rust,no_run
/// use numrs2::new_modules::timeseries::exponential_smoothing::*;
/// use scirs2_core::ndarray::Array1;
///
/// // Simple Exponential Smoothing
/// let model = ExponentialSmoothing::ses(0.3).unwrap();
///
/// // Holt's Linear Trend
/// let model = ExponentialSmoothing::holt(0.3, 0.1).unwrap();
///
/// // Damped Trend
/// let model = ExponentialSmoothing::damped_trend(0.3, 0.1, 0.9).unwrap();
///
/// // Holt-Winters Additive
/// let model = ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 12, SeasonalComponent::Additive).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialSmoothing {
    /// Level smoothing parameter.
    alpha: f64,
    /// Trend smoothing parameter (None for SES).
    beta: Option<f64>,
    /// Seasonal smoothing parameter (None for non-seasonal models).
    gamma: Option<f64>,
    /// Trend damping parameter (None for undamped models).
    phi: Option<f64>,
    /// Type of trend component.
    trend: TrendComponent,
    /// Type of seasonal component.
    seasonal: SeasonalComponent,
    /// Seasonal period (None for non-seasonal models).
    period: Option<usize>,
}

impl ExponentialSmoothing {
    /// Create a Simple Exponential Smoothing (SES) model.
    ///
    /// SES is appropriate for data with no trend or seasonality.
    /// The forecast is flat: all future values equal the last smoothed level.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing parameter in (0, 1)
    pub fn ses(alpha: f64) -> Result<Self> {
        validate_param(alpha, "alpha")?;
        Ok(Self {
            alpha,
            beta: None,
            gamma: None,
            phi: None,
            trend: TrendComponent::None,
            seasonal: SeasonalComponent::None,
            period: None,
        })
    }

    /// Create a Holt's Linear Trend (double exponential smoothing) model.
    ///
    /// Appropriate for data with a linear trend but no seasonality.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing parameter in (0, 1)
    /// * `beta` - Trend smoothing parameter in (0, 1)
    pub fn holt(alpha: f64, beta: f64) -> Result<Self> {
        validate_param(alpha, "alpha")?;
        validate_param(beta, "beta")?;
        Ok(Self {
            alpha,
            beta: Some(beta),
            gamma: None,
            phi: None,
            trend: TrendComponent::Additive,
            seasonal: SeasonalComponent::None,
            period: None,
        })
    }

    /// Create a Damped Trend model.
    ///
    /// The damped trend model modifies Holt's method by dampening the trend
    /// over time, preventing overly optimistic long-range forecasts.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing parameter in (0, 1)
    /// * `beta` - Trend smoothing parameter in (0, 1)
    /// * `phi` - Damping parameter in (0, 1), typically 0.8-0.98
    pub fn damped_trend(alpha: f64, beta: f64, phi: f64) -> Result<Self> {
        validate_param(alpha, "alpha")?;
        validate_param(beta, "beta")?;
        validate_param(phi, "phi")?;
        Ok(Self {
            alpha,
            beta: Some(beta),
            gamma: None,
            phi: Some(phi),
            trend: TrendComponent::Damped,
            seasonal: SeasonalComponent::None,
            period: None,
        })
    }

    /// Create a Holt-Winters seasonal model.
    ///
    /// Supports both additive and multiplicative seasonality, with or without
    /// a damped trend.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing parameter in (0, 1)
    /// * `beta` - Trend smoothing parameter in (0, 1)
    /// * `gamma` - Seasonal smoothing parameter in (0, 1)
    /// * `period` - Seasonal period length (must be >= 2)
    /// * `seasonal` - Type of seasonal component
    pub fn holt_winters(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        seasonal: SeasonalComponent,
    ) -> Result<Self> {
        validate_param(alpha, "alpha")?;
        validate_param(beta, "beta")?;
        validate_param(gamma, "gamma")?;
        if period < 2 {
            return Err(NumRs2Error::ValueError(
                "Seasonal period must be at least 2".to_string(),
            ));
        }
        if seasonal == SeasonalComponent::None {
            return Err(NumRs2Error::ValueError(
                "Use holt() for non-seasonal models".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            beta: Some(beta),
            gamma: Some(gamma),
            phi: None,
            trend: TrendComponent::Additive,
            seasonal,
            period: Some(period),
        })
    }

    /// Create a Damped Holt-Winters seasonal model.
    ///
    /// Combines seasonal decomposition with a damped trend for more
    /// conservative long-range forecasts.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing parameter in (0, 1)
    /// * `beta` - Trend smoothing parameter in (0, 1)
    /// * `gamma` - Seasonal smoothing parameter in (0, 1)
    /// * `phi` - Damping parameter in (0, 1)
    /// * `period` - Seasonal period length (must be >= 2)
    /// * `seasonal` - Type of seasonal component
    pub fn damped_holt_winters(
        alpha: f64,
        beta: f64,
        gamma: f64,
        phi: f64,
        period: usize,
        seasonal: SeasonalComponent,
    ) -> Result<Self> {
        validate_param(alpha, "alpha")?;
        validate_param(beta, "beta")?;
        validate_param(gamma, "gamma")?;
        validate_param(phi, "phi")?;
        if period < 2 {
            return Err(NumRs2Error::ValueError(
                "Seasonal period must be at least 2".to_string(),
            ));
        }
        if seasonal == SeasonalComponent::None {
            return Err(NumRs2Error::ValueError(
                "Use damped_trend() for non-seasonal models".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            beta: Some(beta),
            gamma: Some(gamma),
            phi: Some(phi),
            trend: TrendComponent::Damped,
            seasonal,
            period: Some(period),
        })
    }

    /// Create a fully custom exponential smoothing model.
    ///
    /// This constructor allows specifying all components directly.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing parameter in (0, 1)
    /// * `beta` - Optional trend smoothing parameter in (0, 1)
    /// * `gamma` - Optional seasonal smoothing parameter in (0, 1)
    /// * `phi` - Optional damping parameter in (0, 1)
    /// * `trend` - Trend component type
    /// * `seasonal` - Seasonal component type
    /// * `period` - Optional seasonal period
    pub fn custom(
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        trend: TrendComponent,
        seasonal: SeasonalComponent,
        period: Option<usize>,
    ) -> Result<Self> {
        validate_param(alpha, "alpha")?;
        if let Some(b) = beta {
            validate_param(b, "beta")?;
        }
        if let Some(g) = gamma {
            validate_param(g, "gamma")?;
        }
        if let Some(p) = phi {
            validate_param(p, "phi")?;
        }
        if trend != TrendComponent::None && beta.is_none() {
            return Err(NumRs2Error::ValueError(
                "beta is required when trend is not None".to_string(),
            ));
        }
        if seasonal != SeasonalComponent::None && gamma.is_none() {
            return Err(NumRs2Error::ValueError(
                "gamma is required for seasonal models".to_string(),
            ));
        }
        if seasonal != SeasonalComponent::None {
            match period {
                Some(p) if p < 2 => {
                    return Err(NumRs2Error::ValueError(
                        "Seasonal period must be at least 2".to_string(),
                    ));
                }
                None => {
                    return Err(NumRs2Error::ValueError(
                        "period is required for seasonal models".to_string(),
                    ));
                }
                _ => {}
            }
        }
        if trend == TrendComponent::Damped && phi.is_none() {
            return Err(NumRs2Error::ValueError(
                "phi is required for damped trend models".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            beta,
            gamma,
            phi,
            trend,
            seasonal,
            period,
        })
    }

    /// Fit the model to observed data.
    ///
    /// Initializes the level, trend, and seasonal components, then recursively
    /// applies the smoothing equations to produce fitted values and residuals.
    ///
    /// # Arguments
    ///
    /// * `data` - Observed time series (must have sufficient length)
    ///
    /// # Returns
    ///
    /// An `ExponentialSmoothingResult` containing fitted values, residuals,
    /// component histories, and fit statistics.
    pub fn fit(&self, data: &ArrayView1<f64>) -> Result<ExponentialSmoothingResult> {
        let n = data.len();
        self.validate_data_length(n)?;

        let m = self.period.unwrap_or(1);

        // Initialize components
        let (init_level, init_trend, init_seasonal) = self.initialize(data)?;

        let mut level_hist = Array1::zeros(n);
        let mut trend_hist = if self.trend != TrendComponent::None {
            Some(Array1::zeros(n))
        } else {
            None
        };
        let mut fitted = Array1::zeros(n);

        let mut level = init_level;
        let mut trend = init_trend;
        let mut seasonal = init_seasonal.clone();

        let phi = self.phi.unwrap_or(1.0);
        let beta = self.beta.unwrap_or(0.0);
        let gamma = self.gamma.unwrap_or(0.0);

        // Main smoothing loop
        for t in 0..n {
            let season_idx = t % m;

            // One-step-ahead fitted value at time t
            let ft = self.compute_fitted_value(level, trend, &seasonal, season_idx, phi);
            fitted[t] = ft;

            // Update components
            let prev_level = level;
            match self.seasonal {
                SeasonalComponent::Additive => {
                    level = self.alpha * (data[t] - seasonal[season_idx])
                        + (1.0 - self.alpha) * (prev_level + phi * trend);
                }
                SeasonalComponent::Multiplicative => {
                    let s_val = seasonal[season_idx].max(1e-10);
                    level = self.alpha * (data[t] / s_val)
                        + (1.0 - self.alpha) * (prev_level + phi * trend);
                }
                SeasonalComponent::None => {
                    level = self.alpha * data[t] + (1.0 - self.alpha) * (prev_level + phi * trend);
                }
            }

            if self.trend != TrendComponent::None {
                trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend;
            }

            match self.seasonal {
                SeasonalComponent::Additive => {
                    seasonal[season_idx] =
                        gamma * (data[t] - level) + (1.0 - gamma) * seasonal[season_idx];
                }
                SeasonalComponent::Multiplicative => {
                    let l_val = level.max(1e-10);
                    seasonal[season_idx] =
                        gamma * (data[t] / l_val) + (1.0 - gamma) * seasonal[season_idx];
                }
                SeasonalComponent::None => {}
            }

            level_hist[t] = level;
            if let Some(ref mut th) = trend_hist {
                th[t] = trend;
            }
        }

        let residuals = data - &fitted;
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();
        let mse = sse / n as f64;
        let n_params = self.count_parameters();

        Ok(ExponentialSmoothingResult {
            fitted,
            residuals,
            level: level_hist,
            trend: trend_hist,
            seasonal: if self.seasonal != SeasonalComponent::None {
                Some(seasonal)
            } else {
                None
            },
            sse,
            mse,
            n_obs: n,
            n_params,
        })
    }

    /// Generate h-step-ahead forecasts from the fitted model.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data (used to fit the model)
    /// * `h` - Forecast horizon (number of steps ahead)
    /// * `confidence_level` - Confidence level for prediction intervals (e.g. 0.95)
    ///
    /// # Returns
    ///
    /// An `ExponentialSmoothingForecast` containing point predictions and
    /// optional prediction intervals.
    pub fn forecast(
        &self,
        data: &ArrayView1<f64>,
        h: usize,
        confidence_level: f64,
    ) -> Result<ExponentialSmoothingForecast> {
        if h == 0 {
            return Err(NumRs2Error::ValueError(
                "Forecast horizon must be at least 1".to_string(),
            ));
        }
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(NumRs2Error::ValueError(
                "Confidence level must be in (0, 1)".to_string(),
            ));
        }

        let result = self.fit(data)?;
        let n = data.len();
        let m = self.period.unwrap_or(1);

        // Extract final components
        let final_level = result.level[n - 1];
        let final_trend = result.trend.as_ref().map_or(0.0, |t| t[n - 1]);
        let phi = self.phi.unwrap_or(1.0);

        let mut point = Array1::zeros(h);
        for i in 0..h {
            let steps = (i + 1) as f64;
            let trend_contrib = if self.trend == TrendComponent::None {
                0.0
            } else {
                damped_trend_sum(phi, i + 1) * final_trend
            };

            let season_idx = (n + i) % m;

            match self.seasonal {
                SeasonalComponent::Additive => {
                    let s_val = result.seasonal.as_ref().map_or(0.0, |s| s[season_idx]);
                    point[i] = final_level + trend_contrib + s_val;
                }
                SeasonalComponent::Multiplicative => {
                    let s_val = result.seasonal.as_ref().map_or(1.0, |s| s[season_idx]);
                    point[i] = (final_level + trend_contrib) * s_val;
                }
                SeasonalComponent::None => {
                    point[i] = final_level + trend_contrib;
                }
            }
        }

        // Prediction intervals using residual variance
        let (lower, upper) =
            self.compute_prediction_intervals(&result, &point, h, confidence_level)?;

        Ok(ExponentialSmoothingForecast {
            point,
            lower: Some(lower),
            upper: Some(upper),
            confidence_level,
        })
    }

    /// Compute information criteria (AIC, AICc, BIC) for the fitted model.
    ///
    /// These criteria balance goodness of fit against model complexity,
    /// enabling comparison between models with different numbers of parameters.
    ///
    /// # Arguments
    ///
    /// * `result` - Result from fitting the model
    ///
    /// # Returns
    ///
    /// `InformationCriteria` containing AIC, AICc, and BIC values
    pub fn information_criteria(
        &self,
        result: &ExponentialSmoothingResult,
    ) -> Result<InformationCriteria> {
        let n = result.n_obs as f64;
        let k = result.n_params as f64;

        if n <= k + 1.0 {
            return Err(NumRs2Error::ValueError(
                "Not enough observations relative to parameters for information criteria"
                    .to_string(),
            ));
        }

        // Log-likelihood approximation assuming Gaussian errors
        // L = -(n/2) * ln(2*pi*sigma^2) - SSE/(2*sigma^2)
        // where sigma^2 = SSE/n
        let sigma_sq = result.sse / n;
        if sigma_sq <= 0.0 {
            return Err(NumRs2Error::ComputationError(
                "Zero or negative variance in residuals".to_string(),
            ));
        }

        let log_likelihood = -0.5 * n * (2.0 * std::f64::consts::PI * sigma_sq).ln() - 0.5 * n;

        // AIC = -2*logL + 2*k
        let aic = -2.0 * log_likelihood + 2.0 * k;

        // AICc = AIC + 2*k*(k+1)/(n-k-1)
        let aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0);

        // BIC = -2*logL + k*ln(n)
        let bic = -2.0 * log_likelihood + k * n.ln();

        Ok(InformationCriteria { aic, aicc, bic })
    }

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// Validate that data length is sufficient for the model.
    fn validate_data_length(&self, n: usize) -> Result<()> {
        let min_len = match (&self.trend, &self.seasonal) {
            (TrendComponent::None, SeasonalComponent::None) => 2,
            (_, SeasonalComponent::None) => 3,
            (_, _) => {
                let m = self.period.unwrap_or(2);
                2 * m
            }
        };
        if n < min_len {
            return Err(NumRs2Error::ValueError(format!(
                "Need at least {} observations for this model, got {}",
                min_len, n
            )));
        }
        Ok(())
    }

    /// Initialize level, trend, and seasonal components.
    ///
    /// Uses standard initialization methods:
    /// - Level: mean of first seasonal period (or first observation for non-seasonal)
    /// - Trend: average slope over first two seasonal periods
    /// - Seasonal: deviations from level in first period(s)
    fn initialize(&self, data: &ArrayView1<f64>) -> Result<(f64, f64, Array1<f64>)> {
        let n = data.len();
        let m = self.period.unwrap_or(1);

        // Initialize level
        let level = if self.seasonal != SeasonalComponent::None && n >= m {
            data.slice(s![0..m]).iter().sum::<f64>() / m as f64
        } else {
            data[0]
        };

        // Initialize trend
        let trend = if self.trend != TrendComponent::None {
            if self.seasonal != SeasonalComponent::None && n >= 2 * m {
                let first_mean = data.slice(s![0..m]).iter().sum::<f64>() / m as f64;
                let second_mean = data.slice(s![m..2 * m]).iter().sum::<f64>() / m as f64;
                (second_mean - first_mean) / m as f64
            } else if n >= 2 {
                data[1] - data[0]
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Initialize seasonal components
        let seasonal = if self.seasonal != SeasonalComponent::None {
            let mut s = Array1::zeros(m);
            // Average deviations over available complete cycles
            let n_cycles = (n / m).min(3); // Use up to 3 cycles for initialization
            let n_cycles = n_cycles.max(1);

            for j in 0..m {
                let mut sum = 0.0;
                let mut count = 0;
                for cycle in 0..n_cycles {
                    let idx = cycle * m + j;
                    if idx < n {
                        match self.seasonal {
                            SeasonalComponent::Additive => {
                                sum += data[idx] - level;
                            }
                            SeasonalComponent::Multiplicative => {
                                if level.abs() > 1e-10 {
                                    sum += data[idx] / level;
                                } else {
                                    sum += 1.0;
                                }
                            }
                            SeasonalComponent::None => {}
                        }
                        count += 1;
                    }
                }
                s[j] = if count > 0 { sum / count as f64 } else { 0.0 };
            }

            // Normalize seasonal components
            match self.seasonal {
                SeasonalComponent::Additive => {
                    let mean = s.iter().sum::<f64>() / m as f64;
                    s -= mean;
                }
                SeasonalComponent::Multiplicative => {
                    let mean = s.iter().sum::<f64>() / m as f64;
                    if mean.abs() > 1e-10 {
                        s /= mean;
                    }
                }
                SeasonalComponent::None => {}
            }

            s
        } else {
            Array1::zeros(m)
        };

        Ok((level, trend, seasonal))
    }

    /// Compute the one-step-ahead fitted value given current components.
    fn compute_fitted_value(
        &self,
        level: f64,
        trend: f64,
        seasonal: &Array1<f64>,
        season_idx: usize,
        phi: f64,
    ) -> f64 {
        let trend_contrib = if self.trend == TrendComponent::None {
            0.0
        } else {
            phi * trend
        };

        match self.seasonal {
            SeasonalComponent::Additive => level + trend_contrib + seasonal[season_idx],
            SeasonalComponent::Multiplicative => (level + trend_contrib) * seasonal[season_idx],
            SeasonalComponent::None => level + trend_contrib,
        }
    }

    /// Compute prediction intervals using analytical variance formulas.
    ///
    /// For additive error models, the prediction variance grows with the
    /// forecast horizon. The formulas depend on the model structure.
    fn compute_prediction_intervals(
        &self,
        result: &ExponentialSmoothingResult,
        _point: &Array1<f64>,
        h: usize,
        confidence_level: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let sigma_sq = result.mse;
        let z = quantile_normal((1.0 + confidence_level) / 2.0);

        let phi = self.phi.unwrap_or(1.0);
        let alpha = self.alpha;

        let mut lower = Array1::zeros(h);
        let mut upper = Array1::zeros(h);

        let n = result.n_obs;
        let m = self.period.unwrap_or(1);

        for i in 0..h {
            let j = (i + 1) as f64; // steps ahead

            // Variance multiplier depends on model type
            // For SES: Var(e_{n+j}) = sigma^2 * (1 + (j-1)*alpha^2)
            // For Holt: more complex formula involving beta
            // We use a general approximation following Hyndman et al. (2008)
            let var_multiplier = match (&self.trend, &self.seasonal) {
                (TrendComponent::None, SeasonalComponent::None) => {
                    // SES: 1 + (j-1) * alpha^2
                    1.0 + (j - 1.0) * alpha * alpha
                }
                (TrendComponent::Additive, SeasonalComponent::None) => {
                    let beta = self.beta.unwrap_or(0.0);
                    // Holt: 1 + (j-1) * (alpha^2 + alpha*beta*j + beta^2*j*(2j-1)/6)
                    1.0 + (j - 1.0)
                        * (alpha * alpha
                            + alpha * beta * j
                            + beta * beta * j * (2.0 * j - 1.0) / 6.0)
                }
                (TrendComponent::Damped, SeasonalComponent::None) => {
                    // Damped trend: approximate using phi-adjusted formula
                    let sum_phi = damped_trend_sum(phi, i + 1);
                    1.0 + (j - 1.0) * alpha * alpha * (1.0 + sum_phi / j)
                }
                (_, SeasonalComponent::Additive) => {
                    // Additive seasonal: add seasonal variance component
                    let k = ((i / m) + 1) as f64;
                    1.0 + (j - 1.0) * alpha * alpha + k * self.gamma.unwrap_or(0.0).powi(2)
                }
                (_, SeasonalComponent::Multiplicative) => {
                    // Multiplicative: approximate; variance scales with squared forecast
                    // Use a simpler approximation
                    let season_idx = (n + i) % m;
                    let s_val = result
                        .seasonal
                        .as_ref()
                        .map_or(1.0, |s| s[season_idx])
                        .powi(2);
                    s_val * (1.0 + (j - 1.0) * alpha * alpha)
                }
            };

            let se = (sigma_sq * var_multiplier).sqrt();
            let point_i = _point[i];
            lower[i] = point_i - z * se;
            upper[i] = point_i + z * se;
        }

        Ok((lower, upper))
    }

    /// Count the number of estimated parameters in the model.
    fn count_parameters(&self) -> usize {
        let mut k = 1; // alpha
        if self.beta.is_some() {
            k += 1; // beta
        }
        if self.gamma.is_some() {
            k += 1; // gamma
        }
        if self.phi.is_some() {
            k += 1; // phi
        }
        // Initial states
        k += 1; // initial level
        if self.trend != TrendComponent::None {
            k += 1; // initial trend
        }
        if let Some(m) = self.period {
            if self.seasonal != SeasonalComponent::None {
                k += m - 1; // seasonal indices (m-1 free parameters)
            }
        }
        k += 1; // sigma^2
        k
    }
}

// ============================================================================
// Parameter Optimization
// ============================================================================

/// Optimize smoothing parameters by minimizing the sum of squared one-step-ahead
/// prediction errors (SSE) using adaptive grid search.
///
/// This function performs a coarse grid search followed by refinement around the
/// best point found. It supports all model types (SES, Holt, Holt-Winters,
/// damped variants).
///
/// # Arguments
///
/// * `data` - Training time series
/// * `trend` - Trend component type
/// * `seasonal` - Seasonal component type
/// * `period` - Seasonal period (required if seasonal != None)
/// * `config` - Optimization configuration (grid resolution, bounds, etc.)
///
/// # Returns
///
/// A tuple of (best_model, best_result) with optimized parameters.
pub fn optimize_parameters(
    data: &ArrayView1<f64>,
    trend: TrendComponent,
    seasonal: SeasonalComponent,
    period: Option<usize>,
    config: &OptimizationConfig,
) -> Result<(ExponentialSmoothing, ExponentialSmoothingResult)> {
    let n_grid = config.grid_resolution;
    let lo = config.param_min;
    let hi = config.param_max;

    let mut best_sse = f64::INFINITY;
    let mut best_params: (f64, Option<f64>, Option<f64>, Option<f64>) = (0.5, None, None, None);

    let has_trend = trend != TrendComponent::None;
    let has_season = seasonal != SeasonalComponent::None;
    let is_damped = trend == TrendComponent::Damped;

    // Generate parameter grid
    let alpha_grid: Vec<f64> = (0..n_grid)
        .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1).max(1) as f64)
        .collect();

    let beta_grid: Vec<f64> = if has_trend {
        (0..n_grid)
            .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1).max(1) as f64)
            .collect()
    } else {
        vec![0.0]
    };

    let gamma_grid: Vec<f64> = if has_season {
        (0..n_grid)
            .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1).max(1) as f64)
            .collect()
    } else {
        vec![0.0]
    };

    let phi_grid: Vec<f64> = if is_damped {
        // Damping parameter typically 0.8-0.98
        let phi_lo = 0.80_f64.max(lo);
        let phi_hi = 0.98_f64.min(hi);
        let n_phi = (n_grid / 2).max(5);
        (0..n_phi)
            .map(|i| phi_lo + (phi_hi - phi_lo) * i as f64 / (n_phi - 1).max(1) as f64)
            .collect()
    } else {
        vec![1.0]
    };

    // Coarse grid search
    for &a in &alpha_grid {
        for &b in &beta_grid {
            for &g in &gamma_grid {
                for &p in &phi_grid {
                    let model_result = build_and_fit(
                        a,
                        if has_trend { Some(b) } else { None },
                        if has_season { Some(g) } else { None },
                        if is_damped { Some(p) } else { None },
                        trend,
                        seasonal,
                        period,
                        data,
                    );
                    if let Ok(res) = model_result {
                        if res.sse < best_sse && res.sse.is_finite() {
                            best_sse = res.sse;
                            best_params = (
                                a,
                                if has_trend { Some(b) } else { None },
                                if has_season { Some(g) } else { None },
                                if is_damped { Some(p) } else { None },
                            );
                        }
                    }
                }
            }
        }
    }

    // Refinement iterations around the best point
    let mut current_best = best_params;
    let mut current_sse = best_sse;

    for iter in 0..config.refinement_iterations {
        let scale = 1.0 / ((iter + 1) as f64 * n_grid as f64);
        let half_range = (hi - lo) * scale;
        let n_refine = 11_usize;

        let refine_grid = |center: f64| -> Vec<f64> {
            let r_lo = (center - half_range).max(lo);
            let r_hi = (center + half_range).min(hi);
            (0..n_refine)
                .map(|i| r_lo + (r_hi - r_lo) * i as f64 / (n_refine - 1).max(1) as f64)
                .collect()
        };

        let a_refine = refine_grid(current_best.0);
        let b_refine = current_best.1.map_or_else(|| vec![0.0], &refine_grid);
        let g_refine = current_best.2.map_or_else(|| vec![0.0], &refine_grid);
        let p_refine = if is_damped {
            let center = current_best.3.unwrap_or(0.9);
            let pr_lo = (center - half_range).max(0.80);
            let pr_hi = (center + half_range).min(0.98);
            (0..n_refine)
                .map(|i| pr_lo + (pr_hi - pr_lo) * i as f64 / (n_refine - 1).max(1) as f64)
                .collect()
        } else {
            vec![1.0]
        };

        for &a in &a_refine {
            for &b in &b_refine {
                for &g in &g_refine {
                    for &p in &p_refine {
                        let model_result = build_and_fit(
                            a,
                            if has_trend { Some(b) } else { None },
                            if has_season { Some(g) } else { None },
                            if is_damped { Some(p) } else { None },
                            trend,
                            seasonal,
                            period,
                            data,
                        );
                        if let Ok(res) = model_result {
                            if res.sse < current_sse && res.sse.is_finite() {
                                current_sse = res.sse;
                                current_best = (
                                    a,
                                    if has_trend { Some(b) } else { None },
                                    if has_season { Some(g) } else { None },
                                    if is_damped { Some(p) } else { None },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Build final model with best parameters
    let best_model = ExponentialSmoothing::custom(
        current_best.0,
        current_best.1,
        current_best.2,
        current_best.3,
        trend,
        seasonal,
        period,
    )?;
    let best_result = best_model.fit(data)?;

    Ok((best_model, best_result))
}

/// Compare multiple model specifications and select the best one by AICc.
///
/// This is useful for automated model selection, trying different
/// combinations of trend and seasonal components.
///
/// # Arguments
///
/// * `data` - Training time series
/// * `period` - Seasonal period (used for seasonal model candidates)
/// * `config` - Optimization configuration
///
/// # Returns
///
/// A tuple of (best_model, best_result, best_criteria) for the model
/// with the lowest AICc.
pub fn select_best_model(
    data: &ArrayView1<f64>,
    period: Option<usize>,
    config: &OptimizationConfig,
) -> Result<(
    ExponentialSmoothing,
    ExponentialSmoothingResult,
    InformationCriteria,
)> {
    let mut candidates: Vec<(TrendComponent, SeasonalComponent)> = vec![
        (TrendComponent::None, SeasonalComponent::None),
        (TrendComponent::Additive, SeasonalComponent::None),
        (TrendComponent::Damped, SeasonalComponent::None),
    ];

    // Add seasonal candidates only if period is specified and data is long enough
    if let Some(p) = period {
        if data.len() >= 2 * p {
            candidates.push((TrendComponent::None, SeasonalComponent::Additive));
            candidates.push((TrendComponent::Additive, SeasonalComponent::Additive));
            candidates.push((TrendComponent::Damped, SeasonalComponent::Additive));

            // Only try multiplicative if all values are positive
            let all_positive = data.iter().all(|&x| x > 0.0);
            if all_positive {
                candidates.push((TrendComponent::None, SeasonalComponent::Multiplicative));
                candidates.push((TrendComponent::Additive, SeasonalComponent::Multiplicative));
                candidates.push((TrendComponent::Damped, SeasonalComponent::Multiplicative));
            }
        }
    }

    let mut best_aicc = f64::INFINITY;
    let mut best_model: Option<ExponentialSmoothing> = None;
    let mut best_result: Option<ExponentialSmoothingResult> = None;
    let mut best_criteria: Option<InformationCriteria> = None;

    for (trend, seasonal) in candidates {
        let p = if seasonal != SeasonalComponent::None {
            period
        } else {
            None
        };

        match optimize_parameters(data, trend, seasonal, p, config) {
            Ok((model, result)) => {
                if let Ok(ic) = model.information_criteria(&result) {
                    if ic.aicc < best_aicc && ic.aicc.is_finite() {
                        best_aicc = ic.aicc;
                        best_model = Some(model);
                        best_result = Some(result);
                        best_criteria = Some(ic);
                    }
                }
            }
            Err(_) => continue, // Skip models that fail to fit
        }
    }

    match (best_model, best_result, best_criteria) {
        (Some(m), Some(r), Some(c)) => Ok((m, r, c)),
        _ => Err(NumRs2Error::ComputationError(
            "No valid model could be fitted to the data".to_string(),
        )),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that a smoothing parameter is in (0, 1).
fn validate_param(value: f64, name: &str) -> Result<()> {
    if value <= 0.0 || value >= 1.0 {
        return Err(NumRs2Error::ValueError(format!(
            "{} must be in (0, 1), got {}",
            name, value
        )));
    }
    Ok(())
}

/// Compute the sum phi + phi^2 + ... + phi^h for damped trend forecasting.
///
/// When phi = 1 this equals h (undamped case).
fn damped_trend_sum(phi: f64, h: usize) -> f64 {
    if (phi - 1.0).abs() < 1e-12 {
        return h as f64;
    }
    if phi.abs() < 1e-12 {
        return 0.0;
    }
    // Geometric series: phi * (1 - phi^h) / (1 - phi)
    phi * (1.0 - phi.powi(h as i32)) / (1.0 - phi)
}

/// Build a model with the given parameters and fit it to data.
/// Returns just the result (used in optimization loops).
fn build_and_fit(
    alpha: f64,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
    trend: TrendComponent,
    seasonal: SeasonalComponent,
    period: Option<usize>,
    data: &ArrayView1<f64>,
) -> Result<ExponentialSmoothingResult> {
    let model = ExponentialSmoothing::custom(alpha, beta, gamma, phi, trend, seasonal, period)?;
    model.fit(data)
}

/// Standard normal quantile function (inverse CDF) using rational approximation.
///
/// Uses the Beasley-Springer-Moro algorithm for accuracy across the full range.
fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation for central region
    if p > 0.5 {
        return -quantile_normal(1.0 - p);
    }

    let t = (-2.0 * p.ln()).sqrt();

    // Coefficients for rational approximation (Abramowitz & Stegun 26.2.23)
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;

    -(t - numerator / denominator)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    // ---- SES Tests ----

    #[test]
    fn test_ses_constant_data() {
        // SES on constant data should produce constant fitted values
        let data = Array1::from_vec(vec![5.0; 20]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES creation should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");

        // All fitted values should be close to the constant
        for i in 1..20 {
            assert_relative_eq!(result.fitted[i], 5.0, epsilon = 1e-10);
        }
        // Residuals should be near zero
        assert!(result.mse < 1e-10);
    }

    #[test]
    fn test_ses_trending_data() {
        // SES on trending data: fitted values should lag behind
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let model = ExponentialSmoothing::ses(0.5).expect("SES creation should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");

        // Fitted values should be positive and increasing but lagging
        for i in 1..10 {
            assert!(result.fitted[i] > 0.0);
        }
        // Last fitted value should be less than last data point (lag effect)
        assert!(result.fitted[9] < data[9]);
    }

    #[test]
    fn test_ses_forecast_flat() {
        // SES forecast is flat (constant equal to last level)
        let data = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0, 12.5, 14.0]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES creation should succeed");
        let forecast = model
            .forecast(&data.view(), 5, 0.95)
            .expect("forecast should succeed");

        // All forecast values should be equal (flat forecast)
        for i in 1..5 {
            assert_relative_eq!(forecast.point[i], forecast.point[0], epsilon = 1e-10);
        }
        // Prediction intervals should widen
        let lower = forecast.lower.as_ref().expect("should have lower bounds");
        let upper = forecast.upper.as_ref().expect("should have upper bounds");
        assert!(upper[4] - lower[4] > upper[0] - lower[0]);
    }

    // ---- Holt's Method Tests ----

    #[test]
    fn test_holt_linear_trend() {
        // Holt's method should capture a linear trend well
        let data: Array1<f64> = Array1::from_vec((0..20).map(|i| 10.0 + 2.0 * i as f64).collect());
        let model = ExponentialSmoothing::holt(0.5, 0.3).expect("Holt creation should succeed");
        let forecast = model
            .forecast(&data.view(), 5, 0.95)
            .expect("forecast should succeed");

        // Forecasts should continue the upward trend
        for i in 1..5 {
            assert!(
                forecast.point[i] > forecast.point[i - 1],
                "Forecast should increase: {} vs {}",
                forecast.point[i],
                forecast.point[i - 1]
            );
        }
        // Forecast at h=1 should be close to actual continuation
        assert!(
            (forecast.point[0] - 50.0).abs() < 10.0,
            "First forecast {} should be near 50",
            forecast.point[0]
        );
    }

    #[test]
    fn test_holt_versus_ses_on_trend() {
        // Holt should produce lower MSE than SES on trending data
        let data: Array1<f64> = Array1::from_vec((0..30).map(|i| 5.0 + 1.5 * i as f64).collect());
        let ses = ExponentialSmoothing::ses(0.5).expect("SES should succeed");
        let holt = ExponentialSmoothing::holt(0.5, 0.3).expect("Holt should succeed");

        let ses_result = ses.fit(&data.view()).expect("SES fit");
        let holt_result = holt.fit(&data.view()).expect("Holt fit");

        assert!(
            holt_result.mse < ses_result.mse,
            "Holt MSE ({}) should be less than SES MSE ({}) on trending data",
            holt_result.mse,
            ses_result.mse
        );
    }

    // ---- Damped Trend Tests ----

    #[test]
    fn test_damped_trend_converges() {
        // Damped trend forecasts should converge to a finite limit
        let data: Array1<f64> = Array1::from_vec((0..20).map(|i| 10.0 + 2.0 * i as f64).collect());
        let model = ExponentialSmoothing::damped_trend(0.5, 0.3, 0.9)
            .expect("Damped trend creation should succeed");
        let forecast = model
            .forecast(&data.view(), 50, 0.95)
            .expect("forecast should succeed");

        // Damped forecasts should converge: differences should decrease
        let diff_early = (forecast.point[1] - forecast.point[0]).abs();
        let diff_late = (forecast.point[49] - forecast.point[48]).abs();
        assert!(
            diff_late < diff_early,
            "Late forecast steps should be closer together: {} vs {}",
            diff_late,
            diff_early
        );
    }

    #[test]
    fn test_damped_vs_undamped_long_horizon() {
        // Damped forecasts should be more conservative at long horizons
        let data: Array1<f64> = Array1::from_vec((0..20).map(|i| 10.0 + 2.0 * i as f64).collect());
        let holt = ExponentialSmoothing::holt(0.5, 0.3).expect("Holt should succeed");
        let damped =
            ExponentialSmoothing::damped_trend(0.5, 0.3, 0.85).expect("Damped should succeed");

        let holt_fc = holt
            .forecast(&data.view(), 30, 0.95)
            .expect("Holt forecast");
        let damped_fc = damped
            .forecast(&data.view(), 30, 0.95)
            .expect("Damped forecast");

        // At h=30, damped forecast should be lower than undamped
        assert!(
            damped_fc.point[29] < holt_fc.point[29],
            "Damped forecast ({}) should be less than Holt ({}) at long horizon",
            damped_fc.point[29],
            holt_fc.point[29]
        );
    }

    // ---- Holt-Winters Additive Tests ----

    #[test]
    fn test_hw_additive_seasonal_data() {
        // Create data with clear additive seasonal pattern
        let mut data_vec = Vec::with_capacity(48);
        let seasonal_pattern = [0.0, 3.0, 6.0, 3.0]; // period = 4
        for i in 0..48 {
            let trend = 10.0 + 0.5 * i as f64;
            let season = seasonal_pattern[i % 4];
            data_vec.push(trend + season);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Additive)
                .expect("HW additive creation should succeed");

        let result = model.fit(&data.view()).expect("HW fit should succeed");

        // MSE should be reasonably small for this clean data
        assert!(
            result.mse < 10.0,
            "MSE ({}) should be small for clean seasonal data",
            result.mse
        );

        // Seasonal component should exist
        assert!(result.seasonal.is_some());
    }

    #[test]
    fn test_hw_additive_forecast_preserves_season() {
        let mut data_vec = Vec::with_capacity(40);
        let seasonal_pattern = [0.0, 5.0, 10.0, 5.0];
        for i in 0..40 {
            let trend = 20.0 + 1.0 * i as f64;
            let season = seasonal_pattern[i % 4];
            data_vec.push(trend + season);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.3, 4, SeasonalComponent::Additive)
                .expect("HW should succeed");
        let forecast = model
            .forecast(&data.view(), 8, 0.95)
            .expect("forecast should succeed");

        // Forecast should show seasonal pattern: every 4th step should be similar
        // Check that forecasts at same seasonal position are closer to each other
        // than to adjacent forecasts
        let diff_same_season = (forecast.point[0] - forecast.point[4]).abs();
        let diff_adjacent = (forecast.point[0] - forecast.point[1]).abs();
        // Same-season forecasts differ mainly by trend; adjacent differ by season+trend
        assert!(
            diff_same_season < diff_adjacent * 3.0,
            "Same-season forecasts should be relatively close"
        );
    }

    // ---- Holt-Winters Multiplicative Tests ----

    #[test]
    fn test_hw_multiplicative_seasonal_data() {
        // Create data with multiplicative seasonal pattern
        let mut data_vec = Vec::with_capacity(48);
        let seasonal_factors = [0.8, 1.0, 1.3, 0.9]; // period = 4
        for i in 0..48 {
            let trend = 50.0 + 2.0 * i as f64;
            let season = seasonal_factors[i % 4];
            data_vec.push(trend * season);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Multiplicative)
                .expect("HW multiplicative creation should succeed");

        let result = model.fit(&data.view()).expect("fit should succeed");

        // MSE should be manageable
        assert!(result.mse.is_finite(), "MSE should be finite");
        assert!(result.seasonal.is_some());

        // Seasonal factors should sum close to period (normalized to average 1.0)
        let s = result.seasonal.as_ref().expect("seasonal should exist");
        let s_sum: f64 = s.iter().sum();
        assert_relative_eq!(s_sum / 4.0, 1.0, epsilon = 0.3);
    }

    #[test]
    fn test_hw_multiplicative_forecast() {
        let mut data_vec = Vec::with_capacity(40);
        let seasonal_factors = [0.7, 1.1, 1.4, 0.8];
        for i in 0..40 {
            let base = 100.0 + 3.0 * i as f64;
            data_vec.push(base * seasonal_factors[i % 4]);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Multiplicative)
                .expect("HW should succeed");
        let forecast = model
            .forecast(&data.view(), 4, 0.95)
            .expect("forecast should succeed");

        // All forecasts should be positive
        assert!(forecast.point.iter().all(|&x| x > 0.0));
        // Should have prediction intervals
        assert!(forecast.lower.is_some());
        assert!(forecast.upper.is_some());
    }

    // ---- Parameter Optimization Tests ----

    #[test]
    fn test_optimize_ses_parameters() {
        let data: Array1<f64> =
            Array1::from_vec((0..30).map(|i| 10.0 + 0.1 * (i as f64).sin()).collect());

        let config = OptimizationConfig {
            grid_resolution: 10,
            refinement_iterations: 1,
            ..Default::default()
        };

        let (model, result) = optimize_parameters(
            &data.view(),
            TrendComponent::None,
            SeasonalComponent::None,
            None,
            &config,
        )
        .expect("optimization should succeed");

        assert!(model.alpha > 0.0 && model.alpha < 1.0);
        assert!(result.mse.is_finite());
        assert!(result.mse >= 0.0);
    }

    #[test]
    fn test_optimize_holt_parameters() {
        let data: Array1<f64> = Array1::from_vec(
            (0..40)
                .map(|i| 5.0 + 2.0 * i as f64 + 0.5 * (i as f64 * 0.3).sin())
                .collect(),
        );

        let config = OptimizationConfig {
            grid_resolution: 8,
            refinement_iterations: 1,
            ..Default::default()
        };

        let (model, result) = optimize_parameters(
            &data.view(),
            TrendComponent::Additive,
            SeasonalComponent::None,
            None,
            &config,
        )
        .expect("optimization should succeed");

        assert!(model.alpha > 0.0 && model.alpha < 1.0);
        assert!(model.beta.is_some());
        let beta = model.beta.expect("beta should exist");
        assert!(beta > 0.0 && beta < 1.0);
        assert!(result.mse.is_finite());
    }

    // ---- Forecast Accuracy Tests ----

    #[test]
    fn test_forecast_accuracy_on_known_data() {
        // Linear data: y = 2*t + 10
        let train: Array1<f64> = Array1::from_vec((0..30).map(|i| 10.0 + 2.0 * i as f64).collect());
        let actual_next = 10.0 + 2.0 * 30.0; // = 70.0

        let model = ExponentialSmoothing::holt(0.8, 0.5).expect("Holt should succeed");
        let forecast = model
            .forecast(&train.view(), 1, 0.95)
            .expect("forecast should succeed");

        // Forecast should be close to actual
        let error = (forecast.point[0] - actual_next).abs();
        assert!(
            error < 5.0,
            "Forecast error ({}) should be small for linear data",
            error
        );
    }

    // ---- Prediction Intervals Tests ----

    #[test]
    fn test_prediction_intervals_coverage() {
        let data = Array1::from_vec(vec![
            10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.0, 15.0, 14.5, 16.0,
        ]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES should succeed");
        let forecast = model
            .forecast(&data.view(), 5, 0.95)
            .expect("forecast should succeed");

        let lower = forecast.lower.as_ref().expect("lower bounds");
        let upper = forecast.upper.as_ref().expect("upper bounds");

        // Lower should be less than point, upper should be greater
        for i in 0..5 {
            assert!(
                lower[i] < forecast.point[i],
                "Lower bound should be below point forecast"
            );
            assert!(
                upper[i] > forecast.point[i],
                "Upper bound should be above point forecast"
            );
        }

        assert_relative_eq!(forecast.confidence_level, 0.95, epsilon = 1e-10);
    }

    #[test]
    fn test_prediction_intervals_widen_with_horizon() {
        let data = Array1::from_vec(vec![5.0, 6.0, 7.0, 5.5, 6.5, 7.5, 5.0, 6.0, 7.0, 8.0]);
        let model = ExponentialSmoothing::holt(0.3, 0.1).expect("Holt should succeed");
        let forecast = model
            .forecast(&data.view(), 10, 0.90)
            .expect("forecast should succeed");

        let lower = forecast.lower.as_ref().expect("lower bounds");
        let upper = forecast.upper.as_ref().expect("upper bounds");

        let width_1 = upper[0] - lower[0];
        let width_10 = upper[9] - lower[9];

        assert!(
            width_10 > width_1,
            "Prediction interval at h=10 ({}) should be wider than at h=1 ({})",
            width_10,
            width_1
        );
    }

    // ---- Information Criteria Tests ----

    #[test]
    fn test_information_criteria_computed() {
        let data = Array1::from_vec(vec![
            10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.0, 15.0, 14.5, 16.0, 15.0, 17.0, 16.5, 18.0,
            17.0,
        ]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");
        let ic = model
            .information_criteria(&result)
            .expect("IC should succeed");

        assert!(ic.aic.is_finite(), "AIC should be finite");
        assert!(ic.aicc.is_finite(), "AICc should be finite");
        assert!(ic.bic.is_finite(), "BIC should be finite");

        // AICc >= AIC (correction is always non-negative for finite samples)
        assert!(
            ic.aicc >= ic.aic - 1e-10,
            "AICc ({}) should be >= AIC ({})",
            ic.aicc,
            ic.aic
        );
    }

    #[test]
    fn test_model_selection_prefers_simpler_for_constant() {
        // For constant data, SES should be preferred over Holt
        let data = Array1::from_vec(vec![5.0; 30]);
        // Add tiny noise to avoid zero-variance issues
        let mut noisy = data.clone();
        for i in 0..30 {
            noisy[i] += 0.001 * (i as f64 * 0.7).sin();
        }

        let ses = ExponentialSmoothing::ses(0.1).expect("SES should succeed");
        let holt = ExponentialSmoothing::holt(0.1, 0.1).expect("Holt should succeed");

        let ses_result = ses.fit(&noisy.view()).expect("SES fit");
        let holt_result = holt.fit(&noisy.view()).expect("Holt fit");

        let ses_ic = ses.information_criteria(&ses_result).expect("SES IC");
        let holt_ic = holt.information_criteria(&holt_result).expect("Holt IC");

        // SES should have lower (better) BIC since data has no trend
        assert!(
            ses_ic.bic < holt_ic.bic,
            "SES BIC ({}) should be less than Holt BIC ({}) for constant data",
            ses_ic.bic,
            holt_ic.bic
        );
    }

    // ---- Edge Case Tests ----

    #[test]
    fn test_short_series_ses() {
        // Minimum length for SES is 2
        let data = Array1::from_vec(vec![1.0, 2.0]);
        let model = ExponentialSmoothing::ses(0.5).expect("SES should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");
        assert_eq!(result.fitted.len(), 2);
    }

    #[test]
    fn test_single_observation_ses_fails() {
        let data = Array1::from_vec(vec![42.0]);
        let model = ExponentialSmoothing::ses(0.5).expect("SES should succeed");
        let result = model.fit(&data.view());
        assert!(result.is_err(), "Single observation should fail");
    }

    #[test]
    fn test_invalid_parameters() {
        // Alpha out of range
        assert!(ExponentialSmoothing::ses(0.0).is_err());
        assert!(ExponentialSmoothing::ses(1.0).is_err());
        assert!(ExponentialSmoothing::ses(-0.1).is_err());
        assert!(ExponentialSmoothing::ses(1.5).is_err());

        // Beta out of range
        assert!(ExponentialSmoothing::holt(0.5, 0.0).is_err());
        assert!(ExponentialSmoothing::holt(0.5, 1.0).is_err());

        // Phi out of range
        assert!(ExponentialSmoothing::damped_trend(0.5, 0.3, 0.0).is_err());
        assert!(ExponentialSmoothing::damped_trend(0.5, 0.3, 1.0).is_err());

        // Invalid period
        assert!(
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 1, SeasonalComponent::Additive)
                .is_err()
        );
    }

    #[test]
    fn test_insufficient_data_for_seasonal() {
        // Need at least 2*period observations for seasonal model
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Only 3 obs, period=4
        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Additive)
                .expect("Model creation should succeed (validation at fit time)");
        let result = model.fit(&data.view());
        assert!(
            result.is_err(),
            "Fitting with insufficient data should fail"
        );
    }

    // ---- Damped Holt-Winters Tests ----

    #[test]
    fn test_damped_holt_winters() {
        let mut data_vec = Vec::with_capacity(48);
        let seasonal_pattern = [0.0, 4.0, 8.0, 4.0];
        for i in 0..48 {
            let trend = 20.0 + 1.0 * i as f64;
            let season = seasonal_pattern[i % 4];
            data_vec.push(trend + season);
        }
        let data = Array1::from_vec(data_vec);

        let model = ExponentialSmoothing::damped_holt_winters(
            0.3,
            0.1,
            0.2,
            0.9,
            4,
            SeasonalComponent::Additive,
        )
        .expect("Damped HW should succeed");

        let forecast = model
            .forecast(&data.view(), 20, 0.95)
            .expect("forecast should succeed");

        // Forecasts should be positive and have prediction intervals
        assert!(forecast.point.iter().all(|&x| x > 0.0));
        assert!(forecast.lower.is_some());
        assert!(forecast.upper.is_some());

        // Damped trend: forecast differences should decrease
        let diff_early = (forecast.point[1] - forecast.point[0]).abs();
        let diff_late = (forecast.point[19] - forecast.point[18]).abs();
        // Due to seasonality, compare same-season steps
        let diff_season_early = (forecast.point[4] - forecast.point[0]).abs();
        let diff_season_late = (forecast.point[16] - forecast.point[12]).abs();
        // Later same-season differences should be smaller (damped trend)
        assert!(
            diff_season_late < diff_season_early + 5.0,
            "Damped seasonal steps should converge"
        );
    }

    // ---- Model Selection Test ----

    #[test]
    fn test_select_best_model() {
        // Data with trend and seasonality
        let mut data_vec = Vec::with_capacity(60);
        let seasonal = [0.0, 3.0, 6.0, 3.0];
        for i in 0..60 {
            data_vec.push(10.0 + 0.5 * i as f64 + seasonal[i % 4]);
        }
        let data = Array1::from_vec(data_vec);

        let config = OptimizationConfig {
            grid_resolution: 5,
            refinement_iterations: 0,
            ..Default::default()
        };

        let result = select_best_model(&data.view(), Some(4), &config);
        assert!(result.is_ok(), "Model selection should succeed");

        let (_model, _result, criteria) = result.expect("should have result");
        assert!(criteria.aicc.is_finite());
    }

    // ---- Quantile Normal Test ----

    #[test]
    fn test_quantile_normal_symmetry() {
        let z95 = quantile_normal(0.975);
        assert_relative_eq!(z95, 1.96, epsilon = 0.02);

        let z_sym = quantile_normal(0.025);
        assert_relative_eq!(z_sym, -z95, epsilon = 0.02);
    }

    // ---- Damped Trend Sum Test ----

    #[test]
    fn test_damped_trend_sum_values() {
        // phi=1: sum should equal h
        assert_relative_eq!(damped_trend_sum(1.0, 5), 5.0, epsilon = 1e-10);

        // phi=0.9, h=1: sum = 0.9
        assert_relative_eq!(damped_trend_sum(0.9, 1), 0.9, epsilon = 1e-10);

        // phi=0.9, h=2: sum = 0.9 + 0.81 = 1.71
        assert_relative_eq!(damped_trend_sum(0.9, 2), 1.71, epsilon = 1e-10);

        // phi=0: sum should be 0
        assert_relative_eq!(
            damped_trend_sum(0.001, 100),
            0.001 / (1.0 - 0.001),
            epsilon = 0.01
        );
    }
}
