//! Core `ExponentialSmoothing` struct and fitting/forecasting logic.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, ArrayView1};

use super::helpers::{damped_trend_sum, quantile_normal, validate_param};
use super::types::{
    ExponentialSmoothingForecast, ExponentialSmoothingResult, InformationCriteria,
    SeasonalComponent, TrendComponent,
};

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
    pub(super) alpha: f64,
    /// Trend smoothing parameter (None for SES).
    pub(super) beta: Option<f64>,
    /// Seasonal smoothing parameter (None for non-seasonal models).
    pub(super) gamma: Option<f64>,
    /// Trend damping parameter (None for undamped models).
    pub(super) phi: Option<f64>,
    /// Type of trend component.
    pub(super) trend: TrendComponent,
    /// Type of seasonal component.
    pub(super) seasonal: SeasonalComponent,
    /// Seasonal period (None for non-seasonal models).
    pub(super) period: Option<usize>,
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
    pub(super) fn count_parameters(&self) -> usize {
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
