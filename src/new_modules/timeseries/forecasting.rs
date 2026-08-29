//! Forecasting utilities: accuracy metrics, cross-validation, forecast combination,
//! prediction intervals, and naive baselines.
//!
//! Also re-exports exponential smoothing wrappers (SES, Holt, Holt-Winters) for
//! backward compatibility. For advanced ETS models see `exponential_smoothing` module.
//!
//! ## References
//!
//! - Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice* (3rd ed.).
//! - Hyndman & Koehler (2006). *Int. J. Forecasting*, 22(4), 679-688.
//! - Theil (1966). *Applied Economic Forecasting*.
//! - Timmermann (2006). *Handbook of Economic Forecasting*, 1, 135-196.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use scirs2_core::random::thread_rng;

/// Forecast result with point forecasts and optional intervals.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point forecasts
    pub forecasts: Array1<f64>,
    /// Lower bounds of prediction intervals (if available)
    pub lower_bounds: Option<Array1<f64>>,
    /// Upper bounds of prediction intervals (if available)
    pub upper_bounds: Option<Array1<f64>>,
    /// Fitted values on training data
    pub fitted: Array1<f64>,
    /// Residuals on training data
    pub residuals: Array1<f64>,
}

/// Simple Exponential Smoothing (SES) for level-only data.
///
/// This is a convenience re-export wrapper. For advanced ETS functionality,
/// use the `exponential_smoothing` module directly.
#[derive(Debug, Clone)]
pub struct SimpleExponentialSmoothing {
    /// Smoothing parameter alpha (0 < alpha < 1)
    pub alpha: f64,
}

impl SimpleExponentialSmoothing {
    /// Create new SES model with smoothing parameter alpha in (0, 1).
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(NumRs2Error::ValueError(
                "Alpha must be in (0, 1)".to_string(),
            ));
        }
        Ok(Self { alpha })
    }

    /// Fit model and forecast h steps ahead.
    pub fn fit_forecast(&self, data: &ArrayView1<f64>, h: usize) -> Result<ForecastResult> {
        let n = data.len();
        if n < 2 {
            return Err(NumRs2Error::ValueError(
                "Need at least 2 observations".to_string(),
            ));
        }
        let mut level = data[0];
        let mut fitted = Array1::zeros(n);
        fitted[0] = level;
        for i in 1..n {
            level = self.alpha * data[i] + (1.0 - self.alpha) * level;
            fitted[i] = level;
        }
        let residuals = data - &fitted;
        let forecasts = Array1::from_elem(h, level);
        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }

    /// Optimize alpha using one-step-ahead SSE minimization (grid search).
    pub fn optimize_alpha(data: &ArrayView1<f64>) -> Result<f64> {
        let n = data.len();
        if n < 3 {
            return Err(NumRs2Error::ValueError(
                "Need at least 3 observations".to_string(),
            ));
        }
        let mut best_alpha = 0.5;
        let mut best_sse = f64::INFINITY;
        for a in 1..100 {
            let alpha = a as f64 / 100.0;
            let mut sse = 0.0;
            let mut level = data[0];
            for i in 1..n {
                sse += (data[i] - level).powi(2);
                level = alpha * data[i] + (1.0 - alpha) * level;
            }
            if sse < best_sse {
                best_sse = sse;
                best_alpha = alpha;
            }
        }
        Ok(best_alpha)
    }
}

/// Holt's Linear Trend Method (Double Exponential Smoothing).
#[derive(Debug, Clone)]
pub struct HoltLinearTrend {
    /// Level smoothing parameter
    pub alpha: f64,
    /// Trend smoothing parameter
    pub beta: f64,
}

impl HoltLinearTrend {
    /// Create new Holt's linear trend model. Both parameters must be in (0, 1).
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 || beta <= 0.0 || beta >= 1.0 {
            return Err(NumRs2Error::ValueError(
                "Alpha and beta must be in (0, 1)".to_string(),
            ));
        }
        Ok(Self { alpha, beta })
    }

    /// Fit model and forecast h steps ahead.
    pub fn fit_forecast(&self, data: &ArrayView1<f64>, h: usize) -> Result<ForecastResult> {
        let n = data.len();
        if n < 3 {
            return Err(NumRs2Error::ValueError(
                "Need at least 3 observations".to_string(),
            ));
        }
        let mut level = data[0];
        let mut trend = data[1] - data[0];
        let mut fitted = Array1::zeros(n);
        fitted[0] = level;
        for i in 1..n {
            let prev = level;
            level = self.alpha * data[i] + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - prev) + (1.0 - self.beta) * trend;
            fitted[i] = level + trend;
        }
        let residuals = data - &fitted;
        let mut forecasts = Array1::zeros(h);
        for i in 0..h {
            forecasts[i] = level + (i + 1) as f64 * trend;
        }
        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }
}

/// Holt-Winters Seasonal Method (Triple Exponential Smoothing).
/// Supports additive and multiplicative seasonality.
#[derive(Debug, Clone)]
pub struct HoltWinters {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub period: usize,
    pub multiplicative: bool,
}

impl HoltWinters {
    /// Create new Holt-Winters model. Smoothing parameters in (0,1), period >= 2.
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        multiplicative: bool,
    ) -> Result<Self> {
        if alpha <= 0.0
            || alpha >= 1.0
            || beta <= 0.0
            || beta >= 1.0
            || gamma <= 0.0
            || gamma >= 1.0
        {
            return Err(NumRs2Error::ValueError(
                "Smoothing parameters must be in (0, 1)".to_string(),
            ));
        }
        if period < 2 {
            return Err(NumRs2Error::ValueError(
                "Period must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            beta,
            gamma,
            period,
            multiplicative,
        })
    }

    /// Fit model and forecast h steps ahead.
    pub fn fit_forecast(&self, data: &ArrayView1<f64>, h: usize) -> Result<ForecastResult> {
        let n = data.len();
        let m = self.period;
        if n < 2 * m {
            return Err(NumRs2Error::ValueError(format!(
                "Need at least {} observations (2 * period)",
                2 * m
            )));
        }
        let (mut level, mut trend, mut seasonal) = self.init_components(data)?;
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            let si = i % m;
            if self.multiplicative {
                fitted[i] = (level + trend) * seasonal[si];
                if i < n - 1 {
                    let prev = level;
                    level = self.alpha * (data[i] / seasonal[si].max(1e-10))
                        + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta * (level - prev) + (1.0 - self.beta) * trend;
                    seasonal[si] = self.gamma * (data[i] / level.max(1e-10))
                        + (1.0 - self.gamma) * seasonal[si];
                }
            } else {
                fitted[i] = level + trend + seasonal[si];
                if i < n - 1 {
                    let prev = level;
                    level = self.alpha * (data[i] - seasonal[si])
                        + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta * (level - prev) + (1.0 - self.beta) * trend;
                    seasonal[si] =
                        self.gamma * (data[i] - level) + (1.0 - self.gamma) * seasonal[si];
                }
            }
        }
        let residuals = data - &fitted;
        let mut forecasts = Array1::zeros(h);
        for i in 0..h {
            let si = (n + i) % m;
            forecasts[i] = if self.multiplicative {
                (level + (i + 1) as f64 * trend) * seasonal[si]
            } else {
                level + (i + 1) as f64 * trend + seasonal[si]
            };
        }
        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }

    fn init_components(&self, data: &ArrayView1<f64>) -> Result<(f64, f64, Array1<f64>)> {
        let m = self.period;
        let n = data.len();
        let level = data.slice(s![0..m]).iter().sum::<f64>() / m as f64;
        let first_mean = level;
        let second_mean = if n >= 2 * m {
            data.slice(s![m..2 * m]).iter().sum::<f64>() / m as f64
        } else {
            first_mean
        };
        let trend = (second_mean - first_mean) / m as f64;
        let mut seasonal = Array1::zeros(m);
        for si in 0..m {
            let mut sum = 0.0;
            let mut count = 0;
            for i in (si..n).step_by(m) {
                sum += if self.multiplicative {
                    data[i] / level.max(1e-10)
                } else {
                    data[i] - level
                };
                count += 1;
            }
            seasonal[si] = if count > 0 { sum / count as f64 } else { 0.0 };
        }
        if self.multiplicative {
            let sm = seasonal.iter().sum::<f64>() / m as f64;
            if sm.abs() > 1e-10 {
                seasonal /= sm;
            }
        } else {
            let sm = seasonal.iter().sum::<f64>() / m as f64;
            seasonal -= sm;
        }
        Ok((level, trend, seasonal))
    }
}

/// Forecast accuracy metrics.
pub struct ForecastMetrics;

impl ForecastMetrics {
    /// Mean Absolute Error: MAE = (1/n) Σ|yᵢ - ŷᵢ|
    pub fn mae(actual: &ArrayView1<f64>, forecast: &ArrayView1<f64>) -> Result<f64> {
        if actual.len() != forecast.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Actual and forecast must have same length".to_string(),
            ));
        }

        let n = actual.len() as f64;
        let mae = actual
            .iter()
            .zip(forecast.iter())
            .map(|(&a, &f)| (a - f).abs())
            .sum::<f64>()
            / n;

        Ok(mae)
    }

    /// Root Mean Squared Error: RMSE = sqrt(MSE)
    pub fn rmse(actual: &ArrayView1<f64>, forecast: &ArrayView1<f64>) -> Result<f64> {
        if actual.len() != forecast.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Actual and forecast must have same length".to_string(),
            ));
        }

        let n = actual.len() as f64;
        let mse = actual
            .iter()
            .zip(forecast.iter())
            .map(|(&a, &f)| (a - f).powi(2))
            .sum::<f64>()
            / n;

        Ok(mse.sqrt())
    }

    /// Mean Absolute Percentage Error: MAPE = (100/n) Σ|yᵢ - ŷᵢ| / |yᵢ|
    pub fn mape(actual: &ArrayView1<f64>, forecast: &ArrayView1<f64>) -> Result<f64> {
        if actual.len() != forecast.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Actual and forecast must have same length".to_string(),
            ));
        }

        let n = actual.len() as f64;
        let mape = actual
            .iter()
            .zip(forecast.iter())
            .filter(|(&a, _)| a.abs() > 1e-10) // Avoid division by zero
            .map(|(&a, &f)| (a - f).abs() / a.abs())
            .sum::<f64>()
            / n
            * 100.0;

        Ok(mape)
    }

    /// Mean Absolute Scaled Error: scale-independent metric vs naive forecast.
    pub fn mase(
        actual: &ArrayView1<f64>,
        forecast: &ArrayView1<f64>,
        training: &ArrayView1<f64>,
    ) -> Result<f64> {
        if actual.len() != forecast.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Actual and forecast must have same length".to_string(),
            ));
        }

        if training.len() < 2 {
            return Err(NumRs2Error::ValueError(
                "Training set must have at least 2 observations".to_string(),
            ));
        }

        let mae = Self::mae(actual, forecast)?;

        // Compute scaling factor (mean absolute difference in training set)
        let n_train = training.len();
        let mut scale = 0.0;
        for i in 1..n_train {
            scale += (training[i] - training[i - 1]).abs();
        }
        scale /= (n_train - 1) as f64;

        if scale < 1e-10 {
            return Err(NumRs2Error::ValueError(
                "Training set has zero variation".to_string(),
            ));
        }

        Ok(mae / scale)
    }

    /// Symmetric MAPE: sMAPE = (100/n) Σ 2|yᵢ - ŷᵢ| / (|yᵢ| + |ŷᵢ|)
    pub fn smape(actual: &ArrayView1<f64>, forecast: &ArrayView1<f64>) -> Result<f64> {
        if actual.len() != forecast.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Actual and forecast must have same length".to_string(),
            ));
        }

        let n = actual.len() as f64;
        let smape = actual
            .iter()
            .zip(forecast.iter())
            .map(|(&a, &f)| {
                let denom = a.abs() + f.abs();
                if denom > 1e-10 {
                    2.0 * (a - f).abs() / denom
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n
            * 100.0;

        Ok(smape)
    }

    /// Mean Squared Error: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
    pub fn mse(actual: &ArrayView1<f64>, forecast: &ArrayView1<f64>) -> Result<f64> {
        if actual.len() != forecast.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Actual and forecast must have same length".to_string(),
            ));
        }

        if actual.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Cannot compute MSE for empty arrays".to_string(),
            ));
        }

        let n = actual.len() as f64;
        let mse = actual
            .iter()
            .zip(forecast.iter())
            .map(|(&a, &f)| (a - f).powi(2))
            .sum::<f64>()
            / n;

        Ok(mse)
    }

    /// Theil's U statistic: U = sqrt(SS_forecast / SS_naive). U<1 beats naive.
    pub fn theils_u(
        actual: &ArrayView1<f64>,
        forecast: &ArrayView1<f64>,
        previous: &ArrayView1<f64>,
    ) -> Result<f64> {
        let n = actual.len();
        if n != forecast.len() || n != previous.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "actual, forecast, and previous must have same length".to_string(),
            ));
        }

        if n == 0 {
            return Err(NumRs2Error::ValueError(
                "Cannot compute Theil's U for empty arrays".to_string(),
            ));
        }

        // Numerator: sum of squared forecast errors
        let forecast_ss: f64 = actual
            .iter()
            .zip(forecast.iter())
            .map(|(&a, &f)| (a - f).powi(2))
            .sum();

        // Denominator: sum of squared naive errors (random walk)
        let naive_ss: f64 = actual
            .iter()
            .zip(previous.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum();

        if naive_ss < 1e-15 {
            return Err(NumRs2Error::ComputationError(
                "Naive forecast has zero error; Theil's U is undefined".to_string(),
            ));
        }

        Ok((forecast_ss / naive_ss).sqrt())
    }
}

/// Time series cross-validation for forecast evaluation.
pub struct TimeSeriesCrossValidation;

impl TimeSeriesCrossValidation {
    /// Rolling origin cross-validation with expanding training window.
    pub fn rolling_origin<F>(
        data: &ArrayView1<f64>,
        initial_window: usize,
        horizon: usize,
        step_size: usize,
        forecast_fn: F,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&ArrayView1<f64>, usize) -> Result<Array1<f64>>,
    {
        let n = data.len();
        let mut errors = Vec::new();

        if initial_window + horizon > n {
            return Err(NumRs2Error::ValueError(
                "Initial window + horizon exceeds data length".to_string(),
            ));
        }

        let mut train_end = initial_window;

        while train_end + horizon <= n {
            let train = data.slice(s![0..train_end]);
            let test = data.slice(s![train_end..train_end + horizon]);

            let forecasts = forecast_fn(&train, horizon)?;
            let mae = ForecastMetrics::mae(&test, &forecasts.view())?;
            errors.push(mae);

            train_end += step_size;
        }

        Ok(errors)
    }

    /// Expanding window cross-validation (alias for `rolling_origin`).
    pub fn expanding_window<F>(
        data: &ArrayView1<f64>,
        initial_window: usize,
        horizon: usize,
        step_size: usize,
        forecast_fn: F,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&ArrayView1<f64>, usize) -> Result<Array1<f64>>,
    {
        Self::rolling_origin(data, initial_window, horizon, step_size, forecast_fn)
    }

    /// Rolling (fixed-size) window cross-validation. Training window slides forward.
    pub fn rolling_window<F>(
        data: &ArrayView1<f64>,
        window_size: usize,
        horizon: usize,
        step_size: usize,
        forecast_fn: F,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&ArrayView1<f64>, usize) -> Result<Array1<f64>>,
    {
        let n = data.len();
        let mut errors = Vec::new();

        if window_size + horizon > n {
            return Err(NumRs2Error::ValueError(
                "Window size + horizon exceeds data length".to_string(),
            ));
        }

        if window_size == 0 {
            return Err(NumRs2Error::ValueError(
                "Window size must be positive".to_string(),
            ));
        }

        let mut train_start = 0usize;

        while train_start + window_size + horizon <= n {
            let train = data.slice(s![train_start..train_start + window_size]);
            let test_start = train_start + window_size;
            let test = data.slice(s![test_start..test_start + horizon]);

            let forecasts = forecast_fn(&train, horizon)?;
            let mae = ForecastMetrics::mae(&test, &forecasts.view())?;
            errors.push(mae);

            train_start += step_size;
        }

        Ok(errors)
    }

    /// Temporal train/test split preserving temporal ordering.
    pub fn temporal_split(
        data: &ArrayView1<f64>,
        train_ratio: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(NumRs2Error::ValueError(
                "train_ratio must be in (0, 1)".to_string(),
            ));
        }

        let n = data.len();
        if n < 2 {
            return Err(NumRs2Error::ValueError(
                "Need at least 2 observations for splitting".to_string(),
            ));
        }

        let split_idx = ((n as f64) * train_ratio).round() as usize;
        let split_idx = split_idx.max(1).min(n - 1);

        let train = data.slice(s![0..split_idx]).to_owned();
        let test = data.slice(s![split_idx..n]).to_owned();

        Ok((train, test))
    }
}

/// Methods for combining multiple forecasts (Timmermann, 2006).
pub struct ForecastCombiner;

impl ForecastCombiner {
    /// Simple average (equal weight) combination of multiple forecasts.
    pub fn simple_average(forecasts: &[&ArrayView1<f64>]) -> Result<Array1<f64>> {
        if forecasts.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Need at least one forecast to combine".to_string(),
            ));
        }

        let n = forecasts[0].len();
        for (i, fc) in forecasts.iter().enumerate() {
            if fc.len() != n {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Forecast {} has length {} but expected {}",
                    i,
                    fc.len(),
                    n
                )));
            }
        }

        let k = forecasts.len() as f64;
        let mut combined = Array1::zeros(n);
        for fc in forecasts {
            for j in 0..n {
                combined[j] += fc[j];
            }
        }
        combined /= k;

        Ok(combined)
    }

    /// Weighted average with weights inversely proportional to validation MSE.
    pub fn weighted_average(
        forecasts: &[&ArrayView1<f64>],
        past_actuals: &ArrayView1<f64>,
        past_forecasts: &[&ArrayView1<f64>],
    ) -> Result<Array1<f64>> {
        let k = forecasts.len();
        if k == 0 {
            return Err(NumRs2Error::ValueError(
                "Need at least one forecast to combine".to_string(),
            ));
        }
        if k != past_forecasts.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Number of future forecasts must match number of past forecasts".to_string(),
            ));
        }

        let n = forecasts[0].len();
        for (i, fc) in forecasts.iter().enumerate() {
            if fc.len() != n {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Forecast {} has length {} but expected {}",
                    i,
                    fc.len(),
                    n
                )));
            }
        }

        // Compute MSE for each model over the validation period
        let mut weights = Vec::with_capacity(k);
        let mut weight_sum = 0.0;

        for pf in past_forecasts {
            let mse = ForecastMetrics::mse(past_actuals, pf)?;
            if mse < 1e-15 {
                // Perfect forecast gets all the weight
                let mut combined = Array1::zeros(n);
                let idx = weights.len();
                for j in 0..n {
                    combined[j] = forecasts[idx][j];
                }
                return Ok(combined);
            }
            let inv_mse = 1.0 / mse;
            weights.push(inv_mse);
            weight_sum += inv_mse;
        }

        // Normalize weights
        if weight_sum < 1e-15 {
            return Err(NumRs2Error::ComputationError(
                "All models have infinite MSE".to_string(),
            ));
        }
        for w in &mut weights {
            *w /= weight_sum;
        }

        // Combine
        let mut combined = Array1::zeros(n);
        for (i, fc) in forecasts.iter().enumerate() {
            for j in 0..n {
                combined[j] += weights[i] * fc[j];
            }
        }

        Ok(combined)
    }

    /// Element-wise median combination. Robust to outlier forecasts.
    pub fn median_combination(forecasts: &[&ArrayView1<f64>]) -> Result<Array1<f64>> {
        let k = forecasts.len();
        if k == 0 {
            return Err(NumRs2Error::ValueError(
                "Need at least one forecast to combine".to_string(),
            ));
        }

        let n = forecasts[0].len();
        for (i, fc) in forecasts.iter().enumerate() {
            if fc.len() != n {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Forecast {} has length {} but expected {}",
                    i,
                    fc.len(),
                    n
                )));
            }
        }

        let mut combined = Array1::zeros(n);
        let mut buf = Vec::with_capacity(k);

        for j in 0..n {
            buf.clear();
            for fc in forecasts {
                buf.push(fc[j]);
            }
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            combined[j] = if k.is_multiple_of(2) {
                (buf[k / 2 - 1] + buf[k / 2]) / 2.0
            } else {
                buf[k / 2]
            };
        }

        Ok(combined)
    }
}

/// Methods for constructing prediction intervals around point forecasts.
pub struct PredictionIntervals;

impl PredictionIntervals {
    /// Normal-based prediction intervals: PI = forecast +/- z * sigma * sqrt(h).
    pub fn normal_intervals(
        point_forecasts: &ArrayView1<f64>,
        residuals: &ArrayView1<f64>,
        confidence_level: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(NumRs2Error::ValueError(
                "confidence_level must be in (0, 1)".to_string(),
            ));
        }

        if residuals.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Need residuals to compute prediction intervals".to_string(),
            ));
        }

        let h = point_forecasts.len();

        // Estimate residual standard deviation
        let sigma = {
            let n = residuals.len() as f64;
            let ss: f64 = residuals.iter().map(|&r| r * r).sum();
            (ss / n).sqrt()
        };

        // z-score for the given confidence level
        let alpha = 1.0 - confidence_level;
        let z = quantile_normal(1.0 - alpha / 2.0);

        let mut lower = Array1::zeros(h);
        let mut upper = Array1::zeros(h);

        for i in 0..h {
            let width = z * sigma * ((i + 1) as f64).sqrt();
            lower[i] = point_forecasts[i] - width;
            upper[i] = point_forecasts[i] + width;
        }

        Ok((lower, upper))
    }

    /// Bootstrap prediction intervals via residual resampling.
    pub fn bootstrap_intervals(
        point_forecasts: &ArrayView1<f64>,
        residuals: &ArrayView1<f64>,
        confidence_level: f64,
        n_bootstrap: usize,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(NumRs2Error::ValueError(
                "confidence_level must be in (0, 1)".to_string(),
            ));
        }

        if residuals.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Need residuals to compute bootstrap intervals".to_string(),
            ));
        }

        if n_bootstrap < 10 {
            return Err(NumRs2Error::ValueError(
                "Need at least 10 bootstrap replications".to_string(),
            ));
        }

        let h = point_forecasts.len();
        let n_resid = residuals.len();
        let mut rng = thread_rng();

        // For each horizon step, collect simulated forecast values
        let mut simulations = Array2::<f64>::zeros((n_bootstrap, h));

        for b in 0..n_bootstrap {
            for j in 0..h {
                // Resample a residual uniformly
                let idx = rng.gen_range(0..n_resid);
                simulations[[b, j]] = point_forecasts[j] + residuals[idx];
            }
        }

        // Compute quantiles at each horizon
        let alpha = 1.0 - confidence_level;
        let lower_q = alpha / 2.0;
        let upper_q = 1.0 - alpha / 2.0;

        let mut lower = Array1::zeros(h);
        let mut upper = Array1::zeros(h);

        for j in 0..h {
            let mut col: Vec<f64> = (0..n_bootstrap).map(|b| simulations[[b, j]]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lo_idx = ((n_bootstrap as f64 * lower_q).floor() as usize).min(n_bootstrap - 1);
            let hi_idx = ((n_bootstrap as f64 * upper_q).ceil() as usize).min(n_bootstrap - 1);
            lower[j] = col[lo_idx];
            upper[j] = col[hi_idx];
        }

        Ok((lower, upper))
    }
}

/// Quantile of the standard normal distribution (Abramowitz & Stegun 26.2.23).
fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Coefficients
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let approx = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -approx
    } else {
        approx
    }
}

/// Naive forecasting baselines for benchmarking (Hyndman & Athanasopoulos, Ch. 5).
pub struct NaiveForecaster;

impl NaiveForecaster {
    /// Last observation carried forward (random walk): ŷ_{T+h} = y_T.
    pub fn last_value(data: &ArrayView1<f64>, h: usize) -> Result<ForecastResult> {
        let n = data.len();
        if n == 0 {
            return Err(NumRs2Error::ValueError(
                "Cannot forecast from empty series".to_string(),
            ));
        }

        let last = data[n - 1];
        let forecasts = Array1::from_elem(h, last);

        // Fitted values: each fitted value is the previous observation
        let mut fitted = Array1::zeros(n);
        fitted[0] = data[0]; // No prior observation; use the value itself
        for i in 1..n {
            fitted[i] = data[i - 1];
        }
        let residuals = data - &fitted;

        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }

    /// Seasonal naive: repeat the last seasonal cycle of length `period`.
    pub fn seasonal_naive(
        data: &ArrayView1<f64>,
        period: usize,
        h: usize,
    ) -> Result<ForecastResult> {
        let n = data.len();
        if n < period {
            return Err(NumRs2Error::ValueError(format!(
                "Need at least {} observations (one full season), got {}",
                period, n
            )));
        }
        if period == 0 {
            return Err(NumRs2Error::ValueError(
                "Period must be positive".to_string(),
            ));
        }

        // Forecasts repeat the last complete season
        let mut forecasts = Array1::zeros(h);
        for i in 0..h {
            let lag = period - (i % period);
            let src_idx = n - lag;
            forecasts[i] = data[src_idx];
        }

        // Fitted values: each is the observation one season earlier
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            if i < period {
                fitted[i] = data[i]; // No prior season available
            } else {
                fitted[i] = data[i - period];
            }
        }
        let residuals = data - &fitted;

        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }

    /// Average (mean) forecast: ŷ_{T+h} = mean(y_1..y_T).
    pub fn average(data: &ArrayView1<f64>, h: usize) -> Result<ForecastResult> {
        let n = data.len();
        if n == 0 {
            return Err(NumRs2Error::ValueError(
                "Cannot forecast from empty series".to_string(),
            ));
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let forecasts = Array1::from_elem(h, mean);

        // Fitted: cumulative mean at each point
        let mut fitted = Array1::zeros(n);
        let mut running_sum = 0.0;
        for i in 0..n {
            running_sum += data[i];
            fitted[i] = running_sum / (i + 1) as f64;
        }
        let residuals = data - &fitted;

        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }

    /// Drift method: ŷ_{T+h} = y_T + h * (y_T - y_1) / (T-1).
    pub fn drift(data: &ArrayView1<f64>, h: usize) -> Result<ForecastResult> {
        let n = data.len();
        if n < 2 {
            return Err(NumRs2Error::ValueError(
                "Need at least 2 observations for drift method".to_string(),
            ));
        }

        let drift = (data[n - 1] - data[0]) / (n - 1) as f64;
        let last = data[n - 1];

        let mut forecasts = Array1::zeros(h);
        for i in 0..h {
            forecasts[i] = last + (i + 1) as f64 * drift;
        }

        // Fitted: extrapolation from first observation through each point
        let mut fitted = Array1::zeros(n);
        fitted[0] = data[0];
        for i in 1..n {
            // One-step drift from history up to i-1
            let d = (data[i - 1] - data[0]) / (i.max(1)) as f64;
            fitted[i] = data[i - 1] + d;
        }
        let residuals = data - &fitted;

        Ok(ForecastResult {
            forecasts,
            lower_bounds: None,
            upper_bounds: None,
            fitted,
            residuals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_simple_exponential_smoothing() {
        let data = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0, 12.5, 14.0]);

        let ses = SimpleExponentialSmoothing::new(0.3).expect("SES creation should succeed");
        let result = ses
            .fit_forecast(&data.view(), 3)
            .expect("SES forecast should succeed");

        assert_eq!(result.forecasts.len(), 3);
        assert_eq!(result.fitted.len(), 6);
        assert!(result.forecasts.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_ses_optimize_alpha() {
        let data = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0]);

        let alpha = SimpleExponentialSmoothing::optimize_alpha(&data.view())
            .expect("alpha optimization should succeed");

        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_holt_linear_trend() {
        let data = Array1::from_vec(vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let holt = HoltLinearTrend::new(0.3, 0.1).expect("Holt creation should succeed");
        let result = holt
            .fit_forecast(&data.view(), 3)
            .expect("Holt forecast should succeed");

        assert_eq!(result.forecasts.len(), 3);
        assert!(result.forecasts[2] > result.forecasts[0]);
    }

    #[test]
    fn test_holt_winters_additive() {
        let mut data = Vec::new();
        for i in 0..24 {
            let trend = i as f64 * 0.5;
            let seasonal = (i % 4) as f64 * 2.0;
            data.push(10.0 + trend + seasonal);
        }
        let data = Array1::from_vec(data);

        let hw = HoltWinters::new(0.2, 0.1, 0.3, 4, false).expect("HW creation should succeed");
        let result = hw
            .fit_forecast(&data.view(), 4)
            .expect("HW forecast should succeed");

        assert_eq!(result.forecasts.len(), 4);
        assert!(result.forecasts.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_holt_winters_multiplicative() {
        let mut data = Vec::new();
        for i in 0..20 {
            let base = 10.0 + i as f64 * 0.3;
            let seasonal = 1.0 + ((i % 4) as f64) * 0.1;
            data.push(base * seasonal);
        }
        let data = Array1::from_vec(data);

        let hw = HoltWinters::new(0.2, 0.1, 0.3, 4, true).expect("HW creation should succeed");
        let result = hw
            .fit_forecast(&data.view(), 4)
            .expect("HW forecast should succeed");

        assert_eq!(result.forecasts.len(), 4);
    }

    #[test]
    fn test_mae_metric() {
        let actual = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0]);
        let forecast = Array1::from_vec(vec![10.5, 11.5, 11.5, 12.5]);

        let mae =
            ForecastMetrics::mae(&actual.view(), &forecast.view()).expect("MAE should succeed");

        assert_relative_eq!(mae, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_rmse_metric() {
        let actual = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0]);
        let forecast = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0]);

        let rmse =
            ForecastMetrics::rmse(&actual.view(), &forecast.view()).expect("RMSE should succeed");

        assert!(rmse > 0.0);
        assert!(rmse < 2.0);
    }

    #[test]
    fn test_mape_metric() {
        let actual = Array1::from_vec(vec![100.0, 120.0, 110.0, 130.0]);
        let forecast = Array1::from_vec(vec![105.0, 115.0, 115.0, 125.0]);

        let mape =
            ForecastMetrics::mape(&actual.view(), &forecast.view()).expect("MAPE should succeed");

        assert!(mape >= 0.0);
        assert!(mape < 10.0);
    }

    #[test]
    fn test_mase_metric() {
        let training = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0]);
        let actual = Array1::from_vec(vec![15.0, 16.0]);
        let forecast = Array1::from_vec(vec![14.5, 15.5]);

        let mase = ForecastMetrics::mase(&actual.view(), &forecast.view(), &training.view())
            .expect("MASE should succeed");

        assert!(mase > 0.0);
    }

    #[test]
    fn test_smape_metric() {
        let actual = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let forecast = Array1::from_vec(vec![11.0, 19.0, 31.0]);

        let smape =
            ForecastMetrics::smape(&actual.view(), &forecast.view()).expect("sMAPE should succeed");

        assert!(smape >= 0.0);
        assert!(smape <= 200.0);
    }

    #[test]
    fn test_invalid_alpha() {
        let result = SimpleExponentialSmoothing::new(1.5);
        assert!(result.is_err());

        let result = SimpleExponentialSmoothing::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let actual = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let forecast = Array1::from_vec(vec![1.0, 2.0]);

        let result = ForecastMetrics::mae(&actual.view(), &forecast.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_mse_metric() {
        let actual = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0]);
        let forecast = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0]);

        let mse =
            ForecastMetrics::mse(&actual.view(), &forecast.view()).expect("MSE should succeed");

        // Errors: 0, 1, -1, 0 => squared: 0, 1, 1, 0 => mean = 0.5
        assert_relative_eq!(mse, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_perfect_forecast() {
        let actual = Array1::from_vec(vec![5.0, 10.0, 15.0]);
        let forecast = actual.clone();

        let mse =
            ForecastMetrics::mse(&actual.view(), &forecast.view()).expect("MSE should succeed");

        assert_relative_eq!(mse, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_theils_u_better_than_naive() {
        // Actual: 10, 12, 14 (linear trend)
        // Previous: 8, 10, 12 (lagged actuals)
        // Forecast (good): 10.1, 11.9, 14.1 (close to actual)
        let actual = Array1::from_vec(vec![10.0, 12.0, 14.0]);
        let previous = Array1::from_vec(vec![8.0, 10.0, 12.0]);
        let forecast = Array1::from_vec(vec![10.1, 11.9, 14.1]);

        let u = ForecastMetrics::theils_u(&actual.view(), &forecast.view(), &previous.view())
            .expect("Theil's U should succeed");

        // Good forecast should have U < 1
        assert!(
            u < 1.0,
            "Theil's U = {} should be < 1 for a good forecast",
            u
        );
        assert!(u > 0.0);
    }

    #[test]
    fn test_theils_u_worse_than_naive() {
        // Actual: 10, 12, 14
        // Previous (naive): 9, 11, 13 => naive errors: 1,1,1
        // Bad forecast: 15, 8, 20 => errors: -5, 4, -6
        let actual = Array1::from_vec(vec![10.0, 12.0, 14.0]);
        let previous = Array1::from_vec(vec![9.0, 11.0, 13.0]);
        let forecast = Array1::from_vec(vec![15.0, 8.0, 20.0]);

        let u = ForecastMetrics::theils_u(&actual.view(), &forecast.view(), &previous.view())
            .expect("Theil's U should succeed");

        assert!(
            u > 1.0,
            "Theil's U = {} should be > 1 for a bad forecast",
            u
        );
    }

    #[test]
    fn test_rolling_window_cv() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // Naive last-value forecaster
        let naive_fn = |train: &ArrayView1<f64>, h: usize| -> Result<Array1<f64>> {
            let last = train[train.len() - 1];
            Ok(Array1::from_elem(h, last))
        };

        let errors = TimeSeriesCrossValidation::rolling_window(&data.view(), 5, 2, 1, naive_fn)
            .expect("Rolling window CV should succeed");

        assert!(!errors.is_empty());
        // With window=5, horizon=2, step=1, and n=10:
        // folds: [0..5] test [5..7], [1..6] test [6..8], [2..7] test [7..9], [3..8] test [8..10]
        assert_eq!(errors.len(), 4);
        for e in &errors {
            assert!(*e >= 0.0);
        }
    }

    #[test]
    fn test_expanding_window_cv() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mean_fn = |train: &ArrayView1<f64>, h: usize| -> Result<Array1<f64>> {
            let mean = train.iter().sum::<f64>() / train.len() as f64;
            Ok(Array1::from_elem(h, mean))
        };

        let errors = TimeSeriesCrossValidation::expanding_window(&data.view(), 5, 2, 2, mean_fn)
            .expect("Expanding window CV should succeed");

        assert!(!errors.is_empty());
        // initial_window=5, horizon=2, step=2 => folds at train_end 5, 7 (9+2=11 > 10, so stops)
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_temporal_split() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let (train, test) = TimeSeriesCrossValidation::temporal_split(&data.view(), 0.8)
            .expect("Temporal split should succeed");

        assert_eq!(train.len(), 8);
        assert_eq!(test.len(), 2);
        // Check temporal ordering is preserved
        assert_relative_eq!(train[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(test[0], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_temporal_split_invalid_ratio() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = TimeSeriesCrossValidation::temporal_split(&data.view(), 0.0);
        assert!(result.is_err());

        let result = TimeSeriesCrossValidation::temporal_split(&data.view(), 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_average_combination() {
        let f1 = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let f2 = Array1::from_vec(vec![12.0, 18.0, 32.0]);
        let f3 = Array1::from_vec(vec![11.0, 22.0, 28.0]);

        let combined = ForecastCombiner::simple_average(&[&f1.view(), &f2.view(), &f3.view()])
            .expect("Simple average should succeed");

        assert_eq!(combined.len(), 3);
        assert_relative_eq!(combined[0], 11.0, epsilon = 1e-10);
        assert_relative_eq!(combined[1], 20.0, epsilon = 1e-10);
        assert_relative_eq!(combined[2], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_average_combination() {
        // Model A: better on validation (closer to actuals)
        // Model B: worse on validation
        let val_actual = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let val_a = Array1::from_vec(vec![10.1, 20.1, 30.1]); // very good
        let val_b = Array1::from_vec(vec![12.0, 18.0, 32.0]); // mediocre

        let fc_a = Array1::from_vec(vec![40.0, 50.0]);
        let fc_b = Array1::from_vec(vec![42.0, 48.0]);

        let combined = ForecastCombiner::weighted_average(
            &[&fc_a.view(), &fc_b.view()],
            &val_actual.view(),
            &[&val_a.view(), &val_b.view()],
        )
        .expect("Weighted average should succeed");

        assert_eq!(combined.len(), 2);
        // Model A has much lower MSE, so result should be closer to fc_a
        assert!((combined[0] - 40.0).abs() < (combined[0] - 42.0).abs());
    }

    #[test]
    fn test_median_combination() {
        let f1 = Array1::from_vec(vec![10.0, 100.0]); // outlier at second position
        let f2 = Array1::from_vec(vec![12.0, 20.0]);
        let f3 = Array1::from_vec(vec![11.0, 22.0]);

        let combined = ForecastCombiner::median_combination(&[&f1.view(), &f2.view(), &f3.view()])
            .expect("Median combination should succeed");

        assert_eq!(combined.len(), 2);
        // Median of {10,12,11} = 11, median of {100,20,22} = 22
        assert_relative_eq!(combined[0], 11.0, epsilon = 1e-10);
        assert_relative_eq!(combined[1], 22.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_combination_even_count() {
        let f1 = Array1::from_vec(vec![10.0]);
        let f2 = Array1::from_vec(vec![20.0]);

        let combined = ForecastCombiner::median_combination(&[&f1.view(), &f2.view()])
            .expect("Median combination should succeed");

        // Median of {10, 20} = 15
        assert_relative_eq!(combined[0], 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_prediction_intervals() {
        let forecasts = Array1::from_vec(vec![10.0, 12.0, 14.0]);
        let residuals =
            Array1::from_vec(vec![0.5, -0.3, 0.2, -0.4, 0.6, -0.1, 0.3, -0.5, 0.4, -0.2]);

        let (lower, upper) =
            PredictionIntervals::normal_intervals(&forecasts.view(), &residuals.view(), 0.95)
                .expect("Normal intervals should succeed");

        assert_eq!(lower.len(), 3);
        assert_eq!(upper.len(), 3);

        // Lower must be below forecast, upper above
        for i in 0..3 {
            assert!(lower[i] < forecasts[i]);
            assert!(upper[i] > forecasts[i]);
        }

        // Intervals should widen with horizon
        let width_1 = upper[0] - lower[0];
        let width_3 = upper[2] - lower[2];
        assert!(width_3 > width_1, "Interval width should grow with horizon");
    }

    #[test]
    fn test_bootstrap_prediction_intervals() {
        let forecasts = Array1::from_vec(vec![10.0, 12.0, 14.0, 16.0]);
        let residuals = Array1::from_vec(vec![
            0.5, -0.3, 0.2, -0.4, 0.6, -0.1, 0.3, -0.5, 0.4, -0.2, 0.1, -0.6, 0.7, -0.3, 0.2, -0.1,
            0.5, -0.4, 0.3, -0.2,
        ]);

        let (lower, upper) = PredictionIntervals::bootstrap_intervals(
            &forecasts.view(),
            &residuals.view(),
            0.90,
            500,
        )
        .expect("Bootstrap intervals should succeed");

        assert_eq!(lower.len(), 4);
        assert_eq!(upper.len(), 4);

        for i in 0..4 {
            assert!(
                lower[i] < upper[i],
                "Lower bound must be less than upper bound at horizon {}",
                i + 1
            );
        }
    }

    #[test]
    fn test_naive_last_value() {
        let data = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0]);

        let result =
            NaiveForecaster::last_value(&data.view(), 3).expect("Naive last value should succeed");

        assert_eq!(result.forecasts.len(), 3);
        // All forecasts should be 9.0 (the last observation)
        for &val in result.forecasts.iter() {
            assert_relative_eq!(val, 9.0, epsilon = 1e-10);
        }

        // Fitted: shift-by-one
        assert_relative_eq!(result.fitted[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.fitted[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_seasonal_naive() {
        // Quarterly data: 2 years
        let data = Array1::from_vec(vec![
            10.0, 20.0, 15.0, 25.0, // year 1
            12.0, 22.0, 17.0, 27.0, // year 2
        ]);

        let result = NaiveForecaster::seasonal_naive(&data.view(), 4, 4)
            .expect("Seasonal naive should succeed");

        assert_eq!(result.forecasts.len(), 4);
        // Should repeat last season: 12, 22, 17, 27
        assert_relative_eq!(result.forecasts[0], 12.0, epsilon = 1e-10);
        assert_relative_eq!(result.forecasts[1], 22.0, epsilon = 1e-10);
        assert_relative_eq!(result.forecasts[2], 17.0, epsilon = 1e-10);
        assert_relative_eq!(result.forecasts[3], 27.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_forecast() {
        let data = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);

        let result =
            NaiveForecaster::average(&data.view(), 3).expect("Average forecast should succeed");

        assert_eq!(result.forecasts.len(), 3);
        // Mean = 30
        for &val in result.forecasts.iter() {
            assert_relative_eq!(val, 30.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_drift_method() {
        let data = Array1::from_vec(vec![10.0, 12.0, 14.0, 16.0, 18.0]);

        let result = NaiveForecaster::drift(&data.view(), 3).expect("Drift method should succeed");

        assert_eq!(result.forecasts.len(), 3);
        // drift = (18-10)/(5-1) = 2.0 per step
        // Forecasts: 18+2=20, 18+4=22, 18+6=24
        assert_relative_eq!(result.forecasts[0], 20.0, epsilon = 1e-10);
        assert_relative_eq!(result.forecasts[1], 22.0, epsilon = 1e-10);
        assert_relative_eq!(result.forecasts[2], 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_drift_insufficient_data() {
        let data = Array1::from_vec(vec![5.0]);
        let result = NaiveForecaster::drift(&data.view(), 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_forecast_combiner_empty() {
        let result = ForecastCombiner::simple_average(&[]);
        assert!(result.is_err());

        let result = ForecastCombiner::median_combination(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_prediction_interval_invalid_confidence() {
        let forecasts = Array1::from_vec(vec![1.0, 2.0]);
        let residuals = Array1::from_vec(vec![0.1, -0.1]);

        let result =
            PredictionIntervals::normal_intervals(&forecasts.view(), &residuals.view(), 0.0);
        assert!(result.is_err());

        let result =
            PredictionIntervals::normal_intervals(&forecasts.view(), &residuals.view(), 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rmse_geq_mae() {
        let actual = Array1::from_vec(vec![10.0, 12.0, 8.0, 15.0, 9.0]);
        let forecast = Array1::from_vec(vec![11.0, 11.0, 10.0, 13.0, 10.0]);

        let mae =
            ForecastMetrics::mae(&actual.view(), &forecast.view()).expect("MAE should succeed");
        let rmse =
            ForecastMetrics::rmse(&actual.view(), &forecast.view()).expect("RMSE should succeed");
        let mse =
            ForecastMetrics::mse(&actual.view(), &forecast.view()).expect("MSE should succeed");

        // RMSE >= MAE always holds
        assert!(rmse >= mae, "RMSE={} should be >= MAE={}", rmse, mae);
        // RMSE = sqrt(MSE)
        assert_relative_eq!(rmse, mse.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_naive_last_value_as_theils_u_benchmark() {
        // When forecast equals naive, Theil's U should be 1.0
        let actual = Array1::from_vec(vec![12.0, 14.0, 16.0]);
        let previous = Array1::from_vec(vec![10.0, 12.0, 14.0]);
        // Forecast == naive (previous shifted)
        let forecast = previous.clone();

        let u = ForecastMetrics::theils_u(&actual.view(), &forecast.view(), &previous.view())
            .expect("Theil's U should succeed");

        assert_relative_eq!(u, 1.0, epsilon = 1e-10);
    }
}
