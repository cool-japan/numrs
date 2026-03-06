//! Data structures for exponential smoothing models.

use scirs2_core::ndarray::Array1;

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
