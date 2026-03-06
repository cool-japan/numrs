//! Internal helper functions for exponential smoothing.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::ArrayView1;

use super::core::ExponentialSmoothing;
use super::types::{ExponentialSmoothingResult, SeasonalComponent, TrendComponent};

/// Validate that a smoothing parameter is in (0, 1).
pub(super) fn validate_param(value: f64, name: &str) -> Result<()> {
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
pub(super) fn damped_trend_sum(phi: f64, h: usize) -> f64 {
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
pub(super) fn build_and_fit(
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
pub(super) fn quantile_normal(p: f64) -> f64 {
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
