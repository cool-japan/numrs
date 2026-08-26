//! # Time Series Analysis Module
//!
//! Comprehensive econometric and statistical time series analysis for NumRS2.
//!
//! This module provides production-grade implementations of classical and modern
//! time series methods including ARIMA, state space models, volatility modeling,
//! seasonal decomposition, and advanced forecasting techniques.
//!
//! ## Features
//!
//! - **ARIMA/SARIMA**: AutoRegressive Integrated Moving Average models with seasonal components
//! - **State Space Models**: Kalman filtering (linear, extended, unscented), particle filters
//! - **VAR/VECM**: Vector autoregression and error correction models for multivariate series
//! - **GARCH**: Volatility modeling with ARCH/GARCH family models
//! - **Decomposition**: STL, classical, and X-13 seasonal adjustment
//! - **Stationarity**: Unit root tests (ADF, KPSS, PP), cointegration tests (Johansen, Engle-Granger)
//! - **Spectral Analysis**: Periodogram, spectral density, coherence
//! - **Forecasting**: Exponential smoothing, prediction intervals, accuracy metrics
//! - **Change Point Detection**: PELT, binary segmentation, CUSUM, Bayesian methods
//!
//! ## SCIRS2 Policy Compliance
//!
//! This module strictly follows SCIRS2 ecosystem policies:
//! - Uses `scirs2_core::ndarray` for all array operations
//! - Uses `scirs2_linalg` for linear algebra (OxiBLAS backend)
//! - Uses `scirs2_stats` for statistical functions
//! - Uses `scirs2_fft` for spectral analysis
//! - Uses `scirs2_core::random` for random number generation
//! - Zero C/C++ dependencies (Pure Rust)
//!
//! ## Examples
//!
//! ```
//! use numrs2::new_modules::timeseries::*;
//! use numrs2::prelude::*;
//!
//! // ARIMA modeling
//! let data = Array::from_vec(vec![1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0]);
//! // let model = Arima::new(1, 1, 1); // AR(1), diff=1, MA(1)
//! // let fitted = model.fit(&data).expect("ARIMA fit should succeed");
//! // let forecast = fitted.forecast(5).expect("forecast should succeed");
//!
//! // ACF/PACF analysis
//! // let acf = autocorrelation(&data, 10).expect("ACF computation should succeed");
//! // let pacf = partial_autocorrelation(&data, 10).expect("PACF computation should succeed");
//! ```

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// Submodules
pub mod arima;
pub mod changepoint;
pub mod cointegration;
pub mod exponential_smoothing;
pub mod forecasting;
pub mod garch;
pub mod spectral;
pub mod statespace;
pub mod stationarity;
pub mod stl;
pub mod var;

// Re-exports
pub use arima::{Arima, ArimaParams};
pub use changepoint::{
    compute_penalty, compute_segment_statistics, segment_cost, summarize_changepoints,
    BayesianChangePoint, BinarySegmentation, ChangePointResult, ChangePointSummary, ChangeType,
    CostFunction, Cusum, Pelt, PenaltyType, SegmentStats,
};
pub use cointegration::{
    engle_granger_test, estimate_vecm, johansen_test, phillips_ouliaris_test,
    reduced_rank_regression, EngleGrangerResult, JohansenResult, PhillipsOuliarisResult, TrendSpec,
    VecmResult,
};
pub use exponential_smoothing::{
    optimize_parameters, select_best_model, ExponentialSmoothing, ExponentialSmoothingForecast,
    ExponentialSmoothingResult, InformationCriteria, OptimizationConfig, SeasonalComponent,
    TrendComponent,
};
pub use forecasting::{
    ForecastCombiner, ForecastMetrics, ForecastResult, HoltLinearTrend, HoltWinters,
    NaiveForecaster, PredictionIntervals, SimpleExponentialSmoothing, TimeSeriesCrossValidation,
};
pub use garch::{Arch, Egarch, EgarchParams, Garch, GarchParams, GarchResidualAnalysis};
pub use spectral::{
    ar_psd_burg, ar_psd_yule_walker, bartlett_method, coherence, cross_spectral_density,
    detect_peaks, frequency_grid, periodogram_direct, periodogram_windowed, phase_spectrum,
    power_to_db, welch_method, window_function, WelchConfig, WindowType,
};
pub use statespace::{KalmanFilter, KalmanState, ParticleFilter};
pub use stationarity::{
    adf_test_full, detrend_linear, detrend_polynomial, difference, integration_order,
    kpss_test_full, phillips_perron_test, seasonal_difference, AdfTestResult,
    IntegrationOrderResult, KpssTestResult, LagSelection, PhillipsPerronResult, TrendType,
};
pub use stl::{classical_decomposition, StlDecomposition, StlResult};
pub use var::{Var, VarParams, Vecm};

// =============================================================================
// Core Time Series Functions
// =============================================================================

/// Compute the Autocorrelation Function (ACF) for a time series.
///
/// The ACF measures the linear relationship between observations at different
/// time lags. For a time series {X_t}, the ACF at lag k is:
///
/// ρ(k) = Cov(X_t, X_{t-k}) / Var(X_t)
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `nlags` - Number of lags to compute (must be less than data length)
///
/// # Returns
///
/// Array of autocorrelation coefficients from lag 0 to nlags
///
/// # References
///
/// Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
/// Time series analysis: forecasting and control. John Wiley & Sons.
pub fn autocorrelation(data: &ArrayView1<f64>, nlags: usize) -> Result<Array1<f64>> {
    let n = data.len();

    if nlags >= n {
        return Err(NumRs2Error::ValueError(format!(
            "nlags ({}) must be less than data length ({})",
            nlags, n
        )));
    }

    if n == 0 {
        return Err(NumRs2Error::ValueError(
            "Cannot compute ACF for empty series".to_string(),
        ));
    }

    // Compute mean
    let mean = data.iter().sum::<f64>() / n as f64;

    // Compute variance (lag 0)
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-10 {
        return Err(NumRs2Error::ComputationError(
            "Series has zero or near-zero variance".to_string(),
        ));
    }

    // Compute ACF for each lag
    let mut acf = Array1::zeros(nlags + 1);
    acf[0] = 1.0; // ACF at lag 0 is always 1

    for k in 1..=nlags {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += (data[t] - mean) * (data[t + k] - mean);
        }
        acf[k] = sum / (n as f64 * var);
    }

    Ok(acf)
}

/// Compute the Partial Autocorrelation Function (PACF) using the Durbin-Levinson algorithm.
///
/// The PACF measures the correlation between observations at lag k, after removing
/// the linear dependence on observations at intermediate lags 1, 2, ..., k-1.
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `nlags` - Number of lags to compute
///
/// # Returns
///
/// Array of partial autocorrelation coefficients from lag 0 to nlags
///
/// # References
///
/// Durbin, J. (1960). The fitting of time-series models.
/// Revue de l'Institut International de Statistique, 28(3), 233-244.
pub fn partial_autocorrelation(data: &ArrayView1<f64>, nlags: usize) -> Result<Array1<f64>> {
    // First compute ACF
    let acf = autocorrelation(data, nlags)?;
    let n = data.len();

    if nlags >= n {
        return Err(NumRs2Error::ValueError(format!(
            "nlags ({}) must be less than data length ({})",
            nlags, n
        )));
    }

    // Durbin-Levinson recursion
    let mut pacf = Array1::zeros(nlags + 1);
    pacf[0] = 1.0; // PACF at lag 0 is always 1

    if nlags == 0 {
        return Ok(pacf);
    }

    pacf[1] = acf[1];

    let mut phi = Array2::<f64>::zeros((nlags + 1, nlags + 1));
    phi[[1, 1]] = acf[1];

    for k in 2..=nlags {
        // Compute numerator
        let mut numer = acf[k];
        for j in 1..k {
            numer -= phi[[k - 1, j]] * acf[k - j];
        }

        // Compute denominator
        let mut denom = 1.0;
        for j in 1..k {
            denom -= phi[[k - 1, j]] * acf[j];
        }

        if denom.abs() < 1e-10 {
            return Err(NumRs2Error::ComputationError(
                "Numerical instability in PACF computation".to_string(),
            ));
        }

        pacf[k] = numer / denom;
        phi[[k, k]] = pacf[k];

        // Update phi matrix
        for j in 1..k {
            phi[[k, j]] = phi[[k - 1, j]] - pacf[k] * phi[[k - 1, k - j]];
        }
    }

    Ok(pacf)
}

/// Compute cross-correlation function between two time series.
///
/// The CCF measures the linear relationship between one series and lagged
/// values of another series.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `nlags` - Number of lags to compute
///
/// # Returns
///
/// Array of cross-correlation coefficients from -nlags to +nlags
pub fn cross_correlation(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    nlags: usize,
) -> Result<Array1<f64>> {
    let nx = x.len();
    let ny = y.len();

    if nx != ny {
        return Err(NumRs2Error::ValueError(format!(
            "Series must have same length: x={}, y={}",
            nx, ny
        )));
    }

    let n = nx;

    if nlags >= n {
        return Err(NumRs2Error::ValueError(format!(
            "nlags ({}) must be less than series length ({})",
            nlags, n
        )));
    }

    // Compute means
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    // Compute standard deviations
    let std_x = (x.iter().map(|&val| (val - mean_x).powi(2)).sum::<f64>() / n as f64).sqrt();

    let std_y = (y.iter().map(|&val| (val - mean_y).powi(2)).sum::<f64>() / n as f64).sqrt();

    if std_x < 1e-10 || std_y < 1e-10 {
        return Err(NumRs2Error::ComputationError(
            "One or both series have zero or near-zero variance".to_string(),
        ));
    }

    // Compute CCF for lags from -nlags to +nlags
    let mut ccf = Array1::zeros(2 * nlags + 1);

    for lag in 0..=nlags {
        // Positive lag: y leads x
        let mut sum_pos = 0.0;
        for t in 0..(n - lag) {
            sum_pos += (x[t] - mean_x) * (y[t + lag] - mean_y);
        }
        ccf[nlags + lag] = sum_pos / ((n - lag) as f64 * std_x * std_y);

        // Negative lag: x leads y (symmetric except at lag 0)
        if lag > 0 {
            let mut sum_neg = 0.0;
            for t in 0..(n - lag) {
                sum_neg += (x[t + lag] - mean_x) * (y[t] - mean_y);
            }
            ccf[nlags - lag] = sum_neg / ((n - lag) as f64 * std_x * std_y);
        }
    }

    Ok(ccf)
}

/// Ljung-Box test for autocorrelation in residuals.
///
/// Tests the null hypothesis that the autocorrelations are all zero
/// up to lag h. The test statistic is:
///
/// Q(h) = n(n+2) Σ_{k=1}^h ρ_k^2 / (n-k)
///
/// which follows a chi-squared distribution with h degrees of freedom.
///
/// # Arguments
///
/// * `residuals` - Residual series to test
/// * `lags` - Number of lags to include in test
/// * `df_adjust` - Degrees of freedom adjustment for fitted parameters
///
/// # Returns
///
/// Tuple of (test statistic, p-value)
///
/// # References
///
/// Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in time series models.
/// Biometrika, 65(2), 297-303.
pub fn ljung_box_test(
    residuals: &ArrayView1<f64>,
    lags: usize,
    df_adjust: usize,
) -> Result<(f64, f64)> {
    let n = residuals.len();

    if lags >= n {
        return Err(NumRs2Error::ValueError(format!(
            "lags ({}) must be less than series length ({})",
            lags, n
        )));
    }

    // Compute ACF of residuals
    let acf = autocorrelation(residuals, lags)?;

    // Compute Ljung-Box statistic
    let mut q_stat = 0.0;
    for k in 1..=lags {
        q_stat += acf[k].powi(2) / (n - k) as f64;
    }
    q_stat *= (n * (n + 2)) as f64;

    // Compute p-value from chi-squared distribution
    let df = (lags - df_adjust) as f64;
    if df <= 0.0 {
        return Err(NumRs2Error::ValueError(
            "Degrees of freedom must be positive".to_string(),
        ));
    }

    // Use incomplete gamma function for chi-squared CDF
    // P(χ² > q) = 1 - P(χ² ≤ q) = 1 - γ(df/2, q/2) / Γ(df/2)
    let p_value = 1.0 - chi_squared_cdf(q_stat, df);

    Ok((q_stat, p_value))
}

/// Chi-squared cumulative distribution function.
///
/// Approximation using the incomplete gamma function relationship:
/// P(χ²_k ≤ x) = γ(k/2, x/2) / Γ(k/2)
pub(super) fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Use gamma function approximation
    // For now, use a simple normal approximation for large df
    if df > 30.0 {
        // Wilson-Hilferty transformation
        let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
        return normal_cdf(z);
    }

    // For small df, use series expansion of incomplete gamma
    incomplete_gamma_regularized(df / 2.0, x / 2.0)
}

/// Standard normal CDF using error function.
pub(super) fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun).
pub(super) fn erf(x: f64) -> f64 {
    // Constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Regularized incomplete gamma function P(a,x) = γ(a,x)/Γ(a).
///
/// Uses series expansion for small x and continued fraction for large x.
pub(super) fn incomplete_gamma_regularized(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }

    if x == 0.0 {
        return 0.0;
    }

    // Use series expansion if x < a + 1
    if x < a + 1.0 {
        incomplete_gamma_series(a, x)
    } else {
        1.0 - incomplete_gamma_continued_fraction(a, x)
    }
}

/// Series expansion for incomplete gamma function.
pub(super) fn incomplete_gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let epsilon = 1e-10;

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;

        if term.abs() < epsilon * sum.abs() {
            break;
        }
    }

    sum * (-x).exp() * x.powf(a) / gamma(a)
}

/// Continued fraction for incomplete gamma function.
pub(super) fn incomplete_gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let epsilon = 1e-10;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..max_iter {
        let an = -i as f64 * (i as f64 - a);
        b += 2.0;
        d = an * d + b;

        if d.abs() < 1e-30 {
            d = 1e-30;
        }

        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }

        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < epsilon {
            break;
        }
    }

    h * (-x).exp() * x.powf(a) / gamma(a)
}

/// Gamma function using Lanczos approximation.
pub(super) fn gamma(z: f64) -> f64 {
    // Lanczos coefficients for g=7
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * gamma(1.0 - z))
    } else {
        let z = z - 1.0;
        let mut x = COEF[0];

        for i in 1..9 {
            x += COEF[i] / (z + i as f64);
        }

        let t = z + G + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

/// Durbin-Watson test statistic for serial correlation in residuals.
///
/// Tests for first-order autocorrelation. The test statistic is:
///
/// DW = Σ_{t=2}^n (e_t - e_{t-1})^2 / Σ_{t=1}^n e_t^2
///
/// Values near 2 indicate no autocorrelation, values < 2 indicate positive
/// autocorrelation, and values > 2 indicate negative autocorrelation.
///
/// # Arguments
///
/// * `residuals` - Residual series to test
///
/// # Returns
///
/// Durbin-Watson test statistic
///
/// # References
///
/// Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in
/// least squares regression. Biometrika, 37(3/4), 409-428.
pub fn durbin_watson_test(residuals: &ArrayView1<f64>) -> Result<f64> {
    let n = residuals.len();

    if n < 2 {
        return Err(NumRs2Error::ValueError(
            "Need at least 2 observations for Durbin-Watson test".to_string(),
        ));
    }

    let mut numer = 0.0;
    let mut denom = 0.0;

    for t in 1..n {
        numer += (residuals[t] - residuals[t - 1]).powi(2);
    }

    for t in 0..n {
        denom += residuals[t].powi(2);
    }

    if denom < 1e-10 {
        return Err(NumRs2Error::ComputationError(
            "Residuals have zero or near-zero variance".to_string(),
        ));
    }

    Ok(numer / denom)
}

// =============================================================================
// Unit Root Tests (Stationarity)
// =============================================================================

/// Augmented Dickey-Fuller (ADF) test for unit root.
///
/// Tests the null hypothesis that a time series has a unit root (non-stationary).
/// A significant result (low p-value) indicates the series is stationary.
///
/// Test equation: Δy_t = α + βt + γy_{t-1} + Σδᵢ Δy_{t-i} + ε_t
///
/// # Arguments
///
/// * `data` - Time series to test
/// * `lags` - Number of lagged differences to include
/// * `trend` - Include trend component ("c" for constant, "ct" for constant+trend, "nc" for none)
///
/// # Returns
///
/// Tuple of (ADF statistic, p-value)
///
/// # References
///
/// Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the estimators for autoregressive
/// time series with a unit root." *Journal of the American Statistical Association*, 74(366a), 427-431.
pub fn adf_test(data: &ArrayView1<f64>, lags: usize, trend: &str) -> Result<(f64, f64)> {
    let n = data.len();

    if n < lags + 3 {
        return Err(NumRs2Error::ValueError(
            "Insufficient observations for ADF test".to_string(),
        ));
    }

    // First difference
    let mut diff = Array1::zeros(n - 1);
    for i in 0..(n - 1) {
        diff[i] = data[i + 1] - data[i];
    }

    // Build regression: Δy_t on y_{t-1} and lagged differences
    let n_obs = n - lags - 1;
    let n_vars = match trend {
        "nc" => 1 + lags,
        "c" => 2 + lags,
        "ct" => 3 + lags,
        _ => {
            return Err(NumRs2Error::ValueError(
                "trend must be 'nc', 'c', or 'ct'".to_string(),
            ))
        }
    };

    let mut x = Array2::zeros((n_obs, n_vars));
    let mut y = Array1::zeros(n_obs);

    for i in 0..n_obs {
        let t = i + lags;
        y[i] = diff[t];

        let mut col = 0;

        // Constant
        if trend == "c" || trend == "ct" {
            x[[i, col]] = 1.0;
            col += 1;
        }

        // Time trend
        if trend == "ct" {
            x[[i, col]] = (t + 1) as f64;
            col += 1;
        }

        // Lagged level
        x[[i, col]] = data[t];
        col += 1;

        // Lagged differences
        for lag in 1..=lags {
            if t >= lag {
                x[[i, col]] = diff[t - lag];
                col += 1;
            }
        }
    }

    // OLS estimation
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y);

    let beta = scirs2_linalg::solve(&xtx.view(), &xty.view(), None)
        .map_err(|_| NumRs2Error::ComputationError("Singular matrix".to_string()))?;

    // Extract coefficient on lagged level
    let gamma_idx = if trend == "nc" {
        0
    } else if trend == "c" {
        1
    } else {
        2
    };
    let gamma = beta[gamma_idx];

    // Compute residuals and standard error
    let y_pred = x.dot(&beta);
    let residuals = &y - &y_pred;
    let rss = residuals.iter().map(|&r| r * r).sum::<f64>();
    let se_sq = rss / (n_obs - n_vars) as f64;

    // Standard error of gamma
    let xtx_inv_diag =
        scirs2_linalg::solve_multiple(&xtx.view(), &Array2::eye(n_vars).view(), None)
            .map_err(|_| NumRs2Error::ComputationError("Singular matrix".to_string()))?;
    let se_gamma = (se_sq * xtx_inv_diag[[gamma_idx, gamma_idx]]).sqrt();

    // ADF test statistic
    let adf_stat = gamma / se_gamma;

    // Approximate p-value using MacKinnon (1994) approximations
    let p_value = adf_p_value(adf_stat, n, trend);

    Ok((adf_stat, p_value))
}

/// Approximate p-value for ADF test using MacKinnon (1994) critical values.
fn adf_p_value(stat: f64, _n: usize, trend: &str) -> f64 {
    // Critical values at different significance levels (approximate)
    let critical_values = match trend {
        "nc" => vec![-2.58, -1.95, -1.62], // 1%, 5%, 10%
        "c" => vec![-3.43, -2.86, -2.57],
        "ct" => vec![-3.96, -3.41, -3.12],
        _ => vec![-2.86, -2.57, -2.28],
    };

    // Simple interpolation for p-value
    if stat < critical_values[0] {
        0.01
    } else if stat < critical_values[1] {
        0.05
    } else if stat < critical_values[2] {
        0.10
    } else {
        0.50
    }
}

/// KPSS test for stationarity.
///
/// Tests the null hypothesis that a time series is stationary around a deterministic trend.
/// A significant result (low p-value) indicates non-stationarity.
///
/// This is the reverse of the ADF test: KPSS assumes stationarity under H₀.
///
/// # Arguments
///
/// * `data` - Time series to test
/// * `lags` - Number of lags for Newey-West variance estimator
/// * `trend` - "c" for level stationarity, "ct" for trend stationarity
///
/// # Returns
///
/// Tuple of (KPSS statistic, p-value)
///
/// # References
///
/// Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
/// "Testing the null hypothesis of stationarity against the alternative of a unit root."
/// *Journal of Econometrics*, 54(1-3), 159-178.
pub fn kpss_test(data: &ArrayView1<f64>, lags: usize, trend: &str) -> Result<(f64, f64)> {
    let n = data.len();

    if n < 10 {
        return Err(NumRs2Error::ValueError(
            "Need at least 10 observations for KPSS test".to_string(),
        ));
    }

    // Detrend the series
    let residuals = if trend == "c" {
        // Remove mean
        let mean = data.iter().sum::<f64>() / n as f64;
        let mut res = Array1::zeros(n);
        for i in 0..n {
            res[i] = data[i] - mean;
        }
        res
    } else if trend == "ct" {
        // Remove linear trend using OLS
        let mut x = Array2::zeros((n, 2));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = i as f64;
            y[i] = data[i];
        }

        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let beta = scirs2_linalg::solve(&xtx.view(), &xty.view(), None)
            .map_err(|_| NumRs2Error::ComputationError("Singular matrix".to_string()))?;

        let fitted = x.dot(&beta);
        &y - &fitted
    } else {
        return Err(NumRs2Error::ValueError(
            "trend must be 'c' or 'ct'".to_string(),
        ));
    };

    // Compute partial sums
    let mut partial_sum = Array1::zeros(n);
    partial_sum[0] = residuals[0];
    for i in 1..n {
        partial_sum[i] = partial_sum[i - 1] + residuals[i];
    }

    // Compute KPSS numerator
    let numerator = partial_sum.iter().map(|&s| s * s).sum::<f64>() / (n * n) as f64;

    // Compute long-run variance using Newey-West estimator
    let s0 = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;
    let mut s_sum = s0;

    for lag in 1..=lags.min(n - 1) {
        let weight = 1.0 - lag as f64 / (lags + 1) as f64;
        let mut gamma = 0.0;
        for i in lag..n {
            gamma += residuals[i] * residuals[i - lag];
        }
        gamma /= n as f64;
        s_sum += 2.0 * weight * gamma;
    }

    if s_sum <= 0.0 {
        return Err(NumRs2Error::ComputationError(
            "Long-run variance is non-positive".to_string(),
        ));
    }

    // KPSS statistic
    let kpss_stat = numerator / s_sum;

    // Approximate p-value using critical values
    let p_value = kpss_p_value(kpss_stat, trend);

    Ok((kpss_stat, p_value))
}

/// Approximate p-value for KPSS test.
fn kpss_p_value(stat: f64, trend: &str) -> f64 {
    let critical_values = match trend {
        "c" => vec![0.739, 0.463, 0.347], // 1%, 5%, 10%
        "ct" => vec![0.216, 0.146, 0.119],
        _ => vec![0.463, 0.347, 0.280],
    };

    if stat > critical_values[0] {
        0.01
    } else if stat > critical_values[1] {
        0.05
    } else if stat > critical_values[2] {
        0.10
    } else {
        0.50
    }
}

// =============================================================================
// Spectral Analysis
// =============================================================================

/// Compute periodogram for spectral analysis.
///
/// The periodogram estimates the spectral density using the squared magnitude
/// of the Fourier transform:
///
/// I(ω) = (1/n) |Σ X_t e^{-iωt}|²
///
/// # Arguments
///
/// * `data` - Time series data
///
/// # Returns
///
/// Tuple of (frequencies, power spectral density)
///
/// # References
///
/// Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.
pub fn periodogram(data: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();

    if n < 2 {
        return Err(NumRs2Error::ValueError(
            "Need at least 2 observations for periodogram".to_string(),
        ));
    }

    // Remove mean
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Use scirs2_fft for FFT computation
    let fft_result = crate::fft::rfft(&centered, None)
        .map_err(|_| NumRs2Error::ComputationError("FFT computation failed".to_string()))?;

    // Compute periodogram: I(ω) = (1/n) |FFT|²
    let n_freqs = n / 2 + 1;
    let mut frequencies = Array1::zeros(n_freqs);
    let mut psd = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        frequencies[i] = i as f64 / n as f64;
        let re = fft_result[i].re;
        let im = fft_result[i].im;
        psd[i] = (re * re + im * im) / n as f64;
    }

    Ok((frequencies, psd))
}

/// Compute Welch's method for spectral density estimation.
///
/// Improves periodogram by averaging over overlapping segments.
///
/// # Arguments
///
/// * `data` - Time series data
/// * `segment_length` - Length of each segment
/// * `overlap` - Number of overlapping points between segments
///
/// # Returns
///
/// Tuple of (frequencies, power spectral density)
pub fn welch_psd(
    data: &ArrayView1<f64>,
    segment_length: usize,
    overlap: usize,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();

    if segment_length > n {
        return Err(NumRs2Error::ValueError(
            "Segment length exceeds data length".to_string(),
        ));
    }

    if overlap >= segment_length {
        return Err(NumRs2Error::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    let step = segment_length - overlap;
    let n_segments = ((n - segment_length) as f64 / step as f64).floor() as usize + 1;

    if n_segments == 0 {
        return Err(NumRs2Error::ValueError(
            "No segments available with given parameters".to_string(),
        ));
    }

    // Initialize accumulators
    let n_freqs = segment_length / 2 + 1;
    let mut avg_psd = Array1::zeros(n_freqs);
    let mut frequencies = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        frequencies[i] = i as f64 / segment_length as f64;
    }

    // Process each segment
    for seg in 0..n_segments {
        let start = seg * step;
        let end = (start + segment_length).min(n);

        if end - start < segment_length {
            break;
        }

        let segment = data.slice(scirs2_core::ndarray::s![start..end]);
        let (_freqs, psd) = periodogram(&segment)?;

        for i in 0..n_freqs.min(psd.len()) {
            avg_psd[i] += psd[i];
        }
    }

    // Average
    avg_psd /= n_segments as f64;

    Ok((frequencies, avg_psd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_acf_constant_series() {
        let data = Array1::from_vec(vec![1.0; 10]);
        let result = autocorrelation(&data.view(), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_acf_white_noise() {
        // White noise should have ACF near 0 for all lags > 0
        // Use actual random-like values instead of alternating pattern
        use scirs2_core::random::thread_rng;

        let mut rng = thread_rng();
        let data: Vec<f64> = (0..50).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let data = Array1::from_vec(data);

        let acf = autocorrelation(&data.view(), 5).expect("ACF should succeed");

        assert_relative_eq!(acf[0], 1.0, epsilon = 1e-10);
        // For white noise, ACF at lag > 0 should be small (relaxed tolerance for small samples)
        for i in 1..=5 {
            assert!(
                acf[i].abs() < 0.6,
                "ACF[{}] = {} should be small",
                i,
                acf[i]
            );
        }
    }

    #[test]
    fn test_acf_ar1_process() {
        // AR(1) process: X_t = 0.8 * X_{t-1} + ε_t
        // Theoretical ACF: ρ(k) = 0.8^k
        let data = Array1::from_vec(vec![
            0.0, 0.8, 1.28, 1.424, 1.539, 1.631, 1.705, 1.764, 1.811, 1.849,
        ]);
        let acf = autocorrelation(&data.view(), 3).expect("ACF should succeed");

        assert_relative_eq!(acf[0], 1.0, epsilon = 1e-10);
        // ACF should decay
        assert!(acf[1] > acf[2]);
        assert!(acf[2] > acf[3]);
    }

    #[test]
    fn test_pacf_computation() {
        let data = Array1::from_vec(vec![1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2]);
        let pacf = partial_autocorrelation(&data.view(), 5).expect("PACF should succeed");

        assert_relative_eq!(pacf[0], 1.0, epsilon = 1e-10);
        assert!(pacf.len() == 6);
    }

    #[test]
    fn test_ccf_identical_series() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let ccf = cross_correlation(&data.view(), &data.view(), 2).expect("CCF should succeed");

        // At lag 0, CCF of series with itself should be 1
        assert_relative_eq!(ccf[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ljung_box_white_noise() {
        // White noise should not reject null hypothesis
        let residuals =
            Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, -0.2]);
        let (stat, p_value) =
            ljung_box_test(&residuals.view(), 3, 0).expect("Ljung-Box test should succeed");

        assert!(stat >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_durbin_watson() {
        let residuals = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4]);
        let dw = durbin_watson_test(&residuals.view()).expect("DW test should succeed");

        // DW should be between 0 and 4
        assert!((0.0..=4.0).contains(&dw));
    }

    #[test]
    fn test_acf_nlags_validation() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = autocorrelation(&data.view(), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_gamma_function() {
        // Test known values
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(3.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(4.0), 6.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(5.0), 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adf_test_stationary() {
        // Stationary series (white noise)
        let data = Array1::from_vec(vec![
            0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, -0.2,
            0.1, -0.4, 0.3, -0.1,
        ]);

        let result = adf_test(&data.view(), 2, "c");
        assert!(result.is_ok());

        if let Ok((stat, p_value)) = result {
            assert!(stat.is_finite());
            assert!((0.0..=1.0).contains(&p_value));
        }
    }

    #[test]
    fn test_kpss_test() {
        // Test with stationary series
        let data = Array1::from_vec(vec![
            1.0, 1.1, 0.9, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.2, 1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1,
            0.9, 1.0, 1.1,
        ]);

        let result = kpss_test(&data.view(), 3, "c");
        assert!(result.is_ok());

        if let Ok((stat, p_value)) = result {
            assert!(stat >= 0.0);
            assert!((0.0..=1.0).contains(&p_value));
        }
    }

    #[test]
    fn test_periodogram() {
        // Create sinusoidal signal
        let mut data = Vec::new();
        for i in 0..100 {
            data.push((2.0 * std::f64::consts::PI * 0.1 * i as f64).sin());
        }
        let data = Array1::from_vec(data);

        let result = periodogram(&data.view());
        assert!(result.is_ok());

        if let Ok((freqs, psd)) = result {
            assert_eq!(freqs.len(), 51); // n/2 + 1
            assert_eq!(psd.len(), 51);
            assert!(psd.iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    fn test_welch_psd() {
        let mut data = Vec::new();
        for i in 0..200 {
            data.push((2.0 * std::f64::consts::PI * 0.05 * i as f64).sin());
        }
        let data = Array1::from_vec(data);

        let result = welch_psd(&data.view(), 64, 32);
        assert!(result.is_ok());

        if let Ok((freqs, psd)) = result {
            assert!(!freqs.is_empty());
            assert!(!psd.is_empty());
            assert!(psd.iter().all(|&x| x >= 0.0));
        }
    }
}
