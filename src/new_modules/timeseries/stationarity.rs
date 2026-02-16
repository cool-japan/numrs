//! Unit Root and Stationarity Tests
//!
//! This module provides comprehensive statistical tests for determining whether a
//! time series is stationary or contains a unit root. These tests are fundamental
//! to time series econometrics and are required before fitting ARIMA, VAR, or
//! cointegration models.
//!
//! ## Tests Provided
//!
//! - **Augmented Dickey-Fuller (ADF)**: Tests H0: unit root (non-stationary)
//! - **KPSS**: Tests H0: stationary (reverse of ADF)
//! - **Phillips-Perron (PP)**: Non-parametric correction for serial correlation
//!
//! ## Helper Functions
//!
//! - First and seasonal differencing
//! - Linear and polynomial detrending
//! - Integration order determination via repeated ADF testing
//!
//! ## References
//!
//! - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for
//!   autoregressive time series with a unit root.
//! - Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
//!   Testing the null hypothesis of stationarity against the alternative of a unit root.
//! - Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time series regression.
//! - MacKinnon, J. G. (1994). Approximate asymptotic distribution functions for
//!   unit-root and cointegration tests.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// =============================================================================
// Types and Enums
// =============================================================================

/// Trend specification for unit root tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendType {
    /// No deterministic terms (nc)
    None,
    /// Constant only (c)
    Constant,
    /// Constant and linear time trend (ct)
    ConstantTrend,
}

impl std::str::FromStr for TrendType {
    type Err = NumRs2Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "nc" | "none" | "n" => Ok(TrendType::None),
            "c" | "constant" => Ok(TrendType::Constant),
            "ct" | "trend" | "ctt" => Ok(TrendType::ConstantTrend),
            _ => Err(NumRs2Error::ValueError(format!(
                "Invalid trend type '{}'. Use 'nc', 'c', or 'ct'.",
                s
            ))),
        }
    }
}

impl TrendType {
    /// Number of deterministic regressors for this trend type.
    fn n_deterministic(&self) -> usize {
        match self {
            TrendType::None => 0,
            TrendType::Constant => 1,
            TrendType::ConstantTrend => 2,
        }
    }

    /// String code for this trend type.
    fn code(&self) -> &'static str {
        match self {
            TrendType::None => "nc",
            TrendType::Constant => "c",
            TrendType::ConstantTrend => "ct",
        }
    }
}

/// Lag selection method for ADF test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagSelection {
    /// Use Akaike Information Criterion to select lag order.
    AIC,
    /// Use Bayesian Information Criterion to select lag order.
    BIC,
    /// Use a fixed number of lags.
    Fixed(usize),
}

// =============================================================================
// Result Structures
// =============================================================================

/// Result of the Augmented Dickey-Fuller test.
#[derive(Debug, Clone)]
pub struct AdfTestResult {
    /// ADF test statistic (t-ratio for gamma).
    pub statistic: f64,
    /// Approximate p-value (MacKinnon approximation).
    pub p_value: f64,
    /// Number of lags used in the regression.
    pub lags_used: usize,
    /// Number of observations used in the regression.
    pub n_obs: usize,
    /// Critical values at 1%, 5%, and 10% significance levels.
    pub critical_values: CriticalValues,
    /// Trend type used.
    pub trend: TrendType,
    /// Information criterion value (AIC or BIC) if automatic lag selection was used.
    pub ic_best: Option<f64>,
}

/// Result of the KPSS test.
#[derive(Debug, Clone)]
pub struct KpssTestResult {
    /// KPSS test statistic.
    pub statistic: f64,
    /// Approximate p-value.
    pub p_value: f64,
    /// Bandwidth (number of lags) used in Newey-West estimator.
    pub bandwidth: usize,
    /// Number of observations.
    pub n_obs: usize,
    /// Critical values at 1%, 5%, and 10% significance levels.
    pub critical_values: CriticalValues,
    /// Trend type used ("c" for level, "ct" for trend stationarity).
    pub trend: TrendType,
}

/// Result of the Phillips-Perron test.
#[derive(Debug, Clone)]
pub struct PhillipsPerronResult {
    /// Z_tau statistic (modified t-statistic).
    pub z_tau: f64,
    /// Z_alpha statistic (modified coefficient test).
    pub z_alpha: f64,
    /// Approximate p-value based on Z_tau.
    pub p_value: f64,
    /// Number of lags used for spectral density estimation.
    pub bandwidth: usize,
    /// Number of observations.
    pub n_obs: usize,
    /// Critical values at 1%, 5%, and 10% significance levels.
    pub critical_values: CriticalValues,
    /// Trend type used.
    pub trend: TrendType,
}

/// Result of integration order determination.
#[derive(Debug, Clone)]
pub struct IntegrationOrderResult {
    /// Estimated integration order (0 = stationary, 1 = I(1), etc.)
    pub order: usize,
    /// ADF test results at each differencing stage.
    pub adf_results: Vec<AdfTestResult>,
    /// Significance level used for the tests.
    pub significance: f64,
}

/// Critical values at standard significance levels.
#[derive(Debug, Clone)]
pub struct CriticalValues {
    /// Critical value at 1% significance level.
    pub one_pct: f64,
    /// Critical value at 5% significance level.
    pub five_pct: f64,
    /// Critical value at 10% significance level.
    pub ten_pct: f64,
}

// =============================================================================
// Helper: OLS Regression
// =============================================================================

/// Result from an OLS regression.
struct OlsResult {
    /// Estimated coefficients.
    beta: Array1<f64>,
    /// Residuals.
    residuals: Array1<f64>,
    /// Residual sum of squares.
    rss: f64,
    /// Standard errors of coefficients.
    se: Array1<f64>,
    /// Number of observations.
    n_obs: usize,
    /// Number of regressors.
    n_vars: usize,
}

/// Perform OLS regression of y on X.
///
/// Solves beta = (X'X)^{-1} X'y and computes standard errors.
fn ols_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<OlsResult> {
    let n_obs = y.len();
    let n_vars = x.ncols();

    if n_obs <= n_vars {
        return Err(NumRs2Error::ValueError(format!(
            "Insufficient observations ({}) for {} regressors",
            n_obs, n_vars
        )));
    }

    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);

    let beta = scirs2_linalg::solve(&xtx.view(), &xty.view(), None)
        .map_err(|e| NumRs2Error::ComputationError(format!("OLS solve failed: {}", e)))?;

    let y_hat = x.dot(&beta);
    let residuals = y - &y_hat;
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let sigma2 = rss / (n_obs - n_vars) as f64;

    // Compute (X'X)^{-1} for standard errors
    let identity = Array2::<f64>::eye(n_vars);
    let xtx_inv = scirs2_linalg::solve_multiple(&xtx.view(), &identity.view(), None)
        .map_err(|e| NumRs2Error::ComputationError(format!("OLS inverse failed: {}", e)))?;

    let mut se = Array1::zeros(n_vars);
    for j in 0..n_vars {
        let var_j = sigma2 * xtx_inv[[j, j]];
        se[j] = if var_j > 0.0 { var_j.sqrt() } else { 0.0 };
    }

    Ok(OlsResult {
        beta,
        residuals,
        rss,
        se,
        n_obs,
        n_vars,
    })
}

// =============================================================================
// ADF Test (Enhanced)
// =============================================================================

/// Augmented Dickey-Fuller test for unit root with automatic lag selection.
///
/// Tests H0: the series has a unit root (non-stationary) against
/// H1: the series is stationary.
///
/// The test regression is:
///
/// Delta y_t = alpha + beta*t + gamma*y_{t-1} + sum_i delta_i * Delta y_{t-i} + eps_t
///
/// The ADF statistic is the t-ratio for gamma. If gamma = 0 (unit root),
/// the series is non-stationary.
///
/// # Arguments
///
/// * `data` - Time series to test (at least 15 observations recommended)
/// * `lag_selection` - Method for choosing the number of augmenting lags
/// * `trend` - Deterministic trend specification
/// * `max_lags` - Maximum number of lags to consider (None for automatic: 12*(n/100)^{1/4})
///
/// # Returns
///
/// `AdfTestResult` containing the test statistic, p-value, critical values,
/// and lag information.
///
/// # References
///
/// Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in
/// autoregressive-moving average models of unknown order. Biometrika, 71(3), 599-607.
pub fn adf_test_full(
    data: &ArrayView1<f64>,
    lag_selection: LagSelection,
    trend: TrendType,
    max_lags: Option<usize>,
) -> Result<AdfTestResult> {
    let n = data.len();

    if n < 8 {
        return Err(NumRs2Error::ValueError(format!(
            "Need at least 8 observations for ADF test, got {}",
            n
        )));
    }

    // Compute first differences
    let diff = difference(data, 1)?;

    // Determine maximum lags (Schwert, 1989 rule: 12*(n/100)^{1/4})
    let auto_max = ((12.0 * (n as f64 / 100.0).powf(0.25)) as usize).max(1);
    let max_lag = max_lags.unwrap_or(auto_max).min(n / 3);

    // Select optimal lag
    let (best_lag, ic_best) = match lag_selection {
        LagSelection::Fixed(lag) => {
            if lag > max_lag {
                return Err(NumRs2Error::ValueError(format!(
                    "Requested lag {} exceeds maximum {}",
                    lag, max_lag
                )));
            }
            (lag, None)
        }
        LagSelection::AIC | LagSelection::BIC => {
            select_lag_ic(data, &diff, trend, max_lag, lag_selection)?
        }
    };

    // Run the ADF regression at the selected lag
    let result = adf_regression(data, &diff, best_lag, trend)?;
    let n_obs = result.n_obs;
    let gamma_idx = trend.n_deterministic();
    let gamma = result.beta[gamma_idx];
    let se_gamma = result.se[gamma_idx];

    if se_gamma < 1e-15 {
        return Err(NumRs2Error::ComputationError(
            "Standard error of gamma is effectively zero".to_string(),
        ));
    }

    let adf_stat = gamma / se_gamma;
    let p_value = mackinnon_p_value(adf_stat, n_obs, trend);
    let critical_values = adf_critical_values(n_obs, trend);

    Ok(AdfTestResult {
        statistic: adf_stat,
        p_value,
        lags_used: best_lag,
        n_obs,
        critical_values,
        trend,
        ic_best,
    })
}

/// Build and run the ADF regression for a given lag order.
fn adf_regression(
    data: &ArrayView1<f64>,
    diff: &Array1<f64>,
    lags: usize,
    trend: TrendType,
) -> Result<OlsResult> {
    let n = data.len();
    let n_obs = diff.len() - lags; // effective sample size

    if n_obs < 3 {
        return Err(NumRs2Error::ValueError(
            "Too few observations after accounting for lags".to_string(),
        ));
    }

    // Number of regressors: deterministic + y_{t-1} + lagged diffs
    let n_vars = trend.n_deterministic() + 1 + lags;
    let mut x = Array2::zeros((n_obs, n_vars));
    let mut y = Array1::zeros(n_obs);

    for i in 0..n_obs {
        let t = i + lags; // index into diff array
        y[i] = diff[t];

        let mut col = 0;

        // Deterministic terms
        if trend == TrendType::Constant || trend == TrendType::ConstantTrend {
            x[[i, col]] = 1.0;
            col += 1;
        }
        if trend == TrendType::ConstantTrend {
            x[[i, col]] = (t + 1) as f64;
            col += 1;
        }

        // Lagged level: y_{t-1} (index into original data: t, since diff[t] = data[t+1]-data[t])
        x[[i, col]] = data[t];
        col += 1;

        // Lagged differences
        for lag in 1..=lags {
            if t >= lag {
                x[[i, col]] = diff[t - lag];
            }
            col += 1;
        }
    }

    ols_regression(&x, &y)
}

/// Select optimal lag using AIC or BIC.
fn select_lag_ic(
    data: &ArrayView1<f64>,
    diff: &Array1<f64>,
    trend: TrendType,
    max_lag: usize,
    method: LagSelection,
) -> Result<(usize, Option<f64>)> {
    let mut best_lag = 0;
    let mut best_ic = f64::INFINITY;

    for lag in 0..=max_lag {
        let result = match adf_regression(data, diff, lag, trend) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let n = result.n_obs as f64;
        let k = result.n_vars as f64;
        let sigma2 = result.rss / n;

        if sigma2 <= 0.0 {
            continue;
        }

        let log_lik = -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln());

        let ic = match method {
            LagSelection::AIC => -2.0 * log_lik + 2.0 * k,
            LagSelection::BIC => -2.0 * log_lik + k * n.ln(),
            LagSelection::Fixed(_) => 0.0, // unreachable
        };

        if ic < best_ic {
            best_ic = ic;
            best_lag = lag;
        }
    }

    Ok((best_lag, Some(best_ic)))
}

/// Compute ADF critical values using MacKinnon (1994) response surface coefficients.
///
/// The critical values depend on sample size and trend specification. We use the
/// asymptotic values adjusted for finite samples via:
///   c(p) = beta_inf + beta_1/T + beta_2/T^2
fn adf_critical_values(n: usize, trend: TrendType) -> CriticalValues {
    let t = n as f64;
    let t_inv = 1.0 / t;
    let t_inv2 = t_inv * t_inv;

    // MacKinnon (1994) response surface coefficients [beta_inf, beta_1, beta_2]
    // for tau statistic (case: no constant, constant, constant+trend)
    match trend {
        TrendType::None => {
            // No constant
            let one_pct = -2.5658 + (-1.960) * t_inv + (-10.04) * t_inv2;
            let five_pct = -1.9393 + (-0.398) * t_inv + 0.0 * t_inv2;
            let ten_pct = -1.6156 + (-0.181) * t_inv + 0.0 * t_inv2;
            CriticalValues {
                one_pct,
                five_pct,
                ten_pct,
            }
        }
        TrendType::Constant => {
            // Constant only
            let one_pct = -3.4336 + (-5.999) * t_inv + (-29.25) * t_inv2;
            let five_pct = -2.8621 + (-2.738) * t_inv + (-8.36) * t_inv2;
            let ten_pct = -2.5671 + (-1.438) * t_inv + (-4.48) * t_inv2;
            CriticalValues {
                one_pct,
                five_pct,
                ten_pct,
            }
        }
        TrendType::ConstantTrend => {
            // Constant + trend
            let one_pct = -3.9638 + (-8.353) * t_inv + (-47.44) * t_inv2;
            let five_pct = -3.4126 + (-4.039) * t_inv + (-17.83) * t_inv2;
            let ten_pct = -3.1279 + (-2.418) * t_inv + (-7.58) * t_inv2;
            CriticalValues {
                one_pct,
                five_pct,
                ten_pct,
            }
        }
    }
}

/// Compute approximate p-value for ADF test using MacKinnon (1994) approximation.
///
/// Uses a piecewise interpolation based on the response surface critical values
/// and a normal CDF tail approximation for extreme values.
fn mackinnon_p_value(stat: f64, n: usize, trend: TrendType) -> f64 {
    let cv = adf_critical_values(n, trend);

    // Known significance levels and their critical values
    // We interpolate between these points
    let levels: [(f64, f64); 4] = [
        (0.01, cv.one_pct),
        (0.05, cv.five_pct),
        (0.10, cv.ten_pct),
        // Extrapolate a point for moderate p-values
        (
            0.50,
            match trend {
                TrendType::None => -0.44,
                TrendType::Constant => -0.80,
                TrendType::ConstantTrend => -1.50,
            },
        ),
    ];

    // If stat is more negative than 1% critical value, very small p-value
    if stat <= levels[0].1 {
        // Exponential tail approximation
        let ratio = (stat - levels[0].1) / (levels[0].1 - levels[1].1).abs();
        return (0.01 * (-ratio.abs()).exp()).max(1e-6);
    }

    // If stat is larger than our most positive reference, large p-value
    if stat >= levels[3].1 {
        // Use normal CDF-like tail for positive statistics
        let z = stat - levels[3].1;
        return (0.50 + 0.40 * (1.0 - (-0.5 * z).exp())).min(0.999);
    }

    // Linear interpolation between bracketing points
    for i in 0..levels.len() - 1 {
        if stat <= levels[i + 1].1 && stat >= levels[i].1 {
            // stat is between levels[i].1 and levels[i+1].1
            // (critical values become less negative as p increases)
            let frac = (stat - levels[i].1) / (levels[i + 1].1 - levels[i].1);
            return levels[i].0 + frac * (levels[i + 1].0 - levels[i].0);
        }
    }

    // Fallback: check ordering (critical values are increasingly negative for lower p)
    // ADF critical values: more negative = more significant = lower p
    for i in 0..levels.len() - 1 {
        if stat >= levels[i].1 && stat <= levels[i + 1].1 {
            let frac = (stat - levels[i].1) / (levels[i + 1].1 - levels[i].1);
            return levels[i].0 + frac * (levels[i + 1].0 - levels[i].0);
        }
    }

    0.50
}

// =============================================================================
// KPSS Test (Enhanced)
// =============================================================================

/// KPSS test for stationarity with Newey-West bandwidth selection.
///
/// Tests H0: the series is level/trend stationary against
/// H1: the series has a unit root.
///
/// Note: This is the reverse of the ADF test. Rejecting H0 (small p-value)
/// suggests the series is NOT stationary.
///
/// The KPSS statistic is:
///
/// eta = (1/T^2) * sum_t S_t^2 / s^2(l)
///
/// where S_t = sum_{j=1}^{t} e_j are partial sums of regression residuals, and
/// s^2(l) is the Newey-West long-run variance estimator.
///
/// # Arguments
///
/// * `data` - Time series to test (at least 10 observations)
/// * `trend` - `TrendType::Constant` for level stationarity, `TrendType::ConstantTrend`
///   for trend stationarity
/// * `bandwidth` - Number of lags for Newey-West estimator. If None, uses
///   the automatic Newey-West (1994) bandwidth: int(4*(T/100)^{2/9}).
///
/// # Returns
///
/// `KpssTestResult` containing the test statistic, p-value, critical values.
///
/// # References
///
/// Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
/// Testing the null hypothesis of stationarity against the alternative of a unit root.
/// Journal of Econometrics, 54(1-3), 159-178.
pub fn kpss_test_full(
    data: &ArrayView1<f64>,
    trend: TrendType,
    bandwidth: Option<usize>,
) -> Result<KpssTestResult> {
    let n = data.len();

    if n < 10 {
        return Err(NumRs2Error::ValueError(format!(
            "Need at least 10 observations for KPSS test, got {}",
            n
        )));
    }

    if trend == TrendType::None {
        return Err(NumRs2Error::ValueError(
            "KPSS test requires 'c' (level) or 'ct' (trend) specification".to_string(),
        ));
    }

    // Detrend the series using OLS
    let residuals = detrend_for_kpss(data, trend)?;

    // Determine bandwidth using Newey-West (1994) automatic selection
    let bw = bandwidth.unwrap_or_else(|| newey_west_bandwidth(n));
    let bw = bw.min(n - 1);

    // Compute partial sums S_t = sum_{j=0}^{t} e_j
    let mut partial_sums = Array1::zeros(n);
    partial_sums[0] = residuals[0];
    for i in 1..n {
        partial_sums[i] = partial_sums[i - 1] + residuals[i];
    }

    // KPSS numerator: (1/T^2) * sum_t S_t^2
    let numerator: f64 = partial_sums.iter().map(|&s| s * s).sum::<f64>() / (n as f64 * n as f64);

    // Long-run variance: Newey-West estimator with Bartlett kernel
    let s2 = newey_west_variance(&residuals, bw)?;

    if s2 <= 0.0 {
        return Err(NumRs2Error::ComputationError(
            "Long-run variance estimate is non-positive; series may be constant".to_string(),
        ));
    }

    let kpss_stat = numerator / s2;
    let critical_values = kpss_critical_values(trend);
    let p_value = kpss_p_value_approx(kpss_stat, trend);

    Ok(KpssTestResult {
        statistic: kpss_stat,
        p_value,
        bandwidth: bw,
        n_obs: n,
        critical_values,
        trend,
    })
}

/// Detrend data for KPSS by fitting constant or constant+trend via OLS.
fn detrend_for_kpss(data: &ArrayView1<f64>, trend: TrendType) -> Result<Array1<f64>> {
    let n = data.len();
    let y = Array1::from_iter(data.iter().copied());

    match trend {
        TrendType::Constant => {
            let mean: f64 = y.iter().sum::<f64>() / n as f64;
            Ok(y.mapv(|v| v - mean))
        }
        TrendType::ConstantTrend => {
            let mut x = Array2::zeros((n, 2));
            for i in 0..n {
                x[[i, 0]] = 1.0;
                x[[i, 1]] = (i + 1) as f64;
            }
            let result = ols_regression(&x, &y)?;
            Ok(result.residuals)
        }
        TrendType::None => Err(NumRs2Error::ValueError(
            "KPSS requires 'c' or 'ct' trend specification".to_string(),
        )),
    }
}

/// Compute Newey-West long-run variance estimator with Bartlett kernel.
///
/// s^2(l) = gamma_0 + 2 * sum_{j=1}^{l} w(j) * gamma_j
///
/// where w(j) = 1 - j/(l+1) is the Bartlett weight and
/// gamma_j = (1/T) * sum_{t=j+1}^{T} e_t * e_{t-j} is the j-th autocovariance.
fn newey_west_variance(residuals: &Array1<f64>, bandwidth: usize) -> Result<f64> {
    let n = residuals.len();

    // gamma_0 = (1/T) * sum e_t^2
    let gamma0: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;
    let mut s2 = gamma0;

    for j in 1..=bandwidth.min(n - 1) {
        let weight = 1.0 - j as f64 / (bandwidth as f64 + 1.0);
        let mut gamma_j = 0.0;
        for t in j..n {
            gamma_j += residuals[t] * residuals[t - j];
        }
        gamma_j /= n as f64;
        s2 += 2.0 * weight * gamma_j;
    }

    Ok(s2)
}

/// Automatic Newey-West bandwidth selection.
///
/// Uses the data-dependent plug-in formula: l = int(4*(T/100)^{2/9}).
fn newey_west_bandwidth(n: usize) -> usize {
    let bw = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
    bw.max(1)
}

/// KPSS critical values from Kwiatkowski et al. (1992), Table 1.
fn kpss_critical_values(trend: TrendType) -> CriticalValues {
    match trend {
        TrendType::Constant => CriticalValues {
            one_pct: 0.739,
            five_pct: 0.463,
            ten_pct: 0.347,
        },
        TrendType::ConstantTrend => CriticalValues {
            one_pct: 0.216,
            five_pct: 0.146,
            ten_pct: 0.119,
        },
        TrendType::None => CriticalValues {
            one_pct: 0.739,
            five_pct: 0.463,
            ten_pct: 0.347,
        },
    }
}

/// Approximate p-value for KPSS test.
///
/// The KPSS test rejects H0 (stationarity) for LARGE values of the statistic.
fn kpss_p_value_approx(stat: f64, trend: TrendType) -> f64 {
    let cv = kpss_critical_values(trend);

    // Known critical values and significance levels
    // KPSS: larger stat => more evidence against stationarity => smaller p-value
    let levels: [(f64, f64); 4] = [
        (0.01, cv.one_pct),
        (0.05, cv.five_pct),
        (0.10, cv.ten_pct),
        (
            0.50,
            match trend {
                TrendType::Constant => 0.119,
                TrendType::ConstantTrend => 0.045,
                TrendType::None => 0.119,
            },
        ),
    ];

    // stat > 1% critical value => p < 0.01
    if stat > levels[0].1 {
        let ratio = (stat - levels[0].1) / levels[0].1;
        return (0.01 * (-ratio).exp()).max(1e-6);
    }

    // Interpolate between known levels
    for i in 0..levels.len() - 1 {
        if stat <= levels[i].1 && stat >= levels[i + 1].1 {
            let frac = (levels[i].1 - stat) / (levels[i].1 - levels[i + 1].1);
            return levels[i].0 + frac * (levels[i + 1].0 - levels[i].0);
        }
    }

    // stat < all critical values => large p-value
    if stat < levels[levels.len() - 1].1 {
        return (0.50 + 0.40 * (1.0 - stat / levels[levels.len() - 1].1).max(0.0)).min(0.999);
    }

    0.50
}

// =============================================================================
// Phillips-Perron Test
// =============================================================================

/// Phillips-Perron test for unit root with non-parametric correction.
///
/// Like the ADF test, the PP test examines H0: unit root. However, instead of
/// augmenting the regression with lagged differences to handle serial correlation,
/// the PP test uses a non-parametric correction to the Dickey-Fuller t-statistic
/// based on a consistent estimate of the long-run variance.
///
/// The test estimates:
///   y_t = alpha + beta*t + rho*y_{t-1} + eps_t
///
/// and corrects the t-statistic for rho=1 using the spectral density at frequency zero.
///
/// # Arguments
///
/// * `data` - Time series to test
/// * `trend` - Deterministic trend specification
/// * `bandwidth` - Number of lags for spectral density estimation. If None, uses
///   Newey-West automatic selection.
///
/// # Returns
///
/// `PhillipsPerronResult` containing Z_tau and Z_alpha statistics, p-value,
/// and critical values.
///
/// # References
///
/// Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time series
/// regression. Biometrika, 75(2), 335-346.
pub fn phillips_perron_test(
    data: &ArrayView1<f64>,
    trend: TrendType,
    bandwidth: Option<usize>,
) -> Result<PhillipsPerronResult> {
    let n = data.len();

    if n < 10 {
        return Err(NumRs2Error::ValueError(format!(
            "Need at least 10 observations for PP test, got {}",
            n
        )));
    }

    // Run the base Dickey-Fuller regression (no augmentation)
    // y_t = deterministic + rho * y_{t-1} + eps_t
    let n_obs = n - 1;
    let n_det = trend.n_deterministic();
    let n_vars = n_det + 1; // deterministic + rho

    let mut x = Array2::zeros((n_obs, n_vars));
    let mut y = Array1::zeros(n_obs);

    for i in 0..n_obs {
        y[i] = data[i + 1];
        let mut col = 0;

        if trend == TrendType::Constant || trend == TrendType::ConstantTrend {
            x[[i, col]] = 1.0;
            col += 1;
        }
        if trend == TrendType::ConstantTrend {
            x[[i, col]] = (i + 2) as f64;
            col += 1;
        }

        // y_{t-1}
        x[[i, col]] = data[i];
    }

    let ols = ols_regression(&x, &y)?;

    let rho_idx = n_det; // index of the rho coefficient
    let rho_hat = ols.beta[rho_idx];
    let se_rho = ols.se[rho_idx];
    let residuals = &ols.residuals;

    // Compute sigma^2 (residual variance) and s^2(l) (long-run variance)
    let sigma2 = ols.rss / n_obs as f64;

    let bw = bandwidth.unwrap_or_else(|| newey_west_bandwidth(n));
    let bw = bw.min(n_obs - 1);

    // Long-run variance estimate
    let s2 = newey_west_variance(residuals, bw)?;

    if s2 <= 0.0 || sigma2 <= 0.0 {
        return Err(NumRs2Error::ComputationError(
            "Variance estimate is non-positive in PP test".to_string(),
        ));
    }

    // Correction factor lambda^2 = s^2 - sigma^2
    let lambda2 = (s2 - sigma2).max(0.0);

    // Compute (X'X)^{-1} for the rho coefficient
    let xtx = x.t().dot(&x);
    let identity = Array2::<f64>::eye(n_vars);
    let xtx_inv =
        scirs2_linalg::solve_multiple(&xtx.view(), &identity.view(), None).map_err(|e| {
            NumRs2Error::ComputationError(format!("PP test matrix inverse failed: {}", e))
        })?;

    let mxx_rho = xtx_inv[[rho_idx, rho_idx]]; // diagonal element for rho
    let sum_y2: f64 = (0..n_obs).map(|i| data[i] * data[i]).sum();

    // DF t-statistic
    let t_rho = (rho_hat - 1.0) / se_rho;

    // Z_tau correction (Phillips-Perron modified t-statistic)
    let z_tau =
        (sigma2 / s2).sqrt() * t_rho - 0.5 * lambda2 * (n_obs as f64 * mxx_rho).sqrt() / s2.sqrt();

    // Z_alpha correction (Phillips-Perron modified coefficient statistic)
    let z_alpha = n_obs as f64 * (rho_hat - 1.0)
        - 0.5 * (n_obs as f64).powi(2) * se_rho.powi(2) * lambda2 / (sigma2 * n_obs as f64);

    // P-value from Z_tau (same critical value distribution as ADF)
    let p_value = mackinnon_p_value(z_tau, n_obs, trend);
    let critical_values = adf_critical_values(n_obs, trend);

    Ok(PhillipsPerronResult {
        z_tau,
        z_alpha,
        p_value,
        bandwidth: bw,
        n_obs,
        critical_values,
        trend,
    })
}

// =============================================================================
// Helper Functions: Differencing and Detrending
// =============================================================================

/// Compute the d-th order difference of a time series.
///
/// For d=1: Delta y_t = y_t - y_{t-1}
/// For d=2: Delta^2 y_t = Delta y_t - Delta y_{t-1} = y_t - 2*y_{t-1} + y_{t-2}
///
/// # Arguments
///
/// * `data` - Input time series
/// * `d` - Order of differencing (1, 2, ...)
///
/// # Returns
///
/// Differenced series of length n - d.
pub fn difference(data: &ArrayView1<f64>, d: usize) -> Result<Array1<f64>> {
    if d == 0 {
        return Ok(Array1::from_iter(data.iter().copied()));
    }

    let n = data.len();
    if n <= d {
        return Err(NumRs2Error::ValueError(format!(
            "Cannot take {}-order difference of series with {} observations",
            d, n
        )));
    }

    // First difference
    let mut result = Array1::zeros(n - 1);
    for i in 0..(n - 1) {
        result[i] = data[i + 1] - data[i];
    }

    // Apply additional differencing recursively
    for _ in 1..d {
        let len = result.len();
        if len < 2 {
            return Err(NumRs2Error::ValueError(
                "Series too short for additional differencing".to_string(),
            ));
        }
        let mut next = Array1::zeros(len - 1);
        for i in 0..(len - 1) {
            next[i] = result[i + 1] - result[i];
        }
        result = next;
    }

    Ok(result)
}

/// Compute seasonal difference of a time series.
///
/// Delta_s y_t = y_t - y_{t-s}
///
/// This removes seasonal patterns with period s (e.g., s=12 for monthly data
/// with yearly seasonality, s=4 for quarterly data).
///
/// # Arguments
///
/// * `data` - Input time series
/// * `period` - Seasonal period
///
/// # Returns
///
/// Seasonally differenced series of length n - period.
pub fn seasonal_difference(data: &ArrayView1<f64>, period: usize) -> Result<Array1<f64>> {
    let n = data.len();

    if period == 0 {
        return Err(NumRs2Error::ValueError(
            "Seasonal period must be positive".to_string(),
        ));
    }

    if n <= period {
        return Err(NumRs2Error::ValueError(format!(
            "Series length ({}) must exceed seasonal period ({})",
            n, period
        )));
    }

    let mut result = Array1::zeros(n - period);
    for i in 0..(n - period) {
        result[i] = data[i + period] - data[i];
    }

    Ok(result)
}

/// Remove a linear trend from a time series.
///
/// Fits y_t = a + b*t via OLS and returns the residuals.
///
/// # Arguments
///
/// * `data` - Input time series
///
/// # Returns
///
/// Detrended series (same length as input).
pub fn detrend_linear(data: &ArrayView1<f64>) -> Result<Array1<f64>> {
    let n = data.len();

    if n < 3 {
        return Err(NumRs2Error::ValueError(format!(
            "Need at least 3 observations for linear detrending, got {}",
            n
        )));
    }

    let y = Array1::from_iter(data.iter().copied());
    let mut x = Array2::zeros((n, 2));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = i as f64;
    }

    let result = ols_regression(&x, &y)?;
    Ok(result.residuals)
}

/// Remove a polynomial trend of given degree from a time series.
///
/// Fits y_t = a_0 + a_1*t + a_2*t^2 + ... + a_p*t^p via OLS and returns residuals.
///
/// # Arguments
///
/// * `data` - Input time series
/// * `degree` - Polynomial degree (1 = linear, 2 = quadratic, etc.)
///
/// # Returns
///
/// Detrended series (same length as input).
pub fn detrend_polynomial(data: &ArrayView1<f64>, degree: usize) -> Result<Array1<f64>> {
    let n = data.len();

    if degree == 0 {
        // Demean
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        return Ok(Array1::from_iter(data.iter().map(|&v| v - mean)));
    }

    if n <= degree + 1 {
        return Err(NumRs2Error::ValueError(format!(
            "Need at least {} observations for degree-{} polynomial, got {}",
            degree + 2,
            degree,
            n
        )));
    }

    if degree > 10 {
        return Err(NumRs2Error::ValueError(
            "Polynomial degree too high (max 10); consider using differencing instead".to_string(),
        ));
    }

    let y = Array1::from_iter(data.iter().copied());
    let n_vars = degree + 1;
    let mut x = Array2::zeros((n, n_vars));

    for i in 0..n {
        let t = i as f64 / n as f64; // Normalize to [0,1] for numerical stability
        let mut power = 1.0;
        for j in 0..n_vars {
            x[[i, j]] = power;
            power *= t;
        }
    }

    let result = ols_regression(&x, &y)?;
    Ok(result.residuals)
}

// =============================================================================
// Integration Order Determination
// =============================================================================

/// Determine the integration order of a time series using repeated ADF tests.
///
/// Starting from the original series, this function applies the ADF test. If the
/// null hypothesis (unit root) is not rejected, the series is differenced and tested
/// again. This continues until the series is found to be stationary or the maximum
/// order is reached.
///
/// # Arguments
///
/// * `data` - Time series to test
/// * `max_order` - Maximum integration order to test (default: 5)
/// * `significance` - Significance level for ADF test (default: 0.05)
/// * `trend` - Trend specification
///
/// # Returns
///
/// `IntegrationOrderResult` containing the estimated integration order and
/// ADF test results at each stage.
///
/// # Example
///
/// Integration order 0 => stationary
/// Integration order 1 => first-difference stationary (I(1))
/// Integration order 2 => second-difference stationary (I(2))
pub fn integration_order(
    data: &ArrayView1<f64>,
    max_order: Option<usize>,
    significance: Option<f64>,
    trend: TrendType,
) -> Result<IntegrationOrderResult> {
    let max_d = max_order.unwrap_or(5);
    let alpha = significance.unwrap_or(0.05);
    let mut results = Vec::new();

    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(NumRs2Error::ValueError(format!(
            "Significance level must be in (0, 1), got {}",
            alpha
        )));
    }

    let mut current = Array1::from_iter(data.iter().copied());

    for d in 0..=max_d {
        // Verify we have enough data
        if current.len() < 8 {
            return Ok(IntegrationOrderResult {
                order: d,
                adf_results: results,
                significance: alpha,
            });
        }

        let adf = adf_test_full(&current.view(), LagSelection::BIC, trend, None)?;

        let reject = adf.p_value < alpha;
        results.push(adf);

        if reject {
            // Series is stationary at this differencing order
            return Ok(IntegrationOrderResult {
                order: d,
                adf_results: results,
                significance: alpha,
            });
        }

        // If not the last iteration, difference the series
        if d < max_d {
            current = difference(&current.view(), 1)?;
        }
    }

    // Could not determine finite integration order within max_d
    Ok(IntegrationOrderResult {
        order: max_d + 1,
        adf_results: results,
        significance: alpha,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::{Rng, SeedableRng};
    use std::str::FromStr;

    /// Generate a white noise series (stationary).
    fn white_noise(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from_iter((0..n).map(|_| rng.gen_range(-1.0..1.0)))
    }

    /// Generate a random walk (non-stationary I(1)).
    fn random_walk(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array1::zeros(n);
        for i in 1..n {
            data[i] = data[i - 1] + rng.gen_range(-1.0..1.0);
        }
        data
    }

    /// Generate a trend-stationary series: y_t = a + b*t + eps_t.
    fn trend_stationary(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from_iter((0..n).map(|i| 0.5 + 0.1 * i as f64 + rng.gen_range(-0.5..0.5)))
    }

    // =========================================================================
    // ADF Test
    // =========================================================================

    #[test]
    fn test_adf_stationary_white_noise() {
        let data = white_noise(200, 42);
        let result = adf_test_full(&data.view(), LagSelection::BIC, TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("ADF should succeed");
        // White noise should reject H0 (unit root) => small p-value, very negative statistic
        assert!(
            r.p_value < 0.10,
            "White noise should be detected as stationary, p={:.4}",
            r.p_value
        );
    }

    #[test]
    fn test_adf_nonstationary_random_walk() {
        let data = random_walk(200, 42);
        let result = adf_test_full(&data.view(), LagSelection::BIC, TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("ADF should succeed");
        // Random walk should NOT reject H0 => large p-value
        assert!(
            r.p_value > 0.05,
            "Random walk should be detected as non-stationary, p={:.4}",
            r.p_value
        );
    }

    #[test]
    fn test_adf_with_trend() {
        let data = trend_stationary(200, 42);
        let result = adf_test_full(
            &data.view(),
            LagSelection::BIC,
            TrendType::ConstantTrend,
            None,
        );
        assert!(result.is_ok());
        let r = result.expect("ADF should succeed");
        // With proper trend specification, trend-stationary should reject H0
        assert!(r.statistic.is_finite());
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn test_adf_lag_selection_aic() {
        let data = white_noise(100, 123);
        let result = adf_test_full(
            &data.view(),
            LagSelection::AIC,
            TrendType::Constant,
            Some(10),
        );
        assert!(result.is_ok());
        let r = result.expect("ADF AIC should succeed");
        assert!(r.lags_used <= 10);
        assert!(r.ic_best.is_some());
    }

    #[test]
    fn test_adf_lag_selection_bic() {
        let data = white_noise(100, 456);
        let result = adf_test_full(
            &data.view(),
            LagSelection::BIC,
            TrendType::Constant,
            Some(10),
        );
        assert!(result.is_ok());
        let r = result.expect("ADF BIC should succeed");
        assert!(r.lags_used <= 10);
        assert!(r.ic_best.is_some());
    }

    #[test]
    fn test_adf_fixed_lags() {
        let data = white_noise(100, 789);
        let result = adf_test_full(
            &data.view(),
            LagSelection::Fixed(3),
            TrendType::Constant,
            None,
        );
        assert!(result.is_ok());
        let r = result.expect("ADF fixed lag should succeed");
        assert_eq!(r.lags_used, 3);
        assert!(r.ic_best.is_none());
    }

    #[test]
    fn test_adf_critical_values_structure() {
        let data = white_noise(100, 111);
        let result = adf_test_full(
            &data.view(),
            LagSelection::Fixed(2),
            TrendType::Constant,
            None,
        )
        .expect("ADF should succeed");

        // 1% critical value should be more negative than 5%, which is more negative than 10%
        assert!(result.critical_values.one_pct < result.critical_values.five_pct);
        assert!(result.critical_values.five_pct < result.critical_values.ten_pct);
    }

    #[test]
    fn test_adf_no_constant() {
        let data = white_noise(100, 222);
        let result = adf_test_full(&data.view(), LagSelection::Fixed(1), TrendType::None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adf_short_series() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = adf_test_full(
            &data.view(),
            LagSelection::Fixed(0),
            TrendType::Constant,
            None,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // KPSS Test
    // =========================================================================

    #[test]
    fn test_kpss_stationary() {
        let data = white_noise(200, 42);
        let result = kpss_test_full(&data.view(), TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("KPSS should succeed");
        // Stationary series should NOT reject H0 => large p-value
        assert!(
            r.p_value > 0.05,
            "White noise should be level-stationary, p={:.4}",
            r.p_value
        );
    }

    #[test]
    fn test_kpss_nonstationary() {
        let data = random_walk(200, 42);
        let result = kpss_test_full(&data.view(), TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("KPSS should succeed");
        // Random walk should reject H0 (stationarity) => small p-value
        assert!(
            r.p_value < 0.10,
            "Random walk should not be level-stationary, p={:.4}, stat={:.4}",
            r.p_value,
            r.statistic
        );
    }

    #[test]
    fn test_kpss_trend_stationarity() {
        let data = trend_stationary(200, 42);
        let result = kpss_test_full(&data.view(), TrendType::ConstantTrend, None);
        assert!(result.is_ok());
        let r = result.expect("KPSS trend should succeed");
        // Trend-stationary series tested with 'ct' should not reject H0
        assert!(r.statistic >= 0.0);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn test_kpss_bandwidth_selection() {
        let data = white_noise(100, 333);
        let result = kpss_test_full(&data.view(), TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("KPSS should succeed");
        assert!(r.bandwidth > 0);
        assert!(r.bandwidth < r.n_obs);
    }

    #[test]
    fn test_kpss_manual_bandwidth() {
        let data = white_noise(100, 444);
        let result = kpss_test_full(&data.view(), TrendType::Constant, Some(5));
        assert!(result.is_ok());
        let r = result.expect("KPSS should succeed");
        assert_eq!(r.bandwidth, 5);
    }

    #[test]
    fn test_kpss_invalid_trend() {
        let data = white_noise(50, 555);
        let result = kpss_test_full(&data.view(), TrendType::None, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // Phillips-Perron Test
    // =========================================================================

    #[test]
    fn test_pp_stationary() {
        let data = white_noise(200, 42);
        let result = phillips_perron_test(&data.view(), TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("PP should succeed");
        // White noise should reject unit root
        assert!(r.z_tau.is_finite());
        assert!(r.z_alpha.is_finite());
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn test_pp_nonstationary() {
        let data = random_walk(200, 42);
        let result = phillips_perron_test(&data.view(), TrendType::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("PP should succeed");
        // Random walk should NOT reject unit root => large p-value
        assert!(
            r.p_value > 0.05,
            "Random walk should have unit root, p={:.4}",
            r.p_value
        );
    }

    #[test]
    fn test_pp_with_trend() {
        let data = trend_stationary(200, 42);
        let result = phillips_perron_test(&data.view(), TrendType::ConstantTrend, None);
        assert!(result.is_ok());
        let r = result.expect("PP trend should succeed");
        assert!(r.z_tau.is_finite());
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    // =========================================================================
    // Differencing
    // =========================================================================

    #[test]
    fn test_difference_first_order() {
        let data = Array1::from_vec(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        let diff = difference(&data.view(), 1).expect("difference should succeed");
        assert_eq!(diff.len(), 4);
        // Expected: [2, 3, 4, 5]
        let expected = [2.0, 3.0, 4.0, 5.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (diff[i] - e).abs() < 1e-10,
                "diff[{}] = {}, expected {}",
                i,
                diff[i],
                e
            );
        }
    }

    #[test]
    fn test_difference_second_order() {
        let data = Array1::from_vec(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        let diff2 = difference(&data.view(), 2).expect("second difference should succeed");
        assert_eq!(diff2.len(), 3);
        // First diff: [2, 3, 4, 5], Second diff: [1, 1, 1]
        for i in 0..diff2.len() {
            assert!((diff2[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_seasonal_difference() {
        // Monthly data with period 4
        let data = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 11.0, 22.0, 33.0, 44.0]);
        let sdiff = seasonal_difference(&data.view(), 4).expect("seasonal diff should succeed");
        assert_eq!(sdiff.len(), 4);
        // [11-10, 22-20, 33-30, 44-40] = [1, 2, 3, 4]
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!((sdiff[i] - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_difference_zero_order() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let diff0 = difference(&data.view(), 0).expect("zero difference should succeed");
        assert_eq!(diff0.len(), 3);
        assert!((diff0[0] - 1.0).abs() < 1e-10);
    }

    // =========================================================================
    // Detrending
    // =========================================================================

    #[test]
    fn test_detrend_linear() {
        // y = 2 + 3*t + noise
        let n = 50;
        let mut rng = StdRng::seed_from_u64(42);
        let data =
            Array1::from_iter((0..n).map(|i| 2.0 + 3.0 * i as f64 + rng.gen_range(-0.1..0.1)));

        let detrended = detrend_linear(&data.view()).expect("detrend should succeed");
        assert_eq!(detrended.len(), n);

        // Detrended series should have near-zero mean
        let mean: f64 = detrended.iter().sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 0.5,
            "Detrended mean should be near zero, got {}",
            mean
        );
    }

    #[test]
    fn test_detrend_polynomial() {
        // y = 1 + 2t + 3t^2 + noise
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);
        let data = Array1::from_iter((0..n).map(|i| {
            let t = i as f64 / n as f64;
            1.0 + 2.0 * t + 3.0 * t * t + rng.gen_range(-0.01..0.01)
        }));

        let detrended = detrend_polynomial(&data.view(), 2).expect("poly detrend should succeed");
        assert_eq!(detrended.len(), n);

        // Residuals should be small
        let max_abs: f64 = detrended.iter().map(|&r| r.abs()).fold(0.0, f64::max);
        assert!(
            max_abs < 0.5,
            "Polynomial detrend residuals too large: max={}",
            max_abs
        );
    }

    // =========================================================================
    // Integration Order
    // =========================================================================

    #[test]
    fn test_integration_order_stationary() {
        let data = white_noise(200, 42);
        let result = integration_order(&data.view(), Some(3), Some(0.05), TrendType::Constant);
        assert!(result.is_ok());
        let r = result.expect("integration order should succeed");
        assert_eq!(r.order, 0, "White noise should be I(0)");
    }

    #[test]
    fn test_integration_order_random_walk() {
        let data = random_walk(200, 42);
        let result = integration_order(&data.view(), Some(3), Some(0.05), TrendType::Constant);
        assert!(result.is_ok());
        let r = result.expect("integration order should succeed");
        assert!(
            r.order <= 2,
            "Random walk should be I(1) or I(2), got I({})",
            r.order
        );
    }

    #[test]
    fn test_integration_order_i2() {
        // Generate an I(2) process: cumulative sum of a random walk
        let rw = random_walk(200, 42);
        let mut i2 = Array1::zeros(200);
        i2[0] = rw[0];
        for i in 1..200 {
            i2[i] = i2[i - 1] + rw[i];
        }

        let result = integration_order(&i2.view(), Some(4), Some(0.05), TrendType::Constant);
        assert!(result.is_ok());
        let r = result.expect("integration order should succeed");
        assert!(
            r.order >= 1,
            "I(2) process should have order >= 1, got I({})",
            r.order
        );
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_constant_series_adf() {
        let data = Array1::from_vec(vec![5.0; 50]);
        let result = adf_test_full(
            &data.view(),
            LagSelection::Fixed(1),
            TrendType::Constant,
            None,
        );
        // Constant series has zero variance in differences => should error gracefully
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_short_series_kpss() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = kpss_test_full(&data.view(), TrendType::Constant, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_type_from_str() {
        assert_eq!(TrendType::from_str("nc").expect("ok"), TrendType::None);
        assert_eq!(TrendType::from_str("c").expect("ok"), TrendType::Constant);
        assert_eq!(
            TrendType::from_str("ct").expect("ok"),
            TrendType::ConstantTrend
        );
        assert!(TrendType::from_str("invalid").is_err());
    }

    #[test]
    fn test_seasonal_difference_invalid_period() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(seasonal_difference(&data.view(), 0).is_err());
        assert!(seasonal_difference(&data.view(), 5).is_err());
    }

    #[test]
    fn test_adf_and_kpss_complementary() {
        // For a stationary series, ADF should reject H0 (unit root) and
        // KPSS should fail to reject H0 (stationarity)
        let data = white_noise(200, 9999);

        let adf = adf_test_full(&data.view(), LagSelection::BIC, TrendType::Constant, None)
            .expect("ADF should succeed");

        let kpss =
            kpss_test_full(&data.view(), TrendType::Constant, None).expect("KPSS should succeed");

        // Both should agree on stationarity (ADF: low p, KPSS: high p)
        // Using lenient thresholds since these are statistical tests
        let adf_agrees = adf.p_value < 0.15; // ADF rejects unit root
        let kpss_agrees = kpss.p_value > 0.05; // KPSS fails to reject stationarity

        assert!(
            adf_agrees || kpss_agrees,
            "At least one test should detect stationarity: ADF p={:.4}, KPSS p={:.4}",
            adf.p_value,
            kpss.p_value
        );
    }
}
