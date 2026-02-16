//! Cointegration Testing for Multivariate Time Series
//!
//! This module provides comprehensive cointegration analysis methods for
//! determining long-run equilibrium relationships between non-stationary
//! time series. Cointegration occurs when a linear combination of
//! non-stationary I(1) series is stationary I(0).
//!
//! ## Implemented Tests
//!
//! - **Engle-Granger Two-Step Test**: Residual-based test using OLS regression
//!   followed by ADF test on residuals.
//! - **Johansen Cointegration Test**: System-based test using reduced rank
//!   regression with trace and maximum eigenvalue statistics.
//! - **Phillips-Ouliaris Test**: Residual-based test with Z_alpha and Z_tau
//!   statistics, robust to serial correlation.
//! - **Error Correction Model (VECM)**: Estimates speed of adjustment parameters
//!   and long-run equilibrium from cointegrated systems.
//!
//! ## References
//!
//! - Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction:
//!   Representation, Estimation, and Testing." *Econometrica*, 55(2), 251-276.
//! - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors
//!   in Gaussian Vector Autoregressive Models." *Econometrica*, 59(6), 1551-1580.
//! - Phillips, P.C.B. and Ouliaris, S. (1990). "Asymptotic Properties of Residual
//!   Based Tests for Cointegration." *Econometrica*, 58(1), 165-193.
//!
//! ## SCIRS2 Policy Compliance
//!
//! - Uses `scirs2_core::ndarray` for all array operations
//! - Uses `scirs2_linalg` for eigenvalue decomposition and linear algebra
//! - Uses `scirs2_core::random` for random number generation (tests only)
//! - No unwrap() usage; proper error handling throughout

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

// =============================================================================
// Data Structures
// =============================================================================

/// Trend specification for cointegration tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendSpec {
    /// No constant, no trend
    None,
    /// Constant only (restricted to cointegrating space for Johansen)
    Constant,
    /// Constant and linear trend
    ConstantTrend,
}

/// Result of the Engle-Granger cointegration test.
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    /// ADF test statistic on residuals
    pub adf_statistic: f64,
    /// Approximate p-value
    pub p_value: f64,
    /// OLS regression coefficients (first variable regressed on others)
    pub ols_coefficients: Array1<f64>,
    /// Residuals from OLS regression
    pub residuals: Array1<f64>,
    /// Critical values at 1%, 5%, 10% significance
    pub critical_values: [f64; 3],
    /// Whether null hypothesis of no cointegration is rejected at 5%
    pub cointegrated: bool,
}

/// Result of the Johansen cointegration test.
#[derive(Debug, Clone)]
pub struct JohansenResult {
    /// Trace statistics for each rank hypothesis
    pub trace_statistics: Array1<f64>,
    /// Maximum eigenvalue statistics
    pub max_eigenvalue_statistics: Array1<f64>,
    /// Eigenvalues from reduced rank regression
    pub eigenvalues: Array1<f64>,
    /// Cointegrating vectors (columns of beta matrix)
    pub cointegrating_vectors: Array2<f64>,
    /// Adjustment coefficients (alpha matrix)
    pub adjustment_coefficients: Array2<f64>,
    /// Critical values for trace test (rows: rank, cols: 10%, 5%, 1%)
    pub trace_critical_values: Array2<f64>,
    /// Critical values for max eigenvalue test
    pub max_eig_critical_values: Array2<f64>,
    /// Estimated cointegration rank
    pub rank: usize,
}

/// Result of the Phillips-Ouliaris cointegration test.
#[derive(Debug, Clone)]
pub struct PhillipsOuliarisResult {
    /// Z_alpha statistic
    pub z_alpha: f64,
    /// Z_tau statistic (t-ratio version)
    pub z_tau: f64,
    /// Critical values for Z_alpha at 1%, 5%, 10%
    pub z_alpha_critical_values: [f64; 3],
    /// Critical values for Z_tau at 1%, 5%, 10%
    pub z_tau_critical_values: [f64; 3],
    /// Whether null of no cointegration is rejected at 5%
    pub cointegrated: bool,
    /// Residuals from initial OLS
    pub residuals: Array1<f64>,
}

/// Error Correction Model (VECM) result.
#[derive(Debug, Clone)]
pub struct VecmResult {
    /// Speed of adjustment parameters (alpha): k x r matrix
    pub alpha: Array2<f64>,
    /// Long-run cointegrating vectors (beta): k x r matrix
    pub beta: Array2<f64>,
    /// Short-run dynamics coefficient matrices (Gamma_i for each lag)
    pub gamma: Vec<Array2<f64>>,
    /// Intercept vector
    pub intercept: Array1<f64>,
    /// Residuals from VECM estimation
    pub residuals: Array2<f64>,
    /// Residual covariance matrix
    pub sigma: Array2<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Cointegration rank used
    pub rank: usize,
}

// =============================================================================
// Helper: OLS Regression
// =============================================================================

/// Perform OLS regression: y = X * beta + epsilon.
///
/// Returns (coefficients, residuals, standard_errors).
fn ols_regression(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let n = y.len();
    let (n_x, k) = x.dim();

    if n != n_x {
        return Err(NumRs2Error::ValueError(format!(
            "y length ({}) must match X rows ({})",
            n, n_x
        )));
    }

    if n <= k {
        return Err(NumRs2Error::ValueError(
            "Insufficient observations for OLS regression".to_string(),
        ));
    }

    // beta = (X'X)^{-1} X'y
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);

    let beta = scirs2_linalg::solve(&xtx.view(), &xty.view(), None).map_err(|_| {
        NumRs2Error::ComputationError("Singular matrix in OLS regression".to_string())
    })?;

    // residuals = y - X * beta
    let fitted = x.dot(&beta);
    let residuals = y.to_owned() - &fitted;

    // Standard errors
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let s2 = rss / (n - k) as f64;

    let eye = Array2::<f64>::eye(k);
    let xtx_inv = scirs2_linalg::solve_multiple(&xtx.view(), &eye.view(), None).map_err(|_| {
        NumRs2Error::ComputationError("Cannot compute (X'X)^{-1} for standard errors".to_string())
    })?;

    let mut se = Array1::zeros(k);
    for i in 0..k {
        let var_i = s2 * xtx_inv[[i, i]];
        se[i] = if var_i > 0.0 { var_i.sqrt() } else { 0.0 };
    }

    Ok((beta, residuals, se))
}

/// Perform multivariate OLS regression: Y = X * B + E.
///
/// Y is (n x m), X is (n x k), B is (k x m), E is (n x m).
fn ols_regression_multi(
    y: &ArrayView2<f64>,
    x: &ArrayView2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let (n_y, _m) = y.dim();
    let (n_x, _k) = x.dim();

    if n_y != n_x {
        return Err(NumRs2Error::ValueError(format!(
            "Y rows ({}) must match X rows ({})",
            n_y, n_x
        )));
    }

    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);

    let beta = scirs2_linalg::solve_multiple(&xtx.view(), &xty.view(), None).map_err(|_| {
        NumRs2Error::ComputationError("Singular matrix in multivariate OLS".to_string())
    })?;

    let residuals = y.to_owned() - &x.dot(&beta);

    Ok((beta, residuals))
}

// =============================================================================
// ADF Test for Residuals (Cointegration-specific critical values)
// =============================================================================

/// Run ADF test on residuals with cointegration-specific critical values.
///
/// Critical values for cointegration residuals differ from standard ADF
/// because the residuals are generated regressors (Engle-Granger, 1987).
fn adf_test_residuals(
    residuals: &ArrayView1<f64>,
    lags: usize,
    n_vars: usize,
) -> Result<(f64, f64, [f64; 3])> {
    let n = residuals.len();

    if n < lags + 3 {
        return Err(NumRs2Error::ValueError(
            "Insufficient observations for ADF test on residuals".to_string(),
        ));
    }

    // First difference of residuals
    let mut diff = Array1::zeros(n - 1);
    for i in 0..(n - 1) {
        diff[i] = residuals[i + 1] - residuals[i];
    }

    // Build regression: delta_e_t on e_{t-1} and lagged differences
    let n_obs = n - lags - 1;
    let n_regressors = 1 + lags; // no constant for residual-based test

    if n_obs <= n_regressors {
        return Err(NumRs2Error::ValueError(
            "Too many lags for the sample size".to_string(),
        ));
    }

    let mut x_mat = Array2::zeros((n_obs, n_regressors));
    let mut y_vec = Array1::zeros(n_obs);

    for i in 0..n_obs {
        let t = i + lags;
        y_vec[i] = diff[t];

        // Lagged level
        x_mat[[i, 0]] = residuals[t];

        // Lagged differences
        for lag_idx in 1..=lags {
            if t >= lag_idx {
                x_mat[[i, lag_idx]] = diff[t - lag_idx];
            }
        }
    }

    // OLS
    let (beta, resid, se) = ols_regression(&y_vec.view(), &x_mat.view())?;

    let gamma = beta[0];
    let se_gamma = se[0];

    let adf_stat = if se_gamma.abs() > 1e-15 {
        gamma / se_gamma
    } else {
        // If standard error is essentially zero, the coefficient is essentially zero too
        0.0
    };

    // Cointegration-specific critical values (Engle-Granger / MacKinnon)
    // These depend on the number of variables in the cointegrating regression
    let critical_values = engle_granger_critical_values(n_vars, n);

    // Approximate p-value
    let p_value = if adf_stat < critical_values[0] {
        0.005
    } else if adf_stat < critical_values[1] {
        0.025
    } else if adf_stat < critical_values[2] {
        0.075
    } else {
        0.50
    };

    Ok((adf_stat, p_value, critical_values))
}

/// Engle-Granger critical values for cointegration test.
///
/// Based on MacKinnon (1991, 2010) response surface regressions.
/// Returns [1%, 5%, 10%] critical values.
fn engle_granger_critical_values(n_vars: usize, n: usize) -> [f64; 3] {
    // MacKinnon (2010) asymptotic critical values for cointegration tests
    // (no constant in ADF on residuals; constant was in cointegrating regression)
    // n_vars includes the dependent variable
    let n_f = n as f64;
    let inv_n = 1.0 / n_f;

    match n_vars {
        2 => {
            // Two-variable case
            let cv_1 = -3.9001 - 10.534 * inv_n;
            let cv_5 = -3.3377 - 5.967 * inv_n;
            let cv_10 = -3.0462 - 4.069 * inv_n;
            [cv_1, cv_5, cv_10]
        }
        3 => {
            let cv_1 = -4.2981 - 13.790 * inv_n;
            let cv_5 = -3.7429 - 8.352 * inv_n;
            let cv_10 = -3.4518 - 6.241 * inv_n;
            [cv_1, cv_5, cv_10]
        }
        4 => {
            let cv_1 = -4.6493 - 17.188 * inv_n;
            let cv_5 = -4.1000 - 11.225 * inv_n;
            let cv_10 = -3.8110 - 8.736 * inv_n;
            [cv_1, cv_5, cv_10]
        }
        5 => {
            let cv_1 = -4.9695 - 22.504 * inv_n;
            let cv_5 = -4.4185 - 14.501 * inv_n;
            let cv_10 = -4.1327 - 11.165 * inv_n;
            [cv_1, cv_5, cv_10]
        }
        _ => {
            // Extrapolation for larger systems
            let base_1 = -3.9001 - 0.35 * (n_vars as f64 - 2.0);
            let base_5 = -3.3377 - 0.35 * (n_vars as f64 - 2.0);
            let base_10 = -3.0462 - 0.35 * (n_vars as f64 - 2.0);
            [
                base_1 - 10.0 * inv_n,
                base_5 - 6.0 * inv_n,
                base_10 - 4.0 * inv_n,
            ]
        }
    }
}

// =============================================================================
// Engle-Granger Two-Step Cointegration Test
// =============================================================================

/// Perform the Engle-Granger two-step cointegration test.
///
/// Step 1: Run OLS regression of y1 on y2, ..., yn (and optionally a constant).
/// Step 2: Test the residuals for stationarity using ADF test with
///         cointegration-specific critical values.
///
/// # Arguments
///
/// * `data` - Multivariate time series as (T x k) matrix. The first column
///   is treated as the dependent variable.
/// * `lags` - Number of lags for the ADF test on residuals.
/// * `trend` - Trend specification for the cointegrating regression.
///
/// # Returns
///
/// [`EngleGrangerResult`] containing test statistics and critical values.
///
/// # References
///
/// Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction."
pub fn engle_granger_test(
    data: &ArrayView2<f64>,
    lags: usize,
    trend: TrendSpec,
) -> Result<EngleGrangerResult> {
    let (t, k) = data.dim();

    if k < 2 {
        return Err(NumRs2Error::ValueError(
            "Engle-Granger test requires at least 2 variables".to_string(),
        ));
    }

    if t < 20 {
        return Err(NumRs2Error::ValueError(
            "Engle-Granger test requires at least 20 observations".to_string(),
        ));
    }

    // Step 1: OLS regression y1 = a + b2*y2 + ... + bk*yk + epsilon
    let y = data.column(0);

    // Build design matrix with remaining variables and optional trend
    let n_design_cols = match trend {
        TrendSpec::None => k - 1,
        TrendSpec::Constant => k,
        TrendSpec::ConstantTrend => k + 1,
    };

    let mut x_mat = Array2::zeros((t, n_design_cols));
    let mut col_idx = 0;

    // Constant
    if trend == TrendSpec::Constant || trend == TrendSpec::ConstantTrend {
        for i in 0..t {
            x_mat[[i, col_idx]] = 1.0;
        }
        col_idx += 1;
    }

    // Time trend
    if trend == TrendSpec::ConstantTrend {
        for i in 0..t {
            x_mat[[i, col_idx]] = (i + 1) as f64;
        }
        col_idx += 1;
    }

    // Other variables
    for j in 1..k {
        for i in 0..t {
            x_mat[[i, col_idx]] = data[[i, j]];
        }
        col_idx += 1;
    }

    let (coefficients, residuals, _se) = ols_regression(&y, &x_mat.view())?;

    // Step 2: ADF test on residuals with cointegration critical values
    let (adf_stat, p_value, critical_values) = adf_test_residuals(&residuals.view(), lags, k)?;

    let cointegrated = adf_stat < critical_values[1]; // Reject at 5%

    Ok(EngleGrangerResult {
        adf_statistic: adf_stat,
        p_value,
        ols_coefficients: coefficients,
        residuals,
        critical_values,
        cointegrated,
    })
}

// =============================================================================
// Johansen Cointegration Test
// =============================================================================

/// Johansen critical values for the trace test.
///
/// Returns a matrix where each row corresponds to rank r = 0, 1, ..., k-1
/// and columns are [10%, 5%, 1%] critical values.
///
/// Based on Osterwald-Lenum (1992) tables.
fn johansen_trace_critical_values(k: usize, trend: TrendSpec) -> Array2<f64> {
    let mut cv = Array2::zeros((k, 3));

    // Tables from Osterwald-Lenum (1992) for "constant restricted" case
    // Organized by p-r (number of variables minus rank)
    let tables = match trend {
        TrendSpec::None => {
            // No intercept, no trend (Case 1)
            vec![
                // p-r: 1, 2, 3, 4, 5, 6
                [2.69, 3.76, 6.65],     // p-r = 1
                [13.33, 15.41, 20.04],  // p-r = 2
                [26.79, 29.68, 35.65],  // p-r = 3
                [43.95, 47.21, 54.46],  // p-r = 4
                [64.84, 68.52, 76.07],  // p-r = 5
                [89.48, 94.15, 103.18], // p-r = 6
            ]
        }
        TrendSpec::Constant => {
            // Restricted constant (Case 2/3 - most common)
            vec![
                [7.52, 9.24, 12.97],
                [17.85, 19.96, 24.60],
                [32.00, 34.91, 41.07],
                [49.65, 53.12, 60.16],
                [71.86, 76.07, 84.45],
                [97.18, 102.14, 111.01],
            ]
        }
        TrendSpec::ConstantTrend => {
            // Constant and trend (Case 4)
            vec![
                [10.49, 12.53, 16.31],
                [22.76, 25.32, 30.45],
                [39.06, 42.44, 48.45],
                [59.14, 62.99, 70.05],
                [83.20, 87.31, 96.58],
                [110.42, 114.90, 124.75],
            ]
        }
    };

    for r in 0..k {
        let p_minus_r = k - r;
        if p_minus_r >= 1 && p_minus_r <= tables.len() {
            let row = &tables[p_minus_r - 1];
            cv[[r, 0]] = row[0];
            cv[[r, 1]] = row[1];
            cv[[r, 2]] = row[2];
        } else {
            // Extrapolation for dimensions beyond the table
            let scale = p_minus_r as f64;
            cv[[r, 0]] = 7.5 * scale;
            cv[[r, 1]] = 9.0 * scale;
            cv[[r, 2]] = 12.0 * scale;
        }
    }

    cv
}

/// Johansen critical values for the maximum eigenvalue test.
fn johansen_max_eig_critical_values(k: usize, trend: TrendSpec) -> Array2<f64> {
    let mut cv = Array2::zeros((k, 3));

    let tables = match trend {
        TrendSpec::None => {
            vec![
                [2.69, 3.76, 6.65],
                [12.07, 14.07, 18.63],
                [18.60, 20.97, 25.52],
                [24.73, 27.07, 32.24],
                [30.67, 33.46, 38.77],
                [36.76, 39.37, 44.59],
            ]
        }
        TrendSpec::Constant => {
            vec![
                [7.52, 9.24, 12.97],
                [13.75, 15.67, 20.20],
                [19.77, 22.00, 26.81],
                [25.56, 28.14, 33.24],
                [31.66, 33.32, 39.43],
                [37.45, 39.43, 44.59],
            ]
        }
        TrendSpec::ConstantTrend => {
            vec![
                [10.49, 12.53, 16.31],
                [16.85, 18.96, 23.65],
                [23.11, 25.54, 30.34],
                [29.12, 31.46, 36.65],
                [34.75, 37.52, 42.36],
                [40.91, 43.97, 49.51],
            ]
        }
    };

    for r in 0..k {
        let p_minus_r = k - r;
        if p_minus_r >= 1 && p_minus_r <= tables.len() {
            let row = &tables[p_minus_r - 1];
            cv[[r, 0]] = row[0];
            cv[[r, 1]] = row[1];
            cv[[r, 2]] = row[2];
        } else {
            let scale = p_minus_r as f64;
            cv[[r, 0]] = 7.0 * scale;
            cv[[r, 1]] = 8.5 * scale;
            cv[[r, 2]] = 11.0 * scale;
        }
    }

    cv
}

/// Perform the Johansen cointegration test.
///
/// Uses reduced rank regression to test for the number of cointegrating
/// relationships (rank) in a multivariate time series system.
///
/// The test estimates the VECM:
///   delta_Y_t = Pi * Y_{t-1} + sum_i Gamma_i * delta_Y_{t-i} + epsilon_t
///
/// where Pi = alpha * beta' has reduced rank r equal to the number
/// of cointegrating relationships.
///
/// # Arguments
///
/// * `data` - Multivariate time series as (T x k) matrix
/// * `lags` - Number of lags in the VECM (in levels, so lags-1 differenced lags)
/// * `trend` - Trend specification
///
/// # Returns
///
/// [`JohansenResult`] with trace and max eigenvalue statistics.
///
/// # References
///
/// Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration
/// Vectors in Gaussian Vector Autoregressive Models."
pub fn johansen_test(
    data: &ArrayView2<f64>,
    lags: usize,
    trend: TrendSpec,
) -> Result<JohansenResult> {
    let (t, k) = data.dim();

    if k < 2 {
        return Err(NumRs2Error::ValueError(
            "Johansen test requires at least 2 variables".to_string(),
        ));
    }

    if t < 2 * k + lags + 5 {
        return Err(NumRs2Error::ValueError(format!(
            "Insufficient observations ({}) for Johansen test with {} variables and {} lags",
            t, k, lags
        )));
    }

    let lags_vecm = if lags > 0 { lags } else { 1 };

    // Compute first differences
    let mut delta_y = Array2::zeros((t - 1, k));
    for i in 0..(t - 1) {
        for j in 0..k {
            delta_y[[i, j]] = data[[i + 1, j]] - data[[i, j]];
        }
    }

    // Effective sample after accounting for lags
    let n_eff = t - lags_vecm;
    if n_eff <= k + lags_vecm {
        return Err(NumRs2Error::ValueError(
            "Effective sample size too small".to_string(),
        ));
    }

    // Build matrices for reduced rank regression
    // Y0 = delta_Y[lags_vecm..], Y1 = Y[lags_vecm-1..T-1] (lagged levels)
    // X = lagged differences (if lags_vecm > 1)

    let n_obs = n_eff - 1;
    let mut y0 = Array2::zeros((n_obs, k)); // delta_Y_t
    let mut y1 = Array2::zeros((n_obs, k)); // Y_{t-1}

    for i in 0..n_obs {
        let t_idx = i + lags_vecm;
        for j in 0..k {
            y0[[i, j]] = delta_y[[t_idx - 1, j]]; // delta_Y at time t_idx
            y1[[i, j]] = data[[t_idx - 1, j]]; // Y at t_idx - 1
        }
    }

    // Build matrix of lagged differences and deterministic terms
    let n_det = match trend {
        TrendSpec::None => 0,
        TrendSpec::Constant => 1,
        TrendSpec::ConstantTrend => 2,
    };
    let n_lagged_diffs = (lags_vecm - 1) * k;
    let n_x_cols = n_lagged_diffs + n_det;

    // Concentrate out lagged differences and deterministic terms
    let (r0, r1) = if n_x_cols > 0 {
        let mut x_mat = Array2::zeros((n_obs, n_x_cols));
        let mut col = 0;

        // Deterministic terms
        if trend == TrendSpec::Constant || trend == TrendSpec::ConstantTrend {
            for i in 0..n_obs {
                x_mat[[i, col]] = 1.0;
            }
            col += 1;
        }
        if trend == TrendSpec::ConstantTrend {
            for i in 0..n_obs {
                x_mat[[i, col]] = (i + 1) as f64;
            }
            col += 1;
        }

        // Lagged differences
        for lag in 1..lags_vecm {
            for j in 0..k {
                for i in 0..n_obs {
                    let t_idx = i + lags_vecm;
                    if t_idx > lag {
                        x_mat[[i, col]] = delta_y[[t_idx - 1 - lag, j]];
                    }
                }
                col += 1;
            }
        }

        // Regress Y0 and Y1 on X to get residuals
        let (_b0, res0) = ols_regression_multi(&y0.view(), &x_mat.view())?;
        let (_b1, res1) = ols_regression_multi(&y1.view(), &x_mat.view())?;
        (res0, res1)
    } else {
        // No regressors to concentrate out
        (y0.clone(), y1.clone())
    };

    // Compute moment matrices
    let n_f = n_obs as f64;
    let s00 = r0.t().dot(&r0) / n_f;
    let s01 = r0.t().dot(&r1) / n_f;
    let s10 = r1.t().dot(&r0) / n_f;
    let s11 = r1.t().dot(&r1) / n_f;

    // Solve the generalized eigenvalue problem:
    // |lambda * S11 - S10 * S00^{-1} * S01| = 0
    let eye_k = Array2::<f64>::eye(k);

    // Regularize S00 for numerical stability
    let s00_reg = &s00 + &(&eye_k * 1e-10);
    let s00_inv =
        scirs2_linalg::solve_multiple(&s00_reg.view(), &eye_k.view(), None).map_err(|_| {
            NumRs2Error::ComputationError("Cannot invert S00 in Johansen procedure".to_string())
        })?;

    // Regularize S11
    let s11_reg = &s11 + &(&eye_k * 1e-10);
    let s11_inv =
        scirs2_linalg::solve_multiple(&s11_reg.view(), &eye_k.view(), None).map_err(|_| {
            NumRs2Error::ComputationError("Cannot invert S11 in Johansen procedure".to_string())
        })?;

    // Form the matrix for eigenvalue problem: S11^{-1} * S10 * S00^{-1} * S01
    let m = s11_inv.dot(&s10).dot(&s00_inv).dot(&s01);

    // Compute eigenvalues and eigenvectors
    let (eigenvalues_complex, eigenvectors_complex) =
        scirs2_linalg::eig(&m.view(), None).map_err(|_| {
            NumRs2Error::ComputationError(
                "Eigenvalue decomposition failed in Johansen procedure".to_string(),
            )
        })?;

    // Extract real parts of eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, usize)> = eigenvalues_complex
        .iter()
        .enumerate()
        .map(|(i, c)| (c.re.clamp(0.0, 1.0), i)) // Clamp to [0, 1]
        .collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut eigenvalues = Array1::zeros(k);
    let mut cointegrating_vectors = Array2::zeros((k, k));

    for (new_idx, &(eval, old_idx)) in eig_pairs.iter().enumerate() {
        eigenvalues[new_idx] = eval;
        for row in 0..k {
            cointegrating_vectors[[row, new_idx]] = eigenvectors_complex[[row, old_idx]].re;
        }
    }

    // Compute trace and max eigenvalue statistics
    let mut trace_stats = Array1::zeros(k);
    let mut max_eig_stats = Array1::zeros(k);

    for r in 0..k {
        // Trace statistic: -T * sum_{i=r+1}^{k} ln(1 - lambda_i)
        let mut trace_sum = 0.0;
        for i in r..k {
            let lambda_i = eigenvalues[i].clamp(1e-15, 1.0 - 1e-15);
            trace_sum += (1.0 - lambda_i).ln();
        }
        trace_stats[r] = -(n_obs as f64) * trace_sum;

        // Max eigenvalue statistic: -T * ln(1 - lambda_{r+1})
        if r < k {
            let lambda_r = eigenvalues[r].clamp(1e-15, 1.0 - 1e-15);
            max_eig_stats[r] = -(n_obs as f64) * (1.0 - lambda_r).ln();
        }
    }

    // Get critical values
    let trace_cv = johansen_trace_critical_values(k, trend);
    let max_eig_cv = johansen_max_eig_critical_values(k, trend);

    // Determine rank: smallest r for which we fail to reject H0: rank <= r
    let mut rank = 0;
    for r in 0..k {
        if trace_stats[r] > trace_cv[[r, 1]] {
            // Reject H0: rank <= r at 5%
            rank = r + 1;
        } else {
            break;
        }
    }

    // Compute adjustment coefficients (alpha)
    // alpha = S01 * beta * (beta' * S11 * beta)^{-1}
    let beta = cointegrating_vectors.slice(s![.., ..k]).to_owned();
    let beta_s11_beta = beta.t().dot(&s11).dot(&beta);

    let eye_full = Array2::<f64>::eye(k);
    let beta_s11_beta_inv =
        scirs2_linalg::solve_multiple(&beta_s11_beta.view(), &eye_full.view(), None)
            .unwrap_or_else(|_| eye_full.clone());

    let alpha = s01.dot(&beta).dot(&beta_s11_beta_inv);

    Ok(JohansenResult {
        trace_statistics: trace_stats,
        max_eigenvalue_statistics: max_eig_stats,
        eigenvalues,
        cointegrating_vectors: beta,
        adjustment_coefficients: alpha,
        trace_critical_values: trace_cv,
        max_eig_critical_values: max_eig_cv,
        rank,
    })
}

// =============================================================================
// Phillips-Ouliaris Test
// =============================================================================

/// Long-run variance estimator using the Bartlett (Newey-West) kernel.
///
/// Estimates the long-run variance of a series accounting for serial correlation.
fn long_run_variance(residuals: &ArrayView1<f64>, bandwidth: usize) -> f64 {
    let n = residuals.len();
    let mean = residuals.iter().sum::<f64>() / n as f64;

    // Autocovariance at lag 0
    let gamma0: f64 = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;

    let mut lrv = gamma0;

    for lag in 1..=bandwidth.min(n - 1) {
        let weight = 1.0 - lag as f64 / (bandwidth + 1) as f64; // Bartlett kernel
        let mut gamma_lag = 0.0;
        for i in lag..n {
            gamma_lag += (residuals[i] - mean) * (residuals[i - lag] - mean);
        }
        gamma_lag /= n as f64;
        lrv += 2.0 * weight * gamma_lag;
    }

    lrv.max(1e-15) // Ensure positive
}

/// Perform the Phillips-Ouliaris cointegration test.
///
/// A residual-based cointegration test that is robust to serial correlation
/// through non-parametric corrections. Provides both Z_alpha and Z_tau
/// statistics.
///
/// # Arguments
///
/// * `data` - Multivariate time series as (T x k) matrix
/// * `trend` - Trend specification for the cointegrating regression
/// * `bandwidth` - Bandwidth for long-run variance estimator. If None,
///   uses the Schwert (1989) rule: floor(4 * (T/100)^{1/4}).
///
/// # Returns
///
/// [`PhillipsOuliarisResult`] with Z_alpha and Z_tau statistics.
///
/// # References
///
/// Phillips, P.C.B. and Ouliaris, S. (1990). "Asymptotic Properties of Residual
/// Based Tests for Cointegration."
pub fn phillips_ouliaris_test(
    data: &ArrayView2<f64>,
    trend: TrendSpec,
    bandwidth: Option<usize>,
) -> Result<PhillipsOuliarisResult> {
    let (t, k) = data.dim();

    if k < 2 {
        return Err(NumRs2Error::ValueError(
            "Phillips-Ouliaris test requires at least 2 variables".to_string(),
        ));
    }

    if t < 20 {
        return Err(NumRs2Error::ValueError(
            "Phillips-Ouliaris test requires at least 20 observations".to_string(),
        ));
    }

    // Step 1: OLS regression of y1 on y2,...,yk (with deterministic terms)
    let y = data.column(0);

    let n_det = match trend {
        TrendSpec::None => 0,
        TrendSpec::Constant => 1,
        TrendSpec::ConstantTrend => 2,
    };
    let n_regressors = k - 1 + n_det;
    let mut x_mat = Array2::zeros((t, n_regressors));
    let mut col = 0;

    if trend == TrendSpec::Constant || trend == TrendSpec::ConstantTrend {
        for i in 0..t {
            x_mat[[i, col]] = 1.0;
        }
        col += 1;
    }
    if trend == TrendSpec::ConstantTrend {
        for i in 0..t {
            x_mat[[i, col]] = (i + 1) as f64;
        }
        col += 1;
    }
    for j in 1..k {
        for i in 0..t {
            x_mat[[i, col]] = data[[i, j]];
        }
        col += 1;
    }

    let (_coefficients, residuals, _se) = ols_regression(&y, &x_mat.view())?;

    // Step 2: Compute Phillips-Ouliaris statistics

    // Bandwidth selection (Schwert rule)
    let bw = bandwidth.unwrap_or_else(|| (4.0 * (t as f64 / 100.0).powf(0.25)).floor() as usize);
    let bw = bw.max(1).min(t / 2);

    // Autoregressive coefficient estimate
    // e_t = rho * e_{t-1} + u_t
    let n_resid = residuals.len();
    let mut sum_et_et1 = 0.0;
    let mut sum_et1_sq = 0.0;
    for i in 1..n_resid {
        sum_et_et1 += residuals[i] * residuals[i - 1];
        sum_et1_sq += residuals[i - 1] * residuals[i - 1];
    }

    let rho_hat = if sum_et1_sq.abs() > 1e-15 {
        sum_et_et1 / sum_et1_sq
    } else {
        0.0
    };

    // Residuals from AR(1) on e_t
    let mut u_hat = Array1::zeros(n_resid - 1);
    for i in 1..n_resid {
        u_hat[i - 1] = residuals[i] - rho_hat * residuals[i - 1];
    }

    // Sample variance of innovations
    let sigma_u_sq: f64 = u_hat.iter().map(|&u| u * u).sum::<f64>() / (n_resid - 1) as f64;

    // Long-run variance of u_t
    let omega_sq = long_run_variance(&u_hat.view(), bw);

    // One-sided long-run variance
    let lambda = (omega_sq - sigma_u_sq) / 2.0;

    // Z_alpha statistic
    let t_f = t as f64;
    let n_alpha = t_f * (rho_hat - 1.0) - lambda * t_f / sum_et1_sq;
    let z_alpha = n_alpha;

    // Z_tau statistic
    let se_rho = if sum_et1_sq > 1e-15 {
        (sigma_u_sq / sum_et1_sq).sqrt()
    } else {
        1.0
    };
    let t_stat = (rho_hat - 1.0) / se_rho;
    let correction = lambda / (omega_sq.sqrt() * se_rho * sum_et1_sq.sqrt());
    let z_tau = t_stat - correction;

    // Critical values for Phillips-Ouliaris tests
    // Based on Phillips and Ouliaris (1990) Table IIa/IIb
    let z_alpha_cv = match k {
        2 => [-28.32, -20.49, -17.04],
        3 => [-37.08, -28.30, -24.38],
        4 => [-46.37, -36.74, -32.10],
        _ => {
            let scale = k as f64;
            [-10.0 * scale, -8.0 * scale, -6.5 * scale]
        }
    };

    let z_tau_cv = match k {
        2 => [-3.90, -3.34, -3.05],
        3 => [-4.30, -3.74, -3.45],
        4 => [-4.65, -4.10, -3.81],
        _ => {
            let adj = 0.35 * (k as f64 - 2.0);
            [-3.90 - adj, -3.34 - adj, -3.05 - adj]
        }
    };

    let cointegrated = z_tau < z_tau_cv[1]; // Reject at 5%

    Ok(PhillipsOuliarisResult {
        z_alpha,
        z_tau,
        z_alpha_critical_values: z_alpha_cv,
        z_tau_critical_values: z_tau_cv,
        cointegrated,
        residuals,
    })
}

// =============================================================================
// Error Correction Model (VECM)
// =============================================================================

/// Estimate a Vector Error Correction Model (VECM).
///
/// Given cointegrating vectors from Johansen (or specified externally),
/// estimates the VECM:
///
///   delta_Y_t = alpha * beta' * Y_{t-1} + sum_i Gamma_i * delta_Y_{t-i} + c + epsilon_t
///
/// where:
/// - alpha: speed of adjustment coefficients (k x r)
/// - beta: cointegrating vectors (k x r)
/// - Gamma_i: short-run dynamics matrices
/// - c: intercept
///
/// # Arguments
///
/// * `data` - Multivariate time series as (T x k) matrix
/// * `lags` - Number of lags in the differenced form
/// * `rank` - Number of cointegrating relationships
/// * `beta` - Optional pre-specified cointegrating vectors (k x r). If None,
///   vectors are estimated using Johansen procedure.
///
/// # Returns
///
/// [`VecmResult`] with estimated parameters.
pub fn estimate_vecm(
    data: &ArrayView2<f64>,
    lags: usize,
    rank: usize,
    beta: Option<&ArrayView2<f64>>,
) -> Result<VecmResult> {
    let (t, k) = data.dim();

    if rank == 0 || rank >= k {
        return Err(NumRs2Error::ValueError(format!(
            "Cointegration rank must be between 1 and {} (exclusive)",
            k
        )));
    }

    if t < 2 * k + lags + 5 {
        return Err(NumRs2Error::ValueError(
            "Insufficient observations for VECM estimation".to_string(),
        ));
    }

    // Get cointegrating vectors
    let beta_matrix = if let Some(b) = beta {
        let (b_rows, b_cols) = b.dim();
        if b_rows != k || b_cols != rank {
            return Err(NumRs2Error::ValueError(format!(
                "Beta must be ({} x {}), got ({} x {})",
                k, rank, b_rows, b_cols
            )));
        }
        b.to_owned()
    } else {
        // Estimate using Johansen
        let joh_result = johansen_test(data, lags.max(1), TrendSpec::Constant)?;
        joh_result
            .cointegrating_vectors
            .slice(s![.., ..rank])
            .to_owned()
    };

    // Compute first differences
    let mut delta_y = Array2::zeros((t - 1, k));
    for i in 0..(t - 1) {
        for j in 0..k {
            delta_y[[i, j]] = data[[i + 1, j]] - data[[i, j]];
        }
    }

    // Error correction terms: z_{t-1} = beta' * Y_{t-1}
    let lags_eff = lags.max(1);
    let n_obs = t - lags_eff - 1;

    if n_obs <= rank + k * (lags_eff - 1) + 1 {
        return Err(NumRs2Error::ValueError(
            "Effective sample size too small for VECM estimation".to_string(),
        ));
    }

    // Dependent variable: delta_Y_t
    let mut y_dep = Array2::zeros((n_obs, k));
    for i in 0..n_obs {
        let t_idx = i + lags_eff;
        for j in 0..k {
            y_dep[[i, j]] = delta_y[[t_idx, j]];
        }
    }

    // Build regressor matrix: [z_{t-1}, delta_Y_{t-1}, ..., delta_Y_{t-p+1}, 1]
    let n_gamma_lags = lags_eff.saturating_sub(1);
    let n_regressors = rank + k * n_gamma_lags + 1; // +1 for intercept
    let mut x_mat = Array2::zeros((n_obs, n_regressors));

    for i in 0..n_obs {
        let t_idx = i + lags_eff;
        let mut col = 0;

        // Error correction terms: beta' * Y_{t-1}
        for r in 0..rank {
            let mut z = 0.0;
            for j in 0..k {
                z += beta_matrix[[j, r]] * data[[t_idx, j]];
            }
            x_mat[[i, col]] = z;
            col += 1;
        }

        // Lagged differences
        for lag in 1..=n_gamma_lags {
            for j in 0..k {
                if t_idx >= lag {
                    x_mat[[i, col]] = delta_y[[t_idx - lag, j]];
                }
                col += 1;
            }
        }

        // Intercept
        x_mat[[i, col]] = 1.0;
    }

    // Estimate by OLS
    let (b_hat, residuals) = ols_regression_multi(&y_dep.view(), &x_mat.view())?;

    // Extract parameters
    // Alpha: first 'rank' rows of B_hat transposed appropriately
    let mut alpha = Array2::zeros((k, rank));
    for j in 0..k {
        for r in 0..rank {
            alpha[[j, r]] = b_hat[[r, j]];
        }
    }

    // Gamma matrices
    let mut gamma_matrices = Vec::with_capacity(n_gamma_lags);
    for lag in 0..n_gamma_lags {
        let mut gamma_i = Array2::zeros((k, k));
        let start_col = rank + lag * k;
        for j in 0..k {
            for m in 0..k {
                gamma_i[[j, m]] = b_hat[[start_col + m, j]];
            }
        }
        gamma_matrices.push(gamma_i);
    }

    // Intercept
    let intercept_col = rank + n_gamma_lags * k;
    let mut intercept = Array1::zeros(k);
    for j in 0..k {
        intercept[j] = b_hat[[intercept_col, j]];
    }

    // Residual covariance
    let sigma = residuals.t().dot(&residuals) / n_obs as f64;

    // Log-likelihood (multivariate normal)
    let log_likelihood = compute_mvn_log_likelihood(&residuals.view(), &sigma.view());

    Ok(VecmResult {
        alpha,
        beta: beta_matrix,
        gamma: gamma_matrices,
        intercept,
        residuals,
        sigma,
        log_likelihood,
        rank,
    })
}

/// Compute multivariate normal log-likelihood.
fn compute_mvn_log_likelihood(residuals: &ArrayView2<f64>, sigma: &ArrayView2<f64>) -> f64 {
    let (n, k) = residuals.dim();

    let det_sigma = scirs2_linalg::det(&sigma.view(), None)
        .unwrap_or(1e-10)
        .max(1e-10);

    let log_det = det_sigma.ln();

    let eye_k = Array2::<f64>::eye(k);
    let sigma_inv = scirs2_linalg::solve_multiple(&sigma.view(), &eye_k.view(), None)
        .unwrap_or_else(|_| {
            let mut diag = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                diag[[i, i]] = 1.0 / sigma[[i, i]].max(1e-10);
            }
            diag
        });

    let mut quad_form = 0.0;
    for i in 0..n {
        let r = residuals.row(i);
        let temp = sigma_inv.dot(&r);
        for j in 0..k {
            quad_form += r[j] * temp[j];
        }
    }

    -0.5 * (n as f64 * (k as f64 * (2.0 * std::f64::consts::PI).ln() + log_det) + quad_form)
}

// =============================================================================
// Reduced Rank Regression (Utility)
// =============================================================================

/// Perform reduced rank regression.
///
/// Estimates the matrix Pi = alpha * beta' of rank r in the model:
///   Y = X * Gamma + Z * Pi + epsilon
///
/// by concentrating out X and solving a generalized eigenvalue problem.
///
/// # Arguments
///
/// * `y` - Dependent variable matrix (n x p)
/// * `z` - Variables for reduced rank part (n x q)
/// * `x` - Variables to concentrate out (n x m), or None
/// * `rank` - Desired rank of Pi
///
/// # Returns
///
/// (alpha, beta) such that Pi = alpha * beta'
pub fn reduced_rank_regression(
    y: &ArrayView2<f64>,
    z: &ArrayView2<f64>,
    x: Option<&ArrayView2<f64>>,
    rank: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let (n, p) = y.dim();
    let (_n_z, q) = z.dim();

    if rank > p.min(q) {
        return Err(NumRs2Error::ValueError(format!(
            "Rank {} exceeds min(p={}, q={})",
            rank, p, q
        )));
    }

    // Concentrate out X if provided
    let (r0, r1) = if let Some(x_mat) = x {
        let (_b0, res0) = ols_regression_multi(y, x_mat)?;
        let (_b1, res1) = ols_regression_multi(z, x_mat)?;
        (res0, res1)
    } else {
        (y.to_owned(), z.to_owned())
    };

    let n_f = n as f64;
    let s00 = r0.t().dot(&r0) / n_f;
    let s01 = r0.t().dot(&r1) / n_f;
    let s11 = r1.t().dot(&r1) / n_f;

    // Solve: S11^{-1} * S10 * S00^{-1} * S01
    let eye_p = Array2::<f64>::eye(p);
    let eye_q = Array2::<f64>::eye(q);

    let s00_reg = &s00 + &(&eye_p * 1e-10);
    let s11_reg = &s11 + &(&eye_q * 1e-10);

    let s00_inv =
        scirs2_linalg::solve_multiple(&s00_reg.view(), &eye_p.view(), None).map_err(|_| {
            NumRs2Error::ComputationError(
                "Cannot invert S00 in reduced rank regression".to_string(),
            )
        })?;

    let s11_inv =
        scirs2_linalg::solve_multiple(&s11_reg.view(), &eye_q.view(), None).map_err(|_| {
            NumRs2Error::ComputationError(
                "Cannot invert S11 in reduced rank regression".to_string(),
            )
        })?;

    let mat = s11_inv.dot(&s01.t()).dot(&s00_inv).dot(&s01);

    let (_eigenvalues_c, eigenvectors_c) = scirs2_linalg::eig(&mat.view(), None).map_err(|_| {
        NumRs2Error::ComputationError("Eigenvalue decomposition failed in RRR".to_string())
    })?;

    // Extract first `rank` eigenvectors as beta
    let mut beta = Array2::zeros((q, rank));
    for j in 0..rank {
        for i in 0..q {
            beta[[i, j]] = eigenvectors_c[[i, j]].re;
        }
    }

    // alpha = S01 * beta * (beta' * S11 * beta)^{-1}
    let bsb = beta.t().dot(&s11).dot(&beta);
    let eye_r = Array2::<f64>::eye(rank);
    let bsb_inv = scirs2_linalg::solve_multiple(&bsb.view(), &eye_r.view(), None).unwrap_or(eye_r);

    let alpha = s01.dot(&beta).dot(&bsb_inv);

    Ok((alpha, beta))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_core::random::{Rng, SeedableRng, StdRng};

    /// Generate a cointegrated pair: y1 = beta * y2 + stationary_error
    fn generate_cointegrated_pair(n: usize, beta: f64, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array2::zeros((n, 2));

        // Generate a random walk for y2
        let mut y2 = 0.0;
        // Generate stationary noise for the cointegrating residual
        let mut noise = 0.0;

        for i in 0..n {
            let eps1: f64 = rng.gen_range(-0.5..0.5);
            let eps2: f64 = rng.gen_range(-0.5..0.5);

            y2 += eps1;
            // Mean-reverting noise (AR(1) with phi < 1)
            noise = 0.3 * noise + eps2;

            data[[i, 0]] = beta * y2 + noise;
            data[[i, 1]] = y2;
        }

        data
    }

    /// Generate two independent random walks (no cointegration).
    fn generate_independent_walks(n: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array2::zeros((n, 2));

        let mut y1 = 0.0;
        let mut y2 = 0.0;

        for i in 0..n {
            let eps1: f64 = rng.gen_range(-0.5..0.5);
            let eps2: f64 = rng.gen_range(-0.5..0.5);

            y1 += eps1;
            y2 += eps2;

            data[[i, 0]] = y1;
            data[[i, 1]] = y2;
        }

        data
    }

    /// Generate multivariate cointegrated system with specified rank.
    fn generate_cointegrated_system(n: usize, k: usize, rank: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array2::zeros((n, k));

        // Generate k-rank independent random walks as common trends
        let n_trends = k - rank;
        let mut trends = Array2::zeros((n, n_trends.max(1)));

        for j in 0..n_trends.max(1) {
            let mut val = 0.0;
            for i in 0..n {
                val += rng.gen_range(-0.5..0.5);
                trends[[i, j]] = val;
            }
        }

        // First n_trends variables are driven by trends with stationary noise
        for j in 0..k {
            let mut noise = 0.0;
            for i in 0..n {
                let eps: f64 = rng.gen_range(-0.3..0.3);
                noise = 0.2 * noise + eps;

                // Each variable is a mix of trends + stationary component
                let trend_idx = j % n_trends.max(1);
                let weight = 1.0 + 0.5 * (j as f64);
                data[[i, j]] = weight * trends[[i, trend_idx]] + noise;
            }
        }

        data
    }

    // =========================================================================
    // Test 1: Engle-Granger detects known cointegrated pair
    // =========================================================================
    #[test]
    fn test_engle_granger_cointegrated() {
        let data = generate_cointegrated_pair(300, 2.0, 42);
        let result = engle_granger_test(&data.view(), 2, TrendSpec::Constant);
        assert!(result.is_ok(), "Engle-Granger test should succeed");

        let eg = result.expect("should not fail");
        // The ADF statistic should be significantly negative
        assert!(
            eg.adf_statistic < -1.5,
            "ADF stat {} should be quite negative for cointegrated pair",
            eg.adf_statistic
        );
        assert!(eg.residuals.len() == 300);
    }

    // =========================================================================
    // Test 2: Engle-Granger does NOT reject for independent random walks
    // =========================================================================
    #[test]
    fn test_engle_granger_independent() {
        let data = generate_independent_walks(300, 123);
        let result = engle_granger_test(&data.view(), 2, TrendSpec::Constant);
        assert!(result.is_ok());

        let eg = result.expect("should not fail");
        // For independent walks, the ADF statistic on residuals should
        // generally NOT be very negative (fail to reject null of no cointegration)
        // We use a relaxed check since this is stochastic
        assert!(
            eg.adf_statistic > -5.0,
            "ADF stat {} should not be extremely negative for independent walks",
            eg.adf_statistic
        );
    }

    // =========================================================================
    // Test 3: Johansen test detects cointegration
    // =========================================================================
    #[test]
    fn test_johansen_cointegrated() {
        let data = generate_cointegrated_pair(300, 1.5, 99);
        let result = johansen_test(&data.view(), 2, TrendSpec::Constant);
        assert!(result.is_ok(), "Johansen test should succeed");

        let joh = result.expect("should not fail");
        assert_eq!(joh.trace_statistics.len(), 2);
        assert_eq!(joh.max_eigenvalue_statistics.len(), 2);
        assert_eq!(joh.eigenvalues.len(), 2);
        assert_eq!(joh.cointegrating_vectors.dim(), (2, 2));

        // The first eigenvalue should be non-trivially positive
        assert!(
            joh.eigenvalues[0] > 0.0,
            "First eigenvalue {} should be positive",
            joh.eigenvalues[0]
        );
    }

    // =========================================================================
    // Test 4: Johansen test with independent walks (rank should be 0)
    // =========================================================================
    #[test]
    fn test_johansen_independent() {
        let data = generate_independent_walks(300, 456);
        let result = johansen_test(&data.view(), 2, TrendSpec::Constant);
        assert!(result.is_ok());

        let joh = result.expect("should not fail");
        // Eigenvalues should all be small for independent walks
        // (though not necessarily zero due to finite sample)
        assert!(
            joh.eigenvalues[0] < 0.95,
            "First eigenvalue {} should not be too close to 1 for independent walks",
            joh.eigenvalues[0]
        );
    }

    // =========================================================================
    // Test 5: Johansen with multiple cointegrating relationships
    // =========================================================================
    #[test]
    fn test_johansen_multiple_relationships() {
        // 3-variable system with 1 cointegrating relationship
        let data = generate_cointegrated_system(300, 3, 1, 77);
        let result = johansen_test(&data.view(), 2, TrendSpec::Constant);
        assert!(result.is_ok());

        let joh = result.expect("should not fail");
        assert_eq!(joh.eigenvalues.len(), 3);
        assert_eq!(joh.trace_statistics.len(), 3);
        assert_eq!(joh.trace_critical_values.dim(), (3, 3));
    }

    // =========================================================================
    // Test 6: Phillips-Ouliaris test on cointegrated pair
    // =========================================================================
    #[test]
    fn test_phillips_ouliaris_cointegrated() {
        let data = generate_cointegrated_pair(300, 2.0, 55);
        let result = phillips_ouliaris_test(&data.view(), TrendSpec::Constant, None);
        assert!(result.is_ok());

        let po = result.expect("should not fail");
        // Z_tau and Z_alpha should be finite
        assert!(po.z_tau.is_finite(), "Z_tau should be finite");
        assert!(po.z_alpha.is_finite(), "Z_alpha should be finite");
        assert_eq!(po.residuals.len(), 300);
    }

    // =========================================================================
    // Test 7: Phillips-Ouliaris test on independent walks
    // =========================================================================
    #[test]
    fn test_phillips_ouliaris_independent() {
        let data = generate_independent_walks(300, 789);
        let result = phillips_ouliaris_test(&data.view(), TrendSpec::Constant, None);
        assert!(result.is_ok());

        let po = result.expect("should not fail");
        assert!(po.z_tau.is_finite());
        assert!(po.z_alpha.is_finite());
    }

    // =========================================================================
    // Test 8: VECM estimation
    // =========================================================================
    #[test]
    fn test_vecm_estimation() {
        let data = generate_cointegrated_pair(300, 2.0, 101);
        let result = estimate_vecm(&data.view(), 2, 1, None);
        assert!(
            result.is_ok(),
            "VECM estimation should succeed: {:?}",
            result.err()
        );

        let vecm = result.expect("should not fail");
        assert_eq!(vecm.alpha.dim(), (2, 1));
        assert_eq!(vecm.beta.dim(), (2, 1));
        assert_eq!(vecm.rank, 1);
        assert!(vecm.log_likelihood.is_finite());
    }

    // =========================================================================
    // Test 9: Engle-Granger vs Johansen consistency
    // =========================================================================
    #[test]
    fn test_eg_johansen_consistency() {
        let data = generate_cointegrated_pair(500, 1.0, 200);

        let eg =
            engle_granger_test(&data.view(), 2, TrendSpec::Constant).expect("EG should succeed");
        let joh =
            johansen_test(&data.view(), 2, TrendSpec::Constant).expect("Johansen should succeed");

        // Both should agree on the existence (or not) of cointegration
        // If EG finds strong evidence (very negative ADF), Johansen should
        // have a large first eigenvalue
        if eg.adf_statistic < eg.critical_values[1] {
            // EG rejects no-cointegration
            assert!(
                joh.eigenvalues[0] > 0.01,
                "Johansen should also find evidence if EG does",
            );
        }
    }

    // =========================================================================
    // Test 10: Edge case - identical series
    // =========================================================================
    #[test]
    fn test_identical_series() {
        // Two identical random walks => trivially cointegrated with beta = 1
        let mut data = Array2::zeros((100, 2));
        let mut rng = StdRng::seed_from_u64(999);
        let mut val = 0.0;
        for i in 0..100 {
            val += rng.gen_range(-0.5..0.5);
            data[[i, 0]] = val;
            data[[i, 1]] = val;
        }

        let result = engle_granger_test(&data.view(), 1, TrendSpec::Constant);
        // This should work but the residuals are essentially zero
        // The test might error out due to near-zero variance
        // Either way is acceptable behavior
        match result {
            Ok(eg) => {
                // If it succeeds, residuals should be very small
                let max_resid = eg.residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
                assert!(
                    max_resid < 1e-6,
                    "Residuals for identical series should be near zero, got {}",
                    max_resid
                );
            }
            Err(_) => {
                // Acceptable: numerical issues with identical series
            }
        }
    }

    // =========================================================================
    // Test 11: Edge case - short series
    // =========================================================================
    #[test]
    fn test_short_series() {
        let data = Array2::zeros((5, 2));

        let eg_result = engle_granger_test(&data.view(), 1, TrendSpec::Constant);
        assert!(eg_result.is_err(), "Short series should fail for EG test");

        let joh_result = johansen_test(&data.view(), 1, TrendSpec::Constant);
        assert!(
            joh_result.is_err(),
            "Short series should fail for Johansen test"
        );
    }

    // =========================================================================
    // Test 12: Reduced rank regression
    // =========================================================================
    #[test]
    fn test_reduced_rank_regression() {
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);

        // Create Y (n x 2) and Z (n x 2) with rank-1 relationship
        let mut y_mat = Array2::zeros((n, 2));
        let mut z_mat = Array2::zeros((n, 2));

        for i in 0..n {
            let z1: f64 = rng.gen_range(-1.0..1.0);
            let z2: f64 = rng.gen_range(-1.0..1.0);
            z_mat[[i, 0]] = z1;
            z_mat[[i, 1]] = z2;

            // Y depends on Z through rank-1 matrix
            let noise1: f64 = rng.gen_range(-0.1..0.1);
            let noise2: f64 = rng.gen_range(-0.1..0.1);
            y_mat[[i, 0]] = 2.0 * z1 + 1.0 * z2 + noise1;
            y_mat[[i, 1]] = 4.0 * z1 + 2.0 * z2 + noise2;
        }

        let result = reduced_rank_regression(&y_mat.view(), &z_mat.view(), None, 1);
        assert!(result.is_ok(), "RRR should succeed");

        let (alpha, beta) = result.expect("should not fail");
        assert_eq!(alpha.dim().1, 1);
        assert_eq!(beta.dim().1, 1);
    }

    // =========================================================================
    // Test 13: VECM speed of adjustment
    // =========================================================================
    #[test]
    fn test_vecm_speed_of_adjustment() {
        let data = generate_cointegrated_pair(500, 2.0, 303);
        let result = estimate_vecm(&data.view(), 2, 1, None);

        if let Ok(vecm) = result {
            // At least one alpha coefficient should be non-zero
            let alpha_norm: f64 = vecm.alpha.iter().map(|&a| a * a).sum::<f64>();
            assert!(alpha_norm > 1e-15, "Speed of adjustment should be non-zero");

            // Check residuals exist
            let (n_resid, k_resid) = vecm.residuals.dim();
            assert!(n_resid > 0);
            assert_eq!(k_resid, 2);
        }
    }

    // =========================================================================
    // Test 14: Engle-Granger with different trend specifications
    // =========================================================================
    #[test]
    fn test_engle_granger_trend_specs() {
        let data = generate_cointegrated_pair(200, 2.0, 500);

        // Test all trend specifications
        for trend in &[
            TrendSpec::None,
            TrendSpec::Constant,
            TrendSpec::ConstantTrend,
        ] {
            let result = engle_granger_test(&data.view(), 1, *trend);
            assert!(
                result.is_ok(),
                "EG test should succeed with trend {:?}",
                trend
            );

            let eg = result.expect("should not fail");
            assert!(eg.adf_statistic.is_finite());
        }
    }

    // =========================================================================
    // Test 15: Johansen trace vs max eigenvalue consistency
    // =========================================================================
    #[test]
    fn test_johansen_trace_max_eig_consistency() {
        let data = generate_cointegrated_pair(300, 1.5, 707);
        let result = johansen_test(&data.view(), 2, TrendSpec::Constant);
        assert!(result.is_ok());

        let joh = result.expect("should not fail");

        // Trace statistic at rank 0 should be >= max eigenvalue statistic at rank 0
        // because trace = sum of max_eig for all subsequent ranks
        assert!(
            joh.trace_statistics[0] >= joh.max_eigenvalue_statistics[0] - 1e-10,
            "Trace stat ({}) should be >= max eig stat ({})",
            joh.trace_statistics[0],
            joh.max_eigenvalue_statistics[0]
        );
    }

    // =========================================================================
    // Test 16: OLS helper correctness
    // =========================================================================
    #[test]
    fn test_ols_regression_basic() {
        // y = 2 + 3*x + noise
        let n = 50;
        let mut x_mat = Array2::zeros((n, 2));
        let mut y_vec = Array1::zeros(n);

        for i in 0..n {
            let x_val = i as f64 / 10.0;
            x_mat[[i, 0]] = 1.0;
            x_mat[[i, 1]] = x_val;
            y_vec[i] = 2.0 + 3.0 * x_val;
        }

        let (beta, residuals, _se) =
            ols_regression(&y_vec.view(), &x_mat.view()).expect("OLS should succeed");

        assert!(
            (beta[0] - 2.0).abs() < 1e-10,
            "Intercept should be 2.0, got {}",
            beta[0]
        );
        assert!(
            (beta[1] - 3.0).abs() < 1e-10,
            "Slope should be 3.0, got {}",
            beta[1]
        );

        let max_resid = residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_resid < 1e-10,
            "Residuals should be near zero for perfect fit"
        );
    }
}
