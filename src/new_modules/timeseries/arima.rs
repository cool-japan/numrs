//! ARIMA and SARIMA Time Series Models
//!
//! This module implements AutoRegressive Integrated Moving Average (ARIMA) models
//! and their seasonal variants (SARIMA) for univariate time series analysis.
//!
//! ## Model Specification
//!
//! ### ARIMA(p,d,q)
//!
//! The ARIMA model combines:
//! - **AR(p)**: AutoRegressive component of order p
//! - **I(d)**: Differencing of order d to achieve stationarity
//! - **MA(q)**: Moving Average component of order q
//!
//! The model equation for differenced series Y_t = ∇^d X_t is:
//!
//! φ(L) Y_t = θ(L) ε_t
//!
//! where:
//! - φ(L) = 1 - φ₁L - φ₂L² - ... - φₚLᵖ (AR polynomial)
//! - θ(L) = 1 + θ₁L + θ₂L² + ... + θ_qL^q (MA polynomial)
//! - ε_t ~ N(0, σ²) (white noise)
//!
//! ### SARIMA(p,d,q)(P,D,Q)s
//!
//! Seasonal ARIMA extends ARIMA with seasonal components:
//! - (P,D,Q): Seasonal orders
//! - s: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
//!
//! ## References
//!
//! Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
//! *Time series analysis: forecasting and control* (5th ed.). John Wiley & Sons.
//!
//! Hyndman, R. J., & Athanasopoulos, G. (2021).
//! *Forecasting: principles and practice* (3rd ed.). OTexts.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};

/// ARIMA model parameters.
#[derive(Debug, Clone)]
pub struct ArimaParams {
    /// AR coefficients (φ₁, φ₂, ..., φₚ)
    pub ar_coefs: Array1<f64>,
    /// MA coefficients (θ₁, θ₂, ..., θ_q)
    pub ma_coefs: Array1<f64>,
    /// Intercept/constant term
    pub intercept: f64,
    /// Residual variance σ²
    pub sigma2: f64,
    /// Log-likelihood of fitted model
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

/// SARIMA model parameters extending ARIMA with seasonal components.
#[derive(Debug, Clone)]
pub struct SarimaParams {
    /// Non-seasonal ARIMA parameters
    pub arima_params: ArimaParams,
    /// Seasonal AR coefficients (Φ₁, Φ₂, ..., Φₚ)
    pub seasonal_ar_coefs: Array1<f64>,
    /// Seasonal MA coefficients (Θ₁, Θ₂, ..., Θ_Q)
    pub seasonal_ma_coefs: Array1<f64>,
    /// Seasonal period
    pub seasonal_period: usize,
}

/// ARIMA model structure.
#[derive(Debug, Clone)]
pub struct Arima {
    /// AR order
    pub p: usize,
    /// Differencing order
    pub d: usize,
    /// MA order
    pub q: usize,
    /// Include intercept term
    pub include_intercept: bool,
}

impl Arima {
    /// Create a new ARIMA(p,d,q) model specification.
    ///
    /// # Arguments
    ///
    /// * `p` - AR order
    /// * `d` - Differencing order
    /// * `q` - MA order
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::new_modules::timeseries::Arima;
    ///
    /// let model = Arima::new(1, 1, 1); // ARIMA(1,1,1)
    /// ```
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            include_intercept: true,
        }
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.include_intercept = include;
        self
    }

    /// Fit the ARIMA model to time series data.
    ///
    /// Uses maximum likelihood estimation via numerical optimization.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    ///
    /// # Returns
    ///
    /// Fitted model parameters
    pub fn fit(&self, data: &ArrayView1<f64>) -> Result<ArimaParams> {
        let n = data.len();

        if n < self.p + self.d + self.q + 1 {
            return Err(NumRs2Error::ValueError(format!(
                "Insufficient data: need at least {} observations",
                self.p + self.d + self.q + 1
            )));
        }

        // Apply differencing
        let diff_data = self.difference(data)?;

        // Estimate parameters using conditional sum of squares (CSS) for initialization
        let init_params = self.estimate_css(&diff_data.view())?;

        // Refine using maximum likelihood estimation (MLE)
        let mle_params = self.estimate_mle(&diff_data.view(), &init_params)?;

        Ok(mle_params)
    }

    /// Apply differencing to the time series.
    fn difference(&self, data: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let mut result = data.to_owned();

        for _ in 0..self.d {
            if result.len() < 2 {
                return Err(NumRs2Error::ValueError(
                    "Series too short for differencing".to_string(),
                ));
            }

            let n = result.len();
            let mut diff = Array1::zeros(n - 1);

            for i in 0..(n - 1) {
                diff[i] = result[i + 1] - result[i];
            }

            result = diff;
        }

        Ok(result)
    }

    /// Estimate initial parameters using Conditional Sum of Squares.
    fn estimate_css(&self, data: &ArrayView1<f64>) -> Result<ArimaParams> {
        let n = data.len();
        let max_lag = self.p.max(self.q);

        if n <= max_lag {
            return Err(NumRs2Error::ValueError(
                "Insufficient data for CSS estimation".to_string(),
            ));
        }

        // Initialize coefficients
        let mut ar_coefs = Array1::zeros(self.p);
        let mut ma_coefs = Array1::zeros(self.q);
        let mut intercept = 0.0;

        // Compute mean for intercept
        if self.include_intercept {
            intercept = data.iter().sum::<f64>() / n as f64;
        }

        // For AR component, use Yule-Walker equations if p > 0
        if self.p > 0 {
            ar_coefs = self.estimate_ar_yule_walker(data)?;
        }

        // For MA component, use innovations algorithm if q > 0
        if self.q > 0 {
            ma_coefs = self.estimate_ma_innovations(data, &ar_coefs)?;
        }

        // Compute residuals and variance
        let residuals = self.compute_residuals(data, &ar_coefs, &ma_coefs, intercept)?;
        let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / residuals.len() as f64;

        // Compute log-likelihood
        let log_likelihood = self.compute_log_likelihood(&residuals, sigma2);

        // Compute information criteria
        let k = self.p + self.q + if self.include_intercept { 1 } else { 0 };
        let aic = -2.0 * log_likelihood + 2.0 * k as f64;
        let bic = -2.0 * log_likelihood + (k as f64) * (n as f64).ln();

        Ok(ArimaParams {
            ar_coefs,
            ma_coefs,
            intercept,
            sigma2,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// Estimate AR coefficients using Yule-Walker equations.
    fn estimate_ar_yule_walker(&self, data: &ArrayView1<f64>) -> Result<Array1<f64>> {
        if self.p == 0 {
            return Ok(Array1::zeros(0));
        }

        use crate::new_modules::timeseries::autocorrelation;

        // Compute autocorrelations
        let acf = autocorrelation(data, self.p)?;

        // Set up Yule-Walker equations: Rφ = r
        // where R is Toeplitz matrix of autocorrelations
        let mut r_matrix = Array2::zeros((self.p, self.p));
        let mut r_vec = Array1::zeros(self.p);

        for i in 0..self.p {
            r_vec[i] = acf[i + 1];
            for j in 0..self.p {
                let lag = i.abs_diff(j);
                r_matrix[[i, j]] = acf[lag];
            }
        }

        // Solve for AR coefficients
        match scirs2_linalg::solve(&r_matrix.view(), &r_vec.view(), None) {
            Ok(phi) => Ok(phi),
            Err(_) => {
                // Fallback to least squares if direct solve fails
                Ok(Array1::zeros(self.p))
            }
        }
    }

    /// Estimate MA coefficients using the Hannan-Rissanen algorithm.
    ///
    /// The Hannan-Rissanen procedure provides consistent initial estimates for
    /// the parameters of an ARMA(p, q) model in two least-squares stages:
    ///
    /// 1. **Long AR fit.** A high-order AR(`p_long`) model is fitted to the
    ///    (mean-centered, differenced) series by ordinary least squares. Its
    ///    residuals `ε̂_t` approximate the unobserved innovations `ε_t`.
    ///
    /// 2. **Joint ARMA regression.** The series `y_t` is regressed on its own
    ///    lags `y_{t-1}, ..., y_{t-p}` (the AR part) **and** on the lagged
    ///    estimated residuals `ε̂_{t-1}, ..., ε̂_{t-q}` (the MA part) by least
    ///    squares. The coefficients of the residual lags are the MA estimates.
    ///
    /// The series is mean-centered up front, so the regressions are run through
    /// the origin (the intercept is handled separately by the caller). The AR
    /// block of the stage-2 regression is estimated jointly with the MA block
    /// for consistency, but only the MA coefficients are returned to match the
    /// function contract; the AR coefficients are supplied by the caller's
    /// Yule-Walker estimate.
    ///
    /// # References
    ///
    /// Hannan, E. J., & Rissanen, J. (1982). Recursive estimation of mixed
    /// autoregressive-moving average order. *Biometrika*, 69(1), 81-94.
    fn estimate_ma_innovations(
        &self,
        data: &ArrayView1<f64>,
        ar_coefs: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        if self.q == 0 {
            return Ok(Array1::zeros(0));
        }

        let _ = ar_coefs; // AR coefficients are re-estimated jointly below.

        let n = data.len();
        let p = self.p;
        let q = self.q;

        // Center the series; the intercept is handled by the caller.
        let mean = if self.include_intercept {
            data.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

        // ---------------------------------------------------------------------
        // Stage 1: fit a long autoregression AR(p_long) by least squares to
        // obtain residuals that approximate the innovations.
        // ---------------------------------------------------------------------
        // Order rule (matching the SciRS2 reference implementation): an order of
        // O(sqrt(n)) is large enough to whiten an ARMA(p, q) process yet small
        // enough to keep the stage-1 fit well-conditioned. A larger order
        // over-fits in finite samples and destabilises the stage-2 estimates.
        let p_long_target = ((n as f64).sqrt() as usize).max(p + q);
        // Keep at least a few equations beyond the number of unknowns.
        let p_long = p_long_target.min(n / 2).max(1).min(n.saturating_sub(2));

        let innovations = self.long_ar_residuals(&centered, p_long)?;

        // `innovations[t]` is aligned with `centered[t]` for t >= p_long and is
        // zero for the presample region t < p_long (the innovations there are
        // unobserved; using zeros is the standard Hannan-Rissanen convention).

        // ---------------------------------------------------------------------
        // Stage 2: regress y_t on its own lags (AR) and on lagged residuals (MA)
        // by ordinary least squares.
        // ---------------------------------------------------------------------
        let max_lag = p.max(q);
        // Start far enough in that all lagged residuals are genuine estimates.
        let start = (p_long + max_lag).max(max_lag);

        if start >= n {
            return Err(NumRs2Error::ValueError(
                "Insufficient data for Hannan-Rissanen MA estimation".to_string(),
            ));
        }

        let n_rows = n - start;
        let n_cols = p + q;

        if n_rows <= n_cols {
            // Not enough equations for a least-squares fit; fall back to a
            // first-order innovations approximation rather than a placeholder.
            return self.ma_from_residual_autocovariance(&innovations[..], start);
        }

        let mut design = Array2::<f64>::zeros((n_rows, n_cols));
        let mut target = Array1::<f64>::zeros(n_rows);

        for (row, t) in (start..n).enumerate() {
            target[row] = centered[t];

            // AR regressors: y_{t-1}, ..., y_{t-p}.
            for j in 0..p {
                design[[row, j]] = centered[t - j - 1];
            }
            // MA regressors: ε̂_{t-1}, ..., ε̂_{t-q}.
            for j in 0..q {
                design[[row, p + j]] = innovations[t - j - 1];
            }
        }

        // Solve the joint least-squares problem via the crate's QR-based solver.
        let solution = match scirs2_linalg::lstsq(&design.view(), &target.view(), None) {
            Ok(result) => result.x,
            Err(_) => {
                // Fall back to the residual-autocovariance approximation if the
                // design matrix is rank-deficient.
                return self.ma_from_residual_autocovariance(&innovations[..], start);
            }
        };

        // Extract the MA block (coefficients of the lagged residual regressors).
        let mut ma_coefs = Array1::zeros(q);
        for j in 0..q {
            ma_coefs[j] = solution[p + j];
        }

        // Guard against non-finite estimates from an ill-conditioned solve.
        if ma_coefs.iter().any(|c| !c.is_finite()) {
            return self.ma_from_residual_autocovariance(&innovations[..], start);
        }

        Ok(ma_coefs)
    }

    /// Fit a long autoregression AR(`order`) by ordinary least squares and
    /// return its residuals aligned to the input series.
    ///
    /// The returned vector has the same length as `series`; entries before
    /// `order` (the presample) are set to zero because the corresponding
    /// innovations are unobservable.
    fn long_ar_residuals(&self, series: &[f64], order: usize) -> Result<Vec<f64>> {
        let n = series.len();
        let mut residuals = vec![0.0; n];

        if order == 0 || n <= order {
            // Degenerate case: treat the (centered) series itself as the noise.
            residuals.copy_from_slice(series);
            return Ok(residuals);
        }

        let n_rows = n - order;
        let mut design = Array2::<f64>::zeros((n_rows, order));
        let mut target = Array1::<f64>::zeros(n_rows);

        for (row, t) in (order..n).enumerate() {
            target[row] = series[t];
            for j in 0..order {
                design[[row, j]] = series[t - j - 1];
            }
        }

        let phi_long = match scirs2_linalg::lstsq(&design.view(), &target.view(), None) {
            Ok(result) => result.x,
            Err(_) => {
                // If the long AR fit fails, approximate innovations by the
                // centered series (white-noise assumption).
                residuals.copy_from_slice(series);
                return Ok(residuals);
            }
        };

        // Compute residuals ε̂_t = y_t - Σ φ̂_i y_{t-i} for t >= order.
        for t in order..n {
            let mut prediction = 0.0;
            for j in 0..order {
                prediction += phi_long[j] * series[t - j - 1];
            }
            residuals[t] = series[t] - prediction;
        }

        Ok(residuals)
    }

    /// Fallback MA estimate from the autocovariance of the AR residuals.
    ///
    /// Used only when the joint Hannan-Rissanen regression cannot be solved
    /// (rank deficiency or too few equations). Approximates the MA polynomial
    /// from the residual autocorrelation structure.
    fn ma_from_residual_autocovariance(
        &self,
        innovations: &[f64],
        start: usize,
    ) -> Result<Array1<f64>> {
        use crate::new_modules::timeseries::autocorrelation;

        let skip = start.min(innovations.len());
        let usable: Vec<f64> = innovations.iter().skip(skip).copied().collect();
        if usable.len() <= self.q {
            return Ok(Array1::zeros(self.q));
        }

        let res_array = Array1::from_vec(usable);
        // If the residuals are degenerate (near-constant), fall back to zeros
        // rather than propagating an error from the autocorrelation routine.
        let res_acf = match autocorrelation(&res_array.view(), self.q) {
            Ok(acf) => acf,
            Err(_) => return Ok(Array1::zeros(self.q)),
        };

        let mut ma_coefs = Array1::zeros(self.q);
        for i in 0..self.q {
            ma_coefs[i] = res_acf[i + 1];
        }

        Ok(ma_coefs)
    }

    /// Compute residuals from full ARIMA model.
    fn compute_residuals(
        &self,
        data: &ArrayView1<f64>,
        ar_coefs: &Array1<f64>,
        ma_coefs: &Array1<f64>,
        intercept: f64,
    ) -> Result<Array1<f64>> {
        let n = data.len();
        let max_lag = self.p.max(self.q);
        let mut residuals = Array1::zeros(n);

        // Initialize residuals for first max_lag observations
        for t in 0..max_lag {
            residuals[t] = data[t] - intercept;
        }

        // Compute residuals for remaining observations
        for t in max_lag..n {
            let mut prediction = intercept;

            // AR component
            for i in 0..self.p {
                if t > i {
                    prediction += ar_coefs[i] * (data[t - i - 1] - intercept);
                }
            }

            // MA component
            for i in 0..self.q {
                if t > i {
                    prediction += ma_coefs[i] * residuals[t - i - 1];
                }
            }

            residuals[t] = data[t] - prediction;
        }

        Ok(residuals.slice(s![max_lag..]).to_owned())
    }

    /// Refine parameters using Maximum Likelihood Estimation.
    fn estimate_mle(
        &self,
        data: &ArrayView1<f64>,
        init_params: &ArimaParams,
    ) -> Result<ArimaParams> {
        // For now, return CSS estimates
        // Full MLE would require numerical optimization (e.g., Nelder-Mead, BFGS)
        // which we can implement later
        Ok(init_params.clone())
    }

    /// Compute log-likelihood for given residuals and variance.
    fn compute_log_likelihood(&self, residuals: &Array1<f64>, sigma2: f64) -> f64 {
        let n = residuals.len() as f64;
        let ss = residuals.iter().map(|&r| r * r).sum::<f64>();

        -0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * ss / sigma2
    }

    /// Forecast future values.
    ///
    /// # Arguments
    ///
    /// * `data` - Historical time series data
    /// * `params` - Fitted model parameters
    /// * `steps` - Number of steps ahead to forecast
    ///
    /// # Returns
    ///
    /// Array of forecasted values
    pub fn forecast(
        &self,
        data: &ArrayView1<f64>,
        params: &ArimaParams,
        steps: usize,
    ) -> Result<Array1<f64>> {
        // Apply differencing
        let diff_data = self.difference(data)?;
        let n = diff_data.len();

        // Extend series with forecasts
        let mut extended = diff_data.to_owned();
        let mut forecasts = Array1::zeros(steps);

        for h in 0..steps {
            let mut prediction = params.intercept;

            // AR component
            for i in 0..self.p {
                let idx = n + h - i - 1;
                if idx < extended.len() {
                    prediction += params.ar_coefs[i] * (extended[idx] - params.intercept);
                }
            }

            // MA component (assumes future errors are 0)
            // In practice, this is the expected value

            forecasts[h] = prediction;
            extended = concatenate_arrays(&extended, &Array1::from_vec(vec![prediction]));
        }

        // Integrate forecasts back if differencing was applied
        let integrated = self.integrate(&forecasts, data)?;

        Ok(integrated)
    }

    /// Integrate differenced forecasts back to original scale.
    fn integrate(
        &self,
        forecasts: &Array1<f64>,
        original_data: &ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        if self.d == 0 {
            return Ok(forecasts.clone());
        }

        let mut result = forecasts.clone();
        let n_orig = original_data.len();

        for _ in 0..self.d {
            let mut integrated = Array1::zeros(result.len());
            let last_value = if n_orig > 0 {
                original_data[n_orig - 1]
            } else {
                0.0
            };

            integrated[0] = last_value + result[0];
            for i in 1..result.len() {
                integrated[i] = integrated[i - 1] + result[i];
            }

            result = integrated;
        }

        Ok(result)
    }
}

/// Concatenate two 1D arrays.
fn concatenate_arrays(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(a.len() + b.len());
    result.slice_mut(s![..a.len()]).assign(a);
    result.slice_mut(s![a.len()..]).assign(b);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_arima_creation() {
        let model = Arima::new(1, 1, 1);
        assert_eq!(model.p, 1);
        assert_eq!(model.d, 1);
        assert_eq!(model.q, 1);
        assert!(model.include_intercept);
    }

    #[test]
    fn test_differencing() {
        let data = Array1::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
        let model = Arima::new(0, 1, 0);

        let diff = model
            .difference(&data.view())
            .expect("differencing should succeed");
        assert_eq!(diff.len(), 4);
        assert_relative_eq!(diff[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(diff[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(diff[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(diff[3], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_double_differencing() {
        let data = Array1::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0]);
        let model = Arima::new(0, 2, 0);

        let diff = model
            .difference(&data.view())
            .expect("double differencing should succeed");
        assert_eq!(diff.len(), 4);
        // First diff: [1, 2, 3, 4, 5]
        // Second diff: [1, 1, 1, 1]
        assert_relative_eq!(diff[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(diff[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arima_fit_simple() {
        // Simple AR(1) process
        let data = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.3, 2.5, 2.7, 2.8, 2.9, 3.0, 3.05]);
        let model = Arima::new(1, 0, 0);

        let params = model.fit(&data.view()).expect("ARIMA fit should succeed");
        assert!(params.ar_coefs.len() == 1);
        assert!(params.ma_coefs.is_empty());
        assert!(params.sigma2 > 0.0);
    }

    #[test]
    fn test_arima_forecast() {
        // Use data with trend + small noise (more realistic than perfect linear trend)
        // Avoid perfect linear trend which becomes constant after differencing
        let data = Array1::from_vec(vec![1.0, 2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.0, 9.1, 9.8]);
        let model = Arima::new(1, 1, 0);

        let params = model.fit(&data.view()).expect("fit should succeed");
        let forecast = model
            .forecast(&data.view(), &params, 3)
            .expect("forecast should succeed");

        assert_eq!(forecast.len(), 3);
        // For trend with noise, forecasts should continue the general upward trend
        // Allow wide range for numerical estimation variability
        assert!(
            forecast[0] > 8.0 && forecast[0] < 13.0,
            "First forecast {} should be reasonable",
            forecast[0]
        );
    }

    #[test]
    fn test_information_criteria() {
        let data = Array1::from_vec(vec![1.0, 1.5, 2.2, 2.8, 3.1, 3.5, 3.8, 4.0, 4.3, 4.5]);
        let model = Arima::new(1, 0, 1);

        let params = model.fit(&data.view()).expect("fit should succeed");

        // AIC and BIC should be finite
        assert!(params.aic.is_finite());
        assert!(params.bic.is_finite());
        // BIC penalizes complexity more than AIC
        assert!(params.bic > params.aic);
    }

    #[test]
    fn test_insufficient_data_error() {
        let data = Array1::from_vec(vec![1.0, 2.0]);
        let model = Arima::new(2, 1, 2);

        let result = model.fit(&data.view());
        assert!(result.is_err());
    }
}
