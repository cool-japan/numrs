//! Vector Autoregression (VAR) and Vector Error Correction Models (VECM)
//!
//! Multivariate time series models for analyzing relationships between multiple
//! time series variables.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};

/// VAR model parameters.
#[derive(Debug, Clone)]
pub struct VarParams {
    /// Coefficient matrices for each lag (p matrices of size k×k)
    pub coefficients: Vec<Array2<f64>>,
    /// Intercept vector
    pub intercept: Array1<f64>,
    /// Residual covariance matrix
    pub sigma: Array2<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
}

/// Vector Autoregression model VAR(p).
///
/// Model equation for k-dimensional time series Y_t:
///
/// Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + AₚY_{t-p} + ε_t
///
/// where ε_t ~ N(0, Σ)
#[derive(Debug, Clone)]
pub struct Var {
    /// Number of lags
    pub p: usize,
    /// Include intercept
    pub include_intercept: bool,
}

impl Var {
    /// Create new VAR(p) model.
    pub fn new(p: usize) -> Self {
        Self {
            p,
            include_intercept: true,
        }
    }

    /// Fit VAR model using OLS.
    pub fn fit(&self, data: &ArrayView2<f64>) -> Result<VarParams> {
        let (t, k) = data.dim();

        if t <= self.p {
            return Err(NumRs2Error::ValueError(
                "Insufficient observations for VAR estimation".to_string(),
            ));
        }

        // Construct design matrix X and response matrix Y
        let n_obs = t - self.p;
        let n_vars = k * self.p + if self.include_intercept { 1 } else { 0 };

        let mut x = Array2::zeros((n_obs, n_vars));
        let mut y = Array2::zeros((n_obs, k));

        for i in 0..n_obs {
            let t_idx = i + self.p;
            y.row_mut(i).assign(&data.row(t_idx));

            let mut col_idx = 0;

            // Intercept
            if self.include_intercept {
                x[[i, col_idx]] = 1.0;
                col_idx += 1;
            }

            // Lagged values
            for lag in 1..=self.p {
                let lag_data = data.row(t_idx - lag);
                for j in 0..k {
                    x[[i, col_idx]] = lag_data[j];
                    col_idx += 1;
                }
            }
        }

        // OLS estimation: B = (X'X)^{-1} X'Y
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);

        let beta: Array2<f64> = match scirs2_linalg::solve_multiple(&xtx.view(), &xty.view(), None)
        {
            Ok(b) => b,
            Err(_) => return Err(NumRs2Error::ComputationError("Singular matrix".to_string())),
        };

        // Extract coefficients
        let mut coefficients = Vec::with_capacity(self.p);
        let mut col_idx = if self.include_intercept { 1 } else { 0 };

        let intercept = if self.include_intercept {
            beta.row(0).to_owned()
        } else {
            Array1::zeros(k)
        };

        for _ in 0..self.p {
            let mut coef = Array2::zeros((k, k));
            for i in 0..k {
                for j in 0..k {
                    coef[[j, i]] = beta[[col_idx + i, j]];
                }
            }
            coefficients.push(coef);
            col_idx += k;
        }

        // Compute residuals
        let y_pred = x.dot(&beta);
        let residuals = &y - &y_pred;

        // Residual covariance
        let sigma: Array2<f64> = residuals.t().dot(&residuals) / (n_obs as f64);

        // Log-likelihood (multivariate normal)
        let log_likelihood = self.compute_log_likelihood(&residuals.view(), &sigma.view());

        // Information criteria
        let n_params = n_vars * k;
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (n_params as f64) * (n_obs as f64).ln();

        Ok(VarParams {
            coefficients,
            intercept,
            sigma,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// Compute multivariate normal log-likelihood.
    fn compute_log_likelihood(&self, residuals: &ArrayView2<f64>, sigma: &ArrayView2<f64>) -> f64 {
        let (n, k) = residuals.dim();

        // Compute determinant with regularization for numerical stability
        let det_sigma = match scirs2_linalg::det(&sigma.view(), None) {
            Ok(det) if det > 1e-10 => det,
            _ => {
                // Regularize if nearly singular
                let eye: Array2<f64> = Array2::eye(k);
                let reg_sigma = sigma.to_owned() + &(eye * 1e-6);
                scirs2_linalg::det(&reg_sigma.view(), None)
                    .unwrap_or(1e-6)
                    .max(1e-10)
            }
        };

        let log_det = det_sigma.ln();

        // Compute trace term: sum_i (r_i^T Sigma^{-1} r_i)
        // For small k, use direct inversion
        let eye_matrix: Array2<f64> = Array2::eye(k);
        let sigma_inv = match scirs2_linalg::solve_multiple(&sigma.view(), &eye_matrix.view(), None)
        {
            Ok(inv) => inv,
            Err(_) => {
                // Fallback: use diagonal approximation if inversion fails
                let mut diag_inv = Array2::<f64>::zeros((k, k));
                for i in 0..k {
                    diag_inv[[i, i]] = 1.0 / sigma[[i, i]].max(1e-10);
                }
                diag_inv
            }
        };

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

    /// Forecast h steps ahead.
    pub fn forecast(
        &self,
        data: &ArrayView2<f64>,
        params: &VarParams,
        steps: usize,
    ) -> Result<Array2<f64>> {
        let (t, k) = data.dim();
        let mut forecasts = Array2::zeros((steps, k));

        // Use last p observations
        let mut history: Vec<Array1<f64>> = Vec::new();
        for i in (t.saturating_sub(self.p))..t {
            history.push(data.row(i).to_owned());
        }

        for h in 0..steps {
            let mut forecast = params.intercept.clone();

            for (lag, coef_matrix) in params.coefficients.iter().enumerate() {
                if lag < history.len() {
                    let lagged_vals = &history[history.len() - lag - 1];
                    forecast = forecast + coef_matrix.dot(lagged_vals);
                }
            }

            forecasts.row_mut(h).assign(&forecast);
            history.push(forecast);

            // Keep only last p observations
            if history.len() > self.p {
                history.remove(0);
            }
        }

        Ok(forecasts)
    }
}

/// Vector Error Correction Model (VECM).
#[derive(Debug, Clone)]
pub struct Vecm {
    /// Number of lags in differenced form
    pub p: usize,
    /// Cointegration rank
    pub rank: usize,
}

impl Vecm {
    /// Create new VECM model.
    pub fn new(p: usize, rank: usize) -> Self {
        Self { p, rank }
    }

    /// Fit VECM model (simplified - would need Johansen procedure for full implementation).
    pub fn fit(&self, _data: &ArrayView2<f64>) -> Result<VarParams> {
        // Placeholder - full implementation would estimate cointegration vectors
        // and error correction terms
        Err(NumRs2Error::NotImplemented(
            "Full VECM estimation not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr2, Array2};

    #[test]
    fn test_var_creation() {
        let var = Var::new(2);
        assert_eq!(var.p, 2);
        assert!(var.include_intercept);
    }

    #[test]
    fn test_var_fit() {
        // Create simple bivariate series
        let data = arr2(&[
            [1.0, 2.0],
            [1.5, 2.3],
            [2.0, 2.5],
            [2.5, 2.8],
            [3.0, 3.0],
            [3.5, 3.2],
            [4.0, 3.5],
            [4.5, 3.8],
        ]);

        let var = Var::new(1);
        let params = var.fit(&data.view()).expect("VAR fit should succeed");

        assert_eq!(params.coefficients.len(), 1);
        assert_eq!(params.intercept.len(), 2);
        assert!(params.aic.is_finite());
        assert!(params.bic.is_finite());
    }

    #[test]
    fn test_var_forecast() {
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]);

        let var = Var::new(1);
        let params = var.fit(&data.view()).expect("fit should succeed");
        let forecast = var
            .forecast(&data.view(), &params, 2)
            .expect("forecast should succeed");

        assert_eq!(forecast.dim(), (2, 2));
    }
}
