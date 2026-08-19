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
        let log_likelihood = self.compute_log_likelihood(&residuals.view(), &sigma.view())?;

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
    ///
    /// Returns an error rather than silently substituting a
    /// diag(1/sigma_ii) approximation for Sigma^-1 when the exact solve
    /// fails: a diagonal fallback ignores all off-diagonal covariance and
    /// would silently corrupt the reported log-likelihood/AIC/BIC without
    /// any indication to the caller that Sigma was singular/ill-conditioned.
    fn compute_log_likelihood(
        &self,
        residuals: &ArrayView2<f64>,
        sigma: &ArrayView2<f64>,
    ) -> Result<f64> {
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
        // For small k, use direct inversion. A diag(1/sigma_ii) fallback
        // here would silently discard all off-diagonal covariance and
        // report a wrong log-likelihood/AIC/BIC with no indication to the
        // caller, so propagate the failure instead of masking it.
        let eye_matrix: Array2<f64> = Array2::eye(k);
        let sigma_inv = scirs2_linalg::solve_multiple(&sigma.view(), &eye_matrix.view(), None)
            .map_err(|e| {
                NumRs2Error::ComputationError(format!(
                    "VAR log-likelihood: failed to invert the residual covariance matrix Sigma: {}",
                    e
                ))
            })?;

        let mut quad_form = 0.0;
        for i in 0..n {
            let r = residuals.row(i);
            let temp = sigma_inv.dot(&r);
            for j in 0..k {
                quad_form += r[j] * temp[j];
            }
        }

        Ok(
            -0.5 * (n as f64 * (k as f64 * (2.0 * std::f64::consts::PI).ln() + log_det)
                + quad_form),
        )
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

    /// Fit VECM model via the Johansen (1988) procedure.
    ///
    /// # Algorithm
    /// 1. Form ΔY (first differences) and Y_{t-1} (lagged levels).
    /// 2. Regress both ΔY and Y_{t-1} on lagged ΔY (lags 1..p-1) via OLS.
    ///    Residuals are R0 (from ΔY regression) and R1 (from Y_{t-1} regression).
    /// 3. Compute moment matrices S00 = R0ᵀR0/T, S11 = R1ᵀR1/T, S01 = R0ᵀR1/T.
    /// 4. Form the symmetric PSD matrix M = S11^{-1/2} S10 S00^{-1} S01 S11^{-1/2}
    ///    and solve its symmetric eigendecomposition.
    /// 5. Top `rank` eigenvectors give the cointegrating vectors β;
    ///    α = S01 β diag(λ)^{-1}.
    /// 6. The returned `VarParams` encodes Π = αβᵀ as `coefficients[0]`,
    ///    zero intercept, and S00 as the residual covariance `sigma`.
    pub fn fit(&self, data: &ArrayView2<f64>) -> Result<VarParams> {
        let (t, k) = data.dim();
        let r = self.rank;

        if t < self.p + 2 {
            return Err(NumRs2Error::ValueError(
                "Insufficient observations for VECM estimation".to_string(),
            ));
        }
        if r == 0 || r > k {
            return Err(NumRs2Error::ValueError(format!(
                "Cointegration rank must satisfy 0 < rank <= k, got rank={} k={}",
                r, k
            )));
        }

        // ------------------------------------------------------------------ //
        // Step 1: Build ΔY (first differences) and Y_{t-1} (lagged levels)  //
        // ------------------------------------------------------------------ //
        // Usable observations: indices p..t-1 (after taking p lags of ΔY)
        // For each usable index i (0-based within usable range):
        //   t_idx = i + p  (original index, 1-based from p..t-1)
        //   ΔY_i  = Y[t_idx] - Y[t_idx - 1]
        //   Ylm1_i = Y[t_idx - 1]          (Y_{t-1})

        let n_obs = t - self.p; // number of usable rows
        let mut delta_y = Array2::<f64>::zeros((n_obs, k));
        let mut y_lag1 = Array2::<f64>::zeros((n_obs, k));

        for i in 0..n_obs {
            let t_idx = i + self.p;
            for j in 0..k {
                delta_y[[i, j]] = data[[t_idx, j]] - data[[t_idx - 1, j]];
                y_lag1[[i, j]] = data[[t_idx - 1, j]];
            }
        }

        // ------------------------------------------------------------------ //
        // Step 2: Regress ΔY and Y_{t-1} on lagged ΔY (lags 1..p-1)        //
        // If p == 1 there are no lagged ΔY regressors; residuals = raw data. //
        // ------------------------------------------------------------------ //
        let (resid0, resid1) = if self.p <= 1 {
            (delta_y.clone(), y_lag1.clone())
        } else {
            // Build regressor matrix Z: rows = n_obs, cols = k*(p-1)
            let n_z_cols = k * (self.p - 1);
            let mut z = Array2::<f64>::zeros((n_obs, n_z_cols));
            for i in 0..n_obs {
                let t_idx = i + self.p;
                let mut col = 0usize;
                for lag in 1..self.p {
                    for j in 0..k {
                        // ΔY lagged by `lag` periods
                        z[[i, col]] = data[[t_idx - lag, j]] - data[[t_idx - lag - 1, j]];
                        col += 1;
                    }
                }
            }

            // OLS projection matrices: B = (ZᵀZ)^{-1} Zᵀ Y
            // Residual = Y - Z B
            let ztzt = z.t().dot(&z); // (n_z_cols × n_z_cols)
            let ztdy = z.t().dot(&delta_y); // (n_z_cols × k)
            let zty1 = z.t().dot(&y_lag1); // (n_z_cols × k)

            let eye_z = Array2::<f64>::eye(n_z_cols);
            let ztzt_inv = match scirs2_linalg::solve_multiple(&ztzt.view(), &eye_z.view(), None) {
                Ok(inv) => inv,
                Err(_) => {
                    // Regularise lightly if singular
                    let reg = &ztzt + &(Array2::<f64>::eye(n_z_cols) * 1e-10);
                    scirs2_linalg::solve_multiple(&reg.view(), &eye_z.view(), None).map_err(
                        |_| {
                            NumRs2Error::ComputationError(
                                "VECM: singular regressor matrix in auxiliary regression"
                                    .to_string(),
                            )
                        },
                    )?
                }
            };

            let beta_dy = ztzt_inv.dot(&ztdy); // (n_z_cols × k)
            let beta_y1 = ztzt_inv.dot(&zty1);

            let fitted_dy = z.dot(&beta_dy);
            let fitted_y1 = z.dot(&beta_y1);

            (&delta_y - &fitted_dy, &y_lag1 - &fitted_y1)
        };

        // ------------------------------------------------------------------ //
        // Step 3: Moment matrices                                             //
        // ------------------------------------------------------------------ //
        let t_f = n_obs as f64;
        let s00 = resid0.t().dot(&resid0) / t_f; // k × k
        let s11 = resid1.t().dot(&resid1) / t_f; // k × k
        let s01 = resid0.t().dot(&resid1) / t_f; // k × k  (= S10ᵀ)

        // ------------------------------------------------------------------ //
        // Step 4: Compute S11^{-1/2} via eigendecomposition of S11           //
        //   S11 = V D Vᵀ  =>  S11^{-1/2} = V D^{-1/2} Vᵀ                  //
        // ------------------------------------------------------------------ //
        // Regularise S11 slightly for robustness
        let s11_reg = &s11 + &(Array2::<f64>::eye(k) * 1e-12);

        let (eigvals_s11, eigvecs_s11) = scirs2_linalg::eigh(&s11_reg.view(), None)
            .map_err(|e| NumRs2Error::ComputationError(format!("VECM: eigh(S11) failed: {}", e)))?;

        // D^{-1/2}: invert and sqrt each eigenvalue (clamp for stability)
        let d_inv_sqrt: Array2<f64> = {
            let mut mat = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                let ev = eigvals_s11[i].max(1e-14);
                mat[[i, i]] = 1.0 / ev.sqrt();
            }
            mat
        };

        // S11^{-1/2} = V * D^{-1/2} * Vᵀ
        let s11_inv_sqrt = eigvecs_s11.dot(&d_inv_sqrt).dot(&eigvecs_s11.t());

        // ------------------------------------------------------------------ //
        // Step 5: Build and diagonalise M = S11^{-1/2} S10 S00^{-1} S01 S11^{-1/2}
        // ------------------------------------------------------------------ //
        let eye_k = Array2::<f64>::eye(k);
        let s00_inv = match scirs2_linalg::solve_multiple(&s00.view(), &eye_k.view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                let reg = &s00 + &(Array2::<f64>::eye(k) * 1e-12);
                scirs2_linalg::solve_multiple(&reg.view(), &eye_k.view(), None).map_err(|_| {
                    NumRs2Error::ComputationError("VECM: singular S00 matrix".to_string())
                })?
            }
        };

        // M = S11^{-1/2} * S10 * S00^{-1} * S01 * S11^{-1/2}
        // Note: S10 = S01ᵀ
        let s10 = s01.t().to_owned();
        let mid = s10.dot(&s00_inv).dot(&s01); // k × k
        let m_mat = s11_inv_sqrt.dot(&mid).dot(&s11_inv_sqrt); // k × k

        // Symmetrise M to counteract floating-point asymmetry
        let m_sym = (&m_mat + &m_mat.t()) / 2.0;

        let (eigvals_m, eigvecs_m) = scirs2_linalg::eigh(&m_sym.view(), None)
            .map_err(|e| NumRs2Error::ComputationError(format!("VECM: eigh(M) failed: {}", e)))?;

        // eigh returns eigenvalues in ascending order; "top r" means last r
        let total = eigvals_m.len();

        // ------------------------------------------------------------------ //
        // Step 6: Cointegrating vectors β and loading matrix α               //
        // β columns are in the transformed space — map back via S11^{-1/2}  //
        // ------------------------------------------------------------------ //
        // Top r eigenvectors of M (in S11^{-1/2} coordinates)
        let top_vecs = eigvecs_m.slice(s![.., total - r..]).to_owned(); // k × r
        let top_vals = eigvals_m.slice(s![total - r..]).to_owned(); // r

        // β = S11^{-1/2} * top_vecs  (map back to original space)
        let beta = s11_inv_sqrt.dot(&top_vecs); // k × r

        // α = S01 * β * diag(λ)^{-1}
        //   = S01 * β * D_λ^{-1}
        let s01_beta = s01.dot(&beta); // k × r
        let mut alpha = Array2::<f64>::zeros((k, r));
        for i in 0..r {
            let lam = top_vals[i].max(1e-14);
            for j in 0..k {
                alpha[[j, i]] = s01_beta[[j, i]] / lam;
            }
        }

        // Π = α βᵀ  (k × k), the long-run impact matrix
        let pi = alpha.dot(&beta.t()); // k × k

        // Pack into VarParams
        // coefficients[0] = Π
        // sigma = S00 (residual covariance of the differenced process)
        let log_likelihood = 0.0_f64; // not computed (would require Gaussian likelihood)
        let n_params = k * r * 2; // r*(α + β) columns
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + n_params as f64 * (n_obs as f64).ln();

        Ok(VarParams {
            coefficients: vec![pi],
            intercept: Array1::zeros(k),
            sigma: s00,
            log_likelihood,
            aic,
            bic,
        })
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
        // Bivariate series with irregular (non-perfectly-linear) step
        // sizes in each column, so the OLS residuals -- and hence the
        // residual covariance Sigma -- are genuinely nonzero and
        // well-conditioned, rather than an (exactly or near-)singular
        // all-zero matrix. A perfectly linear series is fit near-exactly
        // by VAR(1), driving Sigma to numerically singular and silently
        // masking whether Sigma^{-1} (needed for the log-likelihood) is
        // actually being computed.
        let data = arr2(&[
            [1.0, 2.0],
            [1.7, 2.1],
            [1.9, 2.8],
            [2.8, 2.6],
            [2.6, 3.4],
            [3.5, 3.0],
            [3.4, 3.9],
            [4.3, 3.5],
            [4.0, 4.3],
            [4.9, 3.9],
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
        // As in `test_var_fit`: irregular step sizes keep the residual
        // covariance nonsingular instead of relying on a perfectly-linear
        // (near-zero-residual) series.
        let data = arr2(&[
            [1.0, 2.0],
            [1.8, 2.7],
            [2.7, 3.8],
            [3.3, 4.2],
            [4.4, 5.5],
            [4.8, 5.7],
            [6.1, 7.0],
            [6.5, 7.3],
        ]);

        let var = Var::new(1);
        let params = var.fit(&data.view()).expect("fit should succeed");
        let forecast = var
            .forecast(&data.view(), &params, 2)
            .expect("forecast should succeed");

        assert_eq!(forecast.dim(), (2, 2));
    }
}
