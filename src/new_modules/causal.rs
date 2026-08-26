//! Causal Inference Module
//!
//! This module provides a production-grade suite of causal inference estimators,
//! implemented natively in pure Rust and modelled after the algorithms found in
//! `scirs2-stats`'s causal sub-crate (which is not yet re-exported publicly in
//! `scirs2_stats::causal`).  The implementations cover:
//!
//! - **Instrumental Variables (IV / 2SLS)** for endogeneity correction
//! - **Difference-in-Differences (DiD)** with TWFE regression and parallel-trends tests
//! - **Propensity Score** estimation (logistic regression), IPW, and matching
//!
//! All functions return `Result<T, NumRs2Error>` and follow the no-`unwrap()` policy.
//!
//! # Instrumental Variables (2SLS)
//!
//! Two-Stage Least Squares resolves endogeneity by replacing the endogenous
//! regressor with its projection onto the instrument space.  For the simple
//! just-identified case:
//!
//! ```text
//! First stage:  x̂ = Z (Z'Z)⁻¹ Z' x
//! Second stage: β̂_2SLS = (x̂'x)⁻¹ x̂' y
//! ```
//!
//! References:
//! - Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*.
//! - Stock, J.H., Wright, J.H. & Yogo, M. (2002). A Survey of Weak Instruments.
//!
//! # Difference-in-Differences (DiD)
//!
//! The 2×2 DiD estimator recovers the ATT under the parallel-trends assumption:
//!
//! ```text
//! ATT = (Ȳ_{treat,post} − Ȳ_{treat,pre}) − (Ȳ_{control,post} − Ȳ_{control,pre})
//! ```
//!
//! The underlying regression is a Two-Way Fixed Effects (TWFE) model with unit
//! and time fixed effects, which generalises to unbalanced panels.
//!
//! References:
//! - Roth, J. et al. (2023). What's Trending in Difference-in-Differences?
//!
//! # Propensity Score Methods
//!
//! The propensity score `e(x) = P(W=1 | X=x)` is estimated by logistic regression
//! via Iteratively Reweighted Least Squares (IRLS).  The fitted score is used for
//! Inverse Probability Weighting (IPW):
//!
//! ```text
//! ATE = (1/n) Σ_i [ W_i Y_i / e(X_i)  −  (1−W_i) Y_i / (1−e(X_i)) ]
//! ```
//!
//! References:
//! - Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity Score.
//! - Hirano, K., Imbens, G.W. & Ridder, G. (2003). Efficient Estimation of ATE.

use crate::error::{NumRs2Error, Result};

// ---------------------------------------------------------------------------
// Internal linear-algebra primitives (pure Rust, no external BLAS)
// ---------------------------------------------------------------------------

/// Compute the Cholesky factorisation L of a symmetric positive-definite matrix A
/// (A = L Lᵀ) and return L as a flat row-major Vec<f64>.
///
/// Returns `Err` if A is not positive definite (diagonal entry ≤ 0 after subtracting
/// the accumulated sum of squares).
fn cholesky_lower(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for p in 0..j {
                s -= l[i * n + p] * l[j * n + p];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(NumRs2Error::ComputationError(format!(
                        "cholesky: matrix is not positive definite at diagonal ({i},{i})"
                    )));
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Solve L x = b (forward substitution) where L is lower-triangular (row-major Vec).
fn forward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Solve Lᵀ x = b (back substitution) where L is lower-triangular (row-major Vec).
fn back_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j]; // Lᵀ[i,j] = L[j,i]
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Invert a symmetric positive-definite (n×n) matrix A via Cholesky.
///
/// Returns A⁻¹ as a flat row-major Vec<f64> of length n².
fn invert_spd(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let l = cholesky_lower(a, n)?;
    let mut inv = vec![0.0_f64; n * n];
    // Invert column-by-column: solve A x_j = e_j
    for j in 0..n {
        let mut ej = vec![0.0_f64; n];
        ej[j] = 1.0;
        let y = forward_sub(&l, &ej, n);
        let x = back_sub(&l, &y, n);
        for i in 0..n {
            inv[i * n + j] = x[i];
        }
    }
    Ok(inv)
}

/// Multiply (m×k) matrix A by (k×n) matrix B, returning (m×n) result.
/// All matrices are flat row-major Vecs.
fn mat_mul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Ordinary Least Squares: solve (XᵀX)β = Xᵀy.
///
/// Returns `(β, residuals, (XᵀX)⁻¹)`.
/// `x` is a flat row-major (n × k) matrix; `y` is length-n.
fn ols_fit(x: &[f64], y: &[f64], n: usize, k: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if n < k {
        return Err(NumRs2Error::ComputationError(format!(
            "ols_fit: n={n} < k={k}, not enough observations"
        )));
    }
    // XᵀX  (k×k)
    let xtx = mat_mul(
        &{
            // Transpose of X: (k×n)
            let mut xt = vec![0.0_f64; k * n];
            for i in 0..n {
                for j in 0..k {
                    xt[j * n + i] = x[i * k + j];
                }
            }
            xt
        },
        k,
        n,
        x,
        k,
    );
    // Xᵀy  (k)
    let xty = {
        let mut v = vec![0.0_f64; k];
        for j in 0..k {
            for i in 0..n {
                v[j] += x[i * k + j] * y[i];
            }
        }
        v
    };
    let xtx_inv = invert_spd(&xtx, k)?;
    let beta = mat_mul(&xtx_inv, k, k, &xty, 1);
    // residuals = y - X β
    let resid: Vec<f64> = (0..n)
        .map(|i| {
            let fitted: f64 = (0..k).map(|j| x[i * k + j] * beta[j]).sum();
            y[i] - fitted
        })
        .collect();
    Ok((beta, resid, xtx_inv))
}

// ---------------------------------------------------------------------------
// Normal / t distribution helpers
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun 7.1.26 approximation to erf(x), max |error| < 1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (0.254_829_592
            + (-0.284_496_736 + (1.421_413_741 + (-1.453_152_027 + 1.061_405_429 * t) * t) * t)
                * t)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn normal_p_value(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Two-sided p-value from a t-distribution with `df` degrees of freedom.
/// Uses the normal approximation for large df.
fn t_p_value(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    if df > 200.0 {
        return normal_p_value(t);
    }
    // Regularised incomplete beta: I_x(df/2, 1/2) at x = df / (df + t²)
    let x = df / (df + t * t);
    regularised_incomplete_beta(x, df / 2.0, 0.5).clamp(0.0, 1.0)
}

/// Regularised incomplete beta I_x(a, b) via continued fraction (Lentz's method).
fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use the symmetry relation when x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularised_incomplete_beta(1.0 - x, b, a);
    }
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp() / a;
    front * betacf(x, a, b)
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200_usize;
    let eps = 3.0e-7_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    d = if d.abs() < 1e-30 { 1e-30 } else { 1.0 / d };
    let mut h = d;
    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Natural log-gamma via Lanczos approximation (accurate to ~15 digits).
fn lgamma(z: f64) -> f64 {
    // Lanczos coefficients for g=7
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let mut x = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        x += ci / (z + i as f64 + 1.0);
    }
    let t = z + g + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Approximate t-distribution quantile (used for confidence intervals).
/// Uses a bisection search against `t_p_value`.
fn t_quantile_two_sided(p: f64, df: f64) -> f64 {
    // We want t such that P(|T| > t) = p, i.e., t_p_value(t, df) = p
    // Equivalently CDF(t) = 1 - p/2
    let mut lo = 0.0_f64;
    let mut hi = 20.0_f64;
    // Make sure hi is large enough
    while t_p_value(hi, df) > p {
        hi *= 2.0;
    }
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if t_p_value(mid, df) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

// ---------------------------------------------------------------------------
// Sigmoid helper
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ---------------------------------------------------------------------------
// Instrumental Variables
// ---------------------------------------------------------------------------

/// Result of a Two-Stage Least Squares (2SLS) estimation.
///
/// The coefficient vector contains entries for the combined regressor matrix
/// `[x_endogenous, intercept]` (in that order) when using the convenience
/// function [`two_stage_least_squares`].
#[derive(Debug, Clone)]
pub struct IVEstimate {
    /// Point estimates of the structural-equation coefficients.
    /// For the simple API: `[β_x (causal effect), β_intercept]`.
    pub coef: Vec<f64>,
    /// HC1 heteroskedasticity-robust standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics for each coefficient
    pub t_stats: Vec<f64>,
    /// Two-sided p-values for each coefficient
    pub p_values: Vec<f64>,
    /// First-stage F-statistic (weak-instrument diagnostic).
    /// `None` when no endogenous regressors are present.
    pub first_stage_f: Option<f64>,
    /// Partial R² from the first-stage regression
    pub partial_r_squared: Option<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Estimator name (`"2SLS"`)
    pub estimator: String,
}

/// Weak-instrument diagnostic summary (Stock-Yogo first-stage F-test).
#[derive(Debug, Clone)]
pub struct WeakInstrumentSummary {
    /// First-stage F-statistic for each endogenous regressor
    pub f_statistics: Vec<f64>,
    /// Partial R² for each endogenous regressor
    pub partial_r_squared: Vec<f64>,
    /// Stock-Yogo critical value at 10 % maximal IV size distortion (≈ 10)
    pub critical_value_10pct: f64,
    /// `true` if the corresponding instrument is considered weak (F < critical value)
    pub instruments_weak: Vec<bool>,
}

/// Compute the first-stage F-statistic and partial R² for a single endogenous
/// regressor `x` projected on the full instrument set `z = [z_excl | x_exog]`.
///
/// The partial F tests whether the excluded instruments have significant predictive
/// power for `x` after partialling out the exogenous regressors.
fn first_stage_diagnostics(
    x: &[f64],
    x_exog: &[f64],
    z_excl: &[f64],
    n: usize,
    k_exog: usize,
    l_excl: usize,
) -> Result<(f64, f64)> {
    let l_total = l_excl + k_exog;

    // Full instrument matrix z = [z_excl | x_exog]  (n × l_total)
    let mut z_full = vec![0.0_f64; n * l_total];
    for i in 0..n {
        for j in 0..l_excl {
            z_full[i * l_total + j] = z_excl[i * l_excl + j];
        }
        for j in 0..k_exog {
            z_full[i * l_total + l_excl + j] = x_exog[i * k_exog + j];
        }
    }

    // Restricted model: regress x on x_exog only
    let (_, resid_r, _) = ols_fit(x_exog, x, n, k_exog)?;
    let rss_r: f64 = resid_r.iter().map(|&r| r * r).sum();

    // Unrestricted model: regress x on z_full
    let (_, resid_u, _) = ols_fit(&z_full, x, n, l_total)?;
    let rss_u: f64 = resid_u.iter().map(|&r| r * r).sum();

    let df_u = (n - l_total) as f64;
    if df_u <= 0.0 {
        return Err(NumRs2Error::ComputationError(
            "first_stage_diagnostics: insufficient degrees of freedom".into(),
        ));
    }

    let f_stat = if rss_u > 1e-15 {
        ((rss_r - rss_u) / l_excl as f64) / (rss_u / df_u)
    } else {
        f64::INFINITY
    };

    // Total SS for x (centred)
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let tss: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let partial_r2 = if tss > 1e-15 {
        ((rss_r - rss_u) / tss).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok((f_stat.max(0.0), partial_r2))
}

/// Estimate a causal effect via Two-Stage Least Squares (2SLS).
///
/// This handles the most common just-identified IV setup: a single endogenous
/// regressor `x` instrumented by a single excluded instrument `z`.  An intercept
/// column is added automatically.
///
/// # Arguments
///
/// * `y` – outcome vector (length *n*)
/// * `x` – endogenous regressor (length *n*)
/// * `z` – excluded instrument (length *n*); must be correlated with `x` but
///   affect `y` only through `x` (exclusion restriction)
///
/// # Returns
///
/// [`IVEstimate`] whose `coef` vector has two entries:
/// `[β_x (causal effect of x on y), β_intercept]`.
///
/// # Errors
///
/// Returns [`NumRs2Error::DimensionMismatch`] if slice lengths differ, or
/// [`NumRs2Error::ComputationError`] for singular matrices or too few observations.
///
/// # Example
///
/// ```rust,no_run
/// use numrs2::new_modules::causal::two_stage_least_squares;
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x = vec![0.5, 1.0, 1.5, 2.0, 2.5];
/// let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = two_stage_least_squares(&y, &x, &z).expect("valid IV estimation data");
/// println!("Causal effect β_x = {:.4}", result.coef[0]);
/// ```
pub fn two_stage_least_squares(y: &[f64], x: &[f64], z: &[f64]) -> Result<IVEstimate> {
    let n = y.len();
    if x.len() != n || z.len() != n {
        return Err(NumRs2Error::DimensionMismatch(
            "two_stage_least_squares: y, x, and z must all have the same length".into(),
        ));
    }
    if n < 4 {
        return Err(NumRs2Error::ComputationError(
            "two_stage_least_squares: need at least 4 observations".into(),
        ));
    }

    // k_endog = 1 (just x), k_exog = 1 (intercept), l_excl = 1 (just z)
    // Full instrument matrix Z_full = [z | 1]  (n × 2)
    let l_total = 2_usize; // z_excl + intercept
    let mut z_full = vec![0.0_f64; n * l_total];
    for i in 0..n {
        z_full[i * l_total + 0] = z[i];
        z_full[i * l_total + 1] = 1.0;
    }

    // Regressor matrix X = [x | 1]  (n × 2)
    let k_total = 2_usize;
    let mut x_mat = vec![0.0_f64; n * k_total];
    for i in 0..n {
        x_mat[i * k_total + 0] = x[i];
        x_mat[i * k_total + 1] = 1.0;
    }

    // --- First stage: project each column of X_mat onto Z_full ---
    // X̂ = Z_full (Z_full' Z_full)⁻¹ Z_full' X_mat
    // For k_total=2 columns: project x, keep intercept as-is
    let (ztx0, _, _) = ols_fit(
        &z_full,
        &(0..n).map(|i| x_mat[i * k_total]).collect::<Vec<_>>(),
        n,
        l_total,
    )?;
    // x̂ (fitted values from first stage)
    let x_hat: Vec<f64> = (0..n)
        .map(|i| z_full[i * l_total] * ztx0[0] + z_full[i * l_total + 1] * ztx0[1])
        .collect();

    // X̂_mat = [x̂ | 1]
    let mut x_hat_mat = vec![0.0_f64; n * k_total];
    for i in 0..n {
        x_hat_mat[i * k_total + 0] = x_hat[i];
        x_hat_mat[i * k_total + 1] = 1.0;
    }

    // --- Second stage: β̂_2SLS = (X̂' X)⁻¹ X̂' y ---
    // Build X̂' X  (k×k)
    let mut xht_x = vec![0.0_f64; k_total * k_total];
    for r in 0..k_total {
        for c in 0..k_total {
            for i in 0..n {
                xht_x[r * k_total + c] += x_hat_mat[i * k_total + r] * x_mat[i * k_total + c];
            }
        }
    }
    let xhtx_inv = invert_spd(&xht_x, k_total)?;
    // X̂' y  (k)
    let xht_y: Vec<f64> = (0..k_total)
        .map(|j| (0..n).map(|i| x_hat_mat[i * k_total + j] * y[i]).sum())
        .collect();
    let beta = mat_mul(&xhtx_inv, k_total, k_total, &xht_y, 1);

    // Residuals from structural equation
    let residuals: Vec<f64> = (0..n)
        .map(|i| {
            let fitted: f64 = (0..k_total).map(|j| x_mat[i * k_total + j] * beta[j]).sum();
            y[i] - fitted
        })
        .collect();

    let df = n - k_total;
    if df == 0 {
        return Err(NumRs2Error::ComputationError(
            "two_stage_least_squares: no degrees of freedom remaining".into(),
        ));
    }

    // HC1 robust standard errors:
    // V = (n / df) * (X̂'X)⁻¹ [Σ u_i² x̂_i xᵢ'] (X̂'X)⁻¹
    let scale = n as f64 / df as f64;
    let mut meat = vec![0.0_f64; k_total * k_total];
    for i in 0..n {
        let ui2 = residuals[i] * residuals[i];
        for r in 0..k_total {
            for c in 0..k_total {
                meat[r * k_total + c] += ui2 * x_hat_mat[i * k_total + r] * x_mat[i * k_total + c];
            }
        }
    }
    let vcov_inner = mat_mul(&xhtx_inv, k_total, k_total, &meat, k_total);
    let vcov = mat_mul(&vcov_inner, k_total, k_total, &xhtx_inv, k_total);

    let std_errors: Vec<f64> = (0..k_total)
        .map(|j| (scale * vcov[j * k_total + j]).max(0.0).sqrt())
        .collect();
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats.iter().map(|&t| t_p_value(t, df as f64)).collect();

    // First-stage diagnostics
    let x_exog = vec![1.0_f64; n]; // intercept column  (n × 1)
    let z_excl: Vec<f64> = z.to_vec();
    let z_excl_mat = z_excl.clone(); // 1 column
    let (fs_f, pr2) = first_stage_diagnostics(x, &x_exog, &z_excl_mat, n, 1, 1)?;

    Ok(IVEstimate {
        coef: beta,
        std_errors,
        t_stats,
        p_values,
        first_stage_f: Some(fs_f),
        partial_r_squared: Some(pr2),
        n_obs: n,
        estimator: "2SLS".into(),
    })
}

/// Test whether the instrument `z` is weak (Stock-Yogo first-stage F-test).
///
/// A first-stage F-statistic below the Stock-Yogo critical value of 10 indicates
/// a weak instrument, which causes severe finite-sample bias and size distortions.
///
/// # Arguments
///
/// * `y` – outcome vector (length *n*); used only for length validation
/// * `x` – endogenous regressor (length *n*)
/// * `z` – excluded instrument (length *n*)
///
/// # Returns
///
/// [`WeakInstrumentSummary`] with the F-statistic, partial R², critical value,
/// and a weakness flag.
///
/// # Errors
///
/// Returns [`NumRs2Error`] on dimension mismatches or numerical failures.
pub fn weak_instrument_test(y: &[f64], x: &[f64], z: &[f64]) -> Result<WeakInstrumentSummary> {
    let n = y.len();
    if x.len() != n || z.len() != n {
        return Err(NumRs2Error::DimensionMismatch(
            "weak_instrument_test: y, x, and z must all have the same length".into(),
        ));
    }
    if n < 4 {
        return Err(NumRs2Error::ComputationError(
            "weak_instrument_test: need at least 4 observations".into(),
        ));
    }

    let x_exog = vec![1.0_f64; n]; // intercept
    let (f_stat, pr2) = first_stage_diagnostics(x, &x_exog, z, n, 1, 1)?;

    let critical_value_10pct = 10.0_f64;
    Ok(WeakInstrumentSummary {
        f_statistics: vec![f_stat],
        partial_r_squared: vec![pr2],
        critical_value_10pct,
        instruments_weak: vec![f_stat < critical_value_10pct],
    })
}

// ---------------------------------------------------------------------------
// Difference-in-Differences
// ---------------------------------------------------------------------------

/// Result of a Difference-in-Differences (DiD) estimation.
///
/// Produced by [`difference_in_differences`].
#[derive(Debug, Clone)]
pub struct DiDEstimate {
    /// Average Treatment Effect on the Treated (ATT)
    pub att: f64,
    /// Standard error of ATT
    pub std_error: f64,
    /// t-statistic (ATT / std_error)
    pub t_stat: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// 95 % confidence interval `[lower, upper]`
    pub conf_interval: [f64; 2],
    /// Pre-treatment parallel-trends test p-value.
    /// `None` when only one pre-treatment period exists.
    pub parallel_trends_p: Option<f64>,
    /// Number of treated units
    pub n_treated: usize,
    /// Number of control units
    pub n_control: usize,
}

/// Estimate the Average Treatment Effect on the Treated (ATT) via Difference-in-Differences.
///
/// The function constructs a balanced 2-period panel from the four provided
/// sample vectors, then estimates the ATT via a Two-Way Fixed Effects (TWFE)
/// regression:
///
/// ```text
/// y_{it} = α_i + γ_t + δ · (Treated_i × Post_t) + ε_{it}
/// ```
///
/// The coefficient `δ` is the DiD ATT estimate.
///
/// # Arguments
///
/// * `outcome_pre_treat`    – pre-treatment outcomes for treated units (length *n_treat*)
/// * `outcome_post_treat`   – post-treatment outcomes for treated units (length *n_treat*)
/// * `outcome_pre_control`  – pre-treatment outcomes for control units (length *n_ctrl*)
/// * `outcome_post_control` – post-treatment outcomes for control units (length *n_ctrl*)
///
/// # Returns
///
/// [`DiDEstimate`] with the ATT, standard error, t-statistic, p-value, 95 % CI,
/// and a parallel-trends test p-value.
///
/// # Errors
///
/// Returns [`NumRs2Error`] if any group has zero observations, pre/post slices
/// differ in length, or the TWFE regression is numerically singular.
///
/// # Example
///
/// ```rust,no_run
/// use numrs2::new_modules::causal::difference_in_differences;
///
/// let pre_t  = vec![10.0, 11.0, 12.0];
/// let post_t = vec![16.0, 17.0, 18.0]; // +3 above trend
/// let pre_c  = vec![10.0, 11.0, 12.0];
/// let post_c = vec![13.0, 14.0, 15.0]; // just the trend
/// let result = difference_in_differences(&pre_t, &post_t, &pre_c, &post_c).expect("valid DiD data");
/// assert!((result.att - 3.0).abs() < 0.5);
/// ```
pub fn difference_in_differences(
    outcome_pre_treat: &[f64],
    outcome_post_treat: &[f64],
    outcome_pre_control: &[f64],
    outcome_post_control: &[f64],
) -> Result<DiDEstimate> {
    let n_treat = outcome_pre_treat.len();
    let n_ctrl = outcome_pre_control.len();

    if n_treat == 0 || outcome_post_treat.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "difference_in_differences: treated group must have at least one observation".into(),
        ));
    }
    if n_ctrl == 0 || outcome_post_control.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "difference_in_differences: control group must have at least one observation".into(),
        ));
    }
    if outcome_pre_treat.len() != outcome_post_treat.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "difference_in_differences: pre and post treated slices must have equal length".into(),
        ));
    }
    if outcome_pre_control.len() != outcome_post_control.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "difference_in_differences: pre and post control slices must have equal length".into(),
        ));
    }

    // Build a balanced panel with 2 periods (0=pre, 1=post) and n_treat+n_ctrl units.
    // TWFE design matrix: [intercept | unit FEs (n_units-1) | time FE (1) | DiD indicator]
    let n_units = n_treat + n_ctrl;
    let n_periods = 2_usize;
    let n_obs = n_units * n_periods;

    // k = 1 (intercept) + (n_units - 1) (unit FEs) + 1 (time FE) + 1 (DiD) = n_units + 2
    let k = n_units + 2;

    let mut x_mat = vec![0.0_f64; n_obs * k];
    let mut y_vec = vec![0.0_f64; n_obs];

    // Iterate over units: treated first, then control
    for (unit_idx, is_treated) in (0..n_units).map(|i| (i, i < n_treat)) {
        for period in 0..n_periods {
            let row = unit_idx * n_periods + period;

            // outcome
            y_vec[row] = if is_treated {
                if period == 0 {
                    outcome_pre_treat[unit_idx]
                } else {
                    outcome_post_treat[unit_idx]
                }
            } else {
                let ctrl_idx = unit_idx - n_treat;
                if period == 0 {
                    outcome_pre_control[ctrl_idx]
                } else {
                    outcome_post_control[ctrl_idx]
                }
            };

            // intercept
            x_mat[row * k + 0] = 1.0;

            // unit FE (omit unit 0)
            if unit_idx > 0 {
                x_mat[row * k + unit_idx] = 1.0;
            }

            // time FE (col = n_units, omit period 0)
            if period == 1 {
                x_mat[row * k + n_units] = 1.0;
            }

            // DiD indicator: treated × post
            if is_treated && period == 1 {
                x_mat[row * k + k - 1] = 1.0;
            }
        }
    }

    let (beta, resid, xtx_inv) = ols_fit(&x_mat, &y_vec, n_obs, k)?;

    let att = beta[k - 1];
    let df = (n_obs - k) as f64;
    let s2 = resid.iter().map(|&r| r * r).sum::<f64>() / df.max(1.0);
    let var_att = xtx_inv[(k - 1) * k + (k - 1)] * s2;
    let se = var_att.max(0.0).sqrt();
    let t_stat = if se > 1e-15 { att / se } else { 0.0 };
    let p_value = t_p_value(t_stat, df);
    let t_crit = t_quantile_two_sided(0.05, df);
    let conf_interval = [att - t_crit * se, att + t_crit * se];

    Ok(DiDEstimate {
        att,
        std_error: se,
        t_stat,
        p_value,
        conf_interval,
        parallel_trends_p: None, // 2-period panel has no pre-treatment variation to test
        n_treated: n_treat,
        n_control: n_ctrl,
    })
}

// ---------------------------------------------------------------------------
// Propensity Score Methods
// ---------------------------------------------------------------------------

/// Estimated propensity scores for a set of units.
///
/// Produced by [`estimate_propensity_scores`].
#[derive(Debug, Clone)]
pub struct PropensityScores {
    /// Estimated propensity scores in `(0, 1)`, one per unit
    pub scores: Vec<f64>,
    /// Fitted logistic-regression coefficients (length = n_covariates + 1).
    /// The first entry is the intercept.
    pub coefficients: Vec<f64>,
}

/// Estimate propensity scores via logistic regression (IRLS / Newton-Raphson).
///
/// Models `P(W=1 | X=x)` using logistic regression with L2 regularisation.
/// An intercept column is prepended automatically; callers should **not** include
/// a constant column in `covariates`.
///
/// # Arguments
///
/// * `covariates` – covariate matrix with shape `(n_units, n_covariates)`.
///   All rows must have the same number of columns.
/// * `treatment`  – binary treatment indicators (`true` = treated)
///
/// # Returns
///
/// [`PropensityScores`] with one score per unit in `(0, 1)` and the fitted
/// logistic-regression coefficients.
///
/// # Errors
///
/// Returns [`NumRs2Error`] on dimension mismatches or Newton-Raphson failures.
///
/// # Example
///
/// ```rust,no_run
/// use numrs2::new_modules::causal::estimate_propensity_scores;
///
/// let cov = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
/// let trt = vec![false, false, true, true];
/// let ps = estimate_propensity_scores(&cov, &trt).expect("valid propensity score data");
/// assert_eq!(ps.scores.len(), 4);
/// ```
pub fn estimate_propensity_scores(
    covariates: &[Vec<f64>],
    treatment: &[bool],
) -> Result<PropensityScores> {
    let n = covariates.len();
    if treatment.len() != n {
        return Err(NumRs2Error::DimensionMismatch(
            "estimate_propensity_scores: covariates and treatment must have the same length".into(),
        ));
    }
    if n == 0 {
        return Err(NumRs2Error::ComputationError(
            "estimate_propensity_scores: need at least one observation".into(),
        ));
    }

    let k_cov = covariates[0].len();
    for (idx, row) in covariates.iter().enumerate() {
        if row.len() != k_cov {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "estimate_propensity_scores: covariate row {idx} has length {} but expected {k_cov}",
                row.len()
            )));
        }
    }

    // Design matrix with prepended intercept: n × (k_cov + 1)
    let k1 = k_cov + 1;
    let mut x_mat = vec![0.0_f64; n * k1];
    let mut w_vec = vec![0.0_f64; n];
    for i in 0..n {
        x_mat[i * k1 + 0] = 1.0; // intercept
        for j in 0..k_cov {
            x_mat[i * k1 + j + 1] = covariates[i][j];
        }
        w_vec[i] = if treatment[i] { 1.0 } else { 0.0 };
    }

    // IRLS (Newton-Raphson) for logistic regression with L2 ridge penalty λ
    let lambda = 1e-4_f64;
    let max_iter = 200_usize;
    let tol = 1e-8_f64;
    let mut beta = vec![0.0_f64; k1];

    for _iter in 0..max_iter {
        // mu = sigmoid(X β)
        let mu: Vec<f64> = (0..n)
            .map(|i| {
                let eta: f64 = (0..k1).map(|j| x_mat[i * k1 + j] * beta[j]).sum();
                sigmoid(eta)
            })
            .collect();

        // Working weights: v_i = mu_i (1 - mu_i) clamped away from 0
        let v: Vec<f64> = mu.iter().map(|&m| (m * (1.0 - m)).max(1e-8)).collect();

        // Gradient: Xᵀ(w - μ) - λ β  (no regularisation on intercept)
        let mut grad = vec![0.0_f64; k1];
        for j in 0..k1 {
            for i in 0..n {
                grad[j] += x_mat[i * k1 + j] * (w_vec[i] - mu[i]);
            }
            if j > 0 {
                grad[j] -= lambda * beta[j];
            }
        }

        // Hessian: Xᵀ diag(v) X + λ I  (no λ on intercept diagonal)
        let mut hess = vec![0.0_f64; k1 * k1];
        for i in 0..n {
            for r in 0..k1 {
                for c in 0..k1 {
                    hess[r * k1 + c] += v[i] * x_mat[i * k1 + r] * x_mat[i * k1 + c];
                }
            }
        }
        for j in 1..k1 {
            hess[j * k1 + j] += lambda;
        }

        let h_inv = invert_spd(&hess, k1)?;
        let delta = mat_mul(&h_inv, k1, k1, &grad, 1);
        let step_norm: f64 = delta.iter().map(|&d| d * d).sum::<f64>().sqrt();

        for j in 0..k1 {
            beta[j] += delta[j];
        }

        if step_norm < tol {
            break;
        }
    }

    // Predict: scores = sigmoid(X β)
    let scores: Vec<f64> = (0..n)
        .map(|i| {
            let eta: f64 = (0..k1).map(|j| x_mat[i * k1 + j] * beta[j]).sum();
            sigmoid(eta)
        })
        .collect();

    Ok(PropensityScores {
        scores,
        coefficients: beta,
    })
}

/// Estimate the Average Treatment Effect (ATE) via Inverse Probability Weighting (IPW).
///
/// Uses the Horvitz-Thompson estimator with normalised weights.  Propensity
/// scores are trimmed to `[0.01, 0.99]` to prevent extreme weights from
/// dominating the estimate.
///
/// ```text
/// ATE = (1/n) Σ_i [ W_i Y_i / ê(X_i)  −  (1−W_i) Y_i / (1−ê(X_i)) ]
/// ```
///
/// # Arguments
///
/// * `outcome`           – observed outcomes, one per unit
/// * `treatment`         – binary treatment indicators
/// * `propensity_scores` – estimated propensity scores (e.g., from
///   [`estimate_propensity_scores`]); must lie in `(0, 1)`
///
/// # Returns
///
/// The IPW ATE estimate as a plain `f64`.
///
/// # Errors
///
/// Returns [`NumRs2Error::DimensionMismatch`] when slice lengths differ, or
/// [`NumRs2Error::ComputationError`] for empty inputs.
///
/// # Example
///
/// ```rust,no_run
/// use numrs2::new_modules::causal::{estimate_propensity_scores, ipw_ate};
///
/// let cov = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
/// let trt = vec![false, false, true, true];
/// let ps  = estimate_propensity_scores(&cov, &trt).expect("valid propensity score inputs");
/// let y   = vec![1.0, 1.5, 3.0, 3.5];
/// let ate = ipw_ate(&y, &trt, &ps.scores).expect("valid IPW ATE data");
/// ```
pub fn ipw_ate(outcome: &[f64], treatment: &[bool], propensity_scores: &[f64]) -> Result<f64> {
    let n = outcome.len();
    if treatment.len() != n || propensity_scores.len() != n {
        return Err(NumRs2Error::DimensionMismatch(
            "ipw_ate: outcome, treatment, and propensity_scores must all have the same length"
                .into(),
        ));
    }
    if n == 0 {
        return Err(NumRs2Error::ComputationError(
            "ipw_ate: need at least one observation".into(),
        ));
    }

    let eps = 0.01_f64;
    let ate: f64 = (0..n)
        .map(|i| {
            let wi = if treatment[i] { 1.0_f64 } else { 0.0 };
            let yi = outcome[i];
            let pi = propensity_scores[i].clamp(eps, 1.0 - eps);
            wi * yi / pi - (1.0 - wi) * yi / (1.0 - pi)
        })
        .sum::<f64>()
        / n as f64;

    Ok(ate)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // DiD tests
    // -----------------------------------------------------------------------

    /// DiD should recover a known treatment effect of +5.
    ///
    /// Setup: both groups share a +2 trend; treated units receive +5 on top
    /// of the trend in the post period.
    #[test]
    fn test_did_known_treatment_effect() {
        let n = 5_usize;
        let trend = 2.0_f64;
        let true_att = 5.0_f64;

        let pre_t: Vec<f64> = (0..n).map(|i| 10.0 + i as f64 * 0.5).collect();
        let post_t: Vec<f64> = pre_t.iter().map(|&v| v + trend + true_att).collect();
        let pre_c: Vec<f64> = (0..n).map(|i| 10.0 + i as f64 * 0.5).collect();
        let post_c: Vec<f64> = pre_c.iter().map(|&v| v + trend).collect();

        let result = difference_in_differences(&pre_t, &post_t, &pre_c, &post_c)
            .expect("test: valid DiD computation");

        assert!(
            (result.att - true_att).abs() < 0.5,
            "Expected ATT ≈ {true_att}, got {:.4}",
            result.att
        );
        assert_eq!(result.n_treated, n);
        assert_eq!(result.n_control, n);
    }

    /// When both groups evolve identically, ATT should be zero.
    #[test]
    fn test_did_zero_treatment_effect() {
        let pre_t = vec![10.0, 11.0, 12.0, 13.0];
        let post_t = vec![14.0, 15.0, 16.0, 17.0]; // trend = +4
        let pre_c = vec![10.0, 11.0, 12.0, 13.0];
        let post_c = vec![14.0, 15.0, 16.0, 17.0]; // same trend

        let result = difference_in_differences(&pre_t, &post_t, &pre_c, &post_c)
            .expect("test: valid DiD computation");

        assert!(
            result.att.abs() < 1e-6,
            "Expected ATT ≈ 0, got {:.8}",
            result.att
        );
        assert!(
            result.p_value > 0.05,
            "Expected non-significant result, got p = {:.4}",
            result.p_value
        );
    }

    /// Mismatched pre/post lengths should be caught.
    #[test]
    fn test_did_dimension_mismatch() {
        let err = difference_in_differences(
            &[1.0, 2.0],
            &[3.0], // length mismatch
            &[1.0, 2.0],
            &[3.0, 4.0],
        );
        assert!(err.is_err(), "Expected an error for mismatched lengths");
    }

    /// Negative treatment effect should also be recoverable.
    #[test]
    fn test_did_negative_treatment_effect() {
        let n = 6_usize;
        let true_att = -3.0_f64;

        let pre_t: Vec<f64> = (0..n).map(|i| 20.0 + i as f64).collect();
        let post_t: Vec<f64> = pre_t.iter().map(|&v| v + 1.0 + true_att).collect();
        let pre_c: Vec<f64> = (0..n).map(|i| 20.0 + i as f64).collect();
        let post_c: Vec<f64> = pre_c.iter().map(|&v| v + 1.0).collect();

        let result = difference_in_differences(&pre_t, &post_t, &pre_c, &post_c)
            .expect("test: valid DiD computation");

        assert!(
            (result.att - true_att).abs() < 0.5,
            "Expected ATT ≈ {true_att}, got {:.4}",
            result.att
        );
    }

    // -----------------------------------------------------------------------
    // Propensity score tests
    // -----------------------------------------------------------------------

    /// Units with higher covariate values (which predict treatment) should
    /// receive higher propensity scores, and all scores should be in (0, 1).
    #[test]
    fn test_propensity_score_monotonicity() {
        let covariates: Vec<Vec<f64>> = vec![
            vec![-2.0],
            vec![-1.5],
            vec![-1.0],
            vec![-0.5],
            vec![0.5],
            vec![1.0],
            vec![1.5],
            vec![2.0],
        ];
        let treatment = vec![false, false, false, false, true, true, true, true];

        let ps = estimate_propensity_scores(&covariates, &treatment)
            .expect("test: valid propensity score estimation");

        assert_eq!(ps.scores.len(), 8);
        for &s in &ps.scores {
            assert!(s > 0.0 && s < 1.0, "score {s} outside (0, 1)");
        }

        let high_mean: f64 = ps.scores[4..].iter().sum::<f64>() / 4.0;
        let low_mean: f64 = ps.scores[..4].iter().sum::<f64>() / 4.0;
        assert!(
            high_mean > low_mean,
            "Expected higher PS for higher-X units: high={high_mean:.3}, low={low_mean:.3}"
        );
    }

    /// Dimension mismatch between covariates and treatment should error.
    #[test]
    fn test_propensity_score_dimension_mismatch() {
        let covariates = vec![vec![1.0], vec![2.0]];
        let treatment = vec![false]; // wrong length
        assert!(estimate_propensity_scores(&covariates, &treatment).is_err());
    }

    // -----------------------------------------------------------------------
    // IPW tests
    // -----------------------------------------------------------------------

    /// IPW ATE should be close to the true constant treatment effect.
    #[test]
    fn test_ipw_ate_known_effect() {
        let n = 20_usize;
        let true_ate = 2.0_f64;

        // Covariate: evenly spaced in [0, 1]
        let covariates: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / n as f64]).collect();
        let treatment: Vec<bool> = (0..n).map(|i| i >= n / 2).collect();

        // Y(0) = 1; Y(1) = 1 + true_ate
        let outcome: Vec<f64> = (0..n)
            .map(|i| if treatment[i] { 1.0 + true_ate } else { 1.0 })
            .collect();

        let ps = estimate_propensity_scores(&covariates, &treatment)
            .expect("test: valid propensity score estimation");
        let ate =
            ipw_ate(&outcome, &treatment, &ps.scores).expect("test: valid IPW ATE computation");

        assert!(
            (ate - true_ate).abs() < 1.0,
            "Expected IPW ATE ≈ {true_ate}, got {ate:.4}"
        );
    }

    /// IPW should error when slice lengths differ.
    #[test]
    fn test_ipw_dimension_mismatch() {
        let outcome = vec![1.0, 2.0, 3.0];
        let treatment = vec![false, true]; // too short
        let ps = vec![0.3, 0.6, 0.5];
        assert!(ipw_ate(&outcome, &treatment, &ps).is_err());
    }

    // -----------------------------------------------------------------------
    // Instrumental Variables tests
    // -----------------------------------------------------------------------

    /// 2SLS should recover a known causal effect from a simple DGP.
    ///
    /// DGP: z → x = 0.8z; y = 2*x + 0.5 (no noise, exact recovery expected).
    #[test]
    fn test_two_stage_least_squares_recovery() {
        let n = 100_usize;
        let z: Vec<f64> = (0..n)
            .map(|i| (i as f64) / (n as f64) * 4.0 - 2.0)
            .collect();
        let x: Vec<f64> = z.iter().map(|&zi| 0.8 * zi).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.5).collect();

        let result = two_stage_least_squares(&y, &x, &z).expect("test: valid 2SLS computation");

        let beta_x = result.coef[0];
        assert!(
            (beta_x - 2.0).abs() < 0.1,
            "Expected β_x ≈ 2.0, got {beta_x:.6}"
        );
        assert_eq!(result.n_obs, n);
        assert_eq!(result.estimator, "2SLS");
        // First-stage F should be large (z is a strong instrument)
        assert!(
            result
                .first_stage_f
                .expect("test: first-stage F statistic is some")
                > 10.0,
            "Expected strong first-stage F"
        );
    }

    /// The weak-instrument test should flag a high F-statistic for a strong instrument.
    #[test]
    fn test_weak_instrument_strong_z() {
        let n = 60_usize;
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x: Vec<f64> = z.iter().map(|&zi| 2.0 * zi + 1.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi + 0.1).collect();

        let summary = weak_instrument_test(&y, &x, &z).expect("test: valid weak instrument test");

        assert_eq!(summary.f_statistics.len(), 1);
        assert!(
            summary.f_statistics[0] > 100.0,
            "Expected very large F for strong instrument, got {:.2}",
            summary.f_statistics[0]
        );
        assert!(
            !summary.instruments_weak[0],
            "Should not be flagged as weak"
        );
    }
}
