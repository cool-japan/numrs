//! Bayesian Optimization with Gaussian Process Surrogate
//!
//! Implements Bayesian Optimization for black-box function minimization using a
//! GP surrogate model with RBF or Matern 5/2 kernels. Supports EI, PI, UCB, and
//! Thompson Sampling acquisition functions with hyperparameter optimization via
//! marginal log-likelihood.
//!
//! # Example
//!
//! ```no_run
//! use numrs2::optimize::bayesian_opt::*;
//!
//! let f = |x: &[f64]| -> f64 { x.iter().map(|xi| (xi - 0.5).powi(2)).sum() };
//! let config = BayesOptConfig {
//!     bounds: vec![(0.0, 1.0), (0.0, 1.0)],
//!     n_initial: 5, n_iterations: 20,
//!     ..Default::default()
//! };
//! let result = bayesian_optimize(f, config).expect("optimization should succeed");
//! assert!(result.f_best < 0.1);
//! ```

use crate::error::{NumRs2Error, Result};
use scirs2_core::random::{thread_rng, Distribution, Normal, Rng, Uniform};
use std::f64::consts::PI;

/// Kernel function type for the GP surrogate model.
#[derive(Debug, Clone, Default)]
pub enum KernelType {
    /// Squared Exponential (RBF): k(x,x') = sigma^2 exp(-||x-x'||^2 / (2l^2))
    SquaredExponential,
    /// Matern 5/2: k(x,x') = sigma^2 (1+sqrt(5)r/l+5r^2/(3l^2)) exp(-sqrt(5)r/l)
    #[default]
    Matern52,
}

/// Internal kernel representation with hyperparameters.
#[derive(Debug, Clone)]
struct Kernel {
    /// The kernel type
    kernel_type: KernelType,
    /// Length scale parameter (l)
    length_scale: f64,
    /// Signal variance parameter (sigma^2)
    signal_variance: f64,
}

impl Kernel {
    /// Create a new kernel with the given type and default hyperparameters.
    fn new(kernel_type: KernelType) -> Self {
        Self {
            kernel_type,
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }

    /// Compute the kernel value between two points.
    fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist = squared_euclidean_distance(x1, x2);

        match self.kernel_type {
            KernelType::SquaredExponential => {
                let l2 = self.length_scale * self.length_scale;
                self.signal_variance * (-sq_dist / (2.0 * l2)).exp()
            }
            KernelType::Matern52 => {
                let r = sq_dist.sqrt();
                let sqrt5_r_l = 5.0_f64.sqrt() * r / self.length_scale;
                let sq_term = 5.0 * sq_dist / (3.0 * self.length_scale * self.length_scale);
                self.signal_variance * (1.0 + sqrt5_r_l + sq_term) * (-sqrt5_r_l).exp()
            }
        }
    }

    /// Compute the full kernel matrix K for a set of training points.
    fn compute_matrix(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            k[i][i] = self.signal_variance;
            for j in (i + 1)..n {
                let val = self.compute(&x[i], &x[j]);
                k[i][j] = val;
                k[j][i] = val;
            }
        }
        k
    }

    /// Compute the kernel vector k* between a new point and training points.
    fn compute_vector(&self, x_new: &[f64], x_train: &[Vec<f64>]) -> Vec<f64> {
        x_train.iter().map(|xi| self.compute(x_new, xi)).collect()
    }

    /// Create a copy with updated hyperparameters.
    fn with_params(&self, length_scale: f64, signal_variance: f64) -> Self {
        Self {
            kernel_type: self.kernel_type.clone(),
            length_scale,
            signal_variance,
        }
    }
}

/// Compute squared Euclidean distance between two vectors.
fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Gaussian Process (GP) surrogate model for function approximation.
///
/// Provides posterior mean and variance predictions at query points using
/// Cholesky-based computations. Supports hyperparameter optimization via NLML.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Kernel function with hyperparameters
    kernel: Kernel,
    /// Training inputs (each inner Vec is one data point)
    x_train: Vec<Vec<f64>>,
    /// Training outputs (normalized)
    y_train: Vec<f64>,
    /// Noise variance for observations
    noise_variance: f64,
    /// Cholesky factor L of (K + sigma_n^2 I)
    cholesky_l: Vec<Vec<f64>>,
    /// Pre-computed alpha = L^T \ (L \ y) for predictions
    alpha: Vec<f64>,
    /// Mean of raw training outputs (for de-normalization)
    y_mean: f64,
    /// Standard deviation of raw training outputs (for de-normalization)
    y_std: f64,
}

impl GaussianProcess {
    /// Create a new unfitted Gaussian Process with the given kernel and noise variance.
    pub fn new(kernel_type: KernelType, noise_variance: f64) -> Self {
        Self {
            kernel: Kernel::new(kernel_type),
            x_train: Vec::new(),
            y_train: Vec::new(),
            noise_variance,
            cholesky_l: Vec::new(),
            alpha: Vec::new(),
            y_mean: 0.0,
            y_std: 1.0,
        }
    }

    /// Fit the GP model to training data (normalizes, computes kernel, Cholesky, alpha).
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(NumRs2Error::InvalidInput(
                "Training data cannot be empty".to_string(),
            ));
        }
        if x.len() != y.len() {
            return Err(NumRs2Error::InvalidInput(format!(
                "x and y must have same length: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        self.x_train = x.to_vec();

        // Normalize y for numerical stability
        self.y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let y_var = y.iter().map(|&yi| (yi - self.y_mean).powi(2)).sum::<f64>() / y.len() as f64;
        self.y_std = if y_var > 1e-12 { y_var.sqrt() } else { 1.0 };

        self.y_train = y
            .iter()
            .map(|&yi| (yi - self.y_mean) / self.y_std)
            .collect();

        // Compute kernel matrix K
        let mut k = self.kernel.compute_matrix(&self.x_train);
        let n = k.len();

        // Add noise to diagonal: K + (sigma_n^2 / y_std^2) * I + jitter
        let noise = self.noise_variance / (self.y_std * self.y_std);
        for i in 0..n {
            k[i][i] += noise + 1e-8;
        }

        // Cholesky decomposition: K = L * L^T
        self.cholesky_l = cholesky_decomposition(&k)?;

        // Solve for alpha = L^T \ (L \ y_normalized)
        let z = forward_substitution(&self.cholesky_l, &self.y_train)?;
        self.alpha = backward_substitution_transpose(&self.cholesky_l, &z)?;

        Ok(())
    }

    /// Predict the posterior mean and variance at a new point (de-normalized).
    pub fn predict(&self, x_new: &[f64]) -> Result<(f64, f64)> {
        if self.x_train.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "GP model has not been fitted yet".to_string(),
            ));
        }

        let k_star = self.kernel.compute_vector(x_new, &self.x_train);

        // Mean: mu = k*^T alpha (then de-normalize)
        let mu_norm: f64 = k_star
            .iter()
            .zip(self.alpha.iter())
            .map(|(&ki, &ai)| ki * ai)
            .sum();
        let mu = mu_norm * self.y_std + self.y_mean;

        // Variance: var = k(x,x) - v^T v where v = L \ k*
        let k_self = self.kernel.signal_variance;
        let v = forward_substitution(&self.cholesky_l, &k_star)?;
        let v_sq: f64 = v.iter().map(|&vi| vi * vi).sum();
        let var_norm = (k_self - v_sq).max(1e-12);
        let var = var_norm * self.y_std * self.y_std;

        Ok((mu, var))
    }

    /// Predict mean and variance for multiple points.
    pub fn predict_batch(&self, x_new: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut means = Vec::with_capacity(x_new.len());
        let mut variances = Vec::with_capacity(x_new.len());
        for x in x_new {
            let (mu, var) = self.predict(x)?;
            means.push(mu);
            variances.push(var);
        }
        Ok((means, variances))
    }

    /// Compute the negative log marginal likelihood (for hyperparameter optimization).
    pub fn negative_log_marginal_likelihood(&self) -> Result<f64> {
        if self.x_train.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "GP model has not been fitted yet".to_string(),
            ));
        }

        let n = self.y_train.len() as f64;

        // Data fit term: 0.5 * y^T alpha
        let data_fit: f64 = self
            .y_train
            .iter()
            .zip(self.alpha.iter())
            .map(|(&yi, &ai)| yi * ai)
            .sum();

        // Complexity penalty: sum of log of diagonal of L
        let log_det: f64 = self
            .cholesky_l
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].ln())
            .sum();

        // Constant term
        let constant = 0.5 * n * (2.0 * PI).ln();

        Ok(0.5 * data_fit + log_det + constant)
    }

    /// Optimize kernel hyperparameters via multi-start coordinate descent on NLML.
    pub fn optimize_hyperparameters(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        n_restarts: usize,
    ) -> Result<()> {
        let mut rng = thread_rng();
        let log_dist = Uniform::new(-2.0_f64, 2.0_f64).map_err(|e| {
            NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e))
        })?;

        let mut best_nll = f64::INFINITY;
        let mut best_ls = self.kernel.length_scale;
        let mut best_sv = self.kernel.signal_variance;

        for restart in 0..n_restarts {
            let (init_ls, init_sv) = if restart == 0 {
                (self.kernel.length_scale, self.kernel.signal_variance)
            } else {
                let log_ls: f64 = log_dist.sample(&mut rng);
                let log_sv: f64 = log_dist.sample(&mut rng);
                (log_ls.exp(), log_sv.exp())
            };

            let refinement = self.refine_hyperparameters(x, y, init_ls, init_sv)?;

            if refinement.0 < best_nll {
                best_nll = refinement.0;
                best_ls = refinement.1;
                best_sv = refinement.2;
            }
        }

        self.kernel = self.kernel.with_params(best_ls, best_sv);
        self.fit(x, y)?;

        Ok(())
    }

    /// Refine hyperparameters via coordinate descent with multiplicative steps.
    fn refine_hyperparameters(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        init_ls: f64,
        init_sv: f64,
    ) -> Result<(f64, f64, f64)> {
        let mut current_ls = init_ls;
        let mut current_sv = init_sv;

        self.kernel = self.kernel.with_params(current_ls, current_sv);
        self.fit(x, y)?;
        let mut current_nll = self.negative_log_marginal_likelihood()?;

        let factors = [0.5, 0.8, 1.0, 1.25, 2.0];

        for _ in 0..5 {
            // Optimize length scale
            for &factor in &factors {
                let trial_ls = (current_ls * factor).clamp(1e-4, 100.0);
                self.kernel = self.kernel.with_params(trial_ls, current_sv);
                if self.fit(x, y).is_ok() {
                    if let Ok(nll) = self.negative_log_marginal_likelihood() {
                        if nll < current_nll {
                            current_nll = nll;
                            current_ls = trial_ls;
                        }
                    }
                }
            }

            // Optimize signal variance
            for &factor in &factors {
                let trial_sv = (current_sv * factor).clamp(1e-4, 100.0);
                self.kernel = self.kernel.with_params(current_ls, trial_sv);
                if self.fit(x, y).is_ok() {
                    if let Ok(nll) = self.negative_log_marginal_likelihood() {
                        if nll < current_nll {
                            current_nll = nll;
                            current_sv = trial_sv;
                        }
                    }
                }
            }
        }

        Ok((current_nll, current_ls, current_sv))
    }

    /// Sample from the GP posterior at the given points (for Thompson Sampling).
    pub fn sample_posterior(&self, x_points: &[Vec<f64>], rng: &mut impl Rng) -> Result<Vec<f64>> {
        let (means, variances) = self.predict_batch(x_points)?;

        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            NumRs2Error::ComputationError(format!("Failed to create Normal distribution: {}", e))
        })?;

        let samples: Vec<f64> = means
            .iter()
            .zip(variances.iter())
            .map(|(&mu, &var)| {
                let z: f64 = normal.sample(rng);
                mu + z * var.sqrt()
            })
            .collect();

        Ok(samples)
    }

    /// Get the current kernel type.
    pub fn kernel_type(&self) -> &KernelType {
        &self.kernel.kernel_type
    }

    /// Get the current length scale.
    pub fn length_scale(&self) -> f64 {
        self.kernel.length_scale
    }

    /// Get the current signal variance.
    pub fn signal_variance(&self) -> f64 {
        self.kernel.signal_variance
    }

    /// Get the number of training points.
    pub fn n_train(&self) -> usize {
        self.x_train.len()
    }
}

/// Cholesky decomposition: A = L * L^T. Returns lower-triangular L.
fn cholesky_decomposition(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Matrix cannot be empty".to_string(),
        ));
    }

    let mut l = vec![vec![0.0; n]; n];

    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j][k] * l[j][k];
        }
        let diag = a[j][j] - sum;
        if diag <= 0.0 {
            return Err(NumRs2Error::NumericalError(format!(
                "Matrix is not positive definite at index {} (diag = {:.6e})",
                j, diag
            )));
        }
        l[j][j] = diag.sqrt();

        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            l[i][j] = (a[i][j] - s) / l[j][j];
        }
    }

    Ok(l)
}

/// Forward substitution: solve L x = b where L is lower triangular.
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = l.len();
    let mut x = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        if l[i][i].abs() < 1e-15 {
            return Err(NumRs2Error::NumericalError(
                "Singular matrix in forward substitution".to_string(),
            ));
        }
        x[i] = (b[i] - sum) / l[i][i];
    }

    Ok(x)
}

/// Backward substitution: solve L^T x = b where L is lower triangular.
fn backward_substitution_transpose(l: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = l.len();
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j];
        }
        if l[i][i].abs() < 1e-15 {
            return Err(NumRs2Error::NumericalError(
                "Singular matrix in backward substitution".to_string(),
            ));
        }
        x[i] = (b[i] - sum) / l[i][i];
    }

    Ok(x)
}

/// Acquisition function type for guiding optimization.
#[derive(Debug, Clone, Default)]
pub enum AcquisitionType {
    /// Expected Improvement: EI(x) = E[max(f_best - f(x), 0)] (uses xi param)
    #[default]
    ExpectedImprovement,
    /// Probability of Improvement: PI(x) = P(f(x) < f_best - xi) (uses xi param)
    ProbabilityOfImprovement,
    /// Upper Confidence Bound: UCB(x) = -(mu(x) - kappa * sigma(x)) (uses kappa param)
    UpperConfidenceBound,
    /// Thompson Sampling: sample from posterior and minimize the sample
    ThompsonSampling,
}

/// Evaluate the acquisition function at a point given GP predictions.
///
/// Returns higher values for more promising points (to be maximized).
fn compute_acquisition(
    acq: &AcquisitionType,
    mu: f64,
    sigma: f64,
    best_y: f64,
    xi: f64,
    kappa: f64,
) -> f64 {
    if sigma < 1e-12 {
        return 0.0;
    }

    match acq {
        AcquisitionType::ExpectedImprovement => {
            let improvement = best_y - mu - xi;
            let z = improvement / sigma;
            improvement * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)
        }
        AcquisitionType::ProbabilityOfImprovement => {
            let z = (best_y - mu - xi) / sigma;
            standard_normal_cdf(z)
        }
        AcquisitionType::UpperConfidenceBound => {
            // For minimization: lower mu is better; higher sigma = more exploration
            // Return negative LCB so higher = more promising
            -(mu - kappa * sigma)
        }
        AcquisitionType::ThompsonSampling => {
            // Handled separately via posterior sampling
            0.0
        }
    }
}

/// Standard normal PDF: phi(z) = exp(-z^2/2) / sqrt(2*pi).
fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let inner = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    sign * (1.0 - inner * (-x * x).exp())
}

/// Configuration for Bayesian Optimization.
#[derive(Debug, Clone)]
pub struct BayesOptConfig {
    /// Number of initial random samples (Latin Hypercube design).
    pub n_initial: usize,

    /// Number of optimization iterations after the initial phase.
    pub n_iterations: usize,

    /// Search space bounds for each dimension: Vec<(lower, upper)>.
    pub bounds: Vec<(f64, f64)>,

    /// Acquisition function type.
    pub acquisition: AcquisitionType,

    /// Kernel type for the GP surrogate.
    pub kernel: KernelType,

    /// Optional RNG seed for reproducibility (currently uses thread_rng).
    pub seed: Option<u64>,

    /// Exploration-exploitation tradeoff for EI and PI (default: 0.01).
    pub xi: f64,
    /// UCB exploration parameter (default: 2.0).
    pub kappa: f64,

    /// Noise variance assumed for observations.
    pub noise_variance: f64,

    /// Number of random candidates for acquisition optimization.
    pub n_candidates: usize,

    /// Number of restarts for hyperparameter optimization.
    pub n_hp_restarts: usize,

    /// Whether to optimize GP hyperparameters during the run.
    pub optimize_hyperparams: bool,

    /// Frequency of hyperparameter optimization (every N iterations).
    pub hp_optimize_interval: usize,
}

impl Default for BayesOptConfig {
    fn default() -> Self {
        Self {
            n_initial: 5,
            n_iterations: 25,
            bounds: vec![(0.0, 1.0)],
            acquisition: AcquisitionType::default(),
            kernel: KernelType::default(),
            seed: None,
            xi: 0.01,
            kappa: 2.0,
            noise_variance: 1e-6,
            n_candidates: 1000,
            n_hp_restarts: 3,
            optimize_hyperparams: true,
            hp_optimize_interval: 5,
        }
    }
}

/// Result of Bayesian Optimization.
#[derive(Debug, Clone)]
pub struct BayesOptResult {
    /// Best found solution (parameter vector).
    pub x_best: Vec<f64>,

    /// Best objective value found.
    pub f_best: f64,

    /// History of all evaluated points.
    pub x_history: Vec<Vec<f64>>,

    /// History of all objective values.
    pub f_history: Vec<f64>,

    /// Total number of function evaluations.
    pub n_evaluations: usize,

    /// Whether optimization improved over the initial samples.
    pub success: bool,
    /// Final GP model for inspection.
    pub model: GaussianProcess,
}

/// Generate a Latin Hypercube Sampling (LHS) design with space-filling properties.
fn latin_hypercube_sampling(
    n_samples: usize,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> Result<Vec<Vec<f64>>> {
    let n_dims = bounds.len();
    if n_dims == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Bounds cannot be empty".to_string(),
        ));
    }
    if n_samples == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Number of samples must be positive".to_string(),
        ));
    }

    let uniform = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e)))?;

    let mut samples = vec![vec![0.0; n_dims]; n_samples];

    for dim in 0..n_dims {
        // Fisher-Yates shuffle to create a random permutation
        let mut perm: Vec<usize> = (0..n_samples).collect();
        for i in (1..n_samples).rev() {
            let j_f64: f64 = uniform.sample(rng);
            let j = ((j_f64 * (i + 1) as f64) as usize).min(i);
            perm.swap(i, j);
        }

        let (low, high) = bounds[dim];
        let step = (high - low) / n_samples as f64;

        for (i, &p) in perm.iter().enumerate() {
            let u: f64 = uniform.sample(rng);
            samples[i][dim] = low + (p as f64 + u) * step;
        }
    }

    Ok(samples)
}

/// Generate uniform random samples within bounds.
fn random_sampling(
    n_samples: usize,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> Result<Vec<Vec<f64>>> {
    let n_dims = bounds.len();
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut point = Vec::with_capacity(n_dims);
        for &(low, high) in bounds {
            let dist = Uniform::new(low, high).map_err(|e| {
                NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e))
            })?;
            let val: f64 = dist.sample(rng);
            point.push(val);
        }
        samples.push(point);
    }

    Ok(samples)
}

/// Optimize the acquisition function via random search + local refinement.
fn optimize_acquisition(
    gp: &GaussianProcess,
    acq: &AcquisitionType,
    bounds: &[(f64, f64)],
    best_y: f64,
    xi: f64,
    kappa: f64,
    n_candidates: usize,
    rng: &mut impl Rng,
) -> Result<Vec<f64>> {
    match acq {
        AcquisitionType::ThompsonSampling => {
            let candidates = random_sampling(n_candidates, bounds, rng)?;
            let posterior_samples = gp.sample_posterior(&candidates, rng)?;

            let mut best_idx = 0;
            let mut best_val = f64::INFINITY;
            for (i, &val) in posterior_samples.iter().enumerate() {
                if val < best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            Ok(candidates[best_idx].clone())
        }
        _ => {
            // Phase 1: Random search
            let candidates = random_sampling(n_candidates, bounds, rng)?;
            let mut best_acq = f64::NEG_INFINITY;
            let mut best_point = candidates[0].clone();

            for candidate in &candidates {
                let (mu, var) = gp.predict(candidate)?;
                let sigma = var.sqrt();
                let acq_val = compute_acquisition(acq, mu, sigma, best_y, xi, kappa);
                if acq_val > best_acq {
                    best_acq = acq_val;
                    best_point = candidate.clone();
                }
            }

            // Phase 2: Local refinement around best candidate
            let n_refine = 50;
            for _ in 0..n_refine {
                let mut perturbed = best_point.clone();
                for (d, &(low, high)) in bounds.iter().enumerate() {
                    let range = (high - low) * 0.05;
                    let pert_dist = Uniform::new(-range, range).map_err(|e| {
                        NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e))
                    })?;
                    let delta: f64 = pert_dist.sample(rng);
                    perturbed[d] = (perturbed[d] + delta).max(low).min(high);
                }

                let (mu, var) = gp.predict(&perturbed)?;
                let sigma = var.sqrt();
                let acq_val = compute_acquisition(acq, mu, sigma, best_y, xi, kappa);
                if acq_val > best_acq {
                    best_acq = acq_val;
                    best_point = perturbed;
                }
            }

            Ok(best_point)
        }
    }
}

/// Run Bayesian Optimization to minimize the given objective function.
///
/// Phase 1: Evaluates `n_initial` points via LHS. Phase 2: Iteratively fits GP,
/// optimizes acquisition function, evaluates objective at suggested point.
///
/// # Errors
///
/// Besides the up-front `config.bounds` validation, this function
/// propagates (does not swallow) failures from the GP fitting machinery,
/// since continuing past one would otherwise hand back a
/// [`BayesOptResult`] whose `model` is stale or was never actually fitted:
///
/// - Every per-iteration [`GaussianProcess::fit`] call (required before
///   that iteration's acquisition step) and the final re-fit after the
///   loop (so `result.model` matches the returned `result.x_history`/
///   `result.f_history`) return `Err` on a non-positive-definite
///   covariance matrix — e.g. an invalid (negative) `noise_variance`, or
///   otherwise numerically degenerate training data.
/// - When `config.optimize_hyperparams` is enabled (the default), each
///   [`GaussianProcess::optimize_hyperparameters`] call at the configured
///   `hp_optimize_interval` can itself fail if a randomly-drawn restart
///   hyperparameter combination produces a degenerate covariance; such a
///   failure aborts the run rather than risking a corrupted GP (mutated
///   kernel, stale cached Cholesky factor) silently feeding the next
///   iteration's predictions.
pub fn bayesian_optimize<F>(objective: F, config: BayesOptConfig) -> Result<BayesOptResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n_dims = config.bounds.len();
    if n_dims == 0 {
        return Err(NumRs2Error::InvalidInput(
            "Bounds cannot be empty".to_string(),
        ));
    }

    // Validate bounds
    for (i, &(low, high)) in config.bounds.iter().enumerate() {
        if low >= high {
            return Err(NumRs2Error::InvalidInput(format!(
                "Invalid bounds for dimension {}: lower ({}) must be less than upper ({})",
                i, low, high
            )));
        }
    }

    // Note: scirs2_core thread_rng does not support seeding directly;
    // we use thread_rng regardless. The seed field is reserved for future use.
    let mut rng = thread_rng();

    // Initialize GP model
    let mut gp = GaussianProcess::new(config.kernel.clone(), config.noise_variance);

    // Phase 1: Initial sampling via LHS
    let initial_points = latin_hypercube_sampling(config.n_initial, &config.bounds, &mut rng)?;

    let mut x_history: Vec<Vec<f64>> = Vec::new();
    let mut f_history: Vec<f64> = Vec::new();
    let mut f_best = f64::INFINITY;
    let mut x_best = initial_points[0].clone();

    for point in &initial_points {
        let value = objective(point);
        x_history.push(point.clone());
        f_history.push(value);

        if value < f_best {
            f_best = value;
            x_best = point.clone();
        }
    }

    let initial_best = f_best;

    // Phase 2: Bayesian Optimization loop
    for iteration in 0..config.n_iterations {
        // Fit GP model to all data collected so far
        gp.fit(&x_history, &f_history)?;

        // Optionally optimize GP hyperparameters.
        //
        // Policy: propagate failures instead of swallowing them. A failure
        // here (e.g. a numerically degenerate trial hyperparameter
        // combination) can leave `gp`'s kernel and its cached Cholesky
        // factor/alpha mutually inconsistent (`optimize_hyperparameters`
        // mutates `gp.kernel` before re-fitting; if that re-fit fails, the
        // kernel no longer matches the stale `cholesky_l`/`alpha` from the
        // prior successful fit). Continuing on such a discarded error would
        // let `predict`/the acquisition step silently use a corrupted GP
        // instead of surfacing the problem, so we bail out via `?` rather
        // than risk that.
        if config.optimize_hyperparams
            && config.hp_optimize_interval > 0
            && iteration % config.hp_optimize_interval == 0
        {
            gp.optimize_hyperparameters(&x_history, &f_history, config.n_hp_restarts)?;
        }

        // Find the next point by maximizing the acquisition function
        let next_point = optimize_acquisition(
            &gp,
            &config.acquisition,
            &config.bounds,
            f_best,
            config.xi,
            config.kappa,
            config.n_candidates,
            &mut rng,
        )?;

        // Evaluate objective at the next point
        let value = objective(&next_point);
        x_history.push(next_point.clone());
        f_history.push(value);

        if value < f_best {
            f_best = value;
            x_best = next_point;
        }
    }

    // Final fit for the returned model.
    //
    // Policy: propagate failures instead of swallowing them. This fit is
    // what makes `result.model` consistent with the full, final
    // `x_history`/`f_history` we are about to return (the last
    // per-iteration fit above only saw the history as of the *start* of
    // the final iteration). Discarding an error here used to mean a
    // failure left `gp` in a stale/inconsistent state (kernel-vs-cached-
    // Cholesky mismatch, or never fitted at all when `n_iterations == 0`)
    // while `bayesian_optimize` still returned `Ok(..)` — silently handing
    // the caller a model whose `predict()` produces nonsense instead of an
    // error. Surface it instead.
    gp.fit(&x_history, &f_history)?;

    let n_evaluations = x_history.len();
    let success = f_best < initial_best;

    Ok(BayesOptResult {
        x_best,
        f_best,
        x_history,
        f_history,
        n_evaluations,
        success,
        model: gp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rbf_kernel_properties() {
        let kernel = Kernel {
            kernel_type: KernelType::SquaredExponential,
            length_scale: 1.0,
            signal_variance: 1.0,
        };

        // Same point => signal_variance
        let k_same = kernel.compute(&[0.0, 0.0], &[0.0, 0.0]);
        assert_relative_eq!(k_same, 1.0, epsilon = 1e-10);

        // Far points => near zero
        let k_far = kernel.compute(&[0.0, 0.0], &[100.0, 100.0]);
        assert!(
            k_far < 1e-10,
            "Far points should have near-zero kernel: {}",
            k_far
        );

        // Symmetry
        let k12 = kernel.compute(&[1.0, 2.0], &[3.0, 4.0]);
        let k21 = kernel.compute(&[3.0, 4.0], &[1.0, 2.0]);
        assert_relative_eq!(k12, k21, epsilon = 1e-15);

        // Monotone decay with distance
        let k_near = kernel.compute(&[0.0], &[0.5]);
        let k_mid = kernel.compute(&[0.0], &[2.0]);
        assert!(k_near > k_mid, "Kernel should decrease with distance");
    }

    #[test]
    fn test_matern52_kernel_properties() {
        let kernel = Kernel {
            kernel_type: KernelType::Matern52,
            length_scale: 1.0,
            signal_variance: 1.0,
        };

        let k_same = kernel.compute(&[0.0], &[0.0]);
        assert_relative_eq!(k_same, 1.0, epsilon = 1e-10);

        let k_near = kernel.compute(&[0.0], &[0.5]);
        let k_far = kernel.compute(&[0.0], &[2.0]);
        assert!(k_near > k_far, "Kernel should decrease with distance");

        // Symmetry in multi-dimensions
        let k12 = kernel.compute(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let k21 = kernel.compute(&[4.0, 5.0, 6.0], &[1.0, 2.0, 3.0]);
        assert_relative_eq!(k12, k21, epsilon = 1e-15);
    }

    #[test]
    fn test_gp_prediction_at_training_points() {
        let mut gp = GaussianProcess::new(KernelType::SquaredExponential, 1e-6);

        let x_train: Vec<Vec<f64>> = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let y_train: Vec<f64> = x_train.iter().map(|x| x[0].sin()).collect();

        gp.fit(&x_train, &y_train).expect("GP fit should succeed");

        for (x, &y_true) in x_train.iter().zip(y_train.iter()) {
            let (mu, var) = gp.predict(x).expect("GP predict should succeed");
            assert!(
                (mu - y_true).abs() < 0.1,
                "Prediction at training point x={:?}: mu={}, y_true={}",
                x,
                mu,
                y_true
            );
            assert!(var >= 0.0, "Variance should be non-negative");
        }
    }

    #[test]
    fn test_gp_variance_increases_away_from_data() {
        let mut gp = GaussianProcess::new(KernelType::SquaredExponential, 1e-6);

        let x_train = vec![vec![0.0], vec![2.0], vec![4.0]];
        let y_train = vec![0.0, 1.0, 0.0];

        gp.fit(&x_train, &y_train).expect("GP fit should succeed");

        let (_, var_at_train) = gp.predict(&[0.0]).expect("predict should succeed");
        let (_, var_far) = gp.predict(&[10.0]).expect("predict should succeed");

        assert!(
            var_far > var_at_train,
            "Variance far from data ({}) should exceed variance at data ({})",
            var_far,
            var_at_train
        );
    }

    #[test]
    fn test_1d_quadratic_optimization() {
        let objective = |x: &[f64]| -> f64 { (x[0] - 0.3).powi(2) };

        let config = BayesOptConfig {
            bounds: vec![(0.0, 1.0)],
            n_initial: 5,
            n_iterations: 15,
            noise_variance: 1e-6,
            n_candidates: 500,
            optimize_hyperparams: true,
            hp_optimize_interval: 5,
            ..Default::default()
        };

        let result = bayesian_optimize(objective, config)
            .expect("Bayesian optimization should succeed for 1D quadratic");

        assert!(
            result.f_best < 0.01,
            "Best value should be close to 0, got {}",
            result.f_best
        );
        assert!(
            (result.x_best[0] - 0.3).abs() < 0.15,
            "Best x should be close to 0.3, got {}",
            result.x_best[0]
        );
        assert_eq!(result.n_evaluations, 20); // 5 initial + 15 iterations
    }

    #[test]
    fn test_2d_branin_optimization() {
        let branin = |x: &[f64]| -> f64 {
            let x1 = x[0] * 15.0 - 5.0;
            let x2 = x[1] * 15.0;

            let a = 1.0;
            let b = 5.1 / (4.0 * PI * PI);
            let c = 5.0 / PI;
            let r = 6.0;
            let s = 10.0;
            let t = 1.0 / (8.0 * PI);

            a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
        };

        let config = BayesOptConfig {
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            n_initial: 10,
            n_iterations: 30,
            noise_variance: 1e-6,
            n_candidates: 1000,
            optimize_hyperparams: true,
            hp_optimize_interval: 5,
            kernel: KernelType::Matern52,
            ..Default::default()
        };

        let result = bayesian_optimize(branin, config)
            .expect("Bayesian optimization should succeed for Branin");

        // Branin global minimum ~ 0.397887; with limited budget, get reasonably close
        assert!(
            result.f_best < 2.0,
            "Branin should find a reasonably good minimum, got {}",
            result.f_best
        );
        assert_eq!(result.x_best.len(), 2);
    }

    #[test]
    fn test_rosenbrock_optimization() {
        let rosenbrock = |x: &[f64]| -> f64 {
            let x0 = x[0];
            let x1 = x[1];
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };

        let config = BayesOptConfig {
            bounds: vec![(0.0, 2.0), (0.0, 2.0)],
            n_initial: 10,
            n_iterations: 40,
            noise_variance: 1e-6,
            n_candidates: 1000,
            optimize_hyperparams: true,
            hp_optimize_interval: 5,
            kernel: KernelType::Matern52,
            ..Default::default()
        };

        let result = bayesian_optimize(rosenbrock, config)
            .expect("Bayesian optimization should succeed for Rosenbrock");

        // Rosenbrock is very hard for BO with limited budget
        assert!(
            result.f_best < 10.0,
            "Rosenbrock best value should be < 10 with 50 evals, got {}",
            result.f_best
        );
    }

    #[test]
    fn test_different_acquisition_functions() {
        let objective = |x: &[f64]| -> f64 { (x[0] - 0.7).powi(2) };

        let acqs: Vec<AcquisitionType> = vec![
            AcquisitionType::ExpectedImprovement,
            AcquisitionType::ProbabilityOfImprovement,
            AcquisitionType::UpperConfidenceBound,
            AcquisitionType::ThompsonSampling,
        ];

        for acq in acqs {
            let acq_name = format!("{:?}", acq);
            let config = BayesOptConfig {
                bounds: vec![(0.0, 1.0)],
                n_initial: 5,
                n_iterations: 10,
                noise_variance: 1e-6,
                n_candidates: 200,
                optimize_hyperparams: false,
                acquisition: acq,
                ..Default::default()
            };

            let result = bayesian_optimize(objective, config);
            assert!(
                result.is_ok(),
                "Optimization with {} should succeed",
                acq_name
            );

            let res = result.expect("should not fail");
            assert!(
                res.f_best < 0.5,
                "{} should find reasonable solution, got f_best={}",
                acq_name,
                res.f_best
            );
        }
    }

    #[test]
    fn test_different_kernels() {
        let objective = |x: &[f64]| -> f64 { x[0].sin() + 0.1 * x[0] };

        let kernels = vec![KernelType::SquaredExponential, KernelType::Matern52];

        for kernel in kernels {
            let kernel_name = format!("{:?}", kernel);
            let config = BayesOptConfig {
                bounds: vec![(0.0, 6.0)],
                n_initial: 5,
                n_iterations: 10,
                noise_variance: 1e-6,
                n_candidates: 200,
                optimize_hyperparams: false,
                kernel,
                ..Default::default()
            };

            let result = bayesian_optimize(objective, config);
            assert!(
                result.is_ok(),
                "Optimization with kernel {} should succeed",
                kernel_name
            );
        }
    }

    #[test]
    fn test_bounded_optimization() {
        // Minimum of x^2 is at x=0, but bounds force x >= 2
        let objective = |x: &[f64]| -> f64 { x[0] * x[0] };

        let config = BayesOptConfig {
            bounds: vec![(2.0, 5.0)],
            n_initial: 3,
            n_iterations: 10,
            noise_variance: 1e-6,
            n_candidates: 300,
            optimize_hyperparams: false,
            ..Default::default()
        };

        let result =
            bayesian_optimize(objective, config).expect("Bounded optimization should succeed");

        // Best should be near x=2 (lower bound)
        assert!(
            (result.x_best[0] - 2.0).abs() < 0.5,
            "Best x should be near lower bound 2.0, got {}",
            result.x_best[0]
        );
        assert!(
            result.f_best < 6.5,
            "Best value should be near 4.0, got {}",
            result.f_best
        );

        // All evaluated points should be within bounds
        for point in &result.x_history {
            assert!(
                point[0] >= 2.0 - 1e-10 && point[0] <= 5.0 + 1e-10,
                "Point {} should be within bounds [2, 5]",
                point[0]
            );
        }
    }

    #[test]
    fn test_ei_acquisition_values() {
        // EI positive when mu < best_y
        let ei = compute_acquisition(
            &AcquisitionType::ExpectedImprovement,
            0.0,
            1.0,
            1.0,
            0.0,
            2.0,
        );
        assert!(
            ei > 0.0,
            "EI should be positive when improvement is possible"
        );

        // EI zero when sigma is 0
        let ei_zero = compute_acquisition(
            &AcquisitionType::ExpectedImprovement,
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
        );
        assert_relative_eq!(ei_zero, 0.0, epsilon = 1e-10);

        // Higher sigma => higher EI
        let ei_low = compute_acquisition(
            &AcquisitionType::ExpectedImprovement,
            0.5,
            0.1,
            1.0,
            0.0,
            2.0,
        );
        let ei_high = compute_acquisition(
            &AcquisitionType::ExpectedImprovement,
            0.5,
            2.0,
            1.0,
            0.0,
            2.0,
        );
        assert!(ei_high > ei_low, "Higher sigma should give higher EI");
    }

    #[test]
    fn test_pi_acquisition_values() {
        // mu << best_y => PI ~ 1
        let pi_good = compute_acquisition(
            &AcquisitionType::ProbabilityOfImprovement,
            -10.0,
            1.0,
            0.0,
            0.0,
            2.0,
        );
        assert!(
            pi_good > 0.9,
            "PI should be high when mu << best_y, got {}",
            pi_good
        );

        // mu >> best_y => PI ~ 0
        let pi_bad = compute_acquisition(
            &AcquisitionType::ProbabilityOfImprovement,
            10.0,
            1.0,
            0.0,
            0.0,
            2.0,
        );
        assert!(
            pi_bad < 0.1,
            "PI should be low when mu >> best_y, got {}",
            pi_bad
        );
    }

    #[test]
    fn test_ucb_kappa_effect() {
        let ucb_low_kappa = compute_acquisition(
            &AcquisitionType::UpperConfidenceBound,
            1.0,
            1.0,
            0.0,
            0.0,
            0.1,
        );
        let ucb_high_kappa = compute_acquisition(
            &AcquisitionType::UpperConfidenceBound,
            1.0,
            1.0,
            0.0,
            0.0,
            5.0,
        );
        assert!(
            ucb_high_kappa > ucb_low_kappa,
            "Higher kappa should favor exploration: low={}, high={}",
            ucb_low_kappa,
            ucb_high_kappa
        );
    }

    #[test]
    fn test_optimization_progress_is_monotonic() {
        let objective = |x: &[f64]| -> f64 { x.iter().map(|xi| (xi - 0.5).powi(2)).sum() };

        let config = BayesOptConfig {
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            n_initial: 5,
            n_iterations: 15,
            noise_variance: 1e-6,
            n_candidates: 300,
            optimize_hyperparams: false,
            ..Default::default()
        };

        let result = bayesian_optimize(objective, config).expect("Optimization should succeed");

        // Verify running best is monotonically non-increasing
        let mut running_best = f64::INFINITY;
        for &val in &result.f_history {
            if val < running_best {
                running_best = val;
            }
        }
        assert_relative_eq!(running_best, result.f_best, epsilon = 1e-15);
    }

    #[test]
    fn test_hyperparameter_optimization() {
        let mut gp = GaussianProcess::new(KernelType::SquaredExponential, 1e-6);
        // Start with bad hyperparameters
        gp.kernel = gp.kernel.with_params(0.1, 10.0);

        let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let y_train: Vec<f64> = x_train.iter().map(|x| x[0].sin()).collect();

        gp.optimize_hyperparameters(&x_train, &y_train, 3)
            .expect("Hyperparameter optimization should succeed");

        let (mu, _var) = gp.predict(&[0.25]).expect("Prediction should succeed");
        let y_true = 0.25_f64.sin();
        assert!(
            (mu - y_true).abs() < 0.5,
            "After HP opt, prediction should be reasonable: mu={}, y_true={}",
            mu,
            y_true
        );
    }

    #[test]
    fn test_invalid_inputs() {
        let objective = |x: &[f64]| -> f64 { x[0] };

        // Empty bounds
        let config = BayesOptConfig {
            bounds: vec![],
            ..Default::default()
        };
        assert!(bayesian_optimize(objective, config).is_err());

        // Inverted bounds
        let config = BayesOptConfig {
            bounds: vec![(1.0, 0.0)],
            ..Default::default()
        };
        assert!(bayesian_optimize(objective, config).is_err());

        // GP operations on unfitted model
        let gp = GaussianProcess::new(KernelType::SquaredExponential, 1e-6);
        assert!(gp.predict(&[0.0]).is_err());
        assert!(gp.negative_log_marginal_likelihood().is_err());

        // Mismatched training data
        let mut gp2 = GaussianProcess::new(KernelType::SquaredExponential, 1e-6);
        assert!(gp2.fit(&[], &[]).is_err());
        assert!(gp2.fit(&[vec![1.0]], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_cholesky_correctness() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];

        let l = cholesky_decomposition(&a).expect("Cholesky should succeed for PD matrix");

        // Verify L * L^T = A
        let n = a.len();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[i][k] * l[j][k];
                }
                assert_relative_eq!(sum, a[i][j], epsilon = 1e-10);
            }
        }

        // Non-PD should fail
        let bad = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky_decomposition(&bad).is_err());
    }

    #[test]
    fn test_erf_and_normal_cdf() {
        assert_relative_eq!(erf_approx(0.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(erf_approx(1.0), 0.842700793, epsilon = 1e-5);
        assert_relative_eq!(erf_approx(-1.0), -0.842700793, epsilon = 1e-5);
        assert_relative_eq!(erf_approx(2.0), 0.995322265, epsilon = 1e-5);

        assert_relative_eq!(standard_normal_cdf(0.0), 0.5, epsilon = 1e-6);
        assert!(standard_normal_cdf(5.0) > 0.999);
        assert!(standard_normal_cdf(-5.0) < 0.001);
    }

    #[test]
    fn test_result_struct_completeness() {
        let objective = |x: &[f64]| -> f64 { x[0] * x[0] };

        let config = BayesOptConfig {
            bounds: vec![(0.0, 1.0)],
            n_initial: 3,
            n_iterations: 5,
            noise_variance: 1e-6,
            n_candidates: 100,
            optimize_hyperparams: false,
            ..Default::default()
        };

        let result = bayesian_optimize(objective, config).expect("Should succeed");

        // Verify all result fields
        assert_eq!(result.x_best.len(), 1);
        assert!(result.f_best.is_finite());
        assert_eq!(result.x_history.len(), 8); // 3 initial + 5 iterations
        assert_eq!(result.f_history.len(), 8);
        assert_eq!(result.n_evaluations, 8);
        // Model should be usable
        let (mu, var) = result.model.predict(&[0.5]).expect("Model should predict");
        assert!(mu.is_finite());
        assert!(var >= 0.0);
    }

    #[test]
    fn test_latin_hypercube_sampling_coverage() {
        let mut rng = thread_rng();
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let n_samples = 20;

        let samples =
            latin_hypercube_sampling(n_samples, &bounds, &mut rng).expect("LHS should succeed");

        assert_eq!(samples.len(), n_samples);

        // All samples must be within bounds
        for sample in &samples {
            assert_eq!(sample.len(), 2);
            for (d, &val) in sample.iter().enumerate() {
                assert!(
                    val >= bounds[d].0 && val <= bounds[d].1,
                    "Sample dim {} value {} out of bounds [{}, {}]",
                    d,
                    val,
                    bounds[d].0,
                    bounds[d].1
                );
            }
        }

        // Check LHS strata coverage
        for dim in 0..2 {
            let mut strata = vec![false; n_samples];
            for sample in &samples {
                let stratum = (sample[dim] * n_samples as f64) as usize;
                let stratum = stratum.min(n_samples - 1);
                strata[stratum] = true;
            }
            let covered = strata.iter().filter(|&&s| s).count();
            assert!(
                covered >= n_samples / 2,
                "LHS should cover most strata in dim {}: only {}/{}",
                dim,
                covered,
                n_samples
            );
        }
    }

    #[test]
    fn test_xi_parameter_effect() {
        let ei_low_xi = compute_acquisition(
            &AcquisitionType::ExpectedImprovement,
            0.8,
            0.5,
            1.0,
            0.0,
            2.0,
        );
        let ei_high_xi = compute_acquisition(
            &AcquisitionType::ExpectedImprovement,
            0.8,
            0.5,
            1.0,
            0.5,
            2.0,
        );
        // Higher xi subtracts more from improvement => lower EI at same point
        assert!(
            ei_low_xi >= ei_high_xi,
            "Higher xi should reduce EI at the same point: low_xi={}, high_xi={}",
            ei_low_xi,
            ei_high_xi
        );
    }

    // ========================================================================
    // Regression tests: discarded `Result`s from `gp.fit(...)` /
    // `gp.optimize_hyperparameters(...)` inside `bayesian_optimize` used to
    // be thrown away with `let _ = ...`, so a failure there silently left
    // the GP in a stale/inconsistent state instead of surfacing an error.
    // ========================================================================

    /// A negative `noise_variance` makes `noise = noise_variance / y_std^2`
    /// negative enough that `K[i][i] + noise + jitter` goes non-positive, so
    /// the covariance matrix built from otherwise perfectly valid, distinct
    /// training points is not positive definite and Cholesky must fail.
    /// This is the "degenerate covariance" failure mode `fit()` is supposed
    /// to reject rather than silently accept.
    #[test]
    fn test_gp_fit_returns_err_on_degenerate_negative_noise_covariance() {
        let mut gp = GaussianProcess::new(KernelType::SquaredExponential, -10.0);
        let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_train = vec![1.0, 2.0, 3.0];

        let err = gp.fit(&x_train, &y_train).expect_err(
            "fit() must surface an error for a degenerate (non-PD) covariance, not silently succeed",
        );
        assert!(
            matches!(err, NumRs2Error::NumericalError(_)),
            "expected NumericalError for a non-positive-definite covariance matrix, got {:?}",
            err
        );
    }

    /// Same degenerate-covariance construction as above, but exercised
    /// through `optimize_hyperparameters` (the call `bayesian_optimize`
    /// makes at each `hp_optimize_interval` step) rather than `fit`
    /// directly. `n_restarts = 1` keeps this deterministic: restart 0 always
    /// reuses the kernel's *current* hyperparameters (no RNG sampling is
    /// involved unless `n_restarts > 1`), so the failure does not depend on
    /// `thread_rng()`.
    #[test]
    fn test_gp_optimize_hyperparameters_returns_err_on_degenerate_negative_noise_covariance() {
        let mut gp = GaussianProcess::new(KernelType::SquaredExponential, -10.0);
        let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_train = vec![1.0, 2.0, 3.0];

        let result = gp.optimize_hyperparameters(&x_train, &y_train, 1);
        assert!(
            result.is_err(),
            "optimize_hyperparameters() must surface the underlying fit failure instead of \
             silently succeeding: got {:?}",
            result
        );
    }

    /// Regression test for the `let _ = gp.fit(&x_history, &f_history);`
    /// discard that used to sit after the optimization loop returned. With
    /// `n_iterations = 0` the loop body (which contains the *already*
    /// correctly-propagated per-iteration `gp.fit(...)?`) never runs, so the
    /// only `fit` call exercised end-to-end is the final one — this isolates
    /// the exact line that used to swallow its error.
    ///
    /// Before the fix, this silently returned `Ok(BayesOptResult { model, .. })`
    /// with `model` left in an inconsistent, never-actually-fitted state:
    /// `fit()` updates `x_train`/`y_mean`/`y_std`/`y_train` *before*
    /// attempting the Cholesky decomposition, so a failure there leaves
    /// `cholesky_l`/`alpha` at their `GaussianProcess::new()` (empty)
    /// values while the rest of the state has already moved on. A later
    /// `predict()` call on such a model does not even error (its `x_train
    /// .is_empty()` guard passes, since `x_train` was updated): it silently
    /// returns `mu == y_mean` for every query point via a zero-length
    /// `zip` against the empty `alpha`. After the fix, the caller gets an
    /// `Err` instead of that silently-wrong model.
    #[test]
    fn test_bayesian_optimize_surfaces_final_fit_error_instead_of_returning_broken_model() {
        let objective = |x: &[f64]| -> f64 { x[0] * x[0] };
        let config = BayesOptConfig {
            bounds: vec![(0.0, 1.0)],
            n_initial: 5,
            n_iterations: 0,
            noise_variance: -10.0,
            optimize_hyperparams: false,
            ..Default::default()
        };

        let err = bayesian_optimize(objective, config).expect_err(
            "bayesian_optimize must surface the final-fit error instead of returning a broken model",
        );
        assert!(
            matches!(err, NumRs2Error::NumericalError(_)),
            "expected NumericalError from the degenerate final fit, got {:?}",
            err
        );
    }
}
