//! CMA-ES configuration types and builder methods.

/// Configuration for the CMA-ES optimizer.
///
/// Provides full control over the algorithm parameters. Most parameters have
/// sensible defaults computed from the problem dimension. Use [`CMAESConfig::default`]
/// for a reasonable starting configuration, then customize via builder methods.
///
/// # Example
///
/// ```
/// use numrs2::optimize::cma_es::CMAESConfig;
///
/// let config = CMAESConfig::default()
///     .with_population_size(50)
///     .with_sigma0(2.0)
///     .with_max_iter(5000)
///     .with_bounds(vec![(-10.0, 10.0); 3]);
/// ```
#[derive(Debug, Clone)]
pub struct CMAESConfig {
    /// Population size (lambda). If 0, computed as 4 + floor(3 * ln(n)).
    pub population_size: usize,
    /// Initial step-size (sigma_0). Must be > 0. Default: 0.5
    pub sigma0: f64,
    /// Maximum number of iterations (generations). Default: 10000
    pub max_iter: usize,
    /// Function value tolerance for convergence. Default: 1e-12
    pub ftol: f64,
    /// Parameter (x) tolerance for convergence. Default: 1e-12
    pub xtol: f64,
    /// Optional box constraints: (lower, upper) for each dimension
    pub bounds: Option<Vec<(f64, f64)>>,
    /// Optional random seed for reproducibility
    pub seed: Option<u64>,
    /// Penalty coefficient for box constraint violations. Default: 1e6
    pub penalty_coefficient: f64,
    /// Enable IPOP restart strategy. Default: false
    pub enable_restarts: bool,
    /// Maximum number of restarts for IPOP. Default: 9
    pub max_restarts: usize,
    /// Population size multiplier on each restart. Default: 2.0
    pub restart_pop_multiplier: f64,
    /// Maximum condition number of covariance matrix before restart. Default: 1e14
    pub max_condition_number: f64,
    /// Optional custom recombination weights
    pub weights: Option<Vec<f64>>,
    /// Optional learning rate for rank-1 update (c1)
    pub c1: Option<f64>,
    /// Optional learning rate for rank-mu update (cmu)
    pub cmu: Option<f64>,
    /// Optional learning rate for cumulation of covariance path (cc)
    pub cc: Option<f64>,
    /// Optional learning rate for cumulation of step-size path (cs)
    pub cs: Option<f64>,
    /// Optional damping for step-size adaptation (damps)
    pub damps: Option<f64>,
}

impl Default for CMAESConfig {
    fn default() -> Self {
        Self {
            population_size: 0,
            sigma0: 0.5,
            max_iter: 10000,
            ftol: 1e-12,
            xtol: 1e-12,
            bounds: None,
            seed: None,
            penalty_coefficient: 1e6,
            enable_restarts: false,
            max_restarts: 9,
            restart_pop_multiplier: 2.0,
            max_condition_number: 1e14,
            weights: None,
            c1: None,
            cmu: None,
            cc: None,
            cs: None,
            damps: None,
        }
    }
}

impl CMAESConfig {
    /// Create a new configuration with sensible defaults for the given dimension.
    ///
    /// The defaults follow the recommendations from Hansen's tutorial (2016).
    ///
    /// # Arguments
    /// * `dimension` - Problem dimensionality (used to compute default lambda)
    pub fn new(dimension: usize) -> Self {
        let lambda = compute_default_lambda(dimension);
        Self {
            population_size: lambda,
            ..Self::default()
        }
    }

    /// Set the population size (lambda).
    pub fn with_population_size(mut self, pop_size: usize) -> Self {
        self.population_size = pop_size;
        self
    }

    /// Set the initial step size (sigma_0).
    pub fn with_sigma0(mut self, sigma0: f64) -> Self {
        self.sigma0 = sigma0;
        self
    }

    /// Backward-compatible alias for [`with_sigma0`](Self::with_sigma0).
    pub fn with_sigma(self, sigma: f64) -> Self {
        self.with_sigma0(sigma)
    }

    /// Set the maximum number of iterations (generations).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Backward-compatible alias for [`with_max_iter`](Self::with_max_iter).
    pub fn with_max_generations(self, max_gen: usize) -> Self {
        self.with_max_iter(max_gen)
    }

    /// Set box constraints for each dimension.
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set function value convergence tolerance.
    pub fn with_ftol(mut self, ftol: f64) -> Self {
        self.ftol = ftol;
        self
    }

    /// Set parameter convergence tolerance.
    pub fn with_xtol(mut self, xtol: f64) -> Self {
        self.xtol = xtol;
        self
    }

    /// Enable IPOP restart strategy.
    pub fn with_restarts(mut self, max_restarts: usize) -> Self {
        self.enable_restarts = true;
        self.max_restarts = max_restarts;
        self
    }

    /// Set population size (lambda) manually.
    pub fn with_lambda(mut self, lambda: usize) -> Self {
        self.population_size = lambda;
        self
    }

    /// Resolve the effective population size for a given dimension.
    pub(crate) fn effective_lambda(&self, dimension: usize) -> usize {
        if self.population_size == 0 {
            compute_default_lambda(dimension)
        } else {
            self.population_size
        }
    }
}

/// Compute default population size (lambda) from dimension.
///
/// Uses the standard formula: lambda = 4 + floor(3 * ln(n)).
pub(crate) fn compute_default_lambda(dimension: usize) -> usize {
    if dimension == 0 {
        return 4;
    }
    4 + (3.0 * (dimension as f64).ln()).floor() as usize
}

/// Compute default recombination weights.
///
/// Uses the standard log-linear weighting scheme from Hansen (2016).
pub(crate) fn compute_default_weights(mu: usize, _lambda: usize) -> Vec<f64> {
    let mu_f = mu as f64;
    let raw: Vec<f64> = (0..mu)
        .map(|i| (mu_f + 0.5).ln() - ((i + 1) as f64).ln())
        .collect();

    let sum: f64 = raw.iter().sum();
    if sum > 0.0 {
        raw.iter().map(|&w| w / sum).collect()
    } else {
        vec![1.0 / mu_f; mu]
    }
}

/// Compute the variance effective selection mass (mu_eff).
pub(crate) fn compute_mu_eff(weights: &[f64], mu: usize) -> f64 {
    let sum_w: f64 = weights.iter().take(mu).sum();
    let sum_w2: f64 = weights.iter().take(mu).map(|w| w * w).sum();
    if sum_w2 > 0.0 {
        (sum_w * sum_w) / sum_w2
    } else {
        1.0
    }
}
